use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::{json, Value};
use sieve_lcm::db::config::LcmConfig;
use sieve_lcm::engine::{ContextEngineInfo, ConversationLookupApi, LcmContextEngineApi};
use sieve_lcm::expansion_auth::{
    create_delegated_expansion_grant, resolve_delegated_expansion_grant_id,
    reset_delegated_expansion_grants_for_tests, CreateDelegatedExpansionGrantInput,
};
use sieve_lcm::retrieval::{
    DescribeResult, DescribeResultType, DescribeSummary, ExpandInput, ExpandResult, GrepInput,
    GrepResult, RetrievalApi,
};
use sieve_lcm::store::conversation_store::ConversationRecord;
use sieve_lcm::store::summary_store::{SummaryKind, SummarySearchResult};
use sieve_lcm::tools::lcm_expand_query_tool::create_lcm_expand_query_tool;
use sieve_lcm::tools::lcm_expansion_recursion_guard::{
    get_delegated_expansion_context_for_tests,
    get_expansion_delegation_telemetry_snapshot_for_tests, reset_expansion_delegation_guard_for_tests,
    stamp_delegated_expansion_context,
};
use sieve_lcm::types::{
    CompletionRequest, CompletionResult, GatewayCallRequest, LcmDependencies, LcmLogger, ModelRef,
};

static TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

fn parse_agent_session_key(session_key: &str) -> Option<(String, String)> {
    let trimmed = session_key.trim();
    if !trimmed.starts_with("agent:") {
        return None;
    }
    let parts = trimmed.split(':').collect::<Vec<&str>>();
    if parts.len() < 3 {
        return None;
    }
    Some((
        parts.get(1).copied().unwrap_or("main").to_string(),
        parts[2..].join(":"),
    ))
}

fn read_latest_assistant_reply(messages: &[Value]) -> Option<String> {
    for message in messages.iter().rev() {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        if let Some(content) = message.get("content").and_then(Value::as_str) {
            return Some(content.to_string());
        }
        if let Some(parts) = message.get("content").and_then(Value::as_array) {
            let text = parts
                .iter()
                .filter_map(|part| {
                    if part.get("type").and_then(Value::as_str) == Some("text") {
                        part.get("text").and_then(Value::as_str)
                    } else {
                        None
                    }
                })
                .collect::<Vec<&str>>()
                .join("\n")
                .trim()
                .to_string();
            if !text.is_empty() {
                return Some(text);
            }
        }
    }
    None
}

#[derive(Default)]
struct TestLogger;

#[async_trait]
impl LcmLogger for TestLogger {
    fn info(&self, _msg: &str) {}
    fn warn(&self, _msg: &str) {}
    fn error(&self, _msg: &str) {}
    fn debug(&self, _msg: &str) {}
}

struct TestDeps {
    config: LcmConfig,
    logger: TestLogger,
    call_gateway: Arc<dyn Fn(GatewayCallRequest) -> anyhow::Result<Value> + Send + Sync>,
}

#[async_trait]
impl LcmDependencies for TestDeps {
    fn config(&self) -> &LcmConfig {
        &self.config
    }

    async fn complete(&self, _request: CompletionRequest) -> anyhow::Result<CompletionResult> {
        Ok(CompletionResult::default())
    }

    async fn call_gateway(&self, request: GatewayCallRequest) -> anyhow::Result<Value> {
        (self.call_gateway)(request)
    }

    fn resolve_model(
        &self,
        _model_ref: Option<&str>,
        _provider_hint: Option<&str>,
    ) -> anyhow::Result<ModelRef> {
        Ok(ModelRef {
            provider: "anthropic".to_string(),
            model: "claude-opus-4-5".to_string(),
        })
    }

    fn get_api_key(&self, _provider: &str, _model: &str) -> Option<String> {
        None
    }

    fn require_api_key(&self, _provider: &str, _model: &str) -> anyhow::Result<String> {
        Ok(String::new())
    }

    fn parse_agent_session_key(&self, session_key: &str) -> Option<(String, String)> {
        parse_agent_session_key(session_key)
    }

    fn is_subagent_session_key(&self, session_key: &str) -> bool {
        session_key.contains(":subagent:")
    }

    fn normalize_agent_id(&self, id: Option<&str>) -> String {
        id.map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or("main")
            .to_string()
    }

    fn build_subagent_system_prompt(
        &self,
        _depth: i32,
        _max_depth: i32,
        _task_summary: Option<&str>,
    ) -> String {
        "subagent prompt".to_string()
    }

    fn read_latest_assistant_reply(&self, messages: &[Value]) -> Option<String> {
        read_latest_assistant_reply(messages)
    }

    fn resolve_agent_dir(&self) -> String {
        "/tmp/openclaw-agent".to_string()
    }

    async fn resolve_session_id_from_session_key(
        &self,
        _session_key: &str,
    ) -> anyhow::Result<Option<String>> {
        Ok(None)
    }

    fn agent_lane_subagent(&self) -> &str {
        "subagent"
    }

    fn logger(&self) -> &dyn LcmLogger {
        &self.logger
    }
}

fn make_deps(
    call_gateway: Arc<dyn Fn(GatewayCallRequest) -> anyhow::Result<Value> + Send + Sync>,
) -> Arc<TestDeps> {
    Arc::new(TestDeps {
        config: LcmConfig {
            enabled: true,
            database_path: ":memory:".to_string(),
            context_threshold: 0.75,
            fresh_tail_count: 8,
            leaf_min_fanout: 8,
            condensed_min_fanout: 4,
            condensed_min_fanout_hard: 2,
            incremental_max_depth: 0,
            leaf_chunk_tokens: 20_000,
            leaf_target_tokens: 600,
            condensed_target_tokens: 900,
            max_expand_tokens: 120,
            large_file_token_threshold: 25_000,
            large_file_summary_provider: String::new(),
            large_file_summary_model: String::new(),
            autocompact_disabled: false,
            timezone: "UTC".to_string(),
            prune_heartbeat_ok: false,
        },
        logger: TestLogger,
        call_gateway,
    })
}

struct MockRetrieval {
    grep_calls: Mutex<Vec<GrepInput>>,
    describe_calls: Mutex<Vec<String>>,
    grep_result: Mutex<GrepResult>,
    describe_result: Mutex<Option<DescribeResult>>,
}

impl Default for MockRetrieval {
    fn default() -> Self {
        Self {
            grep_calls: Mutex::new(vec![]),
            describe_calls: Mutex::new(vec![]),
            grep_result: Mutex::new(GrepResult {
                messages: vec![],
                summaries: vec![],
                total_matches: 0,
            }),
            describe_result: Mutex::new(None),
        }
    }
}

#[async_trait]
impl RetrievalApi for MockRetrieval {
    async fn describe(&self, id: &str) -> anyhow::Result<Option<DescribeResult>> {
        self.describe_calls.lock().push(id.to_string());
        Ok(self.describe_result.lock().clone())
    }

    async fn grep(&self, input: GrepInput) -> anyhow::Result<GrepResult> {
        self.grep_calls.lock().push(input);
        Ok(self.grep_result.lock().clone())
    }

    async fn expand(&self, _input: ExpandInput) -> anyhow::Result<ExpandResult> {
        Ok(ExpandResult {
            children: vec![],
            messages: vec![],
            estimated_tokens: 0,
            truncated: false,
        })
    }
}

#[derive(Clone)]
struct MockConversationLookup {
    conversation: Option<ConversationRecord>,
}

#[async_trait]
impl ConversationLookupApi for MockConversationLookup {
    async fn get_conversation_by_session_id(
        &self,
        _session_id: &str,
    ) -> anyhow::Result<Option<ConversationRecord>> {
        Ok(self.conversation.clone())
    }
}

struct MockEngine {
    info: ContextEngineInfo,
    retrieval: Arc<MockRetrieval>,
    conversation_lookup: Arc<MockConversationLookup>,
}

impl MockEngine {
    fn new(retrieval: Arc<MockRetrieval>, conversation_id: Option<i64>) -> Self {
        let conversation = conversation_id.map(|conversation_id| ConversationRecord {
            conversation_id,
            session_id: "session-1".to_string(),
            title: None,
            bootstrapped_at: None,
            created_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                .unwrap()
                .with_timezone(&Utc),
            updated_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                .unwrap()
                .with_timezone(&Utc),
        });
        Self {
            info: ContextEngineInfo {
                id: "lcm".to_string(),
                name: "LCM".to_string(),
                version: "0.0.0".to_string(),
                owns_compaction: true,
            },
            retrieval,
            conversation_lookup: Arc::new(MockConversationLookup { conversation }),
        }
    }
}

impl LcmContextEngineApi for MockEngine {
    fn info(&self) -> &ContextEngineInfo {
        &self.info
    }
    fn get_retrieval(&self) -> Arc<dyn RetrievalApi> {
        self.retrieval.clone()
    }
    fn get_conversation_store(&self) -> Arc<dyn ConversationLookupApi> {
        self.conversation_lookup.clone()
    }
}

fn summary_describe_result(conversation_id: i64) -> DescribeResult {
    DescribeResult {
        id: "sum_a".to_string(),
        result: DescribeResultType::Summary(DescribeSummary {
            conversation_id,
            kind: SummaryKind::Leaf,
            content: "leaf".to_string(),
            depth: 0,
            token_count: 1,
            descendant_count: 0,
            descendant_token_count: 0,
            source_message_token_count: 1,
            file_ids: vec![],
            parent_ids: vec![],
            child_ids: vec![],
            message_ids: vec![],
            earliest_at: None,
            latest_at: None,
            subtree: vec![],
            created_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                .unwrap()
                .with_timezone(&Utc),
        }),
    }
}

#[tokio::test]
async fn returns_focused_delegated_answer_for_explicit_summary_ids() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    reset_expansion_delegation_guard_for_tests();

    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.describe_result.lock() = Some(summary_describe_result(42));

    let delegated_session_key = Arc::new(Mutex::new(String::new()));
    let delegated_context = Arc::new(Mutex::new(None));
    let gateway_calls = Arc::new(Mutex::new(Vec::<GatewayCallRequest>::new()));
    let key_ref = delegated_session_key.clone();
    let ctx_ref = delegated_context.clone();
    let calls_ref = gateway_calls.clone();
    let deps = make_deps(Arc::new(move |request: GatewayCallRequest| {
        calls_ref.lock().push(request.clone());
        match request.method.as_str() {
            "agent" => {
                let key = request
                    .params
                    .as_ref()
                    .and_then(|v| v.get("sessionKey"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                *key_ref.lock() = key.clone();
                *ctx_ref.lock() = get_delegated_expansion_context_for_tests(&key);
                Ok(json!({ "runId": "run-1" }))
            }
            "agent.wait" => Ok(json!({ "status": "ok" })),
            "sessions.get" => Ok(json!({
                "messages": [{
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "{\"answer\":\"Issue traced to stale token handling.\",\"citedIds\":[\"sum_a\"],\"expandedSummaryCount\":1,\"totalSourceTokens\":45000,\"truncated\":false}"
                    }]
                }]
            })),
            "sessions.delete" => Ok(json!({ "ok": true })),
            _ => Ok(json!({})),
        }
    }));

    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval, None));
    let tool = create_lcm_expand_query_tool(
        deps,
        engine,
        Some("agent:main:main".to_string()),
        Some("agent:main:main".to_string()),
        None,
    );
    let result = tool
        .execute(
            "call-1",
            json!({
                "summaryIds": ["sum_a"],
                "prompt": "What caused the outage?",
                "conversationId": 42,
                "maxTokens": 700
            }),
        )
        .await
        .expect("execute");

    assert_eq!(
        result.details.get("answer").and_then(Value::as_str),
        Some("Issue traced to stale token handling.")
    );
    assert_eq!(
        result.details.get("citedIds").and_then(Value::as_array).map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect::<Vec<String>>()
        }),
        Some(vec!["sum_a".to_string()])
    );
    assert_eq!(
        result.details.get("sourceConversationId").and_then(Value::as_i64),
        Some(42)
    );
    assert_eq!(
        result.details.get("expandedSummaryCount").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        result.details.get("totalSourceTokens").and_then(Value::as_i64),
        Some(45000)
    );
    assert_eq!(
        result.details.get("truncated").and_then(Value::as_bool),
        Some(false)
    );

    let agent_call = gateway_calls
        .lock()
        .iter()
        .find(|entry| entry.method == "agent")
        .cloned()
        .expect("agent call");
    let message = agent_call
        .params
        .as_ref()
        .and_then(|v| v.get("message"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    assert!(message.contains("lcm_expand"));
    assert!(message.contains("lcm_describe"));
    assert!(message.contains("DO NOT call `lcm_expand_query` from this delegated session."));
    assert!(message.contains("Synthesize the final answer from retrieved evidence, not assumptions."));
    assert!(message.contains("Expansion token budget"));

    let delegated_key = delegated_session_key.lock().clone();
    assert!(!delegated_key.is_empty());
    let delegated = delegated_context.lock().clone().expect("delegated ctx");
    assert_eq!(delegated.expansion_depth, 1);
    assert_eq!(delegated.origin_session_key, "agent:main:main");
    assert_eq!(delegated.stamped_by, "lcm_expand_query");
    assert!(!delegated.request_id.is_empty());
    assert_eq!(resolve_delegated_expansion_grant_id(&delegated_key), None);
    let snapshot = get_expansion_delegation_telemetry_snapshot_for_tests();
    assert_eq!(snapshot.get("start"), Some(&1));
    assert_eq!(snapshot.get("block"), Some(&0));
    assert_eq!(snapshot.get("timeout"), Some(&0));
    assert_eq!(snapshot.get("success"), Some(&1));
}

#[tokio::test]
async fn returns_validation_error_when_prompt_missing() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    reset_expansion_delegation_guard_for_tests();

    let retrieval = Arc::new(MockRetrieval::default());
    let call_count = Arc::new(Mutex::new(0_usize));
    let count_ref = call_count.clone();
    let deps = make_deps(Arc::new(move |_request: GatewayCallRequest| {
        *count_ref.lock() += 1;
        Ok(json!({}))
    }));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval, None));
    let tool = create_lcm_expand_query_tool(
        deps,
        engine,
        Some("agent:main:main".to_string()),
        Some("agent:main:main".to_string()),
        None,
    );
    let result = tool
        .execute(
            "call-2",
            json!({
                "summaryIds": ["sum_a"],
                "prompt": "   "
            }),
        )
        .await
        .expect("execute");
    assert_eq!(
        result.details.get("error").and_then(Value::as_str),
        Some("prompt is required.")
    );
    assert_eq!(*call_count.lock(), 0);
}

#[tokio::test]
async fn returns_timeout_when_delegated_run_exceeds_120_seconds() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    reset_expansion_delegation_guard_for_tests();

    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.describe_result.lock() = Some(summary_describe_result(42));

    let delegated_session_key = Arc::new(Mutex::new(String::new()));
    let gateway_calls = Arc::new(Mutex::new(Vec::<GatewayCallRequest>::new()));
    let key_ref = delegated_session_key.clone();
    let calls_ref = gateway_calls.clone();
    let deps = make_deps(Arc::new(move |request: GatewayCallRequest| {
        calls_ref.lock().push(request.clone());
        match request.method.as_str() {
            "agent" => {
                let key = request
                    .params
                    .as_ref()
                    .and_then(|v| v.get("sessionKey"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                *key_ref.lock() = key;
                Ok(json!({ "runId": "run-timeout" }))
            }
            "agent.wait" => Ok(json!({ "status": "timeout" })),
            "sessions.delete" => Ok(json!({ "ok": true })),
            _ => Ok(json!({})),
        }
    }));

    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval, None));
    let tool = create_lcm_expand_query_tool(
        deps,
        engine,
        Some("agent:main:main".to_string()),
        Some("agent:main:main".to_string()),
        None,
    );
    let result = tool
        .execute(
            "call-3",
            json!({
                "summaryIds": ["sum_a"],
                "prompt": "Summarize root cause",
                "conversationId": 42
            }),
        )
        .await
        .expect("execute");
    assert!(result
        .details
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .contains("timed out"));
    let methods = gateway_calls
        .lock()
        .iter()
        .map(|entry| entry.method.clone())
        .collect::<Vec<String>>();
    assert!(methods.contains(&"sessions.delete".to_string()));
    let key = delegated_session_key.lock().clone();
    assert!(!key.is_empty());
    assert_eq!(resolve_delegated_expansion_grant_id(&key), None);
    let snapshot = get_expansion_delegation_telemetry_snapshot_for_tests();
    assert_eq!(snapshot.get("start"), Some(&1));
    assert_eq!(snapshot.get("block"), Some(&0));
    assert_eq!(snapshot.get("timeout"), Some(&1));
    assert_eq!(snapshot.get("success"), Some(&0));
}

#[tokio::test]
async fn cleans_up_delegated_session_and_grant_when_agent_call_fails() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    reset_expansion_delegation_guard_for_tests();

    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.describe_result.lock() = Some(summary_describe_result(42));

    let delegated_session_key = Arc::new(Mutex::new(String::new()));
    let gateway_calls = Arc::new(Mutex::new(Vec::<GatewayCallRequest>::new()));
    let key_ref = delegated_session_key.clone();
    let calls_ref = gateway_calls.clone();
    let deps = make_deps(Arc::new(move |request: GatewayCallRequest| {
        calls_ref.lock().push(request.clone());
        match request.method.as_str() {
            "agent" => {
                let key = request
                    .params
                    .as_ref()
                    .and_then(|v| v.get("sessionKey"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                *key_ref.lock() = key;
                anyhow::bail!("agent spawn failed");
            }
            "sessions.delete" => Ok(json!({ "ok": true })),
            _ => Ok(json!({})),
        }
    }));

    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval, None));
    let tool = create_lcm_expand_query_tool(
        deps,
        engine,
        Some("agent:main:main".to_string()),
        Some("agent:main:main".to_string()),
        None,
    );
    let result = tool
        .execute(
            "call-4",
            json!({
                "summaryIds": ["sum_a"],
                "prompt": "Answer this",
                "conversationId": 42
            }),
        )
        .await
        .expect("execute");
    assert_eq!(
        result.details.get("error").and_then(Value::as_str),
        Some("agent spawn failed")
    );
    let methods = gateway_calls
        .lock()
        .iter()
        .map(|entry| entry.method.clone())
        .collect::<Vec<String>>();
    assert!(methods.contains(&"sessions.delete".to_string()));
    let key = delegated_session_key.lock().clone();
    assert!(!key.is_empty());
    assert_eq!(resolve_delegated_expansion_grant_id(&key), None);
}

#[tokio::test]
async fn greps_summaries_first_when_query_is_provided() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    reset_expansion_delegation_guard_for_tests();

    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.grep_result.lock() = GrepResult {
        messages: vec![],
        summaries: vec![
            SummarySearchResult {
                summary_id: "sum_x".to_string(),
                conversation_id: 7,
                kind: SummaryKind::Leaf,
                snippet: "x".to_string(),
                created_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                    .unwrap()
                    .with_timezone(&Utc),
                rank: None,
            },
            SummarySearchResult {
                summary_id: "sum_y".to_string(),
                conversation_id: 7,
                kind: SummaryKind::Leaf,
                snippet: "y".to_string(),
                created_at: DateTime::parse_from_rfc3339("2026-01-01T00:01:00.000Z")
                    .unwrap()
                    .with_timezone(&Utc),
                rank: None,
            },
        ],
        total_matches: 2,
    };

    let gateway_calls = Arc::new(Mutex::new(Vec::<GatewayCallRequest>::new()));
    let calls_ref = gateway_calls.clone();
    let deps = make_deps(Arc::new(move |request: GatewayCallRequest| {
        calls_ref.lock().push(request.clone());
        match request.method.as_str() {
            "agent" => Ok(json!({ "runId": "run-query" })),
            "agent.wait" => Ok(json!({ "status": "ok" })),
            "sessions.get" => Ok(json!({
                "messages": [{
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "{\"answer\":\"Top regression happened after deploy B.\",\"citedIds\":[\"sum_x\",\"sum_y\"],\"expandedSummaryCount\":2,\"totalSourceTokens\":2500,\"truncated\":false}"
                    }]
                }]
            })),
            "sessions.delete" => Ok(json!({ "ok": true })),
            _ => Ok(json!({})),
        }
    }));

    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), Some(7)));
    let tool = create_lcm_expand_query_tool(
        deps,
        engine,
        Some("session-1".to_string()),
        Some("agent:main:main".to_string()),
        None,
    );
    let result = tool
        .execute(
            "call-5",
            json!({
                "query": "deploy regression",
                "prompt": "What regressed?"
            }),
        )
        .await
        .expect("execute");

    let grep_calls = retrieval.grep_calls.lock();
    assert_eq!(grep_calls.len(), 1);
    assert_eq!(grep_calls[0].query, "deploy regression");
    assert_eq!(grep_calls[0].mode, "full_text");
    assert_eq!(grep_calls[0].scope, "summaries");
    assert_eq!(grep_calls[0].conversation_id, Some(7));

    let agent_call = gateway_calls
        .lock()
        .iter()
        .find(|entry| entry.method == "agent")
        .cloned()
        .expect("agent call");
    let message = agent_call
        .params
        .as_ref()
        .and_then(|v| v.get("message"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    assert!(message.contains("sum_x"));
    assert!(message.contains("sum_y"));

    assert_eq!(
        result.details.get("sourceConversationId").and_then(Value::as_i64),
        Some(7)
    );
    assert_eq!(
        result
            .details
            .get("expandedSummaryCount")
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        result.details.get("citedIds").and_then(Value::as_array).map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect::<Vec<String>>()
        }),
        Some(vec!["sum_x".to_string(), "sum_y".to_string()])
    );
}

#[tokio::test]
async fn blocks_delegated_reentry_with_deterministic_recursion_errors() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    reset_expansion_delegation_guard_for_tests();

    let retrieval = Arc::new(MockRetrieval::default());
    let delegated_session_key = "agent:main:subagent:recursive";
    create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
        delegated_session_key: delegated_session_key.to_string(),
        issuer_session_id: "agent:main:main".to_string(),
        allowed_conversation_ids: vec![42],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: Some(120),
        ttl_ms: None,
    });
    stamp_delegated_expansion_context(
        delegated_session_key,
        "req-recursive",
        1,
        "agent:main:main",
        "test",
    );

    let call_count = Arc::new(Mutex::new(0_usize));
    let count_ref = call_count.clone();
    let deps = make_deps(Arc::new(move |_request: GatewayCallRequest| {
        *count_ref.lock() += 1;
        Ok(json!({}))
    }));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval, None));
    let tool = create_lcm_expand_query_tool(
        deps,
        engine,
        Some(delegated_session_key.to_string()),
        Some(delegated_session_key.to_string()),
        None,
    );

    let first = tool
        .execute(
            "call-recursive-1",
            json!({
                "summaryIds": ["sum_a"],
                "prompt": "Should block recursion",
                "conversationId": 42
            }),
        )
        .await
        .expect("first");
    assert_eq!(
        first.details.get("errorCode").and_then(Value::as_str),
        Some("EXPANSION_RECURSION_BLOCKED")
    );
    assert_eq!(
        first.details.get("reason").and_then(Value::as_str),
        Some("depth_cap")
    );
    assert_eq!(
        first.details.get("requestId").and_then(Value::as_str),
        Some("req-recursive")
    );
    let first_error = first.details.get("error").and_then(Value::as_str).unwrap_or_default();
    assert!(first_error.contains("Recovery: In delegated sub-agent sessions, call `lcm_expand` directly"));
    assert!(first_error.contains("Do NOT call `lcm_expand_query` from delegated context."));

    let second = tool
        .execute(
            "call-recursive-2",
            json!({
                "summaryIds": ["sum_a"],
                "prompt": "Should block recursion again",
                "conversationId": 42
            }),
        )
        .await
        .expect("second");
    assert_eq!(
        second.details.get("errorCode").and_then(Value::as_str),
        Some("EXPANSION_RECURSION_BLOCKED")
    );
    assert_eq!(
        second.details.get("reason").and_then(Value::as_str),
        Some("idempotent_reentry")
    );
    assert_eq!(
        second.details.get("requestId").and_then(Value::as_str),
        Some("req-recursive")
    );

    assert_eq!(*call_count.lock(), 0);
    let snapshot = get_expansion_delegation_telemetry_snapshot_for_tests();
    assert_eq!(snapshot.get("start"), Some(&2));
    assert_eq!(snapshot.get("block"), Some(&2));
    assert_eq!(snapshot.get("timeout"), Some(&0));
    assert_eq!(snapshot.get("success"), Some(&0));
}
