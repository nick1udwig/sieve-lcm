use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::{Value, json};
use sieve_lcm::db::config::LcmConfig;
use sieve_lcm::engine::{ContextEngineInfo, ConversationLookupApi, LcmContextEngineApi};
use sieve_lcm::expansion_auth::{
    CreateDelegatedExpansionGrantInput, create_delegated_expansion_grant,
    reset_delegated_expansion_grants_for_tests, revoke_delegated_expansion_grant_for_session,
};
use sieve_lcm::retrieval::{
    DescribeResult, DescribeResultType, DescribeSummary, ExpandInput, ExpandResult, GrepInput,
    GrepResult, RetrievalApi,
};
use sieve_lcm::store::conversation_store::ConversationRecord;
use sieve_lcm::store::summary_store::{SummaryKind, SummarySearchResult};
use sieve_lcm::tools::lcm_expand_tool::create_lcm_expand_tool;
use sieve_lcm::types::{
    CompletionRequest, CompletionResult, GatewayCallRequest, LcmDependencies, LcmLogger, ModelRef,
};

static TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

const MAIN_SESSION_RESTRICTION_ERROR: &str = "lcm_expand is only available in sub-agent sessions. Use lcm_expand_query to ask a focused question against expanded summaries, or lcm_describe/lcm_grep for lighter lookups.";

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

    fn read_latest_assistant_reply(&self, _messages: &[Value]) -> Option<String> {
        None
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
    expand_calls: Mutex<Vec<ExpandInput>>,
    describe_calls: Mutex<Vec<String>>,
    grep_result: Mutex<GrepResult>,
    expand_results: Mutex<Vec<ExpandResult>>,
    describe_result: Mutex<Option<DescribeResult>>,
}

impl Default for MockRetrieval {
    fn default() -> Self {
        Self {
            grep_calls: Mutex::new(vec![]),
            expand_calls: Mutex::new(vec![]),
            describe_calls: Mutex::new(vec![]),
            grep_result: Mutex::new(GrepResult {
                messages: vec![],
                summaries: vec![],
                total_matches: 0,
            }),
            expand_results: Mutex::new(vec![]),
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

    async fn expand(&self, input: ExpandInput) -> anyhow::Result<ExpandResult> {
        self.expand_calls.lock().push(input);
        let mut queue = self.expand_results.lock();
        if queue.is_empty() {
            return Ok(ExpandResult {
                children: vec![],
                messages: vec![],
                estimated_tokens: 0,
                truncated: false,
            });
        }
        Ok(queue.remove(0))
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
    retrieval: Arc<MockRetrieval>,
    conversation_lookup: Arc<MockConversationLookup>,
    info: ContextEngineInfo,
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
            retrieval,
            conversation_lookup: Arc::new(MockConversationLookup { conversation }),
            info: ContextEngineInfo {
                id: "lcm".to_string(),
                name: "LCM".to_string(),
                version: "0.0.0".to_string(),
                owns_compaction: true,
            },
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

fn summary_describe(conversation_id: i64) -> DescribeResult {
    DescribeResult {
        id: "sum_a".to_string(),
        result: DescribeResultType::Summary(DescribeSummary {
            conversation_id,
            kind: SummaryKind::Leaf,
            content: "summary".to_string(),
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

fn single_expand_result(tokens: i64) -> ExpandResult {
    ExpandResult {
        children: vec![],
        messages: vec![],
        estimated_tokens: tokens,
        truncated: false,
    }
}

#[tokio::test]
async fn rejects_lcm_expand_from_main_sessions() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    let retrieval = Arc::new(MockRetrieval::default());
    let gateway_calls = Arc::new(Mutex::new(0_usize));
    let calls_ref = gateway_calls.clone();
    let deps = make_deps(Arc::new(move |_request| {
        *calls_ref.lock() += 1;
        Ok(json!({}))
    }));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), None));
    let tool = create_lcm_expand_tool(deps, engine, Some("agent:main:main".to_string()), None);
    let result = tool
        .execute("call-main-rejected", json!({ "summaryIds": ["sum_a"] }))
        .await
        .expect("execute");
    assert_eq!(
        result.details.get("error").and_then(Value::as_str),
        Some(MAIN_SESSION_RESTRICTION_ERROR)
    );
    assert!(retrieval.expand_calls.lock().is_empty());
    assert!(retrieval.grep_calls.lock().is_empty());
}

#[tokio::test]
async fn uses_remaining_grant_tokencap_when_omitted_for_summary_expansion() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval
        .expand_results
        .lock()
        .push(single_expand_result(40));
    create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
        delegated_session_key: "agent:main:subagent:unbounded".to_string(),
        issuer_session_id: "main".to_string(),
        allowed_conversation_ids: vec![7],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: Some(120),
        ttl_ms: None,
    });
    let deps = make_deps(Arc::new(move |_request| Ok(json!({}))));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), None));
    let tool = create_lcm_expand_tool(
        deps,
        engine,
        Some("agent:main:subagent:unbounded".to_string()),
        None,
    );
    tool.execute(
        "call-1",
        json!({ "summaryIds": ["sum_a"], "conversationId": 7 }),
    )
    .await
    .expect("execute");
    let calls = retrieval.expand_calls.lock();
    assert!(calls.len() >= 1);
    assert_eq!(calls[0].summary_id, "sum_a");
    assert_eq!(calls[0].token_cap, Some(120));
}

#[tokio::test]
async fn clamps_oversized_tokencap_for_query_expansion_to_remaining_grant_budget() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.grep_result.lock() = GrepResult {
        messages: vec![],
        summaries: vec![SummarySearchResult {
            summary_id: "sum_match".to_string(),
            conversation_id: 7,
            kind: SummaryKind::Leaf,
            snippet: "match".to_string(),
            created_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                .unwrap()
                .with_timezone(&Utc),
            rank: None,
        }],
        total_matches: 1,
    };
    retrieval
        .expand_results
        .lock()
        .push(single_expand_result(25));
    create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
        delegated_session_key: "agent:main:subagent:query".to_string(),
        issuer_session_id: "main".to_string(),
        allowed_conversation_ids: vec![7],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: Some(120),
        ttl_ms: None,
    });
    let deps = make_deps(Arc::new(move |_request| Ok(json!({}))));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), None));
    let tool = create_lcm_expand_tool(
        deps,
        engine,
        Some("agent:main:subagent:query".to_string()),
        None,
    );
    tool.execute(
        "call-2",
        json!({ "query": "auth", "conversationId": 7, "tokenCap": 9_999 }),
    )
    .await
    .expect("execute");
    let calls = retrieval.expand_calls.lock();
    assert!(calls.len() >= 1);
    assert_eq!(calls[0].summary_id, "sum_match");
    assert_eq!(calls[0].token_cap, Some(120));
}

#[tokio::test]
async fn rejects_delegated_subagent_expansion_when_no_grant_is_propagated() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    let retrieval = Arc::new(MockRetrieval::default());
    let deps = make_deps(Arc::new(move |_request| Ok(json!({}))));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), None));
    let tool = create_lcm_expand_tool(
        deps,
        engine,
        Some("agent:main:subagent:no-grant".to_string()),
        None,
    );
    let result = tool
        .execute("call-missing-grant", json!({ "summaryIds": ["sum_a"] }))
        .await
        .expect("execute");
    assert!(
        result
            .details
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .contains("requires a valid grant")
    );
    assert!(retrieval.expand_calls.lock().is_empty());
}

#[tokio::test]
async fn allows_delegated_subagent_expansion_with_a_valid_grant() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.describe_result.lock() = Some(summary_describe(42));
    retrieval
        .expand_results
        .lock()
        .push(single_expand_result(40));
    create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
        delegated_session_key: "agent:main:subagent:granted".to_string(),
        issuer_session_id: "main".to_string(),
        allowed_conversation_ids: vec![42],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: Some(120),
        ttl_ms: None,
    });
    let deps = make_deps(Arc::new(move |_request| Ok(json!({}))));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), None));
    let tool = create_lcm_expand_tool(
        deps,
        engine,
        Some("agent:main:subagent:granted".to_string()),
        None,
    );
    let result = tool
        .execute(
            "call-valid-grant",
            json!({ "summaryIds": ["sum_a"], "conversationId": 42 }),
        )
        .await
        .expect("execute");
    assert_eq!(retrieval.expand_calls.lock().len(), 1);
    assert_eq!(
        result.details.get("expansionCount").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        result.details.get("totalTokens").and_then(Value::as_i64),
        Some(40)
    );
    assert_eq!(
        result.details.get("truncated").and_then(Value::as_bool),
        Some(false)
    );
}

#[tokio::test]
async fn rejects_delegated_expansion_with_an_expired_grant() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.describe_result.lock() = Some(summary_describe(42));
    create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
        delegated_session_key: "agent:main:subagent:expired".to_string(),
        issuer_session_id: "main".to_string(),
        allowed_conversation_ids: vec![42],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: Some(0),
    });
    let deps = make_deps(Arc::new(move |_request| Ok(json!({}))));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), None));
    let tool = create_lcm_expand_tool(
        deps,
        engine,
        Some("agent:main:subagent:expired".to_string()),
        None,
    );
    let result = tool
        .execute(
            "call-expired-grant",
            json!({ "summaryIds": ["sum_a"], "conversationId": 42 }),
        )
        .await
        .expect("execute");
    let error = result
        .details
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(error.to_lowercase().contains("authorization failed"));
    assert!(retrieval.expand_calls.lock().is_empty());
}

#[tokio::test]
async fn rejects_delegated_expansion_with_a_revoked_grant() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.describe_result.lock() = Some(summary_describe(42));
    create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
        delegated_session_key: "agent:main:subagent:revoked".to_string(),
        issuer_session_id: "main".to_string(),
        allowed_conversation_ids: vec![42],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: None,
    });
    revoke_delegated_expansion_grant_for_session("agent:main:subagent:revoked", false);
    let deps = make_deps(Arc::new(move |_request| Ok(json!({}))));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), None));
    let tool = create_lcm_expand_tool(
        deps,
        engine,
        Some("agent:main:subagent:revoked".to_string()),
        None,
    );
    let result = tool
        .execute(
            "call-revoked-grant",
            json!({ "summaryIds": ["sum_a"], "conversationId": 42 }),
        )
        .await
        .expect("execute");
    let error = result
        .details
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(error.to_lowercase().contains("authorization failed"));
    assert!(retrieval.expand_calls.lock().is_empty());
}

#[tokio::test]
async fn rejects_delegated_expansion_outside_conversation_scope() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    let retrieval = Arc::new(MockRetrieval::default());
    create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
        delegated_session_key: "agent:main:subagent:conversation-scope".to_string(),
        issuer_session_id: "main".to_string(),
        allowed_conversation_ids: vec![7],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: Some(120),
        ttl_ms: None,
    });
    let deps = make_deps(Arc::new(move |_request| Ok(json!({}))));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), None));
    let tool = create_lcm_expand_tool(
        deps,
        engine,
        Some("agent:main:subagent:conversation-scope".to_string()),
        None,
    );
    let result = tool
        .execute(
            "call-conv-scope",
            json!({ "summaryIds": ["sum_a"], "conversationId": 8 }),
        )
        .await
        .expect("execute");
    let error = result
        .details
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(error.to_lowercase().contains("conversation 8"));
    assert!(retrieval.expand_calls.lock().is_empty());
}

#[tokio::test]
async fn clamps_delegated_expansion_tokencap_to_grant_budget() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval
        .expand_results
        .lock()
        .push(single_expand_result(5));
    create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
        delegated_session_key: "agent:main:subagent:token-cap".to_string(),
        issuer_session_id: "main".to_string(),
        allowed_conversation_ids: vec![7],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: Some(50),
        ttl_ms: None,
    });
    let deps = make_deps(Arc::new(move |_request| Ok(json!({}))));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), None));
    let tool = create_lcm_expand_tool(
        deps,
        engine,
        Some("agent:main:subagent:token-cap".to_string()),
        None,
    );
    let result = tool
        .execute(
            "call-token-cap",
            json!({ "summaryIds": ["sum_a"], "conversationId": 7, "tokenCap": 120 }),
        )
        .await
        .expect("execute");
    assert_eq!(
        result.details.get("expansionCount").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        result.details.get("totalTokens").and_then(Value::as_i64),
        Some(5)
    );
    assert_eq!(
        result.details.get("truncated").and_then(Value::as_bool),
        Some(false)
    );
    let calls = retrieval.expand_calls.lock();
    assert!(calls.len() >= 1);
    assert_eq!(calls[0].summary_id, "sum_a");
    assert_eq!(calls[0].token_cap, Some(50));
}

#[tokio::test]
async fn keeps_route_only_query_probes_local_when_no_matches() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.grep_result.lock() = GrepResult {
        messages: vec![],
        summaries: vec![],
        total_matches: 0,
    };
    create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
        delegated_session_key: "agent:main:subagent:route-only".to_string(),
        issuer_session_id: "main".to_string(),
        allowed_conversation_ids: vec![7],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: Some(120),
        ttl_ms: None,
    });
    let call_gateway_count = Arc::new(Mutex::new(0_usize));
    let count_ref = call_gateway_count.clone();
    let deps = make_deps(Arc::new(move |_request| {
        *count_ref.lock() += 1;
        Ok(json!({}))
    }));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), None));
    let tool = create_lcm_expand_tool(
        deps,
        engine,
        Some("agent:main:subagent:route-only".to_string()),
        None,
    );
    let result = tool
        .execute(
            "call-route-only",
            json!({ "query": "nothing to see", "conversationId": 7, "tokenCap": 120 }),
        )
        .await
        .expect("execute");
    assert!(retrieval.expand_calls.lock().is_empty());
    assert_eq!(*call_gateway_count.lock(), 0);
    assert_eq!(
        result.details.get("expansionCount").and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        result.details.get("executionPath").and_then(Value::as_str),
        Some("direct")
    );
    assert_eq!(
        result
            .details
            .get("policy")
            .and_then(|v| v.get("action"))
            .and_then(Value::as_str),
        Some("answer_directly")
    );
}

#[tokio::test]
async fn expands_directly_from_subagent_sessions_when_policy_suggests_delegation() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.grep_result.lock() = GrepResult {
        messages: vec![],
        summaries: vec![
            SummarySearchResult {
                summary_id: "sum_1".to_string(),
                conversation_id: 7,
                kind: SummaryKind::Leaf,
                snippet: "1".to_string(),
                created_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                    .unwrap()
                    .with_timezone(&Utc),
                rank: None,
            },
            SummarySearchResult {
                summary_id: "sum_2".to_string(),
                conversation_id: 7,
                kind: SummaryKind::Leaf,
                snippet: "2".to_string(),
                created_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                    .unwrap()
                    .with_timezone(&Utc),
                rank: None,
            },
            SummarySearchResult {
                summary_id: "sum_3".to_string(),
                conversation_id: 7,
                kind: SummaryKind::Leaf,
                snippet: "3".to_string(),
                created_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                    .unwrap()
                    .with_timezone(&Utc),
                rank: None,
            },
            SummarySearchResult {
                summary_id: "sum_4".to_string(),
                conversation_id: 7,
                kind: SummaryKind::Leaf,
                snippet: "4".to_string(),
                created_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                    .unwrap()
                    .with_timezone(&Utc),
                rank: None,
            },
            SummarySearchResult {
                summary_id: "sum_5".to_string(),
                conversation_id: 7,
                kind: SummaryKind::Leaf,
                snippet: "5".to_string(),
                created_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                    .unwrap()
                    .with_timezone(&Utc),
                rank: None,
            },
            SummarySearchResult {
                summary_id: "sum_6".to_string(),
                conversation_id: 7,
                kind: SummaryKind::Leaf,
                snippet: "6".to_string(),
                created_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                    .unwrap()
                    .with_timezone(&Utc),
                rank: None,
            },
        ],
        total_matches: 6,
    };
    retrieval
        .expand_results
        .lock()
        .push(single_expand_result(10));
    create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
        delegated_session_key: "agent:main:subagent:direct-only".to_string(),
        issuer_session_id: "main".to_string(),
        allowed_conversation_ids: vec![7],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: Some(120),
        ttl_ms: None,
    });
    let call_gateway_count = Arc::new(Mutex::new(0_usize));
    let count_ref = call_gateway_count.clone();
    let deps = make_deps(Arc::new(move |_request| {
        *count_ref.lock() += 1;
        Ok(json!({}))
    }));
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), None));
    let tool = create_lcm_expand_tool(
        deps,
        engine,
        Some("agent:main:subagent:direct-only".to_string()),
        None,
    );
    let result = tool
        .execute(
            "call-delegated",
            json!({ "query": "deep chain", "conversationId": 7, "maxDepth": 6, "tokenCap": 120 }),
        )
        .await
        .expect("execute");
    assert!(!retrieval.expand_calls.lock().is_empty());
    assert_eq!(*call_gateway_count.lock(), 0);
    assert_eq!(
        result.details.get("executionPath").and_then(Value::as_str),
        Some("direct")
    );
    assert_eq!(
        result
            .details
            .get("observability")
            .and_then(|v| v.get("decisionPath"))
            .and_then(|v| v.get("policyAction"))
            .and_then(Value::as_str),
        Some("delegate_traversal")
    );
    assert_eq!(
        result
            .details
            .get("observability")
            .and_then(|v| v.get("decisionPath"))
            .and_then(|v| v.get("executionPath"))
            .and_then(Value::as_str),
        Some("direct")
    );
}
