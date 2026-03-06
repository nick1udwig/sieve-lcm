use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde_json::{Value, json};
use sieve_lcm::db::config::LcmConfig;
use sieve_lcm::engine::{ContextEngineInfo, ConversationLookupApi, LcmContextEngineApi};
use sieve_lcm::expansion_auth::{
    CreateDelegatedExpansionGrantInput, create_delegated_expansion_grant,
    reset_delegated_expansion_grants_for_tests,
};
use sieve_lcm::retrieval::{
    DescribeResult, DescribeResultType, DescribeSubtreeNode, DescribeSummary, ExpandInput,
    ExpandResult, GrepInput, GrepResult, RetrievalApi,
};
use sieve_lcm::store::conversation_store::{ConversationRecord, MessageRole, MessageSearchResult};
use sieve_lcm::store::summary_store::{SummaryKind, SummarySearchResult};
use sieve_lcm::tools::lcm_describe_tool::create_lcm_describe_tool;
use sieve_lcm::tools::lcm_expand_tool::create_lcm_expand_tool;
use sieve_lcm::tools::lcm_grep_tool::create_lcm_grep_tool;
use sieve_lcm::types::{
    CompletionRequest, CompletionResult, GatewayCallRequest, LcmDependencies, LcmLogger, ModelRef,
};

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
}

#[async_trait]
impl LcmDependencies for TestDeps {
    fn config(&self) -> &LcmConfig {
        &self.config
    }

    async fn complete(&self, _request: CompletionRequest) -> anyhow::Result<CompletionResult> {
        Ok(CompletionResult::default())
    }

    async fn call_gateway(&self, _request: GatewayCallRequest) -> anyhow::Result<Value> {
        Ok(json!({}))
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

struct MockRetrieval {
    grep_calls: Mutex<Vec<GrepInput>>,
    expand_calls: Mutex<Vec<ExpandInput>>,
    grep_result: Mutex<GrepResult>,
    expand_result: Mutex<ExpandResult>,
    describe_result: Mutex<Option<DescribeResult>>,
}

impl Default for MockRetrieval {
    fn default() -> Self {
        Self {
            grep_calls: Mutex::new(vec![]),
            expand_calls: Mutex::new(vec![]),
            grep_result: Mutex::new(GrepResult {
                messages: vec![],
                summaries: vec![],
                total_matches: 0,
            }),
            expand_result: Mutex::new(ExpandResult {
                children: vec![],
                messages: vec![],
                estimated_tokens: 0,
                truncated: false,
            }),
            describe_result: Mutex::new(None),
        }
    }
}

#[async_trait]
impl RetrievalApi for MockRetrieval {
    async fn describe(&self, _id: &str) -> anyhow::Result<Option<DescribeResult>> {
        Ok(self.describe_result.lock().clone())
    }

    async fn grep(&self, input: GrepInput) -> anyhow::Result<GrepResult> {
        self.grep_calls.lock().push(input);
        Ok(self.grep_result.lock().clone())
    }

    async fn expand(&self, input: ExpandInput) -> anyhow::Result<ExpandResult> {
        self.expand_calls.lock().push(input);
        Ok(self.expand_result.lock().clone())
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

fn make_deps() -> Arc<TestDeps> {
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
    })
}

#[tokio::test]
async fn lcm_expand_query_mode_infers_conversation_from_delegated_grant() {
    reset_delegated_expansion_grants_for_tests();

    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.grep_result.lock() = GrepResult {
        messages: vec![],
        summaries: vec![SummarySearchResult {
            summary_id: "sum_recent".to_string(),
            conversation_id: 42,
            kind: SummaryKind::Leaf,
            snippet: "recent snippet".to_string(),
            created_at: DateTime::parse_from_rfc3339("2026-01-02T00:00:00.000Z")
                .unwrap()
                .with_timezone(&Utc),
            rank: None,
        }],
        total_matches: 1,
    };
    *retrieval.expand_result.lock() = ExpandResult {
        children: vec![],
        messages: vec![],
        estimated_tokens: 5,
        truncated: false,
    };

    create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
        delegated_session_key: "agent:main:subagent:session-1".to_string(),
        issuer_session_id: "main".to_string(),
        allowed_conversation_ids: vec![42],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: Some(120),
        ttl_ms: None,
    });

    let deps = make_deps();
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval.clone(), None));
    let tool = create_lcm_expand_tool(
        deps,
        engine,
        Some("agent:main:subagent:session-1".to_string()),
        None,
    );
    let result = tool
        .execute("call-1", json!({ "query": "recent snippet" }))
        .await
        .expect("execute");

    let grep_calls = retrieval.grep_calls.lock();
    assert!(grep_calls.len() >= 1);
    assert_eq!(grep_calls[0].conversation_id, Some(42));
    assert_eq!(grep_calls[0].query, "recent snippet");
    assert_eq!(
        result
            .details
            .get("expansionCount")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        1
    );
}

#[tokio::test]
async fn lcm_grep_forwards_since_before_and_includes_iso_timestamp_in_output() {
    reset_delegated_expansion_grants_for_tests();

    let created_at = DateTime::parse_from_rfc3339("2026-01-03T00:00:00.000Z")
        .unwrap()
        .with_timezone(&Utc);
    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.grep_result.lock() = GrepResult {
        messages: vec![MessageSearchResult {
            message_id: 101,
            conversation_id: 42,
            role: MessageRole::Assistant,
            snippet: "deployment timeline".to_string(),
            created_at: created_at.clone(),
            rank: Some(0.0),
        }],
        summaries: vec![],
        total_matches: 1,
    };

    let deps = make_deps();
    let engine: Arc<dyn LcmContextEngineApi> =
        Arc::new(MockEngine::new(retrieval.clone(), Some(42)));
    let tool = create_lcm_grep_tool(deps, engine, Some("session-1".to_string()), None);
    let result = tool
        .execute(
            "call-2",
            json!({
                "pattern": "deployment",
                "since": "2026-01-01T00:00:00.000Z",
                "before": "2026-01-04T00:00:00.000Z"
            }),
        )
        .await
        .expect("execute");

    let grep_calls = retrieval.grep_calls.lock();
    assert!(grep_calls.len() >= 1);
    assert_eq!(grep_calls[0].conversation_id, Some(42));
    assert!(grep_calls[0].since.is_some());
    assert!(grep_calls[0].before.is_some());
    assert!(result.content[0].text.contains(&created_at.to_rfc3339()));
}

#[tokio::test]
async fn lcm_describe_blocks_cross_conversation_unless_all_conversations_true() {
    reset_delegated_expansion_grants_for_tests();

    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.describe_result.lock() = Some(DescribeResult {
        id: "sum_foreign".to_string(),
        result: DescribeResultType::Summary(DescribeSummary {
            conversation_id: 99,
            kind: SummaryKind::Leaf,
            content: "foreign summary".to_string(),
            depth: 0,
            token_count: 12,
            descendant_count: 0,
            descendant_token_count: 0,
            source_message_token_count: 12,
            file_ids: vec![],
            parent_ids: vec![],
            child_ids: vec![],
            message_ids: vec![],
            earliest_at: Some(
                DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                    .unwrap()
                    .with_timezone(&Utc),
            ),
            latest_at: Some(
                DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                    .unwrap()
                    .with_timezone(&Utc),
            ),
            subtree: vec![DescribeSubtreeNode {
                summary_id: "sum_foreign".to_string(),
                parent_summary_id: None,
                depth_from_root: 0,
                kind: SummaryKind::Leaf,
                depth: 0,
                token_count: 12,
                descendant_count: 0,
                descendant_token_count: 0,
                source_message_token_count: 12,
                earliest_at: Some(
                    DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                        .unwrap()
                        .with_timezone(&Utc),
                ),
                latest_at: Some(
                    DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                        .unwrap()
                        .with_timezone(&Utc),
                ),
                child_count: 0,
                path: String::new(),
            }],
            created_at: DateTime::parse_from_rfc3339("2026-01-01T00:00:00.000Z")
                .unwrap()
                .with_timezone(&Utc),
        }),
    });

    let deps = make_deps();
    let engine: Arc<dyn LcmContextEngineApi> = Arc::new(MockEngine::new(retrieval, Some(42)));
    let tool = create_lcm_describe_tool(deps, engine, Some("session-1".to_string()), None);
    let scoped = tool
        .execute("call-3", json!({ "id": "sum_foreign" }))
        .await
        .expect("scoped");
    assert!(
        scoped
            .details
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .contains("Not found in conversation 42")
    );

    let cross = tool
        .execute(
            "call-4",
            json!({
                "id": "sum_foreign",
                "allConversations": true
            }),
        )
        .await
        .expect("cross");
    assert!(cross.content[0].text.contains("meta conv=99"));
    assert!(cross.content[0].text.contains("manifest"));
}
