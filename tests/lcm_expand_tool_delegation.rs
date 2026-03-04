use std::sync::Arc;

use async_trait::async_trait;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::{json, Value};
use sieve_lcm::db::config::LcmConfig;
use sieve_lcm::tools::lcm_expand_tool_delegation::{run_delegated_expansion_loop, DelegatedPassStatus};
use sieve_lcm::tools::lcm_expansion_recursion_guard::{
    get_expansion_delegation_telemetry_snapshot_for_tests, reset_expansion_delegation_guard_for_tests,
    stamp_delegated_expansion_context,
};
use sieve_lcm::expansion_auth::reset_delegated_expansion_grants_for_tests;
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
struct TestLogger {
    info: Mutex<Vec<String>>,
    warn: Mutex<Vec<String>>,
}

#[async_trait]
impl LcmLogger for TestLogger {
    fn info(&self, msg: &str) {
        self.info.lock().push(msg.to_string());
    }
    fn warn(&self, msg: &str) {
        self.warn.lock().push(msg.to_string());
    }
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
        logger: TestLogger::default(),
        call_gateway,
    })
}

#[tokio::test]
async fn runs_delegated_expansion_when_not_in_delegated_context() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    reset_expansion_delegation_guard_for_tests();

    let last_agent_message = Arc::new(Mutex::new(String::new()));
    let calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let message_ref = last_agent_message.clone();
    let calls_ref = calls.clone();
    let deps = make_deps(Arc::new(move |request: GatewayCallRequest| {
        calls_ref.lock().push(request.method.clone());
        match request.method.as_str() {
            "agent" => {
                let msg = request
                    .params
                    .as_ref()
                    .and_then(|v| v.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                *message_ref.lock() = msg;
                Ok(json!({ "runId": "run-pass-1" }))
            }
            "agent.wait" => Ok(json!({ "status": "ok" })),
            "sessions.get" => Ok(json!({
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "{\"summary\":\"Expansion succeeded.\",\"citedIds\":[\"sum_a\"],\"followUpSummaryIds\":[],\"totalTokens\":33,\"truncated\":false}"
                            }
                        ]
                    }
                ]
            })),
            "sessions.delete" => Ok(json!({ "ok": true })),
            _ => Ok(json!({})),
        }
    }));

    let result = run_delegated_expansion_loop(
        deps.as_ref(),
        "agent:main:main",
        7,
        vec!["sum_a".to_string()],
        None,
        None,
        false,
        None,
        None,
    )
    .await;

    assert_eq!(result.status, DelegatedPassStatus::Ok);
    assert_eq!(result.cited_ids, vec!["sum_a".to_string()]);
    let last = last_agent_message.lock().clone();
    assert!(last.contains("requestId"));
    assert!(last.contains("DO NOT call `lcm_expand_query` from this delegated session."));
    assert!(last.contains("use `lcm_expand` directly"));
    let snapshot = get_expansion_delegation_telemetry_snapshot_for_tests();
    assert_eq!(snapshot.get("start"), Some(&1));
    assert_eq!(snapshot.get("block"), Some(&0));
    assert_eq!(snapshot.get("timeout"), Some(&0));
    assert_eq!(snapshot.get("success"), Some(&1));
}

#[tokio::test]
async fn blocks_delegated_expansion_helper_reentry_at_depth_cap() {
    let _guard = TEST_LOCK.lock();
    reset_delegated_expansion_grants_for_tests();
    reset_expansion_delegation_guard_for_tests();

    stamp_delegated_expansion_context(
        "agent:main:subagent:blocked",
        "req-loop",
        1,
        "agent:main:main",
        "test",
    );

    let calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let calls_ref = calls.clone();
    let deps = make_deps(Arc::new(move |request: GatewayCallRequest| {
        calls_ref.lock().push(request.method.clone());
        Ok(json!({}))
    }));

    let result = run_delegated_expansion_loop(
        deps.as_ref(),
        "agent:main:subagent:blocked",
        7,
        vec!["sum_a".to_string()],
        None,
        None,
        false,
        None,
        Some("req-loop"),
    )
    .await;

    assert_eq!(result.status, DelegatedPassStatus::Error);
    let error = result.error.unwrap_or_default();
    assert!(error.contains("EXPANSION_RECURSION_BLOCKED"));
    assert!(error.contains("Recovery: In delegated sub-agent sessions, call `lcm_expand` directly"));
    assert!(error.contains("Do NOT call `lcm_expand_query` from delegated context."));
    assert!(calls.lock().is_empty());
    let snapshot = get_expansion_delegation_telemetry_snapshot_for_tests();
    assert_eq!(snapshot.get("start"), Some(&1));
    assert_eq!(snapshot.get("block"), Some(&1));
    assert_eq!(snapshot.get("timeout"), Some(&0));
    assert_eq!(snapshot.get("success"), Some(&0));
}
