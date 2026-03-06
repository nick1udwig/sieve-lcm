use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::{Value, json};
use sieve_lcm::db::config::LcmConfig;
use sieve_lcm::summarize::{
    LcmSummarizeFn, LcmSummarizeOptions, LcmSummarizerLegacyParams,
    create_lcm_summarize_from_legacy_params,
};
use sieve_lcm::types::{
    CompletionRequest, CompletionResult, GatewayCallRequest, LcmDependencies, LcmLogger, ModelRef,
};

type CompleteFuture = Pin<Box<dyn Future<Output = anyhow::Result<CompletionResult>> + Send>>;
type CompleteHandler = Arc<dyn Fn(CompletionRequest) -> CompleteFuture + Send + Sync>;
type ResolveModelHandler =
    Arc<dyn Fn(Option<&str>, Option<&str>) -> anyhow::Result<ModelRef> + Send + Sync>;
type GetApiKeyHandler = Arc<dyn Fn(&str, &str) -> Option<String> + Send + Sync>;

#[derive(Default)]
struct TestLogger {
    infos: Mutex<Vec<String>>,
    warns: Mutex<Vec<String>>,
    errors: Mutex<Vec<String>>,
    debugs: Mutex<Vec<String>>,
}

#[async_trait]
impl LcmLogger for TestLogger {
    fn info(&self, msg: &str) {
        self.infos.lock().push(msg.to_string());
    }

    fn warn(&self, msg: &str) {
        self.warns.lock().push(msg.to_string());
    }

    fn error(&self, msg: &str) {
        self.errors.lock().push(msg.to_string());
    }

    fn debug(&self, msg: &str) {
        self.debugs.lock().push(msg.to_string());
    }
}

struct TestDeps {
    config: LcmConfig,
    logger: Arc<TestLogger>,
    complete_handler: Mutex<CompleteHandler>,
    resolve_model_handler: Mutex<ResolveModelHandler>,
    get_api_key_handler: Mutex<GetApiKeyHandler>,
    complete_calls: Mutex<Vec<CompletionRequest>>,
}

impl TestDeps {
    fn new() -> Self {
        let complete_handler: CompleteHandler = Arc::new(|_request| {
            Box::pin(async {
                Ok(completion_result(json!({
                    "content": [{ "type": "text", "text": "summary output" }]
                })))
            })
        });
        let resolve_model_handler: ResolveModelHandler = Arc::new(|_model_ref, _provider_hint| {
            Ok(ModelRef {
                provider: "anthropic".to_string(),
                model: "claude-opus-4-5".to_string(),
            })
        });
        let get_api_key_handler: GetApiKeyHandler =
            Arc::new(|_provider, _model| Some("test-api-key".to_string()));

        Self {
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
            logger: Arc::new(TestLogger::default()),
            complete_handler: Mutex::new(complete_handler),
            resolve_model_handler: Mutex::new(resolve_model_handler),
            get_api_key_handler: Mutex::new(get_api_key_handler),
            complete_calls: Mutex::new(vec![]),
        }
    }

    fn set_complete_handler(&self, handler: CompleteHandler) {
        *self.complete_handler.lock() = handler;
    }

    fn set_resolve_model_handler(&self, handler: ResolveModelHandler) {
        *self.resolve_model_handler.lock() = handler;
    }

    fn set_get_api_key_handler(&self, handler: GetApiKeyHandler) {
        *self.get_api_key_handler.lock() = handler;
    }

    fn complete_calls(&self) -> Vec<CompletionRequest> {
        self.complete_calls.lock().clone()
    }

    fn error_log_text(&self) -> String {
        self.logger.errors.lock().join(" ")
    }
}

#[async_trait]
impl LcmDependencies for TestDeps {
    fn config(&self) -> &LcmConfig {
        &self.config
    }

    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResult> {
        self.complete_calls.lock().push(request.clone());
        let handler = self.complete_handler.lock().clone();
        handler(request).await
    }

    async fn call_gateway(&self, _request: GatewayCallRequest) -> anyhow::Result<Value> {
        Ok(json!({}))
    }

    fn resolve_model(
        &self,
        model_ref: Option<&str>,
        provider_hint: Option<&str>,
    ) -> anyhow::Result<ModelRef> {
        let handler = self.resolve_model_handler.lock().clone();
        handler(model_ref, provider_hint)
    }

    fn get_api_key(&self, provider: &str, model: &str) -> Option<String> {
        let handler = self.get_api_key_handler.lock().clone();
        handler(provider, model)
    }

    fn require_api_key(&self, provider: &str, model: &str) -> anyhow::Result<String> {
        Ok(self.get_api_key(provider, model).unwrap_or_default())
    }

    fn parse_agent_session_key(&self, _session_key: &str) -> Option<(String, String)> {
        None
    }

    fn is_subagent_session_key(&self, _session_key: &str) -> bool {
        false
    }

    fn normalize_agent_id(&self, _id: Option<&str>) -> String {
        "main".to_string()
    }

    fn build_subagent_system_prompt(
        &self,
        _depth: i32,
        _max_depth: i32,
        _task_summary: Option<&str>,
    ) -> String {
        String::new()
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
        self.logger.as_ref()
    }
}

fn completion_result(value: Value) -> CompletionResult {
    serde_json::from_value(value).expect("valid completion result")
}

fn make_deps() -> Arc<TestDeps> {
    Arc::new(TestDeps::new())
}

fn basic_legacy(provider: &str, model: &str) -> LcmSummarizerLegacyParams {
    LcmSummarizerLegacyParams {
        provider: Some(provider.to_string()),
        model: Some(model.to_string()),
        config: None,
        agent_dir: None,
        auth_profile_id: None,
    }
}

async fn build_summarizer(
    deps: Arc<TestDeps>,
    legacy_params: LcmSummarizerLegacyParams,
    custom_instructions: Option<&str>,
) -> LcmSummarizeFn {
    create_lcm_summarize_from_legacy_params(
        deps,
        legacy_params,
        custom_instructions.map(|v| v.to_string()),
    )
    .await
    .expect("create summarize")
    .expect("summarizer present")
}

fn first_prompt(calls: &[CompletionRequest]) -> String {
    calls[0].messages[0]
        .content
        .as_str()
        .unwrap_or_default()
        .to_string()
}

#[tokio::test]
async fn returns_none_when_model_resolution_fails() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| Err(anyhow::anyhow!("no model"))));

    let summarize = create_lcm_summarize_from_legacy_params(
        deps,
        basic_legacy("anthropic", "claude-opus-4-5"),
        None,
    )
    .await
    .expect("create summarize");

    let _ = summarize;
}

#[tokio::test]
async fn builds_distinct_normal_vs_aggressive_prompts() {
    let deps = make_deps();
    let summarize = create_lcm_summarize_from_legacy_params(
        deps.clone(),
        basic_legacy("anthropic", "claude-opus-4-5"),
        Some("Keep implementation caveats.".to_string()),
    )
    .await
    .expect("create summarize");
    assert!(summarize.is_some());
    let summarize = summarize.expect("summarizer present");

    summarize("A".repeat(8_000), false, None).await;
    summarize("A".repeat(8_000), true, None).await;

    let calls = deps.complete_calls();
    assert_eq!(calls.len(), 2);
    let normal_prompt = calls[0].messages[0]
        .content
        .as_str()
        .unwrap_or_default()
        .to_string();
    let aggressive_prompt = calls[1].messages[0]
        .content
        .as_str()
        .unwrap_or_default()
        .to_string();
    let system_prompt = calls[0].system.as_deref().unwrap_or_default();

    assert!(normal_prompt.contains("Normal summary policy:"));
    assert!(aggressive_prompt.contains("Aggressive summary policy:"));
    assert!(normal_prompt.contains("Keep implementation caveats."));
    assert!(system_prompt.contains("context-compaction summarization engine"));
    assert!(calls[1].max_tokens < calls[0].max_tokens);
    assert_eq!(calls[1].temperature, Some(0.1));
}

#[tokio::test]
async fn uses_condensed_prompt_mode_for_condensed_summaries() {
    let deps = make_deps();
    let summarize = build_summarizer(
        deps.clone(),
        basic_legacy("anthropic", "claude-opus-4-5"),
        None,
    )
    .await;

    summarize(
        "A".repeat(8_000),
        false,
        Some(LcmSummarizeOptions {
            previous_summary: None,
            is_condensed: Some(true),
            depth: None,
        }),
    )
    .await;

    let calls = deps.complete_calls();
    assert_eq!(calls.len(), 1);
    let prompt = first_prompt(&calls);
    assert!(prompt.contains("<conversation_to_condense>"));
    assert_eq!(calls[0].reasoning, None);
}

#[tokio::test]
async fn passes_resolved_api_key_to_completion_calls() {
    let deps = make_deps();
    deps.set_get_api_key_handler(Arc::new(|_, _| Some("resolved-api-key".to_string())));
    let summarize = build_summarizer(
        deps.clone(),
        basic_legacy("anthropic", "claude-opus-4-5"),
        None,
    )
    .await;

    summarize("Summary input".to_string(), false, None).await;

    let calls = deps.complete_calls();
    assert_eq!(calls[0].api_key.as_deref(), Some("resolved-api-key"));
}

#[tokio::test]
async fn falls_back_deterministically_when_model_returns_empty_after_retry() {
    let deps = make_deps();
    deps.set_complete_handler(Arc::new(|_request| {
        Box::pin(async { Ok(completion_result(json!({ "content": [] }))) })
    }));
    let summarize = build_summarizer(
        deps.clone(),
        basic_legacy("anthropic", "claude-opus-4-5"),
        None,
    )
    .await;

    let summary = summarize("A".repeat(12_000), false, None).await;
    assert_eq!(deps.complete_calls().len(), 2);
    assert!(summary.contains("[LCM fallback summary; truncated for context management]"));
    let result = SummaryResult {
        summary: Some(summary),
    };
    assert!(result.summary.is_some());
}

#[derive(Clone, Debug)]
struct SummaryResult {
    summary: Option<String>,
}

#[tokio::test]
async fn normalizes_openai_output_text_and_reasoning_summary_blocks() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| {
        Ok(ModelRef {
            provider: "openai".to_string(),
            model: "gpt-5.3-codex".to_string(),
        })
    }));
    deps.set_complete_handler(Arc::new(|_request| {
        Box::pin(async {
            Ok(completion_result(json!({
                "content": [
                    {
                        "type": "reasoning",
                        "summary": [{ "type": "summary_text", "text": "Reasoning summary line." }]
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": "Final condensed summary." }]
                    }
                ]
            })))
        })
    }));
    let summarize = build_summarizer(deps, basic_legacy("openai", "gpt-5.3-codex"), None).await;
    let summary = summarize("Input segment".to_string(), false, None).await;

    assert!(summary.contains("Reasoning summary line."));
    assert!(summary.contains("Final condensed summary."));
}

#[tokio::test]
async fn logs_provider_model_block_diagnostics_when_summary_is_empty() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| {
        Ok(ModelRef {
            provider: "openai".to_string(),
            model: "gpt-5.3-codex".to_string(),
        })
    }));
    deps.set_complete_handler(Arc::new(|_request| {
        Box::pin(async {
            Ok(completion_result(json!({
                "content": [{ "type": "reasoning" }]
            })))
        })
    }));
    let summarize =
        build_summarizer(deps.clone(), basic_legacy("openai", "gpt-5.3-codex"), None).await;

    let summary = summarize("A".repeat(12_000), false, None).await;
    assert!(summary.contains("[LCM fallback summary; truncated for context management]"));

    let diagnostics = deps.error_log_text();
    assert!(diagnostics.contains("provider=openai"));
    assert!(diagnostics.contains("model=gpt-5.3-codex"));
    assert!(diagnostics.contains("block_types=reasoning"));
    assert!(diagnostics.contains("content_preview="));
}

#[tokio::test]
async fn retries_with_conservative_settings_when_first_attempt_returns_empty_array() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| {
        Ok(ModelRef {
            provider: "openai".to_string(),
            model: "gpt-5.3-codex".to_string(),
        })
    }));

    let call_count = Arc::new(Mutex::new(0));
    let call_count_cloned = call_count.clone();
    deps.set_complete_handler(Arc::new(move |_request| {
        let call_count_cloned = call_count_cloned.clone();
        Box::pin(async move {
            let mut count = call_count_cloned.lock();
            *count += 1;
            if *count == 1 {
                Ok(completion_result(json!({ "content": [] })))
            } else {
                Ok(completion_result(json!({
                    "content": [{ "type": "text", "text": "Recovered summary from retry." }]
                })))
            }
        })
    }));

    let summarize =
        build_summarizer(deps.clone(), basic_legacy("openai", "gpt-5.3-codex"), None).await;
    let summary = summarize("A".repeat(8_000), false, None).await;

    assert_eq!(summary, "Recovered summary from retry.");
    let calls = deps.complete_calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1].temperature, Some(0.05));
    assert_eq!(calls[1].reasoning.as_deref(), Some("low"));
    assert!(deps.error_log_text().contains("retry succeeded"));
}

#[tokio::test]
async fn falls_back_to_truncation_when_retry_returns_non_text_only_blocks() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| {
        Ok(ModelRef {
            provider: "openai".to_string(),
            model: "openai-codex".to_string(),
        })
    }));
    deps.set_complete_handler(Arc::new(|_request| {
        Box::pin(async {
            Ok(completion_result(json!({
                "content": [{ "type": "tool_use", "id": "tu_1", "name": "bash", "input": { "cmd": "ls" } }]
            })))
        })
    }));

    let summarize =
        build_summarizer(deps.clone(), basic_legacy("openai", "openai-codex"), None).await;
    let summary = summarize("B".repeat(10_000), false, None).await;

    assert_eq!(deps.complete_calls().len(), 2);
    assert!(summary.contains("[LCM fallback summary; truncated for context management]"));
    let diagnostics = deps.error_log_text();
    assert!(diagnostics.contains("empty normalized summary on first attempt"));
    assert!(diagnostics.contains("retry also returned empty summary"));
    assert!(diagnostics.contains("block_types=tool_use"));
    assert!(diagnostics.contains("\"type\":\"tool_use\""));
}

#[tokio::test]
async fn falls_back_gracefully_when_retry_throws_exception() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| {
        Ok(ModelRef {
            provider: "openai".to_string(),
            model: "gpt-5.3-codex".to_string(),
        })
    }));

    let call_count = Arc::new(Mutex::new(0));
    let call_count_cloned = call_count.clone();
    deps.set_complete_handler(Arc::new(move |_request| {
        let call_count_cloned = call_count_cloned.clone();
        Box::pin(async move {
            let mut count = call_count_cloned.lock();
            *count += 1;
            if *count == 1 {
                Ok(completion_result(json!({ "content": [] })))
            } else {
                Err(anyhow::anyhow!("rate limit exceeded"))
            }
        })
    }));

    let summarize =
        build_summarizer(deps.clone(), basic_legacy("openai", "gpt-5.3-codex"), None).await;
    let summary = summarize("C".repeat(10_000), false, None).await;
    assert!(summary.contains("[LCM fallback summary; truncated for context management]"));

    let diagnostics = deps.error_log_text();
    assert!(diagnostics.contains("retry failed"));
    assert!(diagnostics.contains("rate limit exceeded"));
}

#[tokio::test]
async fn logs_response_envelope_metadata_in_diagnostics() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| {
        Ok(ModelRef {
            provider: "openai".to_string(),
            model: "gpt-5.3-codex".to_string(),
        })
    }));
    deps.set_complete_handler(Arc::new(|_request| {
        Box::pin(async {
            Ok(completion_result(json!({
                "content": [],
                "id": "req_abc123",
                "provider": "openai-codex",
                "model": "gpt-5.3-codex-20260101",
                "request_provider": "openai-codex",
                "request_model": "gpt-5.3-codex",
                "request_api": "openai-codex-responses",
                "request_reasoning": "low",
                "request_has_system": "true",
                "request_temperature": "(omitted)",
                "request_temperature_sent": "false",
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 0,
                    "total_tokens": 500,
                    "input": 500,
                    "output": 0
                },
                "stopReason": "stop",
                "errorMessage": "upstream timeout while acquiring provider connection",
                "error": { "code": "provider_timeout", "retriable": true }
            })))
        })
    }));

    let summarize =
        build_summarizer(deps.clone(), basic_legacy("openai", "gpt-5.3-codex"), None).await;
    summarize("D".repeat(8_000), false, None).await;

    let diagnostics = deps.error_log_text();
    assert!(diagnostics.contains("id=req_abc123"));
    assert!(diagnostics.contains("resp_provider=openai-codex"));
    assert!(diagnostics.contains("resp_model=gpt-5.3-codex-20260101"));
    assert!(diagnostics.contains("request_api=openai-codex-responses"));
    assert!(diagnostics.contains("request_reasoning=low"));
    assert!(diagnostics.contains("request_has_system=true"));
    assert!(diagnostics.contains("request_temperature=(omitted)"));
    assert!(diagnostics.contains("request_temperature_sent=false"));
    assert!(diagnostics.contains("completion_tokens=0"));
    assert!(diagnostics.contains("input=500"));
    assert!(diagnostics.contains("finish=stop"));
    assert!(diagnostics.contains("error_message=upstream timeout"));
    assert!(diagnostics.contains("error_preview="));
}

#[tokio::test]
async fn redacts_sensitive_keys_from_diagnostic_content_previews() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| {
        Ok(ModelRef {
            provider: "openai".to_string(),
            model: "gpt-5.3-codex".to_string(),
        })
    }));
    deps.set_complete_handler(Arc::new(|_request| {
        Box::pin(async {
            Ok(completion_result(json!({
                "content": [
                    {
                        "type": "tool_use",
                        "name": "http",
                        "input": {
                            "authorization": "Bearer super-secret-token",
                            "body": "x".repeat(1500)
                        }
                    }
                ]
            })))
        })
    }));

    let summarize =
        build_summarizer(deps.clone(), basic_legacy("openai", "gpt-5.3-codex"), None).await;
    summarize("E".repeat(8_000), false, None).await;

    let diagnostics = deps.error_log_text();
    assert!(diagnostics.contains("content_preview="));
    assert!(diagnostics.contains("\"authorization\":\"[redacted]\""));
    assert!(!diagnostics.contains("super-secret-token"));
    assert!(diagnostics.contains("[truncated:"));
}

#[tokio::test]
async fn does_not_retry_when_anthropic_returns_valid_summary() {
    let deps = make_deps();
    let summarize = build_summarizer(
        deps.clone(),
        basic_legacy("anthropic", "claude-opus-4-5"),
        None,
    )
    .await;

    let summary = summarize("Some conversation text".to_string(), false, None).await;
    assert_eq!(summary, "summary output");
    assert_eq!(deps.complete_calls().len(), 1);
}

#[tokio::test]
async fn recovers_summary_from_top_level_output_text_when_content_is_empty() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| {
        Ok(ModelRef {
            provider: "openai".to_string(),
            model: "gpt-5.3-codex".to_string(),
        })
    }));
    deps.set_complete_handler(Arc::new(|_request| {
        Box::pin(async {
            Ok(completion_result(json!({
                "content": [],
                "output_text": "Summary recovered from envelope output_text."
            })))
        })
    }));

    let summarize =
        build_summarizer(deps.clone(), basic_legacy("openai", "gpt-5.3-codex"), None).await;
    let summary = summarize("A".repeat(8_000), false, None).await;

    assert_eq!(summary, "Summary recovered from envelope output_text.");
    assert_eq!(deps.complete_calls().len(), 1);
    let diagnostics = deps.error_log_text();
    assert!(diagnostics.contains("source=envelope"));
    assert!(diagnostics.contains("recovered summary from response envelope"));
    assert!(!diagnostics.contains("retrying with conservative settings"));
}

#[tokio::test]
async fn recovers_summary_from_output_array_when_content_is_empty() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| {
        Ok(ModelRef {
            provider: "openai".to_string(),
            model: "openai-codex".to_string(),
        })
    }));
    deps.set_complete_handler(Arc::new(|_request| {
        Box::pin(async {
            Ok(completion_result(json!({
                "content": [],
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            { "type": "output_text", "text": "Summary from output message." }
                        ]
                    }
                ]
            })))
        })
    }));

    let summarize =
        build_summarizer(deps.clone(), basic_legacy("openai", "openai-codex"), None).await;
    let summary = summarize("B".repeat(8_000), false, None).await;

    assert_eq!(summary, "Summary from output message.");
    assert_eq!(deps.complete_calls().len(), 1);
    assert!(deps.error_log_text().contains("source=envelope"));
}

#[tokio::test]
async fn recovers_from_envelope_when_content_has_reasoning_only_blocks() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| {
        Ok(ModelRef {
            provider: "openai".to_string(),
            model: "gpt-5.3-codex".to_string(),
        })
    }));
    deps.set_complete_handler(Arc::new(|_request| {
        Box::pin(async {
            Ok(completion_result(json!({
                "content": [{ "type": "reasoning" }],
                "output": [
                    { "type": "reasoning", "summary": [] },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": "Actual summary after reasoning." }]
                    }
                ]
            })))
        })
    }));

    let summarize =
        build_summarizer(deps.clone(), basic_legacy("openai", "gpt-5.3-codex"), None).await;
    let summary = summarize("C".repeat(8_000), false, None).await;

    assert_eq!(summary, "Actual summary after reasoning.");
    assert_eq!(deps.complete_calls().len(), 1);
    let diagnostics = deps.error_log_text();
    assert!(diagnostics.contains("source=envelope"));
    assert!(!diagnostics.contains("retrying"));
}

#[tokio::test]
async fn proceeds_to_retry_when_envelope_also_has_no_extractable_text() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| {
        Ok(ModelRef {
            provider: "openai".to_string(),
            model: "openai-codex".to_string(),
        })
    }));
    deps.set_complete_handler(Arc::new(|_request| {
        Box::pin(async {
            Ok(completion_result(json!({
                "content": [],
                "output": [{ "type": "function_call", "name": "run_code", "call_id": "fc_1" }]
            })))
        })
    }));

    let summarize =
        build_summarizer(deps.clone(), basic_legacy("openai", "openai-codex"), None).await;
    let summary = summarize("D".repeat(10_000), false, None).await;

    assert_eq!(deps.complete_calls().len(), 2);
    assert!(summary.contains("[LCM fallback summary; truncated for context management]"));
    let diagnostics = deps.error_log_text();
    assert!(!diagnostics.contains("source=envelope"));
    assert!(diagnostics.contains("retrying with conservative settings"));
}

#[tokio::test]
async fn deduplicates_text_found_in_content_and_envelope_output() {
    let deps = make_deps();
    deps.set_resolve_model_handler(Arc::new(|_, _| {
        Ok(ModelRef {
            provider: "openai".to_string(),
            model: "gpt-5.3-codex".to_string(),
        })
    }));
    deps.set_complete_handler(Arc::new(|_request| {
        Box::pin(async {
            Ok(completion_result(json!({
                "content": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": "Deduplicated summary." }]
                    }
                ],
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": "Deduplicated summary." }]
                    }
                ],
                "output_text": "Deduplicated summary."
            })))
        })
    }));

    let summarize =
        build_summarizer(deps.clone(), basic_legacy("openai", "gpt-5.3-codex"), None).await;
    let summary = summarize("E".repeat(4_000), false, None).await;
    assert_eq!(summary, "Deduplicated summary.");
    assert_eq!(deps.complete_calls().len(), 1);
}
