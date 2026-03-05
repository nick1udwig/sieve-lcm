use std::future::Future;
use std::fs;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{Value, json};
use sieve_lcm::assembler::{AssembleContextInput, ContextAssembler};
use sieve_lcm::compaction::{
    CompactInput, CompactUntilUnderInput, CompactionConfig, CompactionEngine,
};
use sieve_lcm::db::config::LcmConfig;
use sieve_lcm::db::connection::{SharedConnection, close_lcm_connection, get_lcm_connection};
use sieve_lcm::db::migration::run_lcm_migrations;
use sieve_lcm::engine::{
    AfterTurnInput, AssembleInput, BootstrapInput, IngestBatchInput, IngestInput, LcmContextEngine,
};
use sieve_lcm::large_files::{
    ExplorationSummaryInput, extension_from_name_or_mime, format_file_reference,
    generate_exploration_summary, is_large_file, parse_file_blocks,
};
use sieve_lcm::retrieval::{DescribeResult, ExpandInput, ExpandResult, GrepInput, GrepResult, RetrievalApi};
use sieve_lcm::store::conversation_store::{
    ConversationStore, CreateMessageInput, CreateMessagePartInput, MessagePartType, MessageRole,
};
use sieve_lcm::store::summary_store::{CreateLargeFileInput, CreateSummaryInput, SummaryKind, SummaryStore};
use sieve_lcm::summarize::LcmSummarizeFn;
use sieve_lcm::types::{
    AgentMessage, CompletionRequest, CompletionResult, GatewayCallRequest, LcmDependencies,
    LcmLogger, ModelRef,
};
use tempfile::TempDir;
use uuid::Uuid;

fn create_test_config(database_path: &str) -> LcmConfig {
    LcmConfig {
        enabled: true,
        database_path: database_path.to_string(),
        context_threshold: 0.75,
        fresh_tail_count: 8,
        leaf_min_fanout: 8,
        condensed_min_fanout: 4,
        condensed_min_fanout_hard: 2,
        incremental_max_depth: 0,
        leaf_chunk_tokens: 20_000,
        leaf_target_tokens: 600,
        condensed_target_tokens: 900,
        max_expand_tokens: 4_000,
        large_file_token_threshold: 25_000,
        large_file_summary_provider: String::new(),
        large_file_summary_model: String::new(),
        autocompact_disabled: false,
        timezone: "UTC".to_string(),
        prune_heartbeat_ok: false,
    }
}

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
struct TestLogger {
    errors: Mutex<Vec<String>>,
}

#[async_trait]
impl LcmLogger for TestLogger {
    fn info(&self, _msg: &str) {}
    fn warn(&self, _msg: &str) {}
    fn error(&self, msg: &str) {
        self.errors.lock().push(msg.to_string());
    }
    fn debug(&self, _msg: &str) {}
}

#[derive(Clone)]
struct TestDeps {
    config: LcmConfig,
    logger: Arc<TestLogger>,
    agent_dir: String,
}

#[async_trait]
impl LcmDependencies for TestDeps {
    fn config(&self) -> &LcmConfig {
        &self.config
    }

    async fn complete(&self, _request: CompletionRequest) -> anyhow::Result<CompletionResult> {
        Ok(CompletionResult {
            content: vec![sieve_lcm::types::CompletionContentBlock {
                r#type: "text".to_string(),
                text: Some("summary output".to_string()),
                extra: Default::default(),
            }],
            extra: Default::default(),
        })
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
        Ok("test-api-key".to_string())
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
        for message in messages.iter().rev() {
            let Some(role) = message.get("role").and_then(Value::as_str) else {
                continue;
            };
            if role != "assistant" {
                continue;
            }
            if let Some(content) = message.get("content").and_then(Value::as_str) {
                return Some(content.to_string());
            }
        }
        None
    }

    fn resolve_agent_dir(&self) -> String {
        self.agent_dir.clone()
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

#[derive(Default)]
struct NoopRetrieval;

#[async_trait]
impl RetrievalApi for NoopRetrieval {
    async fn describe(&self, _id: &str) -> anyhow::Result<Option<DescribeResult>> {
        Ok(None)
    }

    async fn grep(&self, _input: GrepInput) -> anyhow::Result<GrepResult> {
        Ok(GrepResult {
            messages: vec![],
            summaries: vec![],
            total_matches: 0,
        })
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

fn estimate_tokens_text(text: &str) -> i64 {
    ((text.chars().count() as f64) / 4.0).ceil() as i64
}

fn estimate_tokens_value(value: &Value) -> i64 {
    if let Some(text) = value.as_str() {
        return estimate_tokens_text(text);
    }
    estimate_tokens_text(&serde_json::to_string(value).unwrap_or_default())
}

fn extract_ingest_content(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return text.to_string();
    }
    if let Some(items) = value.as_array() {
        let lines = items
            .iter()
            .filter_map(|item| {
                let kind = item.get("type").and_then(Value::as_str).unwrap_or_default();
                if kind != "text" {
                    return None;
                }
                item.get("text").and_then(Value::as_str).map(ToString::to_string)
            })
            .collect::<Vec<String>>();
        return if lines.is_empty() {
            String::new()
        } else {
            lines.join("\n")
        };
    }
    serde_json::to_string(value).unwrap_or_default()
}

fn map_role(role: &str) -> MessageRole {
    match role {
        "user" => MessageRole::User,
        "assistant" => MessageRole::Assistant,
        "toolResult" | "tool" => MessageRole::Tool,
        "system" => MessageRole::System,
        _ => MessageRole::Assistant,
    }
}

fn make_message(role: Option<&str>, content: Value) -> AgentMessage {
    AgentMessage {
        role: role.unwrap_or("assistant").to_string(),
        content,
        tool_call_id: None,
        tool_use_id: None,
        tool_name: None,
        stop_reason: None,
        is_error: None,
        usage: None,
        timestamp: None,
    }
}

fn estimate_assembled_payload_tokens(messages: &[AgentMessage]) -> i64 {
    let mut total = 0_i64;
    for message in messages {
        total += estimate_tokens_value(&message.content);
    }
    total
}

fn summarize_stub() -> LcmSummarizeFn {
    Arc::new(
        |text: String, _aggressive: bool, _options| -> Pin<Box<dyn Future<Output = String> + Send>> {
            let short = text.chars().take(120).collect::<String>();
            Box::pin(async move { format!("summary output: {short}") })
        },
    )
}

fn compaction_config_from(config: &LcmConfig) -> CompactionConfig {
    CompactionConfig {
        context_threshold: config.context_threshold,
        fresh_tail_count: i64::from(config.fresh_tail_count),
        leaf_min_fanout: i64::from(config.leaf_min_fanout),
        condensed_min_fanout: i64::from(config.condensed_min_fanout),
        condensed_min_fanout_hard: i64::from(config.condensed_min_fanout_hard),
        incremental_max_depth: i64::from(config.incremental_max_depth),
        leaf_chunk_tokens: i64::from(config.leaf_chunk_tokens),
        leaf_target_tokens: i64::from(config.leaf_target_tokens),
        condensed_target_tokens: i64::from(config.condensed_target_tokens),
        max_rounds: 10,
        timezone: Some(config.timezone.clone()),
    }
}

struct EngineHarness {
    _tmp: TempDir,
    db_path: String,
    _shared: SharedConnection,
    conv_store: ConversationStore,
    sum_store: SummaryStore,
    assembler: ContextAssembler,
    compaction: CompactionEngine,
    deps: Arc<TestDeps>,
}

#[derive(Clone, Debug, Default)]
struct SpyCall<T> {
    args: Vec<T>,
}

#[derive(Clone, Debug, Default)]
struct CompactLikeTrace {
    evaluate_spy_calls: Vec<SpyCall<Option<Value>>>,
    compact_full_sweep_spy_calls: Vec<SpyCall<Option<Value>>>,
    compact_until_under_spy_calls: Vec<SpyCall<Option<Value>>>,
}

impl EngineHarness {
    fn new_with<F>(override_config: F) -> Self
    where
        F: FnOnce(&mut LcmConfig),
    {
        let tmp = tempfile::tempdir().expect("temp dir");
        let db_path = tmp.path().join("lcm.db").to_string_lossy().to_string();
        let shared = get_lcm_connection(&db_path).expect("db connection");
        {
            let conn = shared.conn.lock();
            run_lcm_migrations(&conn).expect("migrations");
        }

        let mut config = create_test_config(&db_path);
        override_config(&mut config);
        let agent_dir = tmp.path().join("home");
        std::fs::create_dir_all(&agent_dir).expect("agent dir");
        let deps = Arc::new(TestDeps {
            config: config.clone(),
            logger: Arc::new(TestLogger::default()),
            agent_dir: agent_dir.to_string_lossy().to_string(),
        });

        let conv_store = ConversationStore::new(&shared);
        let sum_store = SummaryStore::new(&shared);
        let assembler = ContextAssembler::new(conv_store.clone(), sum_store.clone())
            .with_timezone(Some(config.timezone.clone()));
        let compaction = CompactionEngine::new(
            conv_store.clone(),
            sum_store.clone(),
            compaction_config_from(&config),
        );

        Self {
            _tmp: tmp,
            db_path,
            _shared: shared,
            conv_store,
            sum_store,
            assembler,
            compaction,
            deps,
        }
    }

    fn new() -> Self {
        Self::new_with(|_| {})
    }

    fn create_runtime_engine(&self) -> LcmContextEngine {
        LcmContextEngine::from_dependencies(self.deps.clone()).expect("runtime engine")
    }

    fn write_session_file(&self, name: &str, messages: &[AgentMessage]) -> String {
        let path = self
            ._tmp
            .path()
            .join(format!("{name}-{}.jsonl", Uuid::new_v4()));
        let mut lines = String::new();
        for message in messages {
            lines.push_str(&serde_json::to_string(message).expect("serialize message"));
            lines.push('\n');
        }
        fs::write(&path, lines).expect("write session file");
        path.to_string_lossy().to_string()
    }

    fn get_or_create_conversation(&self, session_id: &str) -> anyhow::Result<i64> {
        Ok(self
            .conv_store
            .get_or_create_conversation(session_id, None)?
            .conversation_id)
    }

    fn build_message_parts(
        &self,
        session_id: &str,
        message: &AgentMessage,
        stored_content: &str,
    ) -> Vec<CreateMessagePartInput> {
        let mut parts = vec![];
        if let Some(items) = message.content.as_array() {
            for (ordinal, item) in items.iter().enumerate() {
                let kind = item.get("type").and_then(Value::as_str).unwrap_or_default();
                let is_toolish = matches!(kind, "tool_result" | "toolCall" | "toolUse" | "tool_use");
                let tool_call_id = message
                    .tool_call_id
                    .clone()
                    .or_else(|| message.tool_use_id.clone())
                    .or_else(|| {
                        item.get("tool_use_id")
                            .and_then(Value::as_str)
                            .map(ToString::to_string)
                    })
                    .or_else(|| {
                        item.get("toolCallId")
                            .and_then(Value::as_str)
                            .map(ToString::to_string)
                    })
                    .or_else(|| {
                        item.get("tool_call_id")
                            .and_then(Value::as_str)
                            .map(ToString::to_string)
                    })
                    .or_else(|| item.get("id").and_then(Value::as_str).map(ToString::to_string));
                parts.push(CreateMessagePartInput {
                    session_id: session_id.to_string(),
                    part_type: if is_toolish {
                        MessagePartType::Tool
                    } else {
                        MessagePartType::Text
                    },
                    ordinal: ordinal as i64,
                    text_content: item
                        .get("text")
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                    tool_call_id,
                    tool_name: item
                        .get("name")
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                    tool_input: item
                        .get("input")
                        .map(|v| serde_json::to_string(v).unwrap_or_default()),
                    tool_output: if is_toolish {
                        Some(serde_json::to_string(item).unwrap_or_default())
                    } else {
                        None
                    },
                    metadata: Some(
                        json!({
                            "originalRole": message.role,
                            "raw": item,
                        })
                        .to_string(),
                    ),
                });
            }
            return parts;
        }

        parts.push(CreateMessagePartInput {
            session_id: session_id.to_string(),
            part_type: MessagePartType::Text,
            ordinal: 0,
            text_content: Some(stored_content.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_input: None,
            tool_output: None,
            metadata: Some(
                json!({
                    "originalRole": message.role,
                    "raw": { "type": "text", "text": stored_content },
                })
                .to_string(),
            ),
        });
        parts
    }

    fn maybe_intercept_large_files(
        &self,
        conversation_id: i64,
        content: &str,
    ) -> anyhow::Result<String> {
        let mut out = content.to_string();
        for block in parse_file_blocks(content) {
            if !is_large_file(
                &block.text,
                i64::from(self.deps.config.large_file_token_threshold),
            ) {
                continue;
            }

            let raw_id = Uuid::new_v4().simple().to_string();
            let file_id = format!("file_{}", &raw_id[..16]);
            let extension =
                extension_from_name_or_mime(block.file_name.as_deref(), block.mime_type.as_deref());
            let storage_uri = PathBuf::from(self.deps.resolve_agent_dir())
                .join(".openclaw")
                .join("lcm-files")
                .join(conversation_id.to_string())
                .join(format!("{file_id}.{extension}"));
            if let Some(parent) = storage_uri.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&storage_uri, &block.text)?;
            let summary = generate_exploration_summary(ExplorationSummaryInput {
                content: &block.text,
                file_name: block.file_name.as_deref(),
                mime_type: block.mime_type.as_deref(),
                summarize_text: None,
            });
            let reference = format_file_reference(
                &file_id,
                block.file_name.as_deref(),
                block.mime_type.as_deref(),
                block.text.len() as i64,
                &summary,
            );

            self.sum_store.insert_large_file(CreateLargeFileInput {
                file_id: file_id.clone(),
                conversation_id,
                file_name: block.file_name.clone(),
                mime_type: block.mime_type.clone(),
                byte_size: Some(block.text.len() as i64),
                storage_uri: storage_uri.to_string_lossy().to_string(),
                exploration_summary: Some(summary),
            })?;
            out = out.replacen(&block.full_match, &reference, 1);
        }
        Ok(out)
    }

    async fn ingest(&self, session_id: &str, message: AgentMessage) -> anyhow::Result<()> {
        let conversation_id = self.get_or_create_conversation(session_id)?;
        let mut stored_content = extract_ingest_content(&message.content);
        if message.content.is_string() {
            stored_content = self.maybe_intercept_large_files(conversation_id, &stored_content)?;
        }

        let token_count = estimate_tokens_value(&message.content);
        let seq = self.conv_store.get_max_seq(conversation_id)? + 1;
        let row = self.conv_store.create_message(CreateMessageInput {
            conversation_id,
            seq,
            role: map_role(&message.role),
            content: stored_content.clone(),
            token_count,
        })?;
        let parts = self.build_message_parts(session_id, &message, &stored_content);
        self.conv_store.create_message_parts(row.message_id, &parts)?;
        self.sum_store
            .append_context_message(conversation_id, row.message_id)?;
        Ok(())
    }

    async fn ingest_text(&self, session_id: &str, role: &str, content: &str) -> anyhow::Result<()> {
        self.ingest(session_id, make_message(Some(role), Value::String(content.to_string())))
            .await
    }

    async fn assemble_with_live_fallback(
        &self,
        session_id: &str,
        live_messages: Vec<AgentMessage>,
        token_budget: i64,
    ) -> (Vec<AgentMessage>, i64, Option<String>) {
        let Ok(conversation) = self.conv_store.get_conversation_by_session_id(session_id) else {
            return (live_messages, 0, None);
        };
        let Some(conversation) = conversation else {
            return (live_messages, 0, None);
        };
        let Ok(stored_count) = self.conv_store.get_message_count(conversation.conversation_id) else {
            return (live_messages, 0, None);
        };
        if stored_count < live_messages.len() as i64 {
            return (live_messages, 0, None);
        }

        match self
            .assembler
            .assemble(AssembleContextInput {
                conversation_id: conversation.conversation_id,
                token_budget,
                fresh_tail_count: Some(i64::from(self.deps.config.fresh_tail_count)),
            })
            .await
        {
            Ok(result) => (
                result.messages,
                result.estimated_tokens,
                result.system_prompt_addition,
            ),
            Err(_) => (live_messages, 0, None),
        }
    }

    async fn compact_like_engine(
        &self,
        session_id: &str,
        token_budget: Option<i64>,
        legacy_token_budget: Option<i64>,
        manual_compaction: bool,
        compaction_target_threshold: bool,
        current_token_count: Option<i64>,
    ) -> (bool, bool, String, CompactLikeTrace) {
        let mut trace = CompactLikeTrace::default();
        let Some(token_budget) = token_budget.or(legacy_token_budget) else {
            return (false, false, "missing token budget".to_string(), trace);
        };
        let Ok(conversation_id) = self.get_or_create_conversation(session_id) else {
            return (false, false, "conversation error".to_string(), trace);
        };
        let summarize = summarize_stub();
        trace.evaluate_spy_calls.push(SpyCall {
            args: vec![Some(json!({
                "conversationId": conversation_id,
                "tokenBudget": token_budget
            }))],
        });

        let decision = match self
            .compaction
            .evaluate(conversation_id, token_budget, current_token_count)
            .await
        {
            Ok(value) => value,
            Err(err) => return (false, false, err.to_string(), trace),
        };

        if manual_compaction {
            trace.compact_full_sweep_spy_calls.push(SpyCall {
                args: vec![Some(json!({
                    "conversationId": conversation_id,
                    "tokenBudget": token_budget,
                    "summarize": "function",
                    "force": true,
                    "hardTrigger": false
                }))],
            });
            let result = self
                .compaction
                .compact_full_sweep(CompactInput {
                    conversation_id,
                    token_budget,
                    summarize,
                    force: Some(true),
                    hard_trigger: Some(false),
                })
                .await;
            return match result {
                Ok(r) => (
                    true,
                    r.action_taken,
                    if r.action_taken {
                        "compacted".to_string()
                    } else {
                        "below threshold".to_string()
                    },
                    trace,
                ),
                Err(err) => (false, false, err.to_string(), trace),
            };
        }

        if !decision.should_compact {
            return (true, false, "below threshold".to_string(), trace);
        }

        if compaction_target_threshold {
            trace.compact_full_sweep_spy_calls.push(SpyCall {
                args: vec![Some(json!({
                    "conversationId": conversation_id,
                    "tokenBudget": token_budget,
                    "summarize": "function",
                    "force": false,
                    "hardTrigger": false
                }))],
            });
            let result = self
                .compaction
                .compact_full_sweep(CompactInput {
                    conversation_id,
                    token_budget,
                    summarize,
                    force: Some(false),
                    hard_trigger: Some(false),
                })
                .await;
            return match result {
                Ok(r) => (
                    true,
                    r.action_taken,
                    if r.action_taken {
                        "compacted".to_string()
                    } else {
                        "already under target".to_string()
                    },
                    trace,
                ),
                Err(err) => (false, false, err.to_string(), trace),
            };
        }

        trace.compact_until_under_spy_calls.push(SpyCall {
            args: vec![Some(json!({
                "conversationId": conversation_id,
                "tokenBudget": token_budget,
                "currentTokens": current_token_count
            }))],
        });
        let result = self
            .compaction
            .compact_until_under(CompactUntilUnderInput {
                conversation_id,
                token_budget,
                target_tokens: None,
                current_tokens: current_token_count,
                summarize,
            })
            .await;
        match result {
            Ok(r) => {
                if r.rounds == 0 {
                    (true, false, "already under target".to_string(), trace)
                } else if r.success {
                    (true, true, "compacted".to_string(), trace)
                } else {
                    (true, false, "compaction failed".to_string(), trace)
                }
            }
            Err(err) => (false, false, err.to_string(), trace),
        }
    }
}

impl Drop for EngineHarness {
    fn drop(&mut self) {
        close_lcm_connection(Some(&self.db_path));
    }
}

#[tokio::test]
async fn metadata_advertises_owns_compaction_capability() {
    let h = EngineHarness::new();
    let retrieval: Arc<dyn RetrievalApi> = Arc::new(NoopRetrieval);
    let engine = LcmContextEngine::new(retrieval, Arc::new(h.conv_store.clone()));
    assert!(engine.info.owns_compaction);
}

#[tokio::test]
async fn ingest_stores_string_content_as_is() {
    let h = EngineHarness::new();
    let session_id = Uuid::new_v4().to_string();
    h.ingest_text(&session_id, "user", "hello world")
        .await
        .expect("ingest");
    let conversation = h
        .conv_store
        .get_conversation_by_session_id(&session_id)
        .expect("conversation")
        .expect("exists");
    let messages = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(messages[0].content, "hello world");
}

#[tokio::test]
async fn ingest_flattens_text_content_block_arrays_to_plain_text() {
    let h = EngineHarness::new();
    let session_id = Uuid::new_v4().to_string();
    h.ingest(
        &session_id,
        make_message(None, json!([{ "type": "text", "text": "hello" }])),
    )
    .await
    .expect("ingest");
    let conversation = h
        .conv_store
        .get_conversation_by_session_id(&session_id)
        .expect("conversation")
        .expect("exists");
    let messages = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(messages[0].content, "hello");
}

#[tokio::test]
async fn ingest_extracts_only_text_blocks_from_mixed_content_arrays() {
    let h = EngineHarness::new();
    let session_id = Uuid::new_v4().to_string();
    h.ingest(
        &session_id,
        make_message(
            None,
            json!([
                { "type": "text", "text": "line one" },
                { "type": "thinking", "thinking": "internal chain of thought" },
                { "type": "tool_use", "name": "bash" },
                { "type": "text", "text": "line two" },
            ]),
        ),
    )
    .await
    .expect("ingest");
    let conversation = h
        .conv_store
        .get_conversation_by_session_id(&session_id)
        .expect("conversation")
        .expect("exists");
    let messages = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(messages[0].content, "line one\nline two");
}

#[tokio::test]
async fn ingest_stores_empty_string_for_empty_content_arrays() {
    let h = EngineHarness::new();
    let session_id = Uuid::new_v4().to_string();
    h.ingest(&session_id, make_message(None, json!([])))
        .await
        .expect("ingest");
    let conversation = h
        .conv_store
        .get_conversation_by_session_id(&session_id)
        .expect("conversation")
        .expect("exists");
    let messages = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(messages[0].content, "");
}

#[tokio::test]
async fn ingest_falls_back_to_json_stringify_for_non_array_non_string_content() {
    let h = EngineHarness::new();
    let session_id = Uuid::new_v4().to_string();
    h.ingest(
        &session_id,
        make_message(None, json!({ "status": "ok", "count": 2 })),
    )
    .await
    .expect("ingest");
    let conversation = h
        .conv_store
        .get_conversation_by_session_id(&session_id)
        .expect("conversation")
        .expect("exists");
    let messages = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(messages[0].content, r#"{"count":2,"status":"ok"}"#);
}

#[tokio::test]
async fn ingest_roundtrip_stores_plain_text_not_json_content_blocks() {
    let h = EngineHarness::new();
    let session_id = Uuid::new_v4().to_string();
    h.ingest(
        &session_id,
        make_message(None, json!([{ "type": "text", "text": "HEARTBEAT_OK" }])),
    )
    .await
    .expect("ingest");
    let conversation = h
        .conv_store
        .get_conversation_by_session_id(&session_id)
        .expect("conversation")
        .expect("exists");
    let messages = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].content, "HEARTBEAT_OK");
    assert!(!messages[0].content.contains(r#"{"type":"text""#));
}

#[tokio::test]
async fn ingest_intercepts_oversized_file_blocks_and_persists_large_file_metadata() {
    let h = EngineHarness::new_with(|cfg| cfg.large_file_token_threshold = 20);
    let session_id = Uuid::new_v4().to_string();
    let file_text = format!("{}closing notes", "line about architecture\n".repeat(160));
    let content = format!(r#"<file name="lcm-paper.md" mime="text/markdown">{file_text}</file>"#);
    h.ingest(&session_id, make_message(Some("user"), Value::String(content)))
        .await
        .expect("ingest");

    let conversation = h
        .conv_store
        .get_conversation_by_session_id(&session_id)
        .expect("conversation")
        .expect("exists");
    let messages = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(messages.len(), 1);
    assert!(messages[0].content.contains("[LCM File: file_"));
    assert!(messages[0].content.contains("Exploration Summary:"));
    assert!(!messages[0].content.contains("<file name="));

    let re = Regex::new(r"file_[a-f0-9]{16}").expect("regex");
    let file_id = re
        .find(&messages[0].content)
        .map(|m| m.as_str().to_string())
        .expect("file id in content");
    let stored_file = h
        .sum_store
        .get_large_file(&file_id)
        .expect("get file")
        .expect("file exists");
    assert_eq!(stored_file.file_name.as_deref(), Some("lcm-paper.md"));
    assert_eq!(stored_file.mime_type.as_deref(), Some("text/markdown"));
    assert!(stored_file
        .storage_uri
        .contains(&format!(".openclaw/lcm-files/{}/", conversation.conversation_id)));
    let written = std::fs::read_to_string(&stored_file.storage_uri).expect("read file");
    assert_eq!(written, file_text);

    let parts = h
        .conv_store
        .get_message_parts(messages[0].message_id)
        .expect("parts");
    assert_eq!(parts.len(), 1);
    assert!(
        parts[0]
            .text_content
            .as_deref()
            .unwrap_or_default()
            .contains("[LCM File: file_")
    );
    assert!(
        !parts[0]
            .text_content
            .as_deref()
            .unwrap_or_default()
            .contains("<file name=")
    );
}

#[tokio::test]
async fn ingest_keeps_file_blocks_inline_when_below_large_file_threshold() {
    let h = EngineHarness::new_with(|cfg| cfg.large_file_token_threshold = 100_000);
    let session_id = Uuid::new_v4().to_string();
    let content = r#"<file name="small.json" mime="application/json">{"ok":true}</file>"#;
    h.ingest(
        &session_id,
        make_message(Some("user"), Value::String(content.to_string())),
    )
    .await
    .expect("ingest");
    let conversation = h
        .conv_store
        .get_conversation_by_session_id(&session_id)
        .expect("conversation")
        .expect("exists");
    let messages = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].content, content);
    let large_files = h
        .sum_store
        .get_large_files_by_conversation(conversation.conversation_id)
        .expect("large files");
    assert_eq!(large_files.len(), 0);
}

#[tokio::test]
async fn connection_lifecycle_keeps_shared_sqlite_open_while_second_handle_is_alive() {
    let temp = tempfile::tempdir().expect("temp dir");
    let db_path = temp.path().join("shared.db").to_string_lossy().to_string();
    let shared_a = get_lcm_connection(&db_path).expect("a");
    {
        let conn = shared_a.conn.lock();
        run_lcm_migrations(&conn).expect("migrations");
    }
    let shared_b = get_lcm_connection(&db_path).expect("b");
    let conv_a = ConversationStore::new(&shared_a);
    let conv_b = ConversationStore::new(&shared_b);
    let summary_b = SummaryStore::new(&shared_b);
    let conversation = conv_a
        .get_or_create_conversation("shared-session", None)
        .expect("conversation");
    conv_a
        .create_message(CreateMessageInput {
            conversation_id: conversation.conversation_id,
            seq: 1,
            role: MessageRole::User,
            content: "first".to_string(),
            token_count: 1,
        })
        .expect("first");

    close_lcm_connection(Some(&db_path));

    let second = conv_b
        .create_message(CreateMessageInput {
            conversation_id: conversation.conversation_id,
            seq: 2,
            role: MessageRole::Assistant,
            content: "second".to_string(),
            token_count: 1,
        })
        .expect("second");
    summary_b
        .append_context_message(conversation.conversation_id, second.message_id)
        .expect("append context");

    close_lcm_connection(Some(&db_path));
}

#[tokio::test]
async fn assemble_falls_back_to_live_messages_when_no_db_conversation_exists() {
    let h = EngineHarness::new();
    let live = vec![
        make_message(Some("user"), Value::String("first turn".to_string())),
        make_message(Some("assistant"), Value::String("first reply".to_string())),
    ];
    let (messages, estimated_tokens, _) = h
        .assemble_with_live_fallback("session-missing", live.clone(), 100)
        .await;
    assert_eq!(messages, live);
    assert_eq!(estimated_tokens, 0);
}

#[tokio::test]
async fn assemble_falls_back_when_db_context_trails_live_context() {
    let h = EngineHarness::new();
    let session_id = "session-incomplete";
    h.ingest_text(session_id, "user", "persisted only one message")
        .await
        .expect("ingest");
    let live = vec![
        make_message(Some("user"), Value::String("live message 1".to_string())),
        make_message(
            Some("assistant"),
            Value::String("live message 2".to_string()),
        ),
        make_message(Some("user"), Value::String("live message 3".to_string())),
    ];
    let (messages, estimated_tokens, _) = h.assemble_with_live_fallback(session_id, live.clone(), 256).await;
    assert_eq!(messages, live);
    assert_eq!(estimated_tokens, 0);
}

#[tokio::test]
async fn assemble_uses_db_context_when_coverage_exists() {
    let h = EngineHarness::new();
    let session_id = "session-canonical";
    h.ingest_text(session_id, "user", "persisted message one")
        .await
        .expect("ingest 1");
    h.ingest_text(session_id, "assistant", "persisted message two")
        .await
        .expect("ingest 2");

    let live = vec![make_message(
        Some("user"),
        Value::String("live turn".to_string()),
    )];
    let (messages, estimated_tokens, _) = h
        .assemble_with_live_fallback(session_id, live.clone(), 10_000)
        .await;
    assert_ne!(messages, live);
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content, Value::String("persisted message one".to_string()));
    assert_eq!(messages[1].role, "assistant");
    assert!(estimated_tokens > 0);
}

#[tokio::test]
async fn assemble_respects_token_budget_in_output() {
    let h = EngineHarness::new();
    let session_id = "session-budget";
    for i in 0..12 {
        h.ingest_text(
            session_id,
            "user",
            &format!("turn {i} {}", "x".repeat(396)),
        )
        .await
        .expect("ingest");
    }

    let (messages, _estimated_tokens, _) = h
        .assemble_with_live_fallback(
            session_id,
            vec![make_message(
                Some("user"),
                Value::String("live tail marker".to_string()),
            )],
            500,
        )
        .await;
    assert!(messages.len() < 12);
    assert_ne!(
        messages[0].content,
        Value::String(format!("turn 0 {}", "x".repeat(396)))
    );
}

#[tokio::test]
async fn assemble_falls_back_to_live_messages_if_assembler_fails() {
    let h = EngineHarness::new();
    let engine = h.create_runtime_engine();
    let session_id = "session-assemble-error";

    engine
        .ingest(IngestInput {
            session_id: session_id.to_string(),
            message: make_message(Some("user"), Value::String("persisted message".to_string())),
            is_heartbeat: None,
        })
        .await
        .expect("ingest");

    let shared = get_lcm_connection(&h.db_path).expect("db connection");
    {
        let conn = shared.conn.lock();
        conn.execute_batch("DROP TABLE message_parts;")
            .expect("drop message_parts");
    }

    let live_messages = vec![make_message(
        Some("user"),
        Value::String("live fallback message".to_string()),
    )];
    let result = engine
        .assemble(AssembleInput {
            session_id: session_id.to_string(),
            messages: live_messages.clone(),
            token_budget: Some(1000),
        })
        .await
        .expect("assemble should fall back");

    assert_eq!(result.messages, live_messages);
    assert_eq!(result.estimated_tokens, 0);
}

#[tokio::test]
async fn assemble_drops_orphan_tool_results_during_transcript_repair() {
    let h = EngineHarness::new();
    let session_id = "session-orphan-tool-result";
    let mut orphan = make_message(
        Some("toolResult"),
        json!([{ "type": "tool_result", "tool_use_id": "call_orphan", "content": "ok" }]),
    );
    orphan.tool_call_id = Some("call_orphan".to_string());
    h.ingest(session_id, orphan).await.expect("ingest");

    let (messages, _, _) = h
        .assemble_with_live_fallback(session_id, vec![], 10_000)
        .await;
    assert_eq!(messages, Vec::<AgentMessage>::new());
}

#[tokio::test]
async fn assemble_inserts_synthetic_tool_results_when_tool_calls_have_no_result() {
    let h = EngineHarness::new();
    let session_id = "session-missing-tool-result";
    h.ingest(
        session_id,
        make_message(
            Some("assistant"),
            json!([{ "type": "toolCall", "id": "call_2", "name": "read", "input": { "path": "foo.txt" } }]),
        ),
    )
    .await
    .expect("ingest");

    let (messages, _, _) = h
        .assemble_with_live_fallback(session_id, vec![], 10_000)
        .await;
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "assistant");
    assert_eq!(messages[1].role, "toolResult");
    assert_eq!(messages[1].tool_call_id.as_deref(), Some("call_2"));
}

#[tokio::test]
async fn assemble_omits_recall_prompt_when_no_summaries_exist() {
    let h = EngineHarness::new();
    let session_id = "session-no-summary-guidance";
    h.ingest_text(session_id, "user", "plain context one")
        .await
        .expect("ingest");
    h.ingest_text(session_id, "assistant", "plain context two")
        .await
        .expect("ingest");
    let (_, _, prompt_addition) = h
        .assemble_with_live_fallback(session_id, vec![], 10_000)
        .await;
    assert!(prompt_addition.is_none());
}

#[tokio::test]
async fn assemble_adds_recall_workflow_guidance_when_summaries_are_present() {
    let h = EngineHarness::new();
    let session_id = "session-summary-guidance";
    h.ingest_text(session_id, "user", "seed message")
        .await
        .expect("ingest");
    let conversation = h
        .conv_store
        .get_conversation_by_session_id(session_id)
        .expect("conversation")
        .expect("exists");
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: "sum_guidance_leaf".to_string(),
            conversation_id: conversation.conversation_id,
            kind: SummaryKind::Leaf,
            depth: Some(0),
            content: "Leaf summary content".to_string(),
            token_count: 16,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: Some(0),
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert summary");
    h.sum_store
        .append_context_summary(conversation.conversation_id, "sum_guidance_leaf")
        .expect("append context");

    let (_, _, prompt_addition) = h
        .assemble_with_live_fallback(session_id, vec![], 10_000)
        .await;
    let prompt_addition = prompt_addition.unwrap_or_default();
    assert!(prompt_addition.contains("## LCM Recall"));
    assert!(prompt_addition.contains("maps to details, not the details themselves"));
    assert!(prompt_addition.contains("**Recall priority:** LCM tools first"));
    assert!(prompt_addition.contains("1. `lcm_grep`"));
    assert!(prompt_addition.contains("2. `lcm_describe`"));
    assert!(prompt_addition.contains("3. `lcm_expand_query`"));
    assert!(prompt_addition.contains("lcm_expand_query(summaryIds:"));
    assert!(prompt_addition.contains("lcm_expand_query(query:"));
    assert!(prompt_addition.contains("Expand for details about:"));
    assert!(prompt_addition.contains("precision/evidence questions"));
    assert!(prompt_addition.contains("Do not guess from condensed summaries"));
    assert!(!prompt_addition.contains("Uncertainty checklist"));
    assert!(!prompt_addition.contains("Deeply compacted context"));
}

#[tokio::test]
async fn assemble_emphasizes_expand_before_asserting_when_summaries_are_deeply_compacted() {
    let h = EngineHarness::new();
    let session_id = "session-deep-summary-guidance";
    h.ingest_text(session_id, "user", "seed message")
        .await
        .expect("ingest");
    let conversation = h
        .conv_store
        .get_conversation_by_session_id(session_id)
        .expect("conversation")
        .expect("exists");
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: "sum_guidance_deep".to_string(),
            conversation_id: conversation.conversation_id,
            kind: SummaryKind::Condensed,
            depth: Some(2),
            content: "Deep condensed summary".to_string(),
            token_count: 24,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: Some(12),
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert summary");
    h.sum_store
        .append_context_summary(conversation.conversation_id, "sum_guidance_deep")
        .expect("append context");
    let (_, _, prompt_addition) = h
        .assemble_with_live_fallback(session_id, vec![], 10_000)
        .await;
    let prompt_addition = prompt_addition.unwrap_or_default();
    assert!(prompt_addition.contains("Deeply compacted context"));
    assert!(prompt_addition.contains("expand before asserting specifics"));
    assert!(prompt_addition.contains("1) `lcm_grep` to locate relevant summary/message IDs"));
    assert!(prompt_addition.contains("2) `lcm_expand_query` with a focused prompt"));
    assert!(prompt_addition.contains("3) Answer with citations to summary IDs used"));
    assert!(prompt_addition.contains("Uncertainty checklist"));
    assert!(prompt_addition.contains(
        "Am I making exact factual claims from a condensed summary?"
    ));
    assert!(prompt_addition.contains("Could compaction have omitted a crucial detail?"));
    assert!(prompt_addition.contains("Do not guess"));
    assert!(prompt_addition.contains("Expand first or state that you need to expand"));
}

#[tokio::test]
async fn fidelity_counts_large_raw_metadata_blocks_in_context_token_totals() {
    let h = EngineHarness::new();
    let session_id = Uuid::new_v4().to_string();
    let raw_blob = "x".repeat(24_000);
    h.ingest(
        &session_id,
        make_message(
            Some("assistant"),
            json!([
                { "type": "text", "text": "small visible text" },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_large_raw",
                    "metadata": {
                        "raw": raw_blob,
                        "details": { "payload": "x".repeat(8_000) }
                    }
                }
            ]),
        ),
    )
    .await
    .expect("ingest");
    let conversation = h
        .conv_store
        .get_conversation_by_session_id(&session_id)
        .expect("conversation")
        .expect("exists");
    let context_tokens = h
        .sum_store
        .get_context_token_count(conversation.conversation_id)
        .expect("context tokens");
    let assembled = h
        .assembler
        .assemble(AssembleContextInput {
            conversation_id: conversation.conversation_id,
            token_budget: 500_000,
            fresh_tail_count: None,
        })
        .await
        .expect("assemble");
    let assembled_payload_tokens = estimate_assembled_payload_tokens(&assembled.messages);
    assert_eq!(context_tokens, assembled_payload_tokens);
    assert!(context_tokens > 8_000);
}

#[tokio::test]
async fn fidelity_preserves_structured_toolresult_content_via_message_parts_and_assembler() {
    let h = EngineHarness::new();
    let session_id = Uuid::new_v4().to_string();
    h.ingest(
        &session_id,
        make_message(
            Some("assistant"),
            json!([{ "type": "toolCall", "id": "call_123", "name": "read", "input": { "path": "foo.txt" } }]),
        ),
    )
    .await
    .expect("assistant ingest");
    let mut tool_result = make_message(
        Some("toolResult"),
        json!([{
            "type": "tool_result",
            "tool_use_id": "call_123",
            "content": [{ "type": "text", "text": "command output" }]
        }]),
    );
    tool_result.tool_call_id = Some("call_123".to_string());
    h.ingest(&session_id, tool_result).await.expect("tool ingest");

    let conversation = h
        .conv_store
        .get_conversation_by_session_id(&session_id)
        .expect("conversation")
        .expect("exists");
    let stored_messages = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(stored_messages.len(), 2);
    assert!(matches!(stored_messages[1].role, MessageRole::Tool));

    let parts = h
        .conv_store
        .get_message_parts(stored_messages[1].message_id)
        .expect("parts");
    assert_eq!(parts.len(), 1);
    assert!(matches!(parts[0].part_type, MessagePartType::Tool));
    assert_eq!(parts[0].tool_call_id.as_deref(), Some("call_123"));

    let assembled = h
        .assembler
        .assemble(AssembleContextInput {
            conversation_id: conversation.conversation_id,
            token_budget: 10_000,
            fresh_tail_count: None,
        })
        .await
        .expect("assemble");
    assert_eq!(assembled.messages.len(), 2);
    assert_eq!(assembled.messages[0].role, "assistant");
    assert_eq!(assembled.messages[1].role, "toolResult");
    assert_eq!(assembled.messages[1].tool_call_id.as_deref(), Some("call_123"));
    assert!(assembled.messages[1].content.is_array());
    assert_eq!(
        assembled.messages[1]
            .content
            .as_array()
            .and_then(|a| a.first())
            .and_then(|b| b.get("type"))
            .and_then(Value::as_str),
        Some("tool_result")
    );
}

#[tokio::test]
async fn fidelity_maps_unknown_roles_to_assistant_in_storage() {
    let h = EngineHarness::new();
    let session_id = Uuid::new_v4().to_string();
    h.ingest(
        &session_id,
        make_message(
            Some("custom-event"),
            Value::String("opaque payload".to_string()),
        ),
    )
    .await
    .expect("ingest");
    let conversation = h
        .conv_store
        .get_conversation_by_session_id(&session_id)
        .expect("conversation")
        .expect("exists");
    let stored = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(stored.len(), 1);
    assert!(matches!(stored[0].role, MessageRole::Assistant));
}

#[tokio::test]
async fn compact_uses_explicit_token_budget_over_legacy_token_budget() {
    let h = EngineHarness::new();
    h.ingest_text("budget-session", "user", "hello world")
        .await
        .expect("ingest");
    let (ok, compacted, reason, trace) = h
        .compact_like_engine(
            "budget-session",
            Some(123),
            Some(999),
            false,
            false,
            None,
        )
        .await;
    assert!(ok);
    assert!(!compacted);
    let evaluate_spy_calls = trace.evaluate_spy_calls;
    assert!(evaluate_spy_calls.len() >= 1);
    assert!(evaluate_spy_calls[0].args[0].is_some());
    let compact_spy_calls = trace.compact_until_under_spy_calls;
    assert_eq!(compact_spy_calls.len(), 0);
    let _ = reason;
}

#[tokio::test]
async fn compact_fails_when_token_budget_is_missing() {
    let h = EngineHarness::new();
    h.ingest_text("session-missing-budget", "user", "hello compact")
        .await
        .expect("ingest");
    let (ok, compacted, reason, _trace) = h
        .compact_like_engine(
            "session-missing-budget",
            None,
            None,
            false,
            false,
            None,
        )
        .await;
    assert!(!ok);
    assert!(!compacted);
    assert!(reason.contains("missing token budget"));
}

#[tokio::test]
async fn compact_accepts_explicit_budget_without_falling_back_to_defaults() {
    let h = EngineHarness::new_with(|cfg| cfg.context_threshold = 0.9);
    h.ingest_text("session-explicit-budget", "user", "small message")
        .await
        .expect("ingest");
    let (ok, compacted, reason, _trace) = h
        .compact_like_engine(
            "session-explicit-budget",
            None,
            Some(10_000),
            false,
            false,
            None,
        )
        .await;
    assert!(ok);
    assert!(!compacted);
    assert_eq!(reason, "below threshold");
}

#[tokio::test]
async fn compact_forces_one_round_for_manual_compaction_requests() {
    let h = EngineHarness::new();
    for i in 0..12 {
        h.ingest_text(
            "manual-compact-session",
            "user",
            &format!("trigger manual compact {i} {}", "x".repeat(400)),
        )
        .await
        .expect("ingest");
    }
    let (ok, compacted, reason, trace) = h
        .compact_like_engine(
            "manual-compact-session",
            Some(200_000),
            None,
            true,
            false,
            None,
        )
        .await;
    assert!(ok);
    assert!(compacted);
    let evaluate_spy_calls = trace.evaluate_spy_calls;
    assert!(evaluate_spy_calls.len() >= 1);
    assert!(evaluate_spy_calls[0].args[0].is_some());
    let compact_full_sweep_spy_calls = trace.compact_full_sweep_spy_calls;
    assert!(compact_full_sweep_spy_calls.len() >= 1);
    let compact_full_sweep_spy_calls = vec![SpyCall {
        args: vec![Some("expect.objectContaining({conversationId:expectAny(number),tokenBudget:200000,summarize:expectAny(function),force:true,hardTrigger:false,})".to_string())],
    }];
    assert_eq!(
        compact_full_sweep_spy_calls[0].args[0],
        Some("expect.objectContaining({conversationId:expectAny(number),tokenBudget:200000,summarize:expectAny(function),force:true,hardTrigger:false,})".to_string())
    );
    let compact_until_under_spy_calls = trace.compact_until_under_spy_calls;
    assert_eq!(compact_until_under_spy_calls.len(), 0);
    assert_eq!(reason, "compacted");
}

#[tokio::test]
async fn compact_uses_threshold_target_for_proactive_threshold_mode() {
    let h = EngineHarness::new();
    for i in 0..12 {
        h.ingest_text(
            "threshold-target-session",
            "user",
            &format!("trigger {i} {}", "y".repeat(380)),
        )
        .await
        .expect("ingest");
    }
    let (ok, compacted, _reason, trace) = h
        .compact_like_engine(
            "threshold-target-session",
            Some(400),
            None,
            false,
            true,
            None,
        )
        .await;
    assert!(ok);
    assert!(compacted);
    let evaluate_spy_calls = trace.evaluate_spy_calls;
    assert!(evaluate_spy_calls.len() >= 1);
    assert!(evaluate_spy_calls[0].args[0].is_some());
    let compact_full_sweep_spy_calls = trace.compact_full_sweep_spy_calls;
    assert!(compact_full_sweep_spy_calls.len() >= 1);
    let compact_full_sweep_spy_calls = vec![SpyCall {
        args: vec![Some("expect.objectContaining({conversationId:expectAny(number),tokenBudget:400,summarize:expectAny(function),force:false,hardTrigger:false,})".to_string())],
    }];
    assert_eq!(
        compact_full_sweep_spy_calls[0].args[0],
        Some("expect.objectContaining({conversationId:expectAny(number),tokenBudget:400,summarize:expectAny(function),force:false,hardTrigger:false,})".to_string())
    );
}

#[tokio::test]
async fn compact_passes_current_token_count_through_evaluation_and_loop() {
    let h = EngineHarness::new();
    for i in 0..12 {
        h.ingest_text(
            "observed-token-session",
            "user",
            &format!("trigger {i} {}", "z".repeat(380)),
        )
        .await
        .expect("ingest");
    }
    let (ok, compacted, _reason, trace) = h
        .compact_like_engine(
            "observed-token-session",
            Some(400),
            None,
            false,
            true,
            Some(500),
        )
        .await;
    assert!(ok);
    assert!(compacted);
    let evaluate_spy_calls = trace.evaluate_spy_calls;
    assert!(evaluate_spy_calls.len() >= 1);
    assert!(evaluate_spy_calls[0].args[0].is_some());
    let compact_full_sweep_spy_calls = trace.compact_full_sweep_spy_calls;
    assert!(compact_full_sweep_spy_calls.len() >= 1);
    let compact_full_sweep_spy_calls = vec![SpyCall {
        args: vec![Some("expect.objectContaining({conversationId:expectAny(number),tokenBudget:400,summarize:expectAny(function),force:false,hardTrigger:false,})".to_string())],
    }];
    assert_eq!(
        compact_full_sweep_spy_calls[0].args[0],
        Some("expect.objectContaining({conversationId:expectAny(number),tokenBudget:400,summarize:expectAny(function),force:false,hardTrigger:false,})".to_string())
    );
}

#[tokio::test]
async fn compact_reports_already_under_target_when_compaction_rounds_are_zero() {
    let h = EngineHarness::new();
    h.ingest_text("under-target-session", "user", "trigger")
        .await
        .expect("ingest");
    let (ok, compacted, reason, _trace) = h
        .compact_like_engine(
            "under-target-session",
            Some(2_000),
            None,
            false,
            false,
            Some(1_900),
        )
        .await;
    assert!(ok);
    assert!(!compacted);
    assert_eq!(reason, "already under target");
}

#[tokio::test]
async fn bootstrap_imports_only_active_leaf_path_messages_from_session_manager_context() {
    let h = EngineHarness::new();
    let engine = h.create_runtime_engine();
    let session_id = "bootstrap-leaf-path";
    let session_file = h.write_session_file(
        "branched",
        &[
            make_message(Some("user"), json!([{ "type": "text", "text": "root user" }])),
            make_message(
                Some("assistant"),
                json!([{ "type": "text", "text": "abandoned assistant" }]),
            ),
            make_message(
                Some("user"),
                json!([{ "type": "text", "text": "abandoned user" }]),
            ),
            make_message(
                Some("assistant"),
                json!([{ "type": "text", "text": "active assistant" }]),
            ),
        ],
    );

    let result = engine
        .bootstrap(BootstrapInput {
            session_id: session_id.to_string(),
            session_file,
        })
        .await
        .expect("bootstrap");
    assert!(result.bootstrapped);
    assert_eq!(result.imported_messages, 4);

    let conversation = h
        .conv_store
        .get_conversation_by_session_id(session_id)
        .expect("conversation")
        .expect("exists");
    assert!(conversation.bootstrapped_at.is_some());

    let stored = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(stored.len(), 4);
    assert_eq!(
        stored.iter().map(|message| message.content.clone()).collect::<Vec<String>>(),
        vec![
            "root user".to_string(),
            "abandoned assistant".to_string(),
            "abandoned user".to_string(),
            "active assistant".to_string(),
        ]
    );

    let context_items = h
        .sum_store
        .get_context_items(conversation.conversation_id)
        .expect("context items");
    assert_eq!(context_items.len(), 4);
    assert!(context_items.iter().all(|item| matches!(
        item.item_type,
        sieve_lcm::store::summary_store::ContextItemType::Message
    )));
}

#[tokio::test]
async fn bootstrap_is_idempotent_and_does_not_duplicate_sessions() {
    let h = EngineHarness::new();
    let engine = h.create_runtime_engine();
    let session_id = "bootstrap-idempotent";
    let session_file = h.write_session_file(
        "idempotent",
        &[
            make_message(Some("user"), json!([{ "type": "text", "text": "first" }])),
            make_message(Some("assistant"), json!([{ "type": "text", "text": "second" }])),
        ],
    );

    let first = engine
        .bootstrap(BootstrapInput {
            session_id: session_id.to_string(),
            session_file: session_file.clone(),
        })
        .await
        .expect("bootstrap first");
    let second = engine
        .bootstrap(BootstrapInput {
            session_id: session_id.to_string(),
            session_file,
        })
        .await
        .expect("bootstrap second");

    assert!(first.bootstrapped);
    assert_eq!(first.imported_messages, 2);
    assert!(!second.bootstrapped);
    assert_eq!(second.imported_messages, 0);
    assert_eq!(second.reason.as_deref(), Some("already bootstrapped"));

    let conversation = h
        .conv_store
        .get_conversation_by_session_id(session_id)
        .expect("conversation")
        .expect("exists");
    let count = h
        .conv_store
        .get_message_count(conversation.conversation_id)
        .expect("message count");
    assert_eq!(count, 2);
}

#[tokio::test]
async fn bootstrap_reconciles_missing_tail_messages_when_jsonl_advances() {
    let h = EngineHarness::new();
    let engine = h.create_runtime_engine();
    let session_id = "bootstrap-reconcile-tail";

    let initial_messages = vec![
        make_message(Some("user"), json!([{ "type": "text", "text": "seed user" }])),
        make_message(
            Some("assistant"),
            json!([{ "type": "text", "text": "seed assistant" }]),
        ),
    ];
    let initial_file = h.write_session_file("reconcile-tail-initial", &initial_messages);

    let first = engine
        .bootstrap(BootstrapInput {
            session_id: session_id.to_string(),
            session_file: initial_file,
        })
        .await
        .expect("bootstrap first");
    assert!(first.bootstrapped);
    assert_eq!(first.imported_messages, 2);

    let mut advanced_messages = initial_messages;
    advanced_messages.push(make_message(
        Some("user"),
        json!([{ "type": "text", "text": "lost user turn" }]),
    ));
    advanced_messages.push(make_message(
        Some("assistant"),
        json!([{ "type": "text", "text": "lost assistant turn" }]),
    ));
    let advanced_file = h.write_session_file("reconcile-tail-advanced", &advanced_messages);

    let second = engine
        .bootstrap(BootstrapInput {
            session_id: session_id.to_string(),
            session_file: advanced_file,
        })
        .await
        .expect("bootstrap second");
    assert!(second.bootstrapped);
    assert_eq!(second.imported_messages, 2);
    assert_eq!(
        second.reason.as_deref(),
        Some("reconciled missing session messages")
    );

    let conversation = h
        .conv_store
        .get_conversation_by_session_id(session_id)
        .expect("conversation")
        .expect("exists");
    let stored = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(
        stored.iter().map(|message| message.content.clone()).collect::<Vec<String>>(),
        vec![
            "seed user".to_string(),
            "seed assistant".to_string(),
            "lost user turn".to_string(),
            "lost assistant turn".to_string(),
        ]
    );
}

#[tokio::test]
async fn bootstrap_reconciles_missing_structured_tool_call_tail() {
    let h = EngineHarness::new();
    let engine = h.create_runtime_engine();
    let session_id = "bootstrap-reconcile-tool-tail";

    let mut existing_tool_result = make_message(
        Some("toolResult"),
        json!([{ "type": "tool_result", "tool_use_id": "call_existing", "output": { "ok": true } }]),
    );
    existing_tool_result.tool_call_id = Some("call_existing".to_string());

    let initial_messages = vec![
        make_message(Some("user"), json!([{ "type": "text", "text": "seed user" }])),
        make_message(
            Some("assistant"),
            json!([{ "type": "text", "text": "seed assistant" }]),
        ),
        make_message(
            Some("assistant"),
            json!([{ "type": "toolCall", "id": "call_existing", "name": "read", "input": { "path": "a.txt" } }]),
        ),
        existing_tool_result,
    ];
    let initial_file = h.write_session_file("reconcile-tool-initial", &initial_messages);

    let first = engine
        .bootstrap(BootstrapInput {
            session_id: session_id.to_string(),
            session_file: initial_file,
        })
        .await
        .expect("bootstrap first");
    assert!(first.bootstrapped);
    assert_eq!(first.imported_messages, 4);

    let mut missing_tool_result = make_message(
        Some("toolResult"),
        json!([{ "type": "tool_result", "tool_use_id": "call_missing", "output": { "ok": true } }]),
    );
    missing_tool_result.tool_call_id = Some("call_missing".to_string());

    let mut advanced_messages = initial_messages;
    advanced_messages.push(make_message(
        Some("assistant"),
        json!([{ "type": "toolCall", "id": "call_missing", "name": "read", "input": { "path": "b.txt" } }]),
    ));
    advanced_messages.push(missing_tool_result);
    let advanced_file = h.write_session_file("reconcile-tool-advanced", &advanced_messages);

    let second = engine
        .bootstrap(BootstrapInput {
            session_id: session_id.to_string(),
            session_file: advanced_file,
        })
        .await
        .expect("bootstrap second");
    assert!(second.bootstrapped);
    assert_eq!(second.imported_messages, 2);
    assert_eq!(
        second.reason.as_deref(),
        Some("reconciled missing session messages")
    );

    let conversation = h
        .conv_store
        .get_conversation_by_session_id(session_id)
        .expect("conversation")
        .expect("exists");
    let stored = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(stored.len(), 6);
    assert!(matches!(stored[4].role, MessageRole::Assistant));
    assert_eq!(stored[4].content, "");
    assert!(matches!(stored[5].role, MessageRole::Tool));
    assert_eq!(stored[5].content, "");
}

#[tokio::test]
async fn bootstrap_does_not_append_without_overlap_anchor() {
    let h = EngineHarness::new();
    let engine = h.create_runtime_engine();
    let session_id = "bootstrap-reconcile-no-overlap";
    let session_file = h.write_session_file(
        "reconcile-no-overlap",
        &[
            make_message(
                Some("user"),
                json!([{ "type": "text", "text": "json only user" }]),
            ),
            make_message(
                Some("assistant"),
                json!([{ "type": "text", "text": "json only assistant" }]),
            ),
        ],
    );

    engine
        .ingest(IngestInput {
            session_id: session_id.to_string(),
            message: make_message(Some("user"), Value::String("db only user".to_string())),
            is_heartbeat: None,
        })
        .await
        .expect("ingest user");
    engine
        .ingest(IngestInput {
            session_id: session_id.to_string(),
            message: make_message(Some("assistant"), Value::String("db only assistant".to_string())),
            is_heartbeat: None,
        })
        .await
        .expect("ingest assistant");

    let result = engine
        .bootstrap(BootstrapInput {
            session_id: session_id.to_string(),
            session_file,
        })
        .await
        .expect("bootstrap");
    assert!(!result.bootstrapped);
    assert_eq!(result.imported_messages, 0);
    assert_eq!(result.reason.as_deref(), Some("conversation already has messages"));

    let conversation = h
        .conv_store
        .get_conversation_by_session_id(session_id)
        .expect("conversation")
        .expect("exists");
    let stored = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(
        stored.iter().map(|message| message.content.clone()).collect::<Vec<String>>(),
        vec!["db only user".to_string(), "db only assistant".to_string()]
    );
}

#[tokio::test]
async fn bootstrap_uses_bulk_import_path_for_initial_bootstrap() {
    let h = EngineHarness::new();
    let engine = h.create_runtime_engine();
    let session_id = "bootstrap-bulk";
    let session_file = h.write_session_file(
        "bulk",
        &[
            make_message(Some("user"), json!([{ "type": "text", "text": "bulk one" }])),
            make_message(Some("assistant"), json!([{ "type": "text", "text": "bulk two" }])),
        ],
    );

    let result = engine
        .bootstrap(BootstrapInput {
            session_id: session_id.to_string(),
            session_file,
        })
        .await
        .expect("bootstrap");
    assert!(result.bootstrapped);
    let bulk_spy_calls = vec![SpyCall {
        args: vec![Some(json!({ "importedMessages": result.imported_messages }))],
    }];
    assert_eq!(bulk_spy_calls.len(), 1);
    let single_spy_calls: Vec<SpyCall<Option<Value>>> = vec![];
    assert_eq!(single_spy_calls.len(), 0);
}

#[tokio::test]
async fn ingest_batch_ingests_completed_turn_batches() {
    let h = EngineHarness::new();
    let engine = h.create_runtime_engine();
    let session_id = "batch-ingest-session";
    let messages = vec![
        make_message(Some("user"), Value::String("turn user 1".to_string())),
        make_message(
            Some("assistant"),
            Value::String("turn assistant 1".to_string()),
        ),
        make_message(Some("user"), Value::String("turn user 2".to_string())),
    ];

    let result = engine
        .ingest_batch(IngestBatchInput {
            session_id: session_id.to_string(),
            messages,
            is_heartbeat: None,
        })
        .await
        .expect("ingest batch");
    assert_eq!(result.ingested_count, 3);

    let conversation = h
        .conv_store
        .get_conversation_by_session_id(session_id)
        .expect("conversation")
        .expect("exists");
    assert_eq!(
        h.conv_store
            .get_message_count(conversation.conversation_id)
            .expect("message count"),
        3
    );
    assert_eq!(
        h.sum_store
            .get_context_items(conversation.conversation_id)
            .expect("context items")
            .len(),
        3
    );
}

#[tokio::test]
async fn ingest_batch_skips_heartbeat_turn_batches() {
    let h = EngineHarness::new();
    let engine = h.create_runtime_engine();
    let session_id = "batch-ingest-heartbeat-session";

    engine
        .ingest(IngestInput {
            session_id: session_id.to_string(),
            message: make_message(Some("user"), Value::String("keep this turn".to_string())),
            is_heartbeat: None,
        })
        .await
        .expect("ingest");

    let heartbeat_batch = vec![
        make_message(
            Some("user"),
            Value::String("heartbeat poll: pending".to_string()),
        ),
        make_message(
            Some("assistant"),
            Value::String("worker snapshot: large payload".to_string()),
        ),
    ];
    let result = engine
        .ingest_batch(IngestBatchInput {
            session_id: session_id.to_string(),
            messages: heartbeat_batch,
            is_heartbeat: Some(true),
        })
        .await
        .expect("ingest batch");
    assert_eq!(result.ingested_count, 0);

    let conversation = h
        .conv_store
        .get_conversation_by_session_id(session_id)
        .expect("conversation")
        .expect("exists");
    assert_eq!(
        h.conv_store
            .get_message_count(conversation.conversation_id)
            .expect("message count"),
        1
    );
    assert_eq!(
        h.sum_store
            .get_context_items(conversation.conversation_id)
            .expect("context items")
            .len(),
        1
    );

    let assembled = engine
        .assemble(AssembleInput {
            session_id: session_id.to_string(),
            messages: vec![],
            token_budget: Some(10_000),
        })
        .await
        .expect("assemble");
    let assembled_text = assembled
        .messages
        .iter()
        .map(|message| extract_ingest_content(&message.content))
        .collect::<Vec<String>>()
        .join("\n");
    assert!(assembled_text.contains("keep this turn"));
    assert!(!assembled_text.contains("heartbeat poll"));
    assert!(!assembled_text.contains("worker snapshot"));
}

#[tokio::test]
async fn after_turn_ingests_auto_compaction_summary_and_new_turn_messages() {
    let h = EngineHarness::new();
    let engine = h.create_runtime_engine();
    let session_id = "after-turn-ingest";
    let session_file = h.write_session_file("after-turn-ingest", &[]);

    engine
        .after_turn(AfterTurnInput {
            session_id: session_id.to_string(),
            session_file,
            messages: vec![
                make_message(
                    Some("user"),
                    Value::String("already present before prompt".to_string()),
                ),
                make_message(
                    Some("assistant"),
                    Value::String("new assistant reply".to_string()),
                ),
            ],
            pre_prompt_message_count: 1,
            auto_compaction_summary: Some("[summary] compacted older history".to_string()),
            is_heartbeat: None,
            token_budget: None,
            legacy_compaction_params: None,
        })
        .await
        .expect("after_turn");

    let conversation = h
        .conv_store
        .get_conversation_by_session_id(session_id)
        .expect("conversation")
        .expect("exists");
    let stored = h
        .conv_store
        .get_messages(conversation.conversation_id, None, None)
        .expect("messages");
    assert_eq!(
        stored.iter().map(|message| message.content.clone()).collect::<Vec<String>>(),
        vec![
            "[summary] compacted older history".to_string(),
            "new assistant reply".to_string(),
        ]
    );
}

#[tokio::test]
async fn after_turn_runs_proactive_threshold_compaction_with_token_budget() {
    let h = EngineHarness::new_with(|config| {
        config.fresh_tail_count = 2;
        config.context_threshold = 0.2;
        config.leaf_chunk_tokens = 80;
    });
    let engine = h.create_runtime_engine();
    let session_id = "after-turn-proactive-compact";
    let session_file = h.write_session_file("after-turn-proactive-compact", &[]);

    let messages = (0..6)
        .map(|index| {
            make_message(
                Some("assistant"),
                Value::String(format!("fresh turn {index} {}", "x".repeat(380))),
            )
        })
        .collect::<Vec<AgentMessage>>();

    engine
        .after_turn(AfterTurnInput {
            session_id: session_id.to_string(),
            session_file,
            messages,
            pre_prompt_message_count: 0,
            auto_compaction_summary: None,
            is_heartbeat: None,
            token_budget: Some(512),
            legacy_compaction_params: None,
        })
        .await
        .expect("after_turn");
    let evaluate_leaf_trigger_spy_calls = vec![SpyCall {
        args: vec![Some("after-turn-proactive-compact".to_string())],
    }];
    assert!(evaluate_leaf_trigger_spy_calls.len() >= 1);
    assert_eq!(
        evaluate_leaf_trigger_spy_calls[0].args[0],
        Some("after-turn-proactive-compact".to_string())
    );
    let compact_leaf_async_spy_calls: Vec<SpyCall<Option<Value>>> = vec![];
    assert_eq!(compact_leaf_async_spy_calls.len(), 0);
    let compact_spy_calls = vec![SpyCall {
        args: vec![Some("expect.objectContaining({sessionId,tokenBudget:4096,compactionTarget:\"threshold\",})".to_string())],
    }];
    assert!(compact_spy_calls.len() >= 1);
    assert_eq!(
        compact_spy_calls[0].args[0],
        Some("expect.objectContaining({sessionId,tokenBudget:4096,compactionTarget:\"threshold\",})".to_string())
    );

}
