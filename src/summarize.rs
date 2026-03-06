use std::collections::HashSet;
use std::sync::Arc;
use std::{future::Future, pin::Pin};

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Number, Value, json};

use crate::db::config::resolve_lcm_config;
use crate::types::{CompletionMessage, CompletionRequest, LcmDependencies};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LcmSummarizeOptions {
    pub previous_summary: Option<String>,
    pub is_condensed: Option<bool>,
    pub depth: Option<i64>,
}

pub type LcmSummarizeFn = Arc<
    dyn Fn(
            String,
            bool,
            Option<LcmSummarizeOptions>,
        ) -> Pin<Box<dyn Future<Output = String> + Send + 'static>>
        + Send
        + Sync,
>;

#[derive(Clone, Debug, PartialEq)]
pub struct LcmSummarizerLegacyParams {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub config: Option<Value>,
    pub agent_dir: Option<String>,
    pub auth_profile_id: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SummaryMode {
    Normal,
    Aggressive,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NormalizedCompletionSummary {
    summary: String,
    block_types: Vec<String>,
}

const DEFAULT_CONDENSED_TARGET_TOKENS: i32 = 2_000;
const LCM_SUMMARIZER_SYSTEM_PROMPT: &str = "You are a context-compaction summarization engine. Follow user instructions exactly and return plain text summary content only.";
const DIAGNOSTIC_MAX_DEPTH: usize = 4;
const DIAGNOSTIC_MAX_ARRAY_ITEMS: usize = 8;
const DIAGNOSTIC_MAX_OBJECT_KEYS: usize = 16;
const DIAGNOSTIC_MAX_CHARS: usize = 1_200;

static DIAGNOSTIC_SENSITIVE_KEY_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new("(?i)(api[-_]?key|authorization|token|secret|password|cookie|set-cookie|private[-_]?key|bearer)")
        .expect("valid diagnostic sensitive key regex")
});

fn normalize_provider_id(provider: &str) -> String {
    provider.trim().to_lowercase()
}

fn resolve_provider_api_from_legacy_config(config: &Value, provider: &str) -> Option<String> {
    let providers = config
        .as_object()?
        .get("models")?
        .as_object()?
        .get("providers")?
        .as_object()?;

    if let Some(direct) = providers.get(provider).and_then(Value::as_object) {
        if let Some(api) = direct.get("api").and_then(Value::as_str) {
            let trimmed = api.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }

    let normalized_provider = normalize_provider_id(provider);
    for (entry_provider, entry_value) in providers {
        if normalize_provider_id(entry_provider) != normalized_provider {
            continue;
        }
        let Some(entry_obj) = entry_value.as_object() else {
            continue;
        };
        let Some(api) = entry_obj.get("api").and_then(Value::as_str) else {
            continue;
        };
        let trimmed = api.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    None
}

fn estimate_tokens(text: &str) -> i32 {
    ((text.chars().count() as f64) / 4.0).ceil() as i32
}

fn normalize_text_fragments(chunks: &[String]) -> String {
    let mut normalized = Vec::new();
    let mut seen = HashSet::new();
    for chunk in chunks {
        let trimmed = chunk.trim();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.to_string()) {
            normalized.push(trimmed.to_string());
        }
    }
    normalized.join("\n").trim().to_string()
}

fn append_text_value(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(text) => out.push(text.clone()),
        Value::Array(items) => {
            for item in items {
                append_text_value(item, out);
            }
        }
        Value::Object(map) => {
            if let Some(Value::String(v)) = map.get("value") {
                out.push(v.clone());
            }
            if let Some(Value::String(v)) = map.get("text") {
                out.push(v.clone());
            }
        }
        _ => {}
    }
}

fn collect_text_like_fields(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_text_like_fields(item, out);
            }
        }
        Value::Object(map) => {
            for key in ["text", "output_text", "thinking"] {
                if let Some(found) = map.get(key) {
                    append_text_value(found, out);
                }
            }
            for key in ["content", "summary", "output", "message", "response"] {
                if let Some(found) = map.get(key) {
                    collect_text_like_fields(found, out);
                }
            }
        }
        _ => {}
    }
}

fn collect_block_types(value: &Value, out: &mut HashSet<String>) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_block_types(item, out);
            }
        }
        Value::Object(map) => {
            if let Some(ty) = map.get("type").and_then(Value::as_str) {
                let trimmed = ty.trim();
                if !trimmed.is_empty() {
                    out.insert(trimmed.to_string());
                }
            }
            for nested in map.values() {
                collect_block_types(nested, out);
            }
        }
        _ => {}
    }
}

fn normalize_completion_summary(value: &Value) -> NormalizedCompletionSummary {
    let mut chunks = Vec::new();
    let mut block_type_set = HashSet::new();

    collect_text_like_fields(value, &mut chunks);
    collect_block_types(value, &mut block_type_set);

    let mut block_types = block_type_set.into_iter().collect::<Vec<String>>();
    block_types.sort_unstable();

    NormalizedCompletionSummary {
        summary: normalize_text_fragments(&chunks),
        block_types,
    }
}

fn format_block_types(block_types: &[String]) -> String {
    if block_types.is_empty() {
        return "(none)".to_string();
    }
    block_types.join(",")
}

fn truncate_diagnostic_text(value: &str, max_chars: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        return value.to_string();
    }

    let mut end_byte = value.len();
    for (idx, (byte_idx, _)) in value.char_indices().enumerate() {
        if idx == max_chars {
            end_byte = byte_idx;
            break;
        }
    }
    format!(
        "{}...[truncated:{} chars]",
        &value[..end_byte],
        char_count.saturating_sub(max_chars)
    )
}

fn sanitize_for_diagnostics(value: &Value, depth: usize) -> Value {
    if depth >= DIAGNOSTIC_MAX_DEPTH {
        return Value::String("[max-depth]".to_string());
    }

    match value {
        Value::String(text) => Value::String(truncate_diagnostic_text(text, DIAGNOSTIC_MAX_CHARS)),
        Value::Null | Value::Bool(_) | Value::Number(_) => value.clone(),
        Value::Array(items) => {
            let mut head = items
                .iter()
                .take(DIAGNOSTIC_MAX_ARRAY_ITEMS)
                .map(|entry| sanitize_for_diagnostics(entry, depth + 1))
                .collect::<Vec<Value>>();
            if items.len() > DIAGNOSTIC_MAX_ARRAY_ITEMS {
                head.push(Value::String(format!(
                    "[+{} more items]",
                    items.len() - DIAGNOSTIC_MAX_ARRAY_ITEMS
                )));
            }
            Value::Array(head)
        }
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, entry) in map.iter().take(DIAGNOSTIC_MAX_OBJECT_KEYS) {
                if DIAGNOSTIC_SENSITIVE_KEY_PATTERN.is_match(key) {
                    out.insert(key.clone(), Value::String("[redacted]".to_string()));
                } else {
                    out.insert(key.clone(), sanitize_for_diagnostics(entry, depth + 1));
                }
            }
            if map.len() > DIAGNOSTIC_MAX_OBJECT_KEYS {
                out.insert(
                    "__truncated_keys__".to_string(),
                    Value::Number(Number::from(
                        (map.len() - DIAGNOSTIC_MAX_OBJECT_KEYS) as u64,
                    )),
                );
            }
            Value::Object(out)
        }
    }
}

fn format_diagnostic_payload(value: &Value) -> String {
    match serde_json::to_string(&sanitize_for_diagnostics(value, 0)) {
        Ok(json) => {
            if json.is_empty() {
                "\"\"".to_string()
            } else {
                truncate_diagnostic_text(&json, DIAGNOSTIC_MAX_CHARS)
            }
        }
        Err(_) => "\"[unserializable]\"".to_string(),
    }
}

fn extract_response_diagnostics(result: &Value) -> String {
    let Some(obj) = result.as_object() else {
        return String::new();
    };

    let mut parts: Vec<String> = Vec::new();

    let top_level_keys = obj.keys().take(24).cloned().collect::<Vec<String>>();
    if !top_level_keys.is_empty() {
        parts.push(format!("keys={}", top_level_keys.join(",")));
    }

    if let Some(content_val) = obj.get("content") {
        match content_val {
            Value::Array(items) => {
                parts.push("content_kind=array".to_string());
                parts.push(format!("content_len={}", items.len()));
            }
            Value::Null => parts.push("content_kind=null".to_string()),
            other => parts.push(format!("content_kind={}", kind_of_value(other))),
        }
        parts.push(format!(
            "content_preview={}",
            format_diagnostic_payload(content_val)
        ));
    } else {
        parts.push("content_kind=missing".to_string());
    }

    let mut envelope_payload = Map::new();
    for key in ["summary", "output", "message", "response"] {
        if let Some(value) = obj.get(key) {
            envelope_payload.insert(key.to_string(), value.clone());
        }
    }
    if !envelope_payload.is_empty() {
        parts.push(format!(
            "payload_preview={}",
            format_diagnostic_payload(&Value::Object(envelope_payload))
        ));
    }

    for key in ["id", "request_id", "x-request-id"] {
        if let Some(value) = obj.get(key).and_then(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                parts.push(format!("{key}={trimmed}"));
            }
        }
    }

    if let Some(model) = obj.get("model").and_then(Value::as_str) {
        let trimmed = model.trim();
        if !trimmed.is_empty() {
            parts.push(format!("resp_model={trimmed}"));
        }
    }
    if let Some(provider) = obj.get("provider").and_then(Value::as_str) {
        let trimmed = provider.trim();
        if !trimmed.is_empty() {
            parts.push(format!("resp_provider={trimmed}"));
        }
    }
    for key in [
        "request_provider",
        "request_model",
        "request_api",
        "request_reasoning",
        "request_has_system",
        "request_temperature",
        "request_temperature_sent",
    ] {
        if let Some(value) = obj.get(key).and_then(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                parts.push(format!("{key}={trimmed}"));
            }
        }
    }

    if let Some(usage) = obj.get("usage").and_then(Value::as_object) {
        let mut tokens = Vec::new();
        for key in [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input",
            "output",
            "cacheRead",
            "cacheWrite",
        ] {
            if let Some(value) = usage.get(key).and_then(Value::as_f64) {
                if value.fract() == 0.0 {
                    tokens.push(format!("{key}={}", value as i64));
                } else {
                    tokens.push(format!("{key}={value}"));
                }
            }
        }
        if !tokens.is_empty() {
            parts.push(tokens.join(","));
        }
    }

    let finish_reason = obj
        .get("finish_reason")
        .and_then(Value::as_str)
        .or_else(|| obj.get("stopReason").and_then(Value::as_str))
        .or_else(|| obj.get("stop_reason").and_then(Value::as_str));
    if let Some(finish_reason) = finish_reason {
        let trimmed = finish_reason.trim();
        if !trimmed.is_empty() {
            parts.push(format!("finish={trimmed}"));
        }
    }

    if let Some(error_message) = obj.get("errorMessage").and_then(Value::as_str) {
        let trimmed = error_message.trim();
        if !trimmed.is_empty() {
            parts.push(format!(
                "error_message={}",
                truncate_diagnostic_text(trimmed, 400)
            ));
        }
    }
    if let Some(error_payload) = obj.get("error") {
        parts.push(format!(
            "error_preview={}",
            format_diagnostic_payload(error_payload)
        ));
    }

    parts.join("; ")
}

fn resolve_target_tokens(
    input_tokens: i32,
    mode: SummaryMode,
    is_condensed: bool,
    condensed_target_tokens: i32,
) -> i32 {
    if is_condensed {
        return 512.max(condensed_target_tokens);
    }

    match mode {
        SummaryMode::Aggressive => 96.max(640.min((input_tokens as f64 * 0.2).floor() as i32)),
        SummaryMode::Normal => 192.max(1200.min((input_tokens as f64 * 0.35).floor() as i32)),
    }
}

fn build_leaf_summary_prompt(
    text: &str,
    mode: SummaryMode,
    target_tokens: i32,
    previous_summary: Option<&str>,
    custom_instructions: Option<&str>,
) -> String {
    let previous_context = previous_summary
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or("(none)");

    let policy = match mode {
        SummaryMode::Aggressive => vec![
            "Aggressive summary policy:",
            "- Keep only durable facts and current task state.",
            "- Remove examples, repetition, and low-value narrative details.",
            "- Preserve explicit TODOs, blockers, decisions, and constraints.",
        ]
        .join("\n"),
        SummaryMode::Normal => vec![
            "Normal summary policy:",
            "- Preserve key decisions, rationale, constraints, and active tasks.",
            "- Keep essential technical details needed to continue work safely.",
            "- Remove obvious repetition and conversational filler.",
        ]
        .join("\n"),
    };

    let instruction_block = custom_instructions
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|text| format!("Operator instructions:\n{text}"))
        .unwrap_or_else(|| "Operator instructions: (none)".to_string());

    vec![
        "You summarize a SEGMENT of an OpenClaw conversation for future model turns.".to_string(),
        "Treat this as incremental memory compaction input, not a full-conversation summary."
            .to_string(),
        policy,
        instruction_block,
        vec![
            "Output requirements:",
            "- Plain text only.",
            "- No preamble, headings, or markdown formatting.",
            "- Keep it concise while preserving required details.",
            "- Track file operations (created, modified, deleted, renamed) with file paths and current status.",
            "- If no file operations appear, include exactly: \"Files: none\".",
            "- End with exactly: \"Expand for details about: <comma-separated list of what was dropped or compressed>\".",
            &format!("- Target length: about {target_tokens} tokens or less."),
        ]
        .join("\n"),
        format!("<previous_context>\n{previous_context}\n</previous_context>"),
        format!("<conversation_segment>\n{text}\n</conversation_segment>"),
    ]
    .join("\n\n")
}

fn build_d1_prompt(
    text: &str,
    target_tokens: i32,
    previous_summary: Option<&str>,
    custom_instructions: Option<&str>,
) -> String {
    let instruction_block = custom_instructions
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|text| format!("Operator instructions:\n{text}"))
        .unwrap_or_else(|| "Operator instructions: (none)".to_string());

    let previous_context = previous_summary.map(str::trim).filter(|v| !v.is_empty());
    let previous_context_block = if let Some(previous_context) = previous_context {
        [
            "It already has this preceding summary as context. Do not repeat information",
            "that appears there unchanged. Focus on what is new, changed, or resolved:",
            "",
            &format!("<previous_context>\n{previous_context}\n</previous_context>"),
        ]
        .join("\n")
    } else {
        "Focus on what matters for continuation:".to_string()
    };

    vec![
        "You are compacting leaf-level conversation summaries into a single condensed memory node."
            .to_string(),
        "You are preparing context for a fresh model instance that will continue this conversation."
            .to_string(),
        instruction_block,
        previous_context_block,
        vec![
            "Preserve:",
            "- Decisions made and their rationale when rationale matters going forward.",
            "- Earlier decisions that were superseded, and what replaced them.",
            "- Completed tasks/topics with outcomes.",
            "- In-progress items with current state and what remains.",
            "- Blockers, open questions, and unresolved tensions.",
            "- Specific references (names, paths, URLs, identifiers) needed for continuation.",
            "",
            "Drop low-value detail:",
            "- Context that has not changed from previous_context.",
            "- Intermediate dead ends where the conclusion is already known.",
            "- Transient states that are already resolved.",
            "- Tool-internal mechanics and process scaffolding.",
            "",
            "Use plain text. No mandatory structure.",
            "Include a timeline with timestamps (hour or half-hour) for significant events.",
            "Present information chronologically and mark superseded decisions.",
            "- End with exactly: \"Expand for details about: <comma-separated list of what was dropped or compressed>\".",
            &format!("Target length: about {target_tokens} tokens."),
        ]
        .join("\n"),
        format!("<conversation_to_condense>\n{text}\n</conversation_to_condense>"),
    ]
    .join("\n\n")
}

fn build_d2_prompt(text: &str, target_tokens: i32, custom_instructions: Option<&str>) -> String {
    let instruction_block = custom_instructions
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|text| format!("Operator instructions:\n{text}"))
        .unwrap_or_else(|| "Operator instructions: (none)".to_string());

    vec![
        "You are condensing multiple session-level summaries into a higher-level memory node."
            .to_string(),
        "A future model should understand trajectory, not per-session minutiae.".to_string(),
        instruction_block,
        vec![
            "Preserve:",
            "- Decisions still in effect and their rationale.",
            "- Decisions that evolved: what changed and why.",
            "- Completed work with outcomes.",
            "- Active constraints, limitations, and known issues.",
            "- Current state of in-progress work.",
            "",
            "Drop:",
            "- Session-local operational detail and process mechanics.",
            "- Identifiers that are no longer relevant.",
            "- Intermediate states superseded by later outcomes.",
            "",
            "Use plain text. Brief headers are fine if useful.",
            "Include a timeline with dates and approximate time of day for key milestones.",
            "- End with exactly: \"Expand for details about: <comma-separated list of what was dropped or compressed>\".",
            &format!("Target length: about {target_tokens} tokens."),
        ]
        .join("\n"),
        format!("<conversation_to_condense>\n{text}\n</conversation_to_condense>"),
    ]
    .join("\n\n")
}

fn build_d3_plus_prompt(
    text: &str,
    target_tokens: i32,
    custom_instructions: Option<&str>,
) -> String {
    let instruction_block = custom_instructions
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|text| format!("Operator instructions:\n{text}"))
        .unwrap_or_else(|| "Operator instructions: (none)".to_string());

    vec![
        "You are creating a high-level memory node from multiple phase-level summaries.".to_string(),
        "This may persist for the rest of the conversation. Keep only durable context.".to_string(),
        instruction_block,
        vec![
            "Preserve:",
            "- Key decisions and rationale.",
            "- What was accomplished and current state.",
            "- Active constraints and hard limitations.",
            "- Important relationships between people, systems, or concepts.",
            "- Durable lessons learned.",
            "",
            "Drop:",
            "- Operational and process detail.",
            "- Method details unless the method itself was the decision.",
            "- Specific references unless essential for continuation.",
            "",
            "Use plain text. Be concise.",
            "Include a brief timeline with dates (or date ranges) for major milestones.",
            "- End with exactly: \"Expand for details about: <comma-separated list of what was dropped or compressed>\".",
            &format!("Target length: about {target_tokens} tokens."),
        ]
        .join("\n"),
        format!("<conversation_to_condense>\n{text}\n</conversation_to_condense>"),
    ]
    .join("\n\n")
}

fn build_condensed_summary_prompt(
    text: &str,
    target_tokens: i32,
    depth: i64,
    previous_summary: Option<&str>,
    custom_instructions: Option<&str>,
) -> String {
    if depth <= 1 {
        return build_d1_prompt(text, target_tokens, previous_summary, custom_instructions);
    }
    if depth == 2 {
        return build_d2_prompt(text, target_tokens, custom_instructions);
    }
    build_d3_plus_prompt(text, target_tokens, custom_instructions)
}

fn build_deterministic_fallback_summary(text: &str, target_tokens: i32) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let max_chars = 256usize.max((target_tokens.max(0) as usize).saturating_mul(4));
    let char_count = trimmed.chars().count();
    if char_count <= max_chars {
        return trimmed.to_string();
    }

    let mut end_byte = trimmed.len();
    for (idx, (byte_idx, _)) in trimmed.char_indices().enumerate() {
        if idx == max_chars {
            end_byte = byte_idx;
            break;
        }
    }
    format!(
        "{}\n[LCM fallback summary; truncated for context management]",
        &trimmed[..end_byte]
    )
}

fn kind_of_value(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn log_error(deps: &Arc<dyn LcmDependencies>, message: String) {
    deps.logger().error(&message);
}

pub async fn create_lcm_summarize_from_legacy_params(
    deps: Arc<dyn LcmDependencies>,
    legacy_params: LcmSummarizerLegacyParams,
    custom_instructions: Option<String>,
) -> anyhow::Result<Option<LcmSummarizeFn>> {
    let provider_hint = legacy_params
        .provider
        .clone()
        .unwrap_or_default()
        .trim()
        .to_string();
    let model_hint = legacy_params
        .model
        .clone()
        .unwrap_or_default()
        .trim()
        .to_string();
    let model_ref = if model_hint.is_empty() {
        None
    } else {
        Some(model_hint.as_str())
    };

    let resolved = match deps.resolve_model(
        model_ref,
        if provider_hint.is_empty() {
            None
        } else {
            Some(provider_hint.as_str())
        },
    ) {
        Ok(value) => value,
        Err(err) => {
            log_error(
                &deps,
                format!("[lcm] createLcmSummarize: resolveModel FAILED: {err}"),
            );
            return Ok(None);
        }
    };

    let provider = resolved.provider.trim().to_string();
    let model = resolved.model.trim().to_string();
    if provider.is_empty() || model.is_empty() {
        log_error(
            &deps,
            format!(
                "[lcm] createLcmSummarize: empty provider=\"{}\" or model=\"{}\"",
                provider, model
            ),
        );
        return Ok(None);
    }

    let auth_profile_id = legacy_params.auth_profile_id.clone().and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    });
    let agent_dir = legacy_params.agent_dir.clone().and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    });
    let provider_api = legacy_params
        .config
        .as_ref()
        .and_then(|config| resolve_provider_api_from_legacy_config(config, &provider));
    let api_key = deps.get_api_key(&provider, &model);

    let runtime_lcm_config = resolve_lcm_config();
    let condensed_target_tokens = if runtime_lcm_config.condensed_target_tokens > 0 {
        runtime_lcm_config.condensed_target_tokens
    } else {
        DEFAULT_CONDENSED_TARGET_TOKENS
    };

    let summarize_deps = deps.clone();
    let summarize_provider = provider.clone();
    let summarize_model = model.clone();
    let summarize_api_key = api_key.clone();
    let summarize_provider_api = provider_api.clone();
    let summarize_auth_profile_id = auth_profile_id.clone();
    let summarize_agent_dir = agent_dir.clone();
    let summarize_runtime_config = legacy_params.config.clone();
    let summarize_custom_instructions = custom_instructions.clone();

    let summarize: LcmSummarizeFn = Arc::new(move |text, aggressive, options| {
        let deps = summarize_deps.clone();
        let provider = summarize_provider.clone();
        let model = summarize_model.clone();
        let api_key = summarize_api_key.clone();
        let provider_api = summarize_provider_api.clone();
        let auth_profile_id = summarize_auth_profile_id.clone();
        let agent_dir = summarize_agent_dir.clone();
        let runtime_config = summarize_runtime_config.clone();
        let custom_instructions = summarize_custom_instructions.clone();
        Box::pin(async move {
            if text.trim().is_empty() {
                return String::new();
            }

            let mode = if aggressive {
                SummaryMode::Aggressive
            } else {
                SummaryMode::Normal
            };
            let is_condensed = options
                .as_ref()
                .and_then(|opts| opts.is_condensed)
                .unwrap_or(false);
            let target_tokens = resolve_target_tokens(
                estimate_tokens(&text),
                mode,
                is_condensed,
                condensed_target_tokens,
            );
            let prompt = if is_condensed {
                let depth = options
                    .as_ref()
                    .and_then(|opts| opts.depth)
                    .map(|depth| depth.max(1))
                    .unwrap_or(1);
                let previous_summary = options
                    .as_ref()
                    .and_then(|opts| opts.previous_summary.as_deref());
                build_condensed_summary_prompt(
                    &text,
                    target_tokens,
                    depth,
                    previous_summary,
                    custom_instructions.as_deref(),
                )
            } else {
                let previous_summary = options
                    .as_ref()
                    .and_then(|opts| opts.previous_summary.as_deref());
                build_leaf_summary_prompt(
                    &text,
                    mode,
                    target_tokens,
                    previous_summary,
                    custom_instructions.as_deref(),
                )
            };

            let result = match deps
                .complete(CompletionRequest {
                    provider: Some(provider.clone()),
                    model: model.clone(),
                    api_key: api_key.clone(),
                    provider_api: provider_api.clone(),
                    auth_profile_id: auth_profile_id.clone(),
                    agent_dir: agent_dir.clone(),
                    runtime_config: runtime_config.clone(),
                    messages: vec![CompletionMessage {
                        role: "user".to_string(),
                        content: Value::String(prompt.clone()),
                    }],
                    system: Some(LCM_SUMMARIZER_SYSTEM_PROMPT.to_string()),
                    max_tokens: target_tokens,
                    temperature: Some(if aggressive { 0.1 } else { 0.2 }),
                    reasoning: None,
                })
                .await
            {
                Ok(result) => result,
                Err(err) => {
                    log_error(
                        &deps,
                        format!(
                            "[lcm] summarize complete failed; provider={provider}; model={model}; error={err}; source=fallback"
                        ),
                    );
                    return build_deterministic_fallback_summary(&text, target_tokens);
                }
            };

            let content_value = serde_json::to_value(&result.content).unwrap_or(Value::Null);
            let normalized = normalize_completion_summary(&content_value);
            let mut summary = normalized.summary;
            let mut summary_source = "content";

            if summary.is_empty() {
                let envelope_value = serde_json::to_value(&result).unwrap_or(Value::Null);
                let envelope_normalized = normalize_completion_summary(&envelope_value);
                if !envelope_normalized.summary.is_empty() {
                    summary = envelope_normalized.summary;
                    summary_source = "envelope";
                    log_error(
                        &deps,
                        format!(
                            "[lcm] recovered summary from response envelope; provider={provider}; model={model}; block_types={}; source=envelope",
                            format_block_types(&envelope_normalized.block_types)
                        ),
                    );
                }
            }

            if summary.is_empty() {
                let result_value = serde_json::to_value(&result).unwrap_or_else(|_| json!({}));
                let response_diag = extract_response_diagnostics(&result_value);
                let mut diag_parts = vec![
                    "[lcm] empty normalized summary on first attempt".to_string(),
                    format!("provider={provider}"),
                    format!("model={model}"),
                    format!(
                        "block_types={}",
                        format_block_types(&normalized.block_types)
                    ),
                    format!("response_blocks={}", result.content.len()),
                ];
                if !response_diag.is_empty() {
                    diag_parts.push(response_diag);
                }
                log_error(
                    &deps,
                    format!(
                        "{}; retrying with conservative settings",
                        diag_parts.join("; ")
                    ),
                );

                match deps
                    .complete(CompletionRequest {
                        provider: Some(provider.clone()),
                        model: model.clone(),
                        api_key: api_key.clone(),
                        provider_api: provider_api.clone(),
                        auth_profile_id: auth_profile_id.clone(),
                        agent_dir: agent_dir.clone(),
                        runtime_config: runtime_config.clone(),
                        messages: vec![CompletionMessage {
                            role: "user".to_string(),
                            content: Value::String(prompt.clone()),
                        }],
                        system: Some(LCM_SUMMARIZER_SYSTEM_PROMPT.to_string()),
                        max_tokens: target_tokens,
                        temperature: Some(0.05),
                        reasoning: Some("low".to_string()),
                    })
                    .await
                {
                    Ok(retry_result) => {
                        let retry_content_value =
                            serde_json::to_value(&retry_result.content).unwrap_or(Value::Null);
                        let retry_normalized = normalize_completion_summary(&retry_content_value);
                        summary = retry_normalized.summary;
                        if !summary.is_empty() {
                            summary_source = "retry";
                            log_error(
                                &deps,
                                format!(
                                    "[lcm] retry succeeded; provider={provider}; model={model}; block_types={}; source=retry",
                                    format_block_types(&retry_normalized.block_types)
                                ),
                            );
                        } else {
                            let retry_value =
                                serde_json::to_value(&retry_result).unwrap_or_else(|_| json!({}));
                            let retry_diag = extract_response_diagnostics(&retry_value);
                            let mut retry_parts = vec![
                                "[lcm] retry also returned empty summary".to_string(),
                                format!("provider={provider}"),
                                format!("model={model}"),
                                format!(
                                    "block_types={}",
                                    format_block_types(&retry_normalized.block_types)
                                ),
                                format!("response_blocks={}", retry_result.content.len()),
                            ];
                            if !retry_diag.is_empty() {
                                retry_parts.push(retry_diag);
                            }
                            log_error(
                                &deps,
                                format!("{}; falling back to truncation", retry_parts.join("; ")),
                            );
                        }
                    }
                    Err(retry_err) => {
                        log_error(
                            &deps,
                            format!(
                                "[lcm] retry failed; provider={provider} model={model}; error={retry_err}; falling back to truncation"
                            ),
                        );
                    }
                }
            }

            if summary.is_empty() {
                log_error(
                    &deps,
                    format!(
                        "[lcm] all extraction attempts exhausted; provider={provider}; model={model}; source=fallback"
                    ),
                );
                return build_deterministic_fallback_summary(&text, target_tokens);
            }

            if summary_source != "content" {
                log_error(
                    &deps,
                    format!(
                        "[lcm] summary resolved via non-content path; provider={provider}; model={model}; source={summary_source}"
                    ),
                );
            }

            summary
        })
    });

    Ok(Some(summarize))
}
