use chrono::{DateTime, FixedOffset, Utc};
use serde_json::{Value, json};

use crate::store::conversation_store::{
    ConversationStore, MessagePartRecord, MessagePartType, MessageRole,
};
use crate::store::summary_store::{
    ContextItemRecord, ContextItemType, SummaryKind, SummaryRecord, SummaryStore,
};
use crate::transcript_repair::sanitize_tool_use_result_pairing;
use crate::types::{AgentMessage, MessageUsage, UsageCost};

#[derive(Clone, Debug, PartialEq)]
pub struct AssembleContextInput {
    pub conversation_id: i64,
    pub token_budget: i64,
    pub fresh_tail_count: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssembleContextStats {
    pub raw_message_count: usize,
    pub summary_count: usize,
    pub total_context_items: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssembleContextResult {
    pub messages: Vec<AgentMessage>,
    pub estimated_tokens: i64,
    pub system_prompt_addition: Option<String>,
    pub stats: AssembleContextStats,
}

#[derive(Clone, Debug, PartialEq)]
struct SummaryPromptSignal {
    kind: SummaryKind,
    depth: i64,
    descendant_count: i64,
}

#[derive(Clone, Debug, PartialEq)]
struct ResolvedItem {
    ordinal: i64,
    message: AgentMessage,
    tokens: i64,
    is_message: bool,
    summary_signal: Option<SummaryPromptSignal>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeRole {
    User,
    Assistant,
    ToolResult,
}

impl RuntimeRole {
    fn as_str(self) -> &'static str {
        match self {
            RuntimeRole::User => "user",
            RuntimeRole::Assistant => "assistant",
            RuntimeRole::ToolResult => "toolResult",
        }
    }
}

#[derive(Clone)]
pub struct ContextAssembler {
    conversation_store: ConversationStore,
    summary_store: SummaryStore,
    timezone: Option<String>,
}

impl ContextAssembler {
    pub fn new(conversation_store: ConversationStore, summary_store: SummaryStore) -> Self {
        Self {
            conversation_store,
            summary_store,
            timezone: None,
        }
    }

    pub fn with_timezone(mut self, timezone: Option<String>) -> Self {
        self.timezone = timezone;
        self
    }

    pub fn set_timezone(&mut self, timezone: Option<String>) {
        self.timezone = timezone;
    }

    pub async fn assemble(
        &self,
        input: AssembleContextInput,
    ) -> anyhow::Result<AssembleContextResult> {
        let conversation_id = input.conversation_id;
        let token_budget = input.token_budget;
        let fresh_tail_count = input.fresh_tail_count.unwrap_or(8);
        let context_items = self.summary_store.get_context_items(conversation_id)?;

        if context_items.is_empty() {
            return Ok(AssembleContextResult {
                messages: vec![],
                estimated_tokens: 0,
                system_prompt_addition: None,
                stats: AssembleContextStats {
                    raw_message_count: 0,
                    summary_count: 0,
                    total_context_items: 0,
                },
            });
        }

        let resolved = self.resolve_items(&context_items).await?;

        let mut raw_message_count = 0_usize;
        let mut summary_count = 0_usize;
        let mut summary_signals: Vec<SummaryPromptSignal> = vec![];
        for item in &resolved {
            if item.is_message {
                raw_message_count += 1;
            } else {
                summary_count += 1;
                if let Some(signal) = item.summary_signal.clone() {
                    summary_signals.push(signal);
                }
            }
        }
        let system_prompt_addition = build_system_prompt_addition(&summary_signals);

        let tail_start = ((resolved.len() as i64) - fresh_tail_count)
            .max(0)
            .min(resolved.len() as i64) as usize;
        let fresh_tail = resolved[tail_start..].to_vec();
        let evictable = resolved[..tail_start].to_vec();

        let tail_tokens: i64 = fresh_tail.iter().map(|item| item.tokens).sum();
        let remaining_budget = (token_budget - tail_tokens).max(0);
        let mut selected: Vec<ResolvedItem> = vec![];
        let evictable_total_tokens: i64 = evictable.iter().map(|item| item.tokens).sum();
        let evictable_tokens: i64;

        if evictable_total_tokens <= remaining_budget {
            selected.extend(evictable.clone());
            evictable_tokens = evictable_total_tokens;
        } else {
            let mut kept: Vec<ResolvedItem> = vec![];
            let mut accum = 0_i64;
            for idx in (0..evictable.len()).rev() {
                let item = &evictable[idx];
                if accum + item.tokens <= remaining_budget {
                    kept.push(item.clone());
                    accum += item.tokens;
                } else {
                    break;
                }
            }
            kept.reverse();
            selected.extend(kept);
            evictable_tokens = accum;
        }

        selected.extend(fresh_tail);
        let estimated_tokens = evictable_tokens + tail_tokens;

        let mut raw_messages: Vec<AgentMessage> =
            selected.into_iter().map(|item| item.message).collect();
        for message in &mut raw_messages {
            if message.role == "assistant" {
                if let Some(text) = message.content.as_str() {
                    message.content = Value::Array(vec![text_block(text)]);
                }
            }
        }

        Ok(AssembleContextResult {
            messages: sanitize_tool_use_result_pairing(raw_messages),
            estimated_tokens,
            system_prompt_addition,
            stats: AssembleContextStats {
                raw_message_count,
                summary_count,
                total_context_items: resolved.len(),
            },
        })
    }

    async fn resolve_items(
        &self,
        context_items: &[ContextItemRecord],
    ) -> anyhow::Result<Vec<ResolvedItem>> {
        let mut resolved: Vec<ResolvedItem> = vec![];
        for item in context_items {
            if let Some(result) = self.resolve_item(item).await? {
                resolved.push(result);
            }
        }
        Ok(resolved)
    }

    async fn resolve_item(&self, item: &ContextItemRecord) -> anyhow::Result<Option<ResolvedItem>> {
        if matches!(item.item_type, ContextItemType::Message) && item.message_id.is_some() {
            return self.resolve_message_item(item).await;
        }
        if matches!(item.item_type, ContextItemType::Summary) && item.summary_id.is_some() {
            return self.resolve_summary_item(item).await;
        }
        Ok(None)
    }

    async fn resolve_message_item(
        &self,
        item: &ContextItemRecord,
    ) -> anyhow::Result<Option<ResolvedItem>> {
        let Some(message_id) = item.message_id else {
            return Ok(None);
        };
        let Some(msg) = self.conversation_store.get_message_by_id(message_id)? else {
            return Ok(None);
        };

        let parts = self.conversation_store.get_message_parts(msg.message_id)?;
        let role_from_store = to_runtime_role(&msg.role, &parts);
        let tool_call_id = if matches!(role_from_store, RuntimeRole::ToolResult) {
            pick_tool_call_id(&parts)
        } else {
            None
        };
        let role = if matches!(role_from_store, RuntimeRole::ToolResult) && tool_call_id.is_none() {
            RuntimeRole::Assistant
        } else {
            role_from_store
        };
        let content = content_from_parts(&parts, role, &msg.content);
        let content_text = content
            .as_str()
            .map(ToString::to_string)
            .unwrap_or_else(|| {
                serde_json::to_string(&content).unwrap_or_else(|_| msg.content.clone())
            });
        let token_count = if msg.token_count > 0 {
            msg.token_count
        } else {
            estimate_tokens(&content_text)
        };

        let message = match role {
            RuntimeRole::Assistant => AgentMessage {
                role: role.as_str().to_string(),
                content,
                tool_call_id: None,
                tool_use_id: None,
                tool_name: None,
                stop_reason: None,
                is_error: None,
                usage: Some(MessageUsage {
                    input: 0,
                    output: clamp_i64_to_i32(token_count),
                    cache_read: 0,
                    cache_write: 0,
                    total_tokens: clamp_i64_to_i32(token_count),
                    cost: UsageCost {
                        input: 0.0,
                        output: 0.0,
                        cache_read: 0.0,
                        cache_write: 0.0,
                        total: 0.0,
                    },
                }),
                timestamp: None,
            },
            RuntimeRole::User => AgentMessage {
                role: role.as_str().to_string(),
                content,
                tool_call_id: None,
                tool_use_id: None,
                tool_name: None,
                stop_reason: None,
                is_error: None,
                usage: None,
                timestamp: None,
            },
            RuntimeRole::ToolResult => AgentMessage {
                role: role.as_str().to_string(),
                content,
                tool_call_id,
                tool_use_id: None,
                tool_name: None,
                stop_reason: None,
                is_error: None,
                usage: None,
                timestamp: None,
            },
        };

        Ok(Some(ResolvedItem {
            ordinal: item.ordinal,
            message,
            tokens: token_count,
            is_message: true,
            summary_signal: None,
        }))
    }

    async fn resolve_summary_item(
        &self,
        item: &ContextItemRecord,
    ) -> anyhow::Result<Option<ResolvedItem>> {
        let Some(summary_id) = item.summary_id.as_deref() else {
            return Ok(None);
        };
        let Some(summary) = self.summary_store.get_summary(summary_id)? else {
            return Ok(None);
        };

        let content =
            format_summary_content(&summary, &self.summary_store, self.timezone.as_deref()).await?;
        let tokens = estimate_tokens(&content);

        Ok(Some(ResolvedItem {
            ordinal: item.ordinal,
            message: AgentMessage {
                role: RuntimeRole::User.as_str().to_string(),
                content: Value::String(content),
                tool_call_id: None,
                tool_use_id: None,
                tool_name: None,
                stop_reason: None,
                is_error: None,
                usage: None,
                timestamp: None,
            },
            tokens,
            is_message: false,
            summary_signal: Some(SummaryPromptSignal {
                kind: summary.kind,
                depth: summary.depth,
                descendant_count: summary.descendant_count,
            }),
        }))
    }
}

fn clamp_i64_to_i32(value: i64) -> i32 {
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn estimate_tokens(text: &str) -> i64 {
    ((text.len() as f64) / 4.0).ceil() as i64
}

fn text_block(text: &str) -> Value {
    json!({
        "type": "text",
        "text": text,
    })
}

fn build_system_prompt_addition(summary_signals: &[SummaryPromptSignal]) -> Option<String> {
    if summary_signals.is_empty() {
        return None;
    }

    let max_depth = summary_signals
        .iter()
        .map(|signal| signal.depth)
        .max()
        .unwrap_or(0);
    let condensed_count = summary_signals
        .iter()
        .filter(|signal| matches!(signal.kind, SummaryKind::Condensed))
        .count();
    let heavily_compacted = max_depth >= 2 || condensed_count >= 2;
    let _descendant_count_total: i64 = summary_signals
        .iter()
        .map(|signal| signal.descendant_count)
        .sum();

    let mut sections: Vec<String> = vec![
        "## LCM Recall".to_string(),
        "".to_string(),
        "Summaries above are compressed context \u{2014} maps to details, not the details themselves.".to_string(),
        "".to_string(),
        "**Recall priority:** LCM tools first, then qmd (for Granola/Limitless/pre-LCM data), then memory_search as last resort.".to_string(),
        "".to_string(),
        "**Tool escalation:**".to_string(),
        "1. `lcm_grep` \u{2014} search by regex or full-text across messages and summaries".to_string(),
        "2. `lcm_describe` \u{2014} inspect a specific summary (cheap, no sub-agent)".to_string(),
        "3. `lcm_expand_query` \u{2014} deep recall: spawns bounded sub-agent, expands DAG, returns answer with cited summary IDs (~120s, don't ration it)".to_string(),
        "".to_string(),
        "**`lcm_expand_query` usage** \u{2014} two patterns (always requires `prompt`):".to_string(),
        "- With IDs: `lcm_expand_query(summaryIds: [\"sum_xxx\"], prompt: \"What config changes were discussed?\")`".to_string(),
        "- With search: `lcm_expand_query(query: \"database migration\", prompt: \"What strategy was decided?\")`".to_string(),
        "- Optional: `maxTokens` (default 2000), `conversationId`, `allConversations: true`".to_string(),
        "".to_string(),
        "**Summaries include \"Expand for details about:\" footers** listing compressed specifics. Use `lcm_expand_query` with that summary's ID to retrieve them.".to_string(),
    ];

    if heavily_compacted {
        sections.extend(
            [
                "",
                "**\u{26A0} Deeply compacted context \u{2014} expand before asserting specifics.**",
                "",
                "Default recall flow for precision work:",
                "1) `lcm_grep` to locate relevant summary/message IDs",
                "2) `lcm_expand_query` with a focused prompt",
                "3) Answer with citations to summary IDs used",
                "",
                "**Uncertainty checklist (run before answering):**",
                "- Am I making exact factual claims from a condensed summary?",
                "- Could compaction have omitted a crucial detail?",
                "- Would this answer fail if the user asks for proof?",
                "",
                "If yes to any \u{2192} expand first.",
                "",
                "**Do not guess** exact commands, SHAs, file paths, timestamps, config values, or causal claims from condensed summaries. Expand first or state that you need to expand.",
            ]
            .into_iter()
            .map(ToString::to_string),
        );
    } else {
        sections.extend(
            [
                "",
                "**For precision/evidence questions** (exact commands, SHAs, paths, timestamps, config values, root-cause chains): expand before answering.",
                "Do not guess from condensed summaries \u{2014} expand first or state uncertainty.",
            ]
            .into_iter()
            .map(ToString::to_string),
        );
    }

    Some(sections.join("\n"))
}

fn parse_json(value: Option<&str>) -> Option<Value> {
    let raw = value?;
    if raw.trim().is_empty() {
        return None;
    }
    serde_json::from_str(raw).ok()
}

fn get_original_role(parts: &[MessagePartRecord]) -> Option<String> {
    for part in parts {
        let Some(decoded) = parse_json(part.metadata.as_deref()) else {
            continue;
        };
        let role = decoded.get("originalRole").and_then(Value::as_str);
        if let Some(role) = role.filter(|role| !role.is_empty()) {
            return Some(role.to_string());
        }
    }
    None
}

fn to_runtime_role(db_role: &MessageRole, parts: &[MessagePartRecord]) -> RuntimeRole {
    let original_role = get_original_role(parts);
    match original_role.as_deref() {
        Some("toolResult") => RuntimeRole::ToolResult,
        Some("assistant") => RuntimeRole::Assistant,
        Some("user") => RuntimeRole::User,
        Some("system") => RuntimeRole::User,
        _ => match db_role {
            MessageRole::Tool => RuntimeRole::ToolResult,
            MessageRole::Assistant => RuntimeRole::Assistant,
            MessageRole::System | MessageRole::User => RuntimeRole::User,
        },
    }
}

fn block_from_part(part: &MessagePartRecord) -> Value {
    if let Some(decoded) = parse_json(part.metadata.as_deref()) {
        if let Some(raw) = decoded.get("raw").and_then(Value::as_object) {
            return Value::Object(raw.clone());
        }
    }

    if matches!(
        part.part_type,
        MessagePartType::Text | MessagePartType::Reasoning
    ) {
        return text_block(part.text_content.as_deref().unwrap_or_default());
    }
    if matches!(part.part_type, MessagePartType::Tool) {
        if let Some(tool_output) = parse_json(part.tool_output.as_deref()) {
            return tool_output;
        }
        if let Some(text) = part.text_content.as_deref() {
            return text_block(text);
        }
        return text_block(
            part.tool_output
                .as_deref()
                .or(part.tool_input.as_deref())
                .unwrap_or_default(),
        );
    }

    if let Some(text_content) = part.text_content.as_deref().filter(|text| !text.is_empty()) {
        return text_block(text_content);
    }

    if let Some(decoded_fallback) = parse_json(part.metadata.as_deref()) {
        if decoded_fallback.is_object() {
            return text_block(&serde_json::to_string(&decoded_fallback).unwrap_or_default());
        }
    }

    text_block("")
}

fn content_from_parts(
    parts: &[MessagePartRecord],
    role: RuntimeRole,
    fallback_content: &str,
) -> Value {
    if parts.is_empty() {
        return match role {
            RuntimeRole::Assistant => {
                if fallback_content.is_empty() {
                    Value::Array(vec![])
                } else {
                    Value::Array(vec![text_block(fallback_content)])
                }
            }
            RuntimeRole::ToolResult => Value::Array(vec![text_block(fallback_content)]),
            RuntimeRole::User => Value::String(fallback_content.to_string()),
        };
    }

    let blocks: Vec<Value> = parts.iter().map(block_from_part).collect();
    if matches!(role, RuntimeRole::User) && blocks.len() == 1 {
        if let Some(obj) = blocks[0].as_object() {
            if obj.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text) = obj.get("text").and_then(Value::as_str) {
                    return Value::String(text.to_string());
                }
            }
        }
    }
    Value::Array(blocks)
}

fn pick_tool_call_id(parts: &[MessagePartRecord]) -> Option<String> {
    for part in parts {
        if let Some(tool_call_id) = part
            .tool_call_id
            .as_deref()
            .filter(|value| !value.is_empty())
        {
            return Some(tool_call_id.to_string());
        }
        let Some(decoded) = parse_json(part.metadata.as_deref()) else {
            continue;
        };
        let Some(raw) = decoded.get("raw").and_then(Value::as_object) else {
            continue;
        };
        if let Some(id) = raw
            .get("toolCallId")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            return Some(id.to_string());
        }
        if let Some(id) = raw
            .get("tool_call_id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        {
            return Some(id.to_string());
        }
    }
    None
}

fn parse_fixed_offset(timezone: &str) -> Option<FixedOffset> {
    let mut chars = timezone.chars();
    let sign = chars.next()?;
    if sign != '+' && sign != '-' {
        return None;
    }
    let rest = chars.as_str();
    let (hours_raw, minutes_raw) = if let Some((h, m)) = rest.split_once(':') {
        (h, m)
    } else if rest.len() == 4 {
        (&rest[..2], &rest[2..])
    } else if rest.len() == 2 {
        (rest, "0")
    } else {
        return None;
    };
    let hours: i32 = hours_raw.parse().ok()?;
    let minutes: i32 = minutes_raw.parse().ok()?;
    if !(0..=23).contains(&hours) || !(0..=59).contains(&minutes) {
        return None;
    }
    let mut seconds = hours * 3600 + minutes * 60;
    if sign == '-' {
        seconds = -seconds;
    }
    FixedOffset::east_opt(seconds)
}

fn format_date_for_attribute(date: &DateTime<Utc>, timezone: Option<&str>) -> String {
    let tz = timezone.unwrap_or("UTC");
    if tz.eq_ignore_ascii_case("UTC") {
        return date.format("%Y-%m-%dT%H:%M:%S").to_string();
    }
    if let Some(offset) = parse_fixed_offset(tz) {
        return date
            .with_timezone(&offset)
            .format("%Y-%m-%dT%H:%M:%S")
            .to_string();
    }
    date.to_rfc3339()
}

fn summary_kind_label(kind: &SummaryKind) -> &'static str {
    match kind {
        SummaryKind::Leaf => "leaf",
        SummaryKind::Condensed => "condensed",
    }
}

async fn format_summary_content(
    summary: &SummaryRecord,
    summary_store: &SummaryStore,
    timezone: Option<&str>,
) -> anyhow::Result<String> {
    let mut attributes = vec![
        format!("id=\"{}\"", summary.summary_id),
        format!("kind=\"{}\"", summary_kind_label(&summary.kind)),
        format!("depth=\"{}\"", summary.depth),
        format!("descendant_count=\"{}\"", summary.descendant_count),
    ];
    if let Some(earliest_at) = summary.earliest_at.as_ref() {
        attributes.push(format!(
            "earliest_at=\"{}\"",
            format_date_for_attribute(earliest_at, timezone)
        ));
    }
    if let Some(latest_at) = summary.latest_at.as_ref() {
        attributes.push(format!(
            "latest_at=\"{}\"",
            format_date_for_attribute(latest_at, timezone)
        ));
    }

    let mut lines: Vec<String> = vec![];
    lines.push(format!("<summary {}>", attributes.join(" ")));
    if matches!(summary.kind, SummaryKind::Condensed) {
        let parents = summary_store.get_summary_parents(&summary.summary_id)?;
        if !parents.is_empty() {
            lines.push("  <parents>".to_string());
            for parent in parents {
                lines.push(format!("    <summary_ref id=\"{}\" />", parent.summary_id));
            }
            lines.push("  </parents>".to_string());
        }
    }
    lines.push("  <content>".to_string());
    lines.push(summary.content.clone());
    lines.push("  </content>".to_string());
    lines.push("</summary>".to_string());
    Ok(lines.join("\n"))
}
