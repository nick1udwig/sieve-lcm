use std::sync::Arc;

use serde_json::{Value, json};

use crate::engine::LcmContextEngineApi;
use crate::store::conversation_store::MessageRole;
use crate::tools::common::{ToolContentBlock, ToolResult, json_result};
use crate::tools::lcm_conversation_scope::{
    parse_iso_timestamp_param, resolve_lcm_conversation_scope,
};
use crate::types::LcmDependencies;

const MAX_RESULT_CHARS: usize = 40_000;

fn truncate_snippet(content: &str, max_len: usize) -> String {
    let single_line = content.replace('\n', " ").trim().to_string();
    if single_line.len() <= max_len {
        return single_line;
    }
    format!("{}...", &single_line[..max_len.saturating_sub(3)])
}

fn message_role_label(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

#[derive(Clone)]
pub struct LcmGrepTool {
    deps: Arc<dyn LcmDependencies>,
    lcm: Arc<dyn LcmContextEngineApi>,
    session_id: Option<String>,
    session_key: Option<String>,
}

impl LcmGrepTool {
    pub fn new(
        deps: Arc<dyn LcmDependencies>,
        lcm: Arc<dyn LcmContextEngineApi>,
        session_id: Option<String>,
        session_key: Option<String>,
    ) -> Self {
        Self {
            deps,
            lcm,
            session_id,
            session_key,
        }
    }

    pub async fn execute(&self, _tool_call_id: &str, params: Value) -> anyhow::Result<ToolResult> {
        let retrieval = self.lcm.get_retrieval();
        let p = params.as_object().cloned().unwrap_or_default();
        let pattern = p
            .get("pattern")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        let mode = p
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("regex")
            .to_string();
        let scope = p
            .get("scope")
            .and_then(Value::as_str)
            .unwrap_or("both")
            .to_string();
        let limit = p
            .get("limit")
            .and_then(Value::as_f64)
            .map(|v| v.trunc() as i64)
            .unwrap_or(50);
        let since = match parse_iso_timestamp_param(&p, "since") {
            Ok(value) => value,
            Err(error) => {
                return Ok(json_result(json!({
                    "error": error.to_string(),
                })));
            }
        };
        let before = match parse_iso_timestamp_param(&p, "before") {
            Ok(value) => value,
            Err(error) => {
                return Ok(json_result(json!({
                    "error": error.to_string(),
                })));
            }
        };
        if let (Some(since), Some(before)) = (since.as_ref(), before.as_ref()) {
            if since >= before {
                return Ok(json_result(json!({
                    "error": "`since` must be earlier than `before`."
                })));
            }
        }

        let conversation_scope = resolve_lcm_conversation_scope(
            self.lcm.as_ref(),
            &p,
            self.session_id.as_deref(),
            self.session_key.as_deref(),
            Some(self.deps.as_ref()),
        )
        .await?;
        if !conversation_scope.all_conversations && conversation_scope.conversation_id.is_none() {
            return Ok(json_result(json!({
                "error": "No LCM conversation found for this session. Provide conversationId or set allConversations=true."
            })));
        }

        let result = retrieval
            .grep(crate::retrieval::GrepInput {
                query: pattern.clone(),
                mode: mode.clone(),
                scope: scope.clone(),
                conversation_id: conversation_scope.conversation_id,
                since,
                before,
                limit: Some(limit),
            })
            .await?;

        let mut lines = vec![
            "## LCM Grep Results".to_string(),
            format!("**Pattern:** `{}`", pattern),
            format!("**Mode:** {} | **Scope:** {}", mode, scope),
        ];
        if conversation_scope.all_conversations {
            lines.push("**Conversation scope:** all conversations".to_string());
        } else if let Some(conversation_id) = conversation_scope.conversation_id {
            lines.push(format!("**Conversation scope:** {}", conversation_id));
        }
        if since.is_some() || before.is_some() {
            let since_text = since
                .map(|v| format!("since {}", v.to_rfc3339()))
                .unwrap_or_else(|| "since -∞".to_string());
            let before_text = before
                .map(|v| format!("before {}", v.to_rfc3339()))
                .unwrap_or_else(|| "before +∞".to_string());
            lines.push(format!("**Time filter:** {} | {}", since_text, before_text));
        }
        lines.push(format!("**Total matches:** {}", result.total_matches));
        lines.push(String::new());

        let mut current_chars = lines.join("\n").len();

        if !result.messages.is_empty() {
            lines.push("### Messages".to_string());
            lines.push(String::new());
            for msg in &result.messages {
                let snippet = truncate_snippet(&msg.snippet, 200);
                let line = format!(
                    "- [msg#{}] ({}, {}): {}",
                    msg.message_id,
                    message_role_label(&msg.role),
                    msg.created_at.to_rfc3339(),
                    snippet
                );
                if current_chars + line.len() > MAX_RESULT_CHARS {
                    lines.push("*(truncated — more results available)*".to_string());
                    break;
                }
                lines.push(line.clone());
                current_chars += line.len();
            }
            lines.push(String::new());
        }

        if !result.summaries.is_empty() {
            lines.push("### Summaries".to_string());
            lines.push(String::new());
            for sum in &result.summaries {
                let snippet = truncate_snippet(&sum.snippet, 200);
                let line = format!(
                    "- [{}] ({}, {}): {}",
                    sum.summary_id,
                    match sum.kind {
                        crate::store::summary_store::SummaryKind::Leaf => "leaf",
                        crate::store::summary_store::SummaryKind::Condensed => "condensed",
                    },
                    sum.created_at.to_rfc3339(),
                    snippet
                );
                if current_chars + line.len() > MAX_RESULT_CHARS {
                    lines.push("*(truncated — more results available)*".to_string());
                    break;
                }
                lines.push(line.clone());
                current_chars += line.len();
            }
            lines.push(String::new());
        }

        if result.total_matches == 0 {
            lines.push("No matches found.".to_string());
        }

        Ok(ToolResult {
            content: vec![ToolContentBlock {
                r#type: "text".to_string(),
                text: lines.join("\n"),
            }],
            details: json!({
                "messageCount": result.messages.len(),
                "summaryCount": result.summaries.len(),
                "totalMatches": result.total_matches,
            }),
        })
    }
}

pub fn create_lcm_grep_tool(
    deps: Arc<dyn LcmDependencies>,
    lcm: Arc<dyn LcmContextEngineApi>,
    session_id: Option<String>,
    session_key: Option<String>,
) -> LcmGrepTool {
    LcmGrepTool::new(deps, lcm, session_id, session_key)
}
