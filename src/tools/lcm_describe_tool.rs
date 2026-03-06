use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::{Value, json};

use crate::engine::LcmContextEngineApi;
use crate::expansion_auth::{
    get_runtime_expansion_auth_manager, resolve_delegated_expansion_grant_id,
};
use crate::retrieval::{DescribeResultType, DescribeSubtreeNode, DescribeSummary};
use crate::tools::common::{ToolContentBlock, ToolResult, json_result};
use crate::tools::lcm_conversation_scope::resolve_lcm_conversation_scope;
use crate::types::LcmDependencies;

fn normalize_requested_token_cap(value: Option<&Value>) -> Option<i64> {
    value
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .map(|v| (v.trunc() as i64).max(1))
}

fn format_iso(value: Option<&DateTime<Utc>>) -> String {
    value
        .map(|v| v.to_rfc3339())
        .unwrap_or_else(|| "-".to_string())
}

fn summary_kind_label(kind: &crate::store::summary_store::SummaryKind) -> &'static str {
    match kind {
        crate::store::summary_store::SummaryKind::Leaf => "leaf",
        crate::store::summary_store::SummaryKind::Condensed => "condensed",
    }
}

fn format_number_en_us(value: i64) -> String {
    let negative = value < 0;
    let digits = value.abs().to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (idx, ch) in digits.chars().enumerate() {
        if idx > 0 && (digits.len() - idx) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    if negative { format!("-{}", out) } else { out }
}

#[derive(Clone)]
pub struct LcmDescribeTool {
    deps: Arc<dyn LcmDependencies>,
    lcm: Arc<dyn LcmContextEngineApi>,
    session_id: Option<String>,
    session_key: Option<String>,
}

impl LcmDescribeTool {
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

    fn render_summary_text(
        &self,
        id: &str,
        summary: &DescribeSummary,
        resolved_token_cap: i64,
    ) -> String {
        let mut lines = vec![];
        lines.push(format!("LCM_SUMMARY {}", id));
        lines.push(format!(
            "meta conv={} kind={} depth={} tok={} descTok={} srcTok={} desc={} range={}..{} budgetCap={}",
            summary.conversation_id,
            summary_kind_label(&summary.kind),
            summary.depth,
            summary.token_count,
            summary.descendant_token_count,
            summary.source_message_token_count,
            summary.descendant_count,
            format_iso(summary.earliest_at.as_ref()),
            format_iso(summary.latest_at.as_ref()),
            resolved_token_cap
        ));
        if !summary.parent_ids.is_empty() {
            lines.push(format!("parents {}", summary.parent_ids.join(" ")));
        }
        if !summary.child_ids.is_empty() {
            lines.push(format!("children {}", summary.child_ids.join(" ")));
        }
        lines.push("manifest".to_string());
        for node in &summary.subtree {
            lines.push(render_manifest_node(node, resolved_token_cap));
        }
        lines.push("content".to_string());
        lines.push(summary.content.clone());
        lines.join("\n")
    }

    pub async fn execute(&self, _tool_call_id: &str, params: Value) -> anyhow::Result<ToolResult> {
        let retrieval = self.lcm.get_retrieval();
        let p = params.as_object().cloned().unwrap_or_default();
        let id = p
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();

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

        let Some(result) = retrieval.describe(&id).await? else {
            return Ok(json_result(json!({
                "error": format!("Not found: {}", id),
                "hint": "Check the ID format (sum_xxx for summaries, file_xxx for files).",
            })));
        };

        let item_conversation_id = match &result.result {
            DescribeResultType::Summary(summary) => Some(summary.conversation_id),
            DescribeResultType::File(file) => Some(file.conversation_id),
        };
        if let (Some(scope_id), Some(item_id)) =
            (conversation_scope.conversation_id, item_conversation_id)
        {
            if item_id != scope_id {
                return Ok(json_result(json!({
                    "error": format!("Not found in conversation {}: {}", scope_id, id),
                    "hint": "Use allConversations=true for cross-conversation lookup.",
                })));
            }
        }

        match &result.result {
            DescribeResultType::Summary(summary) => {
                let requested_token_cap = normalize_requested_token_cap(p.get("tokenCap"));
                let session_key = self
                    .session_key
                    .as_deref()
                    .or(self.session_id.as_deref())
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                let delegated_grant_id = if self.deps.is_subagent_session_key(&session_key) {
                    resolve_delegated_expansion_grant_id(&session_key)
                } else {
                    None
                };
                let delegated_remaining_budget =
                    delegated_grant_id.as_deref().and_then(|grant_id| {
                        get_runtime_expansion_auth_manager()
                            .lock()
                            .get_remaining_token_budget(grant_id)
                    });
                let default_token_cap = i64::from(self.deps.config().max_expand_tokens.max(1));
                let resolved_token_cap = {
                    let base = requested_token_cap
                        .or(delegated_remaining_budget)
                        .unwrap_or(default_token_cap);
                    if let Some(remaining) = delegated_remaining_budget {
                        base.min(remaining).max(0)
                    } else {
                        base.max(1)
                    }
                };

                let text = self.render_summary_text(&id, summary, resolved_token_cap);
                let manifest_nodes = summary
                    .subtree
                    .iter()
                    .map(|node| {
                        let summaries_only_cost =
                            (node.token_count + node.descendant_token_count).max(0);
                        let with_messages_cost =
                            (summaries_only_cost + node.source_message_token_count).max(0);
                        json!({
                            "summaryId": node.summary_id,
                            "parentSummaryId": node.parent_summary_id,
                            "depthFromRoot": node.depth_from_root,
                            "depth": node.depth,
                            "kind": summary_kind_label(&node.kind),
                            "tokenCount": node.token_count,
                            "descendantCount": node.descendant_count,
                            "descendantTokenCount": node.descendant_token_count,
                            "sourceMessageTokenCount": node.source_message_token_count,
                            "childCount": node.child_count,
                            "earliestAt": node.earliest_at.as_ref().map(|v| v.to_rfc3339()),
                            "latestAt": node.latest_at.as_ref().map(|v| v.to_rfc3339()),
                            "path": node.path,
                            "costs": {
                                "summariesOnly": summaries_only_cost,
                                "withMessages": with_messages_cost
                            },
                            "budgetFit": {
                                "summariesOnly": summaries_only_cost <= resolved_token_cap,
                                "withMessages": with_messages_cost <= resolved_token_cap
                            }
                        })
                    })
                    .collect::<Vec<Value>>();

                Ok(ToolResult {
                    content: vec![ToolContentBlock {
                        r#type: "text".to_string(),
                        text,
                    }],
                    details: json!({
                        "id": result.id,
                        "type": "summary",
                        "summary": {
                            "conversationId": summary.conversation_id,
                            "kind": summary_kind_label(&summary.kind),
                            "content": summary.content,
                            "depth": summary.depth,
                            "tokenCount": summary.token_count,
                            "descendantCount": summary.descendant_count,
                            "descendantTokenCount": summary.descendant_token_count,
                            "sourceMessageTokenCount": summary.source_message_token_count,
                            "fileIds": summary.file_ids,
                            "parentIds": summary.parent_ids,
                            "childIds": summary.child_ids,
                            "messageIds": summary.message_ids,
                            "earliestAt": summary.earliest_at.as_ref().map(|v| v.to_rfc3339()),
                            "latestAt": summary.latest_at.as_ref().map(|v| v.to_rfc3339()),
                            "createdAt": summary.created_at.to_rfc3339(),
                        },
                        "manifest": {
                            "tokenCap": resolved_token_cap,
                            "budgetSource": if requested_token_cap.is_some() {
                                "request"
                            } else if delegated_remaining_budget.is_some() {
                                "delegated_grant_remaining"
                            } else {
                                "config_default"
                            },
                            "nodes": manifest_nodes
                        }
                    }),
                })
            }
            DescribeResultType::File(file) => {
                let mut lines = vec![
                    format!("## LCM File: {}", id),
                    String::new(),
                    format!("**Conversation:** {}", file.conversation_id),
                    format!(
                        "**Name:** {}",
                        file.file_name
                            .clone()
                            .unwrap_or_else(|| "(no name)".to_string())
                    ),
                    format!(
                        "**Type:** {}",
                        file.mime_type
                            .clone()
                            .unwrap_or_else(|| "unknown".to_string())
                    ),
                ];
                if let Some(byte_size) = file.byte_size {
                    lines.push(format!(
                        "**Size:** {} bytes",
                        format_number_en_us(byte_size)
                    ));
                }
                lines.push(format!("**Created:** {}", file.created_at.to_rfc3339()));
                if let Some(summary) = &file.exploration_summary {
                    lines.push(String::new());
                    lines.push("## Exploration Summary".to_string());
                    lines.push(String::new());
                    lines.push(summary.clone());
                } else {
                    lines.push(String::new());
                    lines.push("*No exploration summary available.*".to_string());
                }
                Ok(ToolResult {
                    content: vec![ToolContentBlock {
                        r#type: "text".to_string(),
                        text: lines.join("\n"),
                    }],
                    details: json!({
                        "id": result.id,
                        "type": "file",
                        "file": {
                            "conversationId": file.conversation_id,
                            "fileName": file.file_name,
                            "mimeType": file.mime_type,
                            "byteSize": file.byte_size,
                            "storageUri": file.storage_uri,
                            "explorationSummary": file.exploration_summary,
                            "createdAt": file.created_at.to_rfc3339(),
                        }
                    }),
                })
            }
        }
    }
}

fn render_manifest_node(node: &DescribeSubtreeNode, resolved_token_cap: i64) -> String {
    let summaries_only_cost = (node.token_count + node.descendant_token_count).max(0);
    let with_messages_cost = (summaries_only_cost + node.source_message_token_count).max(0);
    format!(
        "d{} {} k={} tok={} descTok={} srcTok={} desc={} child={} range={}..{} cost[s={},m={}] budget[s={},m={}]",
        node.depth_from_root,
        node.summary_id,
        summary_kind_label(&node.kind),
        node.token_count,
        node.descendant_token_count,
        node.source_message_token_count,
        node.descendant_count,
        node.child_count,
        format_iso(node.earliest_at.as_ref()),
        format_iso(node.latest_at.as_ref()),
        summaries_only_cost,
        with_messages_cost,
        if summaries_only_cost <= resolved_token_cap {
            "in"
        } else {
            "over"
        },
        if with_messages_cost <= resolved_token_cap {
            "in"
        } else {
            "over"
        }
    )
}

pub fn create_lcm_describe_tool(
    deps: Arc<dyn LcmDependencies>,
    lcm: Arc<dyn LcmContextEngineApi>,
    session_id: Option<String>,
    session_key: Option<String>,
) -> LcmDescribeTool {
    LcmDescribeTool::new(deps, lcm, session_id, session_key)
}
