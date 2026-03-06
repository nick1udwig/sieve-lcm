use std::collections::BTreeSet;
use std::sync::Arc;

use crate::retrieval::{
    ExpandInput, ExpandResult as RetrievalExpandResult, GrepInput, RetrievalApi,
};

const SNIPPET_MAX_CHARS: usize = 200;

#[derive(Clone, Debug, PartialEq)]
pub struct ExpansionRequest {
    pub summary_ids: Vec<String>,
    pub max_depth: Option<i64>,
    pub token_cap: Option<i64>,
    pub include_messages: Option<bool>,
    pub conversation_id: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpansionChild {
    pub summary_id: String,
    pub kind: String,
    pub snippet: String,
    pub token_count: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpansionMessage {
    pub message_id: i64,
    pub role: String,
    pub snippet: String,
    pub token_count: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpansionEntry {
    pub summary_id: String,
    pub children: Vec<ExpansionChild>,
    pub messages: Vec<ExpansionMessage>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpansionResult {
    pub expansions: Vec<ExpansionEntry>,
    pub cited_ids: Vec<String>,
    pub total_tokens: i64,
    pub truncated: bool,
}

fn truncate_snippet(content: &str, max_chars: usize) -> String {
    if content.len() <= max_chars {
        content.to_string()
    } else {
        format!("{}...", &content[..max_chars])
    }
}

pub fn resolve_expansion_token_cap(
    requested_token_cap: Option<i64>,
    max_expand_tokens: i64,
) -> i64 {
    let max_expand_tokens = max_expand_tokens.max(1);
    match requested_token_cap {
        Some(value) => value.max(1).min(max_expand_tokens),
        None => max_expand_tokens,
    }
}

fn to_expansion_entry(summary_id: &str, raw: RetrievalExpandResult) -> ExpansionEntry {
    ExpansionEntry {
        summary_id: summary_id.to_string(),
        children: raw
            .children
            .into_iter()
            .map(|c| ExpansionChild {
                summary_id: c.summary_id,
                kind: format!("{:?}", c.kind).to_lowercase(),
                snippet: truncate_snippet(&c.content, SNIPPET_MAX_CHARS),
                token_count: c.token_count,
            })
            .collect(),
        messages: raw
            .messages
            .into_iter()
            .map(|m| ExpansionMessage {
                message_id: m.message_id,
                role: m.role,
                snippet: truncate_snippet(&m.content, SNIPPET_MAX_CHARS),
                token_count: m.token_count,
            })
            .collect(),
    }
}

fn collect_cited_ids(entry: &ExpansionEntry) -> Vec<String> {
    let mut ids = vec![entry.summary_id.clone()];
    ids.extend(entry.children.iter().map(|c| c.summary_id.clone()));
    ids
}

#[derive(Clone)]
pub struct ExpansionOrchestrator {
    retrieval: Arc<dyn RetrievalApi>,
}

impl ExpansionOrchestrator {
    pub fn new(retrieval: Arc<dyn RetrievalApi>) -> Self {
        Self { retrieval }
    }

    pub async fn expand(&self, request: ExpansionRequest) -> anyhow::Result<ExpansionResult> {
        let max_depth = request.max_depth.unwrap_or(3);
        let token_cap = request.token_cap.unwrap_or(i64::MAX);
        let include_messages = request.include_messages.unwrap_or(false);

        let mut out = ExpansionResult {
            expansions: vec![],
            cited_ids: vec![],
            total_tokens: 0,
            truncated: false,
        };
        let mut cited = BTreeSet::new();

        for summary_id in request.summary_ids {
            if out.truncated {
                break;
            }
            let remaining = token_cap.saturating_sub(out.total_tokens);
            if remaining <= 0 {
                out.truncated = true;
                break;
            }
            let raw = self
                .retrieval
                .expand(ExpandInput {
                    summary_id: summary_id.clone(),
                    depth: Some(max_depth),
                    include_messages: Some(include_messages),
                    token_cap: Some(remaining),
                })
                .await?;
            let entry = to_expansion_entry(&summary_id, raw.clone());
            let consumed = raw.estimated_tokens.max(0);
            if consumed > remaining {
                out.total_tokens += remaining;
                out.truncated = true;
            } else {
                out.total_tokens += consumed;
            }
            if raw.truncated {
                out.truncated = true;
            }
            for id in collect_cited_ids(&entry) {
                cited.insert(id);
            }
            out.expansions.push(entry);
        }
        out.cited_ids = cited.into_iter().collect();
        Ok(out)
    }

    pub async fn describe_and_expand(
        &self,
        query: &str,
        mode: &str,
        conversation_id: Option<i64>,
        max_depth: Option<i64>,
        token_cap: Option<i64>,
    ) -> anyhow::Result<ExpansionResult> {
        let grep = self
            .retrieval
            .grep(GrepInput {
                query: query.to_string(),
                mode: mode.to_string(),
                scope: "summaries".to_string(),
                conversation_id,
                since: None,
                before: None,
                limit: None,
            })
            .await?;

        let mut sorted = grep.summaries;
        sorted.sort_by(|a, b| {
            let recency = b.created_at.cmp(&a.created_at);
            if recency != std::cmp::Ordering::Equal {
                return recency;
            }
            let ar = a.rank.unwrap_or(f64::INFINITY);
            let br = b.rank.unwrap_or(f64::INFINITY);
            ar.partial_cmp(&br).unwrap_or(std::cmp::Ordering::Equal)
        });
        let summary_ids: Vec<String> = sorted.into_iter().map(|s| s.summary_id).collect();
        if summary_ids.is_empty() {
            return Ok(ExpansionResult {
                expansions: vec![],
                cited_ids: vec![],
                total_tokens: 0,
                truncated: false,
            });
        }
        self.expand(ExpansionRequest {
            summary_ids,
            max_depth,
            token_cap,
            include_messages: Some(false),
            conversation_id: conversation_id.unwrap_or(0),
        })
        .await
    }
}

pub fn distill_for_subagent(result: &ExpansionResult) -> String {
    let mut lines = vec![format!(
        "## Expansion Results ({} summaries, {} total tokens)",
        result.expansions.len(),
        result.total_tokens
    )];
    lines.push(String::new());

    for entry in &result.expansions {
        let kind = if !entry.children.is_empty() {
            "condensed"
        } else {
            "leaf"
        };
        let token_sum: i64 = entry.children.iter().map(|c| c.token_count).sum::<i64>()
            + entry.messages.iter().map(|m| m.token_count).sum::<i64>();
        lines.push(format!(
            "### {} ({}, {} tokens)",
            entry.summary_id, kind, token_sum
        ));
        if !entry.children.is_empty() {
            lines.push(format!(
                "Children: {}",
                entry
                    .children
                    .iter()
                    .map(|c| c.summary_id.clone())
                    .collect::<Vec<String>>()
                    .join(", ")
            ));
        }
        if !entry.messages.is_empty() {
            lines.push(format!(
                "Messages: {}",
                entry
                    .messages
                    .iter()
                    .map(|m| format!(
                        "msg#{} ({}, {} tokens)",
                        m.message_id, m.role, m.token_count
                    ))
                    .collect::<Vec<String>>()
                    .join(", ")
            ));
        }
        if let Some(first) = entry.children.iter().find(|c| !c.snippet.is_empty()) {
            lines.push(format!(
                "[Snippet: {}]",
                truncate_snippet(&first.snippet, SNIPPET_MAX_CHARS)
            ));
        }
        lines.push(String::new());
    }

    if !result.cited_ids.is_empty() {
        lines.push(format!(
            "Cited IDs for follow-up: {}",
            result.cited_ids.join(", ")
        ));
    }
    lines.push(format!(
        "[Truncated: {}]",
        if result.truncated { "yes" } else { "no" }
    ));
    lines.join("\n")
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolExecuteResult {
    pub text: String,
    pub details: serde_json::Value,
}

pub struct ExpansionToolDefinition {
    orchestrator: Arc<ExpansionOrchestrator>,
    max_expand_tokens: i64,
    conversation_id: i64,
}

impl ExpansionToolDefinition {
    pub fn new(
        orchestrator: Arc<ExpansionOrchestrator>,
        max_expand_tokens: i64,
        conversation_id: i64,
    ) -> Self {
        Self {
            orchestrator,
            max_expand_tokens: max_expand_tokens.max(1),
            conversation_id,
        }
    }

    pub async fn execute(&self, params: serde_json::Value) -> anyhow::Result<ToolExecuteResult> {
        let summary_ids = params
            .get("summaryIds")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(ToString::to_string))
                    .collect::<Vec<String>>()
            });
        let query = params.get("query").and_then(|v| v.as_str()).map(str::trim);
        let max_depth = params.get("maxDepth").and_then(|v| v.as_i64());
        let token_cap = resolve_expansion_token_cap(
            params.get("tokenCap").and_then(|v| v.as_i64()),
            self.max_expand_tokens,
        );
        let include_messages = params
            .get("includeMessages")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let result = if let Some(query) = query.filter(|q| !q.is_empty()) {
            self.orchestrator
                .describe_and_expand(
                    query,
                    "full_text",
                    Some(self.conversation_id),
                    max_depth,
                    Some(token_cap),
                )
                .await?
        } else if let Some(summary_ids) = summary_ids.filter(|v| !v.is_empty()) {
            self.orchestrator
                .expand(ExpansionRequest {
                    summary_ids,
                    max_depth,
                    token_cap: Some(token_cap),
                    include_messages: Some(include_messages),
                    conversation_id: self.conversation_id,
                })
                .await?
        } else {
            return Ok(ToolExecuteResult {
                text: "Error: either summaryIds or query must be provided.".to_string(),
                details: serde_json::json!({ "error": "Error: either summaryIds or query must be provided." }),
            });
        };

        Ok(ToolExecuteResult {
            text: distill_for_subagent(&result),
            details: serde_json::json!({
                "expansionCount": result.expansions.len(),
                "citedIds": result.cited_ids,
                "totalTokens": result.total_tokens,
                "truncated": result.truncated,
            }),
        })
    }
}
