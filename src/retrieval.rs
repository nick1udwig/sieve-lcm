use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::store::conversation_store::{ConversationStore, MessageSearchInput, MessageSearchResult};
use crate::store::summary_store::{SummaryKind, SummarySearchInput, SummarySearchResult, SummaryStore};

#[derive(Clone, Debug, PartialEq)]
pub struct DescribeSummary {
    pub conversation_id: i64,
    pub kind: SummaryKind,
    pub content: String,
    pub depth: i64,
    pub token_count: i64,
    pub descendant_count: i64,
    pub descendant_token_count: i64,
    pub source_message_token_count: i64,
    pub file_ids: Vec<String>,
    pub parent_ids: Vec<String>,
    pub child_ids: Vec<String>,
    pub message_ids: Vec<i64>,
    pub earliest_at: Option<DateTime<Utc>>,
    pub latest_at: Option<DateTime<Utc>>,
    pub subtree: Vec<DescribeSubtreeNode>,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DescribeSubtreeNode {
    pub summary_id: String,
    pub parent_summary_id: Option<String>,
    pub depth_from_root: i64,
    pub kind: SummaryKind,
    pub depth: i64,
    pub token_count: i64,
    pub descendant_count: i64,
    pub descendant_token_count: i64,
    pub source_message_token_count: i64,
    pub earliest_at: Option<DateTime<Utc>>,
    pub latest_at: Option<DateTime<Utc>>,
    pub child_count: i64,
    pub path: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DescribeFile {
    pub conversation_id: i64,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub byte_size: Option<i64>,
    pub storage_uri: String,
    pub exploration_summary: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DescribeResultType {
    Summary(DescribeSummary),
    File(DescribeFile),
}

#[derive(Clone, Debug, PartialEq)]
pub struct DescribeResult {
    pub id: String,
    pub result: DescribeResultType,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrepInput {
    pub query: String,
    pub mode: String,
    pub scope: String,
    pub conversation_id: Option<i64>,
    pub since: Option<DateTime<Utc>>,
    pub before: Option<DateTime<Utc>>,
    pub limit: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrepResult {
    pub messages: Vec<MessageSearchResult>,
    pub summaries: Vec<SummarySearchResult>,
    pub total_matches: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpandInput {
    pub summary_id: String,
    pub depth: Option<i64>,
    pub include_messages: Option<bool>,
    pub token_cap: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpandedChild {
    pub summary_id: String,
    pub kind: SummaryKind,
    pub content: String,
    pub token_count: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpandedMessage {
    pub message_id: i64,
    pub role: String,
    pub content: String,
    pub token_count: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpandResult {
    pub children: Vec<ExpandedChild>,
    pub messages: Vec<ExpandedMessage>,
    pub estimated_tokens: i64,
    pub truncated: bool,
}

#[async_trait]
pub trait RetrievalApi: Send + Sync {
    async fn describe(&self, id: &str) -> anyhow::Result<Option<DescribeResult>>;
    async fn grep(&self, input: GrepInput) -> anyhow::Result<GrepResult>;
    async fn expand(&self, input: ExpandInput) -> anyhow::Result<ExpandResult>;
}

fn estimate_tokens(content: &str) -> i64 {
    ((content.len() as f64) / 4.0).ceil() as i64
}

pub struct RetrievalEngine {
    conversation_store: ConversationStore,
    summary_store: SummaryStore,
}

impl RetrievalEngine {
    pub fn new(conversation_store: ConversationStore, summary_store: SummaryStore) -> Self {
        Self {
            conversation_store,
            summary_store,
        }
    }

    async fn describe_summary(&self, id: &str) -> anyhow::Result<Option<DescribeResult>> {
        let Some(summary) = self.summary_store.get_summary(id)? else {
            return Ok(None);
        };
        let parents = self.summary_store.get_summary_parents(id)?;
        let children = self.summary_store.get_summary_children(id)?;
        let message_ids = self.summary_store.get_summary_messages(id)?;
        let subtree = self.summary_store.get_summary_subtree(id)?;

        Ok(Some(DescribeResult {
            id: id.to_string(),
            result: DescribeResultType::Summary(DescribeSummary {
                conversation_id: summary.conversation_id,
                kind: summary.kind,
                content: summary.content,
                depth: summary.depth,
                token_count: summary.token_count,
                descendant_count: summary.descendant_count,
                descendant_token_count: summary.descendant_token_count,
                source_message_token_count: summary.source_message_token_count,
                file_ids: summary.file_ids,
                parent_ids: parents.iter().map(|p| p.summary_id.clone()).collect(),
                child_ids: children.iter().map(|c| c.summary_id.clone()).collect(),
                message_ids,
                earliest_at: summary.earliest_at,
                latest_at: summary.latest_at,
                subtree: subtree
                    .into_iter()
                    .map(|node| DescribeSubtreeNode {
                        summary_id: node.summary.summary_id,
                        parent_summary_id: node.parent_summary_id,
                        depth_from_root: node.depth_from_root,
                        kind: node.summary.kind,
                        depth: node.summary.depth,
                        token_count: node.summary.token_count,
                        descendant_count: node.summary.descendant_count,
                        descendant_token_count: node.summary.descendant_token_count,
                        source_message_token_count: node.summary.source_message_token_count,
                        earliest_at: node.summary.earliest_at,
                        latest_at: node.summary.latest_at,
                        child_count: node.child_count,
                        path: node.path,
                    })
                    .collect(),
                created_at: summary.created_at,
            }),
        }))
    }

    async fn describe_file(&self, id: &str) -> anyhow::Result<Option<DescribeResult>> {
        let Some(file) = self.summary_store.get_large_file(id)? else {
            return Ok(None);
        };
        Ok(Some(DescribeResult {
            id: id.to_string(),
            result: DescribeResultType::File(DescribeFile {
                conversation_id: file.conversation_id,
                file_name: file.file_name,
                mime_type: file.mime_type,
                byte_size: file.byte_size,
                storage_uri: file.storage_uri,
                exploration_summary: file.exploration_summary,
                created_at: file.created_at,
            }),
        }))
    }

    fn expand_recursive(
        &self,
        summary_id: &str,
        depth: i64,
        include_messages: bool,
        token_cap: i64,
        result: &mut ExpandResult,
    ) -> anyhow::Result<()> {
        if depth <= 0 || result.truncated {
            return Ok(());
        }
        let Some(summary) = self.summary_store.get_summary(summary_id)? else {
            return Ok(());
        };

        match summary.kind {
            SummaryKind::Condensed => {
                let children = self.summary_store.get_summary_children(summary_id)?;
                for child in children {
                    if result.truncated {
                        break;
                    }
                    if result.estimated_tokens + child.token_count > token_cap {
                        result.truncated = true;
                        break;
                    }
                    result.children.push(ExpandedChild {
                        summary_id: child.summary_id.clone(),
                        kind: child.kind.clone(),
                        content: child.content.clone(),
                        token_count: child.token_count,
                    });
                    result.estimated_tokens += child.token_count;
                    if depth > 1 {
                        self.expand_recursive(
                            &child.summary_id,
                            depth - 1,
                            include_messages,
                            token_cap,
                            result,
                        )?;
                    }
                }
            }
            SummaryKind::Leaf => {
                if include_messages {
                    let message_ids = self.summary_store.get_summary_messages(summary_id)?;
                    for message_id in message_ids {
                        if result.truncated {
                            break;
                        }
                        let Some(msg) = self.conversation_store.get_message_by_id(message_id)? else {
                            continue;
                        };
                        let token_count = if msg.token_count > 0 {
                            msg.token_count
                        } else {
                            estimate_tokens(&msg.content)
                        };
                        if result.estimated_tokens + token_count > token_cap {
                            result.truncated = true;
                            break;
                        }
                        result.messages.push(ExpandedMessage {
                            message_id: msg.message_id,
                            role: match msg.role {
                                crate::store::conversation_store::MessageRole::System => "system",
                                crate::store::conversation_store::MessageRole::User => "user",
                                crate::store::conversation_store::MessageRole::Assistant => {
                                    "assistant"
                                }
                                crate::store::conversation_store::MessageRole::Tool => "tool",
                            }
                            .to_string(),
                            content: msg.content,
                            token_count,
                        });
                        result.estimated_tokens += token_count;
                    }
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl RetrievalApi for RetrievalEngine {
    async fn describe(&self, id: &str) -> anyhow::Result<Option<DescribeResult>> {
        if id.starts_with("sum_") {
            return self.describe_summary(id).await;
        }
        if id.starts_with("file_") {
            return self.describe_file(id).await;
        }
        Ok(None)
    }

    async fn grep(&self, input: GrepInput) -> anyhow::Result<GrepResult> {
        let mut messages = vec![];
        let mut summaries = vec![];
        if input.scope == "messages" {
            messages = self.conversation_store.search_messages(MessageSearchInput {
                conversation_id: input.conversation_id,
                query: input.query.clone(),
                mode: input.mode.clone(),
                since: input.since,
                before: input.before,
                limit: input.limit,
            })?;
        } else if input.scope == "summaries" {
            summaries = self.summary_store.search_summaries(SummarySearchInput {
                conversation_id: input.conversation_id,
                query: input.query.clone(),
                mode: input.mode.clone(),
                since: input.since,
                before: input.before,
                limit: input.limit,
            })?;
        } else {
            messages = self.conversation_store.search_messages(MessageSearchInput {
                conversation_id: input.conversation_id,
                query: input.query.clone(),
                mode: input.mode.clone(),
                since: input.since,
                before: input.before,
                limit: input.limit,
            })?;
            summaries = self.summary_store.search_summaries(SummarySearchInput {
                conversation_id: input.conversation_id,
                query: input.query.clone(),
                mode: input.mode.clone(),
                since: input.since,
                before: input.before,
                limit: input.limit,
            })?;
        }
        messages.sort_by_key(|m| std::cmp::Reverse(m.created_at));
        summaries.sort_by_key(|s| std::cmp::Reverse(s.created_at));
        Ok(GrepResult {
            total_matches: messages.len() + summaries.len(),
            messages,
            summaries,
        })
    }

    async fn expand(&self, input: ExpandInput) -> anyhow::Result<ExpandResult> {
        let depth = input.depth.unwrap_or(1).max(1);
        let include_messages = input.include_messages.unwrap_or(false);
        let token_cap = input.token_cap.unwrap_or(i64::MAX).max(1);
        let mut result = ExpandResult {
            children: vec![],
            messages: vec![],
            estimated_tokens: 0,
            truncated: false,
        };
        self.expand_recursive(
            &input.summary_id,
            depth,
            include_messages,
            token_cap,
            &mut result,
        )?;
        Ok(result)
    }
}
