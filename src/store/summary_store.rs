use std::sync::Arc;

use anyhow::Context;
use chrono::{DateTime, NaiveDateTime, Utc};
use parking_lot::Mutex;
use regex::Regex;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::db::connection::SharedConnection;

use super::fts5_sanitize::sanitize_fts5_query;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SummaryKind {
    #[serde(rename = "leaf")]
    Leaf,
    #[serde(rename = "condensed")]
    Condensed,
}

impl SummaryKind {
    fn as_str(&self) -> &'static str {
        match self {
            SummaryKind::Leaf => "leaf",
            SummaryKind::Condensed => "condensed",
        }
    }

    fn from_db(value: &str) -> Self {
        if value == "condensed" {
            Self::Condensed
        } else {
            Self::Leaf
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextItemType {
    #[serde(rename = "message")]
    Message,
    #[serde(rename = "summary")]
    Summary,
}

impl ContextItemType {
    fn from_db(value: &str) -> Self {
        if value == "summary" {
            Self::Summary
        } else {
            Self::Message
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CreateSummaryInput {
    pub summary_id: String,
    pub conversation_id: i64,
    pub kind: SummaryKind,
    pub depth: Option<i64>,
    pub content: String,
    pub token_count: i64,
    pub file_ids: Option<Vec<String>>,
    pub earliest_at: Option<DateTime<Utc>>,
    pub latest_at: Option<DateTime<Utc>>,
    pub descendant_count: Option<i64>,
    pub descendant_token_count: Option<i64>,
    pub source_message_token_count: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SummaryRecord {
    pub summary_id: String,
    pub conversation_id: i64,
    pub kind: SummaryKind,
    pub depth: i64,
    pub content: String,
    pub token_count: i64,
    pub file_ids: Vec<String>,
    pub earliest_at: Option<DateTime<Utc>>,
    pub latest_at: Option<DateTime<Utc>>,
    pub descendant_count: i64,
    pub descendant_token_count: i64,
    pub source_message_token_count: i64,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SummarySubtreeNodeRecord {
    pub summary: SummaryRecord,
    pub depth_from_root: i64,
    pub parent_summary_id: Option<String>,
    pub path: String,
    pub child_count: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContextItemRecord {
    pub conversation_id: i64,
    pub ordinal: i64,
    pub item_type: ContextItemType,
    pub message_id: Option<i64>,
    pub summary_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SummarySearchInput {
    pub conversation_id: Option<i64>,
    pub query: String,
    pub mode: String,
    pub since: Option<DateTime<Utc>>,
    pub before: Option<DateTime<Utc>>,
    pub limit: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SummarySearchResult {
    pub summary_id: String,
    pub conversation_id: i64,
    pub kind: SummaryKind,
    pub snippet: String,
    pub created_at: DateTime<Utc>,
    pub rank: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CreateLargeFileInput {
    pub file_id: String,
    pub conversation_id: i64,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub byte_size: Option<i64>,
    pub storage_uri: String,
    pub exploration_summary: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LargeFileRecord {
    pub file_id: String,
    pub conversation_id: i64,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub byte_size: Option<i64>,
    pub storage_uri: String,
    pub exploration_summary: Option<String>,
    pub created_at: DateTime<Utc>,
}

fn parse_datetime(value: &str) -> DateTime<Utc> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        return dt.with_timezone(&Utc);
    }
    if let Ok(naive) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S") {
        return DateTime::from_naive_utc_and_offset(naive, Utc);
    }
    Utc::now()
}

fn decode_file_ids(value: &str) -> Vec<String> {
    serde_json::from_str::<Vec<String>>(value).unwrap_or_default()
}

fn snippet_from_content(content: &str, max_len: usize) -> String {
    let compact = content.replace('\n', " ").trim().to_string();
    if compact.len() <= max_len {
        compact
    } else {
        let head_limit = max_len.saturating_sub(3);
        let mut cut = 0usize;
        for (idx, _) in compact.char_indices() {
            if idx > head_limit {
                break;
            }
            cut = idx;
        }
        format!("{}...", &compact[..cut])
    }
}

#[derive(Clone)]
pub struct SummaryStore {
    conn: Arc<Mutex<Connection>>,
}

impl SummaryStore {
    pub fn new(shared: &SharedConnection) -> Self {
        Self {
            conn: shared.conn.clone(),
        }
    }

    fn with_conn<T, F>(&self, f: F) -> anyhow::Result<T>
    where
        F: FnOnce(&Connection) -> anyhow::Result<T>,
    {
        let guard = self.conn.lock();
        f(&guard)
    }

    pub fn insert_summary(&self, input: CreateSummaryInput) -> anyhow::Result<SummaryRecord> {
        self.with_conn(|conn| {
            let depth = input.depth.unwrap_or(if matches!(input.kind, SummaryKind::Leaf) {
                0
            } else {
                1
            });
            conn.execute(
                "INSERT INTO summaries (
                    summary_id, conversation_id, kind, depth, content, token_count, file_ids,
                    earliest_at, latest_at, descendant_count, descendant_token_count, source_message_token_count
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    input.summary_id,
                    input.conversation_id,
                    input.kind.as_str(),
                    depth,
                    input.content,
                    input.token_count,
                    serde_json::to_string(&input.file_ids.unwrap_or_default())?,
                    input.earliest_at.map(|v| v.to_rfc3339()),
                    input.latest_at.map(|v| v.to_rfc3339()),
                    input.descendant_count.unwrap_or(0).max(0),
                    input.descendant_token_count.unwrap_or(0).max(0),
                    input.source_message_token_count.unwrap_or(0).max(0),
                ],
            )?;
            let _ = conn.execute(
                "INSERT INTO summaries_fts(summary_id, content) VALUES (?, ?)",
                params![input.summary_id, input.content],
            );

            conn.query_row(
                "SELECT summary_id, conversation_id, kind, depth, content, token_count, file_ids,
                        earliest_at, latest_at, descendant_count, descendant_token_count, source_message_token_count, created_at
                 FROM summaries WHERE summary_id = ?",
                params![input.summary_id],
                |row| {
                    Ok(SummaryRecord {
                        summary_id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        kind: SummaryKind::from_db(&row.get::<_, String>(2)?),
                        depth: row.get(3)?,
                        content: row.get(4)?,
                        token_count: row.get(5)?,
                        file_ids: decode_file_ids(&row.get::<_, String>(6)?),
                        earliest_at: row
                            .get::<_, Option<String>>(7)?
                            .as_deref()
                            .map(parse_datetime),
                        latest_at: row
                            .get::<_, Option<String>>(8)?
                            .as_deref()
                            .map(parse_datetime),
                        descendant_count: row.get::<_, Option<i64>>(9)?.unwrap_or(0).max(0),
                        descendant_token_count: row.get::<_, Option<i64>>(10)?.unwrap_or(0).max(0),
                        source_message_token_count: row.get::<_, Option<i64>>(11)?.unwrap_or(0).max(0),
                        created_at: parse_datetime(&row.get::<_, String>(12)?),
                    })
                },
            )
            .optional()?
            .context("summary insert failed to re-read row")
        })
    }

    pub fn get_summary(&self, summary_id: &str) -> anyhow::Result<Option<SummaryRecord>> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT summary_id, conversation_id, kind, depth, content, token_count, file_ids,
                        earliest_at, latest_at, descendant_count, descendant_token_count, source_message_token_count, created_at
                 FROM summaries WHERE summary_id = ?",
                params![summary_id],
                |row| {
                    Ok(SummaryRecord {
                        summary_id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        kind: SummaryKind::from_db(&row.get::<_, String>(2)?),
                        depth: row.get(3)?,
                        content: row.get(4)?,
                        token_count: row.get(5)?,
                        file_ids: decode_file_ids(&row.get::<_, String>(6)?),
                        earliest_at: row
                            .get::<_, Option<String>>(7)?
                            .as_deref()
                            .map(parse_datetime),
                        latest_at: row
                            .get::<_, Option<String>>(8)?
                            .as_deref()
                            .map(parse_datetime),
                        descendant_count: row.get::<_, Option<i64>>(9)?.unwrap_or(0).max(0),
                        descendant_token_count: row.get::<_, Option<i64>>(10)?.unwrap_or(0).max(0),
                        source_message_token_count: row.get::<_, Option<i64>>(11)?.unwrap_or(0).max(0),
                        created_at: parse_datetime(&row.get::<_, String>(12)?),
                    })
                },
            )
            .optional()
            .map_err(Into::into)
        })
    }

    pub fn get_summaries_by_conversation(
        &self,
        conversation_id: i64,
    ) -> anyhow::Result<Vec<SummaryRecord>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT summary_id, conversation_id, kind, depth, content, token_count, file_ids,
                        earliest_at, latest_at, descendant_count, descendant_token_count, source_message_token_count, created_at
                 FROM summaries
                 WHERE conversation_id = ?
                 ORDER BY created_at",
            )?;
            let mut rows = stmt.query(params![conversation_id])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(SummaryRecord {
                    summary_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    kind: SummaryKind::from_db(&row.get::<_, String>(2)?),
                    depth: row.get(3)?,
                    content: row.get(4)?,
                    token_count: row.get(5)?,
                    file_ids: decode_file_ids(&row.get::<_, String>(6)?),
                    earliest_at: row
                        .get::<_, Option<String>>(7)?
                        .as_deref()
                        .map(parse_datetime),
                    latest_at: row
                        .get::<_, Option<String>>(8)?
                        .as_deref()
                        .map(parse_datetime),
                    descendant_count: row.get::<_, Option<i64>>(9)?.unwrap_or(0).max(0),
                    descendant_token_count: row.get::<_, Option<i64>>(10)?.unwrap_or(0).max(0),
                    source_message_token_count: row.get::<_, Option<i64>>(11)?.unwrap_or(0).max(0),
                    created_at: parse_datetime(&row.get::<_, String>(12)?),
                });
            }
            Ok(out)
        })
    }

    pub fn link_summary_to_messages(
        &self,
        summary_id: &str,
        message_ids: &[i64],
    ) -> anyhow::Result<()> {
        if message_ids.is_empty() {
            return Ok(());
        }
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "INSERT INTO summary_messages (summary_id, message_id, ordinal)
                 VALUES (?, ?, ?)
                 ON CONFLICT (summary_id, message_id) DO NOTHING",
            )?;
            for (idx, message_id) in message_ids.iter().enumerate() {
                stmt.execute(params![summary_id, message_id, idx as i64])?;
            }
            Ok(())
        })
    }

    pub fn link_summary_to_parents(
        &self,
        summary_id: &str,
        parent_summary_ids: &[String],
    ) -> anyhow::Result<()> {
        if parent_summary_ids.is_empty() {
            return Ok(());
        }
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "INSERT INTO summary_parents (summary_id, parent_summary_id, ordinal)
                 VALUES (?, ?, ?)
                 ON CONFLICT (summary_id, parent_summary_id) DO NOTHING",
            )?;
            for (idx, parent_summary_id) in parent_summary_ids.iter().enumerate() {
                stmt.execute(params![summary_id, parent_summary_id, idx as i64])?;
            }
            Ok(())
        })
    }

    pub fn get_summary_messages(&self, summary_id: &str) -> anyhow::Result<Vec<i64>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT message_id FROM summary_messages WHERE summary_id = ? ORDER BY ordinal",
            )?;
            let mut rows = stmt.query(params![summary_id])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(row.get(0)?);
            }
            Ok(out)
        })
    }

    pub fn get_summary_children(
        &self,
        parent_summary_id: &str,
    ) -> anyhow::Result<Vec<SummaryRecord>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT s.summary_id, s.conversation_id, s.kind, s.depth, s.content, s.token_count,
                        s.file_ids, s.earliest_at, s.latest_at, s.descendant_count, s.descendant_token_count,
                        s.source_message_token_count, s.created_at
                 FROM summaries s
                 JOIN summary_parents sp ON sp.summary_id = s.summary_id
                 WHERE sp.parent_summary_id = ?
                 ORDER BY sp.ordinal",
            )?;
            let mut rows = stmt.query(params![parent_summary_id])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(SummaryRecord {
                    summary_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    kind: SummaryKind::from_db(&row.get::<_, String>(2)?),
                    depth: row.get(3)?,
                    content: row.get(4)?,
                    token_count: row.get(5)?,
                    file_ids: decode_file_ids(&row.get::<_, String>(6)?),
                    earliest_at: row.get::<_, Option<String>>(7)?.as_deref().map(parse_datetime),
                    latest_at: row.get::<_, Option<String>>(8)?.as_deref().map(parse_datetime),
                    descendant_count: row.get::<_, Option<i64>>(9)?.unwrap_or(0).max(0),
                    descendant_token_count: row.get::<_, Option<i64>>(10)?.unwrap_or(0).max(0),
                    source_message_token_count: row.get::<_, Option<i64>>(11)?.unwrap_or(0).max(0),
                    created_at: parse_datetime(&row.get::<_, String>(12)?),
                });
            }
            Ok(out)
        })
    }

    pub fn get_summary_parents(&self, summary_id: &str) -> anyhow::Result<Vec<SummaryRecord>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT s.summary_id, s.conversation_id, s.kind, s.depth, s.content, s.token_count,
                        s.file_ids, s.earliest_at, s.latest_at, s.descendant_count, s.descendant_token_count,
                        s.source_message_token_count, s.created_at
                 FROM summaries s
                 JOIN summary_parents sp ON sp.parent_summary_id = s.summary_id
                 WHERE sp.summary_id = ?
                 ORDER BY sp.ordinal",
            )?;
            let mut rows = stmt.query(params![summary_id])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(SummaryRecord {
                    summary_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    kind: SummaryKind::from_db(&row.get::<_, String>(2)?),
                    depth: row.get(3)?,
                    content: row.get(4)?,
                    token_count: row.get(5)?,
                    file_ids: decode_file_ids(&row.get::<_, String>(6)?),
                    earliest_at: row.get::<_, Option<String>>(7)?.as_deref().map(parse_datetime),
                    latest_at: row.get::<_, Option<String>>(8)?.as_deref().map(parse_datetime),
                    descendant_count: row.get::<_, Option<i64>>(9)?.unwrap_or(0).max(0),
                    descendant_token_count: row.get::<_, Option<i64>>(10)?.unwrap_or(0).max(0),
                    source_message_token_count: row.get::<_, Option<i64>>(11)?.unwrap_or(0).max(0),
                    created_at: parse_datetime(&row.get::<_, String>(12)?),
                });
            }
            Ok(out)
        })
    }

    pub fn get_summary_subtree(
        &self,
        summary_id: &str,
    ) -> anyhow::Result<Vec<SummarySubtreeNodeRecord>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "WITH RECURSIVE subtree(summary_id, parent_summary_id, depth_from_root, path) AS (
                   SELECT ?, NULL, 0, ''
                   UNION ALL
                   SELECT
                     sp.summary_id,
                     sp.parent_summary_id,
                     subtree.depth_from_root + 1,
                     CASE
                       WHEN subtree.path = '' THEN printf('%04d', sp.ordinal)
                       ELSE subtree.path || '.' || printf('%04d', sp.ordinal)
                     END
                   FROM summary_parents sp
                   JOIN subtree ON sp.parent_summary_id = subtree.summary_id
                 )
                 SELECT
                   s.summary_id,
                   s.conversation_id,
                   s.kind,
                   s.depth,
                   s.content,
                   s.token_count,
                   s.file_ids,
                   s.earliest_at,
                   s.latest_at,
                   s.descendant_count,
                   s.descendant_token_count,
                   s.source_message_token_count,
                   s.created_at,
                   subtree.depth_from_root,
                   subtree.parent_summary_id,
                   subtree.path,
                   (
                     SELECT COUNT(*) FROM summary_parents sp2
                     WHERE sp2.parent_summary_id = s.summary_id
                   ) AS child_count
                 FROM subtree
                 JOIN summaries s ON s.summary_id = subtree.summary_id
                 ORDER BY subtree.depth_from_root ASC, subtree.path ASC, s.created_at ASC",
            )?;
            let mut rows = stmt.query(params![summary_id])?;
            let mut out: Vec<SummarySubtreeNodeRecord> = vec![];
            let mut seen = std::collections::HashSet::new();
            while let Some(row) = rows.next()? {
                let id: String = row.get(0)?;
                if seen.contains(&id) {
                    continue;
                }
                seen.insert(id.clone());
                let summary = SummaryRecord {
                    summary_id: id,
                    conversation_id: row.get(1)?,
                    kind: SummaryKind::from_db(&row.get::<_, String>(2)?),
                    depth: row.get(3)?,
                    content: row.get(4)?,
                    token_count: row.get(5)?,
                    file_ids: decode_file_ids(&row.get::<_, String>(6)?),
                    earliest_at: row
                        .get::<_, Option<String>>(7)?
                        .as_deref()
                        .map(parse_datetime),
                    latest_at: row
                        .get::<_, Option<String>>(8)?
                        .as_deref()
                        .map(parse_datetime),
                    descendant_count: row.get::<_, Option<i64>>(9)?.unwrap_or(0).max(0),
                    descendant_token_count: row.get::<_, Option<i64>>(10)?.unwrap_or(0).max(0),
                    source_message_token_count: row.get::<_, Option<i64>>(11)?.unwrap_or(0).max(0),
                    created_at: parse_datetime(&row.get::<_, String>(12)?),
                };
                out.push(SummarySubtreeNodeRecord {
                    summary,
                    depth_from_root: row.get::<_, i64>(13)?.max(0),
                    parent_summary_id: row.get(14)?,
                    path: row.get::<_, String>(15).unwrap_or_default(),
                    child_count: row.get::<_, Option<i64>>(16)?.unwrap_or(0).max(0),
                });
            }
            Ok(out)
        })
    }

    pub fn get_context_items(
        &self,
        conversation_id: i64,
    ) -> anyhow::Result<Vec<ContextItemRecord>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT conversation_id, ordinal, item_type, message_id, summary_id, created_at
                 FROM context_items
                 WHERE conversation_id = ?
                 ORDER BY ordinal",
            )?;
            let mut rows = stmt.query(params![conversation_id])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(ContextItemRecord {
                    conversation_id: row.get(0)?,
                    ordinal: row.get(1)?,
                    item_type: ContextItemType::from_db(&row.get::<_, String>(2)?),
                    message_id: row.get(3)?,
                    summary_id: row.get(4)?,
                    created_at: parse_datetime(&row.get::<_, String>(5)?),
                });
            }
            Ok(out)
        })
    }

    pub fn get_distinct_depths_in_context(
        &self,
        conversation_id: i64,
        max_ordinal_exclusive: Option<i64>,
    ) -> anyhow::Result<Vec<i64>> {
        self.with_conn(|conn| {
            let mut out = Vec::new();
            if let Some(bound) = max_ordinal_exclusive.filter(|v| *v != i64::MAX) {
                let mut stmt = conn.prepare(
                    "SELECT DISTINCT s.depth
                     FROM context_items ci
                     JOIN summaries s ON s.summary_id = ci.summary_id
                     WHERE ci.conversation_id = ?
                       AND ci.item_type = 'summary'
                       AND ci.ordinal < ?
                     ORDER BY s.depth ASC",
                )?;
                let mut rows = stmt.query(params![conversation_id, bound])?;
                while let Some(row) = rows.next()? {
                    out.push(row.get::<_, i64>(0)?);
                }
                return Ok(out);
            }
            let mut stmt = conn.prepare(
                "SELECT DISTINCT s.depth
                 FROM context_items ci
                 JOIN summaries s ON s.summary_id = ci.summary_id
                 WHERE ci.conversation_id = ?
                   AND ci.item_type = 'summary'
                 ORDER BY s.depth ASC",
            )?;
            let mut rows = stmt.query(params![conversation_id])?;
            while let Some(row) = rows.next()? {
                out.push(row.get::<_, i64>(0)?);
            }
            Ok(out)
        })
    }

    pub fn append_context_message(
        &self,
        conversation_id: i64,
        message_id: i64,
    ) -> anyhow::Result<()> {
        self.with_conn(|conn| {
            let max_ordinal: i64 = conn.query_row(
                "SELECT COALESCE(MAX(ordinal), -1) FROM context_items WHERE conversation_id = ?",
                params![conversation_id],
                |row| row.get(0),
            )?;
            conn.execute(
                "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id)
                 VALUES (?, ?, 'message', ?)",
                params![conversation_id, max_ordinal + 1, message_id],
            )?;
            Ok(())
        })
    }

    pub fn append_context_messages(
        &self,
        conversation_id: i64,
        message_ids: &[i64],
    ) -> anyhow::Result<()> {
        if message_ids.is_empty() {
            return Ok(());
        }
        self.with_conn(|conn| {
            let max_ordinal: i64 = conn.query_row(
                "SELECT COALESCE(MAX(ordinal), -1) FROM context_items WHERE conversation_id = ?",
                params![conversation_id],
                |row| row.get(0),
            )?;
            let mut stmt = conn.prepare(
                "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id)
                 VALUES (?, ?, 'message', ?)",
            )?;
            for (idx, message_id) in message_ids.iter().enumerate() {
                stmt.execute(params![
                    conversation_id,
                    max_ordinal + 1 + idx as i64,
                    message_id
                ])?;
            }
            Ok(())
        })
    }

    pub fn append_context_summary(
        &self,
        conversation_id: i64,
        summary_id: &str,
    ) -> anyhow::Result<()> {
        self.with_conn(|conn| {
            let max_ordinal: i64 = conn.query_row(
                "SELECT COALESCE(MAX(ordinal), -1) FROM context_items WHERE conversation_id = ?",
                params![conversation_id],
                |row| row.get(0),
            )?;
            conn.execute(
                "INSERT INTO context_items (conversation_id, ordinal, item_type, summary_id)
                 VALUES (?, ?, 'summary', ?)",
                params![conversation_id, max_ordinal + 1, summary_id],
            )?;
            Ok(())
        })
    }

    pub fn replace_context_range_with_summary(
        &self,
        conversation_id: i64,
        start_ordinal: i64,
        end_ordinal: i64,
        summary_id: &str,
    ) -> anyhow::Result<()> {
        self.with_conn(|conn| {
            conn.execute_batch("BEGIN")?;
            let result = (|| -> anyhow::Result<()> {
                conn.execute(
                    "DELETE FROM context_items
                     WHERE conversation_id = ?
                       AND ordinal >= ?
                       AND ordinal <= ?",
                    params![conversation_id, start_ordinal, end_ordinal],
                )?;
                conn.execute(
                    "INSERT INTO context_items (conversation_id, ordinal, item_type, summary_id)
                     VALUES (?, ?, 'summary', ?)",
                    params![conversation_id, start_ordinal, summary_id],
                )?;

                let mut stmt = conn.prepare(
                    "SELECT ordinal
                     FROM context_items
                     WHERE conversation_id = ?
                     ORDER BY ordinal",
                )?;
                let mut rows = stmt.query(params![conversation_id])?;
                let mut ordinals = Vec::new();
                while let Some(row) = rows.next()? {
                    ordinals.push(row.get::<_, i64>(0)?);
                }
                let mut update = conn.prepare(
                    "UPDATE context_items
                     SET ordinal = ?
                     WHERE conversation_id = ? AND ordinal = ?",
                )?;
                for (idx, old) in ordinals.iter().enumerate() {
                    update.execute(params![-((idx as i64) + 1), conversation_id, old])?;
                }
                for idx in 0..ordinals.len() {
                    update.execute(params![idx as i64, conversation_id, -((idx as i64) + 1)])?;
                }
                Ok(())
            })();
            match result {
                Ok(()) => {
                    conn.execute_batch("COMMIT")?;
                    Ok(())
                }
                Err(err) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    Err(err)
                }
            }
        })
    }

    pub fn get_context_token_count(&self, conversation_id: i64) -> anyhow::Result<i64> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT COALESCE(SUM(token_count), 0) AS total
                 FROM (
                   SELECT m.token_count
                   FROM context_items ci
                   JOIN messages m ON m.message_id = ci.message_id
                   WHERE ci.conversation_id = ?
                     AND ci.item_type = 'message'

                   UNION ALL

                   SELECT s.token_count
                   FROM context_items ci
                   JOIN summaries s ON s.summary_id = ci.summary_id
                   WHERE ci.conversation_id = ?
                     AND ci.item_type = 'summary'
                 ) sub",
                params![conversation_id, conversation_id],
                |row| row.get::<_, i64>(0),
            )
            .map_err(Into::into)
        })
    }

    pub fn search_summaries(
        &self,
        input: SummarySearchInput,
    ) -> anyhow::Result<Vec<SummarySearchResult>> {
        let limit = input.limit.unwrap_or(50).clamp(1, 200);
        if input.mode == "full_text" {
            return self.search_full_text(
                &input.query,
                limit,
                input.conversation_id,
                input.since,
                input.before,
            );
        }
        self.search_regex(
            &input.query,
            limit,
            input.conversation_id,
            input.since,
            input.before,
        )
    }

    fn search_full_text(
        &self,
        query: &str,
        limit: i64,
        conversation_id: Option<i64>,
        since: Option<DateTime<Utc>>,
        before: Option<DateTime<Utc>>,
    ) -> anyhow::Result<Vec<SummarySearchResult>> {
        self.with_conn(|conn| {
            let mut sql = String::from(
                "SELECT
                    summaries_fts.summary_id,
                    s.conversation_id,
                    s.kind,
                    s.content,
                    rank,
                    s.created_at
                 FROM summaries_fts
                 JOIN summaries s ON s.summary_id = summaries_fts.summary_id
                 WHERE summaries_fts MATCH ?",
            );
            let mut args: Vec<rusqlite::types::Value> =
                vec![rusqlite::types::Value::from(sanitize_fts5_query(query))];
            if let Some(conversation_id) = conversation_id {
                sql.push_str(" AND s.conversation_id = ?");
                args.push(rusqlite::types::Value::from(conversation_id));
            }
            if let Some(since) = since {
                sql.push_str(" AND julianday(s.created_at) >= julianday(?)");
                args.push(rusqlite::types::Value::from(since.to_rfc3339()));
            }
            if let Some(before) = before {
                sql.push_str(" AND julianday(s.created_at) < julianday(?)");
                args.push(rusqlite::types::Value::from(before.to_rfc3339()));
            }
            sql.push_str(" ORDER BY s.created_at DESC LIMIT ?");
            args.push(rusqlite::types::Value::from(limit));

            let mut stmt = conn.prepare(&sql)?;
            let mut rows = stmt.query(rusqlite::params_from_iter(args))?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                let content: String = row.get(3)?;
                out.push(SummarySearchResult {
                    summary_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    kind: SummaryKind::from_db(&row.get::<_, String>(2)?),
                    snippet: snippet_from_content(&content, 200),
                    rank: row.get::<_, Option<f64>>(4)?,
                    created_at: parse_datetime(&row.get::<_, String>(5)?),
                });
            }
            Ok(out)
        })
    }

    fn search_regex(
        &self,
        pattern: &str,
        limit: i64,
        conversation_id: Option<i64>,
        since: Option<DateTime<Utc>>,
        before: Option<DateTime<Utc>>,
    ) -> anyhow::Result<Vec<SummarySearchResult>> {
        let re = Regex::new(pattern)?;
        self.with_conn(|conn| {
            let mut sql = String::from(
                "SELECT summary_id, conversation_id, kind, depth, content, token_count, file_ids,
                        earliest_at, latest_at, descendant_count, descendant_token_count,
                        source_message_token_count, created_at
                 FROM summaries",
            );
            let mut where_clauses: Vec<&str> = vec![];
            let mut args: Vec<rusqlite::types::Value> = vec![];
            if let Some(conversation_id) = conversation_id {
                where_clauses.push("conversation_id = ?");
                args.push(rusqlite::types::Value::from(conversation_id));
            }
            if let Some(since) = since {
                where_clauses.push("julianday(created_at) >= julianday(?)");
                args.push(rusqlite::types::Value::from(since.to_rfc3339()));
            }
            if let Some(before) = before {
                where_clauses.push("julianday(created_at) < julianday(?)");
                args.push(rusqlite::types::Value::from(before.to_rfc3339()));
            }
            if !where_clauses.is_empty() {
                sql.push_str(" WHERE ");
                sql.push_str(&where_clauses.join(" AND "));
            }
            sql.push_str(" ORDER BY created_at DESC");

            let mut stmt = conn.prepare(&sql)?;
            let mut rows = stmt.query(rusqlite::params_from_iter(args))?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                if out.len() as i64 >= limit {
                    break;
                }
                let content: String = row.get(4)?;
                if let Some(mtch) = re.find(&content) {
                    out.push(SummarySearchResult {
                        summary_id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        kind: SummaryKind::from_db(&row.get::<_, String>(2)?),
                        snippet: mtch.as_str().to_string(),
                        rank: Some(0.0),
                        created_at: parse_datetime(&row.get::<_, String>(12)?),
                    });
                }
            }
            Ok(out)
        })
    }

    pub fn insert_large_file(
        &self,
        input: CreateLargeFileInput,
    ) -> anyhow::Result<LargeFileRecord> {
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO large_files (file_id, conversation_id, file_name, mime_type, byte_size, storage_uri, exploration_summary)
                 VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![
                    input.file_id,
                    input.conversation_id,
                    input.file_name,
                    input.mime_type,
                    input.byte_size,
                    input.storage_uri,
                    input.exploration_summary
                ],
            )?;
            conn.query_row(
                "SELECT file_id, conversation_id, file_name, mime_type, byte_size, storage_uri, exploration_summary, created_at
                 FROM large_files WHERE file_id = ?",
                params![input.file_id],
                |row| {
                    Ok(LargeFileRecord {
                        file_id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        file_name: row.get(2)?,
                        mime_type: row.get(3)?,
                        byte_size: row.get(4)?,
                        storage_uri: row.get(5)?,
                        exploration_summary: row.get(6)?,
                        created_at: parse_datetime(&row.get::<_, String>(7)?),
                    })
                },
            )
            .optional()?
            .context("large file insert failed to re-read row")
        })
    }

    pub fn get_large_file(&self, file_id: &str) -> anyhow::Result<Option<LargeFileRecord>> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT file_id, conversation_id, file_name, mime_type, byte_size, storage_uri, exploration_summary, created_at
                 FROM large_files WHERE file_id = ?",
                params![file_id],
                |row| {
                    Ok(LargeFileRecord {
                        file_id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        file_name: row.get(2)?,
                        mime_type: row.get(3)?,
                        byte_size: row.get(4)?,
                        storage_uri: row.get(5)?,
                        exploration_summary: row.get(6)?,
                        created_at: parse_datetime(&row.get::<_, String>(7)?),
                    })
                },
            )
            .optional()
            .map_err(Into::into)
        })
    }

    pub fn get_large_files_by_conversation(
        &self,
        conversation_id: i64,
    ) -> anyhow::Result<Vec<LargeFileRecord>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT file_id, conversation_id, file_name, mime_type, byte_size, storage_uri, exploration_summary, created_at
                 FROM large_files
                 WHERE conversation_id = ?
                 ORDER BY created_at",
            )?;
            let mut rows = stmt.query(params![conversation_id])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(LargeFileRecord {
                    file_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    file_name: row.get(2)?,
                    mime_type: row.get(3)?,
                    byte_size: row.get(4)?,
                    storage_uri: row.get(5)?,
                    exploration_summary: row.get(6)?,
                    created_at: parse_datetime(&row.get::<_, String>(7)?),
                });
            }
            Ok(out)
        })
    }
}
