use std::sync::Arc;

use anyhow::Context;
use chrono::{DateTime, NaiveDateTime, Utc};
use parking_lot::Mutex;
use regex::Regex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::db::connection::SharedConnection;

use super::fts5_sanitize::sanitize_fts5_query;

pub type ConversationId = i64;
pub type MessageId = i64;
pub type SummaryId = String;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "tool")]
    Tool,
}

impl MessageRole {
    fn as_str(&self) -> &'static str {
        match self {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        }
    }

    fn from_db(value: &str) -> Self {
        match value {
            "system" => Self::System,
            "assistant" => Self::Assistant,
            "tool" => Self::Tool,
            _ => Self::User,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessagePartType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "reasoning")]
    Reasoning,
    #[serde(rename = "tool")]
    Tool,
    #[serde(rename = "patch")]
    Patch,
    #[serde(rename = "file")]
    File,
    #[serde(rename = "subtask")]
    Subtask,
    #[serde(rename = "compaction")]
    Compaction,
    #[serde(rename = "step_start")]
    StepStart,
    #[serde(rename = "step_finish")]
    StepFinish,
    #[serde(rename = "snapshot")]
    Snapshot,
    #[serde(rename = "agent")]
    Agent,
    #[serde(rename = "retry")]
    Retry,
}

impl MessagePartType {
    fn as_str(&self) -> &'static str {
        match self {
            MessagePartType::Text => "text",
            MessagePartType::Reasoning => "reasoning",
            MessagePartType::Tool => "tool",
            MessagePartType::Patch => "patch",
            MessagePartType::File => "file",
            MessagePartType::Subtask => "subtask",
            MessagePartType::Compaction => "compaction",
            MessagePartType::StepStart => "step_start",
            MessagePartType::StepFinish => "step_finish",
            MessagePartType::Snapshot => "snapshot",
            MessagePartType::Agent => "agent",
            MessagePartType::Retry => "retry",
        }
    }

    fn from_db(value: &str) -> Self {
        match value {
            "reasoning" => Self::Reasoning,
            "tool" => Self::Tool,
            "patch" => Self::Patch,
            "file" => Self::File,
            "subtask" => Self::Subtask,
            "compaction" => Self::Compaction,
            "step_start" => Self::StepStart,
            "step_finish" => Self::StepFinish,
            "snapshot" => Self::Snapshot,
            "agent" => Self::Agent,
            "retry" => Self::Retry,
            _ => Self::Text,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CreateMessageInput {
    pub conversation_id: ConversationId,
    pub seq: i64,
    pub role: MessageRole,
    pub content: String,
    pub token_count: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MessageRecord {
    pub message_id: MessageId,
    pub conversation_id: ConversationId,
    pub seq: i64,
    pub role: MessageRole,
    pub content: String,
    pub token_count: i64,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CreateMessagePartInput {
    pub session_id: String,
    pub part_type: MessagePartType,
    pub ordinal: i64,
    pub text_content: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
    pub tool_input: Option<String>,
    pub tool_output: Option<String>,
    pub metadata: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MessagePartRecord {
    pub part_id: String,
    pub message_id: MessageId,
    pub session_id: String,
    pub part_type: MessagePartType,
    pub ordinal: i64,
    pub text_content: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
    pub tool_input: Option<String>,
    pub tool_output: Option<String>,
    pub metadata: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CreateConversationInput {
    pub session_id: String,
    pub title: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConversationRecord {
    pub conversation_id: ConversationId,
    pub session_id: String,
    pub title: Option<String>,
    pub bootstrapped_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MessageSearchInput {
    pub conversation_id: Option<ConversationId>,
    pub query: String,
    pub mode: String,
    pub since: Option<DateTime<Utc>>,
    pub before: Option<DateTime<Utc>>,
    pub limit: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MessageSearchResult {
    pub message_id: MessageId,
    pub conversation_id: ConversationId,
    pub role: MessageRole,
    pub snippet: String,
    pub created_at: DateTime<Utc>,
    pub rank: Option<f64>,
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

fn snippet_from_content(content: &str, max_len: usize) -> String {
    let compact = content.replace('\n', " ").trim().to_string();
    if compact.len() <= max_len {
        compact
    } else {
        format!("{}...", &compact[..max_len.saturating_sub(3)])
    }
}

#[derive(Clone)]
pub struct ConversationStore {
    conn: Arc<Mutex<Connection>>,
}

impl ConversationStore {
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

    pub fn with_transaction<T, F>(&self, f: F) -> anyhow::Result<T>
    where
        F: FnOnce(&Connection) -> anyhow::Result<T>,
    {
        self.with_conn(|conn| {
            conn.execute_batch("BEGIN IMMEDIATE")?;
            match f(conn) {
                Ok(value) => {
                    conn.execute_batch("COMMIT")?;
                    Ok(value)
                }
                Err(err) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    Err(err)
                }
            }
        })
    }

    pub fn create_conversation(
        &self,
        input: CreateConversationInput,
    ) -> anyhow::Result<ConversationRecord> {
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO conversations (session_id, title) VALUES (?, ?)",
                params![input.session_id, input.title],
            )?;
            let id = conn.last_insert_rowid();
            conn.query_row(
                "SELECT conversation_id, session_id, title, bootstrapped_at, created_at, updated_at
                 FROM conversations WHERE conversation_id = ?",
                params![id],
                |row| {
                    Ok(ConversationRecord {
                        conversation_id: row.get(0)?,
                        session_id: row.get(1)?,
                        title: row.get(2)?,
                        bootstrapped_at: row
                            .get::<_, Option<String>>(3)?
                            .as_deref()
                            .map(parse_datetime),
                        created_at: parse_datetime(&row.get::<_, String>(4)?),
                        updated_at: parse_datetime(&row.get::<_, String>(5)?),
                    })
                },
            )
            .optional()?
            .context("conversation insert failed to re-read row")
        })
    }

    pub fn get_conversation(
        &self,
        conversation_id: ConversationId,
    ) -> anyhow::Result<Option<ConversationRecord>> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT conversation_id, session_id, title, bootstrapped_at, created_at, updated_at
                 FROM conversations WHERE conversation_id = ?",
                params![conversation_id],
                |row| {
                    Ok(ConversationRecord {
                        conversation_id: row.get(0)?,
                        session_id: row.get(1)?,
                        title: row.get(2)?,
                        bootstrapped_at: row
                            .get::<_, Option<String>>(3)?
                            .as_deref()
                            .map(parse_datetime),
                        created_at: parse_datetime(&row.get::<_, String>(4)?),
                        updated_at: parse_datetime(&row.get::<_, String>(5)?),
                    })
                },
            )
            .optional()
            .map_err(Into::into)
        })
    }

    pub fn get_conversation_by_session_id(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<ConversationRecord>> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT conversation_id, session_id, title, bootstrapped_at, created_at, updated_at
                 FROM conversations
                 WHERE session_id = ?
                 ORDER BY created_at DESC
                 LIMIT 1",
                params![session_id],
                |row| {
                    Ok(ConversationRecord {
                        conversation_id: row.get(0)?,
                        session_id: row.get(1)?,
                        title: row.get(2)?,
                        bootstrapped_at: row
                            .get::<_, Option<String>>(3)?
                            .as_deref()
                            .map(parse_datetime),
                        created_at: parse_datetime(&row.get::<_, String>(4)?),
                        updated_at: parse_datetime(&row.get::<_, String>(5)?),
                    })
                },
            )
            .optional()
            .map_err(Into::into)
        })
    }

    pub fn get_or_create_conversation(
        &self,
        session_id: &str,
        title: Option<&str>,
    ) -> anyhow::Result<ConversationRecord> {
        if let Some(existing) = self.get_conversation_by_session_id(session_id)? {
            return Ok(existing);
        }
        self.create_conversation(CreateConversationInput {
            session_id: session_id.to_string(),
            title: title.map(ToString::to_string),
        })
    }

    pub fn mark_conversation_bootstrapped(&self, conversation_id: ConversationId) -> anyhow::Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                "UPDATE conversations
                 SET bootstrapped_at = COALESCE(bootstrapped_at, datetime('now')),
                     updated_at = datetime('now')
                 WHERE conversation_id = ?",
                params![conversation_id],
            )?;
            Ok(())
        })
    }

    pub fn create_message(&self, input: CreateMessageInput) -> anyhow::Result<MessageRecord> {
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO messages (conversation_id, seq, role, content, token_count)
                 VALUES (?, ?, ?, ?, ?)",
                params![
                    input.conversation_id,
                    input.seq,
                    input.role.as_str(),
                    input.content,
                    input.token_count
                ],
            )?;
            let message_id = conn.last_insert_rowid();
            let _ = conn.execute(
                "INSERT INTO messages_fts(rowid, content) VALUES (?, ?)",
                params![message_id, input.content],
            );
            conn.query_row(
                "SELECT message_id, conversation_id, seq, role, content, token_count, created_at
                 FROM messages WHERE message_id = ?",
                params![message_id],
                |row| {
                    Ok(MessageRecord {
                        message_id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        seq: row.get(2)?,
                        role: MessageRole::from_db(&row.get::<_, String>(3)?),
                        content: row.get(4)?,
                        token_count: row.get(5)?,
                        created_at: parse_datetime(&row.get::<_, String>(6)?),
                    })
                },
            )
            .optional()?
            .context("message insert failed to re-read row")
        })
    }

    pub fn create_messages_bulk(
        &self,
        inputs: &[CreateMessageInput],
    ) -> anyhow::Result<Vec<MessageRecord>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }
        self.with_transaction(|conn| {
            let mut rows = Vec::with_capacity(inputs.len());
            for input in inputs {
                conn.execute(
                    "INSERT INTO messages (conversation_id, seq, role, content, token_count)
                     VALUES (?, ?, ?, ?, ?)",
                    params![
                        input.conversation_id,
                        input.seq,
                        input.role.as_str(),
                        input.content,
                        input.token_count
                    ],
                )?;
                let message_id = conn.last_insert_rowid();
                let _ = conn.execute(
                    "INSERT INTO messages_fts(rowid, content) VALUES (?, ?)",
                    params![message_id, input.content],
                );
                rows.push(
                    conn.query_row(
                        "SELECT message_id, conversation_id, seq, role, content, token_count, created_at
                         FROM messages WHERE message_id = ?",
                        params![message_id],
                        |row| {
                            Ok(MessageRecord {
                                message_id: row.get(0)?,
                                conversation_id: row.get(1)?,
                                seq: row.get(2)?,
                                role: MessageRole::from_db(&row.get::<_, String>(3)?),
                                content: row.get(4)?,
                                token_count: row.get(5)?,
                                created_at: parse_datetime(&row.get::<_, String>(6)?),
                            })
                        },
                    )
                    .optional()?
                    .context("bulk message insert failed to re-read row")?,
                );
            }
            Ok(rows)
        })
    }

    pub fn get_messages(
        &self,
        conversation_id: ConversationId,
        after_seq: Option<i64>,
        limit: Option<i64>,
    ) -> anyhow::Result<Vec<MessageRecord>> {
        self.with_conn(|conn| {
            let mut out = Vec::new();
            let after_seq = after_seq.unwrap_or(-1);
            if let Some(limit) = limit {
                let mut stmt = conn.prepare(
                    "SELECT message_id, conversation_id, seq, role, content, token_count, created_at
                     FROM messages
                     WHERE conversation_id = ? AND seq > ?
                     ORDER BY seq
                     LIMIT ?",
                )?;
                let mut rows = stmt.query(params![conversation_id, after_seq, limit])?;
                while let Some(row) = rows.next()? {
                    out.push(MessageRecord {
                        message_id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        seq: row.get(2)?,
                        role: MessageRole::from_db(&row.get::<_, String>(3)?),
                        content: row.get(4)?,
                        token_count: row.get(5)?,
                        created_at: parse_datetime(&row.get::<_, String>(6)?),
                    });
                }
                return Ok(out);
            }

            let mut stmt = conn.prepare(
                "SELECT message_id, conversation_id, seq, role, content, token_count, created_at
                 FROM messages
                 WHERE conversation_id = ? AND seq > ?
                 ORDER BY seq",
            )?;
            let mut rows = stmt.query(params![conversation_id, after_seq])?;
            while let Some(row) = rows.next()? {
                out.push(MessageRecord {
                    message_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    seq: row.get(2)?,
                    role: MessageRole::from_db(&row.get::<_, String>(3)?),
                    content: row.get(4)?,
                    token_count: row.get(5)?,
                    created_at: parse_datetime(&row.get::<_, String>(6)?),
                });
            }
            Ok(out)
        })
    }

    pub fn get_last_message(&self, conversation_id: ConversationId) -> anyhow::Result<Option<MessageRecord>> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT message_id, conversation_id, seq, role, content, token_count, created_at
                 FROM messages
                 WHERE conversation_id = ?
                 ORDER BY seq DESC
                 LIMIT 1",
                params![conversation_id],
                |row| {
                    Ok(MessageRecord {
                        message_id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        seq: row.get(2)?,
                        role: MessageRole::from_db(&row.get::<_, String>(3)?),
                        content: row.get(4)?,
                        token_count: row.get(5)?,
                        created_at: parse_datetime(&row.get::<_, String>(6)?),
                    })
                },
            )
            .optional()
            .map_err(Into::into)
        })
    }

    pub fn has_message(
        &self,
        conversation_id: ConversationId,
        role: MessageRole,
        content: &str,
    ) -> anyhow::Result<bool> {
        self.with_conn(|conn| {
            let found = conn
                .query_row(
                    "SELECT 1 FROM messages WHERE conversation_id = ? AND role = ? AND content = ? LIMIT 1",
                    params![conversation_id, role.as_str(), content],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
                .is_some();
            Ok(found)
        })
    }

    pub fn count_messages_by_identity(
        &self,
        conversation_id: ConversationId,
        role: MessageRole,
        content: &str,
    ) -> anyhow::Result<i64> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND role = ? AND content = ?",
                params![conversation_id, role.as_str(), content],
                |row| row.get::<_, i64>(0),
            )
            .map_err(Into::into)
        })
    }

    pub fn get_message_by_id(&self, message_id: MessageId) -> anyhow::Result<Option<MessageRecord>> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT message_id, conversation_id, seq, role, content, token_count, created_at
                 FROM messages WHERE message_id = ?",
                params![message_id],
                |row| {
                    Ok(MessageRecord {
                        message_id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        seq: row.get(2)?,
                        role: MessageRole::from_db(&row.get::<_, String>(3)?),
                        content: row.get(4)?,
                        token_count: row.get(5)?,
                        created_at: parse_datetime(&row.get::<_, String>(6)?),
                    })
                },
            )
            .optional()
            .map_err(Into::into)
        })
    }

    pub fn create_message_parts(
        &self,
        message_id: MessageId,
        parts: &[CreateMessagePartInput],
    ) -> anyhow::Result<()> {
        if parts.is_empty() {
            return Ok(());
        }
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "INSERT INTO message_parts (
                    part_id, message_id, session_id, part_type, ordinal, text_content,
                    tool_call_id, tool_name, tool_input, tool_output, metadata
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )?;
            for part in parts {
                stmt.execute(params![
                    Uuid::new_v4().to_string(),
                    message_id,
                    part.session_id,
                    part.part_type.as_str(),
                    part.ordinal,
                    part.text_content,
                    part.tool_call_id,
                    part.tool_name,
                    part.tool_input,
                    part.tool_output,
                    part.metadata
                ])?;
            }
            Ok(())
        })
    }

    pub fn get_message_parts(&self, message_id: MessageId) -> anyhow::Result<Vec<MessagePartRecord>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT part_id, message_id, session_id, part_type, ordinal, text_content,
                        tool_call_id, tool_name, tool_input, tool_output, metadata
                 FROM message_parts
                 WHERE message_id = ?
                 ORDER BY ordinal",
            )?;
            let mut rows = stmt.query(params![message_id])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(MessagePartRecord {
                    part_id: row.get(0)?,
                    message_id: row.get(1)?,
                    session_id: row.get(2)?,
                    part_type: MessagePartType::from_db(&row.get::<_, String>(3)?),
                    ordinal: row.get(4)?,
                    text_content: row.get(5)?,
                    tool_call_id: row.get(6)?,
                    tool_name: row.get(7)?,
                    tool_input: row.get(8)?,
                    tool_output: row.get(9)?,
                    metadata: row.get(10)?,
                });
            }
            Ok(out)
        })
    }

    pub fn get_message_count(&self, conversation_id: ConversationId) -> anyhow::Result<i64> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                params![conversation_id],
                |row| row.get::<_, i64>(0),
            )
            .map_err(Into::into)
        })
    }

    pub fn get_max_seq(&self, conversation_id: ConversationId) -> anyhow::Result<i64> {
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id = ?",
                params![conversation_id],
                |row| row.get::<_, i64>(0),
            )
            .map_err(Into::into)
        })
    }

    pub fn delete_messages(&self, message_ids: &[MessageId]) -> anyhow::Result<i64> {
        if message_ids.is_empty() {
            return Ok(0);
        }
        self.with_transaction(|conn| {
            let mut deleted = 0_i64;
            for message_id in message_ids {
                let has_ref = conn
                    .query_row(
                        "SELECT 1 FROM summary_messages WHERE message_id = ? LIMIT 1",
                        params![message_id],
                        |row| row.get::<_, i64>(0),
                    )
                    .optional()?
                    .is_some();
                if has_ref {
                    continue;
                }
                conn.execute(
                    "DELETE FROM context_items WHERE item_type = 'message' AND message_id = ?",
                    params![message_id],
                )?;
                let _ = conn.execute("DELETE FROM messages_fts WHERE rowid = ?", params![message_id]);
                conn.execute("DELETE FROM messages WHERE message_id = ?", params![message_id])?;
                deleted += 1;
            }
            Ok(deleted)
        })
    }

    pub fn search_messages(&self, input: MessageSearchInput) -> anyhow::Result<Vec<MessageSearchResult>> {
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
        conversation_id: Option<ConversationId>,
        since: Option<DateTime<Utc>>,
        before: Option<DateTime<Utc>>,
    ) -> anyhow::Result<Vec<MessageSearchResult>> {
        self.with_conn(|conn| {
            let mut sql = String::from(
                "SELECT
                    m.message_id,
                    m.conversation_id,
                    m.role,
                    m.content,
                    rank,
                    m.created_at
                 FROM messages_fts
                 JOIN messages m ON m.message_id = messages_fts.rowid
                 WHERE messages_fts MATCH ?",
            );
            let mut args: Vec<rusqlite::types::Value> =
                vec![rusqlite::types::Value::from(sanitize_fts5_query(query))];
            if let Some(conversation_id) = conversation_id {
                sql.push_str(" AND m.conversation_id = ?");
                args.push(rusqlite::types::Value::from(conversation_id));
            }
            if let Some(since) = since {
                sql.push_str(" AND julianday(m.created_at) >= julianday(?)");
                args.push(rusqlite::types::Value::from(since.to_rfc3339()));
            }
            if let Some(before) = before {
                sql.push_str(" AND julianday(m.created_at) < julianday(?)");
                args.push(rusqlite::types::Value::from(before.to_rfc3339()));
            }
            sql.push_str(" ORDER BY m.created_at DESC LIMIT ?");
            args.push(rusqlite::types::Value::from(limit));

            let mut stmt = conn.prepare(&sql)?;
            let mut rows = stmt.query(rusqlite::params_from_iter(args))?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                let content: String = row.get(3)?;
                out.push(MessageSearchResult {
                    message_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    role: MessageRole::from_db(&row.get::<_, String>(2)?),
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
        conversation_id: Option<ConversationId>,
        since: Option<DateTime<Utc>>,
        before: Option<DateTime<Utc>>,
    ) -> anyhow::Result<Vec<MessageSearchResult>> {
        let re = Regex::new(pattern).with_context(|| "invalid regex pattern")?;
        self.with_conn(|conn| {
            let mut sql = String::from(
                "SELECT message_id, conversation_id, seq, role, content, token_count, created_at
                 FROM messages",
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
                    out.push(MessageSearchResult {
                        message_id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        role: MessageRole::from_db(&row.get::<_, String>(3)?),
                        snippet: mtch.as_str().to_string(),
                        rank: Some(0.0),
                        created_at: parse_datetime(&row.get::<_, String>(6)?),
                    });
                }
            }
            Ok(out)
        })
    }
}
