use std::collections::{HashMap, HashSet};

use chrono::{DateTime, NaiveDateTime, SecondsFormat, Utc};
use rusqlite::{params, Connection, OptionalExtension};

#[derive(Clone)]
struct SummaryDepthRow {
    summary_id: String,
    kind: String,
    token_count: i64,
    created_at: String,
}

#[derive(Clone)]
struct SummaryMessageTimeRangeRow {
    summary_id: String,
    earliest_at: Option<String>,
    latest_at: Option<String>,
    source_message_token_count: i64,
}

#[derive(Clone)]
struct SummaryParentEdgeRow {
    summary_id: String,
    parent_summary_id: String,
}

#[derive(Clone)]
struct SummaryMetadata {
    earliest_at: Option<DateTime<Utc>>,
    latest_at: Option<DateTime<Utc>>,
    descendant_count: i64,
    descendant_token_count: i64,
    source_message_token_count: i64,
}

fn table_columns(db: &Connection, table: &str) -> anyhow::Result<Vec<String>> {
    let sql = format!("PRAGMA table_info({})", table);
    let mut stmt = db.prepare(&sql)?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    let mut out = vec![];
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn ensure_summary_depth_column(db: &Connection) -> anyhow::Result<()> {
    let columns = table_columns(db, "summaries")?;
    if !columns.iter().any(|c| c == "depth") {
        db.execute_batch("ALTER TABLE summaries ADD COLUMN depth INTEGER NOT NULL DEFAULT 0")?;
    }
    Ok(())
}

fn ensure_summary_metadata_columns(db: &Connection) -> anyhow::Result<()> {
    let columns = table_columns(db, "summaries")?;
    if !columns.iter().any(|c| c == "earliest_at") {
        db.execute_batch("ALTER TABLE summaries ADD COLUMN earliest_at TEXT")?;
    }
    if !columns.iter().any(|c| c == "latest_at") {
        db.execute_batch("ALTER TABLE summaries ADD COLUMN latest_at TEXT")?;
    }
    if !columns.iter().any(|c| c == "descendant_count") {
        db.execute_batch(
            "ALTER TABLE summaries ADD COLUMN descendant_count INTEGER NOT NULL DEFAULT 0",
        )?;
    }
    if !columns.iter().any(|c| c == "descendant_token_count") {
        db.execute_batch(
            "ALTER TABLE summaries ADD COLUMN descendant_token_count INTEGER NOT NULL DEFAULT 0",
        )?;
    }
    if !columns.iter().any(|c| c == "source_message_token_count") {
        db.execute_batch(
            "ALTER TABLE summaries ADD COLUMN source_message_token_count INTEGER NOT NULL DEFAULT 0",
        )?;
    }
    Ok(())
}

fn parse_timestamp(value: Option<&str>) -> Option<DateTime<Utc>> {
    let value = value?.trim();
    if value.is_empty() {
        return None;
    }

    if let Ok(parsed) = DateTime::parse_from_rfc3339(value) {
        return Some(parsed.with_timezone(&Utc));
    }

    if let Ok(parsed) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S") {
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(parsed, Utc));
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f") {
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(parsed, Utc));
    }

    let normalized = if value.contains('T') {
        value.to_string()
    } else {
        format!("{}Z", value.replace(' ', "T"))
    };
    DateTime::parse_from_rfc3339(&normalized)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

fn iso_string_or_null(value: Option<DateTime<Utc>>) -> Option<String> {
    value.map(|dt| dt.to_rfc3339_opts(SecondsFormat::Millis, true))
}

fn backfill_summary_depths(db: &Connection) -> anyhow::Result<()> {
    db.execute("UPDATE summaries SET depth = 0 WHERE kind = 'leaf'", [])?;

    let mut conv_stmt =
        db.prepare("SELECT DISTINCT conversation_id FROM summaries WHERE kind = 'condensed'")?;
    let conv_rows = conv_stmt.query_map([], |row| row.get::<_, i64>(0))?;
    let mut conversation_ids = vec![];
    for row in conv_rows {
        conversation_ids.push(row?);
    }
    if conversation_ids.is_empty() {
        return Ok(());
    }

    for conversation_id in conversation_ids {
        let mut summaries_stmt = db.prepare(
            r#"
            SELECT summary_id, kind, token_count, created_at
            FROM summaries
            WHERE conversation_id = ?
            "#,
        )?;
        let summaries_rows = summaries_stmt.query_map([conversation_id], |row| {
            Ok(SummaryDepthRow {
                summary_id: row.get(0)?,
                kind: row.get(1)?,
                token_count: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        let mut summaries = vec![];
        for row in summaries_rows {
            summaries.push(row?);
        }

        let mut depth_by_summary_id: HashMap<String, i64> = HashMap::new();
        let mut unresolved_condensed_ids: HashSet<String> = HashSet::new();
        for summary in &summaries {
            if summary.kind == "leaf" {
                depth_by_summary_id.insert(summary.summary_id.clone(), 0);
            } else {
                unresolved_condensed_ids.insert(summary.summary_id.clone());
            }
        }

        let mut edges_stmt = db.prepare(
            r#"
            SELECT summary_id, parent_summary_id
            FROM summary_parents
            WHERE summary_id IN (
              SELECT summary_id FROM summaries
              WHERE conversation_id = ? AND kind = 'condensed'
            )
            "#,
        )?;
        let edge_rows = edges_stmt.query_map([conversation_id], |row| {
            Ok(SummaryParentEdgeRow {
                summary_id: row.get(0)?,
                parent_summary_id: row.get(1)?,
            })
        })?;
        let mut parents_by_summary_id: HashMap<String, Vec<String>> = HashMap::new();
        for row in edge_rows {
            let edge = row?;
            parents_by_summary_id
                .entry(edge.summary_id)
                .or_default()
                .push(edge.parent_summary_id);
        }

        while !unresolved_condensed_ids.is_empty() {
            let mut progressed = false;
            let unresolved: Vec<String> = unresolved_condensed_ids.iter().cloned().collect();
            for summary_id in unresolved {
                let parent_ids = parents_by_summary_id.get(&summary_id).cloned().unwrap_or_default();
                if parent_ids.is_empty() {
                    depth_by_summary_id.insert(summary_id.clone(), 1);
                    unresolved_condensed_ids.remove(&summary_id);
                    progressed = true;
                    continue;
                }

                let mut max_parent_depth = -1_i64;
                let mut all_parents_resolved = true;
                for parent_id in parent_ids {
                    let Some(parent_depth) = depth_by_summary_id.get(&parent_id).copied() else {
                        all_parents_resolved = false;
                        break;
                    };
                    if parent_depth > max_parent_depth {
                        max_parent_depth = parent_depth;
                    }
                }
                if !all_parents_resolved {
                    continue;
                }

                depth_by_summary_id.insert(summary_id.clone(), max_parent_depth + 1);
                unresolved_condensed_ids.remove(&summary_id);
                progressed = true;
            }

            if !progressed {
                for summary_id in unresolved_condensed_ids.iter() {
                    depth_by_summary_id.insert(summary_id.clone(), 1);
                }
                unresolved_condensed_ids.clear();
            }
        }

        let mut update_stmt = db.prepare("UPDATE summaries SET depth = ? WHERE summary_id = ?")?;
        for summary in summaries {
            if let Some(depth) = depth_by_summary_id.get(&summary.summary_id).copied() {
                update_stmt.execute(params![depth, summary.summary_id])?;
            }
        }
    }

    Ok(())
}

fn backfill_summary_metadata(db: &Connection) -> anyhow::Result<()> {
    let mut conv_stmt = db.prepare("SELECT DISTINCT conversation_id FROM summaries")?;
    let conv_rows = conv_stmt.query_map([], |row| row.get::<_, i64>(0))?;
    let mut conversation_ids = vec![];
    for row in conv_rows {
        conversation_ids.push(row?);
    }
    if conversation_ids.is_empty() {
        return Ok(());
    }

    for conversation_id in conversation_ids {
        let mut summaries_stmt = db.prepare(
            r#"
            SELECT summary_id, kind, token_count, created_at
            FROM summaries
            WHERE conversation_id = ?
            ORDER BY depth ASC, created_at ASC
            "#,
        )?;
        let summary_rows = summaries_stmt.query_map([conversation_id], |row| {
            Ok(SummaryDepthRow {
                summary_id: row.get(0)?,
                kind: row.get(1)?,
                token_count: row.get(2)?,
                created_at: row.get(3)?,
            })
        })?;
        let mut summaries = vec![];
        for row in summary_rows {
            summaries.push(row?);
        }
        if summaries.is_empty() {
            continue;
        }

        let mut leaf_stmt = db.prepare(
            r#"
            SELECT
              sm.summary_id,
              MIN(m.created_at) AS earliest_at,
              MAX(m.created_at) AS latest_at,
              COALESCE(SUM(m.token_count), 0) AS source_message_token_count
            FROM summary_messages sm
            JOIN messages m ON m.message_id = sm.message_id
            JOIN summaries s ON s.summary_id = sm.summary_id
            WHERE s.conversation_id = ? AND s.kind = 'leaf'
            GROUP BY sm.summary_id
            "#,
        )?;
        let leaf_rows = leaf_stmt.query_map([conversation_id], |row| {
            Ok(SummaryMessageTimeRangeRow {
                summary_id: row.get(0)?,
                earliest_at: row.get(1)?,
                latest_at: row.get(2)?,
                source_message_token_count: row.get::<_, Option<i64>>(3)?.unwrap_or(0),
            })
        })?;
        let mut leaf_range_by_summary_id: HashMap<String, SummaryMessageTimeRangeRow> =
            HashMap::new();
        for row in leaf_rows {
            let row = row?;
            leaf_range_by_summary_id.insert(row.summary_id.clone(), row);
        }

        let mut edges_stmt = db.prepare(
            r#"
            SELECT summary_id, parent_summary_id
            FROM summary_parents
            WHERE summary_id IN (
              SELECT summary_id FROM summaries WHERE conversation_id = ?
            )
            "#,
        )?;
        let edge_rows = edges_stmt.query_map([conversation_id], |row| {
            Ok(SummaryParentEdgeRow {
                summary_id: row.get(0)?,
                parent_summary_id: row.get(1)?,
            })
        })?;
        let mut parents_by_summary_id: HashMap<String, Vec<String>> = HashMap::new();
        for row in edge_rows {
            let edge = row?;
            parents_by_summary_id
                .entry(edge.summary_id)
                .or_default()
                .push(edge.parent_summary_id);
        }

        let mut metadata_by_summary_id: HashMap<String, SummaryMetadata> = HashMap::new();
        let mut token_count_by_summary_id: HashMap<String, i64> = HashMap::new();
        for summary in &summaries {
            token_count_by_summary_id.insert(summary.summary_id.clone(), summary.token_count.max(0));
        }

        for summary in &summaries {
            let fallback_date = parse_timestamp(Some(summary.created_at.as_str()));
            if summary.kind == "leaf" {
                let range = leaf_range_by_summary_id.get(&summary.summary_id);
                let earliest_at = parse_timestamp(
                    range
                        .and_then(|r| r.earliest_at.as_deref())
                        .or(Some(summary.created_at.as_str())),
                )
                .or(fallback_date);
                let latest_at = parse_timestamp(
                    range
                        .and_then(|r| r.latest_at.as_deref())
                        .or(Some(summary.created_at.as_str())),
                )
                .or(fallback_date);
                metadata_by_summary_id.insert(
                    summary.summary_id.clone(),
                    SummaryMetadata {
                        earliest_at,
                        latest_at,
                        descendant_count: 0,
                        descendant_token_count: 0,
                        source_message_token_count: range
                            .map(|r| r.source_message_token_count.max(0))
                            .unwrap_or(0),
                    },
                );
                continue;
            }

            let parent_ids = parents_by_summary_id
                .get(&summary.summary_id)
                .cloned()
                .unwrap_or_default();
            if parent_ids.is_empty() {
                metadata_by_summary_id.insert(
                    summary.summary_id.clone(),
                    SummaryMetadata {
                        earliest_at: fallback_date,
                        latest_at: fallback_date,
                        descendant_count: 0,
                        descendant_token_count: 0,
                        source_message_token_count: 0,
                    },
                );
                continue;
            }

            let mut earliest_at: Option<DateTime<Utc>> = None;
            let mut latest_at: Option<DateTime<Utc>> = None;
            let mut descendant_count = 0_i64;
            let mut descendant_token_count = 0_i64;
            let mut source_message_token_count = 0_i64;

            for parent_id in parent_ids {
                let Some(parent_meta) = metadata_by_summary_id.get(&parent_id) else {
                    continue;
                };

                if let Some(parent_earliest) = parent_meta.earliest_at {
                    if earliest_at.map(|dt| parent_earliest < dt).unwrap_or(true) {
                        earliest_at = Some(parent_earliest);
                    }
                }
                if let Some(parent_latest) = parent_meta.latest_at {
                    if latest_at.map(|dt| parent_latest > dt).unwrap_or(true) {
                        latest_at = Some(parent_latest);
                    }
                }

                descendant_count += parent_meta.descendant_count.max(0) + 1;
                let parent_token_count =
                    token_count_by_summary_id.get(&parent_id).copied().unwrap_or(0).max(0);
                descendant_token_count +=
                    parent_token_count + parent_meta.descendant_token_count.max(0);
                source_message_token_count += parent_meta.source_message_token_count.max(0);
            }

            metadata_by_summary_id.insert(
                summary.summary_id.clone(),
                SummaryMetadata {
                    earliest_at: earliest_at.or(fallback_date),
                    latest_at: latest_at.or(fallback_date),
                    descendant_count: descendant_count.max(0),
                    descendant_token_count: descendant_token_count.max(0),
                    source_message_token_count: source_message_token_count.max(0),
                },
            );
        }

        let mut update_stmt = db.prepare(
            r#"
            UPDATE summaries
            SET earliest_at = ?, latest_at = ?, descendant_count = ?,
                descendant_token_count = ?, source_message_token_count = ?
            WHERE summary_id = ?
            "#,
        )?;
        for summary in summaries {
            let Some(metadata) = metadata_by_summary_id.get(&summary.summary_id) else {
                continue;
            };
            update_stmt.execute(params![
                iso_string_or_null(metadata.earliest_at),
                iso_string_or_null(metadata.latest_at),
                metadata.descendant_count.max(0),
                metadata.descendant_token_count.max(0),
                metadata.source_message_token_count.max(0),
                summary.summary_id
            ])?;
        }
    }

    Ok(())
}

pub fn run_lcm_migrations(db: &Connection) -> anyhow::Result<()> {
    db.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS conversations (
          conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT NOT NULL,
          title TEXT,
          bootstrapped_at TEXT,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS messages (
          message_id INTEGER PRIMARY KEY AUTOINCREMENT,
          conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
          seq INTEGER NOT NULL,
          role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
          content TEXT NOT NULL,
          token_count INTEGER NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          UNIQUE (conversation_id, seq)
        );

        CREATE TABLE IF NOT EXISTS summaries (
          summary_id TEXT PRIMARY KEY,
          conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
          kind TEXT NOT NULL CHECK (kind IN ('leaf', 'condensed')),
          depth INTEGER NOT NULL DEFAULT 0,
          content TEXT NOT NULL,
          token_count INTEGER NOT NULL,
          earliest_at TEXT,
          latest_at TEXT,
          descendant_count INTEGER NOT NULL DEFAULT 0,
          descendant_token_count INTEGER NOT NULL DEFAULT 0,
          source_message_token_count INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          file_ids TEXT NOT NULL DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS message_parts (
          part_id TEXT PRIMARY KEY,
          message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
          session_id TEXT NOT NULL,
          part_type TEXT NOT NULL CHECK (part_type IN (
            'text', 'reasoning', 'tool', 'patch', 'file',
            'subtask', 'compaction', 'step_start', 'step_finish',
            'snapshot', 'agent', 'retry'
          )),
          ordinal INTEGER NOT NULL,
          text_content TEXT,
          is_ignored INTEGER,
          is_synthetic INTEGER,
          tool_call_id TEXT,
          tool_name TEXT,
          tool_status TEXT,
          tool_input TEXT,
          tool_output TEXT,
          tool_error TEXT,
          tool_title TEXT,
          patch_hash TEXT,
          patch_files TEXT,
          file_mime TEXT,
          file_name TEXT,
          file_url TEXT,
          subtask_prompt TEXT,
          subtask_desc TEXT,
          subtask_agent TEXT,
          step_reason TEXT,
          step_cost REAL,
          step_tokens_in INTEGER,
          step_tokens_out INTEGER,
          snapshot_hash TEXT,
          compaction_auto INTEGER,
          metadata TEXT,
          UNIQUE (message_id, ordinal)
        );

        CREATE TABLE IF NOT EXISTS summary_messages (
          summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
          message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE RESTRICT,
          ordinal INTEGER NOT NULL,
          PRIMARY KEY (summary_id, message_id)
        );

        CREATE TABLE IF NOT EXISTS summary_parents (
          summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
          parent_summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE RESTRICT,
          ordinal INTEGER NOT NULL,
          PRIMARY KEY (summary_id, parent_summary_id)
        );

        CREATE TABLE IF NOT EXISTS context_items (
          conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
          ordinal INTEGER NOT NULL,
          item_type TEXT NOT NULL CHECK (item_type IN ('message', 'summary')),
          message_id INTEGER REFERENCES messages(message_id) ON DELETE RESTRICT,
          summary_id TEXT REFERENCES summaries(summary_id) ON DELETE RESTRICT,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          PRIMARY KEY (conversation_id, ordinal),
          CHECK (
            (item_type = 'message' AND message_id IS NOT NULL AND summary_id IS NULL) OR
            (item_type = 'summary' AND summary_id IS NOT NULL AND message_id IS NULL)
          )
        );

        CREATE TABLE IF NOT EXISTS large_files (
          file_id TEXT PRIMARY KEY,
          conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
          file_name TEXT,
          mime_type TEXT,
          byte_size INTEGER,
          storage_uri TEXT NOT NULL,
          exploration_summary TEXT,
          created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS messages_conv_seq_idx ON messages (conversation_id, seq);
        CREATE INDEX IF NOT EXISTS summaries_conv_created_idx ON summaries (conversation_id, created_at);
        CREATE INDEX IF NOT EXISTS message_parts_message_idx ON message_parts (message_id);
        CREATE INDEX IF NOT EXISTS message_parts_type_idx ON message_parts (part_type);
        CREATE INDEX IF NOT EXISTS context_items_conv_idx ON context_items (conversation_id, ordinal);
        CREATE INDEX IF NOT EXISTS large_files_conv_idx ON large_files (conversation_id, created_at);
        "#,
    )?;

    let conversation_columns = table_columns(db, "conversations")?;
    if !conversation_columns.iter().any(|c| c == "bootstrapped_at") {
        db.execute_batch("ALTER TABLE conversations ADD COLUMN bootstrapped_at TEXT")?;
    }

    ensure_summary_depth_column(db)?;
    ensure_summary_metadata_columns(db)?;
    backfill_summary_depths(db)?;
    backfill_summary_metadata(db)?;

    let messages_fts_sql: Option<String> = db
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages_fts'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    match messages_fts_sql {
        Some(sql) if sql.contains("content_rowid") => {
            db.execute_batch(
                r#"
                DROP TABLE messages_fts;
                CREATE VIRTUAL TABLE messages_fts USING fts5(
                  content,
                  tokenize='porter unicode61'
                );
                INSERT INTO messages_fts(rowid, content) SELECT message_id, content FROM messages;
                "#,
            )?;
        }
        Some(_) => {}
        None => {
            db.execute_batch(
                r#"
                CREATE VIRTUAL TABLE messages_fts USING fts5(
                  content,
                  tokenize='porter unicode61'
                );
                "#,
            )?;
        }
    }

    let summaries_fts_sql: Option<String> = db
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='summaries_fts'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let summaries_fts_columns = table_columns(db, "summaries_fts").unwrap_or_default();
    let has_summary_id_column = summaries_fts_columns.iter().any(|c| c == "summary_id");
    let sql = summaries_fts_sql.clone().unwrap_or_default();
    let should_recreate_summaries_fts = summaries_fts_sql.is_none()
        || !has_summary_id_column
        || sql.contains("content_rowid='summary_id'")
        || sql.contains("content_rowid=\"summary_id\"");
    if should_recreate_summaries_fts {
        db.execute_batch(
            r#"
            DROP TABLE IF EXISTS summaries_fts;
            CREATE VIRTUAL TABLE summaries_fts USING fts5(
              summary_id UNINDEXED,
              content,
              tokenize='porter unicode61'
            );
            INSERT INTO summaries_fts(summary_id, content)
            SELECT summary_id, content FROM summaries;
            "#,
        )?;
    }

    Ok(())
}
