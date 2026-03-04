use std::collections::HashMap;

use chrono::{DateTime, Utc};
use sieve_lcm::db::connection::{close_lcm_connection, get_lcm_connection};
use sieve_lcm::db::migration::run_lcm_migrations;

fn parse_datetime(value: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(value)
        .expect("valid rfc3339")
        .with_timezone(&Utc)
}

#[test]
fn adds_depth_and_metadata_from_summary_lineage() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let db_path = temp_dir.path().join("legacy.db");
    let db_path_str = db_path.to_string_lossy().to_string();
    let shared = get_lcm_connection(&db_path_str).expect("db");
    let db = shared.conn.lock();

    db.execute_batch(
        r#"
      CREATE TABLE conversations (
        conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        title TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
      );

      CREATE TABLE summaries (
        summary_id TEXT PRIMARY KEY,
        conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        kind TEXT NOT NULL CHECK (kind IN ('leaf', 'condensed')),
        content TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        file_ids TEXT NOT NULL DEFAULT '[]'
      );

      CREATE TABLE messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
        content TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE (conversation_id, seq)
      );

      CREATE TABLE summary_messages (
        summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
        message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE RESTRICT,
        ordinal INTEGER NOT NULL,
        PRIMARY KEY (summary_id, message_id)
      );

      CREATE TABLE summary_parents (
        summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
        parent_summary_id TEXT NOT NULL REFERENCES summaries(summary_id) ON DELETE RESTRICT,
        ordinal INTEGER NOT NULL,
        PRIMARY KEY (summary_id, parent_summary_id)
      );
    "#,
    )
    .expect("legacy schema");

    db.execute(
        "INSERT INTO conversations (conversation_id, session_id) VALUES (?, ?)",
        rusqlite::params![1, "legacy-session"],
    )
    .expect("insert conversation");

    let mut insert_summary_stmt = db
        .prepare(
            r#"
       INSERT INTO summaries (summary_id, conversation_id, kind, content, token_count, file_ids)
       VALUES (?, ?, ?, ?, ?, '[]')
    "#,
        )
        .expect("insert summary stmt");
    insert_summary_stmt
        .execute(rusqlite::params!["sum_leaf_a", 1, "leaf", "leaf-a", 10])
        .expect("sum_leaf_a");
    insert_summary_stmt
        .execute(rusqlite::params!["sum_leaf_b", 1, "leaf", "leaf-b", 10])
        .expect("sum_leaf_b");
    insert_summary_stmt
        .execute(rusqlite::params!["sum_condensed_1", 1, "condensed", "condensed-1", 10])
        .expect("sum_condensed_1");
    insert_summary_stmt
        .execute(rusqlite::params!["sum_condensed_2", 1, "condensed", "condensed-2", 10])
        .expect("sum_condensed_2");
    insert_summary_stmt
        .execute(rusqlite::params![
            "sum_condensed_orphan",
            1,
            "condensed",
            "condensed-orphan",
            10
        ])
        .expect("sum_condensed_orphan");
    drop(insert_summary_stmt);

    let mut insert_message_stmt = db
        .prepare(
            r#"
       INSERT INTO messages (message_id, conversation_id, seq, role, content, token_count, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)
    "#,
        )
        .expect("insert message stmt");
    insert_message_stmt
        .execute(rusqlite::params![1, 1, 1, "user", "m1", 5, "2026-01-01 10:00:00"])
        .expect("msg1");
    insert_message_stmt
        .execute(rusqlite::params![
            2,
            1,
            2,
            "assistant",
            "m2",
            5,
            "2026-01-01 11:30:00"
        ])
        .expect("msg2");
    insert_message_stmt
        .execute(rusqlite::params![3, 1, 3, "user", "m3", 5, "2026-01-01 12:45:00"])
        .expect("msg3");
    drop(insert_message_stmt);

    let mut link_message_stmt = db
        .prepare(
            r#"
       INSERT INTO summary_messages (summary_id, message_id, ordinal)
       VALUES (?, ?, ?)
    "#,
        )
        .expect("link message stmt");
    link_message_stmt
        .execute(rusqlite::params!["sum_leaf_a", 1, 0])
        .expect("link1");
    link_message_stmt
        .execute(rusqlite::params!["sum_leaf_a", 2, 1])
        .expect("link2");
    link_message_stmt
        .execute(rusqlite::params!["sum_leaf_b", 3, 0])
        .expect("link3");
    drop(link_message_stmt);

    let mut link_stmt = db
        .prepare(
            r#"
       INSERT INTO summary_parents (summary_id, parent_summary_id, ordinal)
       VALUES (?, ?, ?)
    "#,
        )
        .expect("summary parent stmt");
    link_stmt
        .execute(rusqlite::params!["sum_condensed_1", "sum_leaf_a", 0])
        .expect("parent1");
    link_stmt
        .execute(rusqlite::params!["sum_condensed_1", "sum_leaf_b", 1])
        .expect("parent2");
    link_stmt
        .execute(rusqlite::params!["sum_condensed_2", "sum_condensed_1", 0])
        .expect("parent3");
    drop(link_stmt);

    run_lcm_migrations(&db).expect("migrate");

    let mut pragma_stmt = db
        .prepare("PRAGMA table_info(summaries)")
        .expect("pragma summaries");
    let columns = pragma_stmt
        .query_map([], |row| row.get::<_, String>(1))
        .expect("columns");
    let mut names = vec![];
    for column in columns {
        names.push(column.expect("column"));
    }
    assert!(names.contains(&"depth".to_string()));
    assert!(names.contains(&"earliest_at".to_string()));
    assert!(names.contains(&"latest_at".to_string()));
    assert!(names.contains(&"descendant_count".to_string()));
    assert!(names.contains(&"descendant_token_count".to_string()));
    assert!(names.contains(&"source_message_token_count".to_string()));

    #[derive(Clone)]
    struct Row {
        summary_id: String,
        depth: i64,
        earliest_at: Option<String>,
        latest_at: Option<String>,
        descendant_count: i64,
        descendant_token_count: i64,
        source_message_token_count: i64,
    }

    let mut row_stmt = db
        .prepare(
            r#"
        SELECT summary_id, depth, earliest_at, latest_at, descendant_count,
               descendant_token_count, source_message_token_count
        FROM summaries
        ORDER BY summary_id
      "#,
        )
        .expect("select rows");
    let rows = row_stmt
        .query_map([], |row| {
            Ok(Row {
                summary_id: row.get(0)?,
                depth: row.get(1)?,
                earliest_at: row.get(2)?,
                latest_at: row.get(3)?,
                descendant_count: row.get(4)?,
                descendant_token_count: row.get(5)?,
                source_message_token_count: row.get(6)?,
            })
        })
        .expect("map rows");
    let mut all = vec![];
    for row in rows {
        all.push(row.expect("row"));
    }

    let depth_by_summary_id: HashMap<String, i64> =
        all.iter().map(|row| (row.summary_id.clone(), row.depth)).collect();
    let earliest_by_summary_id: HashMap<String, Option<String>> = all
        .iter()
        .map(|row| (row.summary_id.clone(), row.earliest_at.clone()))
        .collect();
    let latest_by_summary_id: HashMap<String, Option<String>> = all
        .iter()
        .map(|row| (row.summary_id.clone(), row.latest_at.clone()))
        .collect();
    let descendant_count_by_summary_id: HashMap<String, i64> = all
        .iter()
        .map(|row| (row.summary_id.clone(), row.descendant_count))
        .collect();
    let descendant_token_count_by_summary_id: HashMap<String, i64> = all
        .iter()
        .map(|row| (row.summary_id.clone(), row.descendant_token_count))
        .collect();
    let source_message_token_count_by_summary_id: HashMap<String, i64> = all
        .iter()
        .map(|row| (row.summary_id.clone(), row.source_message_token_count))
        .collect();

    assert_eq!(depth_by_summary_id.get("sum_leaf_a"), Some(&0));
    assert_eq!(depth_by_summary_id.get("sum_leaf_b"), Some(&0));
    assert_eq!(depth_by_summary_id.get("sum_condensed_1"), Some(&1));
    assert_eq!(depth_by_summary_id.get("sum_condensed_2"), Some(&2));
    assert_eq!(depth_by_summary_id.get("sum_condensed_orphan"), Some(&1));

    let leaf_a_earliest = earliest_by_summary_id
        .get("sum_leaf_a")
        .and_then(|v| v.as_deref())
        .expect("leaf a earliest");
    let leaf_a_latest = latest_by_summary_id
        .get("sum_leaf_a")
        .and_then(|v| v.as_deref())
        .expect("leaf a latest");
    let leaf_b_earliest = earliest_by_summary_id
        .get("sum_leaf_b")
        .and_then(|v| v.as_deref())
        .expect("leaf b earliest");
    let leaf_b_latest = latest_by_summary_id
        .get("sum_leaf_b")
        .and_then(|v| v.as_deref())
        .expect("leaf b latest");
    let condensed1_earliest = earliest_by_summary_id
        .get("sum_condensed_1")
        .and_then(|v| v.as_deref())
        .expect("condensed1 earliest");
    let condensed1_latest = latest_by_summary_id
        .get("sum_condensed_1")
        .and_then(|v| v.as_deref())
        .expect("condensed1 latest");
    let condensed2_earliest = earliest_by_summary_id
        .get("sum_condensed_2")
        .and_then(|v| v.as_deref())
        .expect("condensed2 earliest");
    let condensed2_latest = latest_by_summary_id
        .get("sum_condensed_2")
        .and_then(|v| v.as_deref())
        .expect("condensed2 latest");

    assert!(leaf_a_earliest.contains("2026-01-01"));
    assert!(leaf_a_latest.contains("2026-01-01"));
    assert!(leaf_b_earliest.contains("2026-01-01"));
    assert!(leaf_b_latest.contains("2026-01-01"));
    assert!(condensed1_earliest.contains("2026-01-01"));
    assert!(condensed1_latest.contains("2026-01-01"));
    assert!(condensed2_earliest.contains("2026-01-01"));
    assert!(condensed2_latest.contains("2026-01-01"));

    assert!(parse_datetime(leaf_a_earliest) <= parse_datetime(leaf_a_latest));
    assert!(parse_datetime(leaf_b_earliest) <= parse_datetime(leaf_b_latest));
    assert!(parse_datetime(condensed1_earliest) <= parse_datetime(condensed1_latest));
    assert!(parse_datetime(condensed2_earliest) <= parse_datetime(condensed2_latest));
    assert!(parse_datetime(condensed1_earliest) <= parse_datetime(leaf_a_earliest));
    assert!(parse_datetime(condensed1_latest) >= parse_datetime(leaf_b_latest));
    assert!(earliest_by_summary_id
        .get("sum_condensed_orphan")
        .and_then(|v| v.as_deref())
        .is_some());
    assert!(latest_by_summary_id
        .get("sum_condensed_orphan")
        .and_then(|v| v.as_deref())
        .is_some());

    assert_eq!(descendant_count_by_summary_id.get("sum_leaf_a"), Some(&0));
    assert_eq!(descendant_count_by_summary_id.get("sum_leaf_b"), Some(&0));
    assert_eq!(descendant_count_by_summary_id.get("sum_condensed_1"), Some(&2));
    assert_eq!(descendant_count_by_summary_id.get("sum_condensed_2"), Some(&3));
    assert_eq!(descendant_count_by_summary_id.get("sum_condensed_orphan"), Some(&0));

    assert_eq!(descendant_token_count_by_summary_id.get("sum_leaf_a"), Some(&0));
    assert_eq!(descendant_token_count_by_summary_id.get("sum_leaf_b"), Some(&0));
    assert_eq!(
        descendant_token_count_by_summary_id.get("sum_condensed_1"),
        Some(&20)
    );
    assert_eq!(
        descendant_token_count_by_summary_id.get("sum_condensed_2"),
        Some(&30)
    );
    assert_eq!(
        descendant_token_count_by_summary_id.get("sum_condensed_orphan"),
        Some(&0)
    );

    assert_eq!(source_message_token_count_by_summary_id.get("sum_leaf_a"), Some(&10));
    assert_eq!(source_message_token_count_by_summary_id.get("sum_leaf_b"), Some(&5));
    assert_eq!(
        source_message_token_count_by_summary_id.get("sum_condensed_1"),
        Some(&15)
    );
    assert_eq!(
        source_message_token_count_by_summary_id.get("sum_condensed_2"),
        Some(&15)
    );
    assert_eq!(
        source_message_token_count_by_summary_id.get("sum_condensed_orphan"),
        Some(&0)
    );

    close_lcm_connection(Some(&db_path_str));
}
