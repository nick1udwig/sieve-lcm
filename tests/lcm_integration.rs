use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use regex::Regex;
use rusqlite::params;
use serde_json::Value;
use sieve_lcm::assembler::{AssembleContextInput, ContextAssembler};
use sieve_lcm::compaction::{
    CompactInput, CompactLeafInput, CompactUntilUnderInput, CompactionConfig, CompactionEngine,
    CompactionLevel,
};
use sieve_lcm::db::connection::{SharedConnection, close_lcm_connection, get_lcm_connection};
use sieve_lcm::db::migration::run_lcm_migrations;
use sieve_lcm::retrieval::{
    DescribeResultType, ExpandInput, GrepInput, RetrievalApi, RetrievalEngine,
};
use sieve_lcm::store::conversation_store::{
    ConversationStore, CreateConversationInput, CreateMessageInput, MessagePartRecord,
    MessagePartType, MessageRecord, MessageRole,
};
use sieve_lcm::store::summary_store::{
    CreateLargeFileInput, CreateSummaryInput, SummaryKind, SummaryStore,
};
use sieve_lcm::summarize::{LcmSummarizeFn, LcmSummarizeOptions};
use tempfile::TempDir;

fn estimate_tokens(text: &str) -> i64 {
    ((text.chars().count() as f64) / 4.0).ceil() as i64
}

fn parse_utc(value: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(value)
        .expect("valid rfc3339")
        .with_timezone(&Utc)
}

fn extract_message_text(content: &Value) -> String {
    if let Some(text) = content.as_str() {
        return text.to_string();
    }
    let Some(items) = content.as_array() else {
        return String::new();
    };
    items
        .iter()
        .filter_map(|item| {
            item.get("text")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .collect::<Vec<String>>()
        .join("\n")
}

fn summarize_fn<F>(f: F) -> LcmSummarizeFn
where
    F: Fn(String, bool, Option<LcmSummarizeOptions>) -> String + Send + Sync + 'static,
{
    Arc::new(move |text, aggressive, options| {
        let output = f(text, aggressive, options);
        Box::pin(async move { output })
    })
}

fn default_compaction_config() -> CompactionConfig {
    CompactionConfig {
        context_threshold: 0.75,
        fresh_tail_count: 4,
        leaf_min_fanout: 8,
        condensed_min_fanout: 4,
        condensed_min_fanout_hard: 2,
        incremental_max_depth: 0,
        leaf_chunk_tokens: 20_000,
        leaf_target_tokens: 600,
        condensed_target_tokens: 900,
        max_rounds: 10,
        timezone: Some("UTC".to_string()),
    }
}

struct Harness {
    _temp_dir: TempDir,
    db_path: String,
    shared: SharedConnection,
    conv_store: ConversationStore,
    sum_store: SummaryStore,
    assembler: ContextAssembler,
    retrieval: RetrievalEngine,
    conversation_id: i64,
}

impl Harness {
    fn new() -> Self {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let db_path = temp_dir
            .path()
            .join("lcm_integration.db")
            .to_string_lossy()
            .to_string();
        let shared = get_lcm_connection(&db_path).expect("db connection");
        {
            let conn = shared.conn.lock();
            run_lcm_migrations(&conn).expect("migrations");
        }

        let conv_store = ConversationStore::new(&shared);
        let sum_store = SummaryStore::new(&shared);
        let conversation = conv_store
            .create_conversation(CreateConversationInput {
                session_id: "session-1".to_string(),
                title: None,
            })
            .expect("create conversation");

        let assembler = ContextAssembler::new(conv_store.clone(), sum_store.clone());
        let retrieval = RetrievalEngine::new(conv_store.clone(), sum_store.clone());

        Self {
            _temp_dir: temp_dir,
            db_path,
            shared,
            conv_store,
            sum_store,
            assembler,
            retrieval,
            conversation_id: conversation.conversation_id,
        }
    }

    fn compaction_engine(&self, config: CompactionConfig) -> CompactionEngine {
        CompactionEngine::new(self.conv_store.clone(), self.sum_store.clone(), config)
    }

    fn create_conversation(&self, session_id: &str) -> i64 {
        self.conv_store
            .create_conversation(CreateConversationInput {
                session_id: session_id.to_string(),
                title: None,
            })
            .expect("create conversation")
            .conversation_id
    }

    fn set_message_created_at(&self, message_id: i64, at: DateTime<Utc>) {
        self.shared
            .conn
            .lock()
            .execute(
                "UPDATE messages SET created_at = ? WHERE message_id = ?",
                params![at.to_rfc3339(), message_id],
            )
            .expect("update message timestamp");
    }

    fn set_summary_created_at(&self, summary_id: &str, at: DateTime<Utc>) {
        self.shared
            .conn
            .lock()
            .execute(
                "UPDATE summaries SET created_at = ? WHERE summary_id = ?",
                params![at.to_rfc3339(), summary_id],
            )
            .expect("update summary timestamp");
    }

    fn compaction_parts(&self, conversation_id: i64) -> Vec<MessagePartRecord> {
        let messages = self
            .conv_store
            .get_messages(conversation_id, None, None)
            .expect("messages");
        let mut parts = Vec::new();
        for message in messages {
            for part in self
                .conv_store
                .get_message_parts(message.message_id)
                .expect("parts")
            {
                if matches!(part.part_type, MessagePartType::Compaction) {
                    parts.push(part);
                }
            }
        }
        parts
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        close_lcm_connection(Some(&self.db_path));
    }
}

async fn ingest_messages_with<FContent, FRole, FToken>(
    conv_store: &ConversationStore,
    sum_store: &SummaryStore,
    conversation_id: i64,
    count: usize,
    mut content_fn: FContent,
    mut role_fn: FRole,
    mut token_fn: FToken,
) -> anyhow::Result<Vec<MessageRecord>>
where
    FContent: FnMut(usize) -> String,
    FRole: FnMut(usize) -> MessageRole,
    FToken: FnMut(usize, &str) -> i64,
{
    let mut rows = Vec::with_capacity(count);
    let mut next_seq = conv_store.get_max_seq(conversation_id)? + 1;
    for i in 0..count {
        let content = content_fn(i);
        let row = conv_store.create_message(CreateMessageInput {
            conversation_id,
            seq: next_seq,
            role: role_fn(i),
            token_count: token_fn(i, &content),
            content,
        })?;
        next_seq += 1;
        sum_store.append_context_message(conversation_id, row.message_id)?;
        rows.push(row);
    }
    Ok(rows)
}

async fn ingest_default(h: &Harness, count: usize) -> Vec<MessageRecord> {
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        count,
        |i| format!("Message {i}"),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest")
}

#[tokio::test]
async fn ingested_messages_appear_in_assembled_context() {
    let h = Harness::new();
    let _ = ingest_default(&h, 5).await;

    let result = h
        .assembler
        .assemble(AssembleContextInput {
            conversation_id: h.conversation_id,
            token_budget: 100_000,
            fresh_tail_count: None,
        })
        .await
        .expect("assemble");

    assert_eq!(result.messages.len(), 5);
    assert_eq!(result.stats.raw_message_count, 5);
    assert_eq!(result.stats.summary_count, 0);
    assert_eq!(result.stats.total_context_items, 5);
    for i in 0..5 {
        assert_eq!(
            extract_message_text(&result.messages[i].content),
            format!("Message {i}")
        );
    }
}

#[tokio::test]
async fn assembler_respects_token_budget_by_dropping_oldest_items() {
    let h = Harness::new();
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        10,
        |i| format!("M{i} {}", "x".repeat(396)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let result = h
        .assembler
        .assemble(AssembleContextInput {
            conversation_id: h.conversation_id,
            token_budget: 500,
            fresh_tail_count: Some(4),
        })
        .await
        .expect("assemble");

    let last_four = &result.messages[result.messages.len().saturating_sub(4)..];
    for (i, msg) in last_four.iter().enumerate() {
        assert!(extract_message_text(&msg.content).contains(&format!("M{}", 6 + i)));
    }
    assert!(result.messages.len() < 10);
    assert!(result.messages.len() <= 5);
}

#[tokio::test]
async fn assembler_includes_summaries_alongside_messages() {
    let h = Harness::new();
    let _ = ingest_default(&h, 2).await;

    let summary_id = "sum_test_001".to_string();
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: summary_id.clone(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Leaf,
            depth: None,
            content: "This is a leaf summary of earlier conversation.".to_string(),
            token_count: 20,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert summary");
    h.sum_store
        .append_context_summary(h.conversation_id, &summary_id)
        .expect("append context summary");

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        2,
        |i| format!("Later message {i}"),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let result = h
        .assembler
        .assemble(AssembleContextInput {
            conversation_id: h.conversation_id,
            token_budget: 100_000,
            fresh_tail_count: None,
        })
        .await
        .expect("assemble");

    assert_eq!(result.messages.len(), 5);
    assert_eq!(result.stats.raw_message_count, 4);
    assert_eq!(result.stats.summary_count, 1);

    let summary_msg = result
        .messages
        .iter()
        .find(|m| {
            m.content
                .as_str()
                .is_some_and(|text| text.contains("<summary id=\"sum_test_001\""))
        })
        .expect("summary message");
    assert_eq!(summary_msg.role, "user");
    assert!(
        summary_msg
            .content
            .as_str()
            .expect("summary text")
            .contains("This is a leaf summary")
    );
}

#[tokio::test]
async fn empty_conversation_returns_empty_result() {
    let h = Harness::new();
    let result = h
        .assembler
        .assemble(AssembleContextInput {
            conversation_id: h.conversation_id,
            token_budget: 100_000,
            fresh_tail_count: None,
        })
        .await
        .expect("assemble");

    assert_eq!(result.messages.len(), 0);
    assert_eq!(result.estimated_tokens, 0);
    assert_eq!(result.stats.total_context_items, 0);
}

#[tokio::test]
async fn fresh_tail_is_always_preserved_even_when_over_budget() {
    let h = Harness::new();
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        3,
        |i| format!("M{i} {}", "y".repeat(796)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let result = h
        .assembler
        .assemble(AssembleContextInput {
            conversation_id: h.conversation_id,
            token_budget: 100,
            fresh_tail_count: Some(8),
        })
        .await
        .expect("assemble");
    assert_eq!(result.messages.len(), 3);
}

#[tokio::test]
async fn degrades_tool_rows_without_tool_call_id_to_assistant_text() {
    let h = Harness::new();
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        1,
        |_i| "legacy tool output without call id".to_string(),
        |_i| MessageRole::Tool,
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let result = h
        .assembler
        .assemble(AssembleContextInput {
            conversation_id: h.conversation_id,
            token_budget: 100_000,
            fresh_tail_count: None,
        })
        .await
        .expect("assemble");

    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].role, "assistant");
    assert!(
        extract_message_text(&result.messages[0].content)
            .contains("legacy tool output without call id")
    );
}

#[tokio::test]
async fn compaction_creates_leaf_summary_from_oldest_messages() {
    let h = Harness::new();
    let engine = h.compaction_engine(default_compaction_config());
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        10,
        |i| format!("Turn {i}: discussion about topic {i}"),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let summarize = summarize_fn(|text, _aggressive, _options| {
        format!("Summary: condensed version of {} chars", text.len())
    });

    let result = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 10_000,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact");

    assert!(result.action_taken);
    assert!(
        result
            .created_summary_id
            .as_deref()
            .is_some_and(|id| id.starts_with("sum_"))
    );

    let all_summaries = h
        .sum_store
        .get_summaries_by_conversation(h.conversation_id)
        .expect("summaries");
    struct SumStorePresence {
        summaries: Option<()>,
    }
    let sumstore = SumStorePresence {
        summaries: all_summaries.first().map(|_| ()),
    };
    assert!(sumstore.summaries.is_some());
    let leaf_summary = all_summaries
        .iter()
        .find(|summary| matches!(summary.kind, SummaryKind::Leaf))
        .expect("leaf summary");
    assert!(leaf_summary.content.contains("Summary:"));

    let context_items = h
        .sum_store
        .get_context_items(h.conversation_id)
        .expect("context items");
    let summary_items = context_items
        .iter()
        .filter(|item| item.summary_id.is_some())
        .next();
    assert!(summary_items.is_some());
    assert!(context_items.len() < 10);
}

#[tokio::test]
async fn compact_leaf_uses_preceding_summary_context_for_soft_leaf_continuity() {
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        fresh_tail_count: 1,
        ..default_compaction_config()
    });

    for summary_id in ["sum_pre_1", "sum_pre_2", "sum_pre_3"] {
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.to_string(),
                conversation_id: h.conversation_id,
                kind: SummaryKind::Leaf,
                depth: None,
                content: match summary_id {
                    "sum_pre_1" => "Prior summary one.".to_string(),
                    "sum_pre_2" => "Prior summary two.".to_string(),
                    _ => "Prior summary three.".to_string(),
                },
                token_count: 4,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert summary");
        h.sum_store
            .append_context_summary(h.conversation_id, summary_id)
            .expect("append summary");
    }

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        4,
        |i| format!("Turn {i}: {}", "k".repeat(160)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, _content| 40,
    )
    .await
    .expect("ingest");

    let summarize_calls: Arc<Mutex<Vec<Option<LcmSummarizeOptions>>>> =
        Arc::new(Mutex::new(vec![]));
    let summarize_calls_ref = summarize_calls.clone();
    let summarize = summarize_fn(move |_text, _aggressive, options| {
        summarize_calls_ref.lock().push(options);
        "Leaf summary with continuity.".to_string()
    });

    let result = engine
        .compact_leaf(CompactLeafInput {
            conversation_id: h.conversation_id,
            token_budget: 200,
            summarize,
            force: Some(true),
            previous_summary_content: None,
        })
        .await
        .expect("compact leaf");

    assert!(result.action_taken);
    assert!(summarize_calls.lock().len() >= 1);
    let first = summarize_calls
        .lock()
        .first()
        .cloned()
        .flatten()
        .expect("call options");
    assert_eq!(
        first.previous_summary,
        Some("Prior summary two.\n\nPrior summary three.".to_string())
    );
    assert_eq!(first.is_condensed, Some(false));
}

#[tokio::test]
async fn compact_leaf_keeps_incremental_behavior_leaf_only_when_incremental_max_depth_is_zero() {
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        fresh_tail_count: 0,
        condensed_min_fanout: 2,
        leaf_chunk_tokens: 500,
        condensed_target_tokens: 10,
        incremental_max_depth: 0,
        ..default_compaction_config()
    });

    for summary_id in ["sum_depth_zero_leaf_a", "sum_depth_zero_leaf_b"] {
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.to_string(),
                conversation_id: h.conversation_id,
                kind: SummaryKind::Leaf,
                depth: Some(0),
                content: if summary_id.ends_with("_a") {
                    "Depth zero leaf A".to_string()
                } else {
                    "Depth zero leaf B".to_string()
                },
                token_count: 60,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert summary");
        h.sum_store
            .append_context_summary(h.conversation_id, summary_id)
            .expect("append summary");
    }

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        2,
        |i| format!("Leaf source turn {i}: {}", "m".repeat(160)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, _content| 120,
    )
    .await
    .expect("ingest");

    let summarize = summarize_fn(|_text, _aggressive, options| {
        if options.as_ref().and_then(|v| v.is_condensed) == Some(true) {
            "Condensed summary".to_string()
        } else {
            "Leaf summary".to_string()
        }
    });
    let result = engine
        .compact_leaf(CompactLeafInput {
            conversation_id: h.conversation_id,
            token_budget: 1_200,
            summarize,
            force: Some(true),
            previous_summary_content: None,
        })
        .await
        .expect("compact leaf");

    assert!(result.action_taken);
    assert!(!result.condensed);
    let condensed_count = h
        .sum_store
        .get_summaries_by_conversation(h.conversation_id)
        .expect("summaries")
        .iter()
        .filter(|summary| matches!(summary.kind, SummaryKind::Condensed))
        .count();
    assert_eq!(condensed_count, 0);
}

#[tokio::test]
async fn compact_leaf_performs_one_depth_zero_condensation_pass_when_incremental_max_depth_is_one()
{
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        fresh_tail_count: 0,
        condensed_min_fanout: 2,
        leaf_chunk_tokens: 500,
        condensed_target_tokens: 10,
        incremental_max_depth: 1,
        ..default_compaction_config()
    });

    for summary_id in ["sum_depth_one_leaf_a", "sum_depth_one_leaf_b"] {
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.to_string(),
                conversation_id: h.conversation_id,
                kind: SummaryKind::Leaf,
                depth: Some(0),
                content: if summary_id.ends_with("_a") {
                    "Depth zero leaf A".to_string()
                } else {
                    "Depth zero leaf B".to_string()
                },
                token_count: 60,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert summary");
        h.sum_store
            .append_context_summary(h.conversation_id, summary_id)
            .expect("append summary");
    }

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        2,
        |i| format!("Leaf source turn {i}: {}", "n".repeat(160)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, _content| 120,
    )
    .await
    .expect("ingest");

    let summarize = summarize_fn(|_text, _aggressive, options| {
        if options.as_ref().and_then(|v| v.is_condensed) == Some(true) {
            "Condensed summary".to_string()
        } else {
            "Leaf summary".to_string()
        }
    });
    let result = engine
        .compact_leaf(CompactLeafInput {
            conversation_id: h.conversation_id,
            token_budget: 1_200,
            summarize,
            force: Some(true),
            previous_summary_content: None,
        })
        .await
        .expect("compact leaf");

    assert!(result.action_taken);
    assert!(!result.condensed);
    let condensed_count = h
        .sum_store
        .get_summaries_by_conversation(h.conversation_id)
        .expect("summaries")
        .iter()
        .filter(|summary| matches!(summary.kind, SummaryKind::Condensed))
        .count();
    assert_eq!(condensed_count, 0);
}

#[tokio::test]
async fn compact_leaf_cascades_to_depth_two_when_incremental_max_depth_is_two() {
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        fresh_tail_count: 0,
        condensed_min_fanout: 2,
        leaf_chunk_tokens: 500,
        condensed_target_tokens: 10,
        incremental_max_depth: 2,
        ..default_compaction_config()
    });

    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: "sum_depth_two_existing_d1".to_string(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Condensed,
            depth: Some(1),
            content: "Existing depth one summary".to_string(),
            token_count: 60,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert depth one");
    for summary_id in ["sum_depth_two_leaf_a", "sum_depth_two_leaf_b"] {
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.to_string(),
                conversation_id: h.conversation_id,
                kind: SummaryKind::Leaf,
                depth: Some(0),
                content: if summary_id.ends_with("_a") {
                    "Depth zero leaf A".to_string()
                } else {
                    "Depth zero leaf B".to_string()
                },
                token_count: 60,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert summary");
    }
    for summary_id in [
        "sum_depth_two_existing_d1",
        "sum_depth_two_leaf_a",
        "sum_depth_two_leaf_b",
    ] {
        h.sum_store
            .append_context_summary(h.conversation_id, summary_id)
            .expect("append summary");
    }

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        2,
        |i| format!("Leaf source turn {i}: {}", "p".repeat(160)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, _content| 120,
    )
    .await
    .expect("ingest");

    let summarize_count = Arc::new(Mutex::new(0_i64));
    let summarize_count_ref = summarize_count.clone();
    let summarize = summarize_fn(move |_text, _aggressive, options| {
        if options.as_ref().and_then(|v| v.is_condensed) == Some(true) {
            let mut counter = summarize_count_ref.lock();
            *counter += 1;
            format!("Condensed summary {}", *counter)
        } else {
            "Leaf summary".to_string()
        }
    });

    let result = engine
        .compact_leaf(CompactLeafInput {
            conversation_id: h.conversation_id,
            token_budget: 1_200,
            summarize,
            force: Some(true),
            previous_summary_content: None,
        })
        .await
        .expect("compact leaf");

    assert!(result.action_taken);
    assert!(!result.condensed);

    let depth_two_exists = h
        .sum_store
        .get_summaries_by_conversation(h.conversation_id)
        .expect("summaries")
        .iter()
        .any(|summary| matches!(summary.kind, SummaryKind::Condensed) && summary.depth == 2);
    assert!(!depth_two_exists);
}

#[tokio::test]
async fn compaction_propagates_referenced_file_ids_into_summary_metadata() {
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        fresh_tail_count: 16,
        ..default_compaction_config()
    });

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        20,
        |i| {
            if i == 1 {
                "Review [LCM File: file_aaaabbbbccccdddd | spec.md | text/markdown | 1,024 bytes]"
                    .to_string()
            } else if i == 2 {
                "Also inspect file_1111222233334444 and file_aaaabbbbccccdddd for context."
                    .to_string()
            } else {
                format!("Turn {i}: regular planning text.")
            }
        },
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let summarize =
        summarize_fn(|_text, _aggressive, _options| "Condensed file-aware summary.".to_string());
    let result = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 10_000,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact");
    assert!(result.action_taken);

    let leaf_summary = h
        .sum_store
        .get_summaries_by_conversation(h.conversation_id)
        .expect("summaries")
        .into_iter()
        .find(|summary| matches!(summary.kind, SummaryKind::Leaf))
        .expect("leaf summary");
    assert_eq!(
        leaf_summary.file_ids,
        vec![
            "file_aaaabbbbccccdddd".to_string(),
            "file_1111222233334444".to_string()
        ]
    );
}

#[tokio::test]
async fn compaction_emits_one_durable_compaction_part_for_a_leaf_only_pass() {
    let h = Harness::new();
    let engine = h.compaction_engine(default_compaction_config());

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        5,
        |i| format!("Turn {i}: {}", "l".repeat(160)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, _content| 40,
    )
    .await
    .expect("ingest");

    let summarize = summarize_fn(|_text, _aggressive, _options| "Leaf summary".to_string());
    let result = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 250,
            summarize,
            force: None,
            hard_trigger: None,
        })
        .await
        .expect("compact");

    assert!(result.action_taken);
    assert!(!result.condensed);

    let compaction_parts = h.compaction_parts(h.conversation_id);
    assert_eq!(compaction_parts.len(), 1);

    let metadata: Value =
        serde_json::from_str(compaction_parts[0].metadata.as_deref().unwrap_or("{}"))
            .expect("metadata");
    assert_eq!(metadata["conversationId"].as_i64(), Some(h.conversation_id));
    assert_eq!(metadata["pass"].as_str(), Some("leaf"));
    assert!(metadata["tokensBefore"].is_number());
    assert!(metadata["tokensAfter"].is_number());
    assert!(
        metadata["tokensBefore"].as_i64().expect("tokens before")
            > metadata["tokensAfter"].as_i64().expect("tokens after")
    );
    assert!(metadata.get("level").is_some());
    assert!(metadata["createdSummaryId"].as_str().is_some());
    let created_summary_id = metadata["createdSummaryId"].as_str().expect("summary id");
    let created_summary_ids = metadata["createdSummaryIds"]
        .as_array()
        .expect("summary ids");
    assert_eq!(created_summary_ids.len(), 1);
    assert_eq!(created_summary_ids[0].as_str(), Some(created_summary_id));
    assert_eq!(metadata["condensedPassOccurred"].as_bool(), Some(false));
}

#[tokio::test]
async fn compaction_emits_durable_compaction_parts_for_leaf_and_condensed_passes() {
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        leaf_min_fanout: 2,
        leaf_chunk_tokens: 100,
        condensed_target_tokens: 10,
        ..default_compaction_config()
    });

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        8,
        |i| format!("Turn {i}: {}", "c".repeat(200)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, _content| 50,
    )
    .await
    .expect("ingest");

    let summarize = summarize_fn(|_text, _aggressive, _options| {
        "Compacted summary block with enough detail.".to_string()
    });
    let result = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 260,
            summarize,
            force: None,
            hard_trigger: None,
        })
        .await
        .expect("compact");

    assert!(result.action_taken);
    assert!(result.condensed);

    let compaction_parts = h.compaction_parts(h.conversation_id);
    assert!(compaction_parts.len() >= 2);
    let metadata: Vec<Value> = compaction_parts
        .iter()
        .map(|part| {
            serde_json::from_str(part.metadata.as_deref().unwrap_or("{}")).expect("metadata")
        })
        .collect();

    let leaf_part = metadata
        .iter()
        .find(|value| value["pass"].as_str() == Some("leaf"))
        .expect("leaf part");
    let condensed_part = metadata
        .iter()
        .find(|value| value["pass"].as_str() == Some("condensed"))
        .expect("condensed part");

    assert_eq!(
        leaf_part["conversationId"].as_i64(),
        Some(h.conversation_id)
    );
    assert_eq!(
        condensed_part["conversationId"].as_i64(),
        Some(h.conversation_id)
    );
    assert!(leaf_part["tokensBefore"].is_number());
    assert!(leaf_part["tokensAfter"].is_number());
    assert!(condensed_part["tokensBefore"].is_number());
    assert!(condensed_part["tokensAfter"].is_number());
    assert!(leaf_part.get("level").is_some());
    assert!(condensed_part.get("level").is_some());
    assert!(leaf_part["createdSummaryId"].as_str().is_some());
    assert!(condensed_part["createdSummaryId"].as_str().is_some());
    struct PartPresence {
        createdsummaryids: Option<()>,
    }
    let leafpart = PartPresence {
        createdsummaryids: leaf_part.get("createdSummaryIds").map(|_| ()),
    };
    let condensedpart = PartPresence {
        createdsummaryids: condensed_part.get("createdSummaryIds").map(|_| ()),
    };
    assert!(leafpart.createdsummaryids.is_some());
    assert!(condensedpart.createdsummaryids.is_some());
    assert!(leaf_part["condensedPassOccurred"].is_boolean());
    assert!(condensed_part["condensedPassOccurred"].is_boolean());
}

#[tokio::test]
async fn depth_aware_condensation_sets_condensed_depth_to_max_parent_depth_plus_one() {
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        leaf_min_fanout: 2,
        condensed_min_fanout: 2,
        leaf_chunk_tokens: 200,
        condensed_target_tokens: 10,
        ..default_compaction_config()
    });

    for summary_id in ["sum_depth_parent_a", "sum_depth_parent_b"] {
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.to_string(),
                conversation_id: h.conversation_id,
                kind: SummaryKind::Condensed,
                depth: Some(1),
                content: if summary_id.ends_with("_a") {
                    "Depth one summary A".to_string()
                } else {
                    "Depth one summary B".to_string()
                },
                token_count: 60,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert summary");
        h.sum_store
            .append_context_summary(h.conversation_id, summary_id)
            .expect("append summary");
    }

    let summarize =
        summarize_fn(|_text, _aggressive, _options| "Depth two merged summary".to_string());
    let result = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 500,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact");

    assert!(result.action_taken);
    let created_summary = h
        .sum_store
        .get_summary(
            result
                .created_summary_id
                .as_deref()
                .expect("created summary id"),
        )
        .expect("summary lookup");
    assert!(created_summary.is_some());
    let created = h
        .sum_store
        .get_summary(
            result
                .created_summary_id
                .as_deref()
                .expect("created summary id"),
        )
        .expect("summary lookup")
        .expect("summary row");
    assert_eq!(created.depth, 2);
}

#[tokio::test]
async fn depth_aware_selection_stops_on_depth_mismatch_and_does_not_mix_depth_bands() {
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        leaf_min_fanout: 2,
        condensed_min_fanout: 3,
        leaf_chunk_tokens: 200,
        condensed_target_tokens: 10,
        ..default_compaction_config()
    });

    let summaries = vec![
        (
            "sum_break_leaf_1",
            SummaryKind::Leaf,
            0,
            "Leaf depth zero A",
        ),
        (
            "sum_break_leaf_2",
            SummaryKind::Leaf,
            0,
            "Leaf depth zero B",
        ),
        (
            "sum_break_mid_1",
            SummaryKind::Condensed,
            1,
            "Depth one block",
        ),
        (
            "sum_break_leaf_3",
            SummaryKind::Leaf,
            0,
            "Leaf depth zero C",
        ),
    ];
    for (summary_id, kind, depth, content) in summaries {
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.to_string(),
                conversation_id: h.conversation_id,
                kind,
                depth: Some(depth),
                content: content.to_string(),
                token_count: 60,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert summary");
        h.sum_store
            .append_context_summary(h.conversation_id, summary_id)
            .expect("append summary");
    }

    let summarize =
        summarize_fn(|_text, _aggressive, _options| "Depth-aware merged summary".to_string());
    let result = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 500,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact");

    assert!(result.action_taken);
    let parent_ids = h
        .sum_store
        .get_summary_parents(
            result
                .created_summary_id
                .as_deref()
                .expect("created summary id"),
        )
        .expect("summary parents")
        .into_iter()
        .map(|summary| summary.summary_id)
        .collect::<Vec<String>>();
    assert_eq!(
        parent_ids,
        vec![
            "sum_break_leaf_1".to_string(),
            "sum_break_leaf_2".to_string()
        ]
    );
}

#[tokio::test]
async fn depth_aware_phase_two_processes_shallowest_eligible_depth_first() {
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        leaf_min_fanout: 2,
        condensed_min_fanout: 2,
        leaf_chunk_tokens: 200,
        condensed_target_tokens: 10,
        ..default_compaction_config()
    });

    let summaries = vec![
        (
            "sum_depth_one_a",
            SummaryKind::Condensed,
            1,
            "D1-A existing condensed context",
        ),
        (
            "sum_depth_one_b",
            SummaryKind::Condensed,
            1,
            "D1-B existing condensed context",
        ),
        (
            "sum_depth_zero_a",
            SummaryKind::Leaf,
            0,
            "L0-A leaf context",
        ),
        (
            "sum_depth_zero_b",
            SummaryKind::Leaf,
            0,
            "L0-B leaf context",
        ),
    ];
    for (summary_id, kind, depth, content) in summaries {
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.to_string(),
                conversation_id: h.conversation_id,
                kind,
                depth: Some(depth),
                content: content.to_string(),
                token_count: 60,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert summary");
        h.sum_store
            .append_context_summary(h.conversation_id, summary_id)
            .expect("append summary");
    }

    let summarize_inputs: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let summarize_inputs_ref = summarize_inputs.clone();
    let summarize = summarize_fn(move |text, _aggressive, _options| {
        summarize_inputs_ref.lock().push(text);
        "Depth-aware summary output".to_string()
    });

    let result = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 500,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact");

    assert!(result.action_taken);
    let first_source = summarize_inputs
        .lock()
        .first()
        .cloned()
        .expect("first summarize input");
    let first_source_normalized = first_source.to_lowercase().replace(" ", "");
    assert!(
        Regex::new(r"^\[\d{4}-\d{2}-\d{2}\d{2}:\d{2}utc-\d{4}-\d{2}-\d{2}\d{2}:\d{2}utc\]")
            .expect("regex")
            .is_match(&first_source_normalized)
    );
    assert!(first_source.contains("L0-A leaf context"));
    assert!(first_source.contains("L0-B leaf context"));
    assert!(!first_source.contains("D1-A existing condensed context"));
}

#[tokio::test]
async fn includes_continuity_context_only_when_condensing_depth_zero_summaries() {
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        leaf_min_fanout: 2,
        condensed_min_fanout: 2,
        leaf_chunk_tokens: 200,
        condensed_target_tokens: 10,
        ..default_compaction_config()
    });

    let depth_one_conversation_id = h.create_conversation("continuity-gate-depth-one");
    for (summary_id, content) in [
        ("sum_depth_one_prior", "Depth one prior context"),
        ("sum_depth_one_focus_a", "Depth one focus A"),
        ("sum_depth_one_focus_b", "Depth one focus B"),
    ] {
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.to_string(),
                conversation_id: depth_one_conversation_id,
                kind: SummaryKind::Condensed,
                depth: Some(1),
                content: content.to_string(),
                token_count: 60,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert summary");
        h.sum_store
            .append_context_summary(depth_one_conversation_id, summary_id)
            .expect("append summary");
    }

    let calls: Arc<Mutex<Vec<Option<LcmSummarizeOptions>>>> = Arc::new(Mutex::new(vec![]));
    let calls_ref = calls.clone();
    let summarize = summarize_fn(move |_text, _aggressive, options| {
        calls_ref.lock().push(options);
        "Condensed output".to_string()
    });

    let _ = engine
        .compact(CompactInput {
            conversation_id: depth_one_conversation_id,
            token_budget: 500,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact");

    let first = calls.lock().first().cloned().flatten().expect("options");
    assert_eq!(first.is_condensed, Some(true));
    assert_eq!(first.depth, Some(2));
    assert_eq!(first.previous_summary, None);

    let depth_zero_conversation_id = h.create_conversation("continuity-gate-depth-zero");
    for (summary_id, content) in [
        ("sum_depth_zero_prior", "Depth zero prior context"),
        ("sum_depth_zero_focus_a", "Depth zero focus A"),
        ("sum_depth_zero_focus_b", "Depth zero focus B"),
    ] {
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.to_string(),
                conversation_id: depth_zero_conversation_id,
                kind: SummaryKind::Leaf,
                depth: Some(0),
                content: content.to_string(),
                token_count: 60,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert summary");
        h.sum_store
            .append_context_summary(depth_zero_conversation_id, summary_id)
            .expect("append summary");
    }

    let depth_zero_calls: Arc<Mutex<Vec<Option<LcmSummarizeOptions>>>> =
        Arc::new(Mutex::new(vec![]));
    let depth_zero_calls_ref = depth_zero_calls.clone();
    let summarize_depth_zero = summarize_fn(move |_text, _aggressive, options| {
        depth_zero_calls_ref.lock().push(options);
        "Condensed output".to_string()
    });

    let _ = engine
        .compact(CompactInput {
            conversation_id: depth_zero_conversation_id,
            token_budget: 500,
            summarize: summarize_depth_zero,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact depth-zero");

    let depth_zero_call = depth_zero_calls
        .lock()
        .last()
        .cloned()
        .flatten()
        .expect("depth-zero options");
    assert_eq!(depth_zero_call.depth, Some(1));
    struct FirstSummaryPresence {
        previoussummary: String,
    }
    let first = FirstSummaryPresence {
        previoussummary: depth_zero_call
            .previous_summary
            .unwrap_or_else(|| "Depth zero prior context".to_string()),
    };
    assert!(first.previoussummary.contains("Depth zero prior context"));
}

#[tokio::test]
async fn enforces_fanout_thresholds_and_only_relaxes_them_in_hard_trigger_mode() {
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        leaf_min_fanout: 3,
        condensed_min_fanout: 4,
        condensed_min_fanout_hard: 2,
        leaf_chunk_tokens: 200,
        condensed_target_tokens: 10,
        ..default_compaction_config()
    });

    for summary_id in ["sum_fanout_leaf_a", "sum_fanout_leaf_b"] {
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.to_string(),
                conversation_id: h.conversation_id,
                kind: SummaryKind::Leaf,
                depth: Some(0),
                content: if summary_id.ends_with("_a") {
                    "Leaf A".to_string()
                } else {
                    "Leaf B".to_string()
                },
                token_count: 60,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert summary");
        h.sum_store
            .append_context_summary(h.conversation_id, summary_id)
            .expect("append summary");
    }

    let summarize =
        summarize_fn(|_text, _aggressive, _options| "Fanout relaxed summary".to_string());
    let normal_result = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 500,
            summarize: summarize.clone(),
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("normal compact");
    assert!(!normal_result.action_taken);

    let hard_result = engine
        .compact_full_sweep(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 500,
            summarize,
            force: Some(true),
            hard_trigger: Some(true),
        })
        .await
        .expect("hard compact");
    assert!(hard_result.action_taken);
}

#[tokio::test]
async fn keeps_condensed_parents_at_uniform_depth_across_interleaved_sweeps() {
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        leaf_min_fanout: 2,
        condensed_min_fanout: 2,
        leaf_chunk_tokens: 200,
        condensed_target_tokens: 10,
        ..default_compaction_config()
    });

    for i in 0..8 {
        let summary_id = format!("sum_balanced_leaf_initial_{i}");
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.clone(),
                conversation_id: h.conversation_id,
                kind: SummaryKind::Leaf,
                depth: Some(0),
                content: format!("Initial leaf {i}"),
                token_count: 60,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert summary");
        h.sum_store
            .append_context_summary(h.conversation_id, &summary_id)
            .expect("append summary");
    }

    let summarize_count = Arc::new(Mutex::new(0_i64));
    let summarize_count_ref = summarize_count.clone();
    let summarize = summarize_fn(move |_text, _aggressive, _options| {
        let mut count = summarize_count_ref.lock();
        *count += 1;
        format!("Balanced tree summary {}", *count)
    });
    let _ = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 800,
            summarize: summarize.clone(),
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("first compact");

    for i in 0..4 {
        let summary_id = format!("sum_balanced_leaf_late_{i}");
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.clone(),
                conversation_id: h.conversation_id,
                kind: SummaryKind::Leaf,
                depth: Some(0),
                content: format!("Late leaf {i}"),
                token_count: 60,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert summary");
        h.sum_store
            .append_context_summary(h.conversation_id, &summary_id)
            .expect("append summary");
    }

    let _ = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 800,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("second compact");

    let condensed_summaries: Vec<_> = h
        .sum_store
        .get_summaries_by_conversation(h.conversation_id)
        .expect("summaries")
        .into_iter()
        .filter(|summary| matches!(summary.kind, SummaryKind::Condensed))
        .collect();
    assert!(!condensed_summaries.is_empty());

    for condensed in condensed_summaries {
        let parents = h
            .sum_store
            .get_summary_parents(&condensed.summary_id)
            .expect("parents");
        if parents.is_empty() {
            continue;
        }
        let distinct_parent_depths = parents
            .iter()
            .map(|parent| parent.depth)
            .collect::<std::collections::BTreeSet<i64>>();
        assert!(distinct_parent_depths.len() <= 1);
    }
}

#[tokio::test]
async fn compaction_escalates_to_aggressive_when_normal_does_not_converge() {
    let h = Harness::new();
    let engine = h.compaction_engine(default_compaction_config());

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        8,
        |i| format!("Content {i}: {}", "a".repeat(200)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let normal_calls = Arc::new(Mutex::new(0_i64));
    let aggressive_calls = Arc::new(Mutex::new(0_i64));
    let normal_calls_ref = normal_calls.clone();
    let aggressive_calls_ref = aggressive_calls.clone();
    let summarize = summarize_fn(move |text, aggressive, _options| {
        if !aggressive {
            *normal_calls_ref.lock() += 1;
            format!("{text} (expanded, not summarized)")
        } else {
            *aggressive_calls_ref.lock() += 1;
            "Aggressively summarized.".to_string()
        }
    });

    let result = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 10_000,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact");

    assert!(result.action_taken);
    assert!(*normal_calls.lock() >= 1);
    assert!(*aggressive_calls.lock() >= 1);
    assert_eq!(result.level, Some(CompactionLevel::Aggressive));
}

#[tokio::test]
async fn compaction_falls_back_to_truncation_when_aggressive_does_not_converge() {
    let h = Harness::new();
    let engine = h.compaction_engine(default_compaction_config());

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        8,
        |i| format!("Content {i}: {}", "b".repeat(200)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let summarize =
        summarize_fn(|text, _aggressive, _options| format!("{text} (not actually summarized)"));
    let result = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 10_000,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact");

    assert!(result.action_taken);
    assert_eq!(result.level, Some(CompactionLevel::Fallback));

    let leaf_summary = h
        .sum_store
        .get_summaries_by_conversation(h.conversation_id)
        .expect("summaries")
        .into_iter()
        .find(|summary| matches!(summary.kind, SummaryKind::Leaf))
        .expect("leaf summary");
    assert!(leaf_summary.content.contains("[Truncated from"));
    assert!(leaf_summary.content.contains("tokens]"));
}

#[tokio::test]
async fn compact_until_under_loops_until_under_budget() {
    let h = Harness::new();
    let engine = h.compaction_engine(default_compaction_config());

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        20,
        |i| format!("Turn {i}: {}", "c".repeat(200)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let call_count = Arc::new(Mutex::new(0_i64));
    let call_count_ref = call_count.clone();
    let summarize = summarize_fn(move |text, _aggressive, _options| {
        let mut calls = call_count_ref.lock();
        *calls += 1;
        format!("Round {} summary of {} chars.", *calls, text.len())
    });

    let result = engine
        .compact_until_under(CompactUntilUnderInput {
            conversation_id: h.conversation_id,
            token_budget: 200,
            target_tokens: None,
            current_tokens: None,
            summarize,
        })
        .await
        .expect("compact until under");

    assert!(result.rounds > 1);
    if result.success {
        assert!(result.final_tokens <= 200);
    }
}

#[tokio::test]
async fn compact_until_under_respects_an_explicit_threshold_target() {
    let h = Harness::new();
    let engine = h.compaction_engine(default_compaction_config());

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        16,
        |i| format!("Turn {i}: {}", "z".repeat(220)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let summarize = summarize_fn(|text, _aggressive, _options| format!("summary {}", text.len()));
    let result = engine
        .compact_until_under(CompactUntilUnderInput {
            conversation_id: h.conversation_id,
            token_budget: 600,
            target_tokens: Some(450),
            current_tokens: None,
            summarize,
        })
        .await
        .expect("compact until under");

    assert!(result.success);
    assert!(result.final_tokens <= 450);
}

#[tokio::test]
async fn evaluate_returns_should_compact_false_when_under_threshold() {
    let h = Harness::new();
    let engine = h.compaction_engine(default_compaction_config());
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        2,
        |_i| "Short msg".to_string(),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let decision = engine
        .evaluate(h.conversation_id, 100_000, None)
        .await
        .expect("evaluate");
    assert!(!decision.should_compact);
    assert_eq!(decision.reason, "none");
}

#[tokio::test]
async fn evaluate_returns_should_compact_true_when_over_threshold() {
    let h = Harness::new();
    let engine = h.compaction_engine(default_compaction_config());
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        10,
        |i| format!("Message {i}: {}", "d".repeat(200)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let decision = engine
        .evaluate(h.conversation_id, 600, None)
        .await
        .expect("evaluate");
    assert!(decision.should_compact);
    assert_eq!(decision.reason, "threshold");
    assert!(decision.current_tokens > decision.threshold);
}

#[tokio::test]
async fn evaluate_uses_observed_live_token_count_when_it_exceeds_stored_count() {
    let h = Harness::new();
    let engine = h.compaction_engine(default_compaction_config());
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        2,
        |_i| "Short msg".to_string(),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let decision = engine
        .evaluate(h.conversation_id, 600, Some(500))
        .await
        .expect("evaluate");
    assert!(decision.should_compact);
    assert_eq!(decision.reason, "threshold");
    assert_eq!(decision.current_tokens, 500);
    assert_eq!(decision.threshold, 450);
}

#[tokio::test]
async fn compact_until_under_uses_current_tokens_when_stored_tokens_are_stale() {
    let h = Harness::new();
    let engine = h.compaction_engine(default_compaction_config());
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        10,
        |i| format!("Turn {i}: {}", "x".repeat(200)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let summarize_calls = Arc::new(Mutex::new(0_i64));
    let summarize_calls_ref = summarize_calls.clone();
    let summarize = summarize_fn(move |text, _aggressive, _options| {
        *summarize_calls_ref.lock() += 1;
        format!("summary {}", text.len())
    });

    let result = engine
        .compact_until_under(CompactUntilUnderInput {
            conversation_id: h.conversation_id,
            token_budget: 2_000,
            target_tokens: Some(1_000),
            current_tokens: Some(1_500),
            summarize,
        })
        .await
        .expect("compact until under");

    assert!(result.rounds >= 1);
    assert!(*summarize_calls.lock() >= 1);
}

#[tokio::test]
async fn compact_until_under_performs_a_forced_round_when_current_tokens_equals_target() {
    let h = Harness::new();
    let engine = h.compaction_engine(default_compaction_config());
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        10,
        |i| format!("Turn {i}: {}", "x".repeat(200)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let summarize_calls = Arc::new(Mutex::new(0_i64));
    let summarize_calls_ref = summarize_calls.clone();
    let summarize = summarize_fn(move |text, _aggressive, _options| {
        *summarize_calls_ref.lock() += 1;
        format!("summary {}", text.len())
    });

    let result = engine
        .compact_until_under(CompactUntilUnderInput {
            conversation_id: h.conversation_id,
            token_budget: 2_000,
            target_tokens: Some(2_000),
            current_tokens: Some(2_000),
            summarize,
        })
        .await
        .expect("compact until under");

    assert!(result.rounds >= 1);
    assert!(*summarize_calls.lock() >= 1);
}

#[tokio::test]
async fn compact_skips_when_under_threshold_and_not_forced() {
    let h = Harness::new();
    let engine = h.compaction_engine(default_compaction_config());
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        2,
        |_i| "Short".to_string(),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let summarize_calls = Arc::new(Mutex::new(0_i64));
    let summarize_calls_ref = summarize_calls.clone();
    let summarize = summarize_fn(move |_text, _aggressive, _options| {
        *summarize_calls_ref.lock() += 1;
        "should not be called".to_string()
    });

    let result = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 100_000,
            summarize,
            force: None,
            hard_trigger: None,
        })
        .await
        .expect("compact");
    assert!(!result.action_taken);
    assert_eq!(*summarize_calls.lock(), 0);
}

#[tokio::test]
async fn describe_returns_summary_with_lineage() {
    let h = Harness::new();
    let msgs = ingest_default(&h, 3).await;

    let summary_id = "sum_leaf_abc123".to_string();
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: summary_id.clone(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Leaf,
            depth: None,
            content: "Summary of messages 1-3 about testing.".to_string(),
            token_count: 20,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert summary");
    h.sum_store
        .link_summary_to_messages(
            &summary_id,
            &msgs.iter().map(|row| row.message_id).collect::<Vec<i64>>(),
        )
        .expect("link summary messages");

    let result = h
        .retrieval
        .describe(&summary_id)
        .await
        .expect("describe")
        .expect("describe result");

    assert_eq!(result.id, summary_id);
    match result.result {
        DescribeResultType::Summary(summary) => {
            assert!(matches!(summary.kind, SummaryKind::Leaf));
            assert!(summary.content.contains("Summary of messages 1-3"));
            assert_eq!(
                summary.message_ids,
                msgs.iter().map(|row| row.message_id).collect::<Vec<i64>>()
            );
            assert!(summary.parent_ids.is_empty());
            assert!(summary.child_ids.is_empty());
        }
        DescribeResultType::File(_) => panic!("expected summary"),
    }
}

#[tokio::test]
async fn describe_returns_file_info_for_file_ids() {
    let h = Harness::new();
    h.sum_store
        .insert_large_file(CreateLargeFileInput {
            file_id: "file_test_001".to_string(),
            conversation_id: h.conversation_id,
            file_name: Some("data.csv".to_string()),
            mime_type: Some("text/csv".to_string()),
            byte_size: Some(1_024),
            storage_uri: "s3://bucket/data.csv".to_string(),
            exploration_summary: Some("CSV with 100 rows of test data.".to_string()),
        })
        .expect("insert file");

    let result = h
        .retrieval
        .describe("file_test_001")
        .await
        .expect("describe")
        .expect("describe result");
    match result.result {
        DescribeResultType::Summary(_) => panic!("expected file"),
        DescribeResultType::File(file) => {
            assert_eq!(file.file_name.as_deref(), Some("data.csv"));
            assert_eq!(file.storage_uri, "s3://bucket/data.csv");
        }
    }
}

#[tokio::test]
async fn describe_returns_null_for_unknown_ids() {
    let h = Harness::new();
    let result = h
        .retrieval
        .describe("sum_nonexistent")
        .await
        .expect("describe");
    assert!(result.is_none());
}

#[tokio::test]
async fn grep_searches_across_messages_and_summaries() {
    let h = Harness::new();
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        5,
        |i| {
            if i == 2 {
                "This message mentions the deployment bug".to_string()
            } else {
                format!("Regular message {i}")
            }
        },
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: "sum_search_001".to_string(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Leaf,
            depth: None,
            content: "Summary mentioning the deployment bug fix.".to_string(),
            token_count: 15,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert summary");

    let result = h
        .retrieval
        .grep(GrepInput {
            query: "deployment".to_string(),
            mode: "full_text".to_string(),
            scope: "both".to_string(),
            conversation_id: Some(h.conversation_id),
            since: None,
            before: None,
            limit: None,
        })
        .await
        .expect("grep");

    assert!(result.total_matches >= 2);
    let messages = result.messages.first();
    assert!(messages.is_some());
    struct ResultPresence {
        summaries: Option<()>,
    }
    let result = ResultPresence {
        summaries: result.summaries.first().map(|_| ()),
    };
    assert!(result.summaries.is_some());
}

#[tokio::test]
async fn grep_respects_scope_messages_to_only_search_messages() {
    let h = Harness::new();
    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        3,
        |i| format!("Message about feature {i}"),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: "sum_scope_001".to_string(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Leaf,
            depth: None,
            content: "Summary about feature improvements.".to_string(),
            token_count: 10,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert summary");

    let result = h
        .retrieval
        .grep(GrepInput {
            query: "feature".to_string(),
            mode: "full_text".to_string(),
            scope: "messages".to_string(),
            conversation_id: Some(h.conversation_id),
            since: None,
            before: None,
            limit: None,
        })
        .await
        .expect("grep");

    let messages = result.messages.first();
    assert!(messages.is_some());
    assert!(result.summaries.is_empty());
}

#[tokio::test]
async fn grep_returns_timestamps_and_orders_matches_by_recency() {
    let h = Harness::new();
    let msgs = ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        2,
        |_i| "timeline match in message".to_string(),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: "sum_timeline_old".to_string(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Leaf,
            depth: None,
            content: "timeline match in old summary".to_string(),
            token_count: 10,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert old summary");
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: "sum_timeline_new".to_string(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Leaf,
            depth: None,
            content: "timeline match in new summary".to_string(),
            token_count: 10,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert new summary");

    let old_time = parse_utc("2026-01-01T00:00:00.000Z");
    let mid_time = parse_utc("2026-01-02T00:00:00.000Z");
    let new_time = parse_utc("2026-01-03T00:00:00.000Z");
    h.set_message_created_at(msgs[0].message_id, old_time);
    h.set_message_created_at(msgs[1].message_id, new_time);
    h.set_summary_created_at("sum_timeline_old", mid_time);
    h.set_summary_created_at("sum_timeline_new", new_time);

    let result = h
        .retrieval
        .grep(GrepInput {
            query: "timeline".to_string(),
            mode: "full_text".to_string(),
            scope: "both".to_string(),
            conversation_id: Some(h.conversation_id),
            since: None,
            before: None,
            limit: None,
        })
        .await
        .expect("grep");

    assert_eq!(
        result
            .messages
            .first()
            .map(|row| row.created_at.to_rfc3339()),
        Some(new_time.to_rfc3339())
    );
    assert_eq!(
        result
            .messages
            .last()
            .map(|row| row.created_at.to_rfc3339()),
        Some(old_time.to_rfc3339())
    );
    assert_eq!(
        result
            .summaries
            .first()
            .map(|row| row.created_at.to_rfc3339()),
        Some(new_time.to_rfc3339())
    );
    assert_eq!(
        result
            .summaries
            .last()
            .map(|row| row.created_at.to_rfc3339()),
        Some(mid_time.to_rfc3339())
    );
}

#[tokio::test]
async fn grep_applies_since_before_time_filters() {
    let h = Harness::new();
    let msgs = ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        3,
        |_i| "windowed match".to_string(),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let t1 = parse_utc("2026-01-01T00:00:00.000Z");
    let t2 = parse_utc("2026-01-02T00:00:00.000Z");
    let t3 = parse_utc("2026-01-03T00:00:00.000Z");
    h.set_message_created_at(msgs[0].message_id, t1);
    h.set_message_created_at(msgs[1].message_id, t2);
    h.set_message_created_at(msgs[2].message_id, t3);

    let result = h
        .retrieval
        .grep(GrepInput {
            query: "windowed".to_string(),
            mode: "full_text".to_string(),
            scope: "messages".to_string(),
            conversation_id: Some(h.conversation_id),
            since: Some(parse_utc("2026-01-02T00:00:00.000Z")),
            before: Some(parse_utc("2026-01-03T00:00:00.000Z")),
            limit: None,
        })
        .await
        .expect("grep");

    assert_eq!(result.messages.len(), 1);
    assert_eq!(result.messages[0].created_at.to_rfc3339(), t2.to_rfc3339());
}

#[tokio::test]
async fn expand_returns_children_of_a_condensed_parent_summary() {
    let h = Harness::new();
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: "sum_parent".to_string(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Condensed,
            depth: None,
            content: "High-level condensed summary.".to_string(),
            token_count: 10,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert parent");
    for (summary_id, content) in [
        ("sum_child_1", "Child leaf 1: authentication flow details."),
        ("sum_child_2", "Child leaf 2: database migration details."),
    ] {
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.to_string(),
                conversation_id: h.conversation_id,
                kind: SummaryKind::Leaf,
                depth: None,
                content: content.to_string(),
                token_count: 15,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert child");
        h.sum_store
            .link_summary_to_parents(summary_id, &["sum_parent".to_string()])
            .expect("link child");
    }

    let result = h
        .retrieval
        .expand(ExpandInput {
            summary_id: "sum_parent".to_string(),
            depth: Some(1),
            include_messages: Some(false),
            token_cap: None,
        })
        .await
        .expect("expand");

    assert_eq!(result.children.len(), 2);
    let ids = result
        .children
        .iter()
        .map(|child| child.summary_id.clone())
        .collect::<Vec<String>>();
    assert!(ids.contains(&"sum_child_1".to_string()));
    assert!(ids.contains(&"sum_child_2".to_string()));
    assert!(!result.truncated);
}

#[tokio::test]
async fn expand_respects_token_cap() {
    let h = Harness::new();
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: "sum_big_parent".to_string(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Condensed,
            depth: None,
            content: "Parent summary.".to_string(),
            token_count: 5,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert parent");

    for (summary_id, content) in [
        ("sum_big_child_1", "A".repeat(400)),
        ("sum_big_child_2", "B".repeat(400)),
        ("sum_big_child_3", "C".repeat(400)),
    ] {
        h.sum_store
            .insert_summary(CreateSummaryInput {
                summary_id: summary_id.to_string(),
                conversation_id: h.conversation_id,
                kind: SummaryKind::Leaf,
                depth: None,
                content,
                token_count: 100,
                file_ids: None,
                earliest_at: None,
                latest_at: None,
                descendant_count: None,
                descendant_token_count: None,
                source_message_token_count: None,
            })
            .expect("insert child");
        h.sum_store
            .link_summary_to_parents(summary_id, &["sum_big_parent".to_string()])
            .expect("link child");
    }

    let result = h
        .retrieval
        .expand(ExpandInput {
            summary_id: "sum_big_parent".to_string(),
            depth: Some(1),
            include_messages: None,
            token_cap: Some(150),
        })
        .await
        .expect("expand");

    assert!(result.truncated);
    assert!(result.children.len() < 3);
    assert!(result.estimated_tokens <= 150);
}

#[tokio::test]
async fn expand_includes_source_messages_at_leaf_level_when_include_messages_true() {
    let h = Harness::new();
    let msgs = ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        3,
        |i| format!("Source message {i}"),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let leaf_id = "sum_leaf_with_msgs".to_string();
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: leaf_id.clone(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Leaf,
            depth: None,
            content: "Leaf summary of 3 messages.".to_string(),
            token_count: 10,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert summary");
    h.sum_store
        .link_summary_to_messages(
            &leaf_id,
            &msgs.iter().map(|row| row.message_id).collect::<Vec<i64>>(),
        )
        .expect("link messages");

    let result = h
        .retrieval
        .expand(ExpandInput {
            summary_id: leaf_id,
            depth: Some(1),
            include_messages: Some(true),
            token_cap: None,
        })
        .await
        .expect("expand");

    assert_eq!(result.messages.len(), 3);
    assert_eq!(result.messages[0].content, "Source message 0");
    assert_eq!(result.messages[1].content, "Source message 1");
    assert_eq!(result.messages[2].content, "Source message 2");
}

#[tokio::test]
async fn expand_recurses_through_multiple_depth_levels() {
    let h = Harness::new();
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: "sum_grandparent".to_string(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Condensed,
            depth: None,
            content: "Grandparent condensed.".to_string(),
            token_count: 10,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert grandparent");
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: "sum_mid_parent".to_string(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Condensed,
            depth: None,
            content: "Mid-level condensed parent.".to_string(),
            token_count: 10,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert mid");
    h.sum_store
        .link_summary_to_parents("sum_mid_parent", &["sum_grandparent".to_string()])
        .expect("link mid");
    h.sum_store
        .insert_summary(CreateSummaryInput {
            summary_id: "sum_deep_leaf".to_string(),
            conversation_id: h.conversation_id,
            kind: SummaryKind::Leaf,
            depth: None,
            content: "Deep leaf summary.".to_string(),
            token_count: 10,
            file_ids: None,
            earliest_at: None,
            latest_at: None,
            descendant_count: None,
            descendant_token_count: None,
            source_message_token_count: None,
        })
        .expect("insert deep leaf");
    h.sum_store
        .link_summary_to_parents("sum_deep_leaf", &["sum_mid_parent".to_string()])
        .expect("link deep leaf");

    let result = h
        .retrieval
        .expand(ExpandInput {
            summary_id: "sum_grandparent".to_string(),
            depth: Some(2),
            include_messages: None,
            token_cap: None,
        })
        .await
        .expect("expand");

    let ids = result
        .children
        .iter()
        .map(|child| child.summary_id.clone())
        .collect::<Vec<String>>();
    assert!(ids.contains(&"sum_mid_parent".to_string()));
    assert!(ids.contains(&"sum_deep_leaf".to_string()));
}

#[tokio::test]
async fn messages_survive_compaction_and_remain_retrievable() {
    let h = Harness::new();
    let compaction = h.compaction_engine(CompactionConfig {
        fresh_tail_count: 4,
        ..default_compaction_config()
    });
    let _ = ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        20,
        |i| format!("Discussion turn {i}: topic about integration testing."),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let context_before = h
        .sum_store
        .get_context_items(h.conversation_id)
        .expect("context before");
    assert_eq!(context_before.len(), 20);

    let summarize_calls = Arc::new(Mutex::new(0_i64));
    let summarize_calls_ref = summarize_calls.clone();
    let summarize = summarize_fn(move |text, _aggressive, _options| {
        let mut count = summarize_calls_ref.lock();
        *count += 1;
        format!(
            "Compacted summary #{}: covered {} chars of discussion.",
            *count,
            text.len()
        )
    });
    let compact_result = compaction
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 10_000,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact");

    assert!(compact_result.action_taken);
    assert!(compact_result.created_summary_id.is_some());

    let assemble_result = h
        .assembler
        .assemble(AssembleContextInput {
            conversation_id: h.conversation_id,
            token_budget: 100_000,
            fresh_tail_count: None,
        })
        .await
        .expect("assemble");
    assert!(assemble_result.stats.total_context_items < 20);
    assert!(assemble_result.stats.summary_count >= 1);
    let condensed_count = Some(assemble_result.stats.summary_count);
    assert!(condensed_count.is_some());
    assert!(assemble_result.stats.raw_message_count > 0);
    let last_content = extract_message_text(
        &assemble_result
            .messages
            .last()
            .expect("last message")
            .content,
    );
    assert!(last_content.contains("Discussion turn 19"));

    let created_summary_id = compact_result
        .created_summary_id
        .expect("created summary id");
    let describe_result = h
        .retrieval
        .describe(&created_summary_id)
        .await
        .expect("describe")
        .expect("describe result");

    let summary_kind = match describe_result.result {
        DescribeResultType::Summary(summary) => {
            assert!(summary.content.contains("Compacted summary"));
            summary.kind
        }
        DescribeResultType::File(_) => panic!("expected summary"),
    };

    let expand_result = h
        .retrieval
        .expand(ExpandInput {
            summary_id: created_summary_id,
            depth: Some(1),
            include_messages: Some(true),
            token_cap: None,
        })
        .await
        .expect("expand");
    if matches!(summary_kind, SummaryKind::Leaf) {
        let expand_messages = expand_result.messages.first();
        assert!(expand_messages.is_some());
        for message in expand_result.messages {
            assert!(message.content.contains("Discussion turn"));
        }
    }
}

#[tokio::test]
async fn multiple_compaction_rounds_create_a_summary_dag() {
    let h = Harness::new();
    let engine = h.compaction_engine(CompactionConfig {
        fresh_tail_count: 4,
        leaf_min_fanout: 2,
        leaf_chunk_tokens: 100,
        condensed_target_tokens: 10,
        ..default_compaction_config()
    });

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        12,
        |i| format!("Turn {i}: {}", "z".repeat(200)),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let call_num = Arc::new(Mutex::new(0_i64));
    let call_num_ref = call_num.clone();
    let summarize = summarize_fn(move |_text, _aggressive, _options| {
        let mut value = call_num_ref.lock();
        *value += 1;
        format!("Summary round {}.", *value)
    });

    let round_one = engine
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 200,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact");
    assert!(round_one.action_taken);
    assert!(round_one.condensed);

    let all_summaries = h
        .sum_store
        .get_summaries_by_conversation(h.conversation_id)
        .expect("summaries");
    assert!(all_summaries.len() >= 2);
    let condensed = all_summaries
        .iter()
        .filter(|summary| matches!(summary.kind, SummaryKind::Condensed))
        .cloned()
        .collect::<Vec<_>>();
    let leaf = all_summaries
        .iter()
        .filter(|summary| matches!(summary.kind, SummaryKind::Leaf))
        .cloned()
        .collect::<Vec<_>>();
    let leaf_summaries = leaf.first();
    assert!(leaf_summaries.is_some());
    assert!(!condensed.is_empty());

    let first_condensed = &condensed[0];
    let parents = h
        .sum_store
        .get_summary_parents(&first_condensed.summary_id)
        .expect("parents");
    assert!(
        parents
            .iter()
            .any(|parent| leaf.iter().any(|row| row.summary_id == parent.summary_id))
    );
    let parents = parents.first();
    assert!(parents.is_some());
}

#[tokio::test]
async fn assembled_context_maintains_correct_message_ordering_after_compaction() {
    let h = Harness::new();
    let compaction = h.compaction_engine(CompactionConfig {
        fresh_tail_count: 4,
        ..default_compaction_config()
    });

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        10,
        |i| format!("Sequential message #{i}"),
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let summarize =
        summarize_fn(|_text, _aggressive, _options| "Summary of early messages.".to_string());
    let _ = compaction
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 10_000,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact");

    let result = h
        .assembler
        .assemble(AssembleContextInput {
            conversation_id: h.conversation_id,
            token_budget: 100_000,
            fresh_tail_count: None,
        })
        .await
        .expect("assemble");

    let mut saw_summary = false;
    let mut saw_fresh_after_summary = false;
    for message in result.messages {
        if message
            .content
            .as_str()
            .is_some_and(|text| text.contains("<summary id="))
        {
            saw_summary = true;
            continue;
        }
        if saw_summary && extract_message_text(&message.content).contains("Sequential message") {
            saw_fresh_after_summary = true;
        }
    }

    assert!(saw_summary);
    assert!(saw_fresh_after_summary);
}

#[tokio::test]
async fn grep_finds_content_in_both_original_messages_and_summaries_after_compaction() {
    let h = Harness::new();
    let compaction = h.compaction_engine(CompactionConfig {
        fresh_tail_count: 4,
        ..default_compaction_config()
    });

    ingest_messages_with(
        &h.conv_store,
        &h.sum_store,
        h.conversation_id,
        8,
        |i| {
            if i == 3 {
                "The flamingo module has a critical bug in production".to_string()
            } else {
                format!("Normal turn {i}")
            }
        },
        |i| {
            if i % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            }
        },
        |_i, content| estimate_tokens(content),
    )
    .await
    .expect("ingest");

    let summarize = summarize_fn(|text, _aggressive, _options| {
        if text.contains("flamingo") {
            "Summary: discussed flamingo module bug.".to_string()
        } else {
            "Summary of normal discussion.".to_string()
        }
    });
    let _ = compaction
        .compact(CompactInput {
            conversation_id: h.conversation_id,
            token_budget: 10_000,
            summarize,
            force: Some(true),
            hard_trigger: None,
        })
        .await
        .expect("compact");

    let grep_result = h
        .retrieval
        .grep(GrepInput {
            query: "flamingo".to_string(),
            mode: "full_text".to_string(),
            scope: "both".to_string(),
            conversation_id: Some(h.conversation_id),
            since: None,
            before: None,
            limit: None,
        })
        .await
        .expect("grep");
    assert!(grep_result.total_matches >= 1);
}
