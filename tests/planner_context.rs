use std::sync::Arc;

use sieve_lcm::compaction::{CompactionConfig, CompactionLevel};
use sieve_lcm::planner_context::PlannerLaneMemory;
use sieve_lcm::store::conversation_store::MessageRole;
use sieve_lcm::summarize::{LcmSummarizeFn, LcmSummarizeOptions};
use tempfile::TempDir;

fn test_db_path(temp_dir: &TempDir, name: &str) -> String {
    temp_dir.path().join(name).to_string_lossy().to_string()
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

fn compaction_config() -> CompactionConfig {
    CompactionConfig {
        context_threshold: 0.0,
        fresh_tail_count: 2,
        leaf_min_fanout: 2,
        condensed_min_fanout: 2,
        condensed_min_fanout_hard: 2,
        incremental_max_depth: 0,
        leaf_chunk_tokens: 4,
        leaf_target_tokens: 4,
        condensed_target_tokens: 4,
        max_rounds: 4,
        timezone: Some("UTC".to_string()),
    }
}

#[tokio::test]
async fn planner_context_trusted_lane_returns_raw_messages() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let lane = PlannerLaneMemory::open(
        &test_db_path(&temp_dir, "trusted.db"),
        compaction_config(),
        Some("UTC".to_string()),
    )
    .expect("open trusted lane");

    lane.ingest_text_message("main", MessageRole::User, "I live in Livermore ca")
        .expect("ingest user");

    let assembled = lane
        .assemble_trusted_context("main", 1_000)
        .await
        .expect("assemble trusted");
    let encoded =
        serde_json::to_string(&assembled.messages).expect("serialize trusted assembled messages");

    assert!(encoded.contains("Livermore ca"));
}

#[tokio::test]
async fn planner_context_untrusted_lane_returns_only_opaque_refs() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let lane = PlannerLaneMemory::open(
        &test_db_path(&temp_dir, "untrusted.db"),
        compaction_config(),
        Some("UTC".to_string()),
    )
    .expect("open untrusted lane");

    lane.ingest_text_message(
        "main",
        MessageRole::Assistant,
        "secret raw assistant string should never appear in planner context",
    )
    .expect("ingest assistant");

    let refs = lane
        .assemble_opaque_refs("main", 1_000)
        .expect("assemble opaque refs");
    let encoded = serde_json::to_string(&refs).expect("serialize opaque refs");

    assert_eq!(refs.len(), 1);
    assert!(encoded.contains("lcm:untrusted:message:"));
    assert!(!encoded.contains("secret raw assistant string"));
}

#[tokio::test]
async fn planner_context_untrusted_lane_stays_opaque_after_compaction() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let lane = PlannerLaneMemory::open(
        &test_db_path(&temp_dir, "untrusted-compact.db"),
        compaction_config(),
        Some("UTC".to_string()),
    )
    .expect("open untrusted lane");

    lane.ingest_text_message("main", MessageRole::Assistant, "alpha raw tool analysis")
        .expect("ingest a");
    lane.ingest_text_message("main", MessageRole::Assistant, "beta raw tool analysis")
        .expect("ingest b");
    lane.ingest_text_message("main", MessageRole::Assistant, "gamma raw tool analysis")
        .expect("ingest c");

    let result = lane
        .compact_session(
            "main",
            8,
            summarize_fn(|_text, _aggressive, _options| "opaque compacted summary".to_string()),
        )
        .await
        .expect("compact lane");

    assert!(result.action_taken);
    assert_eq!(result.level, Some(CompactionLevel::Normal));

    let refs = lane
        .assemble_opaque_refs("main", 1_000)
        .expect("assemble opaque refs after compaction");
    let encoded = serde_json::to_string(&refs).expect("serialize opaque refs");

    assert!(
        refs.iter()
            .any(|entry| entry.reference.contains("lcm:untrusted:summary:")),
        "expected compacted summary ref, got {refs:?}"
    );
    assert!(!encoded.contains("alpha raw tool analysis"));
    assert!(!encoded.contains("opaque compacted summary"));
}
