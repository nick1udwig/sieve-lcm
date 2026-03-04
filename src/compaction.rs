use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::anyhow;
use chrono::{DateTime, Utc};
use serde_json::json;

use crate::large_files::extract_file_ids_from_content;
use crate::store::conversation_store::{
    ConversationStore, CreateMessageInput, CreateMessagePartInput, MessagePartType, MessageRole,
};
use crate::store::summary_store::{
    ContextItemRecord, ContextItemType, CreateSummaryInput, SummaryKind, SummaryRecord, SummaryStore,
};
use crate::summarize::{LcmSummarizeFn, LcmSummarizeOptions};

const FALLBACK_MAX_CHARS: usize = 512 * 4;
const DEFAULT_LEAF_CHUNK_TOKENS: i64 = 20_000;
const CONDENSED_MIN_INPUT_RATIO: f64 = 0.1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompactionLevel {
    Normal,
    Aggressive,
    Fallback,
}

impl CompactionLevel {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Aggressive => "aggressive",
            Self::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CompactionPass {
    Leaf,
    Condensed,
}

impl CompactionPass {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Leaf => "leaf",
            Self::Condensed => "condensed",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactionDecision {
    pub should_compact: bool,
    pub reason: String,
    pub current_tokens: i64,
    pub threshold: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LeafTriggerDecision {
    pub should_compact: bool,
    pub raw_tokens_outside_tail: i64,
    pub threshold: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactionResult {
    pub action_taken: bool,
    pub tokens_before: i64,
    pub tokens_after: i64,
    pub created_summary_id: Option<String>,
    pub condensed: bool,
    pub level: Option<CompactionLevel>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactUntilUnderResult {
    pub success: bool,
    pub rounds: i64,
    pub final_tokens: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactionConfig {
    pub context_threshold: f64,
    pub fresh_tail_count: i64,
    pub leaf_min_fanout: i64,
    pub condensed_min_fanout: i64,
    pub condensed_min_fanout_hard: i64,
    pub incremental_max_depth: i64,
    pub leaf_chunk_tokens: i64,
    pub leaf_target_tokens: i64,
    pub condensed_target_tokens: i64,
    pub max_rounds: i64,
    pub timezone: Option<String>,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            context_threshold: 0.75,
            fresh_tail_count: 8,
            leaf_min_fanout: 8,
            condensed_min_fanout: 4,
            condensed_min_fanout_hard: 2,
            incremental_max_depth: 0,
            leaf_chunk_tokens: DEFAULT_LEAF_CHUNK_TOKENS,
            leaf_target_tokens: 600,
            condensed_target_tokens: 900,
            max_rounds: 10,
            timezone: Some("UTC".to_string()),
        }
    }
}

#[derive(Clone)]
pub struct CompactInput {
    pub conversation_id: i64,
    pub token_budget: i64,
    pub summarize: LcmSummarizeFn,
    pub force: Option<bool>,
    pub hard_trigger: Option<bool>,
}

#[derive(Clone)]
pub struct CompactLeafInput {
    pub conversation_id: i64,
    pub token_budget: i64,
    pub summarize: LcmSummarizeFn,
    pub force: Option<bool>,
    pub previous_summary_content: Option<String>,
}

#[derive(Clone)]
pub struct CompactUntilUnderInput {
    pub conversation_id: i64,
    pub token_budget: i64,
    pub target_tokens: Option<i64>,
    pub current_tokens: Option<i64>,
    pub summarize: LcmSummarizeFn,
}

#[derive(Clone, Debug)]
struct LeafChunkSelection {
    items: Vec<ContextItemRecord>,
}

#[derive(Clone, Debug)]
struct CondensedChunkSelection {
    items: Vec<ContextItemRecord>,
    summary_tokens: i64,
}

#[derive(Clone, Debug)]
struct CondensedPhaseCandidate {
    target_depth: i64,
    chunk: CondensedChunkSelection,
}

#[derive(Clone, Debug)]
struct LeafPassResult {
    summary_id: String,
    level: CompactionLevel,
    content: String,
}

#[derive(Clone, Debug)]
struct PassResult {
    summary_id: String,
    level: CompactionLevel,
}

#[derive(Clone, Debug)]
struct EventPassResult {
    summary_id: String,
    level: CompactionLevel,
}

#[derive(Clone, Debug)]
struct PersistEventsInput {
    conversation_id: i64,
    tokens_before: i64,
    tokens_after_leaf: i64,
    tokens_after_final: i64,
    leaf_result: Option<EventPassResult>,
    condense_result: Option<EventPassResult>,
}

#[derive(Clone, Debug)]
struct PersistEventInput {
    conversation_id: i64,
    session_id: String,
    pass: CompactionPass,
    level: CompactionLevel,
    tokens_before: i64,
    tokens_after: i64,
    created_summary_id: String,
    created_summary_ids: Vec<String>,
    condensed_pass_occurred: bool,
}

#[derive(Clone, Debug)]
struct LeafMessage {
    message_id: i64,
    content: String,
    created_at: DateTime<Utc>,
    token_count: i64,
}

#[derive(Clone)]
pub struct CompactionEngine {
    conversation_store: ConversationStore,
    summary_store: SummaryStore,
    config: CompactionConfig,
}

fn estimate_tokens(content: &str) -> i64 {
    ((content.chars().count() as f64) / 4.0).ceil() as i64
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    value.chars().take(max_chars).collect()
}

fn format_timestamp(value: &DateTime<Utc>, _timezone: Option<&str>) -> String {
    value.format("%Y-%m-%d %H:%M UTC").to_string()
}

fn generate_summary_id(content: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    nanos.hash(&mut hasher);
    format!("sum_{:016x}", hasher.finish())
}

fn dedupe_ordered_ids(ids: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    for id in ids {
        if seen.insert(id.clone()) {
            ordered.push(id);
        }
    }
    ordered
}

impl CompactionEngine {
    pub fn new(
        conversation_store: ConversationStore,
        summary_store: SummaryStore,
        config: CompactionConfig,
    ) -> Self {
        Self {
            conversation_store,
            summary_store,
            config,
        }
    }

    pub async fn evaluate(
        &self,
        conversation_id: i64,
        token_budget: i64,
        observed_token_count: Option<i64>,
    ) -> anyhow::Result<CompactionDecision> {
        let stored_tokens = self.summary_store.get_context_token_count(conversation_id)?;
        let live_tokens = observed_token_count.filter(|v| *v > 0).unwrap_or(0);
        let current_tokens = stored_tokens.max(live_tokens);
        let threshold = (self.config.context_threshold * token_budget as f64).floor() as i64;

        if current_tokens > threshold {
            return Ok(CompactionDecision {
                should_compact: true,
                reason: "threshold".to_string(),
                current_tokens,
                threshold,
            });
        }

        Ok(CompactionDecision {
            should_compact: false,
            reason: "none".to_string(),
            current_tokens,
            threshold,
        })
    }

    pub async fn evaluate_leaf_trigger(&self, conversation_id: i64) -> anyhow::Result<LeafTriggerDecision> {
        let raw_tokens_outside_tail = self.count_raw_tokens_outside_fresh_tail(conversation_id).await?;
        let threshold = self.resolve_leaf_chunk_tokens();
        Ok(LeafTriggerDecision {
            should_compact: raw_tokens_outside_tail >= threshold,
            raw_tokens_outside_tail,
            threshold,
        })
    }

    pub async fn compact(&self, input: CompactInput) -> anyhow::Result<CompactionResult> {
        self.compact_full_sweep(input).await
    }

    pub async fn compact_leaf(&self, input: CompactLeafInput) -> anyhow::Result<CompactionResult> {
        let force = input.force.unwrap_or(false);
        let tokens_before = self.summary_store.get_context_token_count(input.conversation_id)?;
        let threshold = (self.config.context_threshold * input.token_budget as f64).floor() as i64;
        let leaf_trigger = self.evaluate_leaf_trigger(input.conversation_id).await?;

        if !force && tokens_before <= threshold && !leaf_trigger.should_compact {
            return Ok(CompactionResult {
                action_taken: false,
                tokens_before,
                tokens_after: tokens_before,
                created_summary_id: None,
                condensed: false,
                level: None,
            });
        }

        let leaf_chunk = self.select_oldest_leaf_chunk(input.conversation_id).await?;
        if leaf_chunk.items.is_empty() {
            return Ok(CompactionResult {
                action_taken: false,
                tokens_before,
                tokens_after: tokens_before,
                created_summary_id: None,
                condensed: false,
                level: None,
            });
        }

        let previous_summary_content = match input.previous_summary_content {
            Some(value) => Some(value),
            None => self
                .resolve_prior_leaf_summary_context(input.conversation_id, &leaf_chunk.items)
                .await?,
        };

        let leaf_result = self
            .leaf_pass(
                input.conversation_id,
                leaf_chunk.items,
                &input.summarize,
                previous_summary_content,
            )
            .await?;

        let tokens_after_leaf = self.summary_store.get_context_token_count(input.conversation_id)?;
        self.persist_compaction_events(PersistEventsInput {
            conversation_id: input.conversation_id,
            tokens_before,
            tokens_after_leaf,
            tokens_after_final: tokens_after_leaf,
            leaf_result: Some(EventPassResult {
                summary_id: leaf_result.summary_id.clone(),
                level: leaf_result.level.clone(),
            }),
            condense_result: None,
        })
        .await;

        let mut tokens_after = tokens_after_leaf;
        let mut condensed = false;
        let mut created_summary_id = Some(leaf_result.summary_id.clone());
        let mut level = Some(leaf_result.level.clone());

        let incremental_max_depth = self.resolve_incremental_max_depth();
        let condensed_min_chunk_tokens = self.resolve_condensed_min_chunk_tokens();
        if incremental_max_depth > 0 {
            for target_depth in 0..incremental_max_depth {
                let fanout = self.resolve_fanout_for_depth(target_depth, false);
                let chunk = self
                    .select_oldest_chunk_at_depth(input.conversation_id, target_depth, None)
                    .await?;
                if chunk.items.len() < fanout as usize || chunk.summary_tokens < condensed_min_chunk_tokens {
                    break;
                }

                let pass_tokens_before = self.summary_store.get_context_token_count(input.conversation_id)?;
                let condense_result = self
                    .condensed_pass(
                        input.conversation_id,
                        chunk.items,
                        target_depth,
                        &input.summarize,
                    )
                    .await?;
                let pass_tokens_after = self.summary_store.get_context_token_count(input.conversation_id)?;
                self.persist_compaction_events(PersistEventsInput {
                    conversation_id: input.conversation_id,
                    tokens_before: pass_tokens_before,
                    tokens_after_leaf: pass_tokens_before,
                    tokens_after_final: pass_tokens_after,
                    leaf_result: None,
                    condense_result: Some(EventPassResult {
                        summary_id: condense_result.summary_id.clone(),
                        level: condense_result.level.clone(),
                    }),
                })
                .await;

                tokens_after = pass_tokens_after;
                condensed = true;
                created_summary_id = Some(condense_result.summary_id.clone());
                level = Some(condense_result.level.clone());

                if pass_tokens_after >= pass_tokens_before {
                    break;
                }
            }
        }

        Ok(CompactionResult {
            action_taken: true,
            tokens_before,
            tokens_after,
            created_summary_id,
            condensed,
            level,
        })
    }

    pub async fn compact_full_sweep(&self, input: CompactInput) -> anyhow::Result<CompactionResult> {
        let force = input.force.unwrap_or(false);
        let hard_trigger = input.hard_trigger.unwrap_or(false);

        let tokens_before = self.summary_store.get_context_token_count(input.conversation_id)?;
        let threshold = (self.config.context_threshold * input.token_budget as f64).floor() as i64;
        let leaf_trigger = self.evaluate_leaf_trigger(input.conversation_id).await?;

        if !force && tokens_before <= threshold && !leaf_trigger.should_compact {
            return Ok(CompactionResult {
                action_taken: false,
                tokens_before,
                tokens_after: tokens_before,
                created_summary_id: None,
                condensed: false,
                level: None,
            });
        }

        let context_items = self.summary_store.get_context_items(input.conversation_id)?;
        if context_items.is_empty() {
            return Ok(CompactionResult {
                action_taken: false,
                tokens_before,
                tokens_after: tokens_before,
                created_summary_id: None,
                condensed: false,
                level: None,
            });
        }

        let mut action_taken = false;
        let mut condensed = false;
        let mut created_summary_id = None;
        let mut level = None;
        let mut previous_summary_content: Option<String> = None;
        let mut previous_tokens = tokens_before;

        loop {
            let leaf_chunk = self.select_oldest_leaf_chunk(input.conversation_id).await?;
            if leaf_chunk.items.is_empty() {
                break;
            }

            let pass_tokens_before = self.summary_store.get_context_token_count(input.conversation_id)?;
            let leaf_result = self
                .leaf_pass(
                    input.conversation_id,
                    leaf_chunk.items,
                    &input.summarize,
                    previous_summary_content.clone(),
                )
                .await?;
            let pass_tokens_after = self.summary_store.get_context_token_count(input.conversation_id)?;
            self.persist_compaction_events(PersistEventsInput {
                conversation_id: input.conversation_id,
                tokens_before: pass_tokens_before,
                tokens_after_leaf: pass_tokens_after,
                tokens_after_final: pass_tokens_after,
                leaf_result: Some(EventPassResult {
                    summary_id: leaf_result.summary_id.clone(),
                    level: leaf_result.level.clone(),
                }),
                condense_result: None,
            })
            .await;

            action_taken = true;
            created_summary_id = Some(leaf_result.summary_id.clone());
            level = Some(leaf_result.level.clone());
            previous_summary_content = Some(leaf_result.content.clone());

            if pass_tokens_after >= pass_tokens_before || pass_tokens_after >= previous_tokens {
                break;
            }
            previous_tokens = pass_tokens_after;
        }

        loop {
            let candidate = self
                .select_shallowest_condensation_candidate(input.conversation_id, hard_trigger)
                .await?;
            let Some(candidate) = candidate else {
                break;
            };

            let pass_tokens_before = self.summary_store.get_context_token_count(input.conversation_id)?;
            let condense_result = self
                .condensed_pass(
                    input.conversation_id,
                    candidate.chunk.items,
                    candidate.target_depth,
                    &input.summarize,
                )
                .await?;
            let pass_tokens_after = self.summary_store.get_context_token_count(input.conversation_id)?;
            self.persist_compaction_events(PersistEventsInput {
                conversation_id: input.conversation_id,
                tokens_before: pass_tokens_before,
                tokens_after_leaf: pass_tokens_before,
                tokens_after_final: pass_tokens_after,
                leaf_result: None,
                condense_result: Some(EventPassResult {
                    summary_id: condense_result.summary_id.clone(),
                    level: condense_result.level.clone(),
                }),
            })
            .await;

            action_taken = true;
            condensed = true;
            created_summary_id = Some(condense_result.summary_id.clone());
            level = Some(condense_result.level.clone());

            if pass_tokens_after >= pass_tokens_before || pass_tokens_after >= previous_tokens {
                break;
            }
            previous_tokens = pass_tokens_after;
        }

        let tokens_after = self.summary_store.get_context_token_count(input.conversation_id)?;
        Ok(CompactionResult {
            action_taken,
            tokens_before,
            tokens_after,
            created_summary_id,
            condensed,
            level,
        })
    }

    pub async fn compact_until_under(
        &self,
        input: CompactUntilUnderInput,
    ) -> anyhow::Result<CompactUntilUnderResult> {
        let target_tokens = input
            .target_tokens
            .filter(|value| *value > 0)
            .unwrap_or(input.token_budget);

        let stored_tokens = self.summary_store.get_context_token_count(input.conversation_id)?;
        let live_tokens = input.current_tokens.filter(|v| *v > 0).unwrap_or(0);
        let mut last_tokens = stored_tokens.max(live_tokens);

        if last_tokens < target_tokens {
            return Ok(CompactUntilUnderResult {
                success: true,
                rounds: 0,
                final_tokens: last_tokens,
            });
        }

        let max_rounds = self.config.max_rounds.max(0);
        for round in 1..=max_rounds {
            let result = self
                .compact(CompactInput {
                    conversation_id: input.conversation_id,
                    token_budget: input.token_budget,
                    summarize: input.summarize.clone(),
                    force: Some(true),
                    hard_trigger: Some(false),
                })
                .await?;

            if result.tokens_after <= target_tokens {
                return Ok(CompactUntilUnderResult {
                    success: true,
                    rounds: round,
                    final_tokens: result.tokens_after,
                });
            }

            if !result.action_taken || result.tokens_after >= last_tokens {
                return Ok(CompactUntilUnderResult {
                    success: false,
                    rounds: round,
                    final_tokens: result.tokens_after,
                });
            }

            last_tokens = result.tokens_after;
        }

        let final_tokens = self.summary_store.get_context_token_count(input.conversation_id)?;
        Ok(CompactUntilUnderResult {
            success: final_tokens <= target_tokens,
            rounds: max_rounds,
            final_tokens,
        })
    }

    fn resolve_leaf_chunk_tokens(&self) -> i64 {
        if self.config.leaf_chunk_tokens > 0 {
            self.config.leaf_chunk_tokens
        } else {
            DEFAULT_LEAF_CHUNK_TOKENS
        }
    }

    fn resolve_fresh_tail_count(&self) -> i64 {
        if self.config.fresh_tail_count > 0 {
            self.config.fresh_tail_count
        } else {
            0
        }
    }

    fn resolve_leaf_min_fanout(&self) -> i64 {
        if self.config.leaf_min_fanout > 0 {
            self.config.leaf_min_fanout
        } else {
            8
        }
    }

    fn resolve_condensed_min_fanout(&self) -> i64 {
        if self.config.condensed_min_fanout > 0 {
            self.config.condensed_min_fanout
        } else {
            4
        }
    }

    fn resolve_condensed_min_fanout_hard(&self) -> i64 {
        if self.config.condensed_min_fanout_hard > 0 {
            self.config.condensed_min_fanout_hard
        } else {
            2
        }
    }

    fn resolve_incremental_max_depth(&self) -> i64 {
        if self.config.incremental_max_depth > 0 {
            self.config.incremental_max_depth
        } else {
            0
        }
    }

    fn resolve_fanout_for_depth(&self, target_depth: i64, hard_trigger: bool) -> i64 {
        if hard_trigger {
            return self.resolve_condensed_min_fanout_hard();
        }
        if target_depth == 0 {
            return self.resolve_leaf_min_fanout();
        }
        self.resolve_condensed_min_fanout()
    }

    fn resolve_condensed_min_chunk_tokens(&self) -> i64 {
        let chunk_target = self.resolve_leaf_chunk_tokens();
        let ratio_floor = (chunk_target as f64 * CONDENSED_MIN_INPUT_RATIO).floor() as i64;
        self.config.condensed_target_tokens.max(ratio_floor)
    }

    fn resolve_fresh_tail_ordinal(&self, context_items: &[ContextItemRecord]) -> i64 {
        let fresh_tail_count = self.resolve_fresh_tail_count();
        if fresh_tail_count <= 0 {
            return i64::MAX;
        }

        let raw_message_items: Vec<&ContextItemRecord> = context_items
            .iter()
            .filter(|item| matches!(item.item_type, ContextItemType::Message) && item.message_id.is_some())
            .collect();
        if raw_message_items.is_empty() {
            return i64::MAX;
        }

        let tail_start_idx = raw_message_items
            .len()
            .saturating_sub(fresh_tail_count as usize);
        raw_message_items
            .get(tail_start_idx)
            .map(|item| item.ordinal)
            .unwrap_or(i64::MAX)
    }

    fn resolve_message_token_count(&self, content: &str, token_count: i64) -> i64 {
        if token_count > 0 {
            token_count
        } else {
            estimate_tokens(content)
        }
    }

    fn resolve_summary_token_count(&self, summary: &SummaryRecord) -> i64 {
        if summary.token_count > 0 {
            summary.token_count
        } else {
            estimate_tokens(&summary.content)
        }
    }

    async fn get_message_token_count(&self, message_id: i64) -> anyhow::Result<i64> {
        let Some(message) = self.conversation_store.get_message_by_id(message_id)? else {
            return Ok(0);
        };
        Ok(self.resolve_message_token_count(&message.content, message.token_count))
    }

    async fn count_raw_tokens_outside_fresh_tail(&self, conversation_id: i64) -> anyhow::Result<i64> {
        let context_items = self.summary_store.get_context_items(conversation_id)?;
        let fresh_tail_ordinal = self.resolve_fresh_tail_ordinal(&context_items);
        let mut raw_tokens = 0_i64;

        for item in context_items {
            if item.ordinal >= fresh_tail_ordinal {
                break;
            }
            if !matches!(item.item_type, ContextItemType::Message) {
                continue;
            }
            let Some(message_id) = item.message_id else {
                continue;
            };
            raw_tokens += self.get_message_token_count(message_id).await?;
        }

        Ok(raw_tokens)
    }

    async fn select_oldest_leaf_chunk(&self, conversation_id: i64) -> anyhow::Result<LeafChunkSelection> {
        let context_items = self.summary_store.get_context_items(conversation_id)?;
        let fresh_tail_ordinal = self.resolve_fresh_tail_ordinal(&context_items);
        let threshold = self.resolve_leaf_chunk_tokens();

        let mut chunk = Vec::new();
        let mut chunk_tokens = 0_i64;
        let mut started = false;

        for item in context_items {
            if item.ordinal >= fresh_tail_ordinal {
                break;
            }

            let item_is_message =
                matches!(item.item_type, ContextItemType::Message) && item.message_id.is_some();

            if !started {
                if !item_is_message {
                    continue;
                }
                started = true;
            } else if !item_is_message {
                break;
            }

            let Some(message_id) = item.message_id else {
                continue;
            };

            let message_tokens = self.get_message_token_count(message_id).await?;
            if !chunk.is_empty() && chunk_tokens + message_tokens > threshold {
                break;
            }

            chunk_tokens += message_tokens;
            chunk.push(item);

            if chunk_tokens >= threshold {
                break;
            }
        }

        Ok(LeafChunkSelection { items: chunk })
    }

    async fn resolve_prior_leaf_summary_context(
        &self,
        conversation_id: i64,
        message_items: &[ContextItemRecord],
    ) -> anyhow::Result<Option<String>> {
        let Some(start_ordinal) = message_items.iter().map(|item| item.ordinal).min() else {
            return Ok(None);
        };

        let mut prior_summary_items: Vec<ContextItemRecord> = self
            .summary_store
            .get_context_items(conversation_id)?
            .into_iter()
            .filter(|item| {
                item.ordinal < start_ordinal
                    && matches!(item.item_type, ContextItemType::Summary)
                    && item.summary_id.is_some()
            })
            .collect();

        if prior_summary_items.is_empty() {
            return Ok(None);
        }

        let start_idx = prior_summary_items.len().saturating_sub(2);
        prior_summary_items = prior_summary_items.into_iter().skip(start_idx).collect();

        let mut summary_contents = Vec::new();
        for item in prior_summary_items {
            let Some(summary_id) = item.summary_id else {
                continue;
            };
            let Some(summary) = self.summary_store.get_summary(&summary_id)? else {
                continue;
            };
            let content = summary.content.trim();
            if !content.is_empty() {
                summary_contents.push(content.to_string());
            }
        }

        if summary_contents.is_empty() {
            return Ok(None);
        }

        Ok(Some(summary_contents.join("\n\n")))
    }

    async fn select_shallowest_condensation_candidate(
        &self,
        conversation_id: i64,
        hard_trigger: bool,
    ) -> anyhow::Result<Option<CondensedPhaseCandidate>> {
        let context_items = self.summary_store.get_context_items(conversation_id)?;
        let fresh_tail_ordinal = self.resolve_fresh_tail_ordinal(&context_items);
        let min_chunk_tokens = self.resolve_condensed_min_chunk_tokens();
        let depth_levels = self.summary_store.get_distinct_depths_in_context(
            conversation_id,
            if fresh_tail_ordinal == i64::MAX {
                None
            } else {
                Some(fresh_tail_ordinal)
            },
        )?;

        for target_depth in depth_levels {
            let fanout = self.resolve_fanout_for_depth(target_depth, hard_trigger);
            let chunk = self
                .select_oldest_chunk_at_depth(conversation_id, target_depth, Some(fresh_tail_ordinal))
                .await?;
            if chunk.items.len() < fanout as usize {
                continue;
            }
            if chunk.summary_tokens < min_chunk_tokens {
                continue;
            }
            return Ok(Some(CondensedPhaseCandidate {
                target_depth,
                chunk,
            }));
        }

        Ok(None)
    }

    async fn select_oldest_chunk_at_depth(
        &self,
        conversation_id: i64,
        target_depth: i64,
        fresh_tail_ordinal_override: Option<i64>,
    ) -> anyhow::Result<CondensedChunkSelection> {
        let context_items = self.summary_store.get_context_items(conversation_id)?;
        let fresh_tail_ordinal = fresh_tail_ordinal_override
            .unwrap_or_else(|| self.resolve_fresh_tail_ordinal(&context_items));
        let chunk_token_budget = self.resolve_leaf_chunk_tokens();

        let mut chunk = Vec::new();
        let mut summary_tokens = 0_i64;

        for item in context_items {
            if item.ordinal >= fresh_tail_ordinal {
                break;
            }

            if !matches!(item.item_type, ContextItemType::Summary) || item.summary_id.is_none() {
                if !chunk.is_empty() {
                    break;
                }
                continue;
            }

            let Some(summary_id) = item.summary_id.clone() else {
                if !chunk.is_empty() {
                    break;
                }
                continue;
            };

            let Some(summary) = self.summary_store.get_summary(&summary_id)? else {
                if !chunk.is_empty() {
                    break;
                }
                continue;
            };

            if summary.depth != target_depth {
                if !chunk.is_empty() {
                    break;
                }
                continue;
            }

            let token_count = self.resolve_summary_token_count(&summary);
            if !chunk.is_empty() && summary_tokens + token_count > chunk_token_budget {
                break;
            }

            summary_tokens += token_count;
            chunk.push(item);

            if summary_tokens >= chunk_token_budget {
                break;
            }
        }

        Ok(CondensedChunkSelection {
            items: chunk,
            summary_tokens,
        })
    }

    async fn resolve_prior_summary_context_at_depth(
        &self,
        conversation_id: i64,
        summary_items: &[ContextItemRecord],
        target_depth: i64,
    ) -> anyhow::Result<Option<String>> {
        let Some(start_ordinal) = summary_items.iter().map(|item| item.ordinal).min() else {
            return Ok(None);
        };

        let mut prior_summary_items: Vec<ContextItemRecord> = self
            .summary_store
            .get_context_items(conversation_id)?
            .into_iter()
            .filter(|item| {
                item.ordinal < start_ordinal
                    && matches!(item.item_type, ContextItemType::Summary)
                    && item.summary_id.is_some()
            })
            .collect();
        if prior_summary_items.is_empty() {
            return Ok(None);
        }

        let start_idx = prior_summary_items.len().saturating_sub(4);
        prior_summary_items = prior_summary_items.into_iter().skip(start_idx).collect();

        let mut summary_contents = Vec::new();
        for item in prior_summary_items {
            let Some(summary_id) = item.summary_id else {
                continue;
            };
            let Some(summary) = self.summary_store.get_summary(&summary_id)? else {
                continue;
            };
            if summary.depth != target_depth {
                continue;
            }
            let content = summary.content.trim();
            if !content.is_empty() {
                summary_contents.push(content.to_string());
            }
        }

        if summary_contents.is_empty() {
            return Ok(None);
        }

        let start_idx = summary_contents.len().saturating_sub(2);
        Ok(Some(
            summary_contents
                .into_iter()
                .skip(start_idx)
                .collect::<Vec<String>>()
                .join("\n\n"),
        ))
    }

    async fn summarize_with_escalation(
        &self,
        source_text: &str,
        summarize: &LcmSummarizeFn,
        options: Option<LcmSummarizeOptions>,
    ) -> (String, CompactionLevel) {
        let source_text = source_text.trim();
        if source_text.is_empty() {
            return ("[Truncated from 0 tokens]".to_string(), CompactionLevel::Fallback);
        }

        let input_tokens = estimate_tokens(source_text).max(1);
        let mut summary_text = (summarize)(source_text.to_string(), false, options.clone()).await;
        let mut level = CompactionLevel::Normal;

        if estimate_tokens(&summary_text) >= input_tokens {
            summary_text = (summarize)(source_text.to_string(), true, options.clone()).await;
            level = CompactionLevel::Aggressive;

            if estimate_tokens(&summary_text) >= input_tokens {
                let truncated = truncate_chars(source_text, FALLBACK_MAX_CHARS);
                summary_text = format!("{}\n[Truncated from {} tokens]", truncated, input_tokens);
                level = CompactionLevel::Fallback;
            }
        }

        (summary_text, level)
    }

    async fn leaf_pass(
        &self,
        conversation_id: i64,
        message_items: Vec<ContextItemRecord>,
        summarize: &LcmSummarizeFn,
        previous_summary_content: Option<String>,
    ) -> anyhow::Result<LeafPassResult> {
        if message_items.is_empty() {
            return Err(anyhow!("leaf pass requires non-empty message items"));
        }

        let mut message_contents = Vec::new();
        for item in &message_items {
            let Some(message_id) = item.message_id else {
                continue;
            };
            let Some(message) = self.conversation_store.get_message_by_id(message_id)? else {
                continue;
            };
            message_contents.push(LeafMessage {
                message_id: message.message_id,
                content: message.content.clone(),
                created_at: message.created_at,
                token_count: self.resolve_message_token_count(&message.content, message.token_count),
            });
        }

        let concatenated = message_contents
            .iter()
            .map(|message| {
                format!(
                    "[{}]\n{}",
                    format_timestamp(&message.created_at, self.config.timezone.as_deref()),
                    message.content
                )
            })
            .collect::<Vec<String>>()
            .join("\n\n");

        let mut file_id_candidates = Vec::new();
        for message in &message_contents {
            file_id_candidates.extend(extract_file_ids_from_content(&message.content));
        }
        let file_ids = dedupe_ordered_ids(file_id_candidates);

        let (summary_content, level) = self
            .summarize_with_escalation(
                &concatenated,
                summarize,
                Some(LcmSummarizeOptions {
                    previous_summary: previous_summary_content,
                    is_condensed: Some(false),
                    depth: None,
                }),
            )
            .await;

        let summary_id = generate_summary_id(&summary_content);
        let token_count = estimate_tokens(&summary_content);

        let earliest_at = message_contents
            .iter()
            .map(|message| &message.created_at)
            .min()
            .cloned();
        let latest_at = message_contents
            .iter()
            .map(|message| &message.created_at)
            .max()
            .cloned();
        let source_message_token_count: i64 =
            message_contents.iter().map(|message| message.token_count.max(0)).sum();

        self.summary_store.insert_summary(CreateSummaryInput {
            summary_id: summary_id.clone(),
            conversation_id,
            kind: SummaryKind::Leaf,
            depth: Some(0),
            content: summary_content.clone(),
            token_count,
            file_ids: Some(file_ids),
            earliest_at,
            latest_at,
            descendant_count: Some(0),
            descendant_token_count: Some(0),
            source_message_token_count: Some(source_message_token_count),
        })?;

        let message_ids: Vec<i64> = message_contents.iter().map(|message| message.message_id).collect();
        self.summary_store
            .link_summary_to_messages(&summary_id, &message_ids)?;

        let Some(start_ordinal) = message_items.iter().map(|item| item.ordinal).min() else {
            return Err(anyhow!("leaf pass missing start ordinal"));
        };
        let Some(end_ordinal) = message_items.iter().map(|item| item.ordinal).max() else {
            return Err(anyhow!("leaf pass missing end ordinal"));
        };

        self.summary_store.replace_context_range_with_summary(
            conversation_id,
            start_ordinal,
            end_ordinal,
            &summary_id,
        )?;

        Ok(LeafPassResult {
            summary_id,
            level,
            content: summary_content,
        })
    }

    async fn condensed_pass(
        &self,
        conversation_id: i64,
        summary_items: Vec<ContextItemRecord>,
        target_depth: i64,
        summarize: &LcmSummarizeFn,
    ) -> anyhow::Result<PassResult> {
        if summary_items.is_empty() {
            return Err(anyhow!("condensed pass requires non-empty summary items"));
        }

        let mut summary_records = Vec::new();
        for item in &summary_items {
            let Some(summary_id) = item.summary_id.as_deref() else {
                continue;
            };
            let Some(summary) = self.summary_store.get_summary(summary_id)? else {
                continue;
            };
            summary_records.push(summary);
        }

        let concatenated = summary_records
            .iter()
            .map(|summary| {
                let earliest_at = summary.earliest_at.as_ref().unwrap_or(&summary.created_at);
                let latest_at = summary.latest_at.as_ref().unwrap_or(&summary.created_at);
                format!(
                    "[{} - {}]\n{}",
                    format_timestamp(earliest_at, self.config.timezone.as_deref()),
                    format_timestamp(latest_at, self.config.timezone.as_deref()),
                    summary.content
                )
            })
            .collect::<Vec<String>>()
            .join("\n\n");

        let mut file_id_candidates = Vec::new();
        for summary in &summary_records {
            file_id_candidates.extend(summary.file_ids.iter().cloned());
            file_id_candidates.extend(extract_file_ids_from_content(&summary.content));
        }
        let file_ids = dedupe_ordered_ids(file_id_candidates);

        let previous_summary_content = if target_depth == 0 {
            self.resolve_prior_summary_context_at_depth(conversation_id, &summary_items, target_depth)
                .await?
        } else {
            None
        };

        let (summary_content, level) = self
            .summarize_with_escalation(
                &concatenated,
                summarize,
                Some(LcmSummarizeOptions {
                    previous_summary: previous_summary_content,
                    is_condensed: Some(true),
                    depth: Some(target_depth + 1),
                }),
            )
            .await;

        let summary_id = generate_summary_id(&summary_content);
        let token_count = estimate_tokens(&summary_content);

        let earliest_at = summary_records
            .iter()
            .map(|summary| summary.earliest_at.as_ref().unwrap_or(&summary.created_at))
            .min()
            .cloned();
        let latest_at = summary_records
            .iter()
            .map(|summary| summary.latest_at.as_ref().unwrap_or(&summary.created_at))
            .max()
            .cloned();
        let descendant_count: i64 = summary_records
            .iter()
            .map(|summary| summary.descendant_count.max(0) + 1)
            .sum();
        let descendant_token_count: i64 = summary_records
            .iter()
            .map(|summary| summary.descendant_token_count.max(0) + summary.token_count.max(0))
            .sum();
        let source_message_token_count: i64 = summary_records
            .iter()
            .map(|summary| summary.source_message_token_count.max(0))
            .sum();

        self.summary_store.insert_summary(CreateSummaryInput {
            summary_id: summary_id.clone(),
            conversation_id,
            kind: SummaryKind::Condensed,
            depth: Some(target_depth + 1),
            content: summary_content,
            token_count,
            file_ids: Some(file_ids),
            earliest_at,
            latest_at,
            descendant_count: Some(descendant_count),
            descendant_token_count: Some(descendant_token_count),
            source_message_token_count: Some(source_message_token_count),
        })?;

        let parent_summary_ids: Vec<String> = summary_records
            .iter()
            .map(|summary| summary.summary_id.clone())
            .collect();
        self.summary_store
            .link_summary_to_parents(&summary_id, &parent_summary_ids)?;

        let Some(start_ordinal) = summary_items.iter().map(|item| item.ordinal).min() else {
            return Err(anyhow!("condensed pass missing start ordinal"));
        };
        let Some(end_ordinal) = summary_items.iter().map(|item| item.ordinal).max() else {
            return Err(anyhow!("condensed pass missing end ordinal"));
        };

        self.summary_store.replace_context_range_with_summary(
            conversation_id,
            start_ordinal,
            end_ordinal,
            &summary_id,
        )?;

        Ok(PassResult { summary_id, level })
    }

    async fn persist_compaction_events(&self, input: PersistEventsInput) {
        if input.leaf_result.is_none() && input.condense_result.is_none() {
            return;
        }

        let Ok(conversation) = self.conversation_store.get_conversation(input.conversation_id) else {
            return;
        };
        let Some(conversation) = conversation else {
            return;
        };

        let mut created_summary_ids = Vec::new();
        if let Some(result) = &input.leaf_result {
            created_summary_ids.push(result.summary_id.clone());
        }
        if let Some(result) = &input.condense_result {
            created_summary_ids.push(result.summary_id.clone());
        }
        let condensed_pass_occurred = input.condense_result.is_some();

        if let Some(leaf_result) = input.leaf_result {
            self.persist_compaction_event(PersistEventInput {
                conversation_id: input.conversation_id,
                session_id: conversation.session_id.clone(),
                pass: CompactionPass::Leaf,
                level: leaf_result.level,
                tokens_before: input.tokens_before,
                tokens_after: input.tokens_after_leaf,
                created_summary_id: leaf_result.summary_id,
                created_summary_ids: created_summary_ids.clone(),
                condensed_pass_occurred,
            })
            .await;
        }

        if let Some(condense_result) = input.condense_result {
            self.persist_compaction_event(PersistEventInput {
                conversation_id: input.conversation_id,
                session_id: conversation.session_id,
                pass: CompactionPass::Condensed,
                level: condense_result.level,
                tokens_before: input.tokens_after_leaf,
                tokens_after: input.tokens_after_final,
                created_summary_id: condense_result.summary_id,
                created_summary_ids,
                condensed_pass_occurred,
            })
            .await;
        }
    }

    async fn persist_compaction_event(&self, input: PersistEventInput) {
        let content = format!(
            "LCM compaction {} pass ({}): {} -> {}",
            input.pass.as_str(),
            input.level.as_str(),
            input.tokens_before,
            input.tokens_after
        );
        let metadata = json!({
            "conversationId": input.conversation_id,
            "pass": input.pass.as_str(),
            "level": input.level.as_str(),
            "tokensBefore": input.tokens_before,
            "tokensAfter": input.tokens_after,
            "createdSummaryId": input.created_summary_id,
            "createdSummaryIds": input.created_summary_ids,
            "condensedPassOccurred": input.condensed_pass_occurred,
        })
        .to_string();

        let seq = match self.conversation_store.get_max_seq(input.conversation_id) {
            Ok(max_seq) => max_seq + 1,
            Err(_) => return,
        };

        let event_message = match self.conversation_store.create_message(CreateMessageInput {
            conversation_id: input.conversation_id,
            seq,
            role: MessageRole::System,
            content: content.clone(),
            token_count: estimate_tokens(&content),
        }) {
            Ok(value) => value,
            Err(_) => return,
        };

        let parts = vec![CreateMessagePartInput {
            session_id: input.session_id,
            part_type: MessagePartType::Compaction,
            ordinal: 0,
            text_content: Some(content),
            tool_call_id: None,
            tool_name: None,
            tool_input: None,
            tool_output: None,
            metadata: Some(metadata),
        }];
        let _ = self
            .conversation_store
            .create_message_parts(event_message.message_id, &parts);
    }
}
