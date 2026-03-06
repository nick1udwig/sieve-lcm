use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use async_trait::async_trait;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::assembler::{AssembleContextInput, ContextAssembler};
use crate::compaction::{
    CompactInput as CompactionRunInput, CompactLeafInput as CompactionLeafInput,
    CompactUntilUnderInput as CompactionUntilInput, CompactionEngine, LeafTriggerDecision,
};
use crate::db::config::LcmConfig;
use crate::db::connection::get_lcm_connection;
use crate::db::migration::run_lcm_migrations;
use crate::large_files::{
    ExplorationSummaryInput, extension_from_name_or_mime, format_file_reference,
    generate_exploration_summary, is_large_file, parse_file_blocks,
};
use crate::retrieval::{RetrievalApi, RetrievalEngine};
use crate::store::conversation_store::{
    ConversationRecord, ConversationStore, CreateMessageInput, CreateMessagePartInput,
    MessagePartType, MessageRole,
};
use crate::store::summary_store::{CreateLargeFileInput, SummaryStore};
use crate::summarize::{
    LcmSummarizeFn, LcmSummarizerLegacyParams, create_lcm_summarize_from_legacy_params,
};
use crate::types::{AgentMessage, LcmDependencies};

#[derive(Clone, Debug, PartialEq)]
pub struct ContextEngineInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub owns_compaction: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BootstrapResult {
    pub bootstrapped: bool,
    pub imported_messages: i64,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct IngestResult {
    pub ingested: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct IngestBatchResult {
    pub ingested_count: i64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AssembleResult {
    pub messages: Vec<AgentMessage>,
    pub estimated_tokens: i64,
    pub system_prompt_addition: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompactOutcomeDetails {
    pub rounds: i64,
    pub target_tokens: i64,
    pub mode: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompactOutcome {
    pub tokens_before: i64,
    pub tokens_after: Option<i64>,
    pub details: Option<CompactOutcomeDetails>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompactResult {
    pub ok: bool,
    pub compacted: bool,
    pub reason: String,
    pub result: Option<CompactOutcome>,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum CompactionTarget {
    #[default]
    Budget,
    Threshold,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LegacyCompactionParams {
    pub token_budget: Option<i64>,
    pub current_token_count: Option<i64>,
    pub manual_compaction: Option<bool>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub config: Option<Value>,
    pub agent_dir: Option<String>,
    pub auth_profile_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BootstrapInput {
    pub session_id: String,
    pub session_file: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IngestInput {
    pub session_id: String,
    pub message: AgentMessage,
    pub is_heartbeat: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct IngestBatchInput {
    pub session_id: String,
    pub messages: Vec<AgentMessage>,
    pub is_heartbeat: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AfterTurnInput {
    pub session_id: String,
    pub session_file: String,
    pub messages: Vec<AgentMessage>,
    pub pre_prompt_message_count: usize,
    pub auto_compaction_summary: Option<String>,
    pub is_heartbeat: Option<bool>,
    pub token_budget: Option<i64>,
    pub legacy_compaction_params: Option<LegacyCompactionParams>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AssembleInput {
    pub session_id: String,
    pub messages: Vec<AgentMessage>,
    pub token_budget: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompactLeafInput {
    pub session_id: String,
    pub session_file: String,
    pub token_budget: Option<i64>,
    pub current_token_count: Option<i64>,
    pub custom_instructions: Option<String>,
    pub legacy_params: Option<LegacyCompactionParams>,
    pub force: Option<bool>,
    pub previous_summary_content: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompactInput {
    pub session_id: String,
    pub session_file: String,
    pub token_budget: Option<i64>,
    pub current_token_count: Option<i64>,
    pub compaction_target: Option<CompactionTarget>,
    pub custom_instructions: Option<String>,
    pub legacy_params: Option<LegacyCompactionParams>,
    pub force: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
struct StoredMessage {
    role: MessageRole,
    content: String,
    token_count: i64,
}

#[derive(Clone, Debug, PartialEq)]
struct ReconcileSessionTailResult {
    imported_messages: i64,
    has_overlap: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct InterceptLargeFilesResult {
    rewritten_content: String,
    file_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct NormalizedUnknownBlock {
    block_type: String,
    text: Option<String>,
    raw: Value,
}

#[derive(Clone)]
struct RuntimeState {
    deps: Arc<dyn LcmDependencies>,
    config: LcmConfig,
    db_path: String,
    conversation_store: ConversationStore,
    summary_store: SummaryStore,
    assembler: ContextAssembler,
    compaction: CompactionEngine,
    migrated: Arc<AtomicBool>,
}

#[async_trait]
pub trait ConversationLookupApi: Send + Sync {
    async fn get_conversation_by_session_id(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<ConversationRecord>>;
}

#[async_trait]
impl ConversationLookupApi for ConversationStore {
    async fn get_conversation_by_session_id(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<ConversationRecord>> {
        ConversationStore::get_conversation_by_session_id(self, session_id)
    }
}

pub trait LcmContextEngineApi: Send + Sync {
    fn info(&self) -> &ContextEngineInfo;
    fn get_retrieval(&self) -> Arc<dyn RetrievalApi>;
    fn get_conversation_store(&self) -> Arc<dyn ConversationLookupApi>;
}

#[derive(Clone)]
pub struct LcmContextEngine {
    pub info: ContextEngineInfo,
    retrieval: Arc<dyn RetrievalApi>,
    conversation_store: Arc<dyn ConversationLookupApi>,
    runtime: Option<RuntimeState>,
}

impl LcmContextEngine {
    pub fn new(
        retrieval: Arc<dyn RetrievalApi>,
        conversation_store: Arc<dyn ConversationLookupApi>,
    ) -> Self {
        Self {
            info: default_info(),
            retrieval,
            conversation_store,
            runtime: None,
        }
    }

    pub fn from_dependencies(deps: Arc<dyn LcmDependencies>) -> anyhow::Result<Self> {
        let config = deps.config().clone();
        let shared = get_lcm_connection(&config.database_path)?;
        let conversation_store = ConversationStore::new(&shared);
        let summary_store = SummaryStore::new(&shared);

        let mut assembler =
            ContextAssembler::new(conversation_store.clone(), summary_store.clone());
        assembler.set_timezone(Some(config.timezone.clone()));

        let compaction = CompactionEngine::new(
            conversation_store.clone(),
            summary_store.clone(),
            crate::compaction::CompactionConfig {
                context_threshold: config.context_threshold,
                fresh_tail_count: i64::from(config.fresh_tail_count),
                leaf_min_fanout: i64::from(config.leaf_min_fanout),
                condensed_min_fanout: i64::from(config.condensed_min_fanout),
                condensed_min_fanout_hard: i64::from(config.condensed_min_fanout_hard),
                incremental_max_depth: i64::from(config.incremental_max_depth),
                leaf_chunk_tokens: i64::from(config.leaf_chunk_tokens),
                leaf_target_tokens: i64::from(config.leaf_target_tokens),
                condensed_target_tokens: i64::from(config.condensed_target_tokens),
                max_rounds: 10,
                timezone: Some(config.timezone.clone()),
            },
        );

        let retrieval: Arc<dyn RetrievalApi> = Arc::new(RetrievalEngine::new(
            conversation_store.clone(),
            summary_store.clone(),
        ));

        Ok(Self {
            info: default_info(),
            retrieval,
            conversation_store: Arc::new(conversation_store.clone()),
            runtime: Some(RuntimeState {
                deps,
                config: config.clone(),
                db_path: config.database_path,
                conversation_store,
                summary_store,
                assembler,
                compaction,
                migrated: Arc::new(AtomicBool::new(false)),
            }),
        })
    }

    pub fn is_runtime_enabled(&self) -> bool {
        self.runtime.is_some()
    }

    fn runtime(&self) -> anyhow::Result<&RuntimeState> {
        self.runtime
            .as_ref()
            .context("LCM runtime not configured; use LcmContextEngine::from_dependencies")
    }

    fn ensure_migrated(&self) -> anyhow::Result<()> {
        let runtime = self.runtime()?;
        if runtime.migrated.load(Ordering::SeqCst) {
            return Ok(());
        }
        let shared = get_lcm_connection(&runtime.db_path)?;
        {
            let guard = shared.conn.lock();
            run_lcm_migrations(&guard)?;
        }
        runtime.migrated.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn normalize_observed_token_count(value: Option<i64>) -> Option<i64> {
        value.filter(|value| *value > 0)
    }

    fn resolve_token_budget(
        token_budget: Option<i64>,
        legacy: Option<&LegacyCompactionParams>,
    ) -> Option<i64> {
        token_budget.filter(|value| *value > 0).or_else(|| {
            legacy
                .and_then(|legacy| legacy.token_budget)
                .filter(|value| *value > 0)
        })
    }

    async fn resolve_summarize(
        &self,
        legacy: Option<&LegacyCompactionParams>,
        custom_instructions: Option<String>,
    ) -> LcmSummarizeFn {
        let runtime = match self.runtime() {
            Ok(runtime) => runtime,
            Err(_) => return create_emergency_fallback_summarize(),
        };

        let legacy_params = LcmSummarizerLegacyParams {
            provider: legacy.and_then(|params| params.provider.clone()),
            model: legacy.and_then(|params| params.model.clone()),
            config: legacy.and_then(|params| params.config.clone()),
            agent_dir: legacy.and_then(|params| params.agent_dir.clone()),
            auth_profile_id: legacy.and_then(|params| params.auth_profile_id.clone()),
        };

        match create_lcm_summarize_from_legacy_params(
            runtime.deps.clone(),
            legacy_params,
            custom_instructions,
        )
        .await
        {
            Ok(Some(summarize)) => summarize,
            _ => create_emergency_fallback_summarize(),
        }
    }

    fn get_or_create_conversation_id(&self, session_id: &str) -> anyhow::Result<i64> {
        let runtime = self.runtime()?;
        Ok(runtime
            .conversation_store
            .get_or_create_conversation(session_id, None)?
            .conversation_id)
    }

    async fn intercept_large_files(
        &self,
        conversation_id: i64,
        content: &str,
    ) -> anyhow::Result<Option<InterceptLargeFilesResult>> {
        let runtime = self.runtime()?;
        let blocks = parse_file_blocks(content);
        if blocks.is_empty() {
            return Ok(None);
        }

        let threshold = i64::from(runtime.config.large_file_token_threshold).max(1);
        let mut cursor = 0_usize;
        let mut rewritten_segments = Vec::new();
        let mut file_ids = Vec::new();
        let mut intercepted_any = false;

        for block in blocks {
            if !is_large_file(&block.text, threshold) {
                continue;
            }
            intercepted_any = true;

            let file_id = format!(
                "file_{}",
                Uuid::new_v4().simple().to_string()[..16].to_lowercase()
            );
            let extension =
                extension_from_name_or_mime(block.file_name.as_deref(), block.mime_type.as_deref());
            let storage_uri =
                store_large_file_content(conversation_id, &file_id, &extension, &block.text)?;
            let byte_size = block.text.as_bytes().len() as i64;
            let exploration_summary = generate_exploration_summary(ExplorationSummaryInput {
                content: &block.text,
                file_name: block.file_name.as_deref(),
                mime_type: block.mime_type.as_deref(),
                summarize_text: None,
            });

            runtime
                .summary_store
                .insert_large_file(CreateLargeFileInput {
                    file_id: file_id.clone(),
                    conversation_id,
                    file_name: block.file_name.clone(),
                    mime_type: block.mime_type.clone(),
                    byte_size: Some(byte_size),
                    storage_uri,
                    exploration_summary: Some(exploration_summary.clone()),
                })?;

            rewritten_segments.push(content[cursor..block.start].to_string());
            rewritten_segments.push(format_file_reference(
                &file_id,
                block.file_name.as_deref(),
                block.mime_type.as_deref(),
                byte_size,
                &exploration_summary,
            ));
            cursor = block.end;
            file_ids.push(file_id);
        }

        if !intercepted_any {
            return Ok(None);
        }

        rewritten_segments.push(content[cursor..].to_string());
        Ok(Some(InterceptLargeFilesResult {
            rewritten_content: rewritten_segments.join(""),
            file_ids,
        }))
    }

    async fn ingest_single(&self, input: IngestInput) -> anyhow::Result<IngestResult> {
        if input.is_heartbeat.unwrap_or(false) {
            return Ok(IngestResult { ingested: false });
        }

        let runtime = self.runtime()?;
        let conversation_id = self.get_or_create_conversation_id(&input.session_id)?;
        let mut stored = to_stored_message(&input.message);
        let mut message_for_parts = input.message.clone();

        if matches!(stored.role, MessageRole::User) {
            if let Some(intercepted) = self
                .intercept_large_files(conversation_id, &stored.content)
                .await?
            {
                stored.content = intercepted.rewritten_content;
                stored.token_count = estimate_tokens(&stored.content);
                message_for_parts.content = Value::String(stored.content.clone());
            }
        }

        let seq = runtime.conversation_store.get_max_seq(conversation_id)? + 1;
        let message_record = runtime
            .conversation_store
            .create_message(CreateMessageInput {
                conversation_id,
                seq,
                role: stored.role,
                content: stored.content.clone(),
                token_count: stored.token_count,
            })?;

        let parts = build_message_parts(&input.session_id, &message_for_parts, &stored.content);
        runtime
            .conversation_store
            .create_message_parts(message_record.message_id, &parts)?;
        runtime
            .summary_store
            .append_context_message(conversation_id, message_record.message_id)?;

        Ok(IngestResult { ingested: true })
    }

    async fn reconcile_session_tail(
        &self,
        session_id: &str,
        conversation_id: i64,
        historical_messages: &[AgentMessage],
    ) -> anyhow::Result<ReconcileSessionTailResult> {
        if historical_messages.is_empty() {
            return Ok(ReconcileSessionTailResult {
                imported_messages: 0,
                has_overlap: false,
            });
        }

        let runtime = self.runtime()?;
        let Some(latest_db_message) = runtime
            .conversation_store
            .get_last_message(conversation_id)?
        else {
            return Ok(ReconcileSessionTailResult {
                imported_messages: 0,
                has_overlap: false,
            });
        };

        let stored_historical_messages = historical_messages
            .iter()
            .map(to_stored_message)
            .collect::<Vec<_>>();

        let latest_historical = stored_historical_messages.last().cloned();
        if let Some(latest_historical) = latest_historical {
            let latest_identity =
                message_identity(&latest_db_message.role, &latest_db_message.content);
            if latest_identity
                == message_identity(&latest_historical.role, &latest_historical.content)
            {
                let db_occurrences = runtime.conversation_store.count_messages_by_identity(
                    conversation_id,
                    latest_db_message.role.clone(),
                    &latest_db_message.content,
                )?;
                let historical_occurrences = stored_historical_messages
                    .iter()
                    .filter(|stored| {
                        message_identity(&stored.role, &stored.content) == latest_identity
                    })
                    .count() as i64;

                if db_occurrences == historical_occurrences {
                    return Ok(ReconcileSessionTailResult {
                        imported_messages: 0,
                        has_overlap: true,
                    });
                }
            }
        }

        let mut historical_identity_totals: HashMap<String, i64> = HashMap::new();
        for stored in &stored_historical_messages {
            let identity = message_identity(&stored.role, &stored.content);
            let entry = historical_identity_totals.entry(identity).or_insert(0);
            *entry += 1;
        }

        let mut historical_identity_counts_after_index: HashMap<String, i64> = HashMap::new();
        let mut db_identity_counts: HashMap<String, i64> = HashMap::new();
        let mut anchor_index: i64 = -1;

        for index in (0..stored_historical_messages.len()).rev() {
            let stored = &stored_historical_messages[index];
            let identity = message_identity(&stored.role, &stored.content);
            let seen_after = *historical_identity_counts_after_index
                .get(&identity)
                .unwrap_or(&0);
            let total = *historical_identity_totals.get(&identity).unwrap_or(&0);
            let occurrences_through_index = total - seen_after;

            let exists = runtime.conversation_store.has_message(
                conversation_id,
                stored.role.clone(),
                &stored.content,
            )?;
            historical_identity_counts_after_index.insert(identity.clone(), seen_after + 1);

            if !exists {
                continue;
            }

            let db_count_for_identity = if let Some(value) = db_identity_counts.get(&identity) {
                *value
            } else {
                let count = runtime.conversation_store.count_messages_by_identity(
                    conversation_id,
                    stored.role.clone(),
                    &stored.content,
                )?;
                db_identity_counts.insert(identity.clone(), count);
                count
            };

            if db_count_for_identity != occurrences_through_index {
                continue;
            }

            anchor_index = index as i64;
            break;
        }

        if anchor_index < 0 {
            return Ok(ReconcileSessionTailResult {
                imported_messages: 0,
                has_overlap: false,
            });
        }

        let anchor_index_usize = anchor_index as usize;
        if anchor_index_usize >= historical_messages.len().saturating_sub(1) {
            return Ok(ReconcileSessionTailResult {
                imported_messages: 0,
                has_overlap: true,
            });
        }

        let missing_tail = historical_messages
            .iter()
            .skip(anchor_index_usize + 1)
            .cloned()
            .collect::<Vec<_>>();

        let mut imported_messages = 0_i64;
        for message in missing_tail {
            let result = self
                .ingest_single(IngestInput {
                    session_id: session_id.to_string(),
                    message,
                    is_heartbeat: Some(false),
                })
                .await?;
            if result.ingested {
                imported_messages += 1;
            }
        }

        Ok(ReconcileSessionTailResult {
            imported_messages,
            has_overlap: true,
        })
    }

    pub async fn bootstrap(&self, input: BootstrapInput) -> anyhow::Result<BootstrapResult> {
        self.ensure_migrated()?;
        let runtime = self.runtime()?;

        let conversation = runtime
            .conversation_store
            .get_or_create_conversation(&input.session_id, None)?;
        let conversation_id = conversation.conversation_id;
        let historical_messages = read_leaf_path_messages(&input.session_file);

        let existing_count = runtime
            .conversation_store
            .get_message_count(conversation_id)?;

        let mut result = if existing_count == 0 {
            if historical_messages.is_empty() {
                runtime
                    .conversation_store
                    .mark_conversation_bootstrapped(conversation_id)?;
                BootstrapResult {
                    bootstrapped: false,
                    imported_messages: 0,
                    reason: Some("no leaf-path messages in session".to_string()),
                }
            } else {
                let next_seq = runtime.conversation_store.get_max_seq(conversation_id)? + 1;
                let rows = historical_messages
                    .iter()
                    .enumerate()
                    .map(|(idx, message)| {
                        let stored = to_stored_message(message);
                        CreateMessageInput {
                            conversation_id,
                            seq: next_seq + idx as i64,
                            role: stored.role,
                            content: stored.content,
                            token_count: stored.token_count,
                        }
                    })
                    .collect::<Vec<_>>();

                let inserted = runtime.conversation_store.create_messages_bulk(&rows)?;
                runtime.summary_store.append_context_messages(
                    conversation_id,
                    &inserted
                        .iter()
                        .map(|record| record.message_id)
                        .collect::<Vec<_>>(),
                )?;
                runtime
                    .conversation_store
                    .mark_conversation_bootstrapped(conversation_id)?;

                if runtime.config.prune_heartbeat_ok {
                    let _ = self.prune_heartbeat_ok_turns(conversation_id)?;
                }

                BootstrapResult {
                    bootstrapped: true,
                    imported_messages: inserted.len() as i64,
                    reason: None,
                }
            }
        } else {
            let reconcile = self
                .reconcile_session_tail(&input.session_id, conversation_id, &historical_messages)
                .await?;

            if conversation.bootstrapped_at.is_none() {
                runtime
                    .conversation_store
                    .mark_conversation_bootstrapped(conversation_id)?;
            }

            if reconcile.imported_messages > 0 {
                BootstrapResult {
                    bootstrapped: true,
                    imported_messages: reconcile.imported_messages,
                    reason: Some("reconciled missing session messages".to_string()),
                }
            } else if conversation.bootstrapped_at.is_some() {
                BootstrapResult {
                    bootstrapped: false,
                    imported_messages: 0,
                    reason: Some("already bootstrapped".to_string()),
                }
            } else {
                BootstrapResult {
                    bootstrapped: false,
                    imported_messages: 0,
                    reason: Some(if reconcile.has_overlap {
                        "conversation already up to date".to_string()
                    } else {
                        "conversation already has messages".to_string()
                    }),
                }
            }
        };

        if runtime.config.prune_heartbeat_ok && !result.bootstrapped {
            if let Some(conversation) = runtime
                .conversation_store
                .get_conversation_by_session_id(&input.session_id)?
            {
                let _ = self.prune_heartbeat_ok_turns(conversation.conversation_id)?;
            }
        }

        if result.reason.as_deref() == Some("") {
            result.reason = None;
        }

        Ok(result)
    }

    pub async fn ingest(&self, input: IngestInput) -> anyhow::Result<IngestResult> {
        self.ensure_migrated()?;
        self.ingest_single(input).await
    }

    pub async fn ingest_batch(&self, input: IngestBatchInput) -> anyhow::Result<IngestBatchResult> {
        self.ensure_migrated()?;
        if input.messages.is_empty() {
            return Ok(IngestBatchResult { ingested_count: 0 });
        }

        let mut ingested_count = 0_i64;
        for message in input.messages {
            let result = self
                .ingest_single(IngestInput {
                    session_id: input.session_id.clone(),
                    message,
                    is_heartbeat: input.is_heartbeat,
                })
                .await?;
            if result.ingested {
                ingested_count += 1;
            }
        }

        Ok(IngestBatchResult { ingested_count })
    }

    pub async fn after_turn(&self, input: AfterTurnInput) -> anyhow::Result<()> {
        self.ensure_migrated()?;
        let runtime = self.runtime()?;

        let mut ingest_batch = Vec::new();
        if let Some(auto_compaction_summary) = input
            .auto_compaction_summary
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            ingest_batch.push(AgentMessage {
                role: "user".to_string(),
                content: Value::String(auto_compaction_summary.to_string()),
                tool_call_id: None,
                tool_use_id: None,
                tool_name: None,
                stop_reason: None,
                is_error: None,
                usage: None,
                timestamp: None,
            });
        }

        let new_messages = input
            .messages
            .iter()
            .skip(input.pre_prompt_message_count)
            .cloned()
            .collect::<Vec<_>>();
        ingest_batch.extend(new_messages);

        if !ingest_batch.is_empty() {
            let _ = self
                .ingest_batch(IngestBatchInput {
                    session_id: input.session_id.clone(),
                    messages: ingest_batch,
                    is_heartbeat: Some(input.is_heartbeat.unwrap_or(false)),
                })
                .await;
        }

        if runtime.config.autocompact_disabled {
            return Ok(());
        }

        let token_budget = input.token_budget.filter(|value| *value > 0);
        let Some(token_budget) = token_budget else {
            return Ok(());
        };

        let live_context_tokens = estimate_session_token_count_for_after_turn(&input.messages);

        if let Ok(leaf_trigger) = self.evaluate_leaf_trigger(&input.session_id).await {
            if leaf_trigger.should_compact {
                let _ = self
                    .compact_leaf_async(CompactLeafInput {
                        session_id: input.session_id.clone(),
                        session_file: input.session_file.clone(),
                        token_budget: Some(token_budget),
                        current_token_count: Some(live_context_tokens),
                        custom_instructions: None,
                        legacy_params: input.legacy_compaction_params.clone(),
                        force: None,
                        previous_summary_content: None,
                    })
                    .await;
            }
        }

        let _ = self
            .compact(CompactInput {
                session_id: input.session_id,
                session_file: input.session_file,
                token_budget: Some(token_budget),
                current_token_count: Some(live_context_tokens),
                compaction_target: Some(CompactionTarget::Threshold),
                custom_instructions: None,
                legacy_params: input.legacy_compaction_params,
                force: None,
            })
            .await;

        Ok(())
    }

    pub async fn assemble(&self, input: AssembleInput) -> anyhow::Result<AssembleResult> {
        if self.ensure_migrated().is_err() {
            return Ok(AssembleResult {
                messages: input.messages,
                estimated_tokens: 0,
                system_prompt_addition: None,
            });
        }

        let runtime = self.runtime()?;
        let Some(conversation) = runtime
            .conversation_store
            .get_conversation_by_session_id(&input.session_id)?
        else {
            return Ok(AssembleResult {
                messages: input.messages,
                estimated_tokens: 0,
                system_prompt_addition: None,
            });
        };

        let context_items = runtime
            .summary_store
            .get_context_items(conversation.conversation_id)?;
        if context_items.is_empty() {
            return Ok(AssembleResult {
                messages: input.messages,
                estimated_tokens: 0,
                system_prompt_addition: None,
            });
        }

        let has_summary_items = context_items.iter().any(|item| {
            matches!(
                item.item_type,
                crate::store::summary_store::ContextItemType::Summary
            )
        });
        if !has_summary_items && context_items.len() < input.messages.len() {
            return Ok(AssembleResult {
                messages: input.messages,
                estimated_tokens: 0,
                system_prompt_addition: None,
            });
        }

        let token_budget = input
            .token_budget
            .filter(|value| *value > 0)
            .unwrap_or(128_000);
        match runtime
            .assembler
            .assemble(AssembleContextInput {
                conversation_id: conversation.conversation_id,
                token_budget,
                fresh_tail_count: Some(i64::from(runtime.config.fresh_tail_count)),
            })
            .await
        {
            Ok(assembled) => {
                if assembled.messages.is_empty() && !input.messages.is_empty() {
                    return Ok(AssembleResult {
                        messages: input.messages,
                        estimated_tokens: 0,
                        system_prompt_addition: None,
                    });
                }

                Ok(AssembleResult {
                    messages: assembled.messages,
                    estimated_tokens: assembled.estimated_tokens,
                    system_prompt_addition: assembled.system_prompt_addition,
                })
            }
            Err(_) => Ok(AssembleResult {
                messages: input.messages,
                estimated_tokens: 0,
                system_prompt_addition: None,
            }),
        }
    }

    pub async fn assemble_context(&self, input: AssembleInput) -> anyhow::Result<AssembleResult> {
        self.assemble(input).await
    }

    pub async fn evaluate_leaf_trigger(
        &self,
        session_id: &str,
    ) -> anyhow::Result<LeafTriggerDecision> {
        self.ensure_migrated()?;
        let runtime = self.runtime()?;

        let Some(conversation) = runtime
            .conversation_store
            .get_conversation_by_session_id(session_id)?
        else {
            return Ok(LeafTriggerDecision {
                should_compact: false,
                raw_tokens_outside_tail: 0,
                threshold: i64::from(runtime.config.leaf_chunk_tokens).max(1),
            });
        };

        runtime
            .compaction
            .evaluate_leaf_trigger(conversation.conversation_id)
            .await
    }

    pub async fn compact_leaf_async(
        &self,
        input: CompactLeafInput,
    ) -> anyhow::Result<CompactResult> {
        self.ensure_migrated()?;
        let runtime = self.runtime()?;

        let Some(conversation) = runtime
            .conversation_store
            .get_conversation_by_session_id(&input.session_id)?
        else {
            return Ok(CompactResult {
                ok: true,
                compacted: false,
                reason: "no conversation found for session".to_string(),
                result: None,
            });
        };

        let token_budget =
            Self::resolve_token_budget(input.token_budget, input.legacy_params.as_ref());
        let Some(token_budget) = token_budget else {
            return Ok(CompactResult {
                ok: false,
                compacted: false,
                reason: "missing token budget in compact params".to_string(),
                result: None,
            });
        };

        let observed_tokens =
            Self::normalize_observed_token_count(input.current_token_count.or_else(|| {
                input
                    .legacy_params
                    .as_ref()
                    .and_then(|params| params.current_token_count)
            }));
        let summarize = self
            .resolve_summarize(input.legacy_params.as_ref(), input.custom_instructions)
            .await;

        let result = runtime
            .compaction
            .compact_leaf(CompactionLeafInput {
                conversation_id: conversation.conversation_id,
                token_budget,
                summarize,
                force: input.force,
                previous_summary_content: input.previous_summary_content,
            })
            .await?;

        let tokens_before = observed_tokens.unwrap_or(result.tokens_before);
        Ok(CompactResult {
            ok: true,
            compacted: result.action_taken,
            reason: if result.action_taken {
                "compacted".to_string()
            } else {
                "below threshold".to_string()
            },
            result: Some(CompactOutcome {
                tokens_before,
                tokens_after: Some(result.tokens_after),
                details: Some(CompactOutcomeDetails {
                    rounds: if result.action_taken { 1 } else { 0 },
                    target_tokens: token_budget,
                    mode: Some("leaf".to_string()),
                }),
            }),
        })
    }

    pub async fn compact(&self, input: CompactInput) -> anyhow::Result<CompactResult> {
        self.ensure_migrated()?;
        let runtime = self.runtime()?;

        let Some(conversation) = runtime
            .conversation_store
            .get_conversation_by_session_id(&input.session_id)?
        else {
            return Ok(CompactResult {
                ok: true,
                compacted: false,
                reason: "no conversation found for session".to_string(),
                result: None,
            });
        };

        let token_budget =
            Self::resolve_token_budget(input.token_budget, input.legacy_params.as_ref());
        let Some(token_budget) = token_budget else {
            return Ok(CompactResult {
                ok: false,
                compacted: false,
                reason: "missing token budget in compact params".to_string(),
                result: None,
            });
        };

        let manual_compaction_requested = input
            .legacy_params
            .as_ref()
            .and_then(|params| params.manual_compaction)
            .unwrap_or(false);
        let force_compaction = input.force.unwrap_or(false) || manual_compaction_requested;

        let observed_tokens =
            Self::normalize_observed_token_count(input.current_token_count.or_else(|| {
                input
                    .legacy_params
                    .as_ref()
                    .and_then(|params| params.current_token_count)
            }));

        let summarize = self
            .resolve_summarize(input.legacy_params.as_ref(), input.custom_instructions)
            .await;

        let decision = runtime
            .compaction
            .evaluate(conversation.conversation_id, token_budget, observed_tokens)
            .await?;

        if !force_compaction && !decision.should_compact {
            return Ok(CompactResult {
                ok: true,
                compacted: false,
                reason: "below threshold".to_string(),
                result: Some(CompactOutcome {
                    tokens_before: decision.current_tokens,
                    tokens_after: None,
                    details: None,
                }),
            });
        }

        let use_sweep = force_compaction
            || manual_compaction_requested
            || matches!(input.compaction_target, Some(CompactionTarget::Threshold));

        if use_sweep {
            let sweep_result = runtime
                .compaction
                .compact_full_sweep(CompactionRunInput {
                    conversation_id: conversation.conversation_id,
                    token_budget,
                    summarize,
                    force: Some(force_compaction),
                    hard_trigger: Some(false),
                })
                .await?;

            return Ok(CompactResult {
                ok: true,
                compacted: sweep_result.action_taken,
                reason: if sweep_result.action_taken {
                    "compacted".to_string()
                } else if manual_compaction_requested {
                    "nothing to compact".to_string()
                } else {
                    "already under target".to_string()
                },
                result: Some(CompactOutcome {
                    tokens_before: decision.current_tokens,
                    tokens_after: Some(sweep_result.tokens_after),
                    details: Some(CompactOutcomeDetails {
                        rounds: if sweep_result.action_taken { 1 } else { 0 },
                        target_tokens: if matches!(
                            input.compaction_target,
                            Some(CompactionTarget::Threshold)
                        ) {
                            decision.threshold
                        } else {
                            token_budget
                        },
                        mode: None,
                    }),
                }),
            });
        }

        let target_tokens = if force_compaction {
            token_budget
        } else if matches!(input.compaction_target, Some(CompactionTarget::Threshold)) {
            decision.threshold
        } else {
            token_budget
        };

        let compact_result = runtime
            .compaction
            .compact_until_under(CompactionUntilInput {
                conversation_id: conversation.conversation_id,
                token_budget,
                target_tokens: Some(target_tokens),
                current_tokens: observed_tokens,
                summarize,
            })
            .await?;

        let did_compact = compact_result.rounds > 0;
        Ok(CompactResult {
            ok: compact_result.success,
            compacted: did_compact,
            reason: if compact_result.success {
                if did_compact {
                    "compacted".to_string()
                } else {
                    "already under target".to_string()
                }
            } else {
                "could not reach target".to_string()
            },
            result: Some(CompactOutcome {
                tokens_before: decision.current_tokens,
                tokens_after: Some(compact_result.final_tokens),
                details: Some(CompactOutcomeDetails {
                    rounds: compact_result.rounds,
                    target_tokens,
                    mode: None,
                }),
            }),
        })
    }

    pub fn get_retrieval(&self) -> Arc<dyn RetrievalApi> {
        self.retrieval.clone()
    }

    pub fn get_conversation_store(&self) -> Arc<dyn ConversationLookupApi> {
        self.conversation_store.clone()
    }

    pub fn dispose(&self) {
        // No-op: engine is intended to be reused across runs.
    }

    fn prune_heartbeat_ok_turns(&self, conversation_id: i64) -> anyhow::Result<i64> {
        let runtime = self.runtime()?;
        let all_messages = runtime
            .conversation_store
            .get_messages(conversation_id, None, None)?;
        if all_messages.is_empty() {
            return Ok(0);
        }

        let mut to_delete = Vec::new();
        for idx in 0..all_messages.len() {
            let message = &all_messages[idx];
            if !matches!(message.role, MessageRole::Assistant) {
                continue;
            }
            if !is_heartbeat_ok_content(&message.content) {
                continue;
            }

            let mut turn_message_ids = vec![message.message_id];
            for back in (0..idx).rev() {
                let prev = &all_messages[back];
                turn_message_ids.push(prev.message_id);
                if matches!(prev.role, MessageRole::User) {
                    break;
                }
            }
            to_delete.extend(turn_message_ids);
        }

        if to_delete.is_empty() {
            return Ok(0);
        }

        let mut unique = HashSet::new();
        let unique_ids = to_delete
            .into_iter()
            .filter(|id| unique.insert(*id))
            .collect::<Vec<_>>();
        runtime.conversation_store.delete_messages(&unique_ids)
    }
}

impl LcmContextEngineApi for LcmContextEngine {
    fn info(&self) -> &ContextEngineInfo {
        &self.info
    }

    fn get_retrieval(&self) -> Arc<dyn RetrievalApi> {
        self.retrieval.clone()
    }

    fn get_conversation_store(&self) -> Arc<dyn ConversationLookupApi> {
        self.conversation_store.clone()
    }
}

fn default_info() -> ContextEngineInfo {
    ContextEngineInfo {
        id: "lcm".to_string(),
        name: "Lossless Context Management Engine".to_string(),
        version: "0.1.0".to_string(),
        owns_compaction: true,
    }
}

fn estimate_tokens(text: &str) -> i64 {
    ((text.chars().count() as f64) / 4.0).ceil() as i64
}

fn create_emergency_fallback_summarize() -> LcmSummarizeFn {
    Arc::new(move |text, aggressive, _options| {
        Box::pin(async move {
            let max_chars = if aggressive { 600 * 4 } else { 900 * 4 };
            if text.chars().count() <= max_chars {
                return text;
            }
            let trimmed = text.chars().take(max_chars).collect::<String>();
            format!("{}\n[Truncated for context management]", trimmed)
        })
    })
}

fn message_role_label(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

fn message_identity(role: &MessageRole, content: &str) -> String {
    format!("{}\u{0000}{}", message_role_label(role), content)
}

fn extract_message_content(content: &Value) -> String {
    if let Some(text) = content.as_str() {
        return text.to_string();
    }

    if let Some(items) = content.as_array() {
        return items
            .iter()
            .filter_map(|item| {
                let obj = item.as_object()?;
                if obj.get("type").and_then(Value::as_str) != Some("text") {
                    return None;
                }
                obj.get("text")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
            })
            .collect::<Vec<_>>()
            .join("\n");
    }

    serde_json::to_string(content).unwrap_or_default()
}

fn to_db_role(role: &str) -> MessageRole {
    match role {
        "tool" | "toolResult" => MessageRole::Tool,
        "system" => MessageRole::System,
        "user" => MessageRole::User,
        "assistant" => MessageRole::Assistant,
        _ => MessageRole::Assistant,
    }
}

fn is_text_block(value: &Value) -> bool {
    value
        .as_object()
        .map(|obj| {
            obj.get("type").and_then(Value::as_str) == Some("text")
                && obj.get("text").and_then(Value::as_str).is_some()
        })
        .unwrap_or(false)
}

fn runtime_role_for_token_estimate(role: &str) -> &'static str {
    match role {
        "tool" | "toolResult" => "toolResult",
        "user" | "system" => "user",
        _ => "assistant",
    }
}

fn estimate_content_tokens_for_role(role: &str, content: &Value, fallback_content: &str) -> i64 {
    if let Some(text) = content.as_str() {
        return estimate_tokens(text);
    }

    if let Some(items) = content.as_array() {
        if items.is_empty() {
            return estimate_tokens(fallback_content);
        }
        if role == "user" && items.len() == 1 && is_text_block(&items[0]) {
            return estimate_tokens(
                items[0]
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            );
        }
        return estimate_tokens(&serde_json::to_string(items).unwrap_or_default());
    }

    if content.is_object() {
        if role == "user" && is_text_block(content) {
            return estimate_tokens(
                content
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            );
        }
        return estimate_tokens(&serde_json::to_string(&vec![content]).unwrap_or_default());
    }

    estimate_tokens(fallback_content)
}

fn to_stored_message(message: &AgentMessage) -> StoredMessage {
    let content = extract_message_content(&message.content);
    let runtime_role = runtime_role_for_token_estimate(&message.role);
    let token_count = estimate_content_tokens_for_role(runtime_role, &message.content, &content);

    StoredMessage {
        role: to_db_role(&message.role),
        content,
        token_count,
    }
}

fn normalize_unknown_block(value: &Value) -> NormalizedUnknownBlock {
    if !value.is_object() {
        return NormalizedUnknownBlock {
            block_type: "agent".to_string(),
            text: None,
            raw: value.clone(),
        };
    }

    let obj = value.as_object().expect("checked object");
    let block_type = obj
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("agent")
        .to_string();
    let text = obj
        .get("text")
        .and_then(Value::as_str)
        .or_else(|| obj.get("thinking").and_then(Value::as_str))
        .map(ToString::to_string);

    NormalizedUnknownBlock {
        block_type,
        text,
        raw: value.clone(),
    }
}

fn to_part_type(raw_type: &str) -> MessagePartType {
    match raw_type {
        "text" => MessagePartType::Text,
        "thinking" | "reasoning" => MessagePartType::Reasoning,
        "tool_use" | "tool-use" | "tool_result" | "toolResult" | "tool" => MessagePartType::Tool,
        "patch" => MessagePartType::Patch,
        "file" | "image" => MessagePartType::File,
        "subtask" => MessagePartType::Subtask,
        "compaction" => MessagePartType::Compaction,
        "step_start" | "step-start" => MessagePartType::StepStart,
        "step_finish" | "step-finish" => MessagePartType::StepFinish,
        "snapshot" => MessagePartType::Snapshot,
        "retry" => MessagePartType::Retry,
        _ => MessagePartType::Agent,
    }
}

fn value_string_or_json(value: Option<&Value>) -> Option<String> {
    let value = value?;
    if value.is_null() {
        return None;
    }
    if let Some(text) = value.as_str() {
        return Some(text.to_string());
    }
    serde_json::to_string(value).ok()
}

fn build_message_parts(
    session_id: &str,
    message: &AgentMessage,
    fallback_content: &str,
) -> Vec<CreateMessagePartInput> {
    let role = message.role.clone();

    if let Some(text) = message.content.as_str() {
        return vec![CreateMessagePartInput {
            session_id: session_id.to_string(),
            part_type: MessagePartType::Text,
            ordinal: 0,
            text_content: Some(text.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_input: None,
            tool_output: None,
            metadata: Some(
                json!({
                    "originalRole": role,
                })
                .to_string(),
            ),
        }];
    }

    if !message.content.is_array() {
        return vec![CreateMessagePartInput {
            session_id: session_id.to_string(),
            part_type: MessagePartType::Agent,
            ordinal: 0,
            text_content: if fallback_content.is_empty() {
                None
            } else {
                Some(fallback_content.to_string())
            },
            tool_call_id: None,
            tool_name: None,
            tool_input: None,
            tool_output: None,
            metadata: Some(
                json!({
                    "originalRole": role,
                    "source": "non-array-content",
                    "raw": message.content,
                })
                .to_string(),
            ),
        }];
    }

    let mut parts = Vec::new();
    let top_level_tool_call_id = message
        .tool_call_id
        .as_deref()
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            message
                .tool_use_id
                .as_deref()
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
        });

    for (ordinal, block) in message
        .content
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .enumerate()
    {
        let normalized = normalize_unknown_block(block);
        let raw_obj = normalized.raw.as_object();

        let tool_call_id = raw_obj
            .and_then(|obj| {
                obj.get("toolCallId")
                    .and_then(Value::as_str)
                    .or_else(|| obj.get("tool_call_id").and_then(Value::as_str))
                    .map(ToString::to_string)
            })
            .or_else(|| top_level_tool_call_id.clone());

        parts.push(CreateMessagePartInput {
            session_id: session_id.to_string(),
            part_type: to_part_type(&normalized.block_type),
            ordinal: ordinal as i64,
            text_content: normalized.text,
            tool_call_id,
            tool_name: raw_obj
                .and_then(|obj| {
                    obj.get("name")
                        .and_then(Value::as_str)
                        .or_else(|| obj.get("toolName").and_then(Value::as_str))
                        .or_else(|| obj.get("tool_name").and_then(Value::as_str))
                        .map(ToString::to_string)
                })
                .or_else(|| message.tool_name.clone()),
            tool_input: raw_obj
                .and_then(|obj| value_string_or_json(obj.get("input")))
                .or_else(|| raw_obj.and_then(|obj| value_string_or_json(obj.get("toolInput"))))
                .or_else(|| raw_obj.and_then(|obj| value_string_or_json(obj.get("tool_input")))),
            tool_output: raw_obj
                .and_then(|obj| value_string_or_json(obj.get("output")))
                .or_else(|| raw_obj.and_then(|obj| value_string_or_json(obj.get("toolOutput"))))
                .or_else(|| raw_obj.and_then(|obj| value_string_or_json(obj.get("tool_output")))),
            metadata: Some(
                json!({
                    "originalRole": role,
                    "rawType": normalized.block_type,
                    "raw": normalized.raw,
                })
                .to_string(),
            ),
        });
    }

    parts
}

fn estimate_message_content_tokens_for_after_turn(content: &Value) -> i64 {
    if let Some(text) = content.as_str() {
        return estimate_tokens(text);
    }

    if let Some(items) = content.as_array() {
        let mut total = 0_i64;
        for part in items {
            let text = part
                .as_object()
                .and_then(|obj| {
                    obj.get("text")
                        .and_then(Value::as_str)
                        .or_else(|| obj.get("thinking").and_then(Value::as_str))
                })
                .unwrap_or_default();
            if !text.is_empty() {
                total += estimate_tokens(text);
            }
        }
        return total;
    }

    if content.is_null() {
        return 0;
    }

    estimate_tokens(&serde_json::to_string(content).unwrap_or_default())
}

fn estimate_session_token_count_for_after_turn(messages: &[AgentMessage]) -> i64 {
    messages
        .iter()
        .map(|message| estimate_message_content_tokens_for_after_turn(&message.content))
        .sum()
}

fn read_leaf_path_messages(session_file: &str) -> Vec<AgentMessage> {
    let raw = fs::read_to_string(session_file).unwrap_or_default();
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    if trimmed.starts_with('[') {
        if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
            if let Some(items) = parsed.as_array() {
                return items
                    .iter()
                    .filter_map(value_to_bootstrap_message)
                    .collect();
            }
        }
        return Vec::new();
    }

    raw.lines()
        .filter_map(|line| {
            let item = line.trim();
            if item.is_empty() {
                return None;
            }
            let parsed = serde_json::from_str::<Value>(item).ok()?;
            let candidate = parsed
                .as_object()
                .and_then(|obj| obj.get("message"))
                .cloned()
                .unwrap_or(parsed);
            value_to_bootstrap_message(&candidate)
        })
        .collect()
}

fn value_to_bootstrap_message(value: &Value) -> Option<AgentMessage> {
    let parsed = serde_json::from_value::<AgentMessage>(value.clone()).ok()?;
    if parsed.role.trim().is_empty() {
        return None;
    }
    Some(parsed)
}

fn store_large_file_content(
    conversation_id: i64,
    file_id: &str,
    extension: &str,
    content: &str,
) -> anyhow::Result<String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let mut dir = PathBuf::from(home);
    dir.push(".openclaw");
    dir.push("lcm-files");
    dir.push(conversation_id.to_string());
    fs::create_dir_all(&dir)?;

    let normalized_extension = extension
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_lowercase();
    let normalized_extension = if normalized_extension.is_empty() {
        "txt".to_string()
    } else {
        normalized_extension
    };

    let mut file_path = dir;
    file_path.push(format!("{}.{}", file_id, normalized_extension));
    fs::write(&file_path, content)?;
    Ok(file_path.to_string_lossy().to_string())
}

const HEARTBEAT_OK_TOKEN: &str = "heartbeat_ok";

fn is_heartbeat_ok_content(content: &str) -> bool {
    let trimmed = content.trim().to_lowercase();
    if trimmed.is_empty() {
        return false;
    }

    if let Some(suffix) = trimmed.strip_prefix(HEARTBEAT_OK_TOKEN) {
        if suffix.is_empty() {
            return true;
        }
        if let Some(ch) = suffix.chars().next() {
            if !ch.is_ascii_alphanumeric() && ch != '_' {
                return true;
            }
        }
    }

    trimmed.ends_with(HEARTBEAT_OK_TOKEN)
}
