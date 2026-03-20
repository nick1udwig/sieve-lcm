use crate::assembler::ContextAssembler;
use crate::compaction::{
    CompactInput as CompactionRunInput, CompactionConfig, CompactionEngine, CompactionResult,
};
use crate::db::connection::{SharedConnection, get_lcm_connection};
use crate::db::migration::run_lcm_migrations;
use crate::engine::AssembleResult;
use crate::store::conversation_store::{ConversationStore, CreateMessageInput, MessageRole};
use crate::store::summary_store::{ContextItemType, SummaryKind, SummaryStore};
use crate::summarize::LcmSummarizeFn;
use anyhow::Context;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpaqueContextRef {
    pub reference: String,
    pub source: String,
    pub created_at: String,
    pub summary_kind: Option<String>,
    pub depth: Option<i64>,
    pub token_count: i64,
}

#[derive(Clone)]
pub struct PlannerLaneMemory {
    conversation_store: ConversationStore,
    summary_store: SummaryStore,
    assembler: ContextAssembler,
    compaction: CompactionEngine,
}

#[derive(Clone, Debug, PartialEq)]
struct OpaqueContextRefCandidate {
    ordinal: i64,
    token_count: i64,
    reference: OpaqueContextRef,
}

impl PlannerLaneMemory {
    pub fn open(
        db_path: &str,
        compaction_config: CompactionConfig,
        timezone: Option<String>,
    ) -> anyhow::Result<Self> {
        let shared = get_lcm_connection(db_path)?;
        ensure_migrated(&shared)?;
        let conversation_store = ConversationStore::new(&shared);
        let summary_store = SummaryStore::new(&shared);
        let assembler = ContextAssembler::new(conversation_store.clone(), summary_store.clone())
            .with_timezone(timezone);
        let compaction = CompactionEngine::new(
            conversation_store.clone(),
            summary_store.clone(),
            compaction_config,
        );
        Ok(Self {
            conversation_store,
            summary_store,
            assembler,
            compaction,
        })
    }

    pub fn ingest_text_message(
        &self,
        session_id: &str,
        role: MessageRole,
        content: &str,
    ) -> anyhow::Result<()> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Ok(());
        }

        let conversation = self
            .conversation_store
            .get_or_create_conversation(session_id, None)?;
        let seq = self
            .conversation_store
            .get_max_seq(conversation.conversation_id)?
            + 1;
        let message = self.conversation_store.create_message(CreateMessageInput {
            conversation_id: conversation.conversation_id,
            seq,
            role,
            content: trimmed.to_string(),
            token_count: estimate_tokens(trimmed),
        })?;
        self.summary_store
            .append_context_message(conversation.conversation_id, message.message_id)?;
        Ok(())
    }

    pub async fn compact_session(
        &self,
        session_id: &str,
        token_budget: i64,
        summarize: LcmSummarizeFn,
    ) -> anyhow::Result<CompactionResult> {
        let conversation_id = self
            .conversation_store
            .get_or_create_conversation(session_id, None)?
            .conversation_id;
        self.compaction
            .compact(CompactionRunInput {
                conversation_id,
                token_budget: token_budget.max(1),
                summarize,
                force: None,
                hard_trigger: None,
            })
            .await
    }

    pub async fn assemble_trusted_context(
        &self,
        session_id: &str,
        token_budget: i64,
    ) -> anyhow::Result<AssembleResult> {
        let Some(conversation) = self
            .conversation_store
            .get_conversation_by_session_id(session_id)?
        else {
            return Ok(AssembleResult {
                messages: Vec::new(),
                estimated_tokens: 0,
                system_prompt_addition: None,
            });
        };

        let assembled = self
            .assembler
            .assemble(crate::assembler::AssembleContextInput {
                conversation_id: conversation.conversation_id,
                token_budget: token_budget.max(1),
                fresh_tail_count: None,
            })
            .await?;

        Ok(AssembleResult {
            messages: assembled.messages,
            estimated_tokens: assembled.estimated_tokens,
            system_prompt_addition: assembled.system_prompt_addition,
        })
    }

    pub fn assemble_opaque_refs(
        &self,
        session_id: &str,
        token_budget: i64,
    ) -> anyhow::Result<Vec<OpaqueContextRef>> {
        let Some(conversation) = self
            .conversation_store
            .get_conversation_by_session_id(session_id)?
        else {
            return Ok(Vec::new());
        };
        let items = self
            .summary_store
            .get_context_items(conversation.conversation_id)?;
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let mut candidates = Vec::new();
        for item in items {
            let candidate = match item.item_type {
                ContextItemType::Message => {
                    let Some(message_id) = item.message_id else {
                        continue;
                    };
                    let Some(message) = self.conversation_store.get_message_by_id(message_id)?
                    else {
                        continue;
                    };
                    OpaqueContextRefCandidate {
                        ordinal: item.ordinal,
                        token_count: message.token_count.max(0),
                        reference: OpaqueContextRef {
                            reference: format!("lcm:untrusted:message:{message_id}"),
                            source: "message".to_string(),
                            created_at: format_time(message.created_at),
                            summary_kind: None,
                            depth: None,
                            token_count: message.token_count.max(0),
                        },
                    }
                }
                ContextItemType::Summary => {
                    let Some(summary_id) = item.summary_id.clone() else {
                        continue;
                    };
                    let Some(summary) = self.summary_store.get_summary(&summary_id)? else {
                        continue;
                    };
                    OpaqueContextRefCandidate {
                        ordinal: item.ordinal,
                        token_count: summary.token_count.max(0),
                        reference: OpaqueContextRef {
                            reference: format!("lcm:untrusted:summary:{summary_id}"),
                            source: "summary".to_string(),
                            created_at: format_time(summary.created_at),
                            summary_kind: Some(summary_kind_name(&summary.kind).to_string()),
                            depth: Some(summary.depth),
                            token_count: summary.token_count.max(0),
                        },
                    }
                }
            };
            candidates.push(candidate);
        }

        Ok(select_newest_with_budget(candidates, token_budget))
    }
}

fn ensure_migrated(shared: &SharedConnection) -> anyhow::Result<()> {
    let conn = shared.conn.lock();
    run_lcm_migrations(&conn).context("run lcm migrations")
}

fn estimate_tokens(content: &str) -> i64 {
    ((content.chars().count() as f64) / 4.0).ceil() as i64
}

fn format_time(value: DateTime<Utc>) -> String {
    value.to_rfc3339()
}

fn summary_kind_name(kind: &SummaryKind) -> &'static str {
    match kind {
        SummaryKind::Leaf => "leaf",
        SummaryKind::Condensed => "condensed",
    }
}

fn select_newest_with_budget(
    candidates: Vec<OpaqueContextRefCandidate>,
    token_budget: i64,
) -> Vec<OpaqueContextRef> {
    if token_budget <= 0 {
        return candidates
            .into_iter()
            .map(|candidate| candidate.reference)
            .collect();
    }

    let mut selected = Vec::new();
    let mut tokens = 0_i64;
    for candidate in candidates.into_iter().rev() {
        if !selected.is_empty() && tokens + candidate.token_count > token_budget {
            break;
        }
        tokens += candidate.token_count;
        selected.push(candidate);
    }
    selected.sort_by_key(|candidate| candidate.ordinal);
    selected
        .into_iter()
        .map(|candidate| candidate.reference)
        .collect()
}
