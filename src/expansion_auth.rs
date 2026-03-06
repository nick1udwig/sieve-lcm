use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use uuid::Uuid;

use crate::expansion::{ExpansionOrchestrator, ExpansionRequest, ExpansionResult};

#[derive(Clone, Debug, PartialEq)]
pub struct ExpansionGrant {
    pub grant_id: String,
    pub issuer_session_id: String,
    pub allowed_conversation_ids: Vec<i64>,
    pub allowed_summary_ids: Vec<String>,
    pub max_depth: i64,
    pub token_cap: i64,
    pub expires_at: DateTime<Utc>,
    pub revoked: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CreateGrantInput {
    pub issuer_session_id: String,
    pub allowed_conversation_ids: Vec<i64>,
    pub allowed_summary_ids: Option<Vec<String>>,
    pub max_depth: Option<i64>,
    pub token_cap: Option<i64>,
    pub ttl_ms: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CreateDelegatedExpansionGrantInput {
    pub delegated_session_key: String,
    pub issuer_session_id: String,
    pub allowed_conversation_ids: Vec<i64>,
    pub allowed_summary_ids: Option<Vec<String>>,
    pub max_depth: Option<i64>,
    pub token_cap: Option<i64>,
    pub ttl_ms: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValidationResult {
    pub valid: bool,
    pub reason: Option<String>,
}

const DEFAULT_MAX_DEPTH: i64 = 3;
const DEFAULT_TOKEN_CAP: i64 = 4_000;
const DEFAULT_TTL_MS: i64 = 5 * 60 * 1000;

#[derive(Default)]
pub struct ExpansionAuthManager {
    grants: HashMap<String, ExpansionGrant>,
    consumed_tokens_by_grant_id: HashMap<String, i64>,
}

impl ExpansionAuthManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_grant(&mut self, input: CreateGrantInput) -> ExpansionGrant {
        let now = Utc::now();
        let grant = ExpansionGrant {
            grant_id: format!(
                "grant_{}",
                Uuid::new_v4().simple().to_string()[..12].to_string()
            ),
            issuer_session_id: input.issuer_session_id,
            allowed_conversation_ids: input.allowed_conversation_ids,
            allowed_summary_ids: input.allowed_summary_ids.unwrap_or_default(),
            max_depth: input.max_depth.unwrap_or(DEFAULT_MAX_DEPTH),
            token_cap: input.token_cap.unwrap_or(DEFAULT_TOKEN_CAP),
            expires_at: now + Duration::milliseconds(input.ttl_ms.unwrap_or(DEFAULT_TTL_MS)),
            revoked: false,
            created_at: now,
        };
        self.consumed_tokens_by_grant_id
            .insert(grant.grant_id.clone(), 0);
        self.grants.insert(grant.grant_id.clone(), grant.clone());
        grant
    }

    pub fn get_grant(&self, grant_id: &str) -> Option<ExpansionGrant> {
        let grant = self.grants.get(grant_id)?;
        if grant.revoked {
            return None;
        }
        if grant.expires_at <= Utc::now() {
            return None;
        }
        Some(grant.clone())
    }

    pub fn revoke_grant(&mut self, grant_id: &str) -> bool {
        let Some(grant) = self.grants.get_mut(grant_id) else {
            return false;
        };
        grant.revoked = true;
        true
    }

    pub fn get_remaining_token_budget(&self, grant_id: &str) -> Option<i64> {
        let grant = self.get_grant(grant_id)?;
        let consumed = self
            .consumed_tokens_by_grant_id
            .get(grant_id)
            .copied()
            .unwrap_or(0)
            .max(0);
        Some((grant.token_cap.max(0) - consumed).max(0))
    }

    pub fn consume_token_budget(&mut self, grant_id: &str, consumed_tokens: i64) -> Option<i64> {
        let grant = self.get_grant(grant_id)?;
        let safe = consumed_tokens.max(0);
        let previous = self
            .consumed_tokens_by_grant_id
            .get(grant_id)
            .copied()
            .unwrap_or(0)
            .max(0);
        let next = (previous + safe).min(grant.token_cap.max(1));
        self.consumed_tokens_by_grant_id
            .insert(grant_id.to_string(), next);
        Some((grant.token_cap - next).max(0))
    }

    pub fn validate_expansion(
        &self,
        grant_id: &str,
        conversation_id: i64,
        summary_ids: &[String],
        _depth: i64,
        _token_cap: i64,
    ) -> ValidationResult {
        let Some(grant) = self.grants.get(grant_id) else {
            return ValidationResult {
                valid: false,
                reason: Some("Grant not found".to_string()),
            };
        };
        if grant.revoked {
            return ValidationResult {
                valid: false,
                reason: Some("Grant has been revoked".to_string()),
            };
        }
        if grant.expires_at <= Utc::now() {
            return ValidationResult {
                valid: false,
                reason: Some("Grant has expired".to_string()),
            };
        }
        if !grant.allowed_conversation_ids.contains(&conversation_id) {
            return ValidationResult {
                valid: false,
                reason: Some(format!(
                    "Conversation {} is not in the allowed set",
                    conversation_id
                )),
            };
        }
        if !grant.allowed_summary_ids.is_empty() {
            let allowed: std::collections::HashSet<String> =
                grant.allowed_summary_ids.iter().cloned().collect();
            let unauthorized: Vec<String> = summary_ids
                .iter()
                .filter(|id| !allowed.contains(*id))
                .cloned()
                .collect();
            if !unauthorized.is_empty() {
                return ValidationResult {
                    valid: false,
                    reason: Some(format!(
                        "Summary IDs not authorized: {}",
                        unauthorized.join(", ")
                    )),
                };
            }
        }
        ValidationResult {
            valid: true,
            reason: None,
        }
    }

    pub fn cleanup(&mut self) -> usize {
        let now = Utc::now();
        let mut remove = vec![];
        for (grant_id, grant) in &self.grants {
            if grant.revoked || grant.expires_at <= now {
                remove.push(grant_id.clone());
            }
        }
        let removed = remove.len();
        for key in remove {
            self.grants.remove(&key);
            self.consumed_tokens_by_grant_id.remove(&key);
        }
        removed
    }
}

static RUNTIME_MANAGER: Lazy<Arc<Mutex<ExpansionAuthManager>>> =
    Lazy::new(|| Arc::new(Mutex::new(ExpansionAuthManager::new())));
static DELEGATED_SESSION_GRANTS: Lazy<Mutex<HashMap<String, String>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn get_runtime_expansion_auth_manager() -> Arc<Mutex<ExpansionAuthManager>> {
    RUNTIME_MANAGER.clone()
}

pub fn create_delegated_expansion_grant(
    input: CreateDelegatedExpansionGrantInput,
) -> ExpansionGrant {
    let delegated_session_key = input.delegated_session_key.trim().to_string();
    assert!(
        !delegated_session_key.is_empty(),
        "delegatedSessionKey is required for delegated expansion grants"
    );
    let mut manager = RUNTIME_MANAGER.lock();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: input.issuer_session_id,
        allowed_conversation_ids: input.allowed_conversation_ids,
        allowed_summary_ids: input.allowed_summary_ids,
        max_depth: input.max_depth,
        token_cap: input.token_cap,
        ttl_ms: input.ttl_ms,
    });
    DELEGATED_SESSION_GRANTS
        .lock()
        .insert(delegated_session_key, grant.grant_id.clone());
    grant
}

pub fn resolve_delegated_expansion_grant_id(session_key: &str) -> Option<String> {
    let key = session_key.trim();
    if key.is_empty() {
        return None;
    }
    DELEGATED_SESSION_GRANTS.lock().get(key).cloned()
}

pub fn revoke_delegated_expansion_grant_for_session(
    session_key: &str,
    remove_binding: bool,
) -> bool {
    let key = session_key.trim();
    if key.is_empty() {
        return false;
    }
    let grant_id = DELEGATED_SESSION_GRANTS.lock().get(key).cloned();
    let Some(grant_id) = grant_id else {
        return false;
    };
    let mut manager = RUNTIME_MANAGER.lock();
    let revoked = manager.revoke_grant(&grant_id);
    if remove_binding {
        DELEGATED_SESSION_GRANTS.lock().remove(key);
    }
    revoked
}

pub fn remove_delegated_expansion_grant_for_session(session_key: &str) -> bool {
    let key = session_key.trim();
    if key.is_empty() {
        return false;
    }
    DELEGATED_SESSION_GRANTS.lock().remove(key).is_some()
}

pub fn reset_delegated_expansion_grants_for_tests() {
    DELEGATED_SESSION_GRANTS.lock().clear();
}

#[derive(Clone)]
pub struct AuthorizedExpansionOrchestrator {
    orchestrator: Arc<ExpansionOrchestrator>,
    auth_manager: Arc<Mutex<ExpansionAuthManager>>,
}

impl AuthorizedExpansionOrchestrator {
    pub async fn expand(
        &self,
        grant_id: &str,
        request: ExpansionRequest,
    ) -> anyhow::Result<ExpansionResult> {
        let manager = self.auth_manager.lock();
        let validation = manager.validate_expansion(
            grant_id,
            request.conversation_id,
            &request.summary_ids,
            request.max_depth.unwrap_or(DEFAULT_MAX_DEPTH),
            request.token_cap.unwrap_or(DEFAULT_TOKEN_CAP),
        );
        if !validation.valid {
            anyhow::bail!(
                "Expansion authorization failed: {}",
                validation.reason.unwrap_or_else(|| "unknown".to_string())
            );
        }

        let remaining = manager
            .get_remaining_token_budget(grant_id)
            .ok_or_else(|| anyhow::anyhow!("Expansion authorization failed: Grant not found"))?;
        if remaining <= 0 {
            anyhow::bail!("Expansion authorization failed: Grant token budget exhausted");
        }
        let requested = request.token_cap.unwrap_or(remaining).max(1);
        let effective = requested.min(remaining).max(1);
        drop(manager);

        let result = self
            .orchestrator
            .expand(ExpansionRequest {
                token_cap: Some(effective),
                ..request
            })
            .await?;
        self.auth_manager
            .lock()
            .consume_token_budget(grant_id, result.total_tokens);
        Ok(result)
    }
}

pub fn wrap_with_auth(
    orchestrator: Arc<ExpansionOrchestrator>,
    auth_manager: Arc<Mutex<ExpansionAuthManager>>,
) -> AuthorizedExpansionOrchestrator {
    AuthorizedExpansionOrchestrator {
        orchestrator,
        auth_manager,
    }
}
