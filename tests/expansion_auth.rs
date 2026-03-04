use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use parking_lot::Mutex;
use sieve_lcm::expansion::{distill_for_subagent, ExpansionOrchestrator, ExpansionRequest, ExpansionResult};
use sieve_lcm::expansion_auth::{wrap_with_auth, CreateGrantInput, ExpansionAuthManager};
use sieve_lcm::retrieval::{
    DescribeResult, ExpandInput, ExpandResult, ExpandedChild, ExpandedMessage, GrepInput, GrepResult,
    RetrievalApi,
};
use sieve_lcm::store::summary_store::{SummaryKind, SummarySearchResult};

#[derive(Default)]
struct MockRetrieval {
    grep_calls: Mutex<Vec<GrepInput>>,
    expand_calls: Mutex<Vec<ExpandInput>>,
    expand_results: Mutex<Vec<ExpandResult>>,
    grep_result: Mutex<Option<GrepResult>>,
}

#[async_trait]
impl RetrievalApi for MockRetrieval {
    async fn describe(&self, _id: &str) -> anyhow::Result<Option<DescribeResult>> {
        Ok(None)
    }

    async fn grep(&self, input: GrepInput) -> anyhow::Result<GrepResult> {
        self.grep_calls.lock().push(input);
        Ok(self.grep_result.lock().clone().unwrap_or(GrepResult {
            messages: vec![],
            summaries: vec![],
            total_matches: 0,
        }))
    }

    async fn expand(&self, input: ExpandInput) -> anyhow::Result<ExpandResult> {
        self.expand_calls.lock().push(input);
        let mut results = self.expand_results.lock();
        if results.is_empty() {
            return Ok(default_expand_result());
        }
        Ok(results.remove(0))
    }
}

fn default_expand_result() -> ExpandResult {
    ExpandResult {
        children: vec![],
        messages: vec![],
        estimated_tokens: 0,
        truncated: false,
    }
}

fn default_expansion_result() -> ExpansionResult {
    ExpansionResult {
        expansions: vec![],
        cited_ids: vec![],
        total_tokens: 0,
        truncated: false,
    }
}

fn parse_utc(value: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(value)
        .expect("valid rfc3339")
        .with_timezone(&Utc)
}

#[test]
fn creates_a_grant_with_default_values() {
    let mut manager = ExpansionAuthManager::new();
    let before = Utc::now();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1, 2],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: None,
    });
    let after = Utc::now();

    assert!(grant.grant_id.starts_with("grant_"));
    assert_eq!(grant.issuer_session_id, "sess1");
    assert_eq!(grant.allowed_conversation_ids, vec![1, 2]);
    assert_eq!(grant.allowed_summary_ids, Vec::<String>::new());
    assert_eq!(grant.max_depth, 3);
    assert_eq!(grant.token_cap, 4000);
    assert!(!grant.revoked);

    let lower = before + Duration::minutes(5) - Duration::milliseconds(200);
    let upper = after + Duration::minutes(5) + Duration::milliseconds(200);
    assert!(grant.expires_at >= lower);
    assert!(grant.expires_at <= upper);
}

#[test]
fn creates_a_grant_with_custom_values() {
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess2".to_string(),
        allowed_conversation_ids: vec![10],
        allowed_summary_ids: Some(vec!["sum_a".to_string(), "sum_b".to_string()]),
        max_depth: Some(5),
        token_cap: Some(8000),
        ttl_ms: Some(60_000),
    });

    assert_eq!(grant.max_depth, 5);
    assert_eq!(grant.token_cap, 8000);
    assert_eq!(grant.allowed_summary_ids, vec!["sum_a", "sum_b"]);
    let delta = grant.expires_at - grant.created_at;
    assert!(delta >= Duration::milliseconds(59_900));
    assert!(delta <= Duration::milliseconds(60_100));
}

#[test]
fn returns_a_valid_grant() {
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: None,
    });
    assert_eq!(manager.get_grant(&grant.grant_id), Some(grant));
}

#[test]
fn returns_null_for_expired_grant() {
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: Some(0),
    });
    assert!(manager.get_grant(&grant.grant_id).is_none());
}

#[test]
fn returns_null_for_a_grant_with_negative_ttl() {
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: Some(-1000),
    });
    assert!(manager.get_grant(&grant.grant_id).is_none());
}

#[test]
fn returns_null_for_revoked_grant() {
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: None,
    });
    manager.revoke_grant(&grant.grant_id);
    assert!(manager.get_grant(&grant.grant_id).is_none());
}

#[test]
fn returns_null_for_unknown_grant_id() {
    let manager = ExpansionAuthManager::new();
    assert!(manager.get_grant("grant_doesnotexist").is_none());
}

#[test]
fn returns_true_when_revoking_existing_grant() {
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: None,
    });
    assert!(manager.revoke_grant(&grant.grant_id));
}

#[test]
fn returns_false_when_revoking_unknown_grant() {
    let mut manager = ExpansionAuthManager::new();
    assert!(!manager.revoke_grant("grant_nope"));
}

#[test]
fn makes_the_grant_inaccessible_via_get_grant() {
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: None,
    });
    manager.revoke_grant(&grant.grant_id);
    assert!(manager.get_grant(&grant.grant_id).is_none());
}

fn scoped_grant(manager: &mut ExpansionAuthManager) -> String {
    manager
        .create_grant(CreateGrantInput {
            issuer_session_id: "sess1".to_string(),
            allowed_conversation_ids: vec![1],
            allowed_summary_ids: Some(vec!["sum_a".to_string(), "sum_b".to_string()]),
            max_depth: Some(3),
            token_cap: Some(4000),
            ttl_ms: None,
        })
        .grant_id
}

#[test]
fn accepts_valid_request_within_scope() {
    let mut manager = ExpansionAuthManager::new();
    let grant_id = scoped_grant(&mut manager);
    let result = manager.validate_expansion(&grant_id, 1, &["sum_a".to_string()], 2, 2000);
    assert!(result.valid);
    assert!(result.reason.is_none());
}

#[test]
fn accepts_request_at_exact_depth_and_token_limits() {
    let mut manager = ExpansionAuthManager::new();
    let grant_id = scoped_grant(&mut manager);
    let result = manager.validate_expansion(
        &grant_id,
        1,
        &["sum_a".to_string(), "sum_b".to_string()],
        3,
        4000,
    );
    assert!(result.valid);
}

#[test]
fn rejects_unknown_grant() {
    let manager = ExpansionAuthManager::new();
    let result = manager.validate_expansion("grant_fake", 1, &["sum_a".to_string()], 1, 1000);
    assert!(!result.valid);
    assert!(result.reason.unwrap_or_default().contains("not found"));
}

#[test]
fn rejects_expired_grant() {
    let mut manager = ExpansionAuthManager::new();
    let expired = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: Some(0),
    });
    let result = manager.validate_expansion(&expired.grant_id, 1, &[], 1, 1000);
    assert!(!result.valid);
    assert!(result.reason.unwrap_or_default().contains("expired"));
}

#[test]
fn rejects_revoked_grant() {
    let mut manager = ExpansionAuthManager::new();
    let grant_id = scoped_grant(&mut manager);
    manager.revoke_grant(&grant_id);
    let result = manager.validate_expansion(&grant_id, 1, &["sum_a".to_string()], 1, 1000);
    assert!(!result.valid);
    assert!(result.reason.unwrap_or_default().contains("revoked"));
}

#[test]
fn rejects_unauthorized_conversation_id() {
    let mut manager = ExpansionAuthManager::new();
    let grant_id = scoped_grant(&mut manager);
    let result = manager.validate_expansion(&grant_id, 999, &["sum_a".to_string()], 1, 1000);
    assert!(!result.valid);
    assert!(result.reason.unwrap_or_default().contains("Conversation"));
}

#[test]
fn rejects_unauthorized_summary_ids() {
    let mut manager = ExpansionAuthManager::new();
    let grant_id = scoped_grant(&mut manager);
    let result = manager.validate_expansion(&grant_id, 1, &["sum_c".to_string()], 1, 1000);
    assert!(!result.valid);
    assert!(result.reason.unwrap_or_default().contains("Summary"));
}

#[test]
fn rejects_when_some_summary_ids_are_authorized_and_some_are_not() {
    let mut manager = ExpansionAuthManager::new();
    let grant_id = scoped_grant(&mut manager);
    let result = manager.validate_expansion(
        &grant_id,
        1,
        &["sum_a".to_string(), "sum_c".to_string(), "sum_d".to_string()],
        1,
        1000,
    );
    assert!(!result.valid);
    let reason = result.reason.unwrap_or_default();
    assert!(reason.contains("sum_c"));
    assert!(reason.contains("sum_d"));
}

#[test]
fn allows_any_summary_ids_when_allowed_summary_ids_is_empty() {
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: Some(vec![]),
        max_depth: None,
        token_cap: None,
        ttl_ms: None,
    });
    let result = manager.validate_expansion(
        &grant.grant_id,
        1,
        &["sum_anything".to_string(), "sum_everything".to_string()],
        1,
        1000,
    );
    assert!(result.valid);
}

#[test]
fn allows_any_summary_ids_when_allowed_summary_ids_is_omitted_defaults_to_empty() {
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: None,
    });
    let result = manager.validate_expansion(
        &grant.grant_id,
        1,
        &["sum_x".to_string(), "sum_y".to_string(), "sum_z".to_string()],
        1,
        1000,
    );
    assert!(result.valid);
}

#[test]
fn does_not_enforce_max_depth_against_grant_limits() {
    let mut manager = ExpansionAuthManager::new();
    let grant_id = scoped_grant(&mut manager);
    let result = manager.validate_expansion(&grant_id, 1, &["sum_a".to_string()], 5, 1000);
    assert!(result.valid);
    assert!(result.reason.is_none());
}

#[test]
fn does_not_enforce_token_cap_against_grant_limits() {
    let mut manager = ExpansionAuthManager::new();
    let grant_id = scoped_grant(&mut manager);
    let result = manager.validate_expansion(&grant_id, 1, &["sum_a".to_string()], 1, 5000);
    assert!(result.valid);
    assert!(result.reason.is_none());
}

#[test]
fn checks_validation_in_priority_order_existence_revocation_expiry_scope() {
    let manager = ExpansionAuthManager::new();
    let result = manager.validate_expansion("grant_nope", 999, &["sum_c".to_string()], 100, 999_999);
    assert!(result.reason.unwrap_or_default().contains("not found"));
}

#[test]
fn removes_expired_grants() {
    let mut manager = ExpansionAuthManager::new();
    manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: Some(0),
    });
    let active = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![2],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: Some(300_000),
    });

    let removed = manager.cleanup();
    assert_eq!(removed, 1);
    assert!(manager.get_grant(&active.grant_id).is_some());
}

#[test]
fn removes_revoked_grants() {
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: None,
    });
    manager.revoke_grant(&grant.grant_id);
    assert_eq!(manager.cleanup(), 1);
}

#[test]
fn removes_both_expired_and_revoked_grants_in_one_pass() {
    let mut manager = ExpansionAuthManager::new();
    manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: Some(0),
    });
    let revoked = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![2],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: None,
    });
    manager.revoke_grant(&revoked.grant_id);
    manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![3],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: Some(300_000),
    });
    assert_eq!(manager.cleanup(), 2);
}

#[test]
fn returns_0_when_nothing_to_clean() {
    let mut manager = ExpansionAuthManager::new();
    manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: None,
    });
    assert_eq!(manager.cleanup(), 0);
}

#[tokio::test]
async fn delegates_to_orchestrator_when_grant_is_valid() {
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval.expand_results.lock().push(ExpandResult {
        children: vec![],
        messages: vec![],
        estimated_tokens: 100,
        truncated: false,
    });

    let orchestrator = Arc::new(ExpansionOrchestrator::new(retrieval.clone()));
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: Some(vec![]),
        max_depth: None,
        token_cap: Some(4000),
        ttl_ms: None,
    });

    let authorized = wrap_with_auth(orchestrator, Arc::new(Mutex::new(manager)));
    let request = ExpansionRequest {
        summary_ids: vec!["sum_a".to_string()],
        conversation_id: 1,
        max_depth: Some(2),
        token_cap: Some(2000),
        include_messages: None,
    };
    let result = authorized
        .expand(&grant.grant_id, request)
        .await
        .expect("authorized expand");

    assert_eq!(retrieval.expand_calls.lock().len(), 1);
    assert_eq!(result.total_tokens, 100);
}

#[tokio::test]
async fn throws_when_grant_is_invalid() {
    let retrieval = Arc::new(MockRetrieval::default());
    let orchestrator = Arc::new(ExpansionOrchestrator::new(retrieval.clone()));
    let authorized = wrap_with_auth(orchestrator, Arc::new(Mutex::new(ExpansionAuthManager::new())));
    let request = ExpansionRequest {
        summary_ids: vec!["sum_a".to_string()],
        conversation_id: 1,
        max_depth: Some(1),
        token_cap: Some(1000),
        include_messages: None,
    };
    let err = authorized
        .expand("grant_fake", request)
        .await
        .expect_err("should reject");
    assert!(err.to_string().to_lowercase().contains("authorization failed"));
    assert!(retrieval.expand_calls.lock().is_empty());
}

#[tokio::test]
async fn throws_when_grant_is_expired() {
    let retrieval = Arc::new(MockRetrieval::default());
    let orchestrator = Arc::new(ExpansionOrchestrator::new(retrieval.clone()));
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: Some(0),
    });
    let authorized = wrap_with_auth(orchestrator, Arc::new(Mutex::new(manager)));
    let err = authorized
        .expand(
            &grant.grant_id,
            ExpansionRequest {
                summary_ids: vec![],
                conversation_id: 1,
                max_depth: Some(1),
                token_cap: Some(1000),
                include_messages: None,
            },
        )
        .await
        .expect_err("should reject expired");
    assert!(err.to_string().to_lowercase().contains("expired"));
    assert!(retrieval.expand_calls.lock().is_empty());
}

#[tokio::test]
async fn throws_when_grant_is_revoked() {
    let retrieval = Arc::new(MockRetrieval::default());
    let orchestrator = Arc::new(ExpansionOrchestrator::new(retrieval.clone()));
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: None,
        ttl_ms: None,
    });
    manager.revoke_grant(&grant.grant_id);
    let authorized = wrap_with_auth(orchestrator, Arc::new(Mutex::new(manager)));
    let err = authorized
        .expand(
            &grant.grant_id,
            ExpansionRequest {
                summary_ids: vec![],
                conversation_id: 1,
                max_depth: Some(1),
                token_cap: Some(1000),
                include_messages: None,
            },
        )
        .await
        .expect_err("should reject revoked");
    assert!(err.to_string().to_lowercase().contains("revoked"));
}

#[tokio::test]
async fn passes_through_explicit_token_cap_values() {
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval.expand_results.lock().push(default_expand_result());
    let orchestrator = Arc::new(ExpansionOrchestrator::new(retrieval.clone()));
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: Some(vec![]),
        max_depth: None,
        token_cap: Some(1000),
        ttl_ms: None,
    });

    let authorized = wrap_with_auth(orchestrator, Arc::new(Mutex::new(manager)));
    authorized
        .expand(
            &grant.grant_id,
            ExpansionRequest {
                summary_ids: vec!["sum_a".to_string()],
                conversation_id: 1,
                max_depth: Some(2),
                token_cap: Some(800),
                include_messages: None,
            },
        )
        .await
        .expect("authorized expand");

    let calls = retrieval.expand_calls.lock();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].token_cap, Some(800));
}

#[tokio::test]
async fn injects_remaining_token_cap_when_request_omits_it() {
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval.expand_results.lock().push(default_expand_result());
    let orchestrator = Arc::new(ExpansionOrchestrator::new(retrieval.clone()));
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: Some(vec![]),
        max_depth: None,
        token_cap: Some(4000),
        ttl_ms: None,
    });
    let authorized = wrap_with_auth(orchestrator, Arc::new(Mutex::new(manager)));

    authorized
        .expand(
            &grant.grant_id,
            ExpansionRequest {
                summary_ids: vec!["sum_a".to_string()],
                conversation_id: 1,
                max_depth: None,
                token_cap: None,
                include_messages: None,
            },
        )
        .await
        .expect("authorized expand");

    let calls = retrieval.expand_calls.lock();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].token_cap, Some(4000));
}

#[tokio::test]
async fn clamps_requested_token_cap_to_remaining_grant_budget() {
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval.expand_results.lock().push(default_expand_result());
    retrieval.expand_results.lock().push(default_expand_result());
    {
        let mut results = retrieval.expand_results.lock();
        results[0].estimated_tokens = 700;
        results[1].estimated_tokens = 200;
    }

    let orchestrator = Arc::new(ExpansionOrchestrator::new(retrieval.clone()));
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: Some(1000),
        ttl_ms: None,
    });
    let authorized = wrap_with_auth(orchestrator, Arc::new(Mutex::new(manager)));

    authorized
        .expand(
            &grant.grant_id,
            ExpansionRequest {
                summary_ids: vec!["sum_a".to_string()],
                conversation_id: 1,
                max_depth: None,
                token_cap: Some(700),
                include_messages: None,
            },
        )
        .await
        .expect("first");

    authorized
        .expand(
            &grant.grant_id,
            ExpansionRequest {
                summary_ids: vec!["sum_b".to_string()],
                conversation_id: 1,
                max_depth: None,
                token_cap: Some(900),
                include_messages: None,
            },
        )
        .await
        .expect("second");

    let calls = retrieval.expand_calls.lock();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].token_cap, Some(700));
    assert_eq!(calls[1].token_cap, Some(300));
}

#[tokio::test]
async fn fails_when_grant_token_budget_is_exhausted_across_calls() {
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval.expand_results.lock().push(ExpandResult {
        estimated_tokens: 500,
        ..default_expand_result()
    });
    let orchestrator = Arc::new(ExpansionOrchestrator::new(retrieval));
    let mut manager = ExpansionAuthManager::new();
    let grant = manager.create_grant(CreateGrantInput {
        issuer_session_id: "sess1".to_string(),
        allowed_conversation_ids: vec![1],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap: Some(500),
        ttl_ms: None,
    });
    let authorized = wrap_with_auth(orchestrator, Arc::new(Mutex::new(manager)));
    authorized
        .expand(
            &grant.grant_id,
            ExpansionRequest {
                summary_ids: vec!["sum_a".to_string()],
                conversation_id: 1,
                max_depth: None,
                token_cap: Some(500),
                include_messages: None,
            },
        )
        .await
        .expect("first");

    let err = authorized
        .expand(
            &grant.grant_id,
            ExpansionRequest {
                summary_ids: vec!["sum_b".to_string()],
                conversation_id: 1,
                max_depth: None,
                token_cap: Some(50),
                include_messages: None,
            },
        )
        .await
        .expect_err("should be exhausted");
    assert!(err.to_string().to_lowercase().contains("budget exhausted"));
}

#[tokio::test]
async fn expands_multiple_summary_ids_and_collects_cited_ids() {
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval.expand_results.lock().push(ExpandResult {
        children: vec![ExpandedChild {
            summary_id: "sum_child_1".to_string(),
            kind: SummaryKind::Leaf,
            content: "child 1 content".to_string(),
            token_count: 50,
        }],
        estimated_tokens: 50,
        ..default_expand_result()
    });
    retrieval.expand_results.lock().push(ExpandResult {
        children: vec![ExpandedChild {
            summary_id: "sum_child_2".to_string(),
            kind: SummaryKind::Leaf,
            content: "child 2 content".to_string(),
            token_count: 60,
        }],
        estimated_tokens: 60,
        ..default_expand_result()
    });
    let orchestrator = ExpansionOrchestrator::new(retrieval);

    let result = orchestrator
        .expand(ExpansionRequest {
            summary_ids: vec!["sum_a".to_string(), "sum_b".to_string()],
            conversation_id: 1,
            max_depth: Some(2),
            token_cap: None,
            include_messages: None,
        })
        .await
        .expect("expand");

    assert_eq!(result.expansions.len(), 2);
    assert_eq!(result.expansions[0].summary_id, "sum_a");
    assert_eq!(result.expansions[0].children.len(), 1);
    assert_eq!(result.expansions[1].summary_id, "sum_b");
    assert_eq!(result.expansions[1].children.len(), 1);
    assert_eq!(result.total_tokens, 110);
    assert!(!result.truncated);
    assert!(result.cited_ids.contains(&"sum_a".to_string()));
    assert!(result.cited_ids.contains(&"sum_child_1".to_string()));
    assert!(result.cited_ids.contains(&"sum_b".to_string()));
    assert!(result.cited_ids.contains(&"sum_child_2".to_string()));
}

#[tokio::test]
async fn passes_correct_arguments_to_retrieval_expand() {
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval.expand_results.lock().push(default_expand_result());
    let orchestrator = ExpansionOrchestrator::new(retrieval.clone());

    orchestrator
        .expand(ExpansionRequest {
            summary_ids: vec!["sum_a".to_string()],
            conversation_id: 1,
            max_depth: Some(5),
            token_cap: Some(3000),
            include_messages: Some(true),
        })
        .await
        .expect("expand");

    let calls = retrieval.expand_calls.lock();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0],
        ExpandInput {
            summary_id: "sum_a".to_string(),
            depth: Some(5),
            include_messages: Some(true),
            token_cap: Some(3000),
        }
    );
}

#[tokio::test]
async fn enforces_global_token_cap_across_multiple_expansions() {
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval.expand_results.lock().push(ExpandResult {
        children: vec![ExpandedChild {
            summary_id: "sum_c1".to_string(),
            kind: SummaryKind::Leaf,
            content: "big content".to_string(),
            token_count: 900,
        }],
        estimated_tokens: 900,
        ..default_expand_result()
    });
    retrieval.expand_results.lock().push(ExpandResult {
        children: vec![ExpandedChild {
            summary_id: "sum_c2".to_string(),
            kind: SummaryKind::Leaf,
            content: "more content".to_string(),
            token_count: 50,
        }],
        estimated_tokens: 50,
        truncated: true,
        ..default_expand_result()
    });
    let orchestrator = ExpansionOrchestrator::new(retrieval.clone());

    let result = orchestrator
        .expand(ExpansionRequest {
            summary_ids: vec!["sum_a".to_string(), "sum_b".to_string()],
            conversation_id: 1,
            max_depth: None,
            token_cap: Some(1000),
            include_messages: None,
        })
        .await
        .expect("expand");

    let calls = retrieval.expand_calls.lock();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1].token_cap, Some(100));
    assert!(result.truncated);
    assert_eq!(result.total_tokens, 950);
}

#[tokio::test]
async fn stops_expanding_when_budget_is_exhausted() {
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval.expand_results.lock().push(ExpandResult {
        estimated_tokens: 500,
        ..default_expand_result()
    });
    let orchestrator = ExpansionOrchestrator::new(retrieval.clone());

    let result = orchestrator
        .expand(ExpansionRequest {
            summary_ids: vec!["sum_a".to_string(), "sum_b".to_string(), "sum_c".to_string()],
            conversation_id: 1,
            max_depth: None,
            token_cap: Some(500),
            include_messages: None,
        })
        .await
        .expect("expand");

    assert_eq!(retrieval.expand_calls.lock().len(), 1);
    assert!(result.truncated);
}

#[tokio::test]
async fn handles_expansion_with_messages() {
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval.expand_results.lock().push(ExpandResult {
        messages: vec![
            ExpandedMessage {
                message_id: 10,
                role: "user".to_string(),
                content: "Hello world".to_string(),
                token_count: 3,
            },
            ExpandedMessage {
                message_id: 11,
                role: "assistant".to_string(),
                content: "Hi there".to_string(),
                token_count: 2,
            },
        ],
        estimated_tokens: 5,
        ..default_expand_result()
    });
    let orchestrator = ExpansionOrchestrator::new(retrieval);

    let result = orchestrator
        .expand(ExpansionRequest {
            summary_ids: vec!["sum_leaf".to_string()],
            conversation_id: 1,
            max_depth: None,
            token_cap: None,
            include_messages: Some(true),
        })
        .await
        .expect("expand");

    assert_eq!(result.expansions[0].messages.len(), 2);
    assert_eq!(result.expansions[0].messages[0].message_id, 10);
    assert_eq!(result.expansions[0].messages[0].role, "user");
    assert_eq!(result.total_tokens, 5);
}

#[tokio::test]
async fn truncates_long_content_to_snippets() {
    let retrieval = Arc::new(MockRetrieval::default());
    retrieval.expand_results.lock().push(ExpandResult {
        children: vec![ExpandedChild {
            summary_id: "sum_c1".to_string(),
            kind: SummaryKind::Leaf,
            content: "x".repeat(300),
            token_count: 75,
        }],
        estimated_tokens: 75,
        ..default_expand_result()
    });
    let orchestrator = ExpansionOrchestrator::new(retrieval);

    let result = orchestrator
        .expand(ExpansionRequest {
            summary_ids: vec!["sum_a".to_string()],
            conversation_id: 1,
            max_depth: None,
            token_cap: None,
            include_messages: None,
        })
        .await
        .expect("expand");

    let snippet = &result.expansions[0].children[0].snippet;
    assert_eq!(snippet.len(), 203);
    assert!(snippet.ends_with("..."));
}

#[tokio::test]
async fn describe_and_expand_greps_then_expands_top_results() {
    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.grep_result.lock() = Some(GrepResult {
        messages: vec![],
        summaries: vec![
            SummarySearchResult {
                summary_id: "sum_match1".to_string(),
                conversation_id: 1,
                kind: SummaryKind::Leaf,
                snippet: "found it".to_string(),
                created_at: parse_utc("2026-01-01T00:00:00.000Z"),
                rank: None,
            },
            SummarySearchResult {
                summary_id: "sum_match2".to_string(),
                conversation_id: 1,
                kind: SummaryKind::Condensed,
                snippet: "also found".to_string(),
                created_at: parse_utc("2026-01-02T00:00:00.000Z"),
                rank: None,
            },
        ],
        total_matches: 2,
    });
    retrieval.expand_results.lock().push(ExpandResult {
        estimated_tokens: 30,
        ..default_expand_result()
    });
    retrieval.expand_results.lock().push(ExpandResult {
        estimated_tokens: 40,
        ..default_expand_result()
    });
    let orchestrator = ExpansionOrchestrator::new(retrieval.clone());

    let result = orchestrator
        .describe_and_expand("search term", "full_text", Some(1), Some(2), Some(5000))
        .await
        .expect("describe_and_expand");

    let grep_calls = retrieval.grep_calls.lock();
    assert_eq!(grep_calls.len(), 1);
    assert_eq!(
        grep_calls[0],
        GrepInput {
            query: "search term".to_string(),
            mode: "full_text".to_string(),
            scope: "summaries".to_string(),
            conversation_id: Some(1),
            since: None,
            before: None,
            limit: None,
        }
    );

    assert_eq!(retrieval.expand_calls.lock().len(), 2);
    assert_eq!(result.expansions.len(), 2);
    assert_eq!(result.total_tokens, 70);
}

#[tokio::test]
async fn describe_and_expand_returns_empty_when_grep_finds_nothing() {
    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.grep_result.lock() = Some(GrepResult {
        messages: vec![],
        summaries: vec![],
        total_matches: 0,
    });
    let orchestrator = ExpansionOrchestrator::new(retrieval.clone());

    let result = orchestrator
        .describe_and_expand("nothing matches", "regex", Some(1), None, None)
        .await
        .expect("describe_and_expand");

    assert_eq!(result.expansions.len(), 0);
    assert_eq!(result.cited_ids.len(), 0);
    assert_eq!(result.total_tokens, 0);
    assert!(!result.truncated);
    assert_eq!(retrieval.expand_calls.lock().len(), 0);
}

#[tokio::test]
async fn describe_and_expand_passes_conversation_id_through_to_expand() {
    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.grep_result.lock() = Some(GrepResult {
        messages: vec![],
        summaries: vec![SummarySearchResult {
            summary_id: "sum_x".to_string(),
            conversation_id: 42,
            kind: SummaryKind::Leaf,
            snippet: "match".to_string(),
            created_at: parse_utc("2026-01-03T00:00:00.000Z"),
            rank: None,
        }],
        total_matches: 1,
    });
    retrieval.expand_results.lock().push(ExpandResult {
        estimated_tokens: 10,
        ..default_expand_result()
    });
    let orchestrator = ExpansionOrchestrator::new(retrieval.clone());

    orchestrator
        .describe_and_expand("test", "full_text", Some(42), None, None)
        .await
        .expect("describe_and_expand");

    let grep_calls = retrieval.grep_calls.lock();
    assert_eq!(grep_calls.len(), 1);
    assert_eq!(grep_calls[0].conversation_id, Some(42));
}

#[tokio::test]
async fn describe_and_expand_biases_expansion_order_toward_newer_summaries() {
    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.grep_result.lock() = Some(GrepResult {
        messages: vec![],
        summaries: vec![
            SummarySearchResult {
                summary_id: "sum_old".to_string(),
                conversation_id: 1,
                kind: SummaryKind::Leaf,
                snippet: "older".to_string(),
                created_at: parse_utc("2026-01-01T00:00:00.000Z"),
                rank: None,
            },
            SummarySearchResult {
                summary_id: "sum_new".to_string(),
                conversation_id: 1,
                kind: SummaryKind::Leaf,
                snippet: "newer".to_string(),
                created_at: parse_utc("2026-01-02T00:00:00.000Z"),
                rank: None,
            },
        ],
        total_matches: 2,
    });
    retrieval.expand_results.lock().push(ExpandResult {
        estimated_tokens: 10,
        ..default_expand_result()
    });
    retrieval.expand_results.lock().push(ExpandResult {
        estimated_tokens: 10,
        ..default_expand_result()
    });
    let orchestrator = ExpansionOrchestrator::new(retrieval.clone());

    orchestrator
        .describe_and_expand("recent first", "full_text", Some(1), None, None)
        .await
        .expect("describe_and_expand");

    let calls = retrieval.expand_calls.lock();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].summary_id, "sum_new");
    assert_eq!(calls[1].summary_id, "sum_old");
}

#[tokio::test]
async fn describe_and_expand_allows_query_mode_without_conversation_id() {
    let retrieval = Arc::new(MockRetrieval::default());
    *retrieval.grep_result.lock() = Some(GrepResult {
        messages: vec![],
        summaries: vec![SummarySearchResult {
            summary_id: "sum_any".to_string(),
            conversation_id: 9,
            kind: SummaryKind::Leaf,
            snippet: "match".to_string(),
            created_at: parse_utc("2026-01-01T00:00:00.000Z"),
            rank: None,
        }],
        total_matches: 1,
    });
    retrieval.expand_results.lock().push(ExpandResult {
        estimated_tokens: 5,
        ..default_expand_result()
    });
    let orchestrator = ExpansionOrchestrator::new(retrieval.clone());

    orchestrator
        .describe_and_expand("global query", "full_text", None, None, None)
        .await
        .expect("describe_and_expand");

    let grep_calls = retrieval.grep_calls.lock();
    assert_eq!(grep_calls.len(), 1);
    assert_eq!(grep_calls[0].conversation_id, None);
}

#[test]
fn formats_expansion_result_into_readable_text() {
    let result = ExpansionResult {
        expansions: vec![
            sieve_lcm::expansion::ExpansionEntry {
                summary_id: "sum_a".to_string(),
                children: vec![
                    sieve_lcm::expansion::ExpansionChild {
                        summary_id: "sum_child_1".to_string(),
                        kind: "leaf".to_string(),
                        snippet: "child snippet".to_string(),
                        token_count: 50,
                    },
                    sieve_lcm::expansion::ExpansionChild {
                        summary_id: "sum_child_2".to_string(),
                        kind: "condensed".to_string(),
                        snippet: "another snippet".to_string(),
                        token_count: 80,
                    },
                ],
                messages: vec![],
            },
            sieve_lcm::expansion::ExpansionEntry {
                summary_id: "sum_b".to_string(),
                children: vec![],
                messages: vec![
                    sieve_lcm::expansion::ExpansionMessage {
                        message_id: 5,
                        role: "user".to_string(),
                        snippet: "user said hello".to_string(),
                        token_count: 10,
                    },
                    sieve_lcm::expansion::ExpansionMessage {
                        message_id: 6,
                        role: "assistant".to_string(),
                        snippet: "bot replied".to_string(),
                        token_count: 15,
                    },
                ],
            },
        ],
        cited_ids: vec![
            "sum_a".to_string(),
            "sum_child_1".to_string(),
            "sum_child_2".to_string(),
            "sum_b".to_string(),
        ],
        total_tokens: 155,
        truncated: false,
    };
    let output = distill_for_subagent(&result);

    assert!(output.contains("2 summaries"));
    assert!(output.contains("155 total tokens"));
    assert!(output.contains("### sum_a (condensed"));
    assert!(output.contains("Children: sum_child_1, sum_child_2"));
    assert!(output.contains("[Snippet: child snippet]"));
    assert!(output.contains("### sum_b (leaf"));
    assert!(output.contains("msg#5 (user, 10 tokens)"));
    assert!(output.contains("msg#6 (assistant, 15 tokens)"));
    assert!(output.contains("Cited IDs for follow-up:"));
    assert!(output.contains("sum_a"));
    assert!(output.contains("sum_child_1"));
    assert!(output.contains("[Truncated: no]"));
}

#[test]
fn indicates_truncation_when_truncated() {
    let mut result = default_expansion_result();
    result.truncated = true;
    let output = distill_for_subagent(&result);
    assert!(output.contains("[Truncated: yes]"));
}

#[test]
fn handles_empty_expansion_result() {
    let output = distill_for_subagent(&default_expansion_result());
    assert!(output.contains("0 summaries"));
    assert!(output.contains("0 total tokens"));
    assert!(output.contains("[Truncated: no]"));
    assert!(!output.contains("Cited IDs"));
}

#[test]
fn computes_per_entry_token_sum_from_children_and_messages() {
    let result = ExpansionResult {
        expansions: vec![sieve_lcm::expansion::ExpansionEntry {
            summary_id: "sum_mixed".to_string(),
            children: vec![
                sieve_lcm::expansion::ExpansionChild {
                    summary_id: "sum_c1".to_string(),
                    kind: "leaf".to_string(),
                    snippet: "a".to_string(),
                    token_count: 100,
                },
                sieve_lcm::expansion::ExpansionChild {
                    summary_id: "sum_c2".to_string(),
                    kind: "leaf".to_string(),
                    snippet: "b".to_string(),
                    token_count: 200,
                },
            ],
            messages: vec![sieve_lcm::expansion::ExpansionMessage {
                message_id: 1,
                role: "user".to_string(),
                snippet: "c".to_string(),
                token_count: 50,
            }],
        }],
        total_tokens: 350,
        ..default_expansion_result()
    };
    let output = distill_for_subagent(&result);
    assert!(output.contains("### sum_mixed (condensed, 350 tokens)"));
}
