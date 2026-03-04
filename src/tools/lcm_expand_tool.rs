use std::sync::Arc;

use serde_json::{json, Value};

use crate::engine::LcmContextEngineApi;
use crate::expansion::{distill_for_subagent, ExpansionOrchestrator, ExpansionRequest, ExpansionResult};
use crate::expansion_auth::{get_runtime_expansion_auth_manager, resolve_delegated_expansion_grant_id, wrap_with_auth};
use crate::expansion_policy::{
    decide_lcm_expansion_routing, LcmExpansionRoutingAction, LcmExpansionRoutingDecision,
    LcmExpansionRoutingInput, LcmExpansionRoutingIntent, LcmExpansionTokenRiskLevel,
};
use crate::tools::common::{ToolContentBlock, ToolResult, json_result};
use crate::tools::lcm_conversation_scope::resolve_lcm_conversation_scope;
use crate::tools::lcm_expand_tool_delegation::{normalize_summary_ids, run_delegated_expansion_loop, DelegatedPassStatus};
use crate::types::LcmDependencies;

fn make_empty_expansion_result() -> ExpansionResult {
    ExpansionResult {
        expansions: vec![],
        cited_ids: vec![],
        total_tokens: 0,
        truncated: false,
    }
}

fn routing_action_label(action: &LcmExpansionRoutingAction) -> &'static str {
    match action {
        LcmExpansionRoutingAction::AnswerDirectly => "answer_directly",
        LcmExpansionRoutingAction::ExpandShallow => "expand_shallow",
        LcmExpansionRoutingAction::DelegateTraversal => "delegate_traversal",
    }
}

fn token_risk_label(level: &LcmExpansionTokenRiskLevel) -> &'static str {
    match level {
        LcmExpansionTokenRiskLevel::Low => "low",
        LcmExpansionTokenRiskLevel::Moderate => "moderate",
        LcmExpansionTokenRiskLevel::High => "high",
    }
}

fn policy_to_json(policy: &LcmExpansionRoutingDecision) -> Value {
    json!({
        "action": routing_action_label(&policy.action),
        "normalizedMaxDepth": policy.normalized_max_depth,
        "candidateSummaryCount": policy.candidate_summary_count,
        "estimatedTokens": policy.estimated_tokens,
        "tokenCap": policy.token_cap,
        "tokenRiskRatio": policy.token_risk_ratio,
        "tokenRiskLevel": token_risk_label(&policy.token_risk_level),
        "indicators": {
            "broadTimeRange": policy.indicators.broad_time_range,
            "multiHopRetrieval": policy.indicators.multi_hop_retrieval,
        },
        "triggers": {
            "directByNoCandidates": policy.triggers.direct_by_no_candidates,
            "directByLowComplexityProbe": policy.triggers.direct_by_low_complexity_probe,
            "delegateByDepth": policy.triggers.delegate_by_depth,
            "delegateByCandidateCount": policy.triggers.delegate_by_candidate_count,
            "delegateByTokenRisk": policy.triggers.delegate_by_token_risk,
            "delegateByBroadTimeRangeAndMultiHop": policy.triggers.delegate_by_broad_time_range_and_multi_hop,
        },
        "reasons": policy.reasons,
    })
}

fn build_orchestration_observability(
    policy: &LcmExpansionRoutingDecision,
    execution_path: &str,
    delegated: Option<&crate::tools::lcm_expand_tool_delegation::DelegatedExpansionLoopResult>,
) -> Value {
    let delegated_run_refs = delegated.map(|delegated| {
        delegated
            .passes
            .iter()
            .map(|pass| {
                json!({
                    "pass": pass.pass,
                    "status": match pass.status {
                        DelegatedPassStatus::Ok => "ok",
                        DelegatedPassStatus::Timeout => "timeout",
                        DelegatedPassStatus::Error => "error",
                    },
                    "runId": pass.run_id,
                    "childSessionKey": pass.child_session_key,
                })
            })
            .collect::<Vec<Value>>()
    });
    json!({
        "decisionPath": {
            "policyAction": routing_action_label(&policy.action),
            "executionPath": execution_path,
        },
        "policyReasons": policy.reasons,
        "delegatedRunRefs": delegated_run_refs,
    })
}

#[derive(Clone)]
pub struct LcmExpandTool {
    deps: Arc<dyn LcmDependencies>,
    lcm: Arc<dyn LcmContextEngineApi>,
    session_id: Option<String>,
    session_key: Option<String>,
}

impl LcmExpandTool {
    pub fn new(
        deps: Arc<dyn LcmDependencies>,
        lcm: Arc<dyn LcmContextEngineApi>,
        session_id: Option<String>,
        session_key: Option<String>,
    ) -> Self {
        Self {
            deps,
            lcm,
            session_id,
            session_key,
        }
    }

    pub async fn execute(&self, _tool_call_id: &str, params: Value) -> anyhow::Result<ToolResult> {
        let retrieval = self.lcm.get_retrieval();
        let orchestrator = ExpansionOrchestrator::new(retrieval.clone());
        let runtime_auth_manager = get_runtime_expansion_auth_manager();
        let p = params.as_object().cloned().unwrap_or_default();
        let summary_ids = p
            .get("summaryIds")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<String>>()
            });
        let query = p.get("query").and_then(Value::as_str).map(str::trim).filter(|q| !q.is_empty());
        let max_depth = p
            .get("maxDepth")
            .and_then(Value::as_f64)
            .map(|v| v.trunc() as i64);
        let requested_token_cap = p
            .get("tokenCap")
            .and_then(Value::as_f64)
            .map(|v| v.trunc() as i64);
        let token_cap = requested_token_cap.map(|v| v.max(1));
        let include_messages = p
            .get("includeMessages")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let session_key = self
            .session_key
            .as_deref()
            .or(self.session_id.as_deref())
            .unwrap_or_default()
            .trim()
            .to_string();
        if !self.deps.is_subagent_session_key(&session_key) {
            return Ok(json_result(json!({
                "error": "lcm_expand is only available in sub-agent sessions. Use lcm_expand_query to ask a focused question against expanded summaries, or lcm_describe/lcm_grep for lighter lookups."
            })));
        }

        let is_delegated_session = self.deps.is_subagent_session_key(&session_key);
        let delegated_grant_id = if is_delegated_session {
            resolve_delegated_expansion_grant_id(&session_key)
        } else {
            None
        };
        let delegated_grant = delegated_grant_id
            .as_deref()
            .and_then(|grant_id| runtime_auth_manager.lock().get_grant(grant_id));
        let authorized_orchestrator = delegated_grant_id
            .as_ref()
            .map(|_| wrap_with_auth(Arc::new(orchestrator.clone()), runtime_auth_manager.clone()));
        if is_delegated_session && delegated_grant_id.is_none() {
            return Ok(json_result(json!({
                "error": "Delegated expansion requires a valid grant. This sub-agent session has no propagated expansion grant."
            })));
        }

        let conversation_scope = resolve_lcm_conversation_scope(
            self.lcm.as_ref(),
            &p,
            self.session_id.as_deref(),
            self.session_key.as_deref(),
            Some(self.deps.as_ref()),
        )
        .await?;

        let run_expand = |summary_ids: Vec<String>,
                          conversation_id: i64,
                          max_depth: Option<i64>,
                          token_cap: Option<i64>,
                          include_messages: bool,
                          delegated_grant_id: Option<String>,
                          authorized_orchestrator: Option<crate::expansion_auth::AuthorizedExpansionOrchestrator>,
                          orchestrator: ExpansionOrchestrator| async move {
            let request = ExpansionRequest {
                summary_ids,
                max_depth,
                token_cap,
                include_messages: Some(include_messages),
                conversation_id,
            };
            if let (Some(authorized), Some(grant_id)) = (authorized_orchestrator, delegated_grant_id) {
                authorized.expand(&grant_id, request).await
            } else {
                orchestrator.expand(request).await
            }
        };

        let resolved_conversation_id = conversation_scope.conversation_id.or_else(|| {
            delegated_grant.as_ref().and_then(|grant| {
                if grant.allowed_conversation_ids.len() == 1 {
                    Some(grant.allowed_conversation_ids[0])
                } else {
                    None
                }
            })
        });

        if let Some(query) = query {
            let policy_from = |candidate_count: usize| {
                decide_lcm_expansion_routing(LcmExpansionRoutingInput {
                    intent: LcmExpansionRoutingIntent::QueryProbe,
                    query: Some(query.to_string()),
                    requested_max_depth: max_depth,
                    candidate_summary_count: candidate_count as i64,
                    token_cap: token_cap.unwrap_or(i64::MAX),
                    include_messages: false,
                })
            };

            if resolved_conversation_id.is_none() {
                let result = match orchestrator
                    .describe_and_expand(
                        query,
                        "full_text",
                        None,
                        max_depth,
                        token_cap,
                    )
                    .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        return Ok(json_result(json!({
                            "error": error.to_string()
                        })));
                    }
                };
                let policy = policy_from(result.expansions.len());
                return Ok(ToolResult {
                    content: vec![ToolContentBlock {
                        r#type: "text".to_string(),
                        text: distill_for_subagent(&result),
                    }],
                    details: json!({
                        "expansionCount": result.expansions.len(),
                        "citedIds": result.cited_ids,
                        "totalTokens": result.total_tokens,
                        "truncated": result.truncated,
                        "policy": policy_to_json(&policy),
                        "executionPath": "direct",
                        "observability": build_orchestration_observability(&policy, "direct", None),
                    }),
                });
            }

            let resolved_conversation_id = resolved_conversation_id.unwrap_or_default();
            let grep_result = retrieval
                .grep(crate::retrieval::GrepInput {
                    query: query.to_string(),
                    mode: "full_text".to_string(),
                    scope: "summaries".to_string(),
                    conversation_id: Some(resolved_conversation_id),
                    since: None,
                    before: None,
                    limit: None,
                })
                .await?;
            let matched_summary_ids = grep_result
                .summaries
                .iter()
                .map(|entry| entry.summary_id.clone())
                .collect::<Vec<String>>();
            let policy = policy_from(matched_summary_ids.len());

            let can_delegate = !matched_summary_ids.is_empty()
                && matches!(policy.action, LcmExpansionRoutingAction::DelegateTraversal)
                && !is_delegated_session
                && !is_delegated_session
                && !session_key.is_empty();
            let delegated = if can_delegate {
                Some(
                    run_delegated_expansion_loop(
                        self.deps.as_ref(),
                        &session_key,
                        resolved_conversation_id,
                        matched_summary_ids.clone(),
                        max_depth,
                        token_cap,
                        false,
                        Some(query),
                        None,
                    )
                    .await,
                )
            } else {
                None
            };
            if let Some(delegated) = delegated.clone() {
                if delegated.status == DelegatedPassStatus::Ok {
                    return Ok(ToolResult {
                        content: vec![ToolContentBlock {
                            r#type: "text".to_string(),
                            text: delegated.text.clone(),
                        }],
                        details: json!({
                            "expansionCount": delegated.cited_ids.len(),
                            "citedIds": delegated.cited_ids,
                            "totalTokens": delegated.total_tokens,
                            "truncated": delegated.truncated,
                            "policy": policy_to_json(&policy),
                            "executionPath": "delegated",
                            "delegated": delegated_to_json(&delegated),
                            "observability": build_orchestration_observability(&policy, "delegated", Some(&delegated)),
                        }),
                    });
                }
            }

            let execution_path = if delegated.is_some() {
                "direct_fallback"
            } else {
                "direct"
            };
            let result = if matched_summary_ids.is_empty() {
                make_empty_expansion_result()
            } else {
                match run_expand(
                    matched_summary_ids.clone(),
                    resolved_conversation_id,
                    max_depth,
                    token_cap,
                    false,
                    delegated_grant_id.clone(),
                    authorized_orchestrator.clone(),
                    orchestrator.clone(),
                )
                .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        return Ok(json_result(json!({
                            "error": error.to_string()
                        })));
                    }
                }
            };
            return Ok(ToolResult {
                content: vec![ToolContentBlock {
                    r#type: "text".to_string(),
                    text: distill_for_subagent(&result),
                }],
                details: json!({
                    "expansionCount": result.expansions.len(),
                    "citedIds": result.cited_ids,
                    "totalTokens": result.total_tokens,
                    "truncated": result.truncated,
                    "policy": policy_to_json(&policy),
                    "executionPath": execution_path,
                    "delegated": delegated.as_ref().filter(|d| d.status != DelegatedPassStatus::Ok).map(|d| {
                        json!({
                            "status": delegated_status_label(&d.status),
                            "error": d.error.clone(),
                            "passes": d.passes.iter().map(delegated_pass_to_json).collect::<Vec<Value>>(),
                        })
                    }),
                    "observability": build_orchestration_observability(&policy, execution_path, delegated.as_ref()),
                }),
            });
        }

        if let Some(summary_ids) = summary_ids.filter(|v| !v.is_empty()) {
            if let Some(conversation_id) = conversation_scope.conversation_id {
                let mut out_of_scope = vec![];
                for summary_id in &summary_ids {
                    let described = retrieval.describe(summary_id).await?;
                    if let Some(described) = described {
                        if let crate::retrieval::DescribeResultType::Summary(summary) = described.result {
                            if summary.conversation_id != conversation_id {
                                out_of_scope.push(summary_id.clone());
                            }
                        }
                    }
                }
                if !out_of_scope.is_empty() {
                    return Ok(json_result(json!({
                        "error": format!("Some summaryIds are outside conversation {}: {}", conversation_id, out_of_scope.join(", ")),
                        "hint": "Use allConversations=true for cross-conversation expansion.",
                    })));
                }
            }

            let policy = decide_lcm_expansion_routing(LcmExpansionRoutingInput {
                intent: LcmExpansionRoutingIntent::ExplicitExpand,
                query: None,
                requested_max_depth: max_depth,
                candidate_summary_count: summary_ids.len() as i64,
                token_cap: token_cap.unwrap_or(i64::MAX),
                include_messages,
            });
            let normalized_summary_ids = normalize_summary_ids(Some(&summary_ids));
            let can_delegate = !normalized_summary_ids.is_empty()
                && matches!(policy.action, LcmExpansionRoutingAction::DelegateTraversal)
                && !is_delegated_session
                && !session_key.is_empty()
                && resolved_conversation_id.is_some();
            let delegated = if can_delegate {
                Some(
                    run_delegated_expansion_loop(
                        self.deps.as_ref(),
                        &session_key,
                        resolved_conversation_id.unwrap_or_default(),
                        normalized_summary_ids.clone(),
                        max_depth,
                        token_cap,
                        include_messages,
                        None,
                        None,
                    )
                    .await,
                )
            } else {
                None
            };
            if let Some(delegated) = delegated.clone() {
                if delegated.status == DelegatedPassStatus::Ok {
                    return Ok(ToolResult {
                        content: vec![ToolContentBlock {
                            r#type: "text".to_string(),
                            text: delegated.text.clone(),
                        }],
                        details: json!({
                            "expansionCount": delegated.cited_ids.len(),
                            "citedIds": delegated.cited_ids,
                            "totalTokens": delegated.total_tokens,
                            "truncated": delegated.truncated,
                            "policy": policy_to_json(&policy),
                            "executionPath": "delegated",
                            "delegated": delegated_to_json(&delegated),
                            "observability": build_orchestration_observability(&policy, "delegated", Some(&delegated)),
                        }),
                    });
                }
            }
            let execution_path = if delegated.is_some() {
                "direct_fallback"
            } else {
                "direct"
            };
            let result = match run_expand(
                normalized_summary_ids,
                resolved_conversation_id.unwrap_or(0),
                max_depth,
                token_cap,
                include_messages,
                delegated_grant_id.clone(),
                authorized_orchestrator.clone(),
                orchestrator.clone(),
            )
            .await
            {
                Ok(result) => result,
                Err(error) => {
                    return Ok(json_result(json!({
                        "error": error.to_string()
                    })));
                }
            };
            return Ok(ToolResult {
                content: vec![ToolContentBlock {
                    r#type: "text".to_string(),
                    text: distill_for_subagent(&result),
                }],
                details: json!({
                    "expansionCount": result.expansions.len(),
                    "citedIds": result.cited_ids,
                    "totalTokens": result.total_tokens,
                    "truncated": result.truncated,
                    "policy": policy_to_json(&policy),
                    "executionPath": execution_path,
                    "delegated": delegated.as_ref().filter(|d| d.status != DelegatedPassStatus::Ok).map(|d| {
                        json!({
                            "status": delegated_status_label(&d.status),
                            "error": d.error.clone(),
                            "passes": d.passes.iter().map(delegated_pass_to_json).collect::<Vec<Value>>(),
                        })
                    }),
                    "observability": build_orchestration_observability(&policy, execution_path, delegated.as_ref()),
                }),
            });
        }

        Ok(json_result(json!({
            "error": "Either summaryIds or query must be provided."
        })))
    }
}

fn delegated_status_label(status: &DelegatedPassStatus) -> &'static str {
    match status {
        DelegatedPassStatus::Ok => "ok",
        DelegatedPassStatus::Timeout => "timeout",
        DelegatedPassStatus::Error => "error",
    }
}

fn delegated_pass_to_json(pass: &crate::tools::lcm_expand_tool_delegation::DelegatedExpansionPassResult) -> Value {
    json!({
        "pass": pass.pass,
        "status": delegated_status_label(&pass.status),
        "runId": pass.run_id,
        "childSessionKey": pass.child_session_key,
        "summary": pass.summary,
        "citedIds": pass.cited_ids,
        "followUpSummaryIds": pass.follow_up_summary_ids,
        "totalTokens": pass.total_tokens,
        "truncated": pass.truncated,
        "rawReply": pass.raw_reply,
        "error": pass.error,
    })
}

fn delegated_to_json(delegated: &crate::tools::lcm_expand_tool_delegation::DelegatedExpansionLoopResult) -> Value {
    json!({
        "status": delegated_status_label(&delegated.status),
        "passes": delegated.passes.iter().map(delegated_pass_to_json).collect::<Vec<Value>>(),
        "citedIds": delegated.cited_ids,
        "totalTokens": delegated.total_tokens,
        "truncated": delegated.truncated,
        "text": delegated.text,
        "error": delegated.error,
    })
}

pub fn create_lcm_expand_tool(
    deps: Arc<dyn LcmDependencies>,
    lcm: Arc<dyn LcmContextEngineApi>,
    session_id: Option<String>,
    session_key: Option<String>,
) -> LcmExpandTool {
    LcmExpandTool::new(deps, lcm, session_id, session_key)
}
