use std::collections::HashSet;

use serde_json::{Value, json};
use uuid::Uuid;

use crate::engine::LcmContextEngineApi;
use crate::expansion_auth::{
    CreateDelegatedExpansionGrantInput, create_delegated_expansion_grant,
    revoke_delegated_expansion_grant_for_session,
};
use crate::tools::lcm_expansion_recursion_guard::{
    TelemetryEvent, clear_delegated_expansion_context, evaluate_expansion_recursion_guard,
    record_expansion_delegation_telemetry, resolve_expansion_request_id,
    stamp_delegated_expansion_context,
};
use crate::types::{GatewayCallRequest, LcmDependencies};

const MAX_GATEWAY_TIMEOUT_MS: i64 = 2_147_483_647;

#[derive(Clone, Debug, PartialEq)]
pub enum DelegatedPassStatus {
    Ok,
    Timeout,
    Error,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelegatedExpansionPassResult {
    pub pass: i64,
    pub status: DelegatedPassStatus,
    pub run_id: String,
    pub child_session_key: String,
    pub summary: String,
    pub cited_ids: Vec<String>,
    pub follow_up_summary_ids: Vec<String>,
    pub total_tokens: i64,
    pub truncated: bool,
    pub raw_reply: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelegatedExpansionLoopResult {
    pub status: DelegatedPassStatus,
    pub passes: Vec<DelegatedExpansionPassResult>,
    pub cited_ids: Vec<String>,
    pub total_tokens: i64,
    pub truncated: bool,
    pub text: String,
    pub error: Option<String>,
}

pub fn normalize_summary_ids(input: Option<&[String]>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut normalized = vec![];
    for value in input.unwrap_or(&[]) {
        let trimmed = value.trim();
        if trimmed.is_empty() || seen.contains(trimmed) {
            continue;
        }
        seen.insert(trimmed.to_string());
        normalized.push(trimmed.to_string());
    }
    normalized
}

fn parse_delegated_expansion_reply(
    raw_reply: Option<&str>,
) -> (String, Vec<String>, Vec<String>, i64, bool) {
    let fallback_summary = raw_reply.unwrap_or_default().trim().to_string();
    let reply = raw_reply.unwrap_or_default().trim();
    if reply.is_empty() {
        return (fallback_summary, vec![], vec![], 0, false);
    }

    let mut candidates = vec![reply.to_string()];
    if let Some(start) = reply.find("```") {
        if let Some(end) = reply[start + 3..].find("```") {
            let fenced = &reply[start + 3..start + 3 + end];
            let json_candidate = fenced.trim_start_matches("json").trim().to_string();
            if !json_candidate.is_empty() {
                candidates.insert(0, json_candidate);
            }
        }
    }

    for candidate in candidates {
        let Ok(parsed) = serde_json::from_str::<Value>(&candidate) else {
            continue;
        };
        let summary = parsed
            .get("summary")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(ToString::to_string)
            .unwrap_or_else(|| fallback_summary.clone());
        let cited_candidates = parsed
            .get("citedIds")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        let cited_ids = normalize_summary_ids(Some(&cited_candidates));
        let follow_up_candidates = parsed
            .get("followUpSummaryIds")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        let follow_up_summary_ids = normalize_summary_ids(Some(&follow_up_candidates));
        let total_tokens = parsed
            .get("totalTokens")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
            .map(|v| (v.floor() as i64).max(0))
            .unwrap_or(0);
        let truncated = parsed.get("truncated").and_then(Value::as_bool) == Some(true);
        return (
            summary,
            cited_ids,
            follow_up_summary_ids,
            total_tokens,
            truncated,
        );
    }

    (fallback_summary, vec![], vec![], 0, false)
}

fn format_delegated_expansion_text(passes: &[DelegatedExpansionPassResult]) -> String {
    let mut lines = vec![];
    let mut all_cited = HashSet::new();

    for pass in passes {
        for summary_id in &pass.cited_ids {
            all_cited.insert(summary_id.clone());
        }
        if pass.summary.trim().is_empty() {
            continue;
        }
        if passes.len() > 1 {
            lines.push(format!("Pass {}: {}", pass.pass, pass.summary.trim()));
        } else {
            lines.push(pass.summary.trim().to_string());
        }
    }
    if lines.is_empty() {
        lines.push("Delegated expansion completed with no textual summary.".to_string());
    }
    if !all_cited.is_empty() {
        lines.push(String::new());
        lines.push("Cited IDs:".to_string());
        let mut ids = all_cited.into_iter().collect::<Vec<String>>();
        ids.sort();
        lines.extend(ids.into_iter().map(|id| format!("- {}", id)));
    }
    lines.join("\n")
}

fn build_delegated_expansion_task(
    summary_ids: &[String],
    conversation_id: i64,
    max_depth: Option<i64>,
    token_cap: Option<i64>,
    include_messages: bool,
    pass: i64,
    query: Option<&str>,
    request_id: &str,
    expansion_depth: i64,
    origin_session_key: &str,
) -> String {
    let mut payload = json!({
        "summaryIds": summary_ids,
        "conversationId": conversation_id,
        "maxDepth": max_depth,
        "includeMessages": include_messages,
    });
    if let Some(token_cap) = token_cap {
        payload["tokenCap"] = json!(token_cap);
    }
    let mut lines = vec!["Run LCM expansion and report distilled findings.".to_string()];
    if let Some(query) = query {
        lines.push(format!("Original query: {}", query));
    }
    lines.extend(vec![
        format!("Pass {}", pass),
        String::new(),
        "Call `lcm_expand` using exactly this JSON payload:".to_string(),
        serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string()),
        String::new(),
        "Delegated expansion metadata (for tracing):".to_string(),
        format!("- requestId: {}", request_id),
        format!("- expansionDepth: {}", expansion_depth),
        format!("- originSessionKey: {}", origin_session_key),
        String::new(),
        "Then return ONLY JSON with this shape:".to_string(),
        "{".to_string(),
        r#"  "summary": "string concise findings","#.to_string(),
        r#"  "citedIds": ["sum_xxx"],"#.to_string(),
        r#"  "followUpSummaryIds": ["sum_xxx"],"#.to_string(),
        r#"  "totalTokens": 0,"#.to_string(),
        r#"  "truncated": false"#.to_string(),
        "}".to_string(),
        String::new(),
        "Rules:".to_string(),
        "- In delegated context, use `lcm_expand` directly for retrieval.".to_string(),
        "- DO NOT call `lcm_expand_query` from this delegated session.".to_string(),
        "- Keep summary concise and factual.".to_string(),
        "- Synthesize findings from the `lcm_expand` result before returning.".to_string(),
        "- citedIds/followUpSummaryIds must contain unique summary IDs only.".to_string(),
        "- If no follow-up is needed, return an empty followUpSummaryIds array.".to_string(),
    ]);
    lines.join("\n")
}

async fn run_delegated_expansion_pass(
    deps: &dyn LcmDependencies,
    requester_session_key: &str,
    conversation_id: i64,
    summary_ids: &[String],
    max_depth: Option<i64>,
    token_cap: Option<i64>,
    include_messages: bool,
    query: Option<&str>,
    pass: i64,
    request_id: &str,
    parent_expansion_depth: i64,
    origin_session_key: &str,
) -> DelegatedExpansionPassResult {
    let requester_agent_id = deps.normalize_agent_id(
        deps.parse_agent_session_key(requester_session_key)
            .map(|(agent_id, _)| agent_id)
            .as_deref(),
    );
    let child_session_key = format!("agent:{}:subagent:{}", requester_agent_id, Uuid::new_v4());
    let mut run_id = String::new();

    create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
        delegated_session_key: child_session_key.clone(),
        issuer_session_id: requester_session_key.to_string(),
        allowed_conversation_ids: vec![conversation_id],
        allowed_summary_ids: None,
        max_depth: None,
        token_cap,
        ttl_ms: Some(MAX_GATEWAY_TIMEOUT_MS),
    });
    stamp_delegated_expansion_context(
        &child_session_key,
        request_id,
        parent_expansion_depth + 1,
        origin_session_key,
        "runDelegatedExpansionLoop",
    );

    let result: anyhow::Result<DelegatedExpansionPassResult> = async {
        let message = build_delegated_expansion_task(
            summary_ids,
            conversation_id,
            max_depth,
            token_cap,
            include_messages,
            pass,
            query,
            request_id,
            parent_expansion_depth + 1,
            origin_session_key,
        );

        let response = deps
            .call_gateway(GatewayCallRequest {
                method: "agent".to_string(),
                params: Some(json!({
                    "message": message,
                    "sessionKey": child_session_key,
                    "deliver": false,
                    "lane": deps.agent_lane_subagent(),
                    "extraSystemPrompt": deps.build_subagent_system_prompt(
                        1,
                        8,
                        Some("Run lcm_expand and return JSON findings")
                    )
                })),
                timeout_ms: Some(10_000),
            })
            .await?;
        run_id = response
            .get("runId")
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
            .unwrap_or_default()
            .to_string();
        if run_id.is_empty() {
            run_id = Uuid::new_v4().to_string();
        }

        let wait = deps
            .call_gateway(GatewayCallRequest {
                method: "agent.wait".to_string(),
                params: Some(json!({
                    "runId": run_id,
                    "timeoutMs": MAX_GATEWAY_TIMEOUT_MS
                })),
                timeout_ms: Some(MAX_GATEWAY_TIMEOUT_MS),
            })
            .await?;
        let status = wait
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("error")
            .to_string();
        if status == "timeout" {
            return Ok(DelegatedExpansionPassResult {
                pass,
                status: DelegatedPassStatus::Timeout,
                run_id: run_id.clone(),
                child_session_key: child_session_key.clone(),
                summary: String::new(),
                cited_ids: vec![],
                follow_up_summary_ids: vec![],
                total_tokens: 0,
                truncated: true,
                raw_reply: None,
                error: Some("delegated expansion pass timed out".to_string()),
            });
        }
        if status != "ok" {
            return Ok(DelegatedExpansionPassResult {
                pass,
                status: DelegatedPassStatus::Error,
                run_id: run_id.clone(),
                child_session_key: child_session_key.clone(),
                summary: String::new(),
                cited_ids: vec![],
                follow_up_summary_ids: vec![],
                total_tokens: 0,
                truncated: true,
                raw_reply: None,
                error: Some(
                    wait.get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("delegated expansion pass failed")
                        .to_string(),
                ),
            });
        }

        let reply_payload = deps
            .call_gateway(GatewayCallRequest {
                method: "sessions.get".to_string(),
                params: Some(json!({
                    "key": child_session_key,
                    "limit": 80
                })),
                timeout_ms: Some(10_000),
            })
            .await?;
        let messages = reply_payload
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let reply = deps.read_latest_assistant_reply(&messages);
        let (summary, cited_ids, follow_up_summary_ids, total_tokens, truncated) =
            parse_delegated_expansion_reply(reply.as_deref());
        Ok(DelegatedExpansionPassResult {
            pass,
            status: DelegatedPassStatus::Ok,
            run_id: run_id.clone(),
            child_session_key: child_session_key.clone(),
            summary,
            cited_ids,
            follow_up_summary_ids,
            total_tokens,
            truncated,
            raw_reply: reply,
            error: None,
        })
    }
    .await;

    let output = match result {
        Ok(value) => value,
        Err(error) => DelegatedExpansionPassResult {
            pass,
            status: DelegatedPassStatus::Error,
            run_id: if run_id.is_empty() {
                Uuid::new_v4().to_string()
            } else {
                run_id.clone()
            },
            child_session_key: child_session_key.clone(),
            summary: String::new(),
            cited_ids: vec![],
            follow_up_summary_ids: vec![],
            total_tokens: 0,
            truncated: true,
            raw_reply: None,
            error: Some(error.to_string()),
        },
    };

    let _ = deps
        .call_gateway(GatewayCallRequest {
            method: "sessions.delete".to_string(),
            params: Some(json!({
                "key": child_session_key,
                "deleteTranscript": true
            })),
            timeout_ms: Some(10_000),
        })
        .await;
    revoke_delegated_expansion_grant_for_session(&child_session_key, true);
    clear_delegated_expansion_context(&child_session_key);

    output
}

pub async fn resolve_requester_conversation_scope_id(
    deps: &dyn LcmDependencies,
    requester_session_key: &str,
    lcm: &dyn LcmContextEngineApi,
) -> Option<i64> {
    let requester_session_key = requester_session_key.trim();
    if requester_session_key.is_empty() {
        return None;
    }
    let runtime_session_id = deps
        .resolve_session_id_from_session_key(requester_session_key)
        .await
        .ok()
        .flatten()?;
    lcm.get_conversation_store()
        .get_conversation_by_session_id(&runtime_session_id)
        .await
        .ok()
        .flatten()
        .map(|c| c.conversation_id)
}

pub async fn run_delegated_expansion_loop(
    deps: &dyn LcmDependencies,
    requester_session_key: &str,
    conversation_id: i64,
    summary_ids: Vec<String>,
    max_depth: Option<i64>,
    token_cap: Option<i64>,
    include_messages: bool,
    query: Option<&str>,
    request_id: Option<&str>,
) -> DelegatedExpansionLoopResult {
    let request_id = request_id
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| resolve_expansion_request_id(Some(requester_session_key)));
    let recursion_check =
        evaluate_expansion_recursion_guard(Some(requester_session_key), &request_id);
    record_expansion_delegation_telemetry(
        deps,
        "runDelegatedExpansionLoop",
        TelemetryEvent::Start,
        &request_id,
        Some(requester_session_key),
        recursion_check.expansion_depth,
        &recursion_check.origin_session_key,
        None,
        None,
    );
    if recursion_check.blocked {
        record_expansion_delegation_telemetry(
            deps,
            "runDelegatedExpansionLoop",
            TelemetryEvent::Block,
            &request_id,
            Some(requester_session_key),
            recursion_check.expansion_depth,
            &recursion_check.origin_session_key,
            recursion_check.reason.as_ref().map(|r| r.as_str()),
            None,
        );
        return DelegatedExpansionLoopResult {
            status: DelegatedPassStatus::Error,
            passes: vec![],
            cited_ids: vec![],
            total_tokens: 0,
            truncated: true,
            text: "Delegated expansion blocked by recursion guard.".to_string(),
            error: recursion_check.message,
        };
    }

    let mut passes = vec![];
    let mut visited: HashSet<String> = HashSet::new();
    let mut cited: HashSet<String> = HashSet::new();
    let mut queue = normalize_summary_ids(Some(&summary_ids));
    let mut pass = 1_i64;

    while !queue.is_empty() {
        for summary_id in &queue {
            visited.insert(summary_id.clone());
        }
        let result = run_delegated_expansion_pass(
            deps,
            requester_session_key,
            conversation_id,
            &queue,
            max_depth,
            token_cap,
            include_messages,
            query,
            pass,
            &request_id,
            recursion_check.expansion_depth,
            &recursion_check.origin_session_key,
        )
        .await;
        passes.push(result.clone());

        if result.status != DelegatedPassStatus::Ok {
            if result.status == DelegatedPassStatus::Timeout {
                record_expansion_delegation_telemetry(
                    deps,
                    "runDelegatedExpansionLoop",
                    TelemetryEvent::Timeout,
                    &request_id,
                    Some(requester_session_key),
                    recursion_check.expansion_depth,
                    &recursion_check.origin_session_key,
                    None,
                    Some(&result.run_id),
                );
            }
            let ok_passes = passes
                .iter()
                .filter(|entry| entry.status == DelegatedPassStatus::Ok)
                .cloned()
                .collect::<Vec<_>>();
            for ok_pass in &ok_passes {
                for summary_id in &ok_pass.cited_ids {
                    cited.insert(summary_id.clone());
                }
            }
            let text = if ok_passes.is_empty() {
                "Delegated expansion failed before any pass completed.".to_string()
            } else {
                format_delegated_expansion_text(&ok_passes)
            };
            return DelegatedExpansionLoopResult {
                status: result.status,
                passes,
                cited_ids: cited.into_iter().collect(),
                total_tokens: ok_passes.iter().map(|entry| entry.total_tokens).sum(),
                truncated: true,
                text,
                error: result.error,
            };
        }

        for summary_id in &result.cited_ids {
            cited.insert(summary_id.clone());
        }
        let next_queue = result
            .follow_up_summary_ids
            .iter()
            .filter(|summary_id| !visited.contains(*summary_id))
            .cloned()
            .collect::<Vec<String>>();
        queue = next_queue;
        pass += 1;
    }

    record_expansion_delegation_telemetry(
        deps,
        "runDelegatedExpansionLoop",
        TelemetryEvent::Success,
        &request_id,
        Some(requester_session_key),
        recursion_check.expansion_depth,
        &recursion_check.origin_session_key,
        None,
        None,
    );
    let mut cited_ids = cited.into_iter().collect::<Vec<String>>();
    cited_ids.sort();
    DelegatedExpansionLoopResult {
        status: DelegatedPassStatus::Ok,
        passes: passes.clone(),
        cited_ids,
        total_tokens: passes.iter().map(|entry| entry.total_tokens).sum(),
        truncated: passes.iter().any(|entry| entry.truncated),
        text: format_delegated_expansion_text(&passes),
        error: None,
    }
}
