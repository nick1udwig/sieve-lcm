use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde_json::{Value, json};
use uuid::Uuid;

use crate::engine::LcmContextEngineApi;
use crate::expansion_auth::{
    CreateDelegatedExpansionGrantInput, create_delegated_expansion_grant,
    revoke_delegated_expansion_grant_for_session,
};
use crate::tools::common::{ToolResult, json_result};
use crate::tools::lcm_conversation_scope::resolve_lcm_conversation_scope;
use crate::tools::lcm_expand_tool_delegation::{
    normalize_summary_ids, resolve_requester_conversation_scope_id,
};
use crate::tools::lcm_expansion_recursion_guard::{
    TelemetryEvent, clear_delegated_expansion_context, evaluate_expansion_recursion_guard,
    record_expansion_delegation_telemetry, resolve_expansion_request_id,
    resolve_next_expansion_depth, stamp_delegated_expansion_context,
};
use crate::types::{GatewayCallRequest, LcmDependencies};

const DELEGATED_WAIT_TIMEOUT_MS: i64 = 120_000;
const GATEWAY_TIMEOUT_MS: i64 = 10_000;
const DEFAULT_MAX_ANSWER_TOKENS: i64 = 2_000;

#[derive(Clone, Debug, PartialEq)]
struct ExpandQueryReply {
    answer: String,
    cited_ids: Vec<String>,
    expanded_summary_count: i64,
    total_source_tokens: i64,
    truncated: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct SummaryCandidate {
    summary_id: String,
    conversation_id: i64,
}

fn build_delegated_expand_query_task(
    summary_ids: &[String],
    conversation_id: i64,
    query: Option<&str>,
    prompt: &str,
    max_tokens: i64,
    token_cap: i64,
    request_id: &str,
    expansion_depth: i64,
    origin_session_key: &str,
) -> String {
    let seed_summary_ids = if summary_ids.is_empty() {
        "(none)".to_string()
    } else {
        summary_ids.join(", ")
    };
    let mut lines = vec![
        "You are an autonomous LCM retrieval navigator. Plan and execute retrieval before answering."
            .to_string(),
        String::new(),
        "Available tools: lcm_describe, lcm_expand, lcm_grep".to_string(),
        format!("Conversation scope: {}", conversation_id),
        format!("Expansion token budget (total across this run): {}", token_cap),
        format!("Seed summary IDs: {}", seed_summary_ids),
    ];
    if let Some(query) = query {
        lines.push(format!("Routing query: {}", query));
    }
    lines.extend(vec![
        String::new(),
        "Strategy:".to_string(),
        "1. Start with `lcm_describe` on seed summaries to inspect subtree manifests and branch costs."
            .to_string(),
        "2. If additional candidates are needed, use `lcm_grep` scoped to summaries.".to_string(),
        "3. Select branches that fit remaining budget; prefer high-signal paths first.".to_string(),
        "4. Call `lcm_expand` selectively (do not expand everything blindly).".to_string(),
        "5. Keep includeMessages=false by default; use includeMessages=true only for specific leaf evidence."
            .to_string(),
        format!(
            "6. Stay within {} total expansion tokens across all lcm_expand calls.",
            token_cap
        ),
        String::new(),
        "User prompt to answer:".to_string(),
        prompt.to_string(),
        String::new(),
        "Delegated expansion metadata (for tracing):".to_string(),
        format!("- requestId: {}", request_id),
        format!("- expansionDepth: {}", expansion_depth),
        format!("- originSessionKey: {}", origin_session_key),
        String::new(),
        "Return ONLY JSON with this shape:".to_string(),
        "{".to_string(),
        r#"  "answer": "string","#.to_string(),
        r#"  "citedIds": ["sum_xxx"],"#.to_string(),
        r#"  "expandedSummaryCount": 0,"#.to_string(),
        r#"  "totalSourceTokens": 0,"#.to_string(),
        r#"  "truncated": false"#.to_string(),
        "}".to_string(),
        String::new(),
        "Rules:".to_string(),
        "- In delegated context, call `lcm_expand` directly for source retrieval.".to_string(),
        "- DO NOT call `lcm_expand_query` from this delegated session.".to_string(),
        "- Synthesize the final answer from retrieved evidence, not assumptions.".to_string(),
        format!("- Keep answer concise and focused (target <= {} tokens).", max_tokens),
        "- citedIds must be unique summary IDs.".to_string(),
        "- expandedSummaryCount should reflect how many summaries were expanded/used.".to_string(),
        "- totalSourceTokens should estimate total tokens consumed from expansion calls.".to_string(),
        "- truncated should indicate whether source expansion appears truncated.".to_string(),
    ]);
    lines.join("\n")
}

fn parse_delegated_expand_query_reply(
    raw_reply: Option<&str>,
    fallback_expanded_summary_count: usize,
) -> ExpandQueryReply {
    let fallback = ExpandQueryReply {
        answer: raw_reply.unwrap_or_default().trim().to_string(),
        cited_ids: vec![],
        expanded_summary_count: fallback_expanded_summary_count as i64,
        total_source_tokens: 0,
        truncated: false,
    };
    let reply = raw_reply.unwrap_or_default().trim();
    if reply.is_empty() {
        return fallback;
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
        let answer = parsed
            .get("answer")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(ToString::to_string)
            .unwrap_or_else(|| fallback.answer.clone());
        let cited = parsed
            .get("citedIds")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        let cited_ids = normalize_summary_ids(Some(&cited));
        let expanded_summary_count = parsed
            .get("expandedSummaryCount")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
            .map(|v| (v.floor() as i64).max(0))
            .unwrap_or(fallback.expanded_summary_count);
        let total_source_tokens = parsed
            .get("totalSourceTokens")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
            .map(|v| (v.floor() as i64).max(0))
            .unwrap_or(0);
        let truncated = parsed.get("truncated").and_then(Value::as_bool) == Some(true);
        return ExpandQueryReply {
            answer,
            cited_ids,
            expanded_summary_count,
            total_source_tokens,
            truncated,
        };
    }

    fallback
}

fn resolve_source_conversation_id(
    scoped_conversation_id: Option<i64>,
    all_conversations: bool,
    candidates: &[SummaryCandidate],
) -> anyhow::Result<i64> {
    if let Some(scoped_conversation_id) = scoped_conversation_id {
        let mismatched = candidates
            .iter()
            .filter(|candidate| candidate.conversation_id != scoped_conversation_id)
            .map(|candidate| candidate.summary_id.clone())
            .collect::<Vec<String>>();
        if !mismatched.is_empty() {
            anyhow::bail!(
                "Some summaryIds are outside conversation {}: {}",
                scoped_conversation_id,
                mismatched.join(", ")
            );
        }
        return Ok(scoped_conversation_id);
    }

    let conversation_ids = candidates
        .iter()
        .map(|candidate| candidate.conversation_id)
        .collect::<HashSet<i64>>()
        .into_iter()
        .collect::<Vec<i64>>();
    if conversation_ids.len() == 1 {
        return Ok(conversation_ids[0]);
    }
    if all_conversations && conversation_ids.len() > 1 {
        anyhow::bail!(
            "Query matched summaries from multiple conversations. Provide conversationId or narrow the query."
        );
    }
    anyhow::bail!(
        "Unable to resolve a single conversation scope. Provide conversationId or set a narrower summary scope."
    )
}

async fn resolve_summary_candidates(
    lcm: &dyn LcmContextEngineApi,
    explicit_summary_ids: &[String],
    query: Option<&str>,
    conversation_id: Option<i64>,
) -> anyhow::Result<Vec<SummaryCandidate>> {
    let retrieval = lcm.get_retrieval();
    let mut candidates: HashMap<String, SummaryCandidate> = HashMap::new();

    for summary_id in explicit_summary_ids {
        let described = retrieval.describe(summary_id).await?;
        let Some(described) = described else {
            anyhow::bail!("Summary not found: {}", summary_id);
        };
        match described.result {
            crate::retrieval::DescribeResultType::Summary(summary) => {
                candidates.insert(
                    summary_id.clone(),
                    SummaryCandidate {
                        summary_id: summary_id.clone(),
                        conversation_id: summary.conversation_id,
                    },
                );
            }
            crate::retrieval::DescribeResultType::File(_) => {
                anyhow::bail!("Summary not found: {}", summary_id);
            }
        }
    }

    if let Some(query) = query {
        let grep_result = retrieval
            .grep(crate::retrieval::GrepInput {
                query: query.to_string(),
                mode: "full_text".to_string(),
                scope: "summaries".to_string(),
                conversation_id,
                since: None,
                before: None,
                limit: None,
            })
            .await?;
        for summary in grep_result.summaries {
            candidates.insert(
                summary.summary_id.clone(),
                SummaryCandidate {
                    summary_id: summary.summary_id,
                    conversation_id: summary.conversation_id,
                },
            );
        }
    }

    Ok(candidates.into_values().collect())
}

#[derive(Clone)]
pub struct LcmExpandQueryTool {
    deps: Arc<dyn LcmDependencies>,
    lcm: Arc<dyn LcmContextEngineApi>,
    session_id: Option<String>,
    requester_session_key: Option<String>,
    session_key: Option<String>,
}

impl LcmExpandQueryTool {
    pub fn new(
        deps: Arc<dyn LcmDependencies>,
        lcm: Arc<dyn LcmContextEngineApi>,
        session_id: Option<String>,
        requester_session_key: Option<String>,
        session_key: Option<String>,
    ) -> Self {
        Self {
            deps,
            lcm,
            session_id,
            requester_session_key,
            session_key,
        }
    }

    pub async fn execute(&self, _tool_call_id: &str, params: Value) -> anyhow::Result<ToolResult> {
        let p = params.as_object().cloned().unwrap_or_default();
        let explicit_summary_ids = normalize_summary_ids(Some(
            &p.get("summaryIds")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(ToString::to_string)
                        .collect::<Vec<String>>()
                })
                .unwrap_or_default(),
        ));
        let query = p
            .get("query")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default()
            .to_string();
        let prompt = p
            .get("prompt")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default()
            .to_string();
        let requested_max_tokens = p
            .get("maxTokens")
            .and_then(Value::as_f64)
            .map(|v| v.trunc() as i64);
        let max_tokens = requested_max_tokens
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_MAX_ANSWER_TOKENS);
        let requested_token_cap = p
            .get("tokenCap")
            .and_then(Value::as_f64)
            .map(|v| v.trunc() as i64);
        let expansion_token_cap = requested_token_cap
            .filter(|v| *v > 0)
            .unwrap_or_else(|| i64::from(self.deps.config().max_expand_tokens.max(1)));

        if prompt.is_empty() {
            return Ok(json_result(json!({
                "error": "prompt is required."
            })));
        }
        if explicit_summary_ids.is_empty() && query.is_empty() {
            return Ok(json_result(json!({
                "error": "Either summaryIds or query must be provided."
            })));
        }

        let caller_session_key = self
            .requester_session_key
            .as_deref()
            .or(self.session_id.as_deref())
            .unwrap_or_default()
            .trim()
            .to_string();
        let request_id = resolve_expansion_request_id(Some(&caller_session_key));
        let recursion_check =
            evaluate_expansion_recursion_guard(Some(&caller_session_key), &request_id);
        record_expansion_delegation_telemetry(
            self.deps.as_ref(),
            "lcm_expand_query",
            TelemetryEvent::Start,
            &request_id,
            Some(&caller_session_key),
            recursion_check.expansion_depth,
            &recursion_check.origin_session_key,
            None,
            None,
        );
        if recursion_check.blocked {
            record_expansion_delegation_telemetry(
                self.deps.as_ref(),
                "lcm_expand_query",
                TelemetryEvent::Block,
                &request_id,
                Some(&caller_session_key),
                recursion_check.expansion_depth,
                &recursion_check.origin_session_key,
                recursion_check.reason.as_ref().map(|r| r.as_str()),
                None,
            );
            return Ok(json_result(json!({
                "errorCode": recursion_check.code,
                "error": recursion_check.message,
                "requestId": recursion_check.request_id,
                "expansionDepth": recursion_check.expansion_depth,
                "originSessionKey": recursion_check.origin_session_key,
                "reason": recursion_check.reason.as_ref().map(|r| r.as_str()),
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
        let mut scoped_conversation_id = conversation_scope.conversation_id;
        if !conversation_scope.all_conversations
            && scoped_conversation_id.is_none()
            && !caller_session_key.is_empty()
        {
            scoped_conversation_id = resolve_requester_conversation_scope_id(
                self.deps.as_ref(),
                &caller_session_key,
                self.lcm.as_ref(),
            )
            .await;
        }
        if !conversation_scope.all_conversations && scoped_conversation_id.is_none() {
            return Ok(json_result(json!({
                "error": "No LCM conversation found for this session. Provide conversationId or set allConversations=true."
            })));
        }

        let mut child_session_key = String::new();
        let mut grant_created = false;
        let result: anyhow::Result<ToolResult> = async {
            let candidates = resolve_summary_candidates(
                self.lcm.as_ref(),
                &explicit_summary_ids,
                if query.is_empty() { None } else { Some(query.as_str()) },
                scoped_conversation_id,
            )
            .await?;

            if candidates.is_empty() {
                if scoped_conversation_id.is_none() {
                    return Ok(json_result(json!({
                        "error": "No matching summaries found."
                    })));
                }
                return Ok(json_result(json!({
                    "answer": "No matching summaries found for this scope.",
                    "citedIds": [],
                    "sourceConversationId": scoped_conversation_id,
                    "expandedSummaryCount": 0,
                    "totalSourceTokens": 0,
                    "truncated": false,
                })));
            }

            let source_conversation_id = resolve_source_conversation_id(
                scoped_conversation_id,
                conversation_scope.all_conversations,
                &candidates,
            )?;
            let summary_ids = normalize_summary_ids(Some(
                &candidates
                    .iter()
                    .filter(|candidate| candidate.conversation_id == source_conversation_id)
                    .map(|candidate| candidate.summary_id.clone())
                    .collect::<Vec<String>>(),
            ));
            if summary_ids.is_empty() {
                return Ok(json_result(json!({
                    "error": "No summaryIds available after applying conversation scope."
                })));
            }

            let requester_agent_id = self.deps.normalize_agent_id(
                self.deps
                    .parse_agent_session_key(&caller_session_key)
                    .map(|(agent_id, _)| agent_id)
                    .as_deref(),
            );
            child_session_key = format!("agent:{}:subagent:{}", requester_agent_id, Uuid::new_v4());
            let child_expansion_depth = resolve_next_expansion_depth(Some(&caller_session_key));
            let origin_session_key = if recursion_check.origin_session_key.is_empty() {
                if caller_session_key.is_empty() {
                    "main".to_string()
                } else {
                    caller_session_key.clone()
                }
            } else {
                recursion_check.origin_session_key.clone()
            };

            create_delegated_expansion_grant(CreateDelegatedExpansionGrantInput {
                delegated_session_key: child_session_key.clone(),
                issuer_session_id: if caller_session_key.is_empty() {
                    "main".to_string()
                } else {
                    caller_session_key.clone()
                },
                allowed_conversation_ids: vec![source_conversation_id],
                allowed_summary_ids: None,
                max_depth: None,
                token_cap: Some(expansion_token_cap),
                ttl_ms: Some(DELEGATED_WAIT_TIMEOUT_MS + 30_000),
            });
            stamp_delegated_expansion_context(
                &child_session_key,
                &request_id,
                child_expansion_depth,
                &origin_session_key,
                "lcm_expand_query",
            );
            grant_created = true;

            let task = build_delegated_expand_query_task(
                &summary_ids,
                source_conversation_id,
                if query.is_empty() { None } else { Some(query.as_str()) },
                &prompt,
                max_tokens,
                expansion_token_cap,
                &request_id,
                child_expansion_depth,
                &origin_session_key,
            );
            let child_idem = Uuid::new_v4().to_string();
            let response = self
                .deps
                .call_gateway(GatewayCallRequest {
                    method: "agent".to_string(),
                    params: Some(json!({
                        "message": task,
                        "sessionKey": child_session_key,
                        "deliver": false,
                        "lane": self.deps.agent_lane_subagent(),
                        "idempotencyKey": child_idem,
                        "extraSystemPrompt": self.deps.build_subagent_system_prompt(
                            1,
                            8,
                            Some("Run lcm_expand and return prompt-focused JSON answer")
                        ),
                    })),
                    timeout_ms: Some(GATEWAY_TIMEOUT_MS),
                })
                .await?;
            let run_id = response
                .get("runId")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .unwrap_or_default()
                .to_string();
            if run_id.is_empty() {
                return Ok(json_result(json!({
                    "error": "Delegated expansion did not return a runId."
                })));
            }

            let wait = self
                .deps
                .call_gateway(GatewayCallRequest {
                    method: "agent.wait".to_string(),
                    params: Some(json!({
                        "runId": run_id,
                        "timeoutMs": DELEGATED_WAIT_TIMEOUT_MS
                    })),
                    timeout_ms: Some(DELEGATED_WAIT_TIMEOUT_MS),
                })
                .await?;
            let status = wait
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("error")
                .to_string();
            if status == "timeout" {
                record_expansion_delegation_telemetry(
                    self.deps.as_ref(),
                    "lcm_expand_query",
                    TelemetryEvent::Timeout,
                    &request_id,
                    Some(&caller_session_key),
                    child_expansion_depth,
                    &origin_session_key,
                    None,
                    Some(&run_id),
                );
                return Ok(json_result(json!({
                    "error": "lcm_expand_query timed out waiting for delegated expansion (120s)."
                })));
            }
            if status != "ok" {
                return Ok(json_result(json!({
                    "error": wait.get("error").and_then(Value::as_str).map(str::trim).filter(|v| !v.is_empty()).unwrap_or("Delegated expansion query failed.")
                })));
            }

            let reply_payload = self
                .deps
                .call_gateway(GatewayCallRequest {
                    method: "sessions.get".to_string(),
                    params: Some(json!({
                        "key": child_session_key,
                        "limit": 80
                    })),
                    timeout_ms: Some(GATEWAY_TIMEOUT_MS),
                })
                .await?;
            let messages = reply_payload
                .get("messages")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let reply = self.deps.read_latest_assistant_reply(&messages);
            let parsed = parse_delegated_expand_query_reply(reply.as_deref(), summary_ids.len());
            record_expansion_delegation_telemetry(
                self.deps.as_ref(),
                "lcm_expand_query",
                TelemetryEvent::Success,
                &request_id,
                Some(&caller_session_key),
                child_expansion_depth,
                &origin_session_key,
                None,
                Some(&run_id),
            );

            Ok(json_result(json!({
                "answer": parsed.answer,
                "citedIds": parsed.cited_ids,
                "sourceConversationId": source_conversation_id,
                "expandedSummaryCount": parsed.expanded_summary_count,
                "totalSourceTokens": parsed.total_source_tokens,
                "truncated": parsed.truncated,
            })))
        }
        .await;

        let final_result = match result {
            Ok(value) => value,
            Err(error) => json_result(json!({
                "error": error.to_string()
            })),
        };

        if !child_session_key.is_empty() {
            let _ = self
                .deps
                .call_gateway(GatewayCallRequest {
                    method: "sessions.delete".to_string(),
                    params: Some(json!({
                        "key": child_session_key,
                        "deleteTranscript": true
                    })),
                    timeout_ms: Some(GATEWAY_TIMEOUT_MS),
                })
                .await;
        }
        if grant_created && !child_session_key.is_empty() {
            revoke_delegated_expansion_grant_for_session(&child_session_key, true);
        }
        if !child_session_key.is_empty() {
            clear_delegated_expansion_context(&child_session_key);
        }

        Ok(final_result)
    }
}

pub fn create_lcm_expand_query_tool(
    deps: Arc<dyn LcmDependencies>,
    lcm: Arc<dyn LcmContextEngineApi>,
    session_id: Option<String>,
    requester_session_key: Option<String>,
    session_key: Option<String>,
) -> LcmExpandQueryTool {
    LcmExpandQueryTool::new(deps, lcm, session_id, requester_session_key, session_key)
}
