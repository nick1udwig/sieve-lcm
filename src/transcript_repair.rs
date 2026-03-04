use std::collections::{HashMap, HashSet};

use chrono::Utc;
use serde_json::{json, Value};

use crate::types::AgentMessage;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ToolCallLike {
    id: String,
    name: Option<String>,
}

fn extract_tool_calls_from_assistant(msg: &AgentMessage) -> Vec<ToolCallLike> {
    let Some(content) = msg.content.as_array() else {
        return vec![];
    };
    let allowed = HashSet::from(["toolCall", "toolUse", "functionCall"]);
    let mut out = vec![];
    for block in content {
        let Some(obj) = block.as_object() else {
            continue;
        };
        let Some(id) = obj.get("id").and_then(Value::as_str) else {
            continue;
        };
        let is_tool_type = obj
            .get("type")
            .and_then(Value::as_str)
            .map(|t| allowed.contains(t))
            .unwrap_or(false);
        if !is_tool_type {
            continue;
        }
        out.push(ToolCallLike {
            id: id.to_string(),
            name: obj
                .get("name")
                .and_then(Value::as_str)
                .map(ToString::to_string),
        });
    }
    out
}

fn extract_tool_result_id(msg: &AgentMessage) -> Option<String> {
    msg.tool_call_id
        .clone()
        .or_else(|| msg.tool_use_id.clone())
        .filter(|v| !v.is_empty())
}

fn make_missing_tool_result(tool_call_id: &str, tool_name: Option<&str>) -> AgentMessage {
    AgentMessage {
        role: "toolResult".to_string(),
        tool_call_id: Some(tool_call_id.to_string()),
        tool_use_id: None,
        tool_name: Some(tool_name.unwrap_or("unknown").to_string()),
        content: json!([
            {
                "type": "text",
                "text": "[lossless-claw] missing tool result in session history; inserted synthetic error result for transcript repair."
            }
        ]),
        is_error: Some(true),
        timestamp: Some(Utc::now().timestamp_millis()),
        stop_reason: None,
        usage: None,
    }
}

pub fn sanitize_tool_use_result_pairing(messages: Vec<AgentMessage>) -> Vec<AgentMessage> {
    let mut out: Vec<AgentMessage> = vec![];
    let mut seen_tool_result_ids: HashSet<String> = HashSet::new();
    let mut changed = false;
    let mut i = 0_usize;
    fn push_tool_result(
        msg: AgentMessage,
        out: &mut Vec<AgentMessage>,
        seen_tool_result_ids: &mut HashSet<String>,
        changed: &mut bool,
    ) {
        if let Some(id) = extract_tool_result_id(&msg) {
            if seen_tool_result_ids.contains(&id) {
                *changed = true;
                return;
            }
            seen_tool_result_ids.insert(id);
        }
        out.push(msg);
    }

    while i < messages.len() {
        let msg = messages[i].clone();
        if msg.role != "assistant" {
            if msg.role != "toolResult" {
                out.push(msg);
            } else {
                changed = true;
            }
            i += 1;
            continue;
        }

        if matches!(msg.stop_reason.as_deref(), Some("error" | "aborted")) {
            out.push(msg);
            i += 1;
            continue;
        }

        let tool_calls = extract_tool_calls_from_assistant(&msg);
        if tool_calls.is_empty() {
            out.push(msg);
            i += 1;
            continue;
        }

        let tool_call_ids: HashSet<String> = tool_calls.iter().map(|t| t.id.clone()).collect();
        let mut span_results_by_id: HashMap<String, AgentMessage> = HashMap::new();
        let mut remainder: Vec<AgentMessage> = vec![];
        let mut j = i + 1;

        while j < messages.len() {
            let next = messages[j].clone();
            if next.role == "assistant" {
                break;
            }
            if next.role == "toolResult" {
                if let Some(id) = extract_tool_result_id(&next) {
                    if tool_call_ids.contains(&id) {
                        if seen_tool_result_ids.contains(&id) {
                            changed = true;
                            j += 1;
                            continue;
                        }
                        span_results_by_id.entry(id).or_insert(next);
                        j += 1;
                        continue;
                    }
                }
            }
            if next.role != "toolResult" {
                remainder.push(next);
            } else {
                changed = true;
            }
            j += 1;
        }

        out.push(msg);
        for call in tool_calls {
            if let Some(existing) = span_results_by_id.remove(&call.id) {
                push_tool_result(existing, &mut out, &mut seen_tool_result_ids, &mut changed);
            } else {
                changed = true;
                push_tool_result(
                    make_missing_tool_result(&call.id, call.name.as_deref()),
                    &mut out,
                    &mut seen_tool_result_ids,
                    &mut changed,
                );
            }
        }
        out.extend(remainder);
        i = j;
    }

    if changed {
        out
    } else {
        messages
    }
}
