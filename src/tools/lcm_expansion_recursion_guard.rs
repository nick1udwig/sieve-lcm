use std::collections::{HashMap, HashSet};

use chrono::Utc;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::json;
use uuid::Uuid;

use crate::expansion_auth::resolve_delegated_expansion_grant_id;
use crate::types::LcmDependencies;

pub const EXPANSION_RECURSION_ERROR_CODE: &str = "EXPANSION_RECURSION_BLOCKED";
const EXPANSION_DELEGATION_DEPTH_CAP: i64 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TelemetryEvent {
    Start,
    Block,
    Timeout,
    Success,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DelegatedExpansionContext {
    pub request_id: String,
    pub expansion_depth: i64,
    pub origin_session_key: String,
    pub stamped_by: String,
    pub created_at: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExpansionRecursionBlockReason {
    DepthCap,
    IdempotentReentry,
}

impl ExpansionRecursionBlockReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExpansionRecursionBlockReason::DepthCap => "depth_cap",
            ExpansionRecursionBlockReason::IdempotentReentry => "idempotent_reentry",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpansionRecursionGuardDecision {
    pub blocked: bool,
    pub code: Option<String>,
    pub reason: Option<ExpansionRecursionBlockReason>,
    pub message: Option<String>,
    pub request_id: String,
    pub expansion_depth: i64,
    pub origin_session_key: String,
}

static DELEGATED_CONTEXT_BY_SESSION_KEY: Lazy<Mutex<HashMap<String, DelegatedExpansionContext>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static BLOCKED_REQUEST_IDS_BY_SESSION_KEY: Lazy<Mutex<HashMap<String, HashSet<String>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static TELEMETRY_COUNTERS: Lazy<Mutex<HashMap<TelemetryEvent, i64>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(TelemetryEvent::Start, 0);
    map.insert(TelemetryEvent::Block, 0);
    map.insert(TelemetryEvent::Timeout, 0);
    map.insert(TelemetryEvent::Success, 0);
    Mutex::new(map)
});

fn normalize_session_key(session_key: Option<&str>) -> String {
    session_key.unwrap_or_default().trim().to_string()
}

fn build_expansion_recursion_recovery_guidance(origin_session_key: &str) -> String {
    format!(
        "Recovery: In delegated sub-agent sessions, call `lcm_expand` directly and synthesize \
your answer from that result. Do NOT call `lcm_expand_query` from delegated context. \
If deeper delegation is required, return to the origin session ({}) and call \
`lcm_expand_query` there.",
        origin_session_key
    )
}

fn resolve_fallback_delegated_context(
    session_key: &str,
    request_id: &str,
) -> Option<DelegatedExpansionContext> {
    if session_key.is_empty() {
        return None;
    }
    let grant_id = resolve_delegated_expansion_grant_id(session_key)?;
    if grant_id.is_empty() {
        return None;
    }
    Some(DelegatedExpansionContext {
        request_id: request_id.to_string(),
        expansion_depth: EXPANSION_DELEGATION_DEPTH_CAP,
        origin_session_key: session_key.to_string(),
        stamped_by: "delegated_grant".to_string(),
        created_at: Utc::now().to_rfc3339(),
    })
}

pub fn create_expansion_request_id() -> String {
    Uuid::new_v4().to_string()
}

pub fn resolve_expansion_request_id(session_key: Option<&str>) -> String {
    let key = normalize_session_key(session_key);
    DELEGATED_CONTEXT_BY_SESSION_KEY
        .lock()
        .get(&key)
        .map(|ctx| ctx.request_id.clone())
        .unwrap_or_else(create_expansion_request_id)
}

pub fn resolve_next_expansion_depth(session_key: Option<&str>) -> i64 {
    let key = normalize_session_key(session_key);
    if key.is_empty() {
        return 1;
    }
    if let Some(existing) = DELEGATED_CONTEXT_BY_SESSION_KEY.lock().get(&key) {
        return existing.expansion_depth + 1;
    }
    if resolve_delegated_expansion_grant_id(&key).is_some() {
        return EXPANSION_DELEGATION_DEPTH_CAP + 1;
    }
    1
}

pub fn stamp_delegated_expansion_context(
    session_key: &str,
    request_id: &str,
    expansion_depth: i64,
    origin_session_key: &str,
    stamped_by: &str,
) -> DelegatedExpansionContext {
    let key = normalize_session_key(Some(session_key));
    let context = DelegatedExpansionContext {
        request_id: request_id.to_string(),
        expansion_depth: expansion_depth.max(0),
        origin_session_key: {
            let trimmed = origin_session_key.trim();
            if trimmed.is_empty() {
                "main".to_string()
            } else {
                trimmed.to_string()
            }
        },
        stamped_by: stamped_by.to_string(),
        created_at: Utc::now().to_rfc3339(),
    };
    if !key.is_empty() {
        DELEGATED_CONTEXT_BY_SESSION_KEY
            .lock()
            .insert(key, context.clone());
    }
    context
}

pub fn clear_delegated_expansion_context(session_key: &str) {
    let key = normalize_session_key(Some(session_key));
    if key.is_empty() {
        return;
    }
    DELEGATED_CONTEXT_BY_SESSION_KEY.lock().remove(&key);
    BLOCKED_REQUEST_IDS_BY_SESSION_KEY.lock().remove(&key);
}

pub fn evaluate_expansion_recursion_guard(
    session_key: Option<&str>,
    request_id: &str,
) -> ExpansionRecursionGuardDecision {
    let session_key = normalize_session_key(session_key);
    let request_id = request_id.trim().to_string();
    let delegated_context = DELEGATED_CONTEXT_BY_SESSION_KEY
        .lock()
        .get(&session_key)
        .cloned()
        .or_else(|| resolve_fallback_delegated_context(&session_key, &request_id));

    let Some(context) = delegated_context else {
        return ExpansionRecursionGuardDecision {
            blocked: false,
            code: None,
            reason: None,
            message: None,
            request_id,
            expansion_depth: 0,
            origin_session_key: if session_key.is_empty() {
                "main".to_string()
            } else {
                session_key
            },
        };
    };

    if context.expansion_depth < EXPANSION_DELEGATION_DEPTH_CAP {
        return ExpansionRecursionGuardDecision {
            blocked: false,
            code: None,
            reason: None,
            message: None,
            request_id,
            expansion_depth: context.expansion_depth,
            origin_session_key: context.origin_session_key,
        };
    }

    let mut blocked = BLOCKED_REQUEST_IDS_BY_SESSION_KEY.lock();
    let seen_request_ids = blocked.entry(session_key.clone()).or_default();
    let idempotent_reentry = seen_request_ids.contains(&request_id);
    seen_request_ids.insert(request_id.clone());
    drop(blocked);

    let reason = if idempotent_reentry {
        ExpansionRecursionBlockReason::IdempotentReentry
    } else {
        ExpansionRecursionBlockReason::DepthCap
    };
    let message = format!(
        "{}: Expansion delegation blocked at depth {} ({}; requestId={}; origin={}). {}",
        EXPANSION_RECURSION_ERROR_CODE,
        context.expansion_depth,
        reason.as_str(),
        request_id,
        context.origin_session_key,
        build_expansion_recursion_recovery_guidance(&context.origin_session_key)
    );

    ExpansionRecursionGuardDecision {
        blocked: true,
        code: Some(EXPANSION_RECURSION_ERROR_CODE.to_string()),
        reason: Some(reason),
        message: Some(message),
        request_id,
        expansion_depth: context.expansion_depth,
        origin_session_key: context.origin_session_key,
    }
}

pub fn record_expansion_delegation_telemetry(
    deps: &dyn LcmDependencies,
    component: &str,
    event: TelemetryEvent,
    request_id: &str,
    session_key: Option<&str>,
    expansion_depth: i64,
    origin_session_key: &str,
    reason: Option<&str>,
    run_id: Option<&str>,
) {
    let mut counters = TELEMETRY_COUNTERS.lock();
    let count = counters.entry(event).or_insert(0);
    *count += 1;
    let normalized_session_key = normalize_session_key(session_key);
    let session_key_value = if normalized_session_key.is_empty() {
        None
    } else {
        Some(normalized_session_key)
    };
    let payload = json!({
        "component": component,
        "event": match event {
            TelemetryEvent::Start => "start",
            TelemetryEvent::Block => "block",
            TelemetryEvent::Timeout => "timeout",
            TelemetryEvent::Success => "success",
        },
        "requestId": request_id,
        "sessionKey": session_key_value,
        "expansionDepth": expansion_depth,
        "originSessionKey": origin_session_key,
        "reason": reason,
        "runId": run_id,
        "counters": {
            "start": *counters.get(&TelemetryEvent::Start).unwrap_or(&0),
            "block": *counters.get(&TelemetryEvent::Block).unwrap_or(&0),
            "timeout": *counters.get(&TelemetryEvent::Timeout).unwrap_or(&0),
            "success": *counters.get(&TelemetryEvent::Success).unwrap_or(&0),
        }
    });
    drop(counters);

    let line = format!("[lcm][expansion_delegation] {}", payload);
    match event {
        TelemetryEvent::Start | TelemetryEvent::Success => deps.logger().info(&line),
        TelemetryEvent::Block | TelemetryEvent::Timeout => deps.logger().warn(&line),
    }
}

pub fn get_delegated_expansion_context_for_tests(
    session_key: &str,
) -> Option<DelegatedExpansionContext> {
    DELEGATED_CONTEXT_BY_SESSION_KEY
        .lock()
        .get(&normalize_session_key(Some(session_key)))
        .cloned()
}

pub fn get_expansion_delegation_telemetry_snapshot_for_tests() -> HashMap<String, i64> {
    let counters = TELEMETRY_COUNTERS.lock();
    HashMap::from([
        (
            "start".to_string(),
            *counters.get(&TelemetryEvent::Start).unwrap_or(&0),
        ),
        (
            "block".to_string(),
            *counters.get(&TelemetryEvent::Block).unwrap_or(&0),
        ),
        (
            "timeout".to_string(),
            *counters.get(&TelemetryEvent::Timeout).unwrap_or(&0),
        ),
        (
            "success".to_string(),
            *counters.get(&TelemetryEvent::Success).unwrap_or(&0),
        ),
    ])
}

pub fn reset_expansion_delegation_guard_for_tests() {
    DELEGATED_CONTEXT_BY_SESSION_KEY.lock().clear();
    BLOCKED_REQUEST_IDS_BY_SESSION_KEY.lock().clear();
    let mut counters = TELEMETRY_COUNTERS.lock();
    counters.insert(TelemetryEvent::Start, 0);
    counters.insert(TelemetryEvent::Block, 0);
    counters.insert(TelemetryEvent::Timeout, 0);
    counters.insert(TelemetryEvent::Success, 0);
}
