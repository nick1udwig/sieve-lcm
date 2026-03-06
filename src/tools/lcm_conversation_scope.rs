use chrono::{DateTime, Utc};
use serde_json::{Map, Value};

use crate::engine::LcmContextEngineApi;
use crate::types::LcmDependencies;

#[derive(Clone, Debug, PartialEq)]
pub struct LcmConversationScope {
    pub conversation_id: Option<i64>,
    pub all_conversations: bool,
}

pub fn parse_iso_timestamp_param(
    params: &Map<String, Value>,
    key: &str,
) -> anyhow::Result<Option<DateTime<Utc>>> {
    let Some(value) = params.get(key).and_then(|v| v.as_str()) else {
        return Ok(None);
    };
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }
    let parsed = DateTime::parse_from_rfc3339(value)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|_| anyhow::anyhow!("{} must be a valid ISO timestamp.", key))?;
    Ok(Some(parsed))
}

pub async fn resolve_lcm_conversation_scope(
    lcm: &dyn LcmContextEngineApi,
    params: &Map<String, Value>,
    session_id: Option<&str>,
    session_key: Option<&str>,
    deps: Option<&dyn LcmDependencies>,
) -> anyhow::Result<LcmConversationScope> {
    let explicit_conversation_id = params
        .get("conversationId")
        .and_then(|v| v.as_f64())
        .and_then(|v| {
            if v.is_finite() {
                Some(v.trunc() as i64)
            } else {
                None
            }
        });
    if let Some(conversation_id) = explicit_conversation_id {
        return Ok(LcmConversationScope {
            conversation_id: Some(conversation_id),
            all_conversations: false,
        });
    }

    if params.get("allConversations").and_then(Value::as_bool) == Some(true) {
        return Ok(LcmConversationScope {
            conversation_id: None,
            all_conversations: true,
        });
    }

    let mut normalized_session_id = session_id
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string);
    if normalized_session_id.is_none() {
        if let (Some(key), Some(deps)) =
            (session_key.map(str::trim).filter(|k| !k.is_empty()), deps)
        {
            normalized_session_id = deps.resolve_session_id_from_session_key(key).await?;
        }
    }
    let Some(session_id) = normalized_session_id else {
        return Ok(LcmConversationScope {
            conversation_id: None,
            all_conversations: false,
        });
    };

    let conversation = lcm
        .get_conversation_store()
        .get_conversation_by_session_id(&session_id)
        .await?;
    Ok(LcmConversationScope {
        conversation_id: conversation.map(|c| c.conversation_id),
        all_conversations: false,
    })
}
