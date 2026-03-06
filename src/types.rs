use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::db::config::LcmConfig;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionContentBlock {
    #[serde(default)]
    pub r#type: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionResult {
    #[serde(default)]
    pub content: Vec<CompletionContentBlock>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub provider: Option<String>,
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub provider_api: Option<String>,
    #[serde(default)]
    pub auth_profile_id: Option<String>,
    #[serde(default)]
    pub agent_dir: Option<String>,
    #[serde(default)]
    pub runtime_config: Option<Value>,
    pub messages: Vec<CompletionMessage>,
    #[serde(default)]
    pub system: Option<String>,
    pub max_tokens: i32,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub reasoning: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GatewayCallRequest {
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
    #[serde(default)]
    pub timeout_ms: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageCost {
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(default)]
    pub cache_read: f64,
    #[serde(default)]
    pub cache_write: f64,
    #[serde(default)]
    pub total: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MessageUsage {
    #[serde(default)]
    pub input: i32,
    #[serde(default)]
    pub output: i32,
    #[serde(default)]
    pub cache_read: i32,
    #[serde(default)]
    pub cache_write: i32,
    #[serde(default)]
    pub total_tokens: i32,
    #[serde(default)]
    pub cost: UsageCost,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentMessage {
    pub role: String,
    pub content: Value,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_use_id: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub is_error: Option<bool>,
    #[serde(default)]
    pub usage: Option<MessageUsage>,
    #[serde(default)]
    pub timestamp: Option<i64>,
}

impl AgentMessage {
    pub fn new_text(role: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Value::String(text.into()),
            tool_call_id: None,
            tool_use_id: None,
            tool_name: None,
            stop_reason: None,
            is_error: None,
            usage: None,
            timestamp: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelRef {
    pub provider: String,
    pub model: String,
}

#[async_trait]
pub trait LcmLogger: Send + Sync {
    fn info(&self, msg: &str);
    fn warn(&self, msg: &str);
    fn error(&self, msg: &str);
    fn debug(&self, msg: &str);
}

#[async_trait]
pub trait LcmDependencies: Send + Sync {
    fn config(&self) -> &LcmConfig;

    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResult>;

    async fn call_gateway(&self, request: GatewayCallRequest) -> anyhow::Result<Value>;

    fn resolve_model(
        &self,
        model_ref: Option<&str>,
        provider_hint: Option<&str>,
    ) -> anyhow::Result<ModelRef>;

    fn get_api_key(&self, provider: &str, model: &str) -> Option<String>;

    fn require_api_key(&self, provider: &str, model: &str) -> anyhow::Result<String>;

    fn parse_agent_session_key(&self, session_key: &str) -> Option<(String, String)>;

    fn is_subagent_session_key(&self, session_key: &str) -> bool;

    fn normalize_agent_id(&self, id: Option<&str>) -> String;

    fn build_subagent_system_prompt(
        &self,
        depth: i32,
        max_depth: i32,
        task_summary: Option<&str>,
    ) -> String;

    fn read_latest_assistant_reply(&self, messages: &[Value]) -> Option<String>;

    fn resolve_agent_dir(&self) -> String;

    async fn resolve_session_id_from_session_key(
        &self,
        session_key: &str,
    ) -> anyhow::Result<Option<String>>;

    fn agent_lane_subagent(&self) -> &str;

    fn logger(&self) -> &dyn LcmLogger;
}
