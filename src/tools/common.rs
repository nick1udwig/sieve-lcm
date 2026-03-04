use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolContentBlock {
    pub r#type: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<ToolContentBlock>,
    pub details: Value,
}

pub fn json_result(payload: Value) -> ToolResult {
    ToolResult {
        content: vec![ToolContentBlock {
            r#type: "text".to_string(),
            text: serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string()),
        }],
        details: payload,
    }
}

#[derive(Clone, Debug)]
pub struct ReadStringParamOptions {
    pub required: bool,
    pub trim: bool,
    pub allow_empty: bool,
    pub label: Option<String>,
}

impl Default for ReadStringParamOptions {
    fn default() -> Self {
        Self {
            required: false,
            trim: true,
            allow_empty: false,
            label: None,
        }
    }
}

pub fn read_string_param(
    params: &serde_json::Map<String, Value>,
    key: &str,
    options: ReadStringParamOptions,
) -> anyhow::Result<Option<String>> {
    let raw = params.get(key);
    if raw.is_none() || raw == Some(&Value::Null) {
        if options.required {
            anyhow::bail!("{} is required.", options.label.unwrap_or_else(|| key.to_string()));
        }
        return Ok(None);
    }
    let Some(raw_str) = raw.and_then(Value::as_str) else {
        anyhow::bail!(
            "{} must be a string.",
            options.label.unwrap_or_else(|| key.to_string())
        );
    };
    let value = if options.trim {
        raw_str.trim().to_string()
    } else {
        raw_str.to_string()
    };
    if !options.allow_empty && value.is_empty() {
        if options.required {
            anyhow::bail!("{} is required.", options.label.unwrap_or_else(|| key.to_string()));
        }
        return Ok(None);
    }
    Ok(Some(value))
}
