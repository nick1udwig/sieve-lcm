#[derive(Clone, Debug, PartialEq)]
pub struct CompleteSimpleOptions {
    pub api_key: Option<String>,
    pub max_tokens: i32,
    pub temperature: Option<f64>,
    pub reasoning: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BuildCompleteSimpleOptionsParams {
    pub api: Option<String>,
    pub api_key: Option<String>,
    pub max_tokens: i32,
    pub temperature: Option<f64>,
    pub reasoning: Option<String>,
}

pub fn should_omit_temperature_for_api(api: Option<&str>) -> bool {
    api.unwrap_or_default().trim().to_lowercase() == "openai-codex-responses"
}

pub fn build_complete_simple_options(
    params: BuildCompleteSimpleOptionsParams,
) -> CompleteSimpleOptions {
    let mut out = CompleteSimpleOptions {
        api_key: params.api_key,
        max_tokens: params.max_tokens,
        temperature: None,
        reasoning: None,
    };

    if let Some(temperature) = params.temperature {
        if temperature.is_finite() && !should_omit_temperature_for_api(params.api.as_deref()) {
            out.temperature = Some(temperature);
        }
    }

    if let Some(reasoning) = params.reasoning {
        let value = reasoning.trim();
        if !value.is_empty() {
            out.reasoning = Some(value.to_string());
        }
    }

    out
}
