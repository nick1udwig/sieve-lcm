use sieve_lcm::complete_options::{
    build_complete_simple_options, should_omit_temperature_for_api, BuildCompleteSimpleOptionsParams,
};

#[test]
fn omits_temperature_for_openai_codex_responses() {
    assert!(should_omit_temperature_for_api(Some("openai-codex-responses")));

    let options = build_complete_simple_options(BuildCompleteSimpleOptionsParams {
        api: Some("openai-codex-responses".to_string()),
        api_key: Some("k".to_string()),
        max_tokens: 1024,
        temperature: Some(0.2),
        reasoning: None,
    });

    assert_eq!(options.temperature, None);
}

#[test]
fn keeps_temperature_for_non_codex_apis() {
    assert!(!should_omit_temperature_for_api(Some("openai-responses")));

    let options = build_complete_simple_options(BuildCompleteSimpleOptionsParams {
        api: Some("openai-responses".to_string()),
        api_key: Some("k".to_string()),
        max_tokens: 1024,
        temperature: Some(0.2),
        reasoning: Some("low".to_string()),
    });

    assert_eq!(options.temperature, Some(0.2));
    assert_eq!(options.reasoning.as_deref(), Some("low"));
}
