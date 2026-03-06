use sieve_lcm::expansion_policy::{
    EXPANSION_ROUTING_THRESHOLDS, LcmExpansionRoutingAction, LcmExpansionRoutingInput,
    LcmExpansionRoutingIntent, LcmExpansionTokenRiskLevel, classify_expansion_token_risk,
    decide_lcm_expansion_routing, detect_broad_time_range_indicator, detect_multi_hop_indicator,
    estimate_expansion_tokens,
};

#[test]
fn applies_the_expected_route_vs_delegate_decision_matrix() {
    let cases = vec![
        (
            "query probe with zero candidates",
            LcmExpansionRoutingInput {
                intent: LcmExpansionRoutingIntent::QueryProbe,
                query: Some("recent auth failures".to_string()),
                candidate_summary_count: 0,
                requested_max_depth: Some(3),
                token_cap: 1200,
                include_messages: false,
            },
            LcmExpansionRoutingAction::AnswerDirectly,
            "direct_by_no_candidates",
            true,
        ),
        (
            "query probe at low-complexity bounds",
            LcmExpansionRoutingInput {
                intent: LcmExpansionRoutingIntent::QueryProbe,
                query: Some("failed login".to_string()),
                candidate_summary_count: 1,
                requested_max_depth: Some(2),
                token_cap: 10_000,
                include_messages: false,
            },
            LcmExpansionRoutingAction::AnswerDirectly,
            "direct_by_low_complexity_probe",
            true,
        ),
        (
            "explicit expand under delegation thresholds",
            LcmExpansionRoutingInput {
                intent: LcmExpansionRoutingIntent::ExplicitExpand,
                query: None,
                candidate_summary_count: 2,
                requested_max_depth: Some(2),
                token_cap: 10_000,
                include_messages: false,
            },
            LcmExpansionRoutingAction::ExpandShallow,
            "delegate_by_depth",
            false,
        ),
        (
            "query probe with deep depth does not auto-delegate",
            LcmExpansionRoutingInput {
                intent: LcmExpansionRoutingIntent::QueryProbe,
                query: Some("auth chain".to_string()),
                candidate_summary_count: 2,
                requested_max_depth: Some(4),
                token_cap: 10_000,
                include_messages: false,
            },
            LcmExpansionRoutingAction::ExpandShallow,
            "delegate_by_depth",
            false,
        ),
        (
            "query probe with many candidates does not auto-delegate",
            LcmExpansionRoutingInput {
                intent: LcmExpansionRoutingIntent::QueryProbe,
                query: Some("incident spread".to_string()),
                candidate_summary_count: 6,
                requested_max_depth: Some(2),
                token_cap: 10_000,
                include_messages: false,
            },
            LcmExpansionRoutingAction::ExpandShallow,
            "delegate_by_candidate_count",
            false,
        ),
        (
            "query probe with broad range and multi-hop indicators",
            LcmExpansionRoutingInput {
                intent: LcmExpansionRoutingIntent::QueryProbe,
                query: Some(
                    "build timeline from 2021 to 2025 and explain root cause chain".to_string(),
                ),
                candidate_summary_count: 2,
                requested_max_depth: Some(2),
                token_cap: 10_000,
                include_messages: false,
            },
            LcmExpansionRoutingAction::DelegateTraversal,
            "delegate_by_broad_time_range_and_multi_hop",
            true,
        ),
    ];

    for (_name, input, expected_action, trigger, expected_value) in cases {
        let decision = decide_lcm_expansion_routing(input);
        assert_eq!(decision.action, expected_action);
        let observed = match trigger {
            "direct_by_no_candidates" => decision.triggers.direct_by_no_candidates,
            "direct_by_low_complexity_probe" => decision.triggers.direct_by_low_complexity_probe,
            "delegate_by_depth" => decision.triggers.delegate_by_depth,
            "delegate_by_candidate_count" => decision.triggers.delegate_by_candidate_count,
            "delegate_by_broad_time_range_and_multi_hop" => {
                decision.triggers.delegate_by_broad_time_range_and_multi_hop
            }
            _ => unreachable!("unsupported trigger"),
        };
        assert_eq!(observed, expected_value);
    }
}

#[test]
fn answers_directly_when_no_candidate_summaries_are_available() {
    let decision = decide_lcm_expansion_routing(LcmExpansionRoutingInput {
        intent: LcmExpansionRoutingIntent::QueryProbe,
        query: Some("recent auth failures".to_string()),
        candidate_summary_count: 0,
        requested_max_depth: Some(3),
        token_cap: 1200,
        include_messages: false,
    });
    assert_eq!(decision.action, LcmExpansionRoutingAction::AnswerDirectly);
    assert!(decision.triggers.direct_by_no_candidates);
}

#[test]
fn answers_directly_for_low_complexity_query_probes() {
    let decision = decide_lcm_expansion_routing(LcmExpansionRoutingInput {
        intent: LcmExpansionRoutingIntent::QueryProbe,
        query: Some("failed login".to_string()),
        candidate_summary_count: 1,
        requested_max_depth: Some(2),
        token_cap: 10_000,
        include_messages: false,
    });
    assert_eq!(decision.action, LcmExpansionRoutingAction::AnswerDirectly);
    assert!(decision.triggers.direct_by_low_complexity_probe);
}

#[test]
fn uses_shallow_expansion_for_low_complexity_explicit_expand_requests() {
    let decision = decide_lcm_expansion_routing(LcmExpansionRoutingInput {
        intent: LcmExpansionRoutingIntent::ExplicitExpand,
        query: None,
        candidate_summary_count: 1,
        requested_max_depth: Some(2),
        token_cap: 10_000,
        include_messages: false,
    });
    assert_eq!(decision.action, LcmExpansionRoutingAction::ExpandShallow);
}

#[test]
fn does_not_delegate_solely_due_to_depth() {
    let below = decide_lcm_expansion_routing(LcmExpansionRoutingInput {
        intent: LcmExpansionRoutingIntent::QueryProbe,
        query: Some("auth chain".to_string()),
        candidate_summary_count: 2,
        requested_max_depth: Some(3),
        token_cap: 10_000,
        include_messages: false,
    });
    let at = decide_lcm_expansion_routing(LcmExpansionRoutingInput {
        intent: LcmExpansionRoutingIntent::QueryProbe,
        query: Some("auth chain".to_string()),
        candidate_summary_count: 2,
        requested_max_depth: Some(4),
        token_cap: 10_000,
        include_messages: false,
    });

    assert_eq!(below.action, LcmExpansionRoutingAction::ExpandShallow);
    assert_eq!(at.action, LcmExpansionRoutingAction::ExpandShallow);
    assert!(!at.triggers.delegate_by_depth);
}

#[test]
fn does_not_delegate_solely_due_to_candidate_count() {
    let below = decide_lcm_expansion_routing(LcmExpansionRoutingInput {
        intent: LcmExpansionRoutingIntent::QueryProbe,
        query: Some("incident spread".to_string()),
        candidate_summary_count: 5,
        requested_max_depth: Some(2),
        token_cap: 10_000,
        include_messages: false,
    });
    let at = decide_lcm_expansion_routing(LcmExpansionRoutingInput {
        intent: LcmExpansionRoutingIntent::QueryProbe,
        query: Some("incident spread".to_string()),
        candidate_summary_count: 6,
        requested_max_depth: Some(2),
        token_cap: 10_000,
        include_messages: false,
    });

    assert_eq!(below.action, LcmExpansionRoutingAction::ExpandShallow);
    assert_eq!(at.action, LcmExpansionRoutingAction::ExpandShallow);
    assert!(!at.triggers.delegate_by_candidate_count);
}

#[test]
fn delegates_when_token_risk_crosses_the_high_risk_boundary() {
    let requested_max_depth = Some(3);
    let candidate_summary_count = 3;
    let include_messages = true;
    let broad_time_range_indicator = false;
    let multi_hop_indicator = true;

    let estimated_tokens = estimate_expansion_tokens(
        requested_max_depth,
        candidate_summary_count,
        include_messages,
        broad_time_range_indicator,
        multi_hop_indicator,
    );
    let cap_just_below_high_risk = (estimated_tokens as f64
        / EXPANSION_ROUTING_THRESHOLDS.high_token_risk_ratio)
        .ceil() as i64
        - 1;
    let cap_at_or_above_high_risk = (estimated_tokens as f64
        / EXPANSION_ROUTING_THRESHOLDS.high_token_risk_ratio)
        .ceil() as i64;

    let below = decide_lcm_expansion_routing(LcmExpansionRoutingInput {
        intent: LcmExpansionRoutingIntent::QueryProbe,
        query: Some("root cause chain".to_string()),
        candidate_summary_count,
        requested_max_depth,
        include_messages,
        token_cap: cap_at_or_above_high_risk,
    });
    let at = decide_lcm_expansion_routing(LcmExpansionRoutingInput {
        intent: LcmExpansionRoutingIntent::QueryProbe,
        query: Some("root cause chain".to_string()),
        candidate_summary_count,
        requested_max_depth,
        include_messages,
        token_cap: cap_just_below_high_risk.max(1),
    });

    assert_eq!(below.action, LcmExpansionRoutingAction::ExpandShallow);
    assert_eq!(at.action, LcmExpansionRoutingAction::DelegateTraversal);
    assert!(at.triggers.delegate_by_token_risk);
}

#[test]
fn delegates_for_combined_broad_time_range_and_multi_hop_indicators() {
    let decision = decide_lcm_expansion_routing(LcmExpansionRoutingInput {
        intent: LcmExpansionRoutingIntent::QueryProbe,
        query: Some("build timeline from 2021 to 2025 and explain root cause chain".to_string()),
        candidate_summary_count: 2,
        requested_max_depth: Some(2),
        token_cap: 10_000,
        include_messages: false,
    });
    assert_eq!(
        decision.action,
        LcmExpansionRoutingAction::DelegateTraversal
    );
    assert!(decision.triggers.delegate_by_broad_time_range_and_multi_hop);
}

#[test]
fn detects_broad_time_range_year_windows_of_at_least_two_years() {
    assert!(detect_broad_time_range_indicator(Some(
        "events from 2022 to 2024"
    )));
    assert!(!detect_broad_time_range_indicator(Some(
        "events from 2024 to 2025"
    )));
}

#[test]
fn detects_multi_hop_from_traversal_depth_and_query_language() {
    assert!(detect_multi_hop_indicator(
        Some("normal summary lookup"),
        Some(EXPANSION_ROUTING_THRESHOLDS.multi_hop_depth_threshold),
        1,
    ));
    assert!(detect_multi_hop_indicator(
        Some("explain the chain of events"),
        Some(1),
        1,
    ));
}

#[test]
fn classifies_token_risk_at_exact_ratio_boundaries() {
    let (moderate_ratio, moderate_level) = classify_expansion_token_risk(35, 100);
    let (high_ratio, high_level) = classify_expansion_token_risk(70, 100);

    assert_eq!(moderate_level, LcmExpansionTokenRiskLevel::Moderate);
    assert_eq!(high_level, LcmExpansionTokenRiskLevel::High);
    assert!((moderate_ratio - EXPANSION_ROUTING_THRESHOLDS.moderate_token_risk_ratio).abs() < 1e-8);
    assert!((high_ratio - EXPANSION_ROUTING_THRESHOLDS.high_token_risk_ratio).abs() < 1e-8);
}
