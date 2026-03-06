use regex::Regex;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LcmExpansionRoutingIntent {
    QueryProbe,
    ExplicitExpand,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LcmExpansionRoutingAction {
    AnswerDirectly,
    ExpandShallow,
    DelegateTraversal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LcmExpansionTokenRiskLevel {
    Low,
    Moderate,
    High,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LcmExpansionRoutingInput {
    pub intent: LcmExpansionRoutingIntent,
    pub query: Option<String>,
    pub requested_max_depth: Option<i64>,
    pub candidate_summary_count: i64,
    pub token_cap: i64,
    pub include_messages: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpansionIndicators {
    pub broad_time_range: bool,
    pub multi_hop_retrieval: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpansionTriggers {
    pub direct_by_no_candidates: bool,
    pub direct_by_low_complexity_probe: bool,
    pub delegate_by_depth: bool,
    pub delegate_by_candidate_count: bool,
    pub delegate_by_token_risk: bool,
    pub delegate_by_broad_time_range_and_multi_hop: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LcmExpansionRoutingDecision {
    pub action: LcmExpansionRoutingAction,
    pub normalized_max_depth: i64,
    pub candidate_summary_count: i64,
    pub estimated_tokens: i64,
    pub token_cap: i64,
    pub token_risk_ratio: f64,
    pub token_risk_level: LcmExpansionTokenRiskLevel,
    pub indicators: ExpansionIndicators,
    pub triggers: ExpansionTriggers,
    pub reasons: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpansionRoutingThresholds {
    pub default_depth: i64,
    pub min_depth: i64,
    pub max_depth: i64,
    pub direct_max_depth: i64,
    pub direct_max_candidates: i64,
    pub moderate_token_risk_ratio: f64,
    pub high_token_risk_ratio: f64,
    pub base_tokens_per_summary: i64,
    pub include_messages_token_multiplier: f64,
    pub per_depth_token_growth: f64,
    pub broad_time_range_token_multiplier: f64,
    pub multi_hop_token_multiplier: f64,
    pub multi_hop_depth_threshold: i64,
    pub multi_hop_candidate_threshold: i64,
}

pub const EXPANSION_ROUTING_THRESHOLDS: ExpansionRoutingThresholds = ExpansionRoutingThresholds {
    default_depth: 3,
    min_depth: 1,
    max_depth: 10,
    direct_max_depth: 2,
    direct_max_candidates: 1,
    moderate_token_risk_ratio: 0.35,
    high_token_risk_ratio: 0.7,
    base_tokens_per_summary: 220,
    include_messages_token_multiplier: 1.9,
    per_depth_token_growth: 0.65,
    broad_time_range_token_multiplier: 1.35,
    multi_hop_token_multiplier: 1.25,
    multi_hop_depth_threshold: 3,
    multi_hop_candidate_threshold: 5,
};

fn normalize_depth(requested_max_depth: Option<i64>) -> i64 {
    match requested_max_depth {
        Some(value) => value.clamp(
            EXPANSION_ROUTING_THRESHOLDS.min_depth,
            EXPANSION_ROUTING_THRESHOLDS.max_depth,
        ),
        None => EXPANSION_ROUTING_THRESHOLDS.default_depth,
    }
}

fn normalize_token_cap(token_cap: i64) -> i64 {
    if token_cap <= 0 { 1 } else { token_cap }
}

pub fn detect_broad_time_range_indicator(query: Option<&str>) -> bool {
    let Some(query) = query.map(str::trim).filter(|q| !q.is_empty()) else {
        return false;
    };

    let patterns = [
        Regex::new(r"(?i)\b(last|past)\s+(month|months|quarter|quarters|year|years)\b").unwrap(),
        Regex::new(r"(?i)\b(over|across|throughout)\s+(time|months|quarters|years)\b").unwrap(),
        Regex::new(r"(?i)\b(timeline|chronology|history|long[-\s]?term)\b").unwrap(),
        Regex::new(r"(?i)\bbetween\s+[^.]{0,40}\s+and\s+[^.]{0,40}\b").unwrap(),
    ];
    if patterns.iter().any(|p| p.is_match(query)) {
        return true;
    }

    let year_re = Regex::new(r"\b(?:19|20)\d{2}\b").unwrap();
    let years: Vec<i64> = year_re
        .find_iter(query)
        .filter_map(|m| m.as_str().parse::<i64>().ok())
        .collect();
    if years.len() < 2 {
        return false;
    }
    let earliest = *years.iter().min().unwrap_or(&0);
    let latest = *years.iter().max().unwrap_or(&0);
    latest - earliest >= 2
}

pub fn detect_multi_hop_indicator(
    query: Option<&str>,
    requested_max_depth: Option<i64>,
    candidate_summary_count: i64,
) -> bool {
    let normalized_depth = normalize_depth(requested_max_depth);
    let count = candidate_summary_count.max(0);
    if normalized_depth >= EXPANSION_ROUTING_THRESHOLDS.multi_hop_depth_threshold {
        return true;
    }
    if count >= EXPANSION_ROUTING_THRESHOLDS.multi_hop_candidate_threshold {
        return true;
    }
    let Some(query) = query.map(str::trim).filter(|q| !q.is_empty()) else {
        return false;
    };

    let patterns = [
        Regex::new(r"(?i)\b(root\s+cause|causal\s+chain|chain\s+of\s+events)\b").unwrap(),
        Regex::new(r"(?i)\b(multi[-\s]?hop|multi[-\s]?step|cross[-\s]?summary)\b").unwrap(),
        Regex::new(r"(?i)\bhow\s+did\b.+\blead\s+to\b").unwrap(),
    ];
    patterns.iter().any(|p| p.is_match(query))
}

pub fn estimate_expansion_tokens(
    requested_max_depth: Option<i64>,
    candidate_summary_count: i64,
    include_messages: bool,
    broad_time_range_indicator: bool,
    multi_hop_indicator: bool,
) -> i64 {
    let normalized_depth = normalize_depth(requested_max_depth);
    let count = candidate_summary_count.max(0);
    if count == 0 {
        return 0;
    }
    let include_messages_multiplier = if include_messages {
        EXPANSION_ROUTING_THRESHOLDS.include_messages_token_multiplier
    } else {
        1.0
    };
    let depth_multiplier =
        1.0 + ((normalized_depth - 1) as f64) * EXPANSION_ROUTING_THRESHOLDS.per_depth_token_growth;
    let time_multiplier = if broad_time_range_indicator {
        EXPANSION_ROUTING_THRESHOLDS.broad_time_range_token_multiplier
    } else {
        1.0
    };
    let hop_multiplier = if multi_hop_indicator {
        EXPANSION_ROUTING_THRESHOLDS.multi_hop_token_multiplier
    } else {
        1.0
    };

    let per_summary = (EXPANSION_ROUTING_THRESHOLDS.base_tokens_per_summary as f64)
        * include_messages_multiplier
        * depth_multiplier
        * time_multiplier
        * hop_multiplier;
    ((per_summary * count as f64).ceil() as i64).max(0)
}

pub fn classify_expansion_token_risk(
    estimated_tokens: i64,
    token_cap: i64,
) -> (f64, LcmExpansionTokenRiskLevel) {
    let estimated_tokens = estimated_tokens.max(0);
    let token_cap = normalize_token_cap(token_cap);
    let ratio = estimated_tokens as f64 / token_cap as f64;
    if ratio >= EXPANSION_ROUTING_THRESHOLDS.high_token_risk_ratio {
        return (ratio, LcmExpansionTokenRiskLevel::High);
    }
    if ratio >= EXPANSION_ROUTING_THRESHOLDS.moderate_token_risk_ratio {
        return (ratio, LcmExpansionTokenRiskLevel::Moderate);
    }
    (ratio, LcmExpansionTokenRiskLevel::Low)
}

pub fn decide_lcm_expansion_routing(
    input: LcmExpansionRoutingInput,
) -> LcmExpansionRoutingDecision {
    let normalized_max_depth = normalize_depth(input.requested_max_depth);
    let candidate_summary_count = input.candidate_summary_count.max(0);
    let token_cap = normalize_token_cap(input.token_cap);
    let broad_time_range = detect_broad_time_range_indicator(input.query.as_deref());
    let multi_hop_retrieval = detect_multi_hop_indicator(
        input.query.as_deref(),
        Some(normalized_max_depth),
        candidate_summary_count,
    );
    let estimated_tokens = estimate_expansion_tokens(
        Some(normalized_max_depth),
        candidate_summary_count,
        input.include_messages,
        broad_time_range,
        multi_hop_retrieval,
    );
    let (token_risk_ratio, token_risk_level) =
        classify_expansion_token_risk(estimated_tokens, token_cap);

    let direct_by_no_candidates = candidate_summary_count == 0;
    let direct_by_low_complexity_probe =
        matches!(input.intent, LcmExpansionRoutingIntent::QueryProbe)
            && !direct_by_no_candidates
            && normalized_max_depth <= EXPANSION_ROUTING_THRESHOLDS.direct_max_depth
            && candidate_summary_count <= EXPANSION_ROUTING_THRESHOLDS.direct_max_candidates
            && matches!(token_risk_level, LcmExpansionTokenRiskLevel::Low)
            && !broad_time_range
            && !multi_hop_retrieval;

    let delegate_by_depth = false;
    let delegate_by_candidate_count = false;
    let delegate_by_token_risk = matches!(token_risk_level, LcmExpansionTokenRiskLevel::High);
    let delegate_by_broad_time_range_and_multi_hop = broad_time_range && multi_hop_retrieval;

    let should_direct = direct_by_no_candidates || direct_by_low_complexity_probe;
    let should_delegate =
        !should_direct && (delegate_by_token_risk || delegate_by_broad_time_range_and_multi_hop);

    let action = if should_direct {
        LcmExpansionRoutingAction::AnswerDirectly
    } else if should_delegate {
        LcmExpansionRoutingAction::DelegateTraversal
    } else {
        LcmExpansionRoutingAction::ExpandShallow
    };

    let mut reasons = vec![];
    if direct_by_no_candidates {
        reasons.push("No candidate summary IDs are available.".to_string());
    }
    if direct_by_low_complexity_probe {
        reasons
            .push("Query probe is low complexity and below retrieval-risk thresholds.".to_string());
    }
    if delegate_by_token_risk {
        reasons.push(format!(
            "Estimated token risk ratio {:.2} meets delegate threshold {:.2}.",
            token_risk_ratio, EXPANSION_ROUTING_THRESHOLDS.high_token_risk_ratio
        ));
    }
    if delegate_by_broad_time_range_and_multi_hop {
        reasons.push(
            "Broad time-range request combined with multi-hop retrieval indicators.".to_string(),
        );
    }
    if matches!(action, LcmExpansionRoutingAction::ExpandShallow) {
        reasons.push("Complexity is bounded; use direct/shallow expansion.".to_string());
    }

    LcmExpansionRoutingDecision {
        action,
        normalized_max_depth,
        candidate_summary_count,
        estimated_tokens,
        token_cap,
        token_risk_ratio,
        token_risk_level,
        indicators: ExpansionIndicators {
            broad_time_range,
            multi_hop_retrieval,
        },
        triggers: ExpansionTriggers {
            direct_by_no_candidates,
            direct_by_low_complexity_probe,
            delegate_by_depth,
            delegate_by_candidate_count,
            delegate_by_token_risk,
            delegate_by_broad_time_range_and_multi_hop,
        },
        reasons,
    }
}
