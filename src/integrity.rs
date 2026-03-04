#[derive(Clone, Debug, PartialEq)]
pub struct IntegrityCheck {
    pub name: String,
    pub status: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IntegrityReport {
    pub conversation_id: i64,
    pub checks: Vec<IntegrityCheck>,
    pub pass_count: usize,
    pub fail_count: usize,
    pub warn_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LcmMetrics {
    pub conversation_id: i64,
    pub context_tokens: i64,
    pub message_count: i64,
    pub summary_count: i64,
    pub context_item_count: i64,
    pub leaf_summary_count: i64,
    pub condensed_summary_count: i64,
    pub large_file_count: i64,
}

pub fn repair_plan(_report: &IntegrityReport) -> Vec<String> {
    vec![]
}
