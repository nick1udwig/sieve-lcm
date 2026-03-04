use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct LcmConfig {
    pub enabled: bool,
    pub database_path: String,
    pub context_threshold: f64,
    pub fresh_tail_count: i32,
    pub leaf_min_fanout: i32,
    pub condensed_min_fanout: i32,
    pub condensed_min_fanout_hard: i32,
    pub incremental_max_depth: i32,
    pub leaf_chunk_tokens: i32,
    pub leaf_target_tokens: i32,
    pub condensed_target_tokens: i32,
    pub max_expand_tokens: i32,
    pub large_file_token_threshold: i32,
    pub large_file_summary_provider: String,
    pub large_file_summary_model: String,
    pub autocompact_disabled: bool,
    pub timezone: String,
    pub prune_heartbeat_ok: bool,
}

fn parse_bool(value: Option<String>, default: bool) -> bool {
    match value.as_deref().map(str::trim) {
        Some("true") => true,
        Some("false") => false,
        _ => default,
    }
}

fn parse_i32(value: Option<String>, default: i32) -> i32 {
    value
        .as_deref()
        .map(str::trim)
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(default)
}

fn parse_f64(value: Option<String>, default: f64) -> f64 {
    value
        .as_deref()
        .map(str::trim)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

fn home_dir_string() -> String {
    env::var("HOME").unwrap_or_else(|_| ".".to_string())
}

fn default_db_path() -> String {
    let home = home_dir_string();
    Path::new(&home)
        .join(".openclaw")
        .join("lcm.db")
        .to_string_lossy()
        .to_string()
}

pub fn resolve_lcm_config() -> LcmConfig {
    let timezone = env::var("TZ").unwrap_or_else(|_| "UTC".to_string());
    LcmConfig {
        enabled: !matches!(
            env::var("LCM_ENABLED").ok().as_deref().map(str::trim),
            Some("false")
        ),
        database_path: env::var("LCM_DATABASE_PATH").unwrap_or_else(|_| default_db_path()),
        context_threshold: parse_f64(env::var("LCM_CONTEXT_THRESHOLD").ok(), 0.75),
        fresh_tail_count: parse_i32(env::var("LCM_FRESH_TAIL_COUNT").ok(), 32),
        leaf_min_fanout: parse_i32(env::var("LCM_LEAF_MIN_FANOUT").ok(), 8),
        condensed_min_fanout: parse_i32(env::var("LCM_CONDENSED_MIN_FANOUT").ok(), 4),
        condensed_min_fanout_hard: parse_i32(env::var("LCM_CONDENSED_MIN_FANOUT_HARD").ok(), 2),
        incremental_max_depth: parse_i32(env::var("LCM_INCREMENTAL_MAX_DEPTH").ok(), 0),
        leaf_chunk_tokens: parse_i32(env::var("LCM_LEAF_CHUNK_TOKENS").ok(), 20_000),
        leaf_target_tokens: parse_i32(env::var("LCM_LEAF_TARGET_TOKENS").ok(), 1_200),
        condensed_target_tokens: parse_i32(env::var("LCM_CONDENSED_TARGET_TOKENS").ok(), 2_000),
        max_expand_tokens: parse_i32(env::var("LCM_MAX_EXPAND_TOKENS").ok(), 4_000),
        large_file_token_threshold: parse_i32(
            env::var("LCM_LARGE_FILE_TOKEN_THRESHOLD").ok(),
            25_000,
        ),
        large_file_summary_provider: env::var("LCM_LARGE_FILE_SUMMARY_PROVIDER")
            .unwrap_or_default()
            .trim()
            .to_string(),
        large_file_summary_model: env::var("LCM_LARGE_FILE_SUMMARY_MODEL")
            .unwrap_or_default()
            .trim()
            .to_string(),
        autocompact_disabled: parse_bool(env::var("LCM_AUTOCOMPACT_DISABLED").ok(), false),
        timezone,
        prune_heartbeat_ok: parse_bool(env::var("LCM_PRUNE_HEARTBEAT_OK").ok(), false),
    }
}

impl Default for LcmConfig {
    fn default() -> Self {
        resolve_lcm_config()
    }
}

pub fn ensure_parent_dir(path: &str) -> anyhow::Result<()> {
    let path = PathBuf::from(path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
