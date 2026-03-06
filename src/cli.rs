use chrono::{DateTime, Utc};
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde::Serialize;

use crate::db::connection::get_lcm_connection;
use crate::db::migration::run_lcm_migrations;
use crate::store::conversation_store::{
    ConversationStore, CreateMessageInput, MessageRole, MessageSearchInput,
};
use crate::store::summary_store::{SummarySearchInput, SummaryStore};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliCommand {
    Ingest(IngestArgs),
    Query(QueryArgs),
    Expand(ExpandArgs),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestArgs {
    pub db_path: String,
    pub conversation: String,
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryArgs {
    pub trusted_db_path: Option<String>,
    pub untrusted_db_path: Option<String>,
    pub conversation: String,
    pub query: String,
    pub limit: i64,
    pub lane: QueryLane,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryLane {
    Trusted,
    Untrusted,
    Both,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandArgs {
    pub untrusted_db_path: String,
    pub conversation: String,
    pub reference: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    Usage(UsageError),
    Runtime(String),
}

#[derive(Debug, Serialize)]
struct CliErrorJson {
    ok: bool,
    error: CliErrorBody,
}

#[derive(Debug, Serialize)]
struct CliErrorBody {
    code: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<CliUsage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError {
    pub message: String,
    pub usage: Option<CliUsage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CliUsage {
    pub command: String,
    pub syntax: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub commands: Vec<CliUsageCommand>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub options: Vec<CliUsageOption>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CliUsageCommand {
    pub name: String,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CliUsageOption {
    pub flags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_name: Option<String>,
    pub description: String,
    pub required: bool,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub choices: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct IngestOutput {
    pub ok: bool,
    pub conversation: String,
}

#[derive(Debug, Serialize)]
pub struct QueryOutput {
    pub conversation: String,
    pub query: String,
    pub trusted_hits: Vec<TrustedHit>,
    pub untrusted_refs: Vec<UntrustedRef>,
    pub stats: QueryStats,
}

#[derive(Debug, Serialize)]
pub struct TrustedHit {
    pub id: String,
    pub source: String,
    pub score: Option<f64>,
    pub created_at: String,
    pub excerpt: String,
}

#[derive(Debug, Serialize)]
pub struct UntrustedRef {
    #[serde(rename = "ref")]
    pub opaque_ref: String,
    pub source: String,
    pub score: Option<f64>,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct QueryStats {
    pub trusted_count: usize,
    pub untrusted_count: usize,
    pub trusted_chars: usize,
    pub limit: i64,
}

#[derive(Debug, Serialize)]
pub struct ExpandOutput {
    pub conversation: String,
    #[serde(rename = "ref")]
    pub reference: String,
    pub content: String,
    pub meta: ExpandMeta,
}

#[derive(Debug, Serialize)]
pub struct ExpandMeta {
    pub source: String,
    pub id: String,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum CliSuccess {
    Ingest(IngestOutput),
    Query(QueryOutput),
    Expand(ExpandOutput),
}

pub fn parse_command(raw_args: &[String]) -> Result<CliCommand, CliError> {
    if let Some(error) = help_usage_error(raw_args) {
        return Err(CliError::Usage(error));
    }

    let Some(command) = raw_args.first().map(|value| value.trim()) else {
        return Err(usage_error_with_usage(
            "missing command (expected: ingest|query|expand)",
            root_usage_spec(),
        ));
    };

    if !matches!(command, "ingest" | "query" | "expand") {
        return Err(usage_error_with_usage(
            format!("unknown command `{command}` (expected: ingest|query|expand)"),
            root_usage_spec(),
        ));
    }

    let usage = usage_spec_for_command(command);
    prevalidate_raw_args(&raw_args[1..], usage.as_ref())?;

    let matches = build_cli()
        .try_get_matches_from(raw_args)
        .map_err(|err| map_clap_usage_error(err, usage.as_ref()))?;

    match matches.subcommand() {
        Some(("ingest", args)) => parse_ingest(args),
        Some(("query", args)) => parse_query(args),
        Some(("expand", args)) => parse_expand(args),
        Some((other, _)) => Err(usage_error_with_usage(
            format!("unknown command `{other}` (expected: ingest|query|expand)"),
            root_usage_spec(),
        )),
        None => Err(usage_error_with_usage(
            "missing command (expected: ingest|query|expand)",
            root_usage_spec(),
        )),
    }
}

pub fn execute_command(command: CliCommand) -> Result<CliSuccess, CliError> {
    match command {
        CliCommand::Ingest(args) => ingest(args).map(CliSuccess::Ingest),
        CliCommand::Query(args) => query(args).map(CliSuccess::Query),
        CliCommand::Expand(args) => expand(args).map(CliSuccess::Expand),
    }
}

pub fn serialize_success_json(output: &CliSuccess) -> Result<String, CliError> {
    serde_json::to_string(output)
        .map_err(|err| CliError::Runtime(format!("encode success json failed: {err}")))
}

pub fn serialize_error_json(error: &CliError) -> String {
    let (code, message, usage) = match error {
        CliError::Usage(error) => (
            "usage_error".to_string(),
            error.message.clone(),
            error.usage.clone(),
        ),
        CliError::Runtime(message) => {
            let lowered = message.to_ascii_lowercase();
            let code = if lowered.contains("invalid reference") {
                "invalid_ref"
            } else {
                "runtime_error"
            };
            (code.to_string(), message.clone(), None)
        }
    };

    serde_json::to_string(&CliErrorJson {
        ok: false,
        error: CliErrorBody {
            code,
            message,
            usage,
        },
    })
    .unwrap_or_else(|_| {
        "{\"ok\":false,\"error\":{\"code\":\"runtime_error\",\"message\":\"failed to encode error\"}}"
            .to_string()
    })
}

fn parse_ingest(args: &ArgMatches) -> Result<CliCommand, CliError> {
    let usage = ingest_usage_spec();
    let db_path = required_arg(args, "db", &usage)?;
    let conversation = optional_arg(args, "conversation").unwrap_or_else(|| "global".to_string());
    let role = required_arg(args, "role", &usage)?;
    let content = required_arg(args, "content", &usage)?;

    Ok(CliCommand::Ingest(IngestArgs {
        db_path,
        conversation,
        role,
        content,
    }))
}

fn parse_query(args: &ArgMatches) -> Result<CliCommand, CliError> {
    let usage = query_usage_spec();
    let trusted_db_path = optional_arg(args, "trusted-db");
    let untrusted_db_path = optional_arg(args, "untrusted-db");
    let conversation = optional_arg(args, "conversation")
        .or_else(default_global_conversation)
        .unwrap_or_else(|| "global".to_string());
    let query = required_arg(args, "query", &usage)?;
    let limit = optional_arg(args, "limit")
        .map(|raw| raw.parse::<i64>())
        .transpose()
        .map_err(|err| {
            usage_error_with_usage(format!("invalid --limit value: {err}"), usage.clone())
        })?
        .unwrap_or(5)
        .clamp(1, 20);

    let lane = match optional_arg(args, "lane")
        .unwrap_or_else(|| "both".to_string())
        .as_str()
    {
        "trusted" => QueryLane::Trusted,
        "untrusted" => QueryLane::Untrusted,
        "both" => QueryLane::Both,
        other => {
            return Err(usage_error_with_usage(
                format!("invalid --lane `{other}` (expected trusted|untrusted|both)"),
                usage,
            ));
        }
    };

    Ok(CliCommand::Query(QueryArgs {
        trusted_db_path: trusted_db_path.or_else(default_trusted_db_path),
        untrusted_db_path: untrusted_db_path.or_else(default_untrusted_db_path),
        conversation,
        query,
        limit,
        lane,
    }))
}

fn parse_expand(args: &ArgMatches) -> Result<CliCommand, CliError> {
    let usage = expand_usage_spec();
    let untrusted_db_path = optional_arg(args, "untrusted-db")
        .or_else(default_untrusted_db_path)
        .ok_or_else(|| {
            usage_error_with_usage("missing required flag --untrusted-db", usage.clone())
        })?;
    let conversation = optional_arg(args, "conversation")
        .or_else(default_global_conversation)
        .unwrap_or_else(|| "global".to_string());
    let reference = required_arg(args, "ref", &usage)?;

    Ok(CliCommand::Expand(ExpandArgs {
        untrusted_db_path,
        conversation,
        reference,
    }))
}

fn build_cli() -> Command {
    Command::new("sieve-lcm-cli")
        .no_binary_name(true)
        .disable_help_flag(true)
        .disable_version_flag(true)
        .subcommand_required(true)
        .args_override_self(true)
        .arg(
            Arg::new("json")
                .long("json")
                .global(true)
                .hide(true)
                .action(ArgAction::SetTrue),
        )
        .subcommand(build_ingest_command())
        .subcommand(build_query_command())
        .subcommand(build_expand_command())
}

fn prevalidate_raw_args(args: &[String], usage: Option<&CliUsage>) -> Result<(), CliError> {
    let mut idx = 0usize;
    while idx < args.len() {
        let Some(raw) = args.get(idx) else {
            break;
        };
        if raw == "--json" {
            idx += 1;
            continue;
        }
        if !raw.starts_with("--") {
            return Err(usage_error_with_optional_usage(
                format!("unexpected positional argument `{raw}`"),
                usage,
            ));
        }
        let Some(value) = args.get(idx + 1) else {
            return Err(usage_error_with_optional_usage(
                format!("flag `{raw}` requires a value"),
                usage,
            ));
        };
        if value.starts_with("--") {
            return Err(usage_error_with_optional_usage(
                format!("flag `{raw}` requires a value"),
                usage,
            ));
        }
        idx += 2;
    }
    Ok(())
}

fn build_ingest_command() -> Command {
    Command::new("ingest")
        .args_override_self(true)
        .arg(string_arg("db").long("db").value_name("PATH"))
        .arg(
            string_arg("conversation")
                .long("conversation")
                .value_name("ID"),
        )
        .arg(string_arg("role").long("role").value_name("ROLE"))
        .arg(string_arg("content").long("content").value_name("TEXT"))
}

fn build_query_command() -> Command {
    Command::new("query")
        .args_override_self(true)
        .arg(
            string_arg("trusted-db")
                .long("trusted-db")
                .value_name("PATH"),
        )
        .arg(
            string_arg("untrusted-db")
                .long("untrusted-db")
                .value_name("PATH"),
        )
        .arg(
            string_arg("conversation")
                .long("conversation")
                .value_name("ID"),
        )
        .arg(string_arg("query").long("query").value_name("TEXT"))
        .arg(string_arg("limit").long("limit").value_name("N"))
        .arg(string_arg("lane").long("lane").value_name("LANE"))
}

fn build_expand_command() -> Command {
    Command::new("expand")
        .args_override_self(true)
        .arg(
            string_arg("untrusted-db")
                .long("untrusted-db")
                .value_name("PATH"),
        )
        .arg(
            string_arg("conversation")
                .long("conversation")
                .value_name("ID"),
        )
        .arg(string_arg("ref").long("ref").value_name("REF"))
}

fn usage_spec_for_command(name: &str) -> Option<CliUsage> {
    match name {
        "ingest" => Some(ingest_usage_spec()),
        "query" => Some(query_usage_spec()),
        "expand" => Some(expand_usage_spec()),
        _ => None,
    }
}

fn root_usage_spec() -> CliUsage {
    CliUsage {
        command: "sieve-lcm-cli".to_string(),
        syntax: "sieve-lcm-cli [OPTIONS] <COMMAND>".to_string(),
        summary: Some("Tool-driven memory access for sieve-lcm.".to_string()),
        commands: vec![
            CliUsageCommand {
                name: "ingest".to_string(),
                summary: "Append a message to a lane database.".to_string(),
            },
            CliUsageCommand {
                name: "query".to_string(),
                summary: "Retrieve trusted excerpts plus untrusted opaque refs.".to_string(),
            },
            CliUsageCommand {
                name: "expand".to_string(),
                summary: "Resolve an untrusted opaque ref.".to_string(),
            },
        ],
        options: common_usage_options(),
    }
}

fn ingest_usage_spec() -> CliUsage {
    let mut options = vec![
        CliUsageOption {
            flags: vec!["--db".to_string()],
            value_name: Some("PATH".to_string()),
            description: "Lane database path.".to_string(),
            required: true,
            choices: Vec::new(),
        },
        CliUsageOption {
            flags: vec!["--conversation".to_string()],
            value_name: Some("ID".to_string()),
            description: "Conversation id. Defaults to SIEVE_LCM_GLOBAL_SESSION_ID or `global`."
                .to_string(),
            required: false,
            choices: Vec::new(),
        },
        CliUsageOption {
            flags: vec!["--role".to_string()],
            value_name: Some("ROLE".to_string()),
            description: "Message role.".to_string(),
            required: true,
            choices: vec![
                "user".to_string(),
                "assistant".to_string(),
                "system".to_string(),
                "tool".to_string(),
            ],
        },
        CliUsageOption {
            flags: vec!["--content".to_string()],
            value_name: Some("TEXT".to_string()),
            description: "Message content to ingest.".to_string(),
            required: true,
            choices: Vec::new(),
        },
    ];
    options.extend(common_usage_options());

    CliUsage {
        command: "ingest".to_string(),
        syntax: "sieve-lcm-cli ingest [OPTIONS]".to_string(),
        summary: Some("Append a message to a lane database.".to_string()),
        commands: Vec::new(),
        options,
    }
}

fn query_usage_spec() -> CliUsage {
    let mut options = vec![
        CliUsageOption {
            flags: vec!["--trusted-db".to_string()],
            value_name: Some("PATH".to_string()),
            description:
                "Trusted lane database path. Defaults to SIEVE_LCM_TRUSTED_DB_PATH or ~/.sieve/lcm/trusted.db."
                    .to_string(),
            required: false,
            choices: Vec::new(),
        },
        CliUsageOption {
            flags: vec!["--untrusted-db".to_string()],
            value_name: Some("PATH".to_string()),
            description:
                "Untrusted lane database path. Defaults to SIEVE_LCM_UNTRUSTED_DB_PATH or ~/.sieve/lcm/untrusted.db."
                    .to_string(),
            required: false,
            choices: Vec::new(),
        },
        CliUsageOption {
            flags: vec!["--conversation".to_string()],
            value_name: Some("ID".to_string()),
            description:
                "Conversation id. Defaults to SIEVE_LCM_GLOBAL_SESSION_ID or `global`."
                    .to_string(),
            required: false,
            choices: Vec::new(),
        },
        CliUsageOption {
            flags: vec!["--query".to_string()],
            value_name: Some("TEXT".to_string()),
            description: "Search text to match against stored records.".to_string(),
            required: true,
            choices: Vec::new(),
        },
        CliUsageOption {
            flags: vec!["--limit".to_string()],
            value_name: Some("N".to_string()),
            description: "Maximum hits to return; clamped to 1..20. Defaults to 5."
                .to_string(),
            required: false,
            choices: Vec::new(),
        },
        CliUsageOption {
            flags: vec!["--lane".to_string()],
            value_name: Some("LANE".to_string()),
            description: "Which lane to search. Defaults to `both`.".to_string(),
            required: false,
            choices: vec![
                "trusted".to_string(),
                "untrusted".to_string(),
                "both".to_string(),
            ],
        },
    ];
    options.extend(common_usage_options());

    CliUsage {
        command: "query".to_string(),
        syntax: "sieve-lcm-cli query [OPTIONS]".to_string(),
        summary: Some("Retrieve trusted excerpts plus untrusted opaque refs.".to_string()),
        commands: Vec::new(),
        options,
    }
}

fn expand_usage_spec() -> CliUsage {
    let mut options = vec![
        CliUsageOption {
            flags: vec!["--untrusted-db".to_string()],
            value_name: Some("PATH".to_string()),
            description:
                "Untrusted lane database path. Defaults to SIEVE_LCM_UNTRUSTED_DB_PATH or ~/.sieve/lcm/untrusted.db."
                    .to_string(),
            required: false,
            choices: Vec::new(),
        },
        CliUsageOption {
            flags: vec!["--conversation".to_string()],
            value_name: Some("ID".to_string()),
            description:
                "Conversation id. Defaults to SIEVE_LCM_GLOBAL_SESSION_ID or `global`."
                    .to_string(),
            required: false,
            choices: Vec::new(),
        },
        CliUsageOption {
            flags: vec!["--ref".to_string()],
            value_name: Some("REF".to_string()),
            description: "Opaque untrusted reference returned by `query`.".to_string(),
            required: true,
            choices: Vec::new(),
        },
    ];
    options.extend(common_usage_options());

    CliUsage {
        command: "expand".to_string(),
        syntax: "sieve-lcm-cli expand [OPTIONS]".to_string(),
        summary: Some("Resolve an untrusted opaque ref.".to_string()),
        commands: Vec::new(),
        options,
    }
}

fn common_usage_options() -> Vec<CliUsageOption> {
    vec![
        CliUsageOption {
            flags: vec!["--json".to_string()],
            value_name: None,
            description: "Accepted for compatibility; output is JSON regardless.".to_string(),
            required: false,
            choices: Vec::new(),
        },
        CliUsageOption {
            flags: vec!["-h".to_string(), "--help".to_string()],
            value_name: None,
            description: "Show usage in JSON form.".to_string(),
            required: false,
            choices: Vec::new(),
        },
    ]
}

fn help_usage_error(raw_args: &[String]) -> Option<UsageError> {
    let first = raw_args.first().map(|value| value.trim())?;
    if is_help_flag(first) {
        return Some(UsageError {
            message: "help requested".to_string(),
            usage: Some(root_usage_spec()),
        });
    }

    if matches!(first, "ingest" | "query" | "expand")
        && raw_args
            .iter()
            .skip(1)
            .any(|value| is_help_flag(value.trim()))
    {
        return Some(UsageError {
            message: "help requested".to_string(),
            usage: usage_spec_for_command(first),
        });
    }

    None
}

fn is_help_flag(value: &str) -> bool {
    matches!(value, "-h" | "--help")
}

fn usage_error(message: impl Into<String>) -> CliError {
    CliError::Usage(UsageError {
        message: message.into(),
        usage: None,
    })
}

fn usage_error_with_usage(message: impl Into<String>, usage: CliUsage) -> CliError {
    CliError::Usage(UsageError {
        message: message.into(),
        usage: Some(usage),
    })
}

fn usage_error_with_optional_usage(
    message: impl Into<String>,
    usage: Option<&CliUsage>,
) -> CliError {
    match usage {
        Some(usage) => usage_error_with_usage(message, usage.clone()),
        None => usage_error(message),
    }
}

fn string_arg(name: &'static str) -> Arg {
    Arg::new(name)
        .action(ArgAction::Set)
        .allow_hyphen_values(true)
}

fn map_clap_usage_error(error: clap::Error, usage: Option<&CliUsage>) -> CliError {
    let rendered = error.to_string();
    let compact = rendered
        .split("\n\n")
        .next()
        .unwrap_or(rendered.as_str())
        .trim()
        .trim_start_matches("error: ")
        .trim()
        .to_string();
    usage_error_with_optional_usage(compact, usage)
}

fn required_arg(matches: &ArgMatches, name: &str, usage: &CliUsage) -> Result<String, CliError> {
    matches
        .get_one::<String>(name)
        .cloned()
        .map(|value| value.trim().to_string())
        .ok_or_else(|| {
            usage_error_with_usage(format!("missing required flag --{name}"), usage.clone())
        })
        .and_then(|value| {
            if value.is_empty() {
                Err(usage_error_with_usage(
                    format!("missing required flag --{name}"),
                    usage.clone(),
                ))
            } else {
                Ok(value)
            }
        })
}

fn optional_arg(matches: &ArgMatches, name: &str) -> Option<String> {
    matches
        .get_one::<String>(name)
        .cloned()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn ingest(args: IngestArgs) -> Result<IngestOutput, CliError> {
    let lane = open_lane(&args.db_path)?;
    let role = parse_role(&args.role, &ingest_usage_spec())?;

    let conversation = lane
        .conversation_store
        .get_or_create_conversation(&args.conversation, None)
        .map_err(|err| CliError::Runtime(format!("resolve conversation failed: {err}")))?;

    let seq = lane
        .conversation_store
        .get_max_seq(conversation.conversation_id)
        .map_err(|err| CliError::Runtime(format!("read sequence failed: {err}")))?
        + 1;

    let message = lane
        .conversation_store
        .create_message(CreateMessageInput {
            conversation_id: conversation.conversation_id,
            seq,
            role,
            content: args.content.clone(),
            token_count: estimate_tokens(&args.content),
        })
        .map_err(|err| CliError::Runtime(format!("insert message failed: {err}")))?;

    lane.summary_store
        .append_context_message(conversation.conversation_id, message.message_id)
        .map_err(|err| CliError::Runtime(format!("append context failed: {err}")))?;

    Ok(IngestOutput {
        ok: true,
        conversation: args.conversation,
    })
}

fn query(args: QueryArgs) -> Result<QueryOutput, CliError> {
    let mut trusted_hits = Vec::new();
    let mut untrusted_refs = Vec::new();

    if matches!(args.lane, QueryLane::Trusted | QueryLane::Both) {
        let db_path = args
            .trusted_db_path
            .as_deref()
            .ok_or_else(|| usage_error_with_usage("missing --trusted-db", query_usage_spec()))?;
        let lane = open_lane(db_path)?;
        if let Some(conversation) = lane
            .conversation_store
            .get_conversation_by_session_id(&args.conversation)
            .map_err(|err| {
                CliError::Runtime(format!("trusted conversation lookup failed: {err}"))
            })?
        {
            let messages = lane
                .conversation_store
                .search_messages(MessageSearchInput {
                    conversation_id: Some(conversation.conversation_id),
                    query: args.query.clone(),
                    mode: "full_text".to_string(),
                    since: None,
                    before: None,
                    limit: Some(args.limit),
                })
                .map_err(|err| {
                    CliError::Runtime(format!("trusted message search failed: {err}"))
                })?;

            let summaries = lane
                .summary_store
                .search_summaries(SummarySearchInput {
                    conversation_id: Some(conversation.conversation_id),
                    query: args.query.clone(),
                    mode: "full_text".to_string(),
                    since: None,
                    before: None,
                    limit: Some(args.limit),
                })
                .map_err(|err| {
                    CliError::Runtime(format!("trusted summary search failed: {err}"))
                })?;

            trusted_hits.extend(messages.into_iter().map(|hit| TrustedHit {
                id: format!("trusted:message:{}", hit.message_id),
                source: "message".to_string(),
                score: hit.rank,
                created_at: format_time(hit.created_at),
                excerpt: hit.snippet,
            }));
            trusted_hits.extend(summaries.into_iter().map(|hit| TrustedHit {
                id: format!("trusted:summary:{}", hit.summary_id),
                source: "summary".to_string(),
                score: hit.rank,
                created_at: format_time(hit.created_at),
                excerpt: hit.snippet,
            }));

            trusted_hits.sort_by(|left, right| right.created_at.cmp(&left.created_at));
            trusted_hits.truncate(args.limit as usize);
        }
    }

    if matches!(args.lane, QueryLane::Untrusted | QueryLane::Both) {
        let db_path = args
            .untrusted_db_path
            .as_deref()
            .ok_or_else(|| usage_error_with_usage("missing --untrusted-db", query_usage_spec()))?;
        let lane = open_lane(db_path)?;
        if let Some(conversation) = lane
            .conversation_store
            .get_conversation_by_session_id(&args.conversation)
            .map_err(|err| {
                CliError::Runtime(format!("untrusted conversation lookup failed: {err}"))
            })?
        {
            let messages = lane
                .conversation_store
                .search_messages(MessageSearchInput {
                    conversation_id: Some(conversation.conversation_id),
                    query: args.query.clone(),
                    mode: "full_text".to_string(),
                    since: None,
                    before: None,
                    limit: Some(args.limit),
                })
                .map_err(|err| {
                    CliError::Runtime(format!("untrusted message search failed: {err}"))
                })?;

            let summaries = lane
                .summary_store
                .search_summaries(SummarySearchInput {
                    conversation_id: Some(conversation.conversation_id),
                    query: args.query.clone(),
                    mode: "full_text".to_string(),
                    since: None,
                    before: None,
                    limit: Some(args.limit),
                })
                .map_err(|err| {
                    CliError::Runtime(format!("untrusted summary search failed: {err}"))
                })?;

            untrusted_refs.extend(messages.into_iter().map(|hit| UntrustedRef {
                opaque_ref: format!("lcm:untrusted:message:{}", hit.message_id),
                source: "message".to_string(),
                score: hit.rank,
                created_at: format_time(hit.created_at),
            }));
            untrusted_refs.extend(summaries.into_iter().map(|hit| UntrustedRef {
                opaque_ref: format!("lcm:untrusted:summary:{}", hit.summary_id),
                source: "summary".to_string(),
                score: hit.rank,
                created_at: format_time(hit.created_at),
            }));

            untrusted_refs.sort_by(|left, right| right.created_at.cmp(&left.created_at));
            untrusted_refs.truncate(args.limit as usize);
        }
    }

    let trusted_chars = trusted_hits
        .iter()
        .map(|hit| hit.excerpt.chars().count())
        .sum::<usize>();

    let trusted_count = trusted_hits.len();
    let untrusted_count = untrusted_refs.len();

    Ok(QueryOutput {
        conversation: args.conversation,
        query: args.query,
        trusted_hits,
        untrusted_refs,
        stats: QueryStats {
            trusted_count,
            untrusted_count,
            trusted_chars,
            limit: args.limit,
        },
    })
}

fn expand(args: ExpandArgs) -> Result<ExpandOutput, CliError> {
    let lane = open_lane(&args.untrusted_db_path)?;
    let conversation = lane
        .conversation_store
        .get_conversation_by_session_id(&args.conversation)
        .map_err(|err| CliError::Runtime(format!("conversation lookup failed: {err}")))?
        .ok_or_else(|| {
            CliError::Runtime("invalid reference: conversation not found".to_string())
        })?;

    let parsed = parse_untrusted_ref(&args.reference)?;
    match parsed {
        ParsedRef::Message(message_id) => {
            let message = lane
                .conversation_store
                .get_message_by_id(message_id)
                .map_err(|err| CliError::Runtime(format!("read message failed: {err}")))?
                .ok_or_else(|| {
                    CliError::Runtime("invalid reference: untrusted message not found".to_string())
                })?;
            if message.conversation_id != conversation.conversation_id {
                return Err(CliError::Runtime(
                    "invalid reference: message belongs to a different conversation".to_string(),
                ));
            }
            Ok(ExpandOutput {
                conversation: args.conversation,
                reference: args.reference,
                content: message.content,
                meta: ExpandMeta {
                    source: "message".to_string(),
                    id: message.message_id.to_string(),
                },
            })
        }
        ParsedRef::Summary(summary_id) => {
            let summary = lane
                .summary_store
                .get_summary(&summary_id)
                .map_err(|err| CliError::Runtime(format!("read summary failed: {err}")))?
                .ok_or_else(|| {
                    CliError::Runtime("invalid reference: untrusted summary not found".to_string())
                })?;
            if summary.conversation_id != conversation.conversation_id {
                return Err(CliError::Runtime(
                    "invalid reference: summary belongs to a different conversation".to_string(),
                ));
            }
            Ok(ExpandOutput {
                conversation: args.conversation,
                reference: args.reference,
                content: summary.content,
                meta: ExpandMeta {
                    source: "summary".to_string(),
                    id: summary.summary_id,
                },
            })
        }
    }
}

enum ParsedRef {
    Message(i64),
    Summary(String),
}

fn parse_untrusted_ref(reference: &str) -> Result<ParsedRef, CliError> {
    let trimmed = reference.trim();
    let Some(rest) = trimmed.strip_prefix("lcm:untrusted:") else {
        return Err(CliError::Runtime(
            "invalid reference: expected lcm:untrusted:*".to_string(),
        ));
    };

    if let Some(message) = rest.strip_prefix("message:") {
        let id = message.parse::<i64>().map_err(|err| {
            CliError::Runtime(format!(
                "invalid reference: message id is not an integer: {err}"
            ))
        })?;
        return Ok(ParsedRef::Message(id));
    }

    if let Some(summary) = rest.strip_prefix("summary:") {
        if summary.trim().is_empty() {
            return Err(CliError::Runtime(
                "invalid reference: empty summary id".to_string(),
            ));
        }
        return Ok(ParsedRef::Summary(summary.to_string()));
    }

    if rest.trim().is_empty() {
        return Err(CliError::Runtime(
            "invalid reference: missing identifier".to_string(),
        ));
    }

    Ok(ParsedRef::Summary(rest.to_string()))
}

fn parse_role(raw: &str, usage: &CliUsage) -> Result<MessageRole, CliError> {
    match raw.trim() {
        "user" => Ok(MessageRole::User),
        "assistant" => Ok(MessageRole::Assistant),
        "system" => Ok(MessageRole::System),
        "tool" => Ok(MessageRole::Tool),
        other => Err(usage_error_with_usage(
            format!("invalid --role `{other}` (expected user|assistant|system|tool)"),
            usage.clone(),
        )),
    }
}

fn default_global_conversation() -> Option<String> {
    std::env::var("SIEVE_LCM_GLOBAL_SESSION_ID")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn default_trusted_db_path() -> Option<String> {
    std::env::var("SIEVE_LCM_TRUSTED_DB_PATH")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|home| format!("{home}/.sieve/lcm/trusted.db"))
        })
}

fn default_untrusted_db_path() -> Option<String> {
    std::env::var("SIEVE_LCM_UNTRUSTED_DB_PATH")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|home| format!("{home}/.sieve/lcm/untrusted.db"))
        })
}

fn estimate_tokens(content: &str) -> i64 {
    ((content.len() as f64) / 4.0).ceil() as i64
}

fn format_time(value: DateTime<Utc>) -> String {
    value.to_rfc3339()
}

struct OpenLane {
    conversation_store: ConversationStore,
    summary_store: SummaryStore,
}

fn open_lane(path: &str) -> Result<OpenLane, CliError> {
    let shared = get_lcm_connection(path)
        .map_err(|err| CliError::Runtime(format!("open db failed: {err}")))?;

    {
        let guard = shared.conn.lock();
        run_lcm_migrations(&guard)
            .map_err(|err| CliError::Runtime(format!("run migrations failed: {err}")))?;
    }

    Ok(OpenLane {
        conversation_store: ConversationStore::new(&shared),
        summary_store: SummaryStore::new(&shared),
    })
}
