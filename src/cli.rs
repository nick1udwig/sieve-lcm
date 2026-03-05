use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
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
    Usage(String),
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
    let Some(command) = raw_args.first().map(|value| value.trim()) else {
        return Err(CliError::Usage(
            "missing command (expected: ingest|query|expand)".to_string(),
        ));
    };

    match command {
        "ingest" => parse_ingest(&raw_args[1..]),
        "query" => parse_query(&raw_args[1..]),
        "expand" => parse_expand(&raw_args[1..]),
        other => Err(CliError::Usage(format!(
            "unknown command `{other}` (expected: ingest|query|expand)"
        ))),
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
    let (code, message) = match error {
        CliError::Usage(message) => ("usage_error".to_string(), message.clone()),
        CliError::Runtime(message) => {
            let lowered = message.to_ascii_lowercase();
            let code = if lowered.contains("invalid reference") {
                "invalid_ref"
            } else {
                "runtime_error"
            };
            (code.to_string(), message.clone())
        }
    };

    serde_json::to_string(&CliErrorJson {
        ok: false,
        error: CliErrorBody { code, message },
    })
    .unwrap_or_else(|_| {
        "{\"ok\":false,\"error\":{\"code\":\"runtime_error\",\"message\":\"failed to encode error\"}}"
            .to_string()
    })
}

fn parse_ingest(args: &[String]) -> Result<CliCommand, CliError> {
    let mut flags = parse_flags(args)?;
    let db_path = required_flag(&mut flags, "db")?;
    let conversation = optional_flag(&mut flags, "conversation").unwrap_or_else(|| "global".to_string());
    let role = required_flag(&mut flags, "role")?;
    let content = required_flag(&mut flags, "content")?;

    if !flags.is_empty() {
        return Err(CliError::Usage(format!(
            "unknown ingest flags: {}",
            flags.keys().cloned().collect::<Vec<_>>().join(", ")
        )));
    }

    Ok(CliCommand::Ingest(IngestArgs {
        db_path,
        conversation,
        role,
        content,
    }))
}

fn parse_query(args: &[String]) -> Result<CliCommand, CliError> {
    let mut flags = parse_flags(args)?;
    let trusted_db_path = optional_flag(&mut flags, "trusted-db");
    let untrusted_db_path = optional_flag(&mut flags, "untrusted-db");
    let conversation = optional_flag(&mut flags, "conversation").unwrap_or_else(|| "global".to_string());
    let query = required_flag(&mut flags, "query")?;
    let limit = optional_flag(&mut flags, "limit")
        .map(|raw| raw.parse::<i64>())
        .transpose()
        .map_err(|err| CliError::Usage(format!("invalid --limit value: {err}")))?
        .unwrap_or(5)
        .clamp(1, 20);

    let lane = match optional_flag(&mut flags, "lane")
        .unwrap_or_else(|| "both".to_string())
        .as_str()
    {
        "trusted" => QueryLane::Trusted,
        "untrusted" => QueryLane::Untrusted,
        "both" => QueryLane::Both,
        other => {
            return Err(CliError::Usage(format!(
                "invalid --lane `{other}` (expected trusted|untrusted|both)"
            )));
        }
    };

    if !flags.is_empty() {
        return Err(CliError::Usage(format!(
            "unknown query flags: {}",
            flags.keys().cloned().collect::<Vec<_>>().join(", ")
        )));
    }

    match lane {
        QueryLane::Trusted => {
            if trusted_db_path.is_none() {
                return Err(CliError::Usage(
                    "query --lane trusted requires --trusted-db".to_string(),
                ));
            }
        }
        QueryLane::Untrusted => {
            if untrusted_db_path.is_none() {
                return Err(CliError::Usage(
                    "query --lane untrusted requires --untrusted-db".to_string(),
                ));
            }
        }
        QueryLane::Both => {
            if trusted_db_path.is_none() || untrusted_db_path.is_none() {
                return Err(CliError::Usage(
                    "query --lane both requires --trusted-db and --untrusted-db".to_string(),
                ));
            }
        }
    }

    Ok(CliCommand::Query(QueryArgs {
        trusted_db_path,
        untrusted_db_path,
        conversation,
        query,
        limit,
        lane,
    }))
}

fn parse_expand(args: &[String]) -> Result<CliCommand, CliError> {
    let mut flags = parse_flags(args)?;
    let untrusted_db_path = required_flag(&mut flags, "untrusted-db")?;
    let conversation = optional_flag(&mut flags, "conversation").unwrap_or_else(|| "global".to_string());
    let reference = required_flag(&mut flags, "ref")?;

    if !flags.is_empty() {
        return Err(CliError::Usage(format!(
            "unknown expand flags: {}",
            flags.keys().cloned().collect::<Vec<_>>().join(", ")
        )));
    }

    Ok(CliCommand::Expand(ExpandArgs {
        untrusted_db_path,
        conversation,
        reference,
    }))
}

fn parse_flags(args: &[String]) -> Result<BTreeMap<String, String>, CliError> {
    let mut out = BTreeMap::new();
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
            return Err(CliError::Usage(format!(
                "unexpected positional argument `{raw}`"
            )));
        }
        let name = raw.trim_start_matches("--").to_string();
        let Some(value) = args.get(idx + 1) else {
            return Err(CliError::Usage(format!(
                "flag `--{name}` requires a value"
            )));
        };
        if value.starts_with("--") {
            return Err(CliError::Usage(format!(
                "flag `--{name}` requires a value"
            )));
        }
        out.insert(name, value.clone());
        idx += 2;
    }
    Ok(out)
}

fn required_flag(flags: &mut BTreeMap<String, String>, name: &str) -> Result<String, CliError> {
    flags
        .remove(name)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| CliError::Usage(format!("missing required flag --{name}")))
}

fn optional_flag(flags: &mut BTreeMap<String, String>, name: &str) -> Option<String> {
    flags
        .remove(name)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn ingest(args: IngestArgs) -> Result<IngestOutput, CliError> {
    let lane = open_lane(&args.db_path)?;
    let role = parse_role(&args.role)?;

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
            .ok_or_else(|| CliError::Usage("missing --trusted-db".to_string()))?;
        let lane = open_lane(db_path)?;
        if let Some(conversation) = lane
            .conversation_store
            .get_conversation_by_session_id(&args.conversation)
            .map_err(|err| CliError::Runtime(format!("trusted conversation lookup failed: {err}")))?
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
                .map_err(|err| CliError::Runtime(format!("trusted message search failed: {err}")))?;

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
                .map_err(|err| CliError::Runtime(format!("trusted summary search failed: {err}")))?;

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
            .ok_or_else(|| CliError::Usage("missing --untrusted-db".to_string()))?;
        let lane = open_lane(db_path)?;
        if let Some(conversation) = lane
            .conversation_store
            .get_conversation_by_session_id(&args.conversation)
            .map_err(|err| CliError::Runtime(format!("untrusted conversation lookup failed: {err}")))?
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
                .map_err(|err| CliError::Runtime(format!("untrusted message search failed: {err}")))?;

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
                .map_err(|err| CliError::Runtime(format!("untrusted summary search failed: {err}")))?;

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
        .ok_or_else(|| CliError::Runtime("invalid reference: conversation not found".to_string()))?;

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
            CliError::Runtime(format!("invalid reference: message id is not an integer: {err}"))
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

fn parse_role(raw: &str) -> Result<MessageRole, CliError> {
    match raw.trim() {
        "user" => Ok(MessageRole::User),
        "assistant" => Ok(MessageRole::Assistant),
        "system" => Ok(MessageRole::System),
        "tool" => Ok(MessageRole::Tool),
        other => Err(CliError::Usage(format!(
            "invalid --role `{other}` (expected user|assistant|system|tool)"
        ))),
    }
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
    let shared =
        get_lcm_connection(path).map_err(|err| CliError::Runtime(format!("open db failed: {err}")))?;

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
