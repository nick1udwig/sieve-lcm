use tempfile::tempdir;

use sieve_lcm::cli::{
    CliCommand, CliError, CliSuccess, QueryOutput, execute_command, parse_command,
    serialize_error_json,
};

fn run_cli(args: &[&str]) -> Result<CliSuccess, CliError> {
    let owned = args
        .iter()
        .map(|value| (*value).to_string())
        .collect::<Vec<_>>();
    let command = parse_command(&owned)?;
    execute_command(command)
}

#[test]
fn query_both_returns_trusted_text_and_untrusted_refs_only() {
    let tmp = tempdir().expect("create tempdir");
    let trusted_db = tmp.path().join("trusted.db");
    let untrusted_db = tmp.path().join("untrusted.db");

    run_cli(&[
        "ingest",
        "--db",
        trusted_db.to_str().expect("trusted path utf8"),
        "--conversation",
        "global",
        "--role",
        "user",
        "--content",
        "I live in Livermore California",
        "--json",
    ])
    .expect("trusted ingest should succeed");

    run_cli(&[
        "ingest",
        "--db",
        untrusted_db.to_str().expect("untrusted path utf8"),
        "--conversation",
        "global",
        "--role",
        "assistant",
        "--content",
        "untrusted memory says Livermore might be rainy",
        "--json",
    ])
    .expect("untrusted ingest should succeed");

    let output = run_cli(&[
        "query",
        "--trusted-db",
        trusted_db.to_str().expect("trusted path utf8"),
        "--untrusted-db",
        untrusted_db.to_str().expect("untrusted path utf8"),
        "--conversation",
        "global",
        "--query",
        "Livermore",
        "--lane",
        "both",
        "--limit",
        "5",
        "--json",
    ])
    .expect("query should succeed");

    let CliSuccess::Query(QueryOutput {
        trusted_hits,
        untrusted_refs,
        ..
    }) = output
    else {
        panic!("expected query output")
    };

    assert!(!trusted_hits.is_empty(), "expected trusted hit");
    assert!(
        trusted_hits
            .iter()
            .any(|hit| hit.excerpt.to_ascii_lowercase().contains("livermore")),
        "trusted excerpts should include trusted text"
    );
    assert!(!untrusted_refs.is_empty(), "expected untrusted refs");

    let encoded = serde_json::to_string(&untrusted_refs).expect("encode refs");
    assert!(
        !encoded.to_ascii_lowercase().contains("rainy"),
        "untrusted refs must not leak untrusted text"
    );
}

#[test]
fn query_handles_utf8_snippet_boundaries_without_panicking() {
    let tmp = tempdir().expect("create tempdir");
    let trusted_db = tmp.path().join("trusted.db");

    // Put a multibyte rune right across the old byte-slice boundary.
    let mut content = "a".repeat(196);
    content.push('–');
    content.push_str(" Livermore utf8 snippet regression payload");

    run_cli(&[
        "ingest",
        "--db",
        trusted_db.to_str().expect("trusted path utf8"),
        "--conversation",
        "global",
        "--role",
        "user",
        "--content",
        &content,
        "--json",
    ])
    .expect("trusted ingest should succeed");

    let output = run_cli(&[
        "query",
        "--trusted-db",
        trusted_db.to_str().expect("trusted path utf8"),
        "--conversation",
        "global",
        "--query",
        "Livermore",
        "--lane",
        "trusted",
        "--json",
    ])
    .expect("query should succeed");

    let CliSuccess::Query(output) = output else {
        panic!("expected query output")
    };
    assert!(!output.trusted_hits.is_empty(), "expected trusted hit");
    assert!(
        output.trusted_hits[0].excerpt.ends_with("..."),
        "long unicode content should be truncated safely with ellipsis"
    );
}

#[test]
fn expand_resolves_untrusted_ref_content() {
    let tmp = tempdir().expect("create tempdir");
    let untrusted_db = tmp.path().join("untrusted.db");

    run_cli(&[
        "ingest",
        "--db",
        untrusted_db.to_str().expect("untrusted path utf8"),
        "--conversation",
        "global",
        "--role",
        "assistant",
        "--content",
        "opaque untrusted payload",
        "--json",
    ])
    .expect("untrusted ingest should succeed");

    let query = run_cli(&[
        "query",
        "--untrusted-db",
        untrusted_db.to_str().expect("untrusted path utf8"),
        "--conversation",
        "global",
        "--query",
        "opaque",
        "--lane",
        "untrusted",
        "--json",
    ])
    .expect("query should succeed");

    let ref_id = match query {
        CliSuccess::Query(output) => output
            .untrusted_refs
            .first()
            .expect("expected one ref")
            .opaque_ref
            .clone(),
        _ => panic!("expected query output"),
    };

    let expanded = run_cli(&[
        "expand",
        "--untrusted-db",
        untrusted_db.to_str().expect("untrusted path utf8"),
        "--conversation",
        "global",
        "--ref",
        &ref_id,
        "--json",
    ])
    .expect("expand should succeed");

    let CliSuccess::Expand(output) = expanded else {
        panic!("expected expand output")
    };
    assert_eq!(output.content, "opaque untrusted payload");
}

#[test]
fn parse_rejects_flag_value_that_is_really_json_flag() {
    let owned = [
        "ingest",
        "--db",
        "lane.db",
        "--role",
        "user",
        "--content",
        "--json",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();

    let error = parse_command(&owned).expect_err("expected missing value error");
    assert!(matches!(error, CliError::Usage(_)));
    let CliError::Usage(error) = error else {
        unreachable!("checked above")
    };
    assert!(
        error.message.contains("--content"),
        "expected content flag in error, got: {}",
        error.message
    );
    assert_eq!(
        error.usage.as_ref().map(|usage| usage.command.as_str()),
        Some("ingest")
    );
}

#[test]
fn parse_duplicate_flags_last_value_wins() {
    let owned = [
        "query", "--query", "first", "--query", "second", "--lane", "trusted", "--json",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();

    let command = parse_command(&owned).expect("duplicate flags should parse");
    let CliCommand::Query(args) = command else {
        panic!("expected query command")
    };
    assert_eq!(args.query, "second");
}

#[test]
fn invalid_command_json_error_includes_usage() {
    let owned = ["bogus"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

    let error = parse_command(&owned).expect_err("expected invalid command");
    let encoded = serialize_error_json(&error);
    let payload: serde_json::Value = serde_json::from_str(&encoded).expect("valid json");
    let message = payload["error"]["message"]
        .as_str()
        .expect("json error message");
    let usage = &payload["error"]["usage"];

    assert!(message.contains("unknown command `bogus`"));
    assert_eq!(usage["command"], "sieve-lcm-cli");
    assert_eq!(usage["syntax"], "sieve-lcm-cli [OPTIONS] <COMMAND>");
    assert_eq!(usage["commands"].as_array().map(Vec::len), Some(3));
}

#[test]
fn help_flags_return_usage_in_json_error_message() {
    for args in [
        ["--help"].as_slice(),
        ["-h"].as_slice(),
        ["query", "--help"].as_slice(),
    ] {
        let owned = args
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>();
        let error = parse_command(&owned).expect_err("expected help usage");
        let encoded = serialize_error_json(&error);
        let payload: serde_json::Value = serde_json::from_str(&encoded).expect("valid json");
        let message = payload["error"]["message"]
            .as_str()
            .expect("json error message");
        let usage = &payload["error"]["usage"];

        assert_eq!(message, "help requested");
        assert!(usage.is_object(), "missing usage for {:?}", args);
        assert!(
            usage["syntax"].as_str().is_some(),
            "missing syntax for {:?}",
            args
        );
    }
}

#[test]
fn query_help_usage_lists_options_and_choices() {
    let owned = ["query", "--help"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

    let error = parse_command(&owned).expect_err("expected help usage");
    let encoded = serialize_error_json(&error);
    let payload: serde_json::Value = serde_json::from_str(&encoded).expect("valid json");
    let usage = &payload["error"]["usage"];
    let options = usage["options"].as_array().expect("usage options array");

    assert_eq!(usage["command"], "query");
    assert_eq!(usage["syntax"], "sieve-lcm-cli query [OPTIONS]");
    assert!(
        options
            .iter()
            .any(|option| option["flags"] == serde_json::json!(["--query"]))
    );
    assert!(options.iter().any(|option| {
        option["flags"] == serde_json::json!(["--lane"])
            && option["choices"] == serde_json::json!(["trusted", "untrusted", "both"])
    }));
}
