use tempfile::tempdir;

use sieve_lcm::cli::{
    execute_command, parse_command, CliError, CliSuccess, QueryOutput,
};

fn run_cli(args: &[&str]) -> Result<CliSuccess, CliError> {
    let owned = args.iter().map(|value| (*value).to_string()).collect::<Vec<_>>();
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
