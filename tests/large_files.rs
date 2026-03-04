use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use sieve_lcm::large_files::{
    extension_from_name_or_mime, extract_file_ids_from_content, format_file_reference,
    generate_exploration_summary, parse_file_blocks, ExplorationSummaryInput,
};

#[test]
fn parses_multiple_file_blocks_and_attributes() {
    let content = [
        r#"Before <file name="a.json" mime="application/json">{"a":1}</file>"#,
        "Middle",
        "<file name='notes.md'># Title\nBody</file>",
        "After",
    ]
    .join("\n");

    let blocks = parse_file_blocks(&content);
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].file_name.as_deref(), Some("a.json"));
    assert_eq!(blocks[0].mime_type.as_deref(), Some("application/json"));
    assert_eq!(blocks[0].text, r#"{"a":1}"#);
    assert_eq!(blocks[1].file_name.as_deref(), Some("notes.md"));
    assert_eq!(blocks[1].mime_type, None);
    assert!(blocks[1].text.contains("# Title"));
}

#[test]
fn formats_compact_file_references() {
    let text = format_file_reference(
        "file_aaaaaaaaaaaaaaaa",
        Some("paper.pdf"),
        Some("application/pdf"),
        42_150,
        "A concise summary.",
    );
    assert!(text.contains(
        "[LCM File: file_aaaaaaaaaaaaaaaa | paper.pdf | application/pdf | 42,150 bytes]"
    ));
    assert!(text.contains("Exploration Summary:"));
    assert!(text.contains("A concise summary."));
}

#[test]
fn resolves_extensions_from_name_or_mime() {
    assert_eq!(
        extension_from_name_or_mime(Some("report.csv"), Some("text/plain")),
        "csv"
    );
    assert_eq!(
        extension_from_name_or_mime(None, Some("application/json")),
        "json"
    );
    assert_eq!(extension_from_name_or_mime(None, None), "txt");
}

#[test]
fn extracts_file_ids_in_order_without_duplicates() {
    let ids = extract_file_ids_from_content(
        "See file_aaaaaaaaaaaaaaaa and file_bbbbbbbbbbbbbbbb then file_aaaaaaaaaaaaaaaa again.",
    );
    assert_eq!(
        ids,
        vec![
            "file_aaaaaaaaaaaaaaaa".to_string(),
            "file_bbbbbbbbbbbbbbbb".to_string()
        ]
    );
}

#[test]
fn uses_deterministic_structured_summary_for_json() {
    let summary = generate_exploration_summary(ExplorationSummaryInput {
        content: &serde_json::json!({
            "users": [{ "id": 1, "email": "a@example.com" }],
            "count": 1
        })
        .to_string(),
        file_name: Some("data.json"),
        mime_type: Some("application/json"),
        summarize_text: None,
    });

    assert!(summary.contains("Structured summary (JSON)"));
    assert!(summary.contains("Top-level type"));
}

#[test]
fn uses_deterministic_code_summary_for_code_files() {
    let summary = generate_exploration_summary(ExplorationSummaryInput {
        content: &[
            "import { readFileSync } from 'node:fs';",
            "export function runTask(input: string) {",
            "  return input.trim();",
            "}",
        ]
        .join("\n"),
        file_name: Some("task.ts"),
        mime_type: Some("text/x-typescript"),
        summarize_text: None,
    });

    assert!(summary.contains("Code exploration summary"));
    assert!(summary.contains("Imports/dependencies"));
    assert!(summary.contains("Top-level definitions"));
}

#[test]
fn uses_model_summary_hook_for_text_files_when_available() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_ref = calls.clone();
    let summary = generate_exploration_summary(ExplorationSummaryInput {
        content: &"This is a very long plain-text report.".repeat(500),
        file_name: Some("report.txt"),
        mime_type: Some("text/plain"),
        summarize_text: Some(Box::new(move |_prompt| {
            calls_ref.fetch_add(1, Ordering::SeqCst);
            Some("Model-produced exploration summary.".to_string())
        })),
    });

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(summary, "Model-produced exploration summary.");
}

#[test]
fn falls_back_to_deterministic_text_summary_when_model_summary_missing() {
    let summary = generate_exploration_summary(ExplorationSummaryInput {
        content: &["# Overview", "SYSTEM STATUS", "All systems nominal."].join("\n\n"),
        file_name: Some("status.txt"),
        mime_type: Some("text/plain"),
        summarize_text: Some(Box::new(|_prompt| None)),
    });

    assert!(summary.contains("Text exploration summary"));
    assert!(summary.contains("Detected section headers"));
}
