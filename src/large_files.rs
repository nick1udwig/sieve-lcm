use std::collections::{BTreeSet, HashMap, HashSet};

use regex::Regex;

const TEXT_SUMMARY_SLICE_CHARS: usize = 2400;
const TEXT_HEADER_LIMIT: usize = 18;

fn estimate_tokens(content: &str) -> i64 {
    ((content.len() as f64) / 4.0).ceil() as i64
}

fn format_number_en_us(value: i64) -> String {
    let negative = value < 0;
    let digits = value.abs().to_string();
    let mut out = String::with_capacity(digits.len() + (digits.len() / 3));
    for (idx, ch) in digits.chars().enumerate() {
        if idx > 0 && (digits.len() - idx) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    if negative {
        format!("-{}", out)
    } else {
        out
    }
}

fn code_extensions() -> HashSet<&'static str> {
    [
        "c", "cc", "cpp", "cs", "go", "h", "hpp", "java", "js", "jsx", "kt", "m", "php",
        "py", "rb", "rs", "scala", "sh", "sql", "swift", "ts", "tsx",
    ]
    .into_iter()
    .collect()
}

fn structured_extensions() -> HashSet<&'static str> {
    ["csv", "json", "tsv", "xml", "yaml", "yml"]
        .into_iter()
        .collect()
}

fn mime_extension_map() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("application/json", "json"),
        ("application/xml", "xml"),
        ("application/yaml", "yaml"),
        ("application/x-yaml", "yaml"),
        ("application/x-ndjson", "json"),
        ("application/csv", "csv"),
        ("application/javascript", "js"),
        ("application/typescript", "ts"),
        ("application/x-python-code", "py"),
        ("application/x-rust", "rs"),
        ("application/x-sh", "sh"),
        ("text/csv", "csv"),
        ("text/markdown", "md"),
        ("text/plain", "txt"),
        ("text/tab-separated-values", "tsv"),
        ("text/x-c", "c"),
        ("text/x-c++", "cpp"),
        ("text/x-go", "go"),
        ("text/x-java", "java"),
        ("text/x-python", "py"),
        ("text/x-rust", "rs"),
        ("text/x-script.python", "py"),
        ("text/x-shellscript", "sh"),
        ("text/x-typescript", "ts"),
        ("text/xml", "xml"),
    ])
}

#[derive(Clone, Debug, PartialEq)]
pub struct FileBlock {
    pub full_match: String,
    pub start: usize,
    pub end: usize,
    pub attributes: HashMap<String, String>,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub text: String,
}

pub struct ExplorationSummaryInput<'a> {
    pub content: &'a str,
    pub file_name: Option<&'a str>,
    pub mime_type: Option<&'a str>,
    pub summarize_text: Option<Box<dyn Fn(&str) -> Option<String> + Send + Sync + 'a>>,
}

fn parse_file_attributes(raw: &str) -> HashMap<String, String> {
    let mut attrs = HashMap::new();
    let re = Regex::new(r#"([A-Za-z_:][A-Za-z0-9_:\-.]*)\s*=\s*("([^"]*)"|'([^']*)'|([^\s"'>]+))"#)
        .unwrap();
    for caps in re.captures_iter(raw) {
        let key = caps.get(1).map(|m| m.as_str().trim().to_lowercase());
        let value = caps
            .get(3)
            .or_else(|| caps.get(4))
            .or_else(|| caps.get(5))
            .map(|m| m.as_str().trim().to_string());
        if let (Some(key), Some(value)) = (key, value) {
            if !key.is_empty() && !value.is_empty() {
                attrs.insert(key, value);
            }
        }
    }
    attrs
}

fn normalize_text_for_line(text: &str, max_len: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<&str>>().join(" ");
    if compact.len() <= max_len {
        compact
    } else {
        format!("{}...", &compact[..max_len])
    }
}

fn collect_file_name_extension(file_name: Option<&str>) -> Option<String> {
    let file_name = file_name?.trim();
    let base = file_name
        .rsplit('/')
        .next()
        .unwrap_or(file_name)
        .rsplit('\\')
        .next()
        .unwrap_or(file_name);
    let idx = base.rfind('.')?;
    if idx == 0 || idx == base.len() - 1 {
        return None;
    }
    let ext = base[idx + 1..].to_lowercase();
    let valid = Regex::new(r"^[a-z0-9]{1,10}$").unwrap();
    if valid.is_match(&ext) {
        Some(ext)
    } else {
        None
    }
}

fn guess_mime_extension(mime_type: Option<&str>) -> Option<String> {
    let normalized = mime_type?.trim().to_lowercase();
    mime_extension_map().get(normalized.as_str()).map(|v| v.to_string())
}

fn is_structured(mime_type: Option<&str>, extension: Option<&str>) -> bool {
    let mime = mime_type.unwrap_or_default().trim().to_lowercase();
    let prefixes = [
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/x-ndjson",
        "text/csv",
        "text/tab-separated-values",
        "text/xml",
    ];
    if prefixes.iter().any(|candidate| mime.starts_with(candidate)) {
        return true;
    }
    extension
        .map(|ext| structured_extensions().contains(ext))
        .unwrap_or(false)
}

fn is_code(mime_type: Option<&str>, extension: Option<&str>) -> bool {
    let mime = mime_type.unwrap_or_default().trim().to_lowercase();
    let prefixes = [
        "application/javascript",
        "application/typescript",
        "application/x-python-code",
        "application/x-rust",
        "text/javascript",
        "text/x-c",
        "text/x-c++",
        "text/x-go",
        "text/x-java",
        "text/x-python",
        "text/x-rust",
        "text/x-script.python",
        "text/x-shellscript",
        "text/x-typescript",
    ];
    if prefixes.iter().any(|candidate| mime.starts_with(candidate)) {
        return true;
    }
    extension
        .map(|ext| code_extensions().contains(ext))
        .unwrap_or(false)
}

fn unique_ordered(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = vec![];
    for value in values {
        if seen.insert(value.clone()) {
            out.push(value);
        }
    }
    out
}

fn explore_json(content: &str) -> String {
    let parsed = serde_json::from_str::<serde_json::Value>(content);
    if parsed.is_err() {
        return "Structured summary (JSON): failed to parse as valid JSON.".to_string();
    }
    let parsed = parsed.unwrap();
    fn describe(value: &serde_json::Value, depth: usize) -> String {
        if depth >= 2 {
            return "...".to_string();
        }
        if let Some(arr) = value.as_array() {
            let sample = arr
                .iter()
                .take(3)
                .map(|item| describe(item, depth + 1))
                .collect::<Vec<String>>();
            return format!(
                "array(len={}{}{})",
                arr.len(),
                if sample.is_empty() { "" } else { ", sample=[" },
                if sample.is_empty() {
                    "".to_string()
                } else {
                    format!("{}]", sample.join(", "))
                }
            );
        }
        if let Some(obj) = value.as_object() {
            let mut keys = obj.keys().cloned().collect::<Vec<String>>();
            keys.sort();
            let preview = keys.iter().take(10).cloned().collect::<Vec<String>>().join(", ");
            return format!(
                "object(keys={}{}{})",
                keys.len(),
                if preview.is_empty() { "" } else { ": " },
                preview
            );
        }
        match value {
            serde_json::Value::String(_) => "string".to_string(),
            serde_json::Value::Number(_) => "number".to_string(),
            serde_json::Value::Bool(_) => "boolean".to_string(),
            serde_json::Value::Null => "null".to_string(),
            _ => "unknown".to_string(),
        }
    }
    let top_level = if parsed.is_array() {
        "array"
    } else if parsed.is_object() {
        "object"
    } else {
        "primitive"
    };
    format!(
        "Structured summary (JSON):\nTop-level type: {}.\nShape: {}.",
        top_level,
        describe(&parsed, 0)
    )
}

fn parse_delimited_line(line: &str, delimiter: char) -> Vec<String> {
    line.split(delimiter)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn explore_delimited(content: &str, delimiter: char, kind: &str) -> String {
    let lines: Vec<String> = content
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(ToString::to_string)
        .collect();
    if lines.is_empty() {
        return format!("Structured summary ({}): no rows found.", kind);
    }
    let headers = parse_delimited_line(&lines[0], delimiter);
    let row_count = lines.len().saturating_sub(1);
    let first_data = lines
        .get(1)
        .map(|l| normalize_text_for_line(l, 180))
        .unwrap_or_else(|| "(no data rows)".to_string());
    format!(
        "Structured summary ({}):\nRows: {}.\nColumns ({}): {}.\nFirst row sample: {}.",
        kind,
        format_number_en_us(row_count as i64),
        format_number_en_us(headers.len() as i64),
        if headers.is_empty() {
            "(none detected)".to_string()
        } else {
            headers.join(", ")
        },
        first_data
    )
}

fn explore_yaml(content: &str) -> String {
    let re = Regex::new(r"^([A-Za-z0-9_.-]+):\s*(?:#.*)?$").unwrap();
    let keys = unique_ordered(
        content
            .lines()
            .filter_map(|line| re.captures(line).and_then(|caps| caps.get(1)))
            .map(|m| m.as_str().to_string()),
    );
    format!(
        "Structured summary (YAML):\nTop-level keys ({}): {}.",
        keys.len(),
        if keys.is_empty() {
            "(none detected)".to_string()
        } else {
            keys.into_iter().take(30).collect::<Vec<String>>().join(", ")
        }
    )
}

fn explore_xml(content: &str) -> String {
    let root_re = Regex::new(r"<([A-Za-z0-9_:-]+)(\s|>)").unwrap();
    let root_tag = root_re
        .captures(content)
        .and_then(|c| c.get(1).map(|m| m.as_str().to_string()))
        .unwrap_or_else(|| "unknown".to_string());
    let tags = root_re
        .captures_iter(content)
        .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
        .filter(|t| t != &root_tag)
        .take(30)
        .collect::<Vec<String>>();
    let child_tags = unique_ordered(tags);
    format!(
        "Structured summary (XML):\nRoot element: {}.\nChild elements seen: {}.",
        root_tag,
        if child_tags.is_empty() {
            "(none detected)".to_string()
        } else {
            child_tags.join(", ")
        }
    )
}

pub fn explore_structured_data(content: &str, mime_type: Option<&str>, file_name: Option<&str>) -> String {
    let extension = collect_file_name_extension(file_name).or_else(|| guess_mime_extension(mime_type));
    let normalized_mime = mime_type.unwrap_or_default().trim().to_lowercase();

    if extension.as_deref() == Some("json") || normalized_mime.starts_with("application/json") {
        return explore_json(content);
    }
    if extension.as_deref() == Some("csv") || normalized_mime.starts_with("text/csv") {
        return explore_delimited(content, ',', "CSV");
    }
    if extension.as_deref() == Some("tsv")
        || normalized_mime.starts_with("text/tab-separated-values")
    {
        return explore_delimited(content, '\t', "TSV");
    }
    if extension.as_deref() == Some("xml")
        || normalized_mime.starts_with("text/xml")
        || normalized_mime.starts_with("application/xml")
    {
        return explore_xml(content);
    }
    if matches!(extension.as_deref(), Some("yaml" | "yml")) || normalized_mime.contains("yaml") {
        return explore_yaml(content);
    }
    format!(
        "Structured summary:\nCharacters: {}.\nLines: {}.",
        format_number_en_us(content.len() as i64),
        format_number_en_us(content.lines().count() as i64)
    )
}

pub fn explore_code(content: &str, file_name: Option<&str>) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let import_re =
        Regex::new(r"^\s*(import\s+|from\s+\S+\s+import\s+|const\s+\w+\s*=\s*require\()").unwrap();
    let signature_re = Regex::new(
        r"^(export\s+)?(async\s+)?(function|class|interface|type|const\s+\w+\s*=\s*\(|def\s+\w+\(|struct\s+\w+)",
    )
    .unwrap();
    let imports = unique_ordered(
        lines
            .iter()
            .copied()
            .filter(|line| import_re.is_match(line))
            .map(|line| normalize_text_for_line(line, 180)),
    );
    let signatures = unique_ordered(
        lines
            .iter()
            .map(|line| line.trim())
            .filter(|line| signature_re.is_match(line))
            .map(|line| normalize_text_for_line(line, 200)),
    );
    format!(
        "Code exploration summary{}:\nLines: {}.\nImports/dependencies ({}): {}.\nTop-level definitions ({}): {}.",
        file_name
            .map(|name| format!(" ({})", name))
            .unwrap_or_default(),
        format_number_en_us(lines.len() as i64),
        format_number_en_us(imports.len() as i64),
        if imports.is_empty() {
            "none detected".to_string()
        } else {
            imports.into_iter().take(12).collect::<Vec<String>>().join(" | ")
        },
        format_number_en_us(signatures.len() as i64),
        if signatures.is_empty() {
            "none detected".to_string()
        } else {
            signatures.into_iter().take(24).collect::<Vec<String>>().join(" | ")
        }
    )
}

fn extract_text_headers(content: &str) -> Vec<String> {
    let header_re_a = Regex::new(r"^#{1,6}\s+").unwrap();
    let header_re_b = Regex::new(r"^[A-Z0-9][A-Z0-9\s:_-]{6,}$").unwrap();
    unique_ordered(
        content
            .lines()
            .map(str::trim)
            .filter(|line| line.len() > 1)
            .filter(|line| header_re_a.is_match(line) || header_re_b.is_match(line))
            .map(|line| normalize_text_for_line(line, 160))
            .take(TEXT_HEADER_LIMIT),
    )
}

fn build_text_sample(content: &str) -> String {
    if content.len() <= TEXT_SUMMARY_SLICE_CHARS * 2 {
        return content.to_string();
    }
    let middle_start = (content.len() / 2).saturating_sub(TEXT_SUMMARY_SLICE_CHARS / 2);
    let middle_end = (middle_start + TEXT_SUMMARY_SLICE_CHARS).min(content.len());
    let head = &content[..TEXT_SUMMARY_SLICE_CHARS.min(content.len())];
    let mid = &content[middle_start..middle_end];
    let tail_start = content.len().saturating_sub(TEXT_SUMMARY_SLICE_CHARS);
    let tail = &content[tail_start..];
    format!(
        "[Document Start]\n\n{}\n\n[Document Middle]\n\n{}\n\n[Document End]\n\n{}",
        head, mid, tail
    )
}

fn build_text_prompt(content: &str, file_name: Option<&str>, mime_type: Option<&str>, headers: &[String]) -> String {
    let sample = build_text_sample(content);
    format!(
        "Summarize this large file for retrieval-time context references.\nFile name: {}\nMime type: {}\nLength: {} chars\nLine count: {}\n{}\nProduce 200-300 words with:\n- What the document is about\n- Key sections and topics\n- Important names, dates, and numbers\n- Any action items or constraints\nDo not quote long passages verbatim.\n\nDocument sample:\n{}",
        file_name.unwrap_or("unknown"),
        mime_type.unwrap_or("unknown"),
        format_number_en_us(content.len() as i64),
        format_number_en_us(content.lines().count() as i64),
        if headers.is_empty() {
            "Detected section headers: none".to_string()
        } else {
            format!("Detected section headers: {}", headers.join(" | "))
        },
        sample
    )
}

fn explore_text_deterministic_fallback(content: &str, file_name: Option<&str>) -> String {
    let normalized = content.split_whitespace().collect::<Vec<&str>>().join(" ");
    let headers = extract_text_headers(content);
    let line_count = content.lines().count();
    let word_count = if normalized.is_empty() {
        0
    } else {
        normalized.split_whitespace().count()
    };
    let first = normalize_text_for_line(content.get(..500).unwrap_or_default(), 500);
    let last = normalize_text_for_line(
        content
            .get(content.len().saturating_sub(500)..)
            .unwrap_or_default(),
        500,
    );
    format!(
        "Text exploration summary{}:\nCharacters: {}.\nWords: {}.\nLines: {}.\nDetected section headers: {}.\nOpening excerpt: {}.\nClosing excerpt: {}.",
        file_name
            .map(|n| format!(" ({})", n))
            .unwrap_or_default(),
        format_number_en_us(content.len() as i64),
        format_number_en_us(word_count as i64),
        format_number_en_us(line_count as i64),
        if headers.is_empty() {
            "none detected".to_string()
        } else {
            headers.join(" | ")
        },
        if first.is_empty() { "(empty)".to_string() } else { first },
        if last.is_empty() { "(empty)".to_string() } else { last }
    )
}

fn explore_text(input: &ExplorationSummaryInput<'_>) -> String {
    let headers = extract_text_headers(input.content);
    if let Some(summarize_text) = &input.summarize_text {
        let prompt = build_text_prompt(input.content, input.file_name, input.mime_type, &headers);
        if let Some(summary) = summarize_text(&prompt).map(|s| s.trim().to_string()) {
            if !summary.is_empty() {
                return summary;
            }
        }
    }
    explore_text_deterministic_fallback(input.content, input.file_name)
}

pub fn parse_file_blocks(content: &str) -> Vec<FileBlock> {
    let re = Regex::new(r"(?is)<file\b([^>]*)>(.*?)</file>").unwrap();
    let mut out = vec![];
    for caps in re.captures_iter(content) {
        let full = caps.get(0).unwrap();
        let raw_attrs = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
        let text = caps.get(2).map(|m| m.as_str()).unwrap_or_default();
        let attributes = parse_file_attributes(raw_attrs);
        out.push(FileBlock {
            full_match: full.as_str().to_string(),
            start: full.start(),
            end: full.end(),
            file_name: attributes.get("name").cloned(),
            mime_type: attributes.get("mime").cloned(),
            attributes,
            text: text.to_string(),
        });
    }
    out
}

pub fn extension_from_name_or_mime(file_name: Option<&str>, mime_type: Option<&str>) -> String {
    collect_file_name_extension(file_name)
        .or_else(|| guess_mime_extension(mime_type))
        .unwrap_or_else(|| "txt".to_string())
}

pub fn extract_file_ids_from_content(content: &str) -> Vec<String> {
    let re = Regex::new(r"(?i)\bfile_[a-f0-9]{16}\b").unwrap();
    unique_ordered(
        re.find_iter(content)
            .map(|m| m.as_str().to_lowercase())
            .collect::<Vec<String>>(),
    )
}

pub fn format_file_reference(
    file_id: &str,
    file_name: Option<&str>,
    mime_type: Option<&str>,
    byte_size: i64,
    summary: &str,
) -> String {
    let name = file_name.unwrap_or("unknown").trim();
    let mime = mime_type.unwrap_or("unknown").trim();
    let byte_size = byte_size.max(0);
    format!(
        "[LCM File: {} | {} | {} | {} bytes]\n\nExploration Summary:\n{}",
        file_id,
        name,
        mime,
        format_number_en_us(byte_size),
        if summary.trim().is_empty() {
            "(no summary available)"
        } else {
            summary.trim()
        }
    )
}

pub fn generate_exploration_summary(input: ExplorationSummaryInput<'_>) -> String {
    let extension = extension_from_name_or_mime(input.file_name, input.mime_type);
    if is_structured(input.mime_type, Some(extension.as_str())) {
        return explore_structured_data(input.content, input.mime_type, input.file_name);
    }
    if is_code(input.mime_type, Some(extension.as_str())) {
        return explore_code(input.content, input.file_name);
    }
    explore_text(&input)
}

pub fn is_large_file(content: &str, threshold_tokens: i64) -> bool {
    estimate_tokens(content) >= threshold_tokens.max(1)
}
