pub fn sanitize_fts5_query(raw: &str) -> String {
    let tokens: Vec<&str> = raw
        .split_whitespace()
        .filter(|part| !part.is_empty())
        .collect();
    if tokens.is_empty() {
        return "\"\"".to_string();
    }
    tokens
        .into_iter()
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect::<Vec<String>>()
        .join(" ")
}
