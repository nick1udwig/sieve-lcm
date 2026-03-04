use sieve_lcm::store::fts5_sanitize::sanitize_fts5_query;

#[test]
fn quotes_simple_tokens() {
    assert_eq!(sanitize_fts5_query("hello world"), "\"hello\" \"world\"");
}

#[test]
fn preserves_hyphens_inside_quotes() {
    assert_eq!(sanitize_fts5_query("sub-agent restrict"), "\"sub-agent\" \"restrict\"");
}

#[test]
fn neutralizes_boolean_operators() {
    assert_eq!(sanitize_fts5_query("lcm_expand OR crash"), "\"lcm_expand\" \"OR\" \"crash\"");
}

#[test]
fn strips_internal_double_quotes() {
    assert_eq!(sanitize_fts5_query("hello \"world\""), "\"hello\" \"world\"");
}

#[test]
fn handles_colons() {
    assert_eq!(sanitize_fts5_query("agent:foo"), "\"agent:foo\"");
}

#[test]
fn handles_prefix_star_operator() {
    assert_eq!(sanitize_fts5_query("deploy*"), "\"deploy*\"");
}

#[test]
fn handles_empty_string() {
    assert_eq!(sanitize_fts5_query(""), "\"\"");
}

#[test]
fn handles_whitespace_only() {
    assert_eq!(sanitize_fts5_query("   \n\t  "), "\"\"");
}

#[test]
fn handles_single_token() {
    assert_eq!(sanitize_fts5_query("hello"), "\"hello\"");
}

#[test]
fn collapses_multiple_spaces() {
    assert_eq!(sanitize_fts5_query("a   b    c"), "\"a\" \"b\" \"c\"");
}

#[test]
fn handles_not_operator() {
    assert_eq!(sanitize_fts5_query("foo NOT bar"), "\"foo\" \"NOT\" \"bar\"");
}

#[test]
fn handles_near_operator() {
    assert_eq!(sanitize_fts5_query("foo NEAR bar"), "\"foo\" \"NEAR\" \"bar\"");
}

#[test]
fn handles_caret_initial_token() {
    assert_eq!(sanitize_fts5_query("^start"), "\"^start\"");
}
