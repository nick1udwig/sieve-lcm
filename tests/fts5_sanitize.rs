use sieve_lcm::store::fts5_sanitize::sanitize_fts5_query;

#[test]
fn quotes_simple_tokens() {
    assert_eq!(sanitize_fts5_query("hello world"), "\"hello\" \"world\"");
}

#[test]
fn preserves_hyphens_inside_quotes() {
    assert_eq!(
        sanitize_fts5_query("sub-agent restrict"),
        "\"sub-agent\" \"restrict\""
    );
}

#[test]
fn neutralizes_boolean_operators() {
    assert_eq!(
        sanitize_fts5_query("lcm_expand OR crash"),
        "\"lcm_expand\" \"OR\" \"crash\""
    );
}

#[test]
fn strips_internal_double_quotes() {
    assert_eq!(
        sanitize_fts5_query("hello \"world\""),
        "\"hello\" \"world\""
    );
}

#[test]
fn handles_colons_column_filter_syntax() {
    assert_eq!(
        sanitize_fts5_query("agent:foo bar"),
        "\"agent:foo\" \"bar\""
    );
}

#[test]
fn handles_prefix_star_operator() {
    assert_eq!(sanitize_fts5_query("lcm*"), "\"lcm*\"");
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
    assert_eq!(sanitize_fts5_query("expand"), "\"expand\"");
}

#[test]
fn collapses_multiple_spaces() {
    assert_eq!(sanitize_fts5_query("a   b    c"), "\"a\" \"b\" \"c\"");
}

#[test]
fn handles_not_operator() {
    assert_eq!(sanitize_fts5_query("NOT agent"), "\"NOT\" \"agent\"");
}

#[test]
fn handles_near_operator() {
    assert_eq!(sanitize_fts5_query("NEAR(a b)"), "\"NEAR(a\" \"b)\"");
}

#[test]
fn handles_caret_initial_token() {
    assert_eq!(sanitize_fts5_query("^start"), "\"^start\"");
}
