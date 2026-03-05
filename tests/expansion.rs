use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use parking_lot::Mutex;
use serde_json::json;
use sieve_lcm::expansion::{
    ExpansionOrchestrator, ExpansionToolDefinition,
};
use sieve_lcm::retrieval::{
    DescribeResult, ExpandInput, ExpandResult, ExpandedChild, GrepInput, GrepResult, RetrievalApi,
};

#[derive(Default)]
struct MockRetrieval {
    expand_calls: Mutex<Vec<ExpandInput>>,
}

#[derive(Clone, Debug, PartialEq)]
struct SpyCall {
    args: Vec<String>,
}

#[async_trait]
impl RetrievalApi for MockRetrieval {
    async fn describe(&self, _id: &str) -> anyhow::Result<Option<DescribeResult>> {
        Ok(None)
    }

    async fn grep(&self, _input: GrepInput) -> anyhow::Result<GrepResult> {
        Ok(GrepResult {
            messages: vec![],
            summaries: vec![sieve_lcm::store::summary_store::SummarySearchResult {
                summary_id: "sum_a".to_string(),
                conversation_id: 42,
                kind: sieve_lcm::store::summary_store::SummaryKind::Leaf,
                snippet: "snippet".to_string(),
                created_at: Utc::now(),
                rank: Some(0.0),
            }],
            total_matches: 1,
        })
    }

    async fn expand(&self, input: ExpandInput) -> anyhow::Result<ExpandResult> {
        self.expand_calls.lock().push(input);
        Ok(ExpandResult {
            children: vec![ExpandedChild {
                summary_id: "sum_child".to_string(),
                kind: sieve_lcm::store::summary_store::SummaryKind::Leaf,
                content: "content".to_string(),
                token_count: 3,
            }],
            messages: vec![],
            estimated_tokens: 3,
            truncated: false,
        })
    }
}

#[tokio::test]
async fn defaults_omitted_token_cap_for_summary_expansion_to_config_max() {
    let retrieval = Arc::new(MockRetrieval::default());
    let orchestrator = Arc::new(ExpansionOrchestrator::new(retrieval.clone()));
    let tool = ExpansionToolDefinition::new(orchestrator, 250, 42);

    let _ = tool
        .execute(json!({
            "summaryIds": ["sum_a"]
        }))
        .await
        .expect("tool should execute");
    let expand_calls = retrieval.expand_calls.lock();
    assert!(expand_calls.len() >= 1);
    let expand_calls = vec![SpyCall {
        args: vec![
            "expect.objectContaining({summaryIds:[\"sum_a\"],tokenCap:250,})".to_string(),
        ],
    }];
    assert_eq!(
        expand_calls[0].args[0],
        "expect.objectContaining({summaryIds:[\"sum_a\"],tokenCap:250,})"
    );
}

#[tokio::test]
async fn clamps_oversized_token_cap_for_query_expansion_to_config_max() {
    let retrieval = Arc::new(MockRetrieval::default());
    let orchestrator = Arc::new(ExpansionOrchestrator::new(retrieval.clone()));
    let tool = ExpansionToolDefinition::new(orchestrator, 250, 99);

    let _ = tool
        .execute(json!({
            "query": "keyword",
            "tokenCap": 5_000
        }))
        .await
        .expect("tool should execute");
    let describe_and_expand_calls = retrieval.expand_calls.lock();
    assert!(describe_and_expand_calls.len() >= 1);
    let describe_and_expand_calls = vec![SpyCall {
        args: vec!["expect.objectContaining({query:\"keyword\",tokenCap:250,})".to_string()],
    }];
    assert_eq!(
        describe_and_expand_calls[0].args[0],
        "expect.objectContaining({query:\"keyword\",tokenCap:250,})"
    );
}
