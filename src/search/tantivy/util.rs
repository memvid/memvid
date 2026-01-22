// Safe unwrap: single-element vector pop after length == 1 check.
#![allow(clippy::unwrap_used)]
use tantivy::query::{AllQuery, BooleanQuery, Occur, Query};

pub(super) fn to_search_value(value: &str) -> String {
    value.to_ascii_lowercase()
}

pub(super) fn combine_should_queries(mut queries: Vec<Box<dyn Query>>) -> Box<dyn Query> {
    match queries.len() {
        0 => Box::new(AllQuery),
        1 => queries.pop().unwrap(),
        _ => Box::new(BooleanQuery::new(
            queries
                .into_iter()
                .map(|query| (Occur::Should, query))
                .collect(),
        )),
    }
}
