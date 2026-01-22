// Safe unwrap: single-element vector pop after length check.
#![allow(clippy::unwrap_used)]
use std::ops::Bound;

use super::engine::TantivyEngine;
use super::util::{combine_should_queries, to_search_value};
use crate::search::parser::{Expr, FieldTerm, ParsedQuery, Term as ParsedTerm, TextTerm};
use crate::{MemvidError, Result};
use tantivy::Term;
use tantivy::query::{
    AllQuery, BooleanQuery, Occur, PhraseQuery, Query, RangeQuery, RegexQuery, TermQuery,
    TermSetQuery,
};
use tantivy::schema::IndexRecordOption;

pub(super) fn build_root_query(
    engine: &TantivyEngine,
    parsed: &ParsedQuery,
    uri_filter: Option<&str>,
    scope_filter: Option<&str>,
    frame_filter: Option<&[u64]>,
) -> Result<Box<dyn Query>> {
    QueryPlanner { engine }.build_root_query(parsed, uri_filter, scope_filter, frame_filter)
}

struct QueryPlanner<'a> {
    engine: &'a TantivyEngine,
}

impl<'a> QueryPlanner<'a> {
    fn build_root_query(
        &self,
        parsed: &ParsedQuery,
        uri_filter: Option<&str>,
        scope_filter: Option<&str>,
        frame_filter: Option<&[u64]>,
    ) -> Result<Box<dyn Query>> {
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        clauses.push((Occur::Must, self.build_expr_query(&parsed.expr)?));

        if let Some(uri) = uri_filter {
            let normalized = to_search_value(uri);
            let term = Term::from_field_text(self.engine.uri, &normalized);
            clauses.push((
                Occur::Must,
                Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
            ));
        } else if scope_filter.is_some() {
            // Scope filters are evaluated after fetching the documents.
        }

        if let Some(ids) = frame_filter {
            if !ids.is_empty() {
                let terms: Vec<Term> = ids
                    .iter()
                    .map(|id| Term::from_field_u64(self.engine.frame_id, *id))
                    .collect();
                clauses.push((Occur::Must, Box::new(TermSetQuery::new(terms))));
            }
        }

        if clauses.len() == 1 {
            Ok(clauses.into_iter().next().unwrap().1)
        } else {
            Ok(Box::new(BooleanQuery::new(clauses)))
        }
    }

    fn build_expr_query(&self, expr: &Expr) -> Result<Box<dyn Query>> {
        match expr {
            Expr::Or(children) => {
                if children.is_empty() {
                    return Ok(Box::new(AllQuery));
                }
                if children.len() == 1 {
                    return self.build_expr_query(&children[0]);
                }
                let mut clauses = Vec::with_capacity(children.len());
                for child in children {
                    clauses.push((Occur::Should, self.build_expr_query(child)?));
                }
                Ok(Box::new(BooleanQuery::new(clauses)))
            }
            Expr::And(children) => {
                if children.is_empty() {
                    return Ok(Box::new(AllQuery));
                }
                if children.len() == 1 {
                    return self.build_expr_query(&children[0]);
                }
                let mut clauses = Vec::with_capacity(children.len());
                for child in children {
                    clauses.push((Occur::Must, self.build_expr_query(child)?));
                }
                Ok(Box::new(BooleanQuery::new(clauses)))
            }
            Expr::Not(child) => Ok(Box::new(BooleanQuery::new(vec![
                (Occur::Must, Box::new(AllQuery)),
                (Occur::MustNot, self.build_expr_query(child)?),
            ]))),
            Expr::Term(term) => self.build_term_query(term),
        }
    }

    fn build_term_query(&self, term: &ParsedTerm) -> Result<Box<dyn Query>> {
        match term {
            ParsedTerm::Text(text) => self.build_text_query(text),
            ParsedTerm::Field(field) => self.build_field_query(field),
        }
    }

    fn build_text_query(&self, text: &TextTerm) -> Result<Box<dyn Query>> {
        match text {
            TextTerm::Word(word) => self.build_word_query(word),
            TextTerm::Phrase(phrase) => self.build_phrase_query(phrase),
            TextTerm::Wildcard(pattern) => {
                let regex = pattern.regex.as_str().to_ascii_lowercase();
                let query =
                    RegexQuery::from_pattern(&regex, self.engine.content).map_err(|err| {
                        MemvidError::Tantivy {
                            reason: err.to_string(),
                        }
                    })?;
                Ok(Box::new(query))
            }
        }
    }

    fn build_field_query(&self, field: &FieldTerm) -> Result<Box<dyn Query>> {
        match field {
            FieldTerm::Uri(value) => {
                let normalized = to_search_value(value);
                Ok(Box::new(TermQuery::new(
                    Term::from_field_text(self.engine.uri, &normalized),
                    IndexRecordOption::Basic,
                )))
            }
            FieldTerm::Scope(_) => Ok(Box::new(AllQuery)),
            FieldTerm::Track(value) => {
                let normalized = to_search_value(value);
                Ok(Box::new(TermQuery::new(
                    Term::from_field_text(self.engine.track, &normalized),
                    IndexRecordOption::Basic,
                )))
            }
            FieldTerm::Tag(value) => {
                let normalized = to_search_value(value);
                Ok(Box::new(TermQuery::new(
                    Term::from_field_text(self.engine.tags, &normalized),
                    IndexRecordOption::Basic,
                )))
            }
            FieldTerm::Label(value) => {
                let normalized = to_search_value(value);
                Ok(Box::new(TermQuery::new(
                    Term::from_field_text(self.engine.labels, &normalized),
                    IndexRecordOption::Basic,
                )))
            }
            FieldTerm::DateRange(range) => {
                let lower = range
                    .start
                    .map(|value| {
                        Bound::Included(Term::from_field_i64(self.engine.timestamp, value))
                    })
                    .unwrap_or(Bound::Unbounded);
                let upper = range
                    .end
                    .map(|value| {
                        Bound::Included(Term::from_field_i64(self.engine.timestamp, value))
                    })
                    .unwrap_or(Bound::Unbounded);
                Ok(Box::new(RangeQuery::new(lower, upper)))
            }
        }
    }

    fn build_word_query(&self, word: &str) -> Result<Box<dyn Query>> {
        // Handle empty words gracefully (from punctuation-only tokens like "-")
        if word.is_empty() {
            return Ok(Box::new(AllQuery));
        }

        let tokens = self.engine.analyse_text(word);
        if tokens.is_empty() {
            // Word produced no tokens after analysis - match all instead of erroring
            // This can happen with punctuation-only or stop-word-only terms
            return Ok(Box::new(AllQuery));
        }
        let mut queries: Vec<Box<dyn Query>> = Vec::new();
        if tokens.len() == 1 {
            queries.push(Box::new(TermQuery::new(
                Term::from_field_text(self.engine.content, &tokens[0]),
                IndexRecordOption::WithFreqsAndPositions,
            )));
        } else {
            let terms: Vec<Term> = tokens
                .iter()
                .map(|token| Term::from_field_text(self.engine.content, token))
                .collect();
            queries.push(Box::new(PhraseQuery::new(terms)));
        }

        let normalized = to_search_value(word);
        queries.push(Box::new(TermQuery::new(
            Term::from_field_text(self.engine.tags, &normalized),
            IndexRecordOption::Basic,
        )));
        queries.push(Box::new(TermQuery::new(
            Term::from_field_text(self.engine.labels, &normalized),
            IndexRecordOption::Basic,
        )));
        queries.push(Box::new(TermQuery::new(
            Term::from_field_text(self.engine.track, &normalized),
            IndexRecordOption::Basic,
        )));
        queries.push(Box::new(TermQuery::new(
            Term::from_field_text(self.engine.uri, &normalized),
            IndexRecordOption::Basic,
        )));

        Ok(combine_should_queries(queries))
    }

    fn build_phrase_query(&self, phrase: &str) -> Result<Box<dyn Query>> {
        // Handle empty phrases gracefully
        if phrase.is_empty() {
            return Ok(Box::new(AllQuery));
        }

        let tokens = self.engine.analyse_text(phrase);
        if tokens.is_empty() {
            // Phrase produced no tokens after analysis - match all instead of erroring
            return Ok(Box::new(AllQuery));
        }
        let mut queries: Vec<Box<dyn Query>> = Vec::new();
        if tokens.len() == 1 {
            queries.push(Box::new(TermQuery::new(
                Term::from_field_text(self.engine.content, &tokens[0]),
                IndexRecordOption::WithFreqsAndPositions,
            )));
        } else {
            let terms: Vec<Term> = tokens
                .iter()
                .map(|token| Term::from_field_text(self.engine.content, token))
                .collect();
            queries.push(Box::new(PhraseQuery::new(terms)));
        }

        let normalized = to_search_value(phrase);
        queries.push(Box::new(TermQuery::new(
            Term::from_field_text(self.engine.tags, &normalized),
            IndexRecordOption::Basic,
        )));
        queries.push(Box::new(TermQuery::new(
            Term::from_field_text(self.engine.labels, &normalized),
            IndexRecordOption::Basic,
        )));
        queries.push(Box::new(TermQuery::new(
            Term::from_field_text(self.engine.track, &normalized),
            IndexRecordOption::Basic,
        )));
        queries.push(Box::new(TermQuery::new(
            Term::from_field_text(self.engine.uri, &normalized),
            IndexRecordOption::Basic,
        )));

        Ok(combine_should_queries(queries))
    }
}
