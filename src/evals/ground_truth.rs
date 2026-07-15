//! ground truth fixture loader.
//!
//! each eval case pairs a query with expected uris and graded relevance
//! (1.0 = perfect, 0.5 = partial, 0.0 = irrelevant).
//!
//! fixtures use uris instead of frame ids so they're portable across
//! different .mv2 files. embedded via include_str! — no runtime io.

use serde::{Deserialize, Serialize};

use crate::{MemvidError, Result};

const EMBEDDED_FIXTURE: &str = include_str!("../../data/eval_fixtures.json");

/// a single ground truth case: query + expected uris + relevance grades
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCase {
    pub query: String,
    pub relevant_uris: Vec<String>,
    pub grades: Vec<f32>,
}

impl EvalCase {
    fn validate(&self) -> Result<()> {
        if self.query.trim().is_empty() {
            return Err(MemvidError::SchemaValidation {
                reason: "EvalCase query must not be empty".into(),
            });
        }
        if self.relevant_uris.len() != self.grades.len() {
            return Err(MemvidError::SchemaValidation {
                reason: format!(
                    "EvalCase '{}': relevant_uris ({}) and grades ({}) must have equal length",
                    self.query,
                    self.relevant_uris.len(),
                    self.grades.len()
                ),
            });
        }
        for &g in &self.grades {
            if !(0.0..=1.0).contains(&g) {
                return Err(MemvidError::SchemaValidation {
                    reason: format!(
                        "EvalCase '{}': grade {g} out of range [0.0, 1.0]",
                        self.query
                    ),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct EvalSuite {
    pub cases: Vec<EvalCase>,
}

impl EvalSuite {
    /// load the bundled fixture from data/eval_fixtures.json
    pub fn from_embedded() -> Result<Self> {
        Self::from_json(EMBEDDED_FIXTURE)
    }

    /// parse a custom fixture from a json string
    pub fn from_json(json: &str) -> Result<Self> {
        let cases: Vec<EvalCase> = serde_json::from_str(json)
            .map_err(|e| MemvidError::SchemaValidation {
                reason: format!("failed to parse eval fixtures: {e}"),
            })?;

        for case in &cases {
            case.validate()?;
        }

        Ok(Self { cases })
    }

    /// load from a file on disk (handy during dev without recompiling)
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| MemvidError::Io { source: e, path: Some(path.to_path_buf()) })?;
        Self::from_json(&json)
    }

    /// Number of cases in the suite.
    pub fn len(&self) -> usize {
        self.cases.len()
    }

    /// Whether the suite is empty.
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty()
    }

    /// keep only cases whose query contains keyword (case-insensitive)
    pub fn filter_by_keyword(&self, keyword: &str) -> Self {
        let lower = keyword.to_lowercase();
        Self {
            cases: self
                .cases
                .iter()
                .filter(|c| c.query.to_lowercase().contains(&lower))
                .cloned()
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_fixture_parses_and_validates() {
        let suite = EvalSuite::from_embedded().expect("embedded fixture must be valid");
        assert!(
            !suite.is_empty(),
            "embedded fixture must contain at least one case"
        );
        for case in &suite.cases {
            assert!(!case.query.is_empty());
            assert_eq!(case.relevant_uris.len(), case.grades.len());
        }
    }

    #[test]
    fn from_json_rejects_mismatched_lengths() {
        let json = r#"[{"query":"test","relevant_uris":["a","b"],"grades":[1.0]}]"#;
        assert!(EvalSuite::from_json(json).is_err());
    }

    #[test]
    fn from_json_rejects_grade_out_of_range() {
        let json = r#"[{"query":"test","relevant_uris":["a"],"grades":[1.5]}]"#;
        assert!(EvalSuite::from_json(json).is_err());
    }

    #[test]
    fn from_json_rejects_empty_query() {
        let json = r#"[{"query":"  ","relevant_uris":["a"],"grades":[1.0]}]"#;
        assert!(EvalSuite::from_json(json).is_err());
    }

    #[test]
    fn filter_by_keyword_works() {
        let suite = EvalSuite::from_embedded().unwrap();
        let filtered = suite.filter_by_keyword("crash");
        for case in &filtered.cases {
            assert!(
                case.query.to_lowercase().contains("crash"),
                "unexpected case: {}",
                case.query
            );
        }
    }
}
