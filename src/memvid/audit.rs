//! Audit functionality for generating provenance reports.
//!
//! This module provides the `audit` method on `Memvid` that generates
//! structured audit reports showing all sources used to answer a question.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::memvid::lifecycle::Memvid;
use crate::types::ask::{AskMode, AskRequest};
use crate::types::audit::{AuditOptions, AuditReport, SourceSpan};
use crate::{Result, VecEmbedder};

/// Default top-k for audit queries.
const DEFAULT_AUDIT_TOP_K: usize = 10;
/// Default snippet length for audit reports.
const DEFAULT_AUDIT_SNIPPET_CHARS: usize = 500;
/// Audit report version.
const AUDIT_REPORT_VERSION: &str = "1.0";

#[cfg(feature = "lex")]
impl Memvid {
    /// Generate an audit report for a question.
    ///
    /// This method performs a retrieval-augmented query and returns a structured
    /// report containing all sources used to generate the answer, along with
    /// provenance metadata for compliance and debugging.
    ///
    /// # Arguments
    ///
    /// * `question` - The question to audit.
    /// * `options` - Optional configuration for the audit query.
    /// * `embedder` - Optional embedder for semantic search.
    ///
    /// # Returns
    ///
    /// An `AuditReport` containing the question, answer, and all source spans.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let report = mem.audit("What is Memvid?", None, None::<&dyn VecEmbedder>)?;
    /// println!("{}", report.to_text());
    /// ```
    pub fn audit<E>(
        &mut self,
        question: &str,
        options: Option<AuditOptions>,
        embedder: Option<&E>,
    ) -> Result<AuditReport>
    where
        E: VecEmbedder + ?Sized,
    {
        let opts = options.unwrap_or_default();
        let top_k = opts.top_k.unwrap_or(DEFAULT_AUDIT_TOP_K);
        let snippet_chars = opts.snippet_chars.unwrap_or(DEFAULT_AUDIT_SNIPPET_CHARS);
        let mode = opts.mode.unwrap_or(AskMode::Hybrid);

        let request = AskRequest {
            question: question.to_string(),
            top_k,
            snippet_chars,
            uri: None,
            frames: Vec::new(),
            scope: opts.scope.clone(),
            cursor: None,
            start: opts.start,
            end: opts.end,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            context_only: false,
            mode,
            as_of_frame: None,
            as_of_ts: None,
            adaptive: None,
        };

        let response = self.ask(request, embedder)?;

        // Build source spans from the response
        let mut sources: Vec<SourceSpan> = Vec::new();
        let mut notes: Vec<String> = Vec::new();

        for (idx, citation) in response.citations.iter().enumerate() {
            // Get the corresponding hit for additional context
            let hit = response
                .retrieval
                .hits
                .iter()
                .find(|h| h.frame_id == citation.frame_id);

            // Get frame metadata for rich source information
            let frame_data = self.frame_by_id(citation.frame_id).ok();

            let snippet = if opts.include_snippets {
                hit.map(|h| h.chunk_text.clone().unwrap_or_else(|| h.text.clone()))
            } else {
                None
            };

            let source = SourceSpan {
                index: idx + 1,
                frame_id: citation.frame_id,
                uri: citation.uri.clone(),
                title: frame_data.as_ref().and_then(|f| f.title.clone()),
                chunk_range: citation.chunk_range,
                score: citation.score,
                tags: frame_data
                    .as_ref()
                    .map(|f| f.tags.clone())
                    .unwrap_or_default(),
                labels: frame_data
                    .as_ref()
                    .map(|f| f.labels.clone())
                    .unwrap_or_default(),
                frame_timestamp: frame_data.as_ref().map(|f| f.timestamp),
                content_dates: frame_data
                    .as_ref()
                    .map(|f| f.content_dates.clone())
                    .unwrap_or_default(),
                snippet,
            };

            sources.push(source);
        }

        // Add notes about retrieval behavior
        if response.retriever != response.mode.into() {
            notes.push(format!(
                "Retriever fell back from {:?} to {:?}",
                response.mode, response.retriever
            ));
        }

        let generated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        Ok(AuditReport {
            version: AUDIT_REPORT_VERSION.to_string(),
            generated_at,
            question: question.to_string(),
            answer: response.answer,
            mode: response.mode,
            retriever: response.retriever,
            sources,
            total_hits: response.retrieval.total_hits,
            stats: response.stats,
            notes,
        })
    }

    /// Build source spans from an existing ask response.
    ///
    /// This is useful when you've already called `ask()` and want to extract
    /// structured source information without re-running the query.
    pub fn build_sources_from_response(
        &mut self,
        response: &crate::AskResponse,
        include_snippets: bool,
    ) -> Result<Vec<SourceSpan>> {
        let mut sources: Vec<SourceSpan> = Vec::new();

        for (idx, citation) in response.citations.iter().enumerate() {
            let hit = response
                .retrieval
                .hits
                .iter()
                .find(|h| h.frame_id == citation.frame_id);

            let frame_data = self.frame_by_id(citation.frame_id).ok();

            let snippet = if include_snippets {
                hit.map(|h| h.chunk_text.clone().unwrap_or_else(|| h.text.clone()))
            } else {
                None
            };

            let source = SourceSpan {
                index: idx + 1,
                frame_id: citation.frame_id,
                uri: citation.uri.clone(),
                title: frame_data.as_ref().and_then(|f| f.title.clone()),
                chunk_range: citation.chunk_range,
                score: citation.score,
                tags: frame_data
                    .as_ref()
                    .map(|f| f.tags.clone())
                    .unwrap_or_default(),
                labels: frame_data
                    .as_ref()
                    .map(|f| f.labels.clone())
                    .unwrap_or_default(),
                frame_timestamp: frame_data.as_ref().map(|f| f.timestamp),
                content_dates: frame_data
                    .as_ref()
                    .map(|f| f.content_dates.clone())
                    .unwrap_or_default(),
                snippet,
            };

            sources.push(source);
        }

        Ok(sources)
    }
}

#[cfg(not(feature = "lex"))]
impl Memvid {
    pub fn audit<E>(
        &mut self,
        _question: &str,
        _options: Option<AuditOptions>,
        _embedder: Option<&E>,
    ) -> Result<AuditReport>
    where
        E: VecEmbedder + ?Sized,
    {
        Err(crate::MemvidError::LexNotEnabled)
    }

    pub fn build_sources_from_response(
        &mut self,
        _response: &crate::AskResponse,
        _include_snippets: bool,
    ) -> Result<Vec<SourceSpan>> {
        Err(crate::MemvidError::LexNotEnabled)
    }
}

// Convert AskMode to AskRetriever for comparison
impl From<AskMode> for crate::AskRetriever {
    fn from(mode: AskMode) -> Self {
        match mode {
            AskMode::Lex => crate::AskRetriever::Lex,
            AskMode::Sem => crate::AskRetriever::Semantic,
            AskMode::Hybrid => crate::AskRetriever::Hybrid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PutOptions, run_serial_test};
    use tempfile::tempdir;

    #[test]
    #[cfg(feature = "lex")]
    fn test_audit_basic() {
        run_serial_test(|| {
            let dir = tempdir().expect("tmp");
            let path = dir.path().join("audit.mv2");

            let mut mem = Memvid::create(&path).expect("create");
            mem.enable_lex().expect("enable lex");

            let options = PutOptions::builder()
                .uri("mv2://docs/intro.md")
                .title("Introduction to Memvid")
                .push_tag("documentation")
                .push_tag("intro")
                .build();

            mem.put_bytes_with_options(
                b"Memvid is a single-file AI memory system that provides instant retrieval.",
                options,
            )
            .expect("put");
            mem.commit().expect("commit");

            let report = mem
                .audit::<dyn VecEmbedder>("What is Memvid?", None, None)
                .expect("audit");

            assert_eq!(report.version, "1.0");
            assert_eq!(report.question, "What is Memvid?");
            assert!(!report.sources.is_empty());

            let source = &report.sources[0];
            assert_eq!(source.uri, "mv2://docs/intro.md");
            assert_eq!(source.title.as_deref(), Some("Introduction to Memvid"));
            assert!(source.tags.contains(&"documentation".to_string()));

            // Test text output
            let text = report.to_text();
            assert!(text.contains("MEMVID AUDIT REPORT"));
            assert!(text.contains("What is Memvid?"));

            // Test markdown output
            let md = report.to_markdown();
            assert!(md.contains("# Memvid Audit Report"));
            assert!(md.contains("mv2://docs/intro.md"));
        });
    }

    #[test]
    #[cfg(feature = "lex")]
    fn test_audit_with_snippets() {
        run_serial_test(|| {
            let dir = tempdir().expect("tmp");
            let path = dir.path().join("audit_snippets.mv2");

            let mut mem = Memvid::create(&path).expect("create");
            mem.enable_lex().expect("enable lex");

            mem.put_bytes(b"Memvid provides fast semantic search capabilities.")
                .expect("put");
            mem.commit().expect("commit");

            let opts = AuditOptions {
                include_snippets: true,
                ..Default::default()
            };

            let report = mem
                .audit::<dyn VecEmbedder>("semantic search", Some(opts), None)
                .expect("audit");

            if !report.sources.is_empty() {
                let source = &report.sources[0];
                assert!(source.snippet.is_some());
            }
        });
    }
}
