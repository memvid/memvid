//! offline ndcg eval harness for measuring search ranking quality.
//!
//! runs each query from a ground truth fixture against mem.search(),
//! computes ndcg@5 and ndcg@10, and produces an aggregate report.
//!
//! pr2 will add centralised reporting — the report struct is already
//! shaped for that (version, timestamp, blake3-hashed queries).

pub mod ground_truth;
pub mod ndcg;

use std::time::{SystemTime, UNIX_EPOCH};

pub use ground_truth::{EvalCase, EvalSuite};
pub use ndcg::compute_ndcg;

use crate::types::{SearchEngineKind, SearchRequest};
use crate::{Memvid, Result};

/// result for a single eval query
#[derive(Debug, Clone)]
pub struct QueryEvalResult {
    pub query: String,
    /// blake3 hex — the only id sent in pr2 (raw query stays local)
    pub query_hash: String,
    pub ndcg_at_5: f32,
    pub ndcg_at_10: f32,
    pub hit_count: usize,
    pub elapsed_ms: u128,
    pub engine: SearchEngineKind,
    pub sketch_active: bool,
}

/// aggregate eval report — shaped for future centralised reporting in pr2
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub memvid_version: &'static str,
    pub timestamp: i64,
    pub mean_ndcg_at_5: f32,
    pub mean_ndcg_at_10: f32,
    pub cases_with_hits: usize,
    pub total_cases: usize,
    /// per-query detail (stays local, not sent in pr2)
    pub per_query: Vec<QueryEvalResult>,
}

impl EvalReport {
    pub fn print_summary(&self) {
        println!("┌─────────────────────────────────────────────┐");
        println!(
            "│  Memvid NDCG Eval — v{:<23} │",
            self.memvid_version
        );
        println!(
            "│  Cases: {:<3}  Hits: {}/{}{}│",
            self.total_cases,
            self.cases_with_hits,
            self.total_cases,
            " ".repeat(18usize.saturating_sub(
                format!("{}/{}", self.cases_with_hits, self.total_cases).len()
            ))
        );
        println!("├───────────────────┬──────────┬──────────────┤");
        println!("│ Metric            │  Score   │  Grade       │");
        println!("├───────────────────┼──────────┼──────────────┤");
        println!(
            "│ Mean NDCG@5       │  {:.4}  │  {:<12} │",
            self.mean_ndcg_at_5,
            grade(self.mean_ndcg_at_5)
        );
        println!(
            "│ Mean NDCG@10      │  {:.4}  │  {:<12} │",
            self.mean_ndcg_at_10,
            grade(self.mean_ndcg_at_10)
        );
        println!("└───────────────────┴──────────┴──────────────┘");
    }

    /// worst-first per-query breakdown
    pub fn print_per_query(&self) {
        let mut rows = self.per_query.clone();
        rows.sort_by(|a, b| {
            a.ndcg_at_5
                .partial_cmp(&b.ndcg_at_5)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        println!("\n{:<40} {:>8} {:>8}  {:>6}", "Query", "NDCG@5", "NDCG@10", "Hits");
        println!("{}", "─".repeat(70));
        for r in &rows {
            let q = if r.query.len() > 38 {
                format!("{}…", &r.query[..37])
            } else {
                r.query.clone()
            };
            println!(
                "{:<40} {:>8.4} {:>8.4}  {:>6}",
                q, r.ndcg_at_5, r.ndcg_at_10, r.hit_count
            );
        }
    }
}

fn grade(score: f32) -> &'static str {
    match score {
        s if s >= 0.9 => "Excellent",
        s if s >= 0.75 => "Good",
        s if s >= 0.55 => "Fair",
        s if s >= 0.35 => "Poor",
        _ => "Very Poor",
    }
}

/// run every case in the suite against mem.search() and return an aggregate report
#[cfg(feature = "lex")]
pub fn run_eval_suite(mem: &mut Memvid, suite: &EvalSuite) -> Result<EvalReport> {
    let mut per_query: Vec<QueryEvalResult> = Vec::with_capacity(suite.cases.len());

    for case in &suite.cases {
        let request = SearchRequest {
            query: case.query.clone(),
            top_k: 10,
            snippet_chars: 0,
            uri: None,
            scope: None,
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: false,
            acl_context: None,
            acl_enforcement_mode: crate::types::AclEnforcementMode::Audit,
        };

        let (hits, elapsed_ms, engine) = match mem.search(request) {
            Ok(r) => (r.hits, r.elapsed_ms, r.engine),
            Err(_) => (Vec::new(), 0, crate::types::SearchEngineKind::Tantivy),
        };
        let ndcg5 = compute_ndcg(&hits, case, 5);
        let ndcg10 = compute_ndcg(&hits, case, 10);
        let hit_count = hits.len();

        let query_hash = blake3_hex(case.query.as_bytes());

        per_query.push(QueryEvalResult {
            query: case.query.clone(),
            query_hash,
            ndcg_at_5: ndcg5,
            ndcg_at_10: ndcg10,
            hit_count,
            elapsed_ms,
            engine,
            sketch_active: !case.query.is_empty(),
        });
    }

    let n = per_query.len() as f32;
    let mean_ndcg_at_5 = if n > 0.0 {
        per_query.iter().map(|r| r.ndcg_at_5).sum::<f32>() / n
    } else {
        0.0
    };
    let mean_ndcg_at_10 = if n > 0.0 {
        per_query.iter().map(|r| r.ndcg_at_10).sum::<f32>() / n
    } else {
        0.0
    };
    let cases_with_hits = per_query.iter().filter(|r| r.hit_count > 0).count();

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    Ok(EvalReport {
        memvid_version: crate::MEMVID_CORE_VERSION,
        timestamp,
        mean_ndcg_at_5,
        mean_ndcg_at_10,
        cases_with_hits,
        total_cases: per_query.len(),
        per_query,
    })
}

pub(crate) fn blake3_hex(data: &[u8]) -> String {
    let hash = blake3::hash(data);
    hash.to_hex().to_string()
}
