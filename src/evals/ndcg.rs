//! pure ndcg computation — no io, no deps.
//!
//! ndcg measures ranked retrieval quality (0–1, higher = better).
//! rewards both relevance grade and rank position, so burying
//! the best result at rank 7 scores worse than having it at rank 1.

use crate::types::SearchHit;

use super::ground_truth::EvalCase;

/// ndcg@k for a list of search hits vs ground truth. matches by uri.
pub fn compute_ndcg(hits: &[SearchHit], case: &EvalCase, k: usize) -> f32 {
    if k == 0 || case.relevant_uris.is_empty() {
        return 0.0;
    }

    let grade_map: std::collections::HashMap<&str, f32> = case
        .relevant_uris
        .iter()
        .zip(case.grades.iter())
        .map(|(uri, &grade)| (uri.as_str(), grade))
        .collect();

    let actual_grades: Vec<f32> = hits
        .iter()
        .take(k)
        .map(|hit| *grade_map.get(hit.uri.as_str()).unwrap_or(&0.0))
        .collect();

    let dcg = discounted_cumulative_gain(&actual_grades);

    let mut ideal_grades: Vec<f32> = case.grades.clone();
    ideal_grades.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    ideal_grades.truncate(k);
    let idcg = discounted_cumulative_gain(&ideal_grades);

    if idcg == 0.0 {
        return 0.0;
    }

    (dcg / idcg).min(1.0)
}

/// dcg = sum of grade[i] / log2(rank + 1)
fn discounted_cumulative_gain(grades: &[f32]) -> f32 {
    grades
        .iter()
        .enumerate()
        .map(|(i, &grade)| {
            let rank = (i + 1) as f32; // 1-indexed rank
            grade / (rank + 1.0_f32).log2()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evals::ground_truth::EvalCase;
    use crate::types::SearchHit;

    fn make_hit(uri: &str) -> SearchHit {
        SearchHit {
            rank: 0,
            frame_id: 0,
            uri: uri.to_string(),
            title: None,
            range: (0, 0),
            text: String::new(),
            matches: 0,
            chunk_range: None,
            chunk_text: None,
            score: None,
            metadata: None,
        }
    }

    fn make_case(uris: &[&str], grades: &[f32]) -> EvalCase {
        EvalCase {
            query: "test query".to_string(),
            relevant_uris: uris.iter().map(|s| s.to_string()).collect(),
            grades: grades.to_vec(),
        }
    }

    #[test]
    fn perfect_ranking_scores_1() {
        let hits = vec![make_hit("mv2://docs/wal.md"), make_hit("mv2://docs/other.md")];
        let case = make_case(&["mv2://docs/wal.md"], &[1.0]);
        let score = compute_ndcg(&hits, &case, 5);
        assert!(
            (score - 1.0).abs() < 1e-5,
            "expected 1.0 got {score}"
        );
    }

    #[test]
    fn wrong_ranking_scores_less_than_perfect() {
        let hits = vec![
            make_hit("mv2://docs/secondary.md"),
            make_hit("mv2://docs/primary.md"),
        ];
        let case = make_case(
            &["mv2://docs/primary.md", "mv2://docs/secondary.md"],
            &[1.0, 0.5],
        );
        let score = compute_ndcg(&hits, &case, 5);
        assert!(score < 1.0, "expected score < 1.0 got {score}");
        assert!(score > 0.0, "expected score > 0.0 got {score}");
    }

    #[test]
    fn no_relevant_docs_in_hits_scores_0() {
        let hits = vec![make_hit("mv2://docs/irrelevant.md")];
        let case = make_case(&["mv2://docs/target.md"], &[1.0]);
        let score = compute_ndcg(&hits, &case, 5);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn empty_hits_scores_0() {
        let hits: Vec<SearchHit> = vec![];
        let case = make_case(&["mv2://docs/wal.md"], &[1.0]);
        let score = compute_ndcg(&hits, &case, 5);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn k_zero_scores_0() {
        let hits = vec![make_hit("mv2://docs/wal.md")];
        let case = make_case(&["mv2://docs/wal.md"], &[1.0]);
        let score = compute_ndcg(&hits, &case, 0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn empty_ground_truth_scores_0() {
        let hits = vec![make_hit("mv2://docs/wal.md")];
        let case = make_case(&[], &[]);
        let score = compute_ndcg(&hits, &case, 5);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn graded_relevance_rank1_beats_rank2() {
        let case = make_case(
            &["mv2://docs/primary.md", "mv2://docs/secondary.md"],
            &[1.0, 0.5],
        );

        let ideal_hits = vec![
            make_hit("mv2://docs/primary.md"),
            make_hit("mv2://docs/secondary.md"),
        ];
        let swapped_hits = vec![
            make_hit("mv2://docs/secondary.md"),
            make_hit("mv2://docs/primary.md"),
        ];

        let ideal_score = compute_ndcg(&ideal_hits, &case, 5);
        let swapped_score = compute_ndcg(&swapped_hits, &case, 5);

        assert!(
            ideal_score > swapped_score,
            "ideal {ideal_score} should beat swapped {swapped_score}"
        );
        assert!(
            (ideal_score - 1.0).abs() < 1e-5,
            "ideal ordering should be 1.0, got {ideal_score}"
        );
    }

    #[test]
    fn dcg_discount_decreases_with_rank() {
        let hit_at_rank1 = vec![make_hit("mv2://target.md")];
        let hit_at_rank5 = vec![
            make_hit("mv2://irrelevant1.md"),
            make_hit("mv2://irrelevant2.md"),
            make_hit("mv2://irrelevant3.md"),
            make_hit("mv2://irrelevant4.md"),
            make_hit("mv2://target.md"),
        ];
        let case = make_case(&["mv2://target.md"], &[1.0]);

        let score_rank1 = compute_ndcg(&hit_at_rank1, &case, 5);
        let score_rank5 = compute_ndcg(&hit_at_rank5, &case, 5);

        assert!(
            score_rank1 > score_rank5,
            "rank-1 score {score_rank1} should beat rank-5 score {score_rank5}"
        );
    }
}
