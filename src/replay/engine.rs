//! Replay execution engine for time-travel debugging.
//!
//! The replay engine can execute recorded sessions deterministically,
//! compare results with original recordings, and support checkpoint-based
//! partial replay.

use super::types::{ActionType, ReplaySession};
use crate::MemvidError;
use crate::error::Result;
use crate::memvid::lifecycle::Memvid;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use uuid::Uuid;

/// Result of replaying a single action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionReplayResult {
    /// Sequence number of the action
    pub sequence: u64,
    /// Whether the replay matched the original
    pub matched: bool,
    /// Description of any differences
    pub diff: Option<String>,
    /// Duration of the replay in milliseconds
    pub duration_ms: u64,
    /// Original action type
    pub action_type: String,
}

/// Summary of a full session replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResult {
    /// Session that was replayed
    pub session_id: Uuid,
    /// Total actions replayed
    pub total_actions: usize,
    /// Actions that matched their recorded state
    pub matched_actions: usize,
    /// Actions that differed from recorded state
    pub mismatched_actions: usize,
    /// Actions that were skipped
    pub skipped_actions: usize,
    /// Detailed results per action
    pub action_results: Vec<ActionReplayResult>,
    /// Total replay duration in milliseconds
    pub total_duration_ms: u64,
    /// Checkpoint used as starting point (if any)
    pub from_checkpoint: Option<u64>,
}

impl ReplayResult {
    /// Check if the replay was successful (all actions matched).
    pub fn is_success(&self) -> bool {
        self.mismatched_actions == 0
    }

    /// Get the match rate as a percentage.
    pub fn match_rate(&self) -> f64 {
        if self.total_actions == 0 {
            100.0
        } else {
            (self.matched_actions as f64 / self.total_actions as f64) * 100.0
        }
    }
}

/// Configuration for replay execution.
#[derive(Debug, Clone)]
pub struct ReplayExecutionConfig {
    /// Skip put actions (useful for read-only replay)
    pub skip_puts: bool,
    /// Skip find actions
    pub skip_finds: bool,
    /// Skip ask actions (useful when LLM not available)
    pub skip_asks: bool,
    /// Stop on first mismatch
    pub stop_on_mismatch: bool,
    /// Verbose logging
    pub verbose: bool,
    /// Override top-k for find actions (None = use original values)
    /// Higher values reveal documents that may have been missed
    pub top_k: Option<usize>,
    /// Use adaptive retrieval based on score distribution
    pub adaptive: bool,
    /// Minimum relevancy score for adaptive mode (0.0-1.0)
    pub min_relevancy: f32,
    /// AUDIT MODE: Use frozen retrieval for ASK actions
    /// When true, ASK actions use recorded frame IDs instead of re-executing search
    pub audit_mode: bool,
    /// Override model for audit replay (format: "provider:model")
    /// When set, re-executes LLM call with frozen context using this model
    pub use_model: Option<String>,
    /// Generate diff report comparing original vs new answers
    pub generate_diff: bool,
}

impl Default for ReplayExecutionConfig {
    fn default() -> Self {
        Self {
            skip_puts: false,
            skip_finds: false,
            skip_asks: false,
            stop_on_mismatch: false,
            verbose: false,
            top_k: None,
            adaptive: false,
            min_relevancy: 0.5,
            audit_mode: false,
            use_model: None,
            generate_diff: false,
        }
    }
}

/// The replay engine executes recorded sessions.
pub struct ReplayEngine<'a> {
    /// The memory file to replay against
    mem: &'a mut Memvid,
    /// Configuration for replay
    config: ReplayExecutionConfig,
}

impl<'a> ReplayEngine<'a> {
    /// Create a new replay engine.
    pub fn new(mem: &'a mut Memvid, config: ReplayExecutionConfig) -> Self {
        Self { mem, config }
    }

    /// Replay a full session from the beginning.
    pub fn replay_session(&mut self, session: &ReplaySession) -> Result<ReplayResult> {
        self.replay_session_from(session, None)
    }

    /// Replay a session starting from a specific checkpoint.
    pub fn replay_session_from(
        &mut self,
        session: &ReplaySession,
        from_checkpoint: Option<u64>,
    ) -> Result<ReplayResult> {
        let start_time = Instant::now();
        let mut result = ReplayResult {
            session_id: session.session_id,
            total_actions: 0,
            matched_actions: 0,
            mismatched_actions: 0,
            skipped_actions: 0,
            action_results: Vec::new(),
            total_duration_ms: 0,
            from_checkpoint,
        };

        // Determine starting sequence
        let start_sequence = if let Some(checkpoint_id) = from_checkpoint {
            let checkpoint = session
                .checkpoints
                .iter()
                .find(|c| c.id == checkpoint_id)
                .ok_or_else(|| MemvidError::InvalidQuery {
                    reason: format!("Checkpoint {} not found in session", checkpoint_id),
                })?;
            checkpoint.at_sequence
        } else {
            0
        };

        // Filter actions to replay
        let actions_to_replay: Vec<_> = session
            .actions
            .iter()
            .filter(|a| a.sequence >= start_sequence)
            .collect();

        result.total_actions = actions_to_replay.len();

        for action in actions_to_replay {
            let action_start = Instant::now();
            let mut action_result = ActionReplayResult {
                sequence: action.sequence,
                matched: false,
                diff: None,
                duration_ms: 0,
                action_type: action.action_type.name().to_string(),
            };

            match &action.action_type {
                ActionType::Put { frame_id } => {
                    if self.config.skip_puts {
                        result.skipped_actions += 1;
                        action_result.diff = Some("skipped".to_string());
                    } else {
                        // Put actions can't be replayed deterministically (they create new frame IDs)
                        // The frame_id recorded is the WAL sequence, not the frame index
                        // Just verify that frames exist (we can't verify the exact ID)
                        let frame_count = self.mem.toc.frames.len();
                        if frame_count > 0 {
                            action_result.matched = true;
                            action_result.diff = Some(format!(
                                "Put verified (seq {}, {} frames total)",
                                frame_id, frame_count
                            ));
                            result.matched_actions += 1;
                        } else {
                            action_result.matched = false;
                            action_result.diff = Some("No frames found".to_string());
                            result.mismatched_actions += 1;
                        }
                    }
                }

                ActionType::Find {
                    query,
                    mode: _,
                    result_count,
                } => {
                    if self.config.skip_finds {
                        result.skipped_actions += 1;
                        action_result.diff = Some("skipped".to_string());
                    } else {
                        // Determine the top_k to use:
                        // 1. Use config override if specified (for time-travel analysis)
                        // 2. Otherwise use the original value to verify consistency
                        let replay_top_k = self.config.top_k.unwrap_or(*result_count);

                        // Re-execute the search using the search() API which handles
                        // both lex-only and hybrid search modes
                        let search_request = crate::types::SearchRequest {
                            query: query.clone(),
                            top_k: replay_top_k,
                            snippet_chars: 120,
                            uri: None,
                            frames: Vec::new(),
                            scope: None,
                            cursor: None,
                            #[cfg(feature = "temporal_track")]
                            temporal: None,
                            as_of_frame: None,
                            as_of_ts: None,
                            no_sketch: false,
                        };
                        match self.mem.search(search_request) {
                            Ok(response) => {
                                let replay_count = if self.config.adaptive {
                                    // Adaptive mode: count only results above min_relevancy
                                    response
                                        .hits
                                        .iter()
                                        .filter(|h| {
                                            h.score.unwrap_or(0.0) >= self.config.min_relevancy
                                        })
                                        .count()
                                } else {
                                    response.hits.len()
                                };

                                // If we're using a custom top_k, show analysis instead of mismatch
                                if self.config.top_k.is_some() && replay_count != *result_count {
                                    // Build document details string - always show what was found
                                    let mut doc_details = String::new();

                                    if replay_count > *result_count {
                                        // Discovery UP: replay found more docs (higher top-k reveals missed docs)
                                        let extra_count = replay_count - *result_count;
                                        doc_details.push_str(
                                            "\n    Documents discovered with higher top-k:",
                                        );
                                        for (i, hit) in response.hits.iter().enumerate() {
                                            let score = hit.score.unwrap_or(0.0);
                                            let uri = &hit.uri;
                                            let marker =
                                                if i >= *result_count { " [NEW]" } else { "" };
                                            doc_details.push_str(&format!(
                                                "\n      [{}] {} (score: {:.2}){}",
                                                i + 1,
                                                uri,
                                                score,
                                                marker
                                            ));
                                        }
                                        action_result.matched = false;
                                        action_result.diff = Some(format!(
                                            "DISCOVERY: original found {}, replay with top-k={} found {} (+{} docs). Query: \"{}\"{}",
                                            result_count,
                                            replay_top_k,
                                            replay_count,
                                            extra_count,
                                            query,
                                            doc_details
                                        ));
                                    } else {
                                        // Discovery DOWN: replay found fewer docs (lower top-k would miss docs)
                                        let missed_count = *result_count - replay_count;
                                        doc_details.push_str(
                                            "\n    With lower top-k, only these would be found:",
                                        );
                                        for (i, hit) in response.hits.iter().enumerate() {
                                            let score = hit.score.unwrap_or(0.0);
                                            let uri = &hit.uri;
                                            doc_details.push_str(&format!(
                                                "\n      [{}] {} (score: {:.2})",
                                                i + 1,
                                                uri,
                                                score
                                            ));
                                        }
                                        doc_details.push_str(&format!(
                                            "\n    {} document(s) would be MISSED with top-k={}",
                                            missed_count, replay_top_k
                                        ));
                                        action_result.matched = false;
                                        action_result.diff = Some(format!(
                                            "FILTER: original found {}, replay with top-k={} would only find {} (-{} docs). Query: \"{}\"{}",
                                            result_count,
                                            replay_top_k,
                                            replay_count,
                                            missed_count,
                                            query,
                                            doc_details
                                        ));
                                    }

                                    result.mismatched_actions += 1;
                                } else if replay_count == *result_count {
                                    action_result.matched = true;
                                    if self.config.adaptive {
                                        action_result.diff = Some(format!(
                                            "Matched with adaptive (min_relevancy={})",
                                            self.config.min_relevancy
                                        ));
                                    }
                                    result.matched_actions += 1;
                                } else {
                                    action_result.matched = false;
                                    action_result.diff = Some(format!(
                                        "Result count mismatch: expected {}, got {}",
                                        result_count, replay_count
                                    ));
                                    result.mismatched_actions += 1;
                                }
                            }
                            Err(e) => {
                                action_result.matched = false;
                                action_result.diff = Some(format!("Search failed: {}", e));
                                result.mismatched_actions += 1;
                            }
                        }
                    }
                }

                ActionType::Ask {
                    query,
                    provider,
                    model,
                } => {
                    if self.config.skip_asks {
                        result.skipped_actions += 1;
                        action_result.diff = Some("skipped".to_string());
                    } else if self.config.audit_mode {
                        // AUDIT MODE: Frozen retrieval with optional model override
                        let frames_str = if action.affected_frames.is_empty() {
                            "none recorded".to_string()
                        } else {
                            action
                                .affected_frames
                                .iter()
                                .map(|f| f.to_string())
                                .collect::<Vec<_>>()
                                .join(", ")
                        };

                        let original_answer = if action.output_preview.is_empty() {
                            "(no answer recorded)".to_string()
                        } else {
                            action.output_preview.clone()
                        };

                        // Build audit output
                        let mut details = format!(
                            "Question: \"{}\"\n         Mode: AUDIT (frozen retrieval)\n         Original Model: {}:{}\n         Frozen frames: [{}]",
                            query, provider, model, frames_str
                        );

                        // If model override is set, show it (CLI handles actual LLM re-execution)
                        if let Some(ref override_model) = self.config.use_model {
                            details.push_str(&format!(
                                "\n         Override Model: {}",
                                override_model
                            ));
                        }

                        // Show original answer
                        let answer_preview = if original_answer.len() > 200 {
                            format!("{}...", &original_answer[..200])
                        } else {
                            original_answer.clone()
                        };
                        details.push_str(&format!(
                            "\n         Original Answer: \"{}\"",
                            answer_preview
                        ));

                        // In audit mode with frozen frames, we consider it verified
                        if !action.affected_frames.is_empty() {
                            details.push_str("\n         Context: VERIFIED (frames frozen)");
                            action_result.matched = true;
                            result.matched_actions += 1;
                        } else {
                            details.push_str("\n         Context: MISSING (no frames recorded - session recorded before Phase 1)");
                            action_result.matched = false;
                            result.mismatched_actions += 1;
                        }

                        action_result.diff = Some(details);
                    } else {
                        // Display the recorded ask action details
                        // LLM responses vary, so we show the original rather than re-executing
                        let frames_str = if action.affected_frames.is_empty() {
                            "none recorded".to_string()
                        } else {
                            action
                                .affected_frames
                                .iter()
                                .map(|f| f.to_string())
                                .collect::<Vec<_>>()
                                .join(", ")
                        };

                        let answer_preview = if action.output_preview.is_empty() {
                            "(no answer recorded)".to_string()
                        } else {
                            // Truncate long answers for display
                            let preview = &action.output_preview;
                            if preview.len() > 200 {
                                format!("{}...", &preview[..200])
                            } else {
                                preview.clone()
                            }
                        };

                        // Build detailed output
                        let details = format!(
                            "Question: \"{}\"\n         Model: {}:{}\n         Retrieved frames: [{}]\n         Answer: \"{}\"",
                            query, provider, model, frames_str, answer_preview
                        );

                        action_result.matched = true;
                        action_result.diff = Some(details);
                        result.matched_actions += 1;
                    }
                }

                ActionType::Checkpoint { checkpoint_id } => {
                    // Checkpoints don't need replay, just verification
                    action_result.matched = true;
                    action_result.diff = Some(format!("Checkpoint {} verified", checkpoint_id));
                    result.matched_actions += 1;
                }

                ActionType::PutMany { frame_ids, count } => {
                    if self.config.skip_puts {
                        result.skipped_actions += 1;
                        action_result.diff = Some("skipped".to_string());
                    } else {
                        // Verify all frames exist
                        let existing: Vec<_> = frame_ids
                            .iter()
                            .filter(|id| self.mem.frame_by_id(**id).is_ok())
                            .collect();
                        if existing.len() == *count {
                            action_result.matched = true;
                            result.matched_actions += 1;
                        } else {
                            action_result.matched = false;
                            action_result.diff = Some(format!(
                                "Expected {} frames, found {}",
                                count,
                                existing.len()
                            ));
                            result.mismatched_actions += 1;
                        }
                    }
                }

                ActionType::Update { frame_id } => {
                    if self.config.skip_puts {
                        result.skipped_actions += 1;
                        action_result.diff = Some("skipped".to_string());
                    } else {
                        // Verify the frame exists (update would have modified it)
                        if self.mem.frame_by_id(*frame_id).is_ok() {
                            action_result.matched = true;
                            result.matched_actions += 1;
                        } else {
                            action_result.matched = false;
                            action_result.diff = Some(format!("Frame {} not found", frame_id));
                            result.mismatched_actions += 1;
                        }
                    }
                }

                ActionType::Delete { frame_id } => {
                    if self.config.skip_puts {
                        result.skipped_actions += 1;
                        action_result.diff = Some("skipped".to_string());
                    } else {
                        // Verify the frame is deleted (should not exist)
                        if self.mem.frame_by_id(*frame_id).is_err() {
                            action_result.matched = true;
                            result.matched_actions += 1;
                        } else {
                            action_result.matched = false;
                            action_result.diff = Some(format!("Frame {} still exists", frame_id));
                            result.mismatched_actions += 1;
                        }
                    }
                }

                ActionType::ToolCall { name, args_hash: _ } => {
                    // Tool calls can't be replayed deterministically
                    result.skipped_actions += 1;
                    action_result.diff = Some(format!("Tool call '{}' skipped", name));
                }
            }

            action_result.duration_ms = action_start.elapsed().as_millis() as u64;
            result.action_results.push(action_result);

            // Stop on mismatch if configured
            if self.config.stop_on_mismatch
                && result.mismatched_actions > 0
                && result.action_results.last().map_or(false, |r| !r.matched)
            {
                break;
            }
        }

        result.total_duration_ms = start_time.elapsed().as_millis() as u64;

        if self.config.verbose {
            tracing::info!(
                "Replay completed: {}/{} actions matched ({}%)",
                result.matched_actions,
                result.total_actions,
                result.match_rate()
            );
        }

        Ok(result)
    }

    /// Compare two sessions to find differences.
    pub fn compare_sessions(
        session_a: &ReplaySession,
        session_b: &ReplaySession,
    ) -> SessionComparison {
        let mut comparison = SessionComparison {
            session_a_id: session_a.session_id,
            session_b_id: session_b.session_id,
            actions_only_in_a: Vec::new(),
            actions_only_in_b: Vec::new(),
            differing_actions: Vec::new(),
            matching_actions: 0,
        };

        // Create maps for faster lookup
        let a_actions: std::collections::HashMap<_, _> =
            session_a.actions.iter().map(|a| (a.sequence, a)).collect();
        let b_actions: std::collections::HashMap<_, _> =
            session_b.actions.iter().map(|a| (a.sequence, a)).collect();

        // Find actions only in A
        for (seq, _action) in &a_actions {
            if !b_actions.contains_key(seq) {
                comparison.actions_only_in_a.push(*seq);
            }
        }

        // Find actions only in B
        for (seq, _action) in &b_actions {
            if !a_actions.contains_key(seq) {
                comparison.actions_only_in_b.push(*seq);
            }
        }

        // Compare common actions
        for (seq, action_a) in &a_actions {
            if let Some(action_b) = b_actions.get(seq) {
                if action_a.action_type.name() != action_b.action_type.name() {
                    comparison.differing_actions.push(ActionDiff {
                        sequence: *seq,
                        action_type_a: action_a.action_type.name().to_string(),
                        action_type_b: action_b.action_type.name().to_string(),
                        description: format!(
                            "Action type mismatch: {} vs {}",
                            action_a.action_type.name(),
                            action_b.action_type.name()
                        ),
                    });
                } else {
                    // Same type, check details
                    let same = match (&action_a.action_type, &action_b.action_type) {
                        (ActionType::Put { frame_id: a }, ActionType::Put { frame_id: b }) => {
                            a == b
                        }
                        (
                            ActionType::Find {
                                query: qa,
                                result_count: ra,
                                ..
                            },
                            ActionType::Find {
                                query: qb,
                                result_count: rb,
                                ..
                            },
                        ) => qa == qb && ra == rb,
                        (ActionType::Ask { query: qa, .. }, ActionType::Ask { query: qb, .. }) => {
                            qa == qb
                        }
                        (
                            ActionType::Checkpoint { checkpoint_id: a },
                            ActionType::Checkpoint { checkpoint_id: b },
                        ) => a == b,
                        _ => false,
                    };

                    if same {
                        comparison.matching_actions += 1;
                    } else {
                        comparison.differing_actions.push(ActionDiff {
                            sequence: *seq,
                            action_type_a: action_a.action_type.name().to_string(),
                            action_type_b: action_b.action_type.name().to_string(),
                            description: "Action details differ".to_string(),
                        });
                    }
                }
            }
        }

        comparison
    }
}

/// Comparison result between two sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionComparison {
    /// ID of the first session
    pub session_a_id: Uuid,
    /// ID of the second session
    pub session_b_id: Uuid,
    /// Action sequences only in session A
    pub actions_only_in_a: Vec<u64>,
    /// Action sequences only in session B
    pub actions_only_in_b: Vec<u64>,
    /// Actions that differ between sessions
    pub differing_actions: Vec<ActionDiff>,
    /// Number of matching actions
    pub matching_actions: usize,
}

impl SessionComparison {
    /// Check if the sessions are identical.
    pub fn is_identical(&self) -> bool {
        self.actions_only_in_a.is_empty()
            && self.actions_only_in_b.is_empty()
            && self.differing_actions.is_empty()
    }
}

/// A difference between two actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionDiff {
    /// Sequence number of the action
    pub sequence: u64,
    /// Action type in session A
    pub action_type_a: String,
    /// Action type in session B
    pub action_type_b: String,
    /// Description of the difference
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::types::ReplayAction;

    #[test]
    fn test_replay_result_success() {
        let result = ReplayResult {
            session_id: Uuid::new_v4(),
            total_actions: 10,
            matched_actions: 10,
            mismatched_actions: 0,
            skipped_actions: 0,
            action_results: Vec::new(),
            total_duration_ms: 100,
            from_checkpoint: None,
        };
        assert!(result.is_success());
        assert_eq!(result.match_rate(), 100.0);
    }

    #[test]
    fn test_replay_result_partial() {
        let result = ReplayResult {
            session_id: Uuid::new_v4(),
            total_actions: 10,
            matched_actions: 7,
            mismatched_actions: 3,
            skipped_actions: 0,
            action_results: Vec::new(),
            total_duration_ms: 100,
            from_checkpoint: None,
        };
        assert!(!result.is_success());
        assert_eq!(result.match_rate(), 70.0);
    }

    #[test]
    fn test_session_comparison_identical() {
        use std::collections::HashMap;

        let session_a = ReplaySession {
            session_id: Uuid::new_v4(),
            name: Some("A".to_string()),
            created_secs: 0,
            ended_secs: Some(100),
            actions: vec![ReplayAction::new(
                0,
                ActionType::Find {
                    query: "test".to_string(),
                    mode: "lex".to_string(),
                    result_count: 5,
                },
            )],
            checkpoints: Vec::new(),
            metadata: HashMap::new(),
            version: 1,
        };

        let session_b = ReplaySession {
            session_id: Uuid::new_v4(),
            name: Some("B".to_string()),
            created_secs: 0,
            ended_secs: Some(100),
            actions: vec![ReplayAction::new(
                0,
                ActionType::Find {
                    query: "test".to_string(),
                    mode: "lex".to_string(),
                    result_count: 5,
                },
            )],
            checkpoints: Vec::new(),
            metadata: HashMap::new(),
            version: 1,
        };

        let comparison = ReplayEngine::compare_sessions(&session_a, &session_b);
        assert!(comparison.is_identical());
        assert_eq!(comparison.matching_actions, 1);
    }
}
