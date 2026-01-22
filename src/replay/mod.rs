// Safe unwrap: fixed-size byte conversions.
#![allow(clippy::unwrap_used)]
//! Time-travel replay for agent sessions.
//!
//! This module provides deterministic recording and replay of agent sessions,
//! enabling debugging, testing, and auditing of AI agent behavior.
//!
//! # Overview
//!
//! The replay system records every action (put, find, ask) performed on a memory
//! file along with checkpoints of the state. Sessions can then be replayed with
//! different parameters (models, search settings) to detect divergences.
//!
//! # Storage
//!
//! Sessions are stored in a dedicated replay segment within the .mv2 file,
//! maintaining single-file portability.
//!
//! # Example
//!
//! ```ignore
//! // Start recording
//! let session_id = memvid.start_session(Some("Debug Session"))?;
//!
//! // Normal operations are recorded
//! memvid.put_bytes(b"test data")?;
//! let hits = memvid.find("test")?;
//!
//! // End recording
//! let session = memvid.end_session()?;
//!
//! // Later, replay with different model
//! let result = memvid.replay(&session, ReplayOptions {
//!     model: Some("gpt-4".into()),
//!     ..Default::default()
//! })?;
//! ```

mod engine;
mod types;

pub use engine::{
    ActionDiff, ActionReplayResult, ReplayEngine, ReplayExecutionConfig,
    ReplayResult as EngineReplayResult, SessionComparison,
};
pub use types::{
    ActionType, Checkpoint, ComparisonReport, ComparisonSummary, Divergence, DivergenceType,
    ModelResult, REPLAY_SEGMENT_MAGIC, REPLAY_SEGMENT_VERSION, ReplayAction, ReplayManifest,
    ReplayOptions, ReplayResult, ReplaySession, SessionSummary, StateSnapshot,
};

use crate::MemvidError;
use crate::error::Result;
use uuid::Uuid;

/// Configuration for replay recording
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ReplayConfig {
    /// Automatically checkpoint every N actions (0 = disabled)
    pub auto_checkpoint_interval: u64,
    /// Maximum actions per session before auto-ending
    pub max_actions_per_session: Option<u64>,
    /// Enable recording by default when opening files
    pub auto_record: bool,
}

/// Active recording state for a session (serializable for persistence)
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ActiveSession {
    /// The session being recorded
    pub session: ReplaySession,
    /// Counter for generating checkpoint IDs
    pub next_checkpoint_id: u64,
    /// Actions since last checkpoint
    pub actions_since_checkpoint: u64,
    /// Configuration for this recording
    pub config: ReplayConfig,
}

impl ActiveSession {
    /// Create a new active session
    #[must_use]
    pub fn new(name: Option<String>, config: ReplayConfig) -> Self {
        Self {
            session: ReplaySession::new(name),
            next_checkpoint_id: 0,
            actions_since_checkpoint: 0,
            config,
        }
    }

    /// Record an action
    pub fn record_action(&mut self, action: ReplayAction) {
        self.session.add_action(action);
        self.actions_since_checkpoint += 1;
    }

    /// Check if auto-checkpoint is due
    #[must_use]
    pub fn should_checkpoint(&self) -> bool {
        self.config.auto_checkpoint_interval > 0
            && self.actions_since_checkpoint >= self.config.auto_checkpoint_interval
    }

    /// Create a checkpoint
    pub fn create_checkpoint(&mut self, snapshot: StateSnapshot) -> Checkpoint {
        let checkpoint = Checkpoint::new(
            self.next_checkpoint_id,
            self.session.next_sequence().saturating_sub(1),
            snapshot,
        );
        self.session.add_checkpoint(checkpoint.clone());
        self.next_checkpoint_id += 1;
        self.actions_since_checkpoint = 0;
        checkpoint
    }

    /// End the session and return it
    pub fn end(mut self) -> ReplaySession {
        self.session.end();
        self.session
    }

    /// Get the session ID
    #[must_use]
    pub fn session_id(&self) -> Uuid {
        self.session.session_id
    }
}

/// Storage operations for replay segments
pub mod storage {
    use super::*;
    use bincode::config::{self, Config};
    use std::io::{Read, Write};

    fn bincode_config() -> impl Config {
        config::standard()
            .with_fixed_int_encoding()
            .with_little_endian()
    }

    /// Header for the replay segment
    #[derive(Debug)]
    pub struct ReplaySegmentHeader {
        pub magic: [u8; 8],
        pub version: u32,
        pub session_count: u32,
        pub total_size: u64,
    }

    impl ReplaySegmentHeader {
        /// Create a new header
        #[must_use]
        pub fn new(session_count: u32, total_size: u64) -> Self {
            Self {
                magic: *REPLAY_SEGMENT_MAGIC,
                version: REPLAY_SEGMENT_VERSION,
                session_count,
                total_size,
            }
        }

        /// Write the header to a writer
        pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
            writer.write_all(&self.magic)?;
            writer.write_all(&self.version.to_le_bytes())?;
            writer.write_all(&self.session_count.to_le_bytes())?;
            writer.write_all(&self.total_size.to_le_bytes())?;
            Ok(())
        }

        /// Read the header from a reader
        pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
            let mut magic = [0u8; 8];
            reader.read_exact(&mut magic)?;
            if &magic != REPLAY_SEGMENT_MAGIC {
                return Err(MemvidError::InvalidToc {
                    reason: "Invalid replay segment magic".into(),
                });
            }

            let mut version_bytes = [0u8; 4];
            reader.read_exact(&mut version_bytes)?;
            let version = u32::from_le_bytes(version_bytes);

            let mut session_count_bytes = [0u8; 4];
            reader.read_exact(&mut session_count_bytes)?;
            let session_count = u32::from_le_bytes(session_count_bytes);

            let mut total_size_bytes = [0u8; 8];
            reader.read_exact(&mut total_size_bytes)?;
            let total_size = u64::from_le_bytes(total_size_bytes);

            Ok(Self {
                magic,
                version,
                session_count,
                total_size,
            })
        }

        /// Size of the header in bytes
        pub const SIZE: usize = 8 + 4 + 4 + 8; // magic + version + session_count + total_size
    }

    /// Serialize a session to bytes
    pub fn serialize_session(session: &ReplaySession) -> Result<Vec<u8>> {
        bincode::serde::encode_to_vec(session, bincode_config()).map_err(|e| {
            MemvidError::InvalidToc {
                reason: format!("Failed to serialize replay session: {}", e).into(),
            }
        })
    }

    /// Deserialize a session from bytes
    pub fn deserialize_session(data: &[u8]) -> Result<ReplaySession> {
        bincode::serde::decode_from_slice(data, bincode_config())
            .map(|(session, _)| session)
            .map_err(|e| MemvidError::InvalidToc {
                reason: format!("Failed to deserialize replay session: {}", e).into(),
            })
    }

    /// Build a complete replay segment from sessions
    pub fn build_segment(sessions: &[ReplaySession]) -> Result<Vec<u8>> {
        let mut session_data: Vec<Vec<u8>> = Vec::with_capacity(sessions.len());
        let mut total_session_bytes = 0u64;

        for session in sessions {
            let data = serialize_session(session)?;
            total_session_bytes += data.len() as u64 + 8; // +8 for length prefix
            session_data.push(data);
        }

        let header = ReplaySegmentHeader::new(
            sessions.len() as u32,
            ReplaySegmentHeader::SIZE as u64 + total_session_bytes,
        );

        let mut segment = Vec::with_capacity(header.total_size as usize);
        header.write(&mut segment)?;

        // Write each session with length prefix
        for data in session_data {
            segment.extend_from_slice(&(data.len() as u64).to_le_bytes());
            segment.extend_from_slice(&data);
        }

        Ok(segment)
    }

    /// Read sessions from a replay segment
    pub fn read_segment(data: &[u8]) -> Result<Vec<ReplaySession>> {
        let mut cursor = std::io::Cursor::new(data);
        let header = ReplaySegmentHeader::read(&mut cursor)?;

        let mut sessions = Vec::with_capacity(header.session_count as usize);
        for _ in 0..header.session_count {
            let mut len_bytes = [0u8; 8];
            cursor.read_exact(&mut len_bytes)?;
            let len = u64::from_le_bytes(len_bytes) as usize;

            let mut session_data = vec![0u8; len];
            cursor.read_exact(&mut session_data)?;

            let session = deserialize_session(&session_data)?;
            sessions.push(session);
        }

        Ok(sessions)
    }

    /// Magic bytes for active session marker file
    pub const ACTIVE_SESSION_MAGIC: &[u8; 8] = b"MV2ACTIV";

    /// Serialize an active session to bytes
    pub fn serialize_active_session(session: &super::ActiveSession) -> Result<Vec<u8>> {
        let mut data = Vec::new();
        data.extend_from_slice(ACTIVE_SESSION_MAGIC);
        let session_bytes =
            bincode::serde::encode_to_vec(session, bincode_config()).map_err(|e| {
                MemvidError::InvalidToc {
                    reason: format!("Failed to serialize active session: {}", e).into(),
                }
            })?;
        data.extend_from_slice(&(session_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(&session_bytes);
        Ok(data)
    }

    /// Deserialize an active session from bytes
    pub fn deserialize_active_session(data: &[u8]) -> Result<super::ActiveSession> {
        if data.len() < 16 {
            return Err(MemvidError::InvalidToc {
                reason: "Active session data too short".into(),
            });
        }
        if &data[0..8] != ACTIVE_SESSION_MAGIC {
            return Err(MemvidError::InvalidToc {
                reason: "Invalid active session magic".into(),
            });
        }
        let len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        if data.len() < 16 + len {
            return Err(MemvidError::InvalidToc {
                reason: "Active session data truncated".into(),
            });
        }
        bincode::serde::decode_from_slice(&data[16..16 + len], bincode_config())
            .map(|(session, _)| session)
            .map_err(|e| MemvidError::InvalidToc {
                reason: format!("Failed to deserialize active session: {}", e).into(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_session() {
        let mut active = ActiveSession::new(
            Some("Test".to_string()),
            ReplayConfig {
                auto_checkpoint_interval: 2,
                ..Default::default()
            },
        );

        assert!(!active.should_checkpoint());

        active.record_action(ReplayAction::new(0, ActionType::Put { frame_id: 1 }));
        assert!(!active.should_checkpoint());

        active.record_action(ReplayAction::new(1, ActionType::Put { frame_id: 2 }));
        assert!(active.should_checkpoint());

        let checkpoint = active.create_checkpoint(StateSnapshot::default());
        assert_eq!(checkpoint.id, 0);
        assert!(!active.should_checkpoint());

        let session = active.end();
        assert!(!session.is_recording());
        assert_eq!(session.actions.len(), 2);
        assert_eq!(session.checkpoints.len(), 1);
    }

    #[test]
    fn test_segment_roundtrip() {
        let mut session1 = ReplaySession::new(Some("Session 1".to_string()));
        session1.add_action(ReplayAction::new(0, ActionType::Put { frame_id: 1 }));
        session1.end();

        let mut session2 = ReplaySession::new(Some("Session 2".to_string()));
        session2.add_action(ReplayAction::new(
            0,
            ActionType::Find {
                query: "test".into(),
                mode: "lexical".into(),
                result_count: 5,
            },
        ));
        session2.end();

        let segment = storage::build_segment(&[session1.clone(), session2.clone()]).unwrap();
        let restored = storage::read_segment(&segment).unwrap();

        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].session_id, session1.session_id);
        assert_eq!(restored[1].session_id, session2.session_id);
    }
}
