//! Core `Memvid` type orchestrating `.mv2` lifecycle and mutations.

pub mod ask;
pub mod audit;
#[cfg(feature = "parallel_segments")]
pub mod builder;
pub mod chunks;
pub mod doctor;
pub mod enrichment;
pub mod frame;
mod helpers;
pub mod lifecycle;
pub mod maintenance;
pub mod memory;
pub mod mesh;
pub mod mutation;
#[cfg(feature = "parallel_segments")]
pub mod planner;
#[cfg(feature = "replay")]
pub mod replay_ops;
pub mod search;
mod segments;
pub mod sketch;
pub mod tags;
pub mod ticket;
pub mod timeline;
#[cfg(feature = "parallel_segments")]
pub mod workers;

#[cfg(feature = "parallel_segments")]
pub use builder::{BuildOpts, ParallelInput, ParallelPayload};
pub use enrichment::{
    EnrichmentHandle, EnrichmentStats, start_enrichment_worker,
    start_enrichment_worker_with_embeddings,
};
pub use frame::BlobReader;
pub use lifecycle::{LockSettings, Memvid, OpenReadOptions};
pub use sketch::{SketchCandidate, SketchSearchOptions, SketchSearchStats};
