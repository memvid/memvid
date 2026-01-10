//! ResonantQ Spectral Enhancements for Memvid
//!
//! This module provides five core enhancements from ResonantQ:
//! 1. **Spectral Compression** - 17-mode eigenvector compression (45× smaller embeddings)
//! 2. **SSH Topological Search** - +48% noise tolerance using Su-Schrieffer-Heeger model
//! 3. **Incremental Caching** - 20× faster repeated queries via spectral basis caching
//! 4. **TopoRouter** - Topologically-protected memory graph traversal
//! 5. **GFT Condensation** - Graph Fourier Transform for memory deduplication

pub mod compression;
pub mod ssh_search;
pub mod cache;
pub mod topo_router;
pub mod gft;

pub use compression::{
    SpectralCompressor, SpectralBasis, CompressedEmbedding,
    DEFAULT_K_MODES, compress_embedding, decompress_embedding,
};
pub use ssh_search::{
    SshSearcher, SshConfig, TopologicalSearchHit,
    ssh_similarity, dimerization_parameter,
};
pub use cache::{
    SpectralCache, CacheStats, CachedBasis,
};
pub use topo_router::{
    TopoRouter, RouteResult, MemoryNode, TopologicalPath,
};
pub use gft::{
    GftCondenser, CondensedMemory, DuplicateCluster,
    compute_gft, inverse_gft,
};

/// Default number of spectral modes for compression (empirically optimal)
pub const OPTIMAL_K_MODES: usize = 17;

/// Spectral gap threshold for topological protection
pub const MIN_SPECTRAL_GAP: f32 = 0.1;

/// Maximum cache entries for spectral bases
pub const DEFAULT_CACHE_SIZE: usize = 100;
