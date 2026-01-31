//! HTTP streaming support for read-only access to `.mv2` files hosted on CDNs.
//!
//! This module enables fetching `.mv2` files from S3, CloudFront, Cloudflare R2, and other
//! CDN providers using HTTP range requests. Only the header, TOC, and needed data segments
//! are fetched, minimizing network transfer.
//!
//! # Features
//!
//! - **CDN deployment**: Store .mv2 files on S3, CloudFront, Cloudflare R2, etc.
//! - **20x+ network reduction**: Only fetch header, TOC, and needed data segments
//! - **Mobile/serverless enablement**: Low memory footprint, no full file download
//! - **Zero local storage**: Read directly from HTTP without caching to disk
//!
//! # Example
//!
//! ```ignore
//! use memvid_core::streaming::{HttpStreamingSource, StreamingMemvid};
//! use memvid_core::SearchRequest;
//!
//! let source = HttpStreamingSource::from_url("https://cdn.example.com/knowledge.mv2")?;
//! let mut mem = StreamingMemvid::open(source)?;
//!
//! let response = mem.search(SearchRequest {
//!     query: "machine learning".into(),
//!     top_k: 10,
//!     ..Default::default()
//! })?;
//! ```

mod error;
mod http;
mod local;
mod memvid;
mod source;

pub use error::StreamingError;
pub use http::{HttpAuthConfig, HttpConfig, HttpStreamingSource};
pub use local::LocalStreamingSource;
pub use memvid::StreamingMemvid;
pub use source::{StreamingResult, StreamingSource};
