// Safe expect: Static CLIP model lookup with guaranteed default.
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! CLIP (Contrastive Language-Image Pre-training) visual embeddings module.
//!
//! This module provides visual understanding capabilities using MobileCLIP-S2,
//! enabling semantic search across images and PDF pages with natural language queries.
//!
//! # Design Philosophy
//!
//! - **Synchronous with Parallelism**: CLIP runs in parallel with text embedding via rayon.
//!   Since CLIP (~25ms) is faster than text embedding (~200-500ms), it adds zero latency.
//! - **Separate Index**: CLIP embeddings (512 dims) are stored in a separate index from
//!   text embeddings (384/768/1536 dims) because dimensions must match within an index.
//! - **Auto-detection**: Images and PDFs with images are automatically processed without flags.
//! - **Graceful Degradation**: Works without CLIP, just loses visual search capability.

use blake3::hash;
#[cfg(feature = "clip")]
use image::DynamicImage;
#[cfg(all(feature = "clip", not(feature = "pdfium")))]
use image::{ImageBuffer, Luma, Rgb};
#[cfg(all(feature = "clip", not(feature = "pdfium")))]
use lopdf::{Dictionary, Document, Object, ObjectId};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
#[cfg(all(feature = "clip", not(feature = "pdfium")))]
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::{MemvidError, Result, types::FrameId};

// ============================================================================
// Configuration Constants
// ============================================================================

/// CLIP index decode limit (512MB max)
const CLIP_DECODE_LIMIT: usize = crate::MAX_INDEX_BYTES as usize;

/// MobileCLIP-S2 embedding dimensions
pub const MOBILECLIP_DIMS: u32 = 512;

/// SigLIP-base embedding dimensions
pub const SIGLIP_DIMS: u32 = 768;

/// Default input resolution for MobileCLIP-S2
pub const MOBILECLIP_INPUT_SIZE: u32 = 256;

/// Default input resolution for SigLIP
pub const SIGLIP_INPUT_SIZE: u32 = 224;

/// Minimum image dimension to process (skip icons, bullets)
pub const MIN_IMAGE_DIM: u32 = 64;

/// Maximum aspect ratio deviation from 1:1 (skip dividers, lines)
pub const MAX_ASPECT_RATIO: f32 = 10.0;

/// Minimum color variance threshold (skip solid backgrounds)
pub const MIN_COLOR_VARIANCE: f32 = 0.01;

/// Model unload timeout (5 minutes idle)
pub const MODEL_UNLOAD_TIMEOUT: Duration = Duration::from_secs(300);

// ============================================================================
// Bincode Configuration
// ============================================================================

fn clip_config() -> impl bincode::config::Config {
    bincode::config::standard()
        .with_fixed_int_encoding()
        .with_little_endian()
}

// ============================================================================
// Model Registry
// ============================================================================

/// Available CLIP models with verified HuggingFace URLs
#[derive(Debug, Clone)]
pub struct ClipModelInfo {
    /// Model identifier
    pub name: &'static str,
    /// URL for vision encoder ONNX model
    pub vision_url: &'static str,
    /// URL for text encoder ONNX model
    pub text_url: &'static str,
    /// URL for tokenizer JSON (BPE)
    pub tokenizer_url: &'static str,
    /// Vision model size in MB
    pub vision_size_mb: f32,
    /// Text model size in MB
    pub text_size_mb: f32,
    /// Output embedding dimensions
    pub dims: u32,
    /// Input image resolution
    pub input_resolution: u32,
    /// Whether this is the default model
    pub is_default: bool,
}

/// Available CLIP models registry
pub static CLIP_MODELS: &[ClipModelInfo] = &[
    // MobileCLIP-S2 int8 quantized (smallest, but requires INT8 ONNX support)
    // Note: INT8 quantized models don't work on all platforms (ConvInteger not supported)
    ClipModelInfo {
        name: "mobileclip-s2-int8",
        vision_url: "https://huggingface.co/Xenova/mobileclip_s2/resolve/main/onnx/vision_model_int8.onnx",
        text_url: "https://huggingface.co/Xenova/mobileclip_s2/resolve/main/onnx/text_model_int8.onnx",
        tokenizer_url: "https://huggingface.co/Xenova/mobileclip_s2/resolve/main/tokenizer.json",
        vision_size_mb: 36.7,
        text_size_mb: 64.1,
        dims: MOBILECLIP_DIMS,
        input_resolution: MOBILECLIP_INPUT_SIZE,
        is_default: false,
    },
    // Alternative: SigLIP-base quantized (higher quality, but may have INT8 issues)
    ClipModelInfo {
        name: "siglip-base",
        vision_url: "https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/main/onnx/vision_model_quantized.onnx",
        text_url: "https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/main/onnx/text_model_quantized.onnx",
        tokenizer_url: "https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/main/tokenizer.json",
        vision_size_mb: 99.5,
        text_size_mb: 111.0,
        dims: SIGLIP_DIMS,
        input_resolution: SIGLIP_INPUT_SIZE,
        is_default: false,
    },
    // Default: MobileCLIP-S2 fp16 (works on all platforms, good balance of size/quality)
    ClipModelInfo {
        name: "mobileclip-s2",
        vision_url: "https://huggingface.co/Xenova/mobileclip_s2/resolve/main/onnx/vision_model_fp16.onnx",
        text_url: "https://huggingface.co/Xenova/mobileclip_s2/resolve/main/onnx/text_model_fp16.onnx",
        tokenizer_url: "https://huggingface.co/Xenova/mobileclip_s2/resolve/main/tokenizer.json",
        vision_size_mb: 71.7,
        text_size_mb: 127.0,
        dims: MOBILECLIP_DIMS,
        input_resolution: MOBILECLIP_INPUT_SIZE,
        is_default: true,
    },
];

/// Get model info by name, defaults to mobileclip-s2
pub fn get_model_info(name: &str) -> &'static ClipModelInfo {
    CLIP_MODELS
        .iter()
        .find(|m| m.name == name)
        .unwrap_or_else(|| {
            CLIP_MODELS
                .iter()
                .find(|m| m.is_default)
                .expect("default model")
        })
}

/// Get the default model info
pub fn default_model_info() -> &'static ClipModelInfo {
    CLIP_MODELS
        .iter()
        .find(|m| m.is_default)
        .expect("default model exists")
}

// ============================================================================
// CLIP Document and Index Types (mirrors vec.rs pattern)
// ============================================================================

/// A document with CLIP embedding stored in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipDocument {
    /// Frame ID this embedding belongs to
    pub frame_id: FrameId,
    /// CLIP embedding vector (512 or 768 dims depending on model)
    pub embedding: Vec<f32>,
    /// Optional page number (for PDFs)
    #[serde(default)]
    pub page: Option<u32>,
}

/// Builder for constructing CLIP index artifacts
#[derive(Default)]
pub struct ClipIndexBuilder {
    documents: Vec<ClipDocument>,
}

impl ClipIndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a document with its CLIP embedding
    pub fn add_document<I>(&mut self, frame_id: FrameId, page: Option<u32>, embedding: I)
    where
        I: Into<Vec<f32>>,
    {
        self.documents.push(ClipDocument {
            frame_id,
            embedding: embedding.into(),
            page,
        });
    }

    /// Finish building and produce the index artifact
    pub fn finish(self) -> Result<ClipIndexArtifact> {
        let bytes = bincode::serde::encode_to_vec(&self.documents, clip_config())?;

        let checksum = *hash(&bytes).as_bytes();
        let dimension = self
            .documents
            .first()
            .map(|doc| doc.embedding.len() as u32)
            .unwrap_or(0);

        Ok(ClipIndexArtifact {
            bytes,
            vector_count: self.documents.len() as u64,
            dimension,
            checksum,
        })
    }
}

/// Artifact produced by the CLIP index builder
#[derive(Debug, Clone)]
pub struct ClipIndexArtifact {
    /// Serialized index bytes
    pub bytes: Vec<u8>,
    /// Number of vectors in the index
    pub vector_count: u64,
    /// Embedding dimension (512 for MobileCLIP, 768 for SigLIP)
    pub dimension: u32,
    /// Blake3 checksum of the bytes
    pub checksum: [u8; 32],
}

/// In-memory CLIP index for similarity search
#[derive(Debug, Clone)]
pub struct ClipIndex {
    documents: Vec<ClipDocument>,
}

impl ClipIndex {
    /// Create a new empty CLIP index
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
        }
    }

    /// Add a document with its CLIP embedding
    pub fn add_document<I>(&mut self, frame_id: FrameId, page: Option<u32>, embedding: I)
    where
        I: Into<Vec<f32>>,
    {
        self.documents.push(ClipDocument {
            frame_id,
            embedding: embedding.into(),
            page,
        });
    }

    /// Decode CLIP index from bytes
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let (documents, read) = bincode::serde::decode_from_slice::<Vec<ClipDocument>, _>(
            bytes,
            bincode::config::standard()
                .with_fixed_int_encoding()
                .with_little_endian()
                .with_limit::<CLIP_DECODE_LIMIT>(),
        )?;

        if read != bytes.len() {
            return Err(MemvidError::InvalidToc {
                reason: Cow::Owned(format!(
                    "CLIP index decode: expected {} bytes, read {}",
                    bytes.len(),
                    read
                )),
            });
        }

        tracing::debug!(
            bytes_len = bytes.len(),
            docs_count = documents.len(),
            "decoded CLIP index"
        );

        Ok(Self { documents })
    }

    /// Search for similar embeddings using L2 distance
    pub fn search(&self, query: &[f32], limit: usize) -> Vec<ClipSearchHit> {
        if query.is_empty() {
            return Vec::new();
        }

        let mut hits: Vec<ClipSearchHit> = self
            .documents
            .iter()
            .map(|doc| {
                let distance = l2_distance(query, &doc.embedding);
                ClipSearchHit {
                    frame_id: doc.frame_id,
                    page: doc.page,
                    distance,
                }
            })
            .collect();

        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        hits
    }

    /// Get all entries in the index
    pub fn entries(&self) -> impl Iterator<Item = (FrameId, Option<u32>, &[f32])> + '_ {
        self.documents
            .iter()
            .map(|doc| (doc.frame_id, doc.page, doc.embedding.as_slice()))
    }

    /// Get embedding for a specific frame
    pub fn embedding_for(&self, frame_id: FrameId) -> Option<&[f32]> {
        self.documents
            .iter()
            .find(|doc| doc.frame_id == frame_id)
            .map(|doc| doc.embedding.as_slice())
    }

    /// Remove a document from the index
    pub fn remove(&mut self, frame_id: FrameId) {
        self.documents.retain(|doc| doc.frame_id != frame_id);
    }

    /// Number of documents in the index
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Encode the CLIP index to bytes and produce an artifact for persistence
    pub fn encode(&self) -> Result<ClipIndexArtifact> {
        let bytes = bincode::serde::encode_to_vec(&self.documents, clip_config())?;

        let checksum = *hash(&bytes).as_bytes();
        let dimension = self
            .documents
            .first()
            .map(|doc| doc.embedding.len() as u32)
            .unwrap_or(0);

        Ok(ClipIndexArtifact {
            bytes,
            vector_count: self.documents.len() as u64,
            dimension,
            checksum,
        })
    }
}

/// Search result from CLIP index
#[derive(Debug, Clone, PartialEq)]
pub struct ClipSearchHit {
    /// Frame ID of the matched document
    pub frame_id: FrameId,
    /// Optional page number (for PDFs)
    pub page: Option<u32>,
    /// L2 distance to query (lower is more similar)
    pub distance: f32,
}

/// L2 (Euclidean) distance between two vectors
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ============================================================================
// Image Filtering (Junk Detection)
// ============================================================================

/// Metadata about an image for filtering
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub color_variance: f32,
}

impl ImageInfo {
    /// Check if this image should be processed for CLIP embedding
    pub fn should_embed(&self) -> bool {
        // Skip tiny images (icons, bullets)
        if self.width < MIN_IMAGE_DIM || self.height < MIN_IMAGE_DIM {
            return false;
        }

        // Skip extreme aspect ratios (dividers, lines)
        let aspect = self.width as f32 / self.height as f32;
        if aspect > MAX_ASPECT_RATIO || aspect < (1.0 / MAX_ASPECT_RATIO) {
            return false;
        }

        // Skip near-solid colors (backgrounds, spacers)
        if self.color_variance < MIN_COLOR_VARIANCE {
            return false;
        }

        true
    }
}

/// Filter a list of images, keeping only those worth embedding
pub fn filter_junk_images<T, F>(images: Vec<T>, get_info: F) -> Vec<T>
where
    F: Fn(&T) -> ImageInfo,
{
    images
        .into_iter()
        .filter(|img| get_info(img).should_embed())
        .collect()
}

// ============================================================================
// CLIP Model Configuration
// ============================================================================

/// Configuration for CLIP model initialization
#[derive(Debug, Clone)]
pub struct ClipConfig {
    /// Model name (e.g., "mobileclip-s2", "siglip-base")
    pub model_name: String,
    /// Directory where models are cached
    pub models_dir: PathBuf,
    /// Whether to run in offline mode (no downloads)
    pub offline: bool,
}

impl Default for ClipConfig {
    fn default() -> Self {
        // Use ~/.memvid/models as default, consistent with CLI's model installation
        let models_dir = std::env::var("MEMVID_MODELS_DIR")
            .ok()
            .map(PathBuf::from)
            .or_else(|| dirs_next::home_dir().map(|d| d.join(".memvid/models")))
            .unwrap_or_else(|| PathBuf::from(".memvid/models"));

        let model_name =
            std::env::var("MEMVID_CLIP_MODEL").unwrap_or_else(|_| "mobileclip-s2".to_string());

        let offline = std::env::var("MEMVID_OFFLINE").is_ok();

        Self {
            model_name,
            models_dir,
            offline,
        }
    }
}

// ============================================================================
// CLIP Error Types
// ============================================================================

/// CLIP-specific errors
#[derive(Debug, thiserror::Error)]
pub enum ClipError {
    /// Model not found and offline mode enabled
    #[error("CLIP model '{model}' not found. {hint}")]
    ModelNotFound { model: String, hint: String },

    /// Image decode failed
    #[error("Failed to decode image at {path:?}: {cause}")]
    ImageDecodeError { path: PathBuf, cause: String },

    /// Image bytes decode failed
    #[error("Failed to decode image bytes: {cause}")]
    ImageBytesDecodeError { cause: String },

    /// ONNX runtime error
    #[error("CLIP inference error: {cause}")]
    InferenceError { cause: String },

    /// Model download failed
    #[error("Failed to download CLIP model: {cause}")]
    DownloadError { cause: String },

    /// Model file corrupted or invalid
    #[error("CLIP model file is corrupted: {cause}")]
    ModelCorrupted { cause: String },
}

impl From<ClipError> for MemvidError {
    fn from(err: ClipError) -> Self {
        MemvidError::EmbeddingFailed {
            reason: err.to_string().into_boxed_str(),
        }
    }
}

// ============================================================================
// CLIP Model (Feature-gated implementation)
// ============================================================================

#[cfg(feature = "clip")]
mod model {
    use super::*;
    use image::{DynamicImage, GenericImageView, imageops::FilterType};
    use ndarray::{Array, Array4};
    use ort::session::{Session, builder::GraphOptimizationLevel};
    use ort::value::Tensor;
    use std::sync::Mutex;
    use std::time::Instant;
    use tokenizers::{
        PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
        TruncationParams, TruncationStrategy,
    };

    /// CLIP model with lazy-loaded vision and text encoders
    pub struct ClipModel {
        config: ClipConfig,
        model_info: &'static ClipModelInfo,
        /// Lazy-loaded vision encoder session
        vision_session: Mutex<Option<Session>>,
        /// Lazy-loaded text encoder session
        text_session: Mutex<Option<Session>>,
        /// Lazy-loaded tokenizer matching the text encoder
        tokenizer: Mutex<Option<Tokenizer>>,
        /// Last time the model was used (for idle unloading)
        last_used: Mutex<Instant>,
    }

    impl ClipModel {
        /// Create a new CLIP model with the given configuration
        pub fn new(config: ClipConfig) -> Result<Self> {
            let model_info = get_model_info(&config.model_name);

            Ok(Self {
                config,
                model_info,
                vision_session: Mutex::new(None),
                text_session: Mutex::new(None),
                tokenizer: Mutex::new(None),
                last_used: Mutex::new(Instant::now()),
            })
        }

        /// Create with default configuration
        pub fn default_model() -> Result<Self> {
            Self::new(ClipConfig::default())
        }

        /// Get model info
        pub fn model_info(&self) -> &'static ClipModelInfo {
            self.model_info
        }

        /// Get embedding dimensions
        pub fn dims(&self) -> u32 {
            self.model_info.dims
        }

        /// Ensure model file exists, downloading if necessary
        fn ensure_model_file(&self, kind: &str) -> Result<PathBuf> {
            let filename = format!("{}_{}.onnx", self.model_info.name, kind);
            let path = self.config.models_dir.join(&filename);

            if path.exists() {
                return Ok(path);
            }

            if self.config.offline {
                return Err(ClipError::ModelNotFound {
                    model: self.model_info.name.to_string(),
                    hint: format!(
                        "Run: memvid model download {} (or disable MEMVID_OFFLINE)",
                        self.model_info.name
                    ),
                }
                .into());
            }

            // Create models directory if needed
            std::fs::create_dir_all(&self.config.models_dir).map_err(|e| {
                ClipError::DownloadError {
                    cause: format!("Failed to create models directory: {}", e),
                }
            })?;

            // Provide manual download instructions
            Err(ClipError::DownloadError {
                cause: format!(
                    "Automatic download not yet implemented. Please download manually:\n\
                     curl -L '{}' -o '{}'",
                    if kind == "vision" {
                        self.model_info.vision_url
                    } else {
                        self.model_info.text_url
                    },
                    path.display()
                ),
            }
            .into())
        }

        /// Ensure tokenizer file exists, downloading if necessary
        fn ensure_tokenizer_file(&self) -> Result<PathBuf> {
            let filename = format!("{}_tokenizer.json", self.model_info.name);
            let path = self.config.models_dir.join(&filename);

            if path.exists() {
                return Ok(path);
            }

            if self.config.offline {
                return Err(ClipError::ModelNotFound {
                    model: self.model_info.name.to_string(),
                    hint: format!(
                        "Tokenizer missing at {}. Copy tokenizer.json from {}",
                        path.display(),
                        self.model_info.tokenizer_url
                    ),
                }
                .into());
            }

            std::fs::create_dir_all(&self.config.models_dir).map_err(|e| {
                ClipError::DownloadError {
                    cause: format!("Failed to create models directory: {}", e),
                }
            })?;

            Err(ClipError::DownloadError {
                cause: format!(
                    "Automatic download not yet implemented. Please download manually:\n\
                     curl -L '{}' -o '{}'",
                    self.model_info.tokenizer_url,
                    path.display()
                ),
            }
            .into())
        }

        /// Load vision session lazily
        fn load_vision_session(&self) -> Result<()> {
            let mut session_guard = self
                .vision_session
                .lock()
                .map_err(|_| MemvidError::Lock("Failed to lock vision session".into()))?;

            if session_guard.is_some() {
                return Ok(());
            }

            let vision_path = self.ensure_model_file("vision")?;

            tracing::debug!(path = %vision_path.display(), "Loading CLIP vision model");

            let session = Session::builder()
                .map_err(|e| ClipError::InferenceError {
                    cause: e.to_string(),
                })?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| ClipError::InferenceError {
                    cause: e.to_string(),
                })?
                .with_intra_threads(4)
                .map_err(|e| ClipError::InferenceError {
                    cause: e.to_string(),
                })?
                .commit_from_file(&vision_path)
                .map_err(|e| ClipError::InferenceError {
                    cause: format!("Failed to load vision model: {}", e),
                })?;

            *session_guard = Some(session);
            tracing::info!(model = %self.model_info.name, "CLIP vision model loaded");

            Ok(())
        }

        /// Load text session lazily
        fn load_text_session(&self) -> Result<()> {
            let mut session_guard = self
                .text_session
                .lock()
                .map_err(|_| MemvidError::Lock("Failed to lock text session".into()))?;

            if session_guard.is_some() {
                return Ok(());
            }

            let text_path = self.ensure_model_file("text")?;

            tracing::debug!(path = %text_path.display(), "Loading CLIP text model");

            let session = Session::builder()
                .map_err(|e| ClipError::InferenceError {
                    cause: e.to_string(),
                })?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| ClipError::InferenceError {
                    cause: e.to_string(),
                })?
                .with_intra_threads(4)
                .map_err(|e| ClipError::InferenceError {
                    cause: e.to_string(),
                })?
                .commit_from_file(&text_path)
                .map_err(|e| ClipError::InferenceError {
                    cause: format!("Failed to load text model: {}", e),
                })?;

            *session_guard = Some(session);
            tracing::info!(model = %self.model_info.name, "CLIP text model loaded");

            Ok(())
        }

        /// Load tokenizer lazily (matches the text model vocab/BPE)
        fn load_tokenizer(&self) -> Result<()> {
            let mut tokenizer_guard = self
                .tokenizer
                .lock()
                .map_err(|_| MemvidError::Lock("Failed to lock CLIP tokenizer".into()))?;

            if tokenizer_guard.is_some() {
                return Ok(());
            }

            let tokenizer_path = self.ensure_tokenizer_file()?;

            tracing::debug!(path = %tokenizer_path.display(), "Loading CLIP tokenizer");

            let mut tokenizer =
                Tokenizer::from_file(&tokenizer_path).map_err(|e| ClipError::InferenceError {
                    cause: format!("Failed to load tokenizer: {}", e),
                })?;

            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(77),
                direction: PaddingDirection::Right,
                pad_to_multiple_of: None,
                pad_id: 0,
                pad_type_id: 0,
                pad_token: "[PAD]".to_string(),
            }));

            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: 77,
                    strategy: TruncationStrategy::LongestFirst,
                    stride: 0,
                    direction: TruncationDirection::Right,
                }))
                .map_err(|e| ClipError::InferenceError {
                    cause: format!("Failed to apply truncation config: {}", e),
                })?;

            *tokenizer_guard = Some(tokenizer);
            tracing::info!(model = %self.model_info.name, "CLIP tokenizer loaded");

            Ok(())
        }

        /// Preprocess image for CLIP inference
        ///
        /// MobileCLIP-S2 uses:
        /// - Input size: 256x256
        /// - Resize: shortest edge to 256, preserve aspect, center-crop
        /// - Normalization: scale to [0, 1] (no mean/std shift per preprocessor_config)
        /// - Format: NCHW (batch, channels, height, width)
        fn preprocess_image(&self, image: &DynamicImage) -> Array4<f32> {
            let size = self.model_info.input_resolution;
            let rgb_input = image.to_rgb8();
            let (w, h) = rgb_input.dimensions();

            // Resize shortest edge to target while preserving aspect ratio
            let scale = size as f32 / w.min(h) as f32;
            let new_w = ((w as f32) * scale).round().max(1.0) as u32;
            let new_h = ((h as f32) * scale).round().max(1.0) as u32;
            let resized = image.resize_exact(new_w, new_h, FilterType::Triangle);

            // Center crop to (size, size)
            let start_x = (resized.width().saturating_sub(size)) / 2;
            let start_y = (resized.height().saturating_sub(size)) / 2;

            // Create array in NCHW format: [1, 3, H, W]
            let mut array = Array4::<f32>::zeros((1, 3, size as usize, size as usize));

            for y in 0..size as usize {
                for x in 0..size as usize {
                    let pixel = resized.get_pixel(start_x + x as u32, start_y + y as u32);
                    array[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
                    array[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
                    array[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
                }
            }

            array
        }

        /// Encode an image to CLIP embedding
        pub fn encode_image(&self, image: &DynamicImage) -> Result<Vec<f32>> {
            // Ensure vision session is loaded
            self.load_vision_session()?;

            // Preprocess the image
            let pixel_values = self.preprocess_image(image);

            // Update last used timestamp
            if let Ok(mut last) = self.last_used.lock() {
                *last = Instant::now();
            }

            // Run inference
            let mut session_guard = self
                .vision_session
                .lock()
                .map_err(|_| MemvidError::Lock("Failed to lock vision session".into()))?;

            let session = session_guard
                .as_mut()
                .ok_or_else(|| ClipError::InferenceError {
                    cause: "Vision session not loaded".to_string(),
                })?;

            // Get input and output names from session before running
            let input_name = session
                .inputs
                .first()
                .map(|i| i.name.clone())
                .unwrap_or_else(|| "pixel_values".into());
            let output_name = session
                .outputs
                .first()
                .map(|o| o.name.clone())
                .unwrap_or_else(|| "image_embeds".into());

            // Create tensor from ndarray
            let input_tensor =
                Tensor::from_array(pixel_values).map_err(|e| ClipError::InferenceError {
                    cause: format!("Failed to create input tensor: {}", e),
                })?;

            // Run the model
            let outputs = session
                .run(ort::inputs![input_name => input_tensor])
                .map_err(|e| ClipError::InferenceError {
                    cause: format!("Vision inference failed: {}", e),
                })?;

            // Extract embeddings from first output
            let output = outputs
                .get(&output_name)
                .ok_or_else(|| ClipError::InferenceError {
                    cause: format!("No output '{}' from vision model", output_name),
                })?;

            let (_shape, data) =
                output
                    .try_extract_tensor::<f32>()
                    .map_err(|e| ClipError::InferenceError {
                        cause: format!("Failed to extract embeddings: {}", e),
                    })?;

            // Get the embedding from the raw data
            let embedding: Vec<f32> = data.to_vec();
            if embedding.iter().any(|v| !v.is_finite()) {
                return Err(ClipError::InferenceError {
                    cause: "Vision embedding contains non-finite values".to_string(),
                }
                .into());
            }
            let normalized = l2_normalize(&embedding);

            tracing::debug!(dims = normalized.len(), "Generated CLIP image embedding");

            Ok(normalized)
        }

        /// Encode image bytes to CLIP embedding
        pub fn encode_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
            let image =
                image::load_from_memory(bytes).map_err(|e| ClipError::ImageBytesDecodeError {
                    cause: e.to_string(),
                })?;
            self.encode_image(&image)
        }

        /// Encode an image file to CLIP embedding
        pub fn encode_image_file(&self, path: &Path) -> Result<Vec<f32>> {
            let image = image::open(path).map_err(|e| ClipError::ImageDecodeError {
                path: path.to_path_buf(),
                cause: e.to_string(),
            })?;
            self.encode_image(&image)
        }

        /// Encode text to CLIP embedding (for query)
        pub fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
            // Ensure text session is loaded
            self.load_text_session()?;
            self.load_tokenizer()?;

            // Tokenize the text using the model's tokenizer
            let encoding = {
                let tokenizer_guard = self
                    .tokenizer
                    .lock()
                    .map_err(|_| MemvidError::Lock("Failed to lock CLIP tokenizer".into()))?;
                let tokenizer =
                    tokenizer_guard
                        .as_ref()
                        .ok_or_else(|| ClipError::InferenceError {
                            cause: "Tokenizer not loaded".to_string(),
                        })?;

                tokenizer
                    .encode(text, true)
                    .map_err(|e| ClipError::InferenceError {
                        cause: format!("Text tokenization failed: {}", e),
                    })?
            };

            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|id| *id as i64).collect();
            let attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|id| *id as i64)
                .collect();
            let max_length = input_ids.len();

            // Create input arrays
            let input_ids_array =
                Array::from_shape_vec((1, max_length), input_ids).map_err(|e| {
                    ClipError::InferenceError {
                        cause: e.to_string(),
                    }
                })?;
            let attention_mask_array = Array::from_shape_vec((1, max_length), attention_mask)
                .map_err(|e| ClipError::InferenceError {
                    cause: e.to_string(),
                })?;

            // Update last used timestamp
            if let Ok(mut last) = self.last_used.lock() {
                *last = Instant::now();
            }

            // Run inference
            let mut session_guard = self
                .text_session
                .lock()
                .map_err(|_| MemvidError::Lock("Failed to lock text session".into()))?;

            let session = session_guard
                .as_mut()
                .ok_or_else(|| ClipError::InferenceError {
                    cause: "Text session not loaded".to_string(),
                })?;

            // Get input and output names from session before running
            let input_names: Vec<String> = session.inputs.iter().map(|i| i.name.clone()).collect();
            let output_name = session
                .outputs
                .first()
                .map(|o| o.name.clone())
                .unwrap_or_else(|| "text_embeds".into());

            // Create tensors from ndarray
            let input_ids_tensor =
                Tensor::from_array(input_ids_array).map_err(|e| ClipError::InferenceError {
                    cause: format!("Failed to create input_ids tensor: {}", e),
                })?;
            let attention_mask_tensor = Tensor::from_array(attention_mask_array).map_err(|e| {
                ClipError::InferenceError {
                    cause: format!("Failed to create attention_mask tensor: {}", e),
                }
            })?;

            // Build inputs based on what the model expects
            let outputs = if input_names.len() >= 2 {
                session
                    .run(ort::inputs![
                        input_names[0].clone() => input_ids_tensor,
                        input_names[1].clone() => attention_mask_tensor
                    ])
                    .map_err(|e| ClipError::InferenceError {
                        cause: format!("Text inference failed: {}", e),
                    })?
            } else {
                // Single input model
                let name = input_names
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "input_ids".to_string());
                session
                    .run(ort::inputs![name => input_ids_tensor])
                    .map_err(|e| ClipError::InferenceError {
                        cause: format!("Text inference failed: {}", e),
                    })?
            };

            // Extract embeddings from output
            let output = outputs
                .get(&output_name)
                .ok_or_else(|| ClipError::InferenceError {
                    cause: format!("No output '{}' from text model", output_name),
                })?;

            let (_shape, data) =
                output
                    .try_extract_tensor::<f32>()
                    .map_err(|e| ClipError::InferenceError {
                        cause: format!("Failed to extract text embeddings: {}", e),
                    })?;

            // Flatten and normalize the embedding
            let embedding: Vec<f32> = data.to_vec();
            if embedding.iter().any(|v| !v.is_finite()) {
                return Err(ClipError::InferenceError {
                    cause: "Text embedding contains non-finite values".to_string(),
                }
                .into());
            }
            let normalized = l2_normalize(&embedding);

            tracing::debug!(
                text_len = text.len(),
                dims = normalized.len(),
                "Generated CLIP text embedding"
            );

            Ok(normalized)
        }

        /// Maybe unload model if unused for too long (memory management)
        pub fn maybe_unload(&self) -> Result<()> {
            let last_used = self
                .last_used
                .lock()
                .map_err(|_| MemvidError::Lock("Failed to check last_used".into()))?;

            if last_used.elapsed() > MODEL_UNLOAD_TIMEOUT {
                tracing::debug!(model = %self.model_info.name, "Model idle, unloading sessions");

                // Unload vision session
                if let Ok(mut guard) = self.vision_session.lock() {
                    *guard = None;
                }

                // Unload text session
                if let Ok(mut guard) = self.text_session.lock() {
                    *guard = None;
                }

                // Unload tokenizer
                if let Ok(mut guard) = self.tokenizer.lock() {
                    *guard = None;
                }
            }

            Ok(())
        }

        /// Force unload all sessions
        pub fn unload(&self) -> Result<()> {
            if let Ok(mut guard) = self.vision_session.lock() {
                *guard = None;
            }
            if let Ok(mut guard) = self.text_session.lock() {
                *guard = None;
            }
            if let Ok(mut guard) = self.tokenizer.lock() {
                *guard = None;
            }
            tracing::debug!(model = %self.model_info.name, "CLIP sessions unloaded");
            Ok(())
        }

        /// Check if vision model is loaded
        pub fn is_vision_loaded(&self) -> bool {
            self.vision_session
                .lock()
                .map(|g| g.is_some())
                .unwrap_or(false)
        }

        /// Check if text model is loaded
        pub fn is_text_loaded(&self) -> bool {
            self.text_session
                .lock()
                .map(|g| g.is_some())
                .unwrap_or(false)
        }
    }

    /// L2 normalize a vector (unit length)
    fn l2_normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm.is_finite() && norm > 1e-10 {
            v.iter().map(|x| x / norm).collect()
        } else {
            // Fall back to zeros to avoid NaNs propagating through distances
            vec![0.0; v.len()]
        }
    }

    /// Calculate color variance of an image
    pub fn calculate_color_variance(image: &DynamicImage) -> f32 {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        let total_pixels = (width * height) as f32;

        if total_pixels == 0.0 {
            return 0.0;
        }

        // Calculate mean
        let mut sum_r = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;

        for pixel in rgb.pixels() {
            sum_r += pixel[0] as f32;
            sum_g += pixel[1] as f32;
            sum_b += pixel[2] as f32;
        }

        let mean_r = sum_r / total_pixels;
        let mean_g = sum_g / total_pixels;
        let mean_b = sum_b / total_pixels;

        // Calculate variance
        let mut var_r = 0.0f32;
        let mut var_g = 0.0f32;
        let mut var_b = 0.0f32;

        for pixel in rgb.pixels() {
            var_r += (pixel[0] as f32 - mean_r).powi(2);
            var_g += (pixel[1] as f32 - mean_g).powi(2);
            var_b += (pixel[2] as f32 - mean_b).powi(2);
        }

        // Average variance across channels, normalized to 0-1
        ((var_r + var_g + var_b) / (3.0 * total_pixels)) / (255.0 * 255.0)
    }

    /// Get ImageInfo from a DynamicImage
    pub fn get_image_info(image: &DynamicImage) -> ImageInfo {
        let (width, height) = image.dimensions();
        let color_variance = calculate_color_variance(image);

        ImageInfo {
            width,
            height,
            color_variance,
        }
    }
}

#[cfg(feature = "clip")]
pub use model::*;

#[cfg(all(feature = "clip", feature = "pdfium"))]
use pdfium_render::prelude::{PdfPageRenderRotation, PdfRenderConfig, Pdfium};

/// Render PDF pages to images suitable for CLIP embedding (feature-gated).
#[cfg(all(feature = "clip", feature = "pdfium"))]
pub fn render_pdf_pages_for_clip(
    path: &Path,
    max_pages: usize,
    target_px: u32,
) -> Result<Vec<(u32, DynamicImage)>> {
    let bindings = Pdfium::bind_to_system_library().map_err(|e| ClipError::InferenceError {
        cause: format!("Failed to bind pdfium: {}", e),
    })?;
    let pdfium = Pdfium::new(bindings);
    let document =
        pdfium
            .load_pdf_from_file(path, None)
            .map_err(|e| ClipError::InferenceError {
                cause: format!("Failed to load PDF for CLIP rendering: {}", e),
            })?;

    let render_config = PdfRenderConfig::new()
        .set_target_width(target_px as i32)
        .set_maximum_height(target_px as i32)
        .set_maximum_width(target_px as i32)
        .rotate_if_landscape(PdfPageRenderRotation::None, false);

    let mut pages = Vec::new();
    for (index, page) in document.pages().iter().enumerate() {
        if index >= max_pages {
            break;
        }
        let rendered = page
            .render_with_config(&render_config)
            .map_err(|e| ClipError::InferenceError {
                cause: format!("Failed to render PDF page {}: {}", index + 1, e),
            })?
            .as_image();
        pages.push(((index + 1) as u32, rendered));
    }

    Ok(pages)
}

#[cfg(all(feature = "clip", not(feature = "pdfium")))]
pub fn render_pdf_pages_for_clip(
    path: &Path,
    max_pages: usize,
    _target_px: u32,
) -> Result<Vec<(u32, DynamicImage)>> {
    fn extract_images_from_page(
        doc: &Document,
        page_id: ObjectId,
        remaining: &mut usize,
        out: &mut Vec<(u32, DynamicImage)>,
    ) -> Result<()> {
        if *remaining == 0 {
            return Ok(());
        }

        let (resources_opt, resource_ids) =
            doc.get_page_resources(page_id)
                .map_err(|e| ClipError::InferenceError {
                    cause: format!("Failed to read PDF resources: {}", e),
                })?;

        let mut seen = HashSet::new();
        let mut resource_dicts: Vec<Dictionary> = Vec::new();

        if let Some(dict) = resources_opt {
            resource_dicts.push(dict.clone());
        }
        for res_id in resource_ids {
            if seen.insert(res_id) {
                if let Ok(dict) = doc.get_dictionary(res_id) {
                    resource_dicts.push(dict.clone());
                }
            }
        }

        for dict in resource_dicts {
            if let Ok(xobjects) = dict.get(b"XObject") {
                let xobj_dict = match xobjects {
                    Object::Dictionary(d) => Some(d),
                    Object::Reference(id) => doc.get_dictionary(*id).ok(),
                    _ => None,
                };
                if let Some(xobj_dict) = xobj_dict {
                    for (_, obj) in xobj_dict.iter() {
                        let id = match obj {
                            Object::Reference(id) => *id,
                            _ => continue,
                        };
                        let stream = match doc.get_object(id).and_then(Object::as_stream) {
                            Ok(s) => s,
                            Err(_) => continue,
                        };
                        let subtype = stream.dict.get(b"Subtype").ok();
                        let is_image = matches!(subtype, Some(Object::Name(n)) if n == b"Image");
                        if !is_image {
                            continue;
                        }

                        let width = stream
                            .dict
                            .get(b"Width")
                            .ok()
                            .and_then(|o| o.as_i64().ok())
                            .unwrap_or(0);
                        let height = stream
                            .dict
                            .get(b"Height")
                            .ok()
                            .and_then(|o| o.as_i64().ok())
                            .unwrap_or(0);
                        if width <= 0 || height <= 0 {
                            continue;
                        }

                        let filters = stream
                            .dict
                            .get(b"Filter")
                            .ok()
                            .and_then(|f| match f {
                                Object::Name(n) => Some(vec![n.clone()]),
                                Object::Array(arr) => Some(
                                    arr.iter()
                                        .filter_map(|o| o.as_name().ok().map(|n| n.to_vec()))
                                        .collect(),
                                ),
                                _ => None,
                            })
                            .unwrap_or_default();

                        let data = stream
                            .decompressed_content()
                            .unwrap_or_else(|_| stream.content.clone());

                        // If DCT/JPX, hand to image crate directly
                        if filters
                            .iter()
                            .any(|f| f == b"DCTDecode" || f == b"JPXDecode")
                        {
                            if let Ok(img) = image::load_from_memory(&data) {
                                out.push((1, img));
                                if out.len() >= *remaining {
                                    *remaining = 0;
                                    return Ok(());
                                }
                                *remaining -= 1;
                                continue;
                            }
                        }

                        let color_space = stream
                            .dict
                            .get(b"ColorSpace")
                            .ok()
                            .and_then(|o| o.as_name().ok())
                            .unwrap_or(b"DeviceRGB");
                        let channels = if color_space == b"DeviceGray" { 1 } else { 3 };

                        let expected = width as usize * height as usize * channels;
                        if data.len() >= expected && channels == 3 {
                            if let Some(buf) = ImageBuffer::<Rgb<u8>, _>::from_raw(
                                width as u32,
                                height as u32,
                                data.clone(),
                            ) {
                                out.push((1, DynamicImage::ImageRgb8(buf)));
                                if out.len() >= *remaining {
                                    *remaining = 0;
                                    return Ok(());
                                }
                                *remaining -= 1;
                                continue;
                            }
                        } else if data.len() >= expected && channels == 1 {
                            if let Some(buf) = ImageBuffer::<Luma<u8>, _>::from_raw(
                                width as u32,
                                height as u32,
                                data.clone(),
                            ) {
                                out.push((1, DynamicImage::ImageLuma8(buf)));
                                if out.len() >= *remaining {
                                    *remaining = 0;
                                    return Ok(());
                                }
                                *remaining -= 1;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    let doc = Document::load(path).map_err(|e| ClipError::InferenceError {
        cause: format!("Failed to load PDF for image extraction: {}", e),
    })?;

    let mut remaining = max_pages;
    let mut pages: Vec<(u32, DynamicImage)> = Vec::new();

    for (page_num, page_id) in doc.get_pages() {
        if remaining == 0 {
            break;
        }
        let start_len = pages.len();
        extract_images_from_page(&doc, page_id, &mut remaining, &mut pages)?;
        if pages.len() > start_len {
            for entry in pages.iter_mut().skip(start_len) {
                entry.0 = page_num as u32;
            }
        }
    }

    Ok(pages)
}

// ============================================================================
// CLIP Embedding Provider Trait
// ============================================================================

/// Trait for CLIP visual embedding providers.
///
/// Unlike text `EmbeddingProvider`, CLIP providers handle both:
/// - **Image encoding**: Generate embeddings from images (for indexing)
/// - **Text encoding**: Generate embeddings from text (for queries)
///
/// This allows natural language queries against visual content.
///
/// # Example
///
/// ```ignore
/// use memvid_core::clip::{ClipEmbeddingProvider, ClipConfig};
///
/// // Create provider
/// let provider = ClipModel::new(ClipConfig::default())?;
///
/// // Encode image for indexing
/// let image_embedding = provider.embed_image_file(&path)?;
///
/// // Encode query text for search
/// let query_embedding = provider.embed_query("a photo of a cat")?;
///
/// // Search uses cosine similarity between query and image embeddings
/// ```
pub trait ClipEmbeddingProvider: Send + Sync {
    /// Return the provider kind (e.g., "mobileclip", "siglip").
    fn kind(&self) -> &str;

    /// Return the model identifier.
    fn model(&self) -> &str;

    /// Return the embedding dimension.
    fn dimension(&self) -> usize;

    /// Generate an embedding for an image file.
    fn embed_image_file(&self, path: &Path) -> Result<Vec<f32>>;

    /// Generate an embedding for image bytes.
    fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>>;

    /// Generate an embedding for a text query (for searching).
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple image files.
    ///
    /// Default implementation calls `embed_image_file` in a loop.
    /// Providers should override this for efficient batch processing.
    fn embed_image_batch(&self, paths: &[&Path]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(paths.len());
        for path in paths {
            embeddings.push(self.embed_image_file(path)?);
        }
        Ok(embeddings)
    }

    /// Check if the provider is ready to generate embeddings.
    fn is_ready(&self) -> bool {
        true
    }

    /// Initialize the provider (e.g., load models).
    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    /// Unload models to free memory.
    fn unload(&self) -> Result<()> {
        Ok(())
    }
}

/// Result type for CLIP embedding operations
pub type ClipEmbeddingResult = Result<Vec<f32>>;
pub type ClipBatchEmbeddingResult = Result<Vec<Vec<f32>>>;

// ============================================================================
// ClipEmbeddingProvider Implementation (Feature-gated)
// ============================================================================

#[cfg(feature = "clip")]
impl ClipEmbeddingProvider for ClipModel {
    fn kind(&self) -> &str {
        "clip"
    }

    fn model(&self) -> &str {
        self.model_info().name
    }

    fn dimension(&self) -> usize {
        self.model_info().dims as usize
    }

    fn embed_image_file(&self, path: &Path) -> Result<Vec<f32>> {
        self.encode_image_file(path)
    }

    fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        self.encode_image_bytes(bytes)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.encode_text(text)
    }

    fn embed_image_batch(&self, paths: &[&Path]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(paths.len());
        for path in paths {
            embeddings.push(self.encode_image_file(path)?);
        }
        Ok(embeddings)
    }

    fn is_ready(&self) -> bool {
        // CLIP models are lazy-loaded, so always "ready"
        true
    }

    fn unload(&self) -> Result<()> {
        ClipModel::unload(self)
    }
}

// ============================================================================
// CLIP Index Manifest (for TOC)
// ============================================================================

/// Manifest for CLIP index stored in TOC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipIndexManifest {
    /// Byte offset in file
    pub bytes_offset: u64,
    /// Length in bytes
    pub bytes_length: u64,
    /// Number of vectors
    pub vector_count: u64,
    /// Embedding dimensions
    pub dimension: u32,
    /// Blake3 checksum
    pub checksum: [u8; 32],
    /// Model name used to generate embeddings
    pub model_name: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_index_builder_roundtrip() {
        let mut builder = ClipIndexBuilder::new();
        builder.add_document(1, None, vec![0.1, 0.2, 0.3, 0.4]);
        builder.add_document(2, None, vec![0.5, 0.6, 0.7, 0.8]);

        let artifact = builder.finish().expect("finish");
        assert_eq!(artifact.vector_count, 2);
        assert_eq!(artifact.dimension, 4);

        let index = ClipIndex::decode(&artifact.bytes).expect("decode");
        assert_eq!(index.len(), 2);

        let hits = index.search(&[0.1, 0.2, 0.3, 0.4], 10);
        assert_eq!(hits[0].frame_id, 1);
        assert!(hits[0].distance < 0.001); // Should be very close
    }

    #[test]
    fn clip_index_search() {
        let mut builder = ClipIndexBuilder::new();
        builder.add_document(1, None, vec![1.0, 0.0, 0.0]);
        builder.add_document(2, None, vec![0.0, 1.0, 0.0]);
        builder.add_document(3, None, vec![0.0, 0.0, 1.0]);

        let artifact = builder.finish().expect("finish");
        let index = ClipIndex::decode(&artifact.bytes).expect("decode");

        // Search for [1, 0, 0] - should find frame 1 first
        let hits = index.search(&[1.0, 0.0, 0.0], 3);
        assert_eq!(hits[0].frame_id, 1);

        // Search for [0, 1, 0] - should find frame 2 first
        let hits = index.search(&[0.0, 1.0, 0.0], 3);
        assert_eq!(hits[0].frame_id, 2);
    }

    #[test]
    fn l2_distance_calculation() {
        let d = l2_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-6);

        let d = l2_distance(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn image_info_filtering() {
        // Tiny image - should skip
        let tiny = ImageInfo {
            width: 32,
            height: 32,
            color_variance: 0.5,
        };
        assert!(!tiny.should_embed());

        // Good image
        let good = ImageInfo {
            width: 256,
            height: 256,
            color_variance: 0.5,
        };
        assert!(good.should_embed());

        // Extreme aspect ratio
        let wide = ImageInfo {
            width: 1000,
            height: 10,
            color_variance: 0.5,
        };
        assert!(!wide.should_embed());

        // Solid color
        let solid = ImageInfo {
            width: 256,
            height: 256,
            color_variance: 0.001,
        };
        assert!(!solid.should_embed());
    }

    #[test]
    fn model_registry() {
        let default = default_model_info();
        assert_eq!(default.name, "mobileclip-s2");
        assert_eq!(default.dims, 512);
        assert!(default.is_default);

        let siglip = get_model_info("siglip-base");
        assert_eq!(siglip.dims, 768);

        // Unknown model returns default
        let unknown = get_model_info("nonexistent");
        assert_eq!(unknown.name, "mobileclip-s2");
    }

    #[test]
    fn clip_config_defaults() {
        // Clear the env vars to test true defaults
        // SAFETY: No other threads are modifying these env vars in this test
        unsafe {
            std::env::remove_var("MEMVID_CLIP_MODEL");
            std::env::remove_var("MEMVID_OFFLINE");
        }

        let config = ClipConfig::default();
        assert_eq!(config.model_name, "mobileclip-s2");
        assert!(!config.offline);
    }

    #[test]
    fn clip_embedding_provider_trait() {
        // Test that the trait is properly defined
        fn assert_send_sync<T: Send + Sync>() {}

        // The trait should require Send + Sync
        assert_send_sync::<Box<dyn super::ClipEmbeddingProvider>>();
    }
}
