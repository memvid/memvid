// Safe expect: Static NER model lookup with guaranteed default.
#![allow(clippy::expect_used)]
//! Named Entity Recognition (NER) module using DistilBERT-NER ONNX.
//!
//! This module provides entity extraction capabilities using DistilBERT-NER,
//! a fast and accurate NER model fine-tuned on CoNLL-03.
//!
//! # Model
//!
//! Uses dslim/distilbert-NER ONNX (~261 MB) with 92% F1 score.
//! Entities: Person (PER), Organization (ORG), Location (LOC), Miscellaneous (MISC)
//!
//! # Simple Interface
//!
//! Unlike GLiNER, DistilBERT-NER uses standard BERT tokenization:
//! - Input: `input_ids`, `attention_mask`
//! - Output: per-token logits for B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O

use crate::types::{EntityKind, FrameId};
use crate::{MemvidError, Result};
use std::path::{Path, PathBuf};

// ============================================================================
// Configuration Constants
// ============================================================================

/// Model name for downloads and caching
pub const NER_MODEL_NAME: &str = "distilbert-ner";

/// Model download URL (HuggingFace)
pub const NER_MODEL_URL: &str =
    "https://huggingface.co/dslim/distilbert-NER/resolve/main/onnx/model.onnx";

/// Tokenizer URL
pub const NER_TOKENIZER_URL: &str =
    "https://huggingface.co/dslim/distilbert-NER/resolve/main/tokenizer.json";

/// Approximate model size in MB
pub const NER_MODEL_SIZE_MB: f32 = 261.0;

/// Maximum sequence length for the model
pub const NER_MAX_SEQ_LEN: usize = 512;

/// Minimum confidence threshold for entity extraction
#[cfg_attr(not(feature = "logic_mesh"), allow(dead_code))]
pub const NER_MIN_CONFIDENCE: f32 = 0.5;

/// NER label mapping (CoNLL-03 format)
/// O=0, B-PER=1, I-PER=2, B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6, B-MISC=7, I-MISC=8
#[cfg_attr(not(feature = "logic_mesh"), allow(dead_code))]
pub const NER_LABELS: &[&str] = &[
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC",
];

// ============================================================================
// Model Info
// ============================================================================

/// NER model info for the models registry
#[derive(Debug, Clone)]
pub struct NerModelInfo {
    /// Model identifier
    pub name: &'static str,
    /// URL for ONNX model
    pub model_url: &'static str,
    /// URL for tokenizer JSON
    pub tokenizer_url: &'static str,
    /// Model size in MB
    pub size_mb: f32,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Whether this is the default model
    pub is_default: bool,
}

/// Available NER models registry
pub static NER_MODELS: &[NerModelInfo] = &[NerModelInfo {
    name: NER_MODEL_NAME,
    model_url: NER_MODEL_URL,
    tokenizer_url: NER_TOKENIZER_URL,
    size_mb: NER_MODEL_SIZE_MB,
    max_seq_len: NER_MAX_SEQ_LEN,
    is_default: true,
}];

/// Get NER model info by name
pub fn get_ner_model_info(name: &str) -> Option<&'static NerModelInfo> {
    NER_MODELS.iter().find(|m| m.name == name)
}

/// Get default NER model info
pub fn default_ner_model_info() -> &'static NerModelInfo {
    NER_MODELS
        .iter()
        .find(|m| m.is_default)
        .expect("default NER model must exist")
}

// ============================================================================
// Entity Extraction Types
// ============================================================================

/// Raw entity mention extracted from text
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    /// The extracted text span
    pub text: String,
    /// Entity type (PER, ORG, LOC, MISC)
    pub entity_type: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Byte offset start in original text
    pub byte_start: usize,
    /// Byte offset end in original text
    pub byte_end: usize,
}

impl ExtractedEntity {
    /// Convert the raw entity type to our EntityKind enum
    pub fn to_entity_kind(&self) -> EntityKind {
        match self.entity_type.to_uppercase().as_str() {
            "PER" | "PERSON" | "B-PER" | "I-PER" => EntityKind::Person,
            "ORG" | "ORGANIZATION" | "B-ORG" | "I-ORG" => EntityKind::Organization,
            "LOC" | "LOCATION" | "B-LOC" | "I-LOC" => EntityKind::Location,
            "MISC" | "B-MISC" | "I-MISC" => EntityKind::Other,
            _ => EntityKind::Other,
        }
    }
}

/// Result of extracting entities from a frame
#[derive(Debug, Clone)]
pub struct FrameEntities {
    /// Frame ID the entities were extracted from
    pub frame_id: FrameId,
    /// Extracted entities
    pub entities: Vec<ExtractedEntity>,
}

// ============================================================================
// NER Model (Feature-gated)
// ============================================================================

#[cfg(feature = "logic_mesh")]
pub use model_impl::*;

#[cfg(feature = "logic_mesh")]
mod model_impl {
    use super::*;
    use ort::session::{Session, builder::GraphOptimizationLevel};
    use ort::value::Tensor;
    use std::sync::Mutex;
    use tokenizers::{
        PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
        TruncationParams, TruncationStrategy,
    };

    /// DistilBERT-NER model for entity extraction
    pub struct NerModel {
        /// ONNX runtime session
        session: Session,
        /// Tokenizer for text preprocessing
        tokenizer: Mutex<Tokenizer>,
        /// Model path for reference
        model_path: PathBuf,
        /// Minimum confidence threshold
        min_confidence: f32,
    }

    impl NerModel {
        /// Load NER model from path
        ///
        /// # Arguments
        /// * `model_path` - Path to the ONNX model file
        /// * `tokenizer_path` - Path to the tokenizer.json file
        /// * `min_confidence` - Minimum confidence threshold (default: 0.5)
        pub fn load(
            model_path: impl AsRef<Path>,
            tokenizer_path: impl AsRef<Path>,
            min_confidence: Option<f32>,
        ) -> Result<Self> {
            let model_path = model_path.as_ref().to_path_buf();
            let tokenizer_path = tokenizer_path.as_ref();

            // Load tokenizer
            let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
                MemvidError::NerModelNotAvailable {
                    reason: format!("failed to load tokenizer from {:?}: {}", tokenizer_path, e)
                        .into(),
                }
            })?;

            // Configure padding and truncation
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                direction: PaddingDirection::Right,
                pad_to_multiple_of: None,
                pad_id: 0,
                pad_type_id: 0,
                pad_token: "[PAD]".to_string(),
            }));

            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: NER_MAX_SEQ_LEN,
                    strategy: TruncationStrategy::LongestFirst,
                    stride: 0,
                    direction: TruncationDirection::Right,
                }))
                .map_err(|e| MemvidError::NerModelNotAvailable {
                    reason: format!("failed to set truncation: {}", e).into(),
                })?;

            // Initialize ONNX Runtime
            let session = Session::builder()
                .map_err(|e| MemvidError::NerModelNotAvailable {
                    reason: format!("failed to create session builder: {}", e).into(),
                })?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| MemvidError::NerModelNotAvailable {
                    reason: format!("failed to set optimization level: {}", e).into(),
                })?
                .with_intra_threads(4)
                .map_err(|e| MemvidError::NerModelNotAvailable {
                    reason: format!("failed to set threads: {}", e).into(),
                })?
                .commit_from_file(&model_path)
                .map_err(|e| MemvidError::NerModelNotAvailable {
                    reason: format!("failed to load model from {:?}: {}", model_path, e).into(),
                })?;

            tracing::info!(
                model = %model_path.display(),
                "DistilBERT-NER model loaded"
            );

            Ok(Self {
                session,
                tokenizer: Mutex::new(tokenizer),
                model_path,
                min_confidence: min_confidence.unwrap_or(NER_MIN_CONFIDENCE),
            })
        }

        /// Extract entities from text
        pub fn extract(&mut self, text: &str) -> Result<Vec<ExtractedEntity>> {
            if text.trim().is_empty() {
                return Ok(Vec::new());
            }

            // Tokenize
            let tokenizer = self
                .tokenizer
                .lock()
                .map_err(|_| MemvidError::Lock("failed to lock tokenizer".into()))?;

            let encoding =
                tokenizer
                    .encode(text, true)
                    .map_err(|e| MemvidError::NerModelNotAvailable {
                        reason: format!("tokenization failed: {}", e).into(),
                    })?;

            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&x| x as i64)
                .collect();
            let tokens = encoding.get_tokens().to_vec();
            let offsets = encoding.get_offsets().to_vec();

            drop(tokenizer); // Release lock before inference

            let seq_len = input_ids.len();

            // Create input tensors using Tensor::from_array
            let input_ids_array = ndarray::Array2::from_shape_vec((1, seq_len), input_ids)
                .map_err(|e| MemvidError::NerModelNotAvailable {
                    reason: format!("failed to create input_ids array: {}", e).into(),
                })?;

            let attention_mask_array =
                ndarray::Array2::from_shape_vec((1, seq_len), attention_mask).map_err(|e| {
                    MemvidError::NerModelNotAvailable {
                        reason: format!("failed to create attention_mask array: {}", e).into(),
                    }
                })?;

            let input_ids_tensor = Tensor::from_array(input_ids_array).map_err(|e| {
                MemvidError::NerModelNotAvailable {
                    reason: format!("failed to create input_ids tensor: {}", e).into(),
                }
            })?;

            let attention_mask_tensor = Tensor::from_array(attention_mask_array).map_err(|e| {
                MemvidError::NerModelNotAvailable {
                    reason: format!("failed to create attention_mask tensor: {}", e).into(),
                }
            })?;

            // Get output name before inference (avoid borrow conflict)
            let output_name = self
                .session
                .outputs
                .first()
                .map(|o| o.name.clone())
                .unwrap_or_else(|| "logits".into());

            // Run inference
            let outputs = self
                .session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                ])
                .map_err(|e| MemvidError::NerModelNotAvailable {
                    reason: format!("inference failed: {}", e).into(),
                })?;

            let logits =
                outputs
                    .get(&output_name)
                    .ok_or_else(|| MemvidError::NerModelNotAvailable {
                        reason: format!("no output '{}' found", output_name).into(),
                    })?;

            // Parse logits to get predictions
            let entities = Self::decode_predictions_static(
                text,
                &tokens,
                &offsets,
                logits,
                self.min_confidence,
            )?;

            Ok(entities)
        }

        /// Decode model predictions into entities (static to avoid borrow issues)
        fn decode_predictions_static(
            original_text: &str,
            tokens: &[String],
            offsets: &[(usize, usize)],
            logits: &ort::value::Value,
            min_confidence: f32,
        ) -> Result<Vec<ExtractedEntity>> {
            // Extract the logits tensor - shape: [1, seq_len, num_labels]
            let (shape, data) = logits.try_extract_tensor::<f32>().map_err(|e| {
                MemvidError::NerModelNotAvailable {
                    reason: format!("failed to extract logits: {}", e).into(),
                }
            })?;

            // Shape is iterable, convert to Vec
            let shape_vec: Vec<i64> = shape.iter().copied().collect();

            if shape_vec.len() != 3 {
                return Err(MemvidError::NerModelNotAvailable {
                    reason: format!("unexpected logits shape: {:?}", shape_vec).into(),
                });
            }

            let seq_len = shape_vec[1] as usize;
            let num_labels = shape_vec[2] as usize;

            // Helper to index into flat data: [batch, seq, labels] -> flat index
            let idx = |i: usize, j: usize| -> usize { i * num_labels + j };

            let mut entities = Vec::new();
            let mut current_entity: Option<(String, usize, usize, f32)> = None;

            for i in 0..seq_len {
                if i >= tokens.len() || i >= offsets.len() {
                    break;
                }

                // Skip special tokens
                let token = &tokens[i];
                if token == "[CLS]" || token == "[SEP]" || token == "[PAD]" {
                    // Finalize any current entity
                    if let Some((entity_type, start, end, conf)) = current_entity.take() {
                        if end > start && end <= original_text.len() {
                            let text = original_text[start..end].trim().to_string();
                            if !text.is_empty() {
                                entities.push(ExtractedEntity {
                                    text,
                                    entity_type,
                                    confidence: conf,
                                    byte_start: start,
                                    byte_end: end,
                                });
                            }
                        }
                    }
                    continue;
                }

                // Get prediction for this token
                let mut max_score = f32::NEG_INFINITY;
                let mut max_label = 0usize;

                for j in 0..num_labels {
                    let score = data[idx(i, j)];
                    if score > max_score {
                        max_score = score;
                        max_label = j;
                    }
                }

                // Apply softmax to get confidence
                let mut exp_sum = 0.0f32;
                for j in 0..num_labels {
                    exp_sum += (data[idx(i, j)] - max_score).exp();
                }
                let confidence = 1.0 / exp_sum;

                let label = NER_LABELS.get(max_label).unwrap_or(&"O");
                let (start_offset, end_offset) = offsets[i];

                if *label == "O" || confidence < min_confidence {
                    // End any current entity
                    if let Some((entity_type, start, end, conf)) = current_entity.take() {
                        if end > start && end <= original_text.len() {
                            let text = original_text[start..end].trim().to_string();
                            if !text.is_empty() {
                                entities.push(ExtractedEntity {
                                    text,
                                    entity_type,
                                    confidence: conf,
                                    byte_start: start,
                                    byte_end: end,
                                });
                            }
                        }
                    }
                } else if label.starts_with("B-") {
                    // Start new entity (end previous if any)
                    if let Some((entity_type, start, end, conf)) = current_entity.take() {
                        if end > start && end <= original_text.len() {
                            let text = original_text[start..end].trim().to_string();
                            if !text.is_empty() {
                                entities.push(ExtractedEntity {
                                    text,
                                    entity_type,
                                    confidence: conf,
                                    byte_start: start,
                                    byte_end: end,
                                });
                            }
                        }
                    }
                    let entity_type = label[2..].to_string(); // Remove "B-" prefix
                    current_entity = Some((entity_type, start_offset, end_offset, confidence));
                } else if label.starts_with("I-") {
                    // Continue entity
                    if let Some((ref entity_type, start, _, ref mut conf)) = current_entity {
                        let expected_type = &label[2..];
                        if entity_type == expected_type {
                            current_entity = Some((
                                entity_type.clone(),
                                start,
                                end_offset,
                                (*conf + confidence) / 2.0,
                            ));
                        }
                    }
                }
            }

            // Finalize last entity
            if let Some((entity_type, start, end, conf)) = current_entity {
                if end > start && end <= original_text.len() {
                    let text = original_text[start..end].trim().to_string();
                    if !text.is_empty() {
                        entities.push(ExtractedEntity {
                            text,
                            entity_type,
                            confidence: conf,
                            byte_start: start,
                            byte_end: end,
                        });
                    }
                }
            }

            Ok(entities)
        }

        /// Extract entities from a frame's content
        pub fn extract_from_frame(
            &mut self,
            frame_id: FrameId,
            content: &str,
        ) -> Result<FrameEntities> {
            let entities = self.extract(content)?;
            Ok(FrameEntities { frame_id, entities })
        }

        /// Get minimum confidence threshold
        pub fn min_confidence(&self) -> f32 {
            self.min_confidence
        }

        /// Get model path
        pub fn model_path(&self) -> &Path {
            &self.model_path
        }
    }

    impl std::fmt::Debug for NerModel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("NerModel")
                .field("model_path", &self.model_path)
                .field("min_confidence", &self.min_confidence)
                .finish()
        }
    }
}

// ============================================================================
// Stub Implementation (when feature is disabled)
// ============================================================================

#[cfg(not(feature = "logic_mesh"))]
#[allow(dead_code)]
pub struct NerModel {
    _private: (),
}

#[cfg(not(feature = "logic_mesh"))]
#[allow(dead_code)]
impl NerModel {
    pub fn load(
        _model_path: impl AsRef<Path>,
        _tokenizer_path: impl AsRef<Path>,
        _min_confidence: Option<f32>,
    ) -> Result<Self> {
        Err(MemvidError::FeatureUnavailable {
            feature: "logic_mesh",
        })
    }

    pub fn extract(&self, _text: &str) -> Result<Vec<ExtractedEntity>> {
        Err(MemvidError::FeatureUnavailable {
            feature: "logic_mesh",
        })
    }

    pub fn extract_from_frame(&self, _frame_id: FrameId, _content: &str) -> Result<FrameEntities> {
        Err(MemvidError::FeatureUnavailable {
            feature: "logic_mesh",
        })
    }
}

// ============================================================================
// Model Path Utilities
// ============================================================================

/// Get the expected path for the NER model in the models directory
pub fn ner_model_path(models_dir: &Path) -> PathBuf {
    models_dir.join(NER_MODEL_NAME).join("model.onnx")
}

/// Get the expected path for the NER tokenizer in the models directory
pub fn ner_tokenizer_path(models_dir: &Path) -> PathBuf {
    models_dir.join(NER_MODEL_NAME).join("tokenizer.json")
}

/// Check if NER model is installed
pub fn is_ner_model_installed(models_dir: &Path) -> bool {
    ner_model_path(models_dir).exists() && ner_tokenizer_path(models_dir).exists()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_kind_mapping() {
        let cases = vec![
            ("PER", EntityKind::Person),
            ("B-PER", EntityKind::Person),
            ("I-PER", EntityKind::Person),
            ("ORG", EntityKind::Organization),
            ("B-ORG", EntityKind::Organization),
            ("LOC", EntityKind::Location),
            ("B-LOC", EntityKind::Location),
            ("MISC", EntityKind::Other),
            ("B-MISC", EntityKind::Other),
            ("unknown", EntityKind::Other),
        ];

        for (entity_type, expected_kind) in cases {
            let entity = ExtractedEntity {
                text: "test".to_string(),
                entity_type: entity_type.to_string(),
                confidence: 0.9,
                byte_start: 0,
                byte_end: 4,
            };
            assert_eq!(
                entity.to_entity_kind(),
                expected_kind,
                "Failed for entity_type: {}",
                entity_type
            );
        }
    }

    #[test]
    fn test_model_info() {
        let info = default_ner_model_info();
        assert_eq!(info.name, NER_MODEL_NAME);
        assert!(info.is_default);
        assert!(info.size_mb > 200.0);
    }

    #[test]
    fn test_model_paths() {
        let models_dir = PathBuf::from("/tmp/models");
        let model_path = ner_model_path(&models_dir);
        let tokenizer_path = ner_tokenizer_path(&models_dir);

        assert!(model_path.to_string_lossy().contains("model.onnx"));
        assert!(tokenizer_path.to_string_lossy().contains("tokenizer.json"));
    }

    #[test]
    fn test_ner_labels() {
        assert_eq!(NER_LABELS.len(), 9);
        assert_eq!(NER_LABELS[0], "O");
        assert_eq!(NER_LABELS[1], "B-PER");
        assert_eq!(NER_LABELS[3], "B-ORG");
        assert_eq!(NER_LABELS[5], "B-LOC");
        assert_eq!(NER_LABELS[7], "B-MISC");
    }
}
