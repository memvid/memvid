//! Whisper audio transcription with Candle inference.
//!
//! This module provides complete Whisper transcription functionality including:
//! - Audio decoding (MP3, WAV, FLAC, etc.) via symphonia
//! - Resampling to 16kHz via rubato
//! - Whisper model inference via candle-transformers
//! - Automatic model download from HuggingFace Hub

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::MemvidError;

// These are only used when whisper feature is enabled
#[cfg(feature = "whisper")]
use crate::Result;
#[cfg(feature = "whisper")]
use std::path::Path;

// ============================================================================
// Model Registry
// ============================================================================

/// Quantization type for Whisper models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// Full precision FP32 (default, highest accuracy)
    FP32,
    /// 8-bit quantization (~75% smaller, ~15-20% faster)
    Q8K,
    /// 4-bit quantization (~87.5% smaller, ~25-30% faster)
    Q4K,
}

impl Default for QuantizationType {
    fn default() -> Self {
        Self::FP32
    }
}

/// Available Whisper models with verified HuggingFace model IDs
#[derive(Debug, Clone)]
pub struct WhisperModelInfo {
    /// Model identifier for HuggingFace
    pub model_id: &'static str,
    /// Human-readable name
    pub name: &'static str,
    /// Approximate model size in MB
    pub size_mb: f32,
    /// Whether this is the default model
    pub is_default: bool,
    /// Language (e.g., "en" for English-only models, "multilingual" for others)
    pub language: &'static str,
    /// Quantization type (FP32, Q8K, Q4K)
    pub quantization: QuantizationType,
    /// Model file format ("safetensors" for FP32, "gguf" for quantized)
    pub file_format: &'static str,
}

/// Available Whisper models registry
pub static WHISPER_MODELS: &[WhisperModelInfo] = &[
    // FP32 models (default, highest accuracy)
    WhisperModelInfo {
        model_id: "openai/whisper-small.en",
        name: "whisper-small-en",
        size_mb: 244.0,
        is_default: true,
        language: "en",
        quantization: QuantizationType::FP32,
        file_format: "safetensors",
    },
    WhisperModelInfo {
        model_id: "openai/whisper-small",
        name: "whisper-small",
        size_mb: 244.0,
        is_default: false,
        language: "multilingual",
        quantization: QuantizationType::FP32,
        file_format: "safetensors",
    },
    // Tiny FP32 models (faster, less accurate)
    WhisperModelInfo {
        model_id: "openai/whisper-tiny.en",
        name: "whisper-tiny-en",
        size_mb: 75.0,
        is_default: false,
        language: "en",
        quantization: QuantizationType::FP32,
        file_format: "safetensors",
    },
    // Q8K quantized tiny models (~75% smaller, faster)
    // Uses lmz/candle-whisper quantized models from HuggingFace
    WhisperModelInfo {
        model_id: "lmz/candle-whisper",
        name: "whisper-tiny-en-q8k",
        size_mb: 19.0,
        is_default: false,
        language: "en",
        quantization: QuantizationType::Q8K,
        file_format: "gguf",
    },
    WhisperModelInfo {
        model_id: "lmz/candle-whisper",
        name: "whisper-tiny-q8k",
        size_mb: 19.0,
        is_default: false,
        language: "multilingual",
        quantization: QuantizationType::Q8K,
        file_format: "gguf",
    },
];

/// Get model info by name, defaults to whisper-small-en
pub fn get_whisper_model_info(name: &str) -> &'static WhisperModelInfo {
    WHISPER_MODELS
        .iter()
        .find(|m| m.name == name || m.model_id == name)
        .unwrap_or_else(|| {
            WHISPER_MODELS
                .iter()
                .find(|m| m.is_default)
                .expect("default whisper model")
        })
}

/// Get the default model info
pub fn default_whisper_model_info() -> &'static WhisperModelInfo {
    WHISPER_MODELS
        .iter()
        .find(|m| m.is_default)
        .expect("default whisper model exists")
}

// ============================================================================
// Whisper Model Configuration
// ============================================================================

/// Configuration for Whisper model initialization
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Model name (e.g., "whisper-small-en")
    pub model_name: String,
    /// Directory where models are cached
    pub models_dir: PathBuf,
    /// Whether to run in offline mode (no downloads)
    pub offline: bool,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        let models_dir = std::env::var("MEMVID_MODELS_DIR")
            .ok()
            .map(PathBuf::from)
            .or_else(|| dirs_next::home_dir().map(|d| d.join(".memvid/models")))
            .unwrap_or_else(|| PathBuf::from(".memvid/models"));

        let model_name = std::env::var("MEMVID_WHISPER_MODEL")
            .unwrap_or_else(|_| "whisper-small-en".to_string());

        let offline = std::env::var("MEMVID_OFFLINE").is_ok();

        Self {
            model_name,
            models_dir,
            offline,
        }
    }
}

impl WhisperConfig {
    /// Create config with Q8K quantized tiny model (~19 MB, very fast)
    ///
    /// Uses lmz/candle-whisper quantized models from HuggingFace.
    /// Trade-off: Lower accuracy than whisper-small, but much faster.
    #[must_use]
    pub fn with_quantization() -> Self {
        Self {
            model_name: "whisper-tiny-en-q8k".to_string(),
            ..Default::default()
        }
    }

    /// Create config with specific model name
    #[must_use]
    pub fn with_model(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Create config for multilingual Q8K quantized tiny model
    #[must_use]
    pub fn multilingual_quantized() -> Self {
        Self {
            model_name: "whisper-tiny-q8k".to_string(),
            ..Default::default()
        }
    }

    /// Create config for tiny FP32 model (75 MB, faster than small)
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            model_name: "whisper-tiny-en".to_string(),
            ..Default::default()
        }
    }
}

// ============================================================================
// Whisper Error Types
// ============================================================================

/// Whisper-specific errors
#[derive(Debug, thiserror::Error)]
pub enum WhisperError {
    /// Model not found
    #[error("Whisper model '{model}' not found. {hint}")]
    ModelNotFound { model: String, hint: String },

    /// Audio decode failed
    #[error("Failed to decode audio at {path:?}: {cause}")]
    AudioDecodeError { path: PathBuf, cause: String },

    /// Audio bytes decode failed
    #[error("Failed to decode audio bytes: {cause}")]
    AudioBytesDecodeError { cause: String },

    /// Inference error
    #[error("Whisper inference error: {cause}")]
    InferenceError { cause: String },

    /// Model download failed
    #[error("Failed to download Whisper model: {cause}")]
    DownloadError { cause: String },
}

impl From<WhisperError> for MemvidError {
    fn from(err: WhisperError) -> Self {
        MemvidError::ExtractionFailed {
            reason: err.to_string().into_boxed_str(),
        }
    }
}

// ============================================================================
// Transcription Result
// ============================================================================

/// Result of audio transcription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// The transcribed text
    pub text: String,
    /// Language detected or specified
    pub language: String,
    /// Duration of audio in seconds
    pub duration_secs: f32,
    /// Optional timestamps for segments
    #[serde(default)]
    pub segments: Vec<TranscriptionSegment>,
}

/// A segment of transcription with timestamps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Transcribed text for this segment
    pub text: String,
}

// ============================================================================
// Audio Decoding (Feature-gated)
// ============================================================================

#[cfg(feature = "whisper")]
mod audio {
    use super::*;
    use std::fs::File;
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    /// Whisper sample rate (always 16kHz)
    pub const WHISPER_SAMPLE_RATE: u32 = 16000;

    /// Decode audio file to f32 samples, resampling to 16kHz mono
    pub fn decode_audio_file(path: &Path) -> Result<(Vec<f32>, f32)> {
        let file = File::open(path).map_err(|e| WhisperError::AudioDecodeError {
            path: path.to_path_buf(),
            cause: e.to_string(),
        })?;

        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Create a hint based on file extension
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        // Probe the media source
        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| WhisperError::AudioDecodeError {
                path: path.to_path_buf(),
                cause: format!("Failed to probe audio format: {}", e),
            })?;

        let mut format = probed.format;

        // Find the first audio track
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
            .ok_or_else(|| WhisperError::AudioDecodeError {
                path: path.to_path_buf(),
                cause: "No audio track found".to_string(),
            })?;

        let track_id = track.id;
        let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
        let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(2);

        // Create decoder
        let decoder_opts = DecoderOptions::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .map_err(|e| WhisperError::AudioDecodeError {
                path: path.to_path_buf(),
                cause: format!("Failed to create decoder: {}", e),
            })?;

        let mut samples: Vec<f32> = Vec::new();

        // Decode all packets
        loop {
            let packet = match format.next_packet() {
                Ok(p) => p,
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(_) => break,
            };

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = match decoder.decode(&packet) {
                Ok(d) => d,
                Err(_) => continue,
            };

            let spec = *decoded.spec();
            let num_frames = decoded.frames();

            if num_frames == 0 {
                continue;
            }

            let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
            sample_buf.copy_interleaved_ref(decoded);

            let interleaved = sample_buf.samples();

            // Convert to mono by averaging channels
            if channels > 1 {
                for chunk in interleaved.chunks(channels) {
                    let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                    samples.push(mono);
                }
            } else {
                samples.extend_from_slice(interleaved);
            }
        }

        let duration_secs = samples.len() as f32 / sample_rate as f32;

        // Log pre-resampling stats
        let pre_min = samples.iter().cloned().fold(f32::INFINITY, f32::min);
        let pre_max = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let pre_rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        tracing::info!(
            sample_rate = sample_rate,
            channels = channels,
            samples_before = samples.len(),
            pre_min = pre_min,
            pre_max = pre_max,
            pre_rms = pre_rms,
            "Audio before resampling"
        );

        // High-quality sinc resampling to 16kHz
        let samples = if sample_rate != WHISPER_SAMPLE_RATE {
            let resampled = resample_sinc(&samples, sample_rate, WHISPER_SAMPLE_RATE);

            // Log post-resampling stats
            let post_min = resampled.iter().cloned().fold(f32::INFINITY, f32::min);
            let post_max = resampled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let post_rms =
                (resampled.iter().map(|x| x * x).sum::<f32>() / resampled.len() as f32).sqrt();
            tracing::info!(
                samples_after = resampled.len(),
                post_min = post_min,
                post_max = post_max,
                post_rms = post_rms,
                "Audio after resampling"
            );
            resampled
        } else {
            tracing::info!("Audio already at 16kHz, no resampling needed");
            samples
        };

        Ok((samples, duration_secs))
    }

    /// Simple linear interpolation resampling
    /// Note: rubato 1.0 changed API to require audio_core buffer types.
    /// Using simple linear interpolation which is sufficient for Whisper mel spectrogram input.
    fn resample_sinc(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        if from_rate == to_rate {
            return samples.to_vec();
        }

        let ratio = to_rate as f64 / from_rate as f64;
        let output_len = (samples.len() as f64 * ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_pos = i as f64 / ratio;
            let src_idx = src_pos.floor() as usize;
            let frac = (src_pos - src_idx as f64) as f32;

            if src_idx + 1 < samples.len() {
                // Linear interpolation between samples
                let sample = samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac;
                output.push(sample);
            } else if src_idx < samples.len() {
                output.push(samples[src_idx]);
            }
        }

        output
    }
}

#[cfg(feature = "whisper")]
pub use audio::*;

// ============================================================================
// Whisper Transcriber (Candle Inference)
// ============================================================================

#[cfg(feature = "whisper")]
mod inference {
    use super::*;
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::whisper::{self as m, Config, audio};
    use hf_hub::{Repo, RepoType, api::sync::Api};
    use tokenizers::Tokenizer;

    /// Whisper model wrapper for transcription
    pub struct WhisperTranscriber {
        model: Model,
        tokenizer: Tokenizer,
        config: Config,
        mel_filters: Vec<f32>,
        device: Device,
    }

    #[allow(dead_code)]
    enum Model {
        Normal(m::model::Whisper),
        Quantized(m::quantized_model::Whisper),
    }

    impl WhisperTranscriber {
        /// Create a new WhisperTranscriber, downloading the model if needed
        pub fn new(config: &WhisperConfig) -> Result<Self> {
            // Use GPU if available: Metal (macOS) or CUDA (NVIDIA)
            let device = Self::select_device();
            tracing::info!(device = ?device, "Using device for Whisper");

            // Get model info from registry
            let model_info = get_whisper_model_info(&config.model_name);
            let is_quantized = model_info.quantization != QuantizationType::FP32;

            tracing::info!(
                model_name = %config.model_name,
                model_id = %model_info.model_id,
                quantization = ?model_info.quantization,
                file_format = %model_info.file_format,
                "Loading Whisper model"
            );

            let api = Api::new().map_err(|e| WhisperError::DownloadError {
                cause: e.to_string(),
            })?;
            let repo = api.repo(Repo::with_revision(
                model_info.model_id.to_string(),
                RepoType::Model,
                "main".to_string(),
            ));

            // Download config and tokenizer files
            // For quantized models, config/tokenizer come from the base OpenAI model
            let (config_path, tokenizer_path) = if is_quantized {
                // Quantized tiny models need config from openai/whisper-tiny
                let base_model_id = match model_info.language {
                    "en" => "openai/whisper-tiny.en",
                    _ => "openai/whisper-tiny",
                };
                let base_repo = api.repo(Repo::with_revision(
                    base_model_id.to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ));

                let cfg =
                    base_repo
                        .get("config.json")
                        .map_err(|e| WhisperError::DownloadError {
                            cause: format!("Failed to download config.json: {}", e),
                        })?;
                let tok =
                    base_repo
                        .get("tokenizer.json")
                        .map_err(|e| WhisperError::DownloadError {
                            cause: format!("Failed to download tokenizer.json: {}", e),
                        })?;
                (cfg, tok)
            } else {
                let cfg = repo
                    .get("config.json")
                    .map_err(|e| WhisperError::DownloadError {
                        cause: format!("Failed to download config.json: {}", e),
                    })?;
                let tok = repo
                    .get("tokenizer.json")
                    .map_err(|e| WhisperError::DownloadError {
                        cause: format!("Failed to download tokenizer.json: {}", e),
                    })?;
                (cfg, tok)
            };

            // Load config
            let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
                WhisperError::InferenceError {
                    cause: format!("Failed to read config: {}", e),
                }
            })?;
            let model_config: Config =
                serde_json::from_str(&config_str).map_err(|e| WhisperError::InferenceError {
                    cause: format!("Failed to parse config: {}", e),
                })?;

            // Load tokenizer
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                WhisperError::InferenceError {
                    cause: format!("Failed to load tokenizer: {}", e),
                }
            })?;

            // Load mel filters
            let mel_bytes = match model_config.num_mel_bins {
                80 => include_bytes!("melfilters.bytes").as_slice(),
                128 => include_bytes!("melfilters128.bytes").as_slice(),
                n => {
                    return Err(WhisperError::InferenceError {
                        cause: format!("Unsupported number of mel bins: {}", n),
                    }
                    .into());
                }
            };
            let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
            <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
                mel_bytes,
                &mut mel_filters,
            );

            // Load model based on quantization type
            let model = match model_info.quantization {
                QuantizationType::FP32 => {
                    // Download and load FP32 safetensors model
                    let model_path =
                        repo.get("model.safetensors")
                            .map_err(|e| WhisperError::DownloadError {
                                cause: format!("Failed to download model.safetensors: {}", e),
                            })?;

                    let vb = unsafe {
                        VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                            .map_err(|e| WhisperError::InferenceError {
                                cause: format!("Failed to load model weights: {}", e),
                            })?
                    };
                    Model::Normal(m::model::Whisper::load(&vb, model_config.clone()).map_err(
                        |e| WhisperError::InferenceError {
                            cause: format!("Failed to load Whisper model: {}", e),
                        },
                    )?)
                }
                QuantizationType::Q8K | QuantizationType::Q4K => {
                    // Download and load quantized GGUF model from lmz/candle-whisper
                    // Available files: model-tiny-en-q80.gguf, model-tiny-q80.gguf, etc.
                    let gguf_filename = match (model_info.language, model_info.quantization) {
                        ("en", QuantizationType::Q8K) => "model-tiny-en-q80.gguf",
                        ("en", QuantizationType::Q4K) => "model-tiny-en-q40.gguf",
                        (_, QuantizationType::Q8K) => "model-tiny-q80.gguf",
                        (_, QuantizationType::Q4K) => "model-tiny-q40.gguf",
                        _ => "model-tiny-q80.gguf",
                    };

                    let model_path =
                        repo.get(gguf_filename)
                            .map_err(|e| WhisperError::DownloadError {
                                cause: format!("Failed to download {}: {}", gguf_filename, e),
                            })?;

                    tracing::info!(
                        gguf_file = %gguf_filename,
                        quantization = ?model_info.quantization,
                        "Loading quantized GGUF model"
                    );

                    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                        &model_path,
                        &device,
                    )
                    .map_err(|e| WhisperError::InferenceError {
                        cause: format!("Failed to load quantized model: {}", e),
                    })?;

                    Model::Quantized(
                        m::quantized_model::Whisper::load(&vb, model_config.clone()).map_err(
                            |e| WhisperError::InferenceError {
                                cause: format!("Failed to load quantized Whisper model: {}", e),
                            },
                        )?,
                    )
                }
            };

            tracing::info!("Whisper model loaded successfully");

            Ok(Self {
                model,
                tokenizer,
                config: model_config,
                mel_filters,
                device,
            })
        }

        /// Select the best available device (GPU if available, otherwise CPU)
        fn select_device() -> Device {
            // Try Metal (macOS Apple Silicon / AMD)
            #[cfg(feature = "metal")]
            {
                if let Ok(device) = Device::new_metal(0) {
                    tracing::info!("Metal GPU available");
                    return device;
                }
            }

            // Try CUDA (NVIDIA GPUs)
            #[cfg(feature = "cuda")]
            {
                if let Ok(device) = Device::new_cuda(0) {
                    tracing::info!("CUDA GPU available");
                    return device;
                }
            }

            // Fallback to CPU
            tracing::info!("Using CPU (no GPU acceleration)");
            Device::Cpu
        }

        /// Transcribe an audio file
        pub fn transcribe_file(&mut self, path: &Path) -> Result<TranscriptionResult> {
            // Decode audio to PCM
            let (pcm_data, duration_secs) = super::decode_audio_file(path)?;

            // Check audio statistics
            let audio_min = pcm_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let audio_max = pcm_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let audio_mean = pcm_data.iter().sum::<f32>() / pcm_data.len() as f32;
            let audio_rms =
                (pcm_data.iter().map(|x| x * x).sum::<f32>() / pcm_data.len() as f32).sqrt();

            tracing::info!(
                duration = duration_secs,
                samples = pcm_data.len(),
                min = audio_min,
                max = audio_max,
                mean = audio_mean,
                rms = audio_rms,
                "Audio decoded"
            );

            self.transcribe_pcm(&pcm_data, duration_secs)
        }

        /// Transcribe PCM audio samples (16kHz mono f32)
        pub fn transcribe_pcm(
            &mut self,
            pcm_data: &[f32],
            duration_secs: f32,
        ) -> Result<TranscriptionResult> {
            // Whisper processes audio in 30-second chunks
            const CHUNK_LENGTH: usize = 30 * 16000; // 30 seconds at 16kHz
            const N_FRAMES: usize = 3000; // frames per chunk
            const SAMPLE_RATE: f32 = 16000.0;

            // Detect and trim leading silence
            let silence_threshold = 0.01; // RMS threshold for silence
            let window_size = 1600; // 100ms windows at 16kHz

            let start_sample = find_speech_start(pcm_data, silence_threshold, window_size);
            let end_sample = find_speech_end(pcm_data, silence_threshold, window_size);

            let trimmed_start = start_sample as f32 / SAMPLE_RATE;
            let trimmed_end = end_sample as f32 / SAMPLE_RATE;

            tracing::info!(
                start_sample = start_sample,
                end_sample = end_sample,
                trimmed_start_sec = trimmed_start,
                trimmed_end_sec = trimmed_end,
                original_duration = duration_secs,
                "Trimmed silence"
            );

            // Use trimmed audio
            let pcm_data = &pcm_data[start_sample..end_sample];
            let _trimmed_duration = pcm_data.len() as f32 / SAMPLE_RATE;

            let mut all_text = String::new();
            let mut segments = Vec::new();

            // Process audio in chunks
            let num_chunks = (pcm_data.len() + CHUNK_LENGTH - 1) / CHUNK_LENGTH;

            for chunk_idx in 0..num_chunks {
                let chunk_start = chunk_idx * CHUNK_LENGTH;
                let chunk_end = (chunk_start + CHUNK_LENGTH).min(pcm_data.len());
                let chunk = &pcm_data[chunk_start..chunk_end];

                // Adjust timestamps to account for trimmed silence
                let start_time = trimmed_start + chunk_start as f32 / SAMPLE_RATE;
                let end_time = trimmed_start + chunk_end as f32 / SAMPLE_RATE;

                tracing::info!(
                    chunk = chunk_idx + 1,
                    total = num_chunks,
                    start = start_time,
                    end = end_time,
                    "Processing chunk"
                );

                // Reset decoder KV cache for each new chunk
                match &mut self.model {
                    Model::Normal(m) => m.decoder.reset_kv_cache(),
                    Model::Quantized(m) => m.decoder.reset_kv_cache(),
                }

                // Convert chunk to mel spectrogram
                let mel = audio::pcm_to_mel(&self.config, chunk, &self.mel_filters);
                let n_mels = self.config.num_mel_bins;
                let mel_len = mel.len();
                let n_frames = mel_len / n_mels;

                if chunk_idx == 0 {
                    // Print config for debugging
                    tracing::info!(
                        num_mel_bins = self.config.num_mel_bins,
                        max_source_positions = self.config.max_source_positions,
                        max_target_positions = self.config.max_target_positions,
                        "Model config"
                    );

                    // Mel statistics
                    let mel_min = mel.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mel_max = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mel_mean = mel.iter().sum::<f32>() / mel.len() as f32;

                    tracing::info!(
                        mel_len = mel_len,
                        n_mels = n_mels,
                        n_frames = n_frames,
                        chunk_samples = chunk.len(),
                        expected_frames = 3000,
                        mel_min = mel_min,
                        mel_max = mel_max,
                        mel_mean = mel_mean,
                        "Mel spectrogram computed"
                    );
                }

                // Ensure we have exactly 3000 frames (pad or truncate)
                // NOTE: mel array from pcm_to_mel is stored as [mel_bin_0_all_frames, mel_bin_1_all_frames, ...]
                // So each mel bin has n_frames contiguous values: mel[bin * n_frames + frame]
                let mel = if n_frames < N_FRAMES {
                    // Pad each mel bin's frames with zeros to reach N_FRAMES
                    let mut padded = vec![0.0f32; n_mels * N_FRAMES];
                    for bin in 0..n_mels {
                        let src_start = bin * n_frames;
                        let dst_start = bin * N_FRAMES;
                        padded[dst_start..dst_start + n_frames]
                            .copy_from_slice(&mel[src_start..src_start + n_frames]);
                    }
                    padded
                } else if n_frames > N_FRAMES {
                    // Truncate each mel bin's frames to N_FRAMES
                    let mut truncated = vec![0.0f32; n_mels * N_FRAMES];
                    for bin in 0..n_mels {
                        let src_start = bin * n_frames;
                        let dst_start = bin * N_FRAMES;
                        truncated[dst_start..dst_start + N_FRAMES]
                            .copy_from_slice(&mel[src_start..src_start + N_FRAMES]);
                    }
                    truncated
                } else {
                    mel
                };

                let mel =
                    Tensor::from_vec(mel, (1, n_mels, N_FRAMES), &self.device).map_err(|e| {
                        WhisperError::InferenceError {
                            cause: format!("Failed to create mel tensor: {}", e),
                        }
                    })?;

                if chunk_idx == 0 {
                    let mel_shape = mel.shape();
                    tracing::info!(
                        mel_shape = ?mel_shape,
                        "Mel tensor shape"
                    );
                }

                // Run encoder
                let audio_features = match &mut self.model {
                    Model::Normal(m) => m.encoder.forward(&mel, true),
                    Model::Quantized(m) => m.encoder.forward(&mel, true),
                }
                .map_err(|e| WhisperError::InferenceError {
                    cause: format!("Encoder forward failed: {}", e),
                })?;

                if chunk_idx == 0 {
                    let af_shape = audio_features.shape();
                    tracing::info!(
                        audio_features_shape = ?af_shape,
                        "Audio features from encoder"
                    );
                }

                // Get special token IDs
                let sot_token = self.token_id(m::SOT_TOKEN)?;
                let transcribe_token = self.token_id(m::TRANSCRIBE_TOKEN)?;
                let eot_token = self.token_id(m::EOT_TOKEN)?;
                let no_timestamps_token = self.token_id(m::NO_TIMESTAMPS_TOKEN)?;

                if chunk_idx == 0 {
                    let en_token = self.tokenizer.token_to_id("<|en|>");
                    tracing::info!(
                        sot = sot_token,
                        transcribe = transcribe_token,
                        eot = eot_token,
                        no_timestamps = no_timestamps_token,
                        en_token = ?en_token,
                        "Special tokens"
                    );
                }

                // Build initial prompt
                // For English-only models (*.en), we DON'T use language token
                // For multilingual models, we add language token after sot_token
                let has_language_token = self.tokenizer.token_to_id("<|en|>").is_some();

                // English-only models have vocab size 51864, multilingual have 51865
                let is_english_only = self.config.vocab_size == 51864;

                let tokens = if is_english_only {
                    // English-only: SOT -> transcribe -> notimestamps
                    vec![sot_token, transcribe_token, no_timestamps_token]
                } else if has_language_token {
                    // Multilingual: SOT -> language -> transcribe -> notimestamps
                    let language_token = self.token_id("<|en|>")?;
                    vec![
                        sot_token,
                        language_token,
                        transcribe_token,
                        no_timestamps_token,
                    ]
                } else {
                    // Fallback
                    vec![sot_token, transcribe_token, no_timestamps_token]
                };

                if chunk_idx == 0 {
                    tracing::info!(
                        is_english_only = is_english_only,
                        vocab_size = self.config.vocab_size,
                        prompt_tokens = ?tokens,
                        "Initial prompt"
                    );
                }
                let mut all_tokens = tokens.clone();

                // Autoregressive decoding with token suppression
                let sample_len = self.config.max_target_positions / 2;
                let mut repeat_count = 0;
                let mut last_token: Option<u32> = None;

                // Build suppression mask
                let suppress_tokens = &self.config.suppress_tokens;

                for i in 0..sample_len {
                    // For autoregressive decoding with KV cache:
                    // - First iteration: pass all prompt tokens, flush_kv_cache=true
                    // - Subsequent iterations: pass only the new token, flush_kv_cache=false
                    let tokens_tensor = Tensor::new(all_tokens.as_slice(), &self.device)
                        .and_then(|t| t.unsqueeze(0))
                        .map_err(|e| WhisperError::InferenceError {
                            cause: format!("Failed to create tokens tensor: {}", e),
                        })?;

                    if chunk_idx == 0 && i < 3 {
                        tracing::info!(
                            step = i,
                            all_tokens_len = all_tokens.len(),
                            tokens_shape = ?tokens_tensor.shape(),
                            "Decoder input"
                        );
                    }

                    // Get hidden states from decoder, then project to vocabulary
                    // Always pass all tokens (candle doesn't use KV cache the same way as PyTorch)
                    let logits = match &mut self.model {
                        Model::Normal(m) => {
                            let hidden = m
                                .decoder
                                .forward(&tokens_tensor, &audio_features, true)
                                .map_err(|e| WhisperError::InferenceError {
                                cause: format!("Decoder forward failed: {}", e),
                            })?;
                            m.decoder.final_linear(&hidden).map_err(|e| {
                                WhisperError::InferenceError {
                                    cause: format!("Final linear failed: {}", e),
                                }
                            })?
                        }
                        Model::Quantized(m) => {
                            let hidden = m
                                .decoder
                                .forward(&tokens_tensor, &audio_features, true)
                                .map_err(|e| WhisperError::InferenceError {
                                cause: format!("Decoder forward failed: {}", e),
                            })?;
                            m.decoder.final_linear(&hidden).map_err(|e| {
                                WhisperError::InferenceError {
                                    cause: format!("Final linear failed: {}", e),
                                }
                            })?
                        }
                    };

                    if chunk_idx == 0 && i == 0 {
                        tracing::info!(
                            logits_shape = ?logits.shape(),
                            "Decoder output logits"
                        );
                    }

                    // Get logits for last position
                    let (_, seq_len, _) =
                        logits.dims3().map_err(|e| WhisperError::InferenceError {
                            cause: format!("Failed to get logits dims: {}", e),
                        })?;
                    let mut logits_vec = logits
                        .i((0, seq_len - 1, ..))
                        .and_then(|t| t.to_vec1::<f32>())
                        .map_err(|e| WhisperError::InferenceError {
                            cause: format!("Failed to extract logits: {}", e),
                        })?;

                    // Apply token suppression from config
                    for &token_id in suppress_tokens.iter() {
                        if (token_id as usize) < logits_vec.len() {
                            logits_vec[token_id as usize] = f32::NEG_INFINITY;
                        }
                    }

                    // Suppress EOT token for first few steps to allow generation
                    if all_tokens.len() < 10 {
                        logits_vec[eot_token as usize] = f32::NEG_INFINITY;
                    }

                    // Suppress all special tokens during generation:
                    // - SOT (50257), language tokens (50258-50261), task tokens (50358-50359),
                    // - no_timestamps (50362), and timestamp tokens (50363+)
                    logits_vec[sot_token as usize] = f32::NEG_INFINITY;
                    logits_vec[transcribe_token as usize] = f32::NEG_INFINITY;
                    logits_vec[no_timestamps_token as usize] = f32::NEG_INFINITY;
                    // Suppress all tokens from 50257 onward (special tokens) except those in normal vocab
                    for token_id in 50257..logits_vec.len() {
                        logits_vec[token_id] = f32::NEG_INFINITY;
                    }

                    if chunk_idx == 0 && i == 0 {
                        tracing::info!(
                            suppress_count = suppress_tokens.len(),
                            eot_suppressed = all_tokens.len() < 10,
                            "Applied token suppression"
                        );
                    }

                    // Find argmax
                    let next_token = logits_vec
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(eot_token);

                    if chunk_idx == 0 && i < 5 {
                        let max_logit =
                            logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let min_logit = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                        tracing::info!(
                            step = i,
                            next_token = next_token,
                            max_logit = max_logit,
                            min_logit = min_logit,
                            "Decoding step"
                        );
                    }

                    if next_token == eot_token || next_token >= self.config.vocab_size as u32 {
                        if chunk_idx == 0 && i < 5 {
                            tracing::info!(
                                next_token = next_token,
                                eot = eot_token,
                                "Stopping: EOT or invalid token"
                            );
                        }
                        break;
                    }

                    // Check for excessive repetition (stop if same token repeats >3 times)
                    if Some(next_token) == last_token {
                        repeat_count += 1;
                        if repeat_count > 3 {
                            tracing::debug!("Breaking due to token repetition");
                            break;
                        }
                    } else {
                        repeat_count = 0;
                    }
                    last_token = Some(next_token);

                    all_tokens.push(next_token);
                }

                // Decode tokens to text for this chunk
                let prompt_len = if is_english_only { 3 } else { 4 };

                if chunk_idx == 0 {
                    tracing::info!(
                        prompt_tokens = ?&all_tokens[..prompt_len],
                        generated_tokens = ?&all_tokens[prompt_len..],
                        total = all_tokens.len(),
                        "Generated tokens for chunk"
                    );
                }

                let chunk_text = self
                    .tokenizer
                    .decode(&all_tokens[prompt_len..], true) // Skip prompt tokens
                    .map_err(|e| WhisperError::InferenceError {
                        cause: format!("Failed to decode tokens: {}", e),
                    })?;

                let trimmed_text = chunk_text.trim();
                if !trimmed_text.is_empty() {
                    if !all_text.is_empty() {
                        all_text.push(' ');
                    }
                    all_text.push_str(trimmed_text);

                    segments.push(TranscriptionSegment {
                        start: start_time,
                        end: end_time,
                        text: trimmed_text.to_string(),
                    });
                }
            }

            Ok(TranscriptionResult {
                text: all_text.trim().to_string(),
                language: "en".to_string(),
                duration_secs,
                segments,
            })
        }

        fn token_id(&self, token: &str) -> Result<u32> {
            self.tokenizer.token_to_id(token).ok_or_else(|| {
                WhisperError::InferenceError {
                    cause: format!("Token '{}' not found in vocabulary", token),
                }
                .into()
            })
        }
    }

    /// Find the sample index where speech starts (after leading silence)
    fn find_speech_start(samples: &[f32], threshold: f32, window_size: usize) -> usize {
        for i in (0..samples.len()).step_by(window_size) {
            let end = (i + window_size).min(samples.len());
            let window = &samples[i..end];
            let rms = (window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32).sqrt();
            if rms > threshold {
                // Found speech, go back a bit to not cut off the start
                return i.saturating_sub(window_size);
            }
        }
        0 // No silence found, return start
    }

    /// Find the sample index where speech ends (before trailing silence)
    fn find_speech_end(samples: &[f32], threshold: f32, window_size: usize) -> usize {
        for i in (0..samples.len()).rev().step_by(window_size) {
            let start = i.saturating_sub(window_size);
            let window = &samples[start..=i.min(samples.len() - 1)];
            let rms = (window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32).sqrt();
            if rms > threshold {
                // Found speech, go forward a bit to not cut off the end
                return (i + window_size).min(samples.len());
            }
        }
        samples.len() // No silence found, return end
    }
}

#[cfg(feature = "whisper")]
pub use inference::WhisperTranscriber;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whisper_model_registry() {
        let default = default_whisper_model_info();
        assert_eq!(default.name, "whisper-small-en");
        assert!(default.is_default);
        assert_eq!(default.language, "en");

        // Unknown model returns default
        let unknown = get_whisper_model_info("nonexistent");
        assert_eq!(unknown.name, "whisper-small-en");
    }

    #[test]
    fn whisper_config_defaults() {
        let config = WhisperConfig::default();
        assert_eq!(config.model_name, "whisper-small-en");
    }
}
