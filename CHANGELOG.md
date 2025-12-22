# Changelog

All notable changes to Memvid will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **MP4V Codec Optimization**: Set MP4V as the default codec for 1.84x faster encoding compared to H.265/HEVC
  - Uses OpenCV's native MP4V codec (no FFmpeg subprocess required)
  - Optimized for speed while maintaining good compression
  - Smaller file sizes compared to uncompressed formats

- **Automatic Parallel Processing**: Smart multi-core QR generation for large datasets
  - Automatic detection with configurable threshold (default: 500+ chunks)
  - Provides 1.69x speedup for large files (tested with 8,131 chunks)
  - Windows-optimized to overcome process spawn overhead (~120s)
  - Manual control available via `enable_parallel` parameter
  - Performance benchmarks:
    - Small files (<500 chunks): Serial mode is faster
    - Large files (8000+ chunks): Parallel mode provides 1.69x speedup
    - 151MB PDF (8,131 chunks): Parallel saves 37.79 minutes (40.8% faster)

- **VP9/WebM Codec Support**: Added VP9_PARAMETERS for WebM video format
  - Available as alternative codec option
  - Note: MP4V is 1.84x faster for typical use cases

### Changed
- Updated default codec from H.265 to MP4V in `config.py`
- Modified `build_video()` method to support parallel processing with `enable_parallel='auto'` parameter
- Updated `_generate_qr_frames()` to use ProcessPoolExecutor for CPU-bound parallel tasks

### Performance
- MP4V encoding: 1.84x faster than H.265/HEVC
- Parallel QR generation (8000+ chunks): 1.69x speedup
- Real-world test (151MB PDF, 8,131 chunks):
  - Serial: 5560.03s (92.67 min)
  - Parallel: 3293.01s (54.88 min)
  - Speedup: 40.8% faster

## [0.1.0] - Previous Release

### Added
- Initial release with QR code video encoding
- Semantic search and retrieval
- PDF support
- Chat interface
- Multiple codec support (H.265, H.264, AV1)
