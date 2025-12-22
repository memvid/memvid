"""
Fast configuration for memvid - optimized for speed
"""

# Performance mode
PERFORMANCE_MODE = 'fast'

# QR settings - smaller/faster
QR_VERSION = 20
QR_BOX_SIZE = 3

# Use MP4V codec (OpenCV-based, much faster than FFmpeg)
VIDEO_CODEC = 'mp4v'

# More parallel workers
MAX_WORKERS = 8
