"""
MemvidRetriever - Fast semantic search, QR frame extraction, and context assembly
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import cv2

from .utils import (
    batch_extract_and_decode,
    extract_and_decode_cached
)
from .index import IndexManager
from .config import get_default_config

logger = logging.getLogger(__name__)


class MemvidRetriever:
    """Fast retrieval from QR code videos using semantic search"""
    
    def __init__(self, video_file: str, index_file: str, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize MemvidRetriever
        
        Args:
            video_file: Path to QR code video
            index_file: Path to index file
            config: Optional configuration
        """
        self.video_file = str(Path(video_file).absolute())
        self.index_file = str(Path(index_file).absolute())
        self.config = config or get_default_config()
        
        # Load index
        self.index_manager = IndexManager(self.config)
        self.index_manager.load(str(Path(index_file).with_suffix('')))
        
        # Cache for decoded frames
        self._frame_cache = {}
        self._cache_size = self.config["retrieval"]["cache_size"]
        
        # Verify video file
        self._verify_video()
        
        logger.info(f"Initialized retriever with {self.index_manager.get_stats()['total_chunks']} chunks")
    
    def _verify_video(self):
        """Verify video file is accessible"""
        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_file}")
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        logger.info(f"Video has {self.total_frames} frames at {self.fps} fps")
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search for relevant chunks using semantic search
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant text chunks
        """
        start_time = time.time()
        
        # Semantic search in index
        search_results = self.index_manager.search(query, top_k)
        
        # Extract unique frame numbers
        frame_numbers = list(set(result[2]["frame"] for result in search_results))
        
        # Decode frames in parallel
        decoded_frames = self._decode_frames_parallel(frame_numbers)
        
        # Extract text from decoded data
        results = []
        for chunk_id, distance, metadata in search_results:
            frame_num = metadata["frame"]
            if frame_num in decoded_frames:
                try:
                    chunk_data = json.loads(decoded_frames[frame_num])
                    results.append(chunk_data["text"])
                except (json.JSONDecodeError, KeyError):
                    # Fallback to metadata text
                    results.append(metadata["text"])
            else:
                # Fallback to metadata text if decode failed
                results.append(metadata["text"])
        
        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed:.3f}s for query: '{query[:50]}...'")
        
        return results
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[str]:
        """
        Get specific chunk by ID
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk text or None
        """
        metadata = self.index_manager.get_chunk_by_id(chunk_id)
        if metadata:
            frame_num = metadata["frame"]
            decoded = self._decode_single_frame(frame_num)
            if decoded:
                try:
                    chunk_data = json.loads(decoded)
                    return chunk_data["text"]
                except (json.JSONDecodeError, KeyError):
                    pass
            return metadata["text"]
        return None
    
    def _decode_single_frame(self, frame_number: int) -> Optional[str]:
        """Decode single frame with caching"""
        # Check cache
        if frame_number in self._frame_cache:
            return self._frame_cache[frame_number]
        
        # Decode frame
        result = extract_and_decode_cached(self.video_file, frame_number)
        
        # Update cache
        if result and len(self._frame_cache) < self._cache_size:
            self._frame_cache[frame_number] = result
        
        return result
    
    def _decode_frames_parallel(self, frame_numbers: List[int]) -> Dict[int, str]:
        """
        Decode multiple frames in parallel
        
        Args:
            frame_numbers: List of frame numbers to decode
            
        Returns:
            Dict mapping frame number to decoded data
        """
        # Check cache first
        results = {}
        uncached_frames = []
        
        for frame_num in frame_numbers:
            if frame_num in self._frame_cache:
                results[frame_num] = self._frame_cache[frame_num]
            else:
                uncached_frames.append(frame_num)
        
        if not uncached_frames:
            return results
        
        # Decode uncached frames in parallel
        max_workers = self.config["retrieval"]["max_workers"]
        decoded = batch_extract_and_decode(
            self.video_file, 
            uncached_frames, 
            max_workers=max_workers
        )
        
        # Update results and cache
        for frame_num, data in decoded.items():
            results[frame_num] = data
            if len(self._frame_cache) < self._cache_size:
                self._frame_cache[frame_num] = data
        
        return results
    
    def search_with_metadata(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search with full metadata
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of result dictionaries with text, score, and metadata
        """
        start_time = time.time()
        
        # Semantic search
        search_results = self.index_manager.search(query, top_k)
        
        # Extract frame numbers
        frame_numbers = list(set(result[2]["frame"] for result in search_results))
        
        # Decode frames
        decoded_frames = self._decode_frames_parallel(frame_numbers)
        
        # Build results with metadata
        results = []
        for chunk_id, distance, metadata in search_results:
            frame_num = metadata["frame"]
            
            # Get text from decoded frame or metadata
            if frame_num in decoded_frames:
                try:
                    chunk_data = json.loads(decoded_frames[frame_num])
                    text = chunk_data["text"]
                except (json.JSONDecodeError, KeyError):
                    text = metadata["text"]
            else:
                text = metadata["text"]
            
            results.append({
                "text": text,
                "score": 1.0 / (1.0 + distance),  # Convert distance to similarity score
                "chunk_id": chunk_id,
                "frame": frame_num,
                "metadata": metadata
            })
        
        elapsed = time.time() - start_time
        logger.info(f"Search with metadata completed in {elapsed:.3f}s")
        
        return results
    
    def get_context_window(self, chunk_id: int, window_size: int = 2) -> List[str]:
        """
        Get chunk with surrounding context
        
        Args:
            chunk_id: Central chunk ID
            window_size: Number of chunks before/after to include
            
        Returns:
            List of chunks in context window
        """
        chunks = []
        
        # Get chunks in window
        for offset in range(-window_size, window_size + 1):
            target_id = chunk_id + offset
            chunk = self.get_chunk_by_id(target_id)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def prefetch_frames(self, frame_numbers: List[int]):
        """
        Prefetch frames into cache for faster retrieval
        
        Args:
            frame_numbers: List of frame numbers to prefetch
        """
        # Only prefetch frames not in cache
        to_prefetch = [f for f in frame_numbers if f not in self._frame_cache]
        
        if to_prefetch:
            logger.info(f"Prefetching {len(to_prefetch)} frames...")
            decoded = self._decode_frames_parallel(to_prefetch)
            logger.info(f"Prefetched {len(decoded)} frames")
    
    def clear_cache(self):
        """Clear frame cache"""
        self._frame_cache.clear()
        extract_and_decode_cached.cache_clear()
        logger.info("Cleared frame cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "video_file": self.video_file,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "cache_size": len(self._frame_cache),
            "max_cache_size": self._cache_size,
            "index_stats": self.index_manager.get_stats()
        }