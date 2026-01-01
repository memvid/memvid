#!/usr/bin/env python3
"""
Phase 4.2.1: Multi-threaded Frame Ordering

Concurrent implementation of frame ordering optimization for large datasets.
Uses thread pools for signature extraction and parallel processing.

Features:
- Concurrent signature extraction across multiple threads
- Parallel progressive sorting at each resolution level
- Memory-efficient chunked processing for very large datasets
- Configurable thread pool sizes for optimal performance

This module is compatible with the existing frame_ordering.py API
but provides significant performance improvements for large datasets.
"""

import concurrent.futures
import threading
from typing import List, Dict, Tuple, Any, Optional
import time
import numpy as np
import cv2
from pathlib import Path
import queue
import math

from .frame_ordering import FrameOrderingOptimizer, ResolutionSignature


class ConcurrentFrameOrderingOptimizer:
    """Multi-threaded frame ordering optimizer for large datasets"""
    
    def __init__(self, 
                 power_base: int = 2,
                 max_resolution: int = 8,
                 start_with_1x1: bool = True,
                 max_workers: Optional[int] = None,
                 chunk_size: Optional[int] = None):
        """
        Initialize concurrent frame ordering optimizer
        
        Args:
            power_base: Base for resolution progression (2 or 3)
            max_resolution: Maximum resolution for signature extraction
            start_with_1x1: Whether to start with 1x1 resolution
            max_workers: Maximum number of worker threads (default: CPU count)
            chunk_size: Chunk size for batch processing (default: auto-calculate)
        """
        self.power_base = power_base
        self.max_resolution = max_resolution
        self.start_with_1x1 = start_with_1x1
        self.max_workers = max_workers or min(8, (threading.active_count() or 1) + 4)
        self.chunk_size = chunk_size
        
        # Thread-safe components
        self._lock = threading.Lock()
        self._signature_cache = {}
        
        # Create fallback optimizer for compatibility
        self._fallback_optimizer = FrameOrderingOptimizer(
            power_base=power_base,
            max_resolution=max_resolution,
            start_with_1x1=start_with_1x1
        )
    
    def _determine_chunk_size(self, frame_count: int) -> int:
        """Determine optimal chunk size based on frame count and worker count"""
        if self.chunk_size:
            return self.chunk_size
        
        # Aim for 2-4 chunks per worker for good load balancing
        chunks_per_worker = 3
        total_chunks = self.max_workers * chunks_per_worker
        
        # Ensure minimum chunk size of 10, maximum of 1000
        chunk_size = max(10, min(1000, frame_count // total_chunks))
        return chunk_size
    
    def _extract_signature_batch(self, frame_files: List[str], 
                                resolution: int) -> List[Tuple[int, np.ndarray]]:
        """Extract signatures for a batch of frames at given resolution"""
        results = []
        
        for i, frame_file in enumerate(frame_files):
            try:
                # Load and process image
                img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # Create empty signature for invalid frames
                    signature = np.zeros(resolution * resolution, dtype=np.uint8)
                else:
                    # Resize and extract signature
                    resized = cv2.resize(img, (resolution, resolution), 
                                       interpolation=cv2.INTER_AREA)
                    signature = resized.flatten()
                
                results.append((i, signature))
                
            except Exception as e:
                # Handle errors gracefully with empty signature
                empty_signature = np.zeros(resolution * resolution, dtype=np.uint8)
                results.append((i, empty_signature))
        
        return results
    
    def extract_signatures_at_resolution_concurrent(self, 
                                                   frame_files: List[str], 
                                                   resolution: int) -> List[np.ndarray]:
        """Extract signatures at given resolution using concurrent processing"""
        frame_count = len(frame_files)
        
        # For small datasets, use single-threaded processing
        if frame_count < 50 or self.max_workers == 1:
            # Load frames as numpy arrays for fallback
            frames = []
            for frame_file in frame_files:
                img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    frames.append(img)
            return self._fallback_optimizer.extract_signatures_at_resolution(
                frames, resolution
            )
        
        # Determine chunk size for batch processing
        chunk_size = self._determine_chunk_size(frame_count)
        
        # Create chunks of frame files
        chunks = []
        for i in range(0, frame_count, chunk_size):
            chunk = frame_files[i:i + chunk_size]
            chunks.append((chunk, i))  # Include starting index
        
        # Process chunks concurrently
        signatures = [None] * frame_count
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {}
            for chunk_frames, start_idx in chunks:
                future = executor.submit(
                    self._extract_signature_batch, 
                    chunk_frames, 
                    resolution
                )
                future_to_chunk[future] = start_idx
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                start_idx = future_to_chunk[future]
                try:
                    batch_results = future.result()
                    
                    # Place results in correct positions
                    for relative_idx, signature in batch_results:
                        absolute_idx = start_idx + relative_idx
                        signatures[absolute_idx] = signature
                        
                except Exception as e:
                    # Handle batch errors by filling with empty signatures
                    chunk_size_actual = min(chunk_size, frame_count - start_idx)
                    for i in range(chunk_size_actual):
                        signatures[start_idx + i] = np.zeros(resolution * resolution, dtype=np.uint8)
        
        return signatures
    
    def _parallel_sort_chunk(self, frame_indices: List[int], 
                           multi_resolution_signatures: List[List[np.ndarray]],
                           resolution_sequence: List[int]) -> List[int]:
        """Sort a chunk of frames using multi-resolution signatures"""
        
        def compare_frames(idx1: int, idx2: int) -> int:
            """Compare two frames by their multi-resolution signatures"""
            sigs1 = multi_resolution_signatures[idx1]
            sigs2 = multi_resolution_signatures[idx2]
            
            # Compare at each resolution level
            for i, resolution in enumerate(resolution_sequence):
                if i >= len(sigs1) or i >= len(sigs2):
                    break
                
                # Convert numpy arrays to tuples for lexicographic comparison
                sig1_tuple = tuple(sigs1[i].flatten())
                sig2_tuple = tuple(sigs2[i].flatten())
                
                if sig1_tuple < sig2_tuple:
                    return -1
                elif sig1_tuple > sig2_tuple:
                    return 1
            
            # If all signatures are equal, maintain original order
            return idx1 - idx2
        
        # Use Python's stable sort with custom comparison
        from functools import cmp_to_key
        sorted_indices = sorted(frame_indices, key=cmp_to_key(compare_frames))
        return sorted_indices
    
    def progressive_sort_concurrent(self, frame_files: List[str]) -> Tuple[List[int], Dict[str, Any]]:
        """
        Perform progressive sorting using concurrent processing
        
        Returns:
            Tuple of (optimized_order, metadata)
        """
        start_time = time.time()
        frame_count = len(frame_files)
        
        # For very small datasets, use single-threaded fallback
        if frame_count < 20:
            # Load frames as numpy arrays for fallback
            frames = []
            for frame_file in frame_files:
                img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    frames.append(img)
            
            frame_indices = list(range(len(frames)))
            optimized_order = self._fallback_optimizer.progressive_sort(frame_indices, frames)
            
            metadata = {
                "frame_count": len(frames),
                "optimization_method": "single_threaded_fallback",
                "optimization_time": time.time() - start_time,
                "frames_reordered": sum(1 for i, orig_i in enumerate(optimized_order) if i != orig_i)
            }
            
            return optimized_order, metadata
        
        metadata = {
            "frame_count": frame_count,
            "max_workers": self.max_workers,
            "optimization_method": "concurrent_progressive_sort"
        }
        
        try:
            # Generate resolution sequence
            resolution_sequence = self._fallback_optimizer.generate_resolution_sequence()
            metadata["resolution_sequence"] = resolution_sequence
            
            # Extract signatures at all resolutions concurrently
            print(f"🔄 Extracting signatures at {len(resolution_sequence)} resolutions using {self.max_workers} workers...")
            
            # Process each resolution level concurrently
            multi_resolution_signatures = [[] for _ in range(frame_count)]
            
            for resolution in resolution_sequence:
                resolution_start = time.time()
                
                signatures = self.extract_signatures_at_resolution_concurrent(
                    frame_files, resolution
                )
                
                # Add signatures to multi-resolution data
                for i, signature in enumerate(signatures):
                    multi_resolution_signatures[i].append(signature)
                
                resolution_time = time.time() - resolution_start
                print(f"   ✅ Resolution {resolution}x{resolution}: {resolution_time:.3f}s")
            
            # Perform progressive sorting with parallel chunk processing
            print(f"🔄 Progressive sorting with {self.max_workers} workers...")
            sort_start = time.time()
            
            # For large datasets, sort in parallel chunks then merge
            if frame_count > 1000:
                chunk_size = max(100, frame_count // (self.max_workers * 2))
                chunks = []
                for i in range(0, frame_count, chunk_size):
                    chunk_indices = list(range(i, min(i + chunk_size, frame_count)))
                    chunks.append(chunk_indices)
                
                # Sort chunks in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for chunk in chunks:
                        future = executor.submit(
                            self._parallel_sort_chunk,
                            chunk,
                            multi_resolution_signatures,
                            resolution_sequence
                        )
                        futures.append(future)
                    
                    # Collect sorted chunks
                    sorted_chunks = []
                    for future in concurrent.futures.as_completed(futures):
                        sorted_chunk = future.result()
                        sorted_chunks.append(sorted_chunk)
                
                # Merge sorted chunks (this could be optimized further with parallel merge)
                optimized_order = []
                for chunk in sorted_chunks:
                    optimized_order.extend(chunk)
                
                metadata["sorting_method"] = "parallel_chunk_sort"
                metadata["chunk_count"] = len(chunks)
                
            else:
                # For medium datasets, use single-threaded sort but concurrent signature extraction
                frame_indices = list(range(frame_count))
                optimized_order = self._parallel_sort_chunk(
                    frame_indices, 
                    multi_resolution_signatures,
                    resolution_sequence
                )
                metadata["sorting_method"] = "single_threaded_sort"
            
            sort_time = time.time() - sort_start
            total_time = time.time() - start_time
            
            # Calculate effectiveness metrics
            reorder_count = sum(1 for i, orig_i in enumerate(optimized_order) if i != orig_i)
            metadata.update({
                "frames_reordered": reorder_count,
                "reorder_percentage": reorder_count / frame_count * 100,
                "signature_extraction_time": total_time - sort_time,
                "sorting_time": sort_time,
                "optimization_time": total_time,
                "time_per_frame_ms": total_time / frame_count * 1000
            })
            
            print(f"✅ Concurrent optimization complete: {total_time:.3f}s ({reorder_count}/{frame_count} frames reordered)")
            
            return optimized_order, metadata
            
        except Exception as e:
            # Fallback to single-threaded processing
            print(f"⚠️  Concurrent processing failed, falling back to single-threaded: {e}")
            metadata["fallback_reason"] = str(e)
            
            # Load frames as numpy arrays for fallback
            frames = []
            for frame_file in frame_files:
                img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    frames.append(img)
            
            frame_indices = list(range(len(frames)))
            optimized_order = self._fallback_optimizer.progressive_sort(frame_indices, frames)
            
            metadata.update({
                "optimization_time": time.time() - start_time,
                "frames_reordered": sum(1 for i, orig_i in enumerate(optimized_order) if i != orig_i)
            })
            
            return optimized_order, metadata
    
    def optimize_frame_order(self, frame_files: List[str]) -> Dict[str, Any]:
        """
        Optimize frame order using concurrent processing
        
        Returns complete optimization result with metadata
        """
        if not frame_files:
            return {
                "optimized_order": [],
                "optimization_time": 0,
                "frames_reordered": 0,
                "method": "concurrent"
            }
        
        # Validate frame files exist
        valid_files = []
        for f in frame_files:
            if Path(f).exists():
                valid_files.append(f)
        
        if not valid_files:
            return {
                "optimized_order": [],
                "optimization_time": 0,
                "frames_reordered": 0,
                "error": "No valid frame files found"
            }
        
        # Perform concurrent optimization
        optimized_order, metadata = self.progressive_sort_concurrent(valid_files)
        
        # Format result
        result = {
            "optimized_order": optimized_order,
            "original_order": list(range(len(valid_files))),
            "frames_reordered": metadata.get("frames_reordered", 0),
            "optimization_time": metadata.get("optimization_time", 0),
            "method": "concurrent",
            "max_workers": self.max_workers,
            "metadata": metadata
        }
        
        return result


def create_concurrent_optimizer(config: Dict[str, Any]) -> ConcurrentFrameOrderingOptimizer:
    """
    Factory function to create concurrent optimizer from configuration
    
    Args:
        config: Configuration dictionary with optimization parameters
        
    Returns:
        Configured ConcurrentFrameOrderingOptimizer instance
    """
    return ConcurrentFrameOrderingOptimizer(
        power_base=config.get("power_base", 2),
        max_resolution=config.get("max_resolution", 8),
        start_with_1x1=config.get("start_with_1x1", True),
        max_workers=config.get("max_workers"),
        chunk_size=config.get("chunk_size")
    ) 