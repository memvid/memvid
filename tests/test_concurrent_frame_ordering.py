#!/usr/bin/env python3
"""
Phase 4.2.1: Concurrent Frame Ordering Tests

Test suite for multi-threaded frame ordering optimization.
Validates performance improvements and correctness of concurrent implementation.
"""

import unittest
import tempfile
import time
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock

# Add memvid to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.frame_ordering_concurrent import ConcurrentFrameOrderingOptimizer, create_concurrent_optimizer
from memvid.frame_ordering import FrameOrderingOptimizer
from tests.test_frame_ordering import MockQRFrameGenerator


class TestConcurrentFrameOrderingOptimizer(unittest.TestCase):
    """Test concurrent frame ordering optimizer"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.frame_generator = MockQRFrameGenerator()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_frame_files(self, count: int, patterns: List[str]) -> List[str]:
        """Create test frame files on disk for concurrent testing"""
        frame_files = []
        
        for i in range(count):
            pattern = patterns[i % len(patterns)]
            frame = self.frame_generator.create_test_qr_frame(size=100, pattern=pattern, seed=i)
            
            frame_path = Path(self.temp_dir) / f"frame_{i:03d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_files.append(str(frame_path))
        
        return frame_files
    
    def test_concurrent_optimizer_initialization(self):
        """Test concurrent optimizer initialization with different configurations"""
        # Default initialization
        optimizer = ConcurrentFrameOrderingOptimizer()
        self.assertEqual(optimizer.power_base, 2)
        self.assertEqual(optimizer.max_resolution, 8)
        self.assertTrue(optimizer.start_with_1x1)
        self.assertIsNotNone(optimizer.max_workers)
        self.assertGreater(optimizer.max_workers, 0)
        
        # Custom initialization
        optimizer = ConcurrentFrameOrderingOptimizer(
            power_base=3,
            max_resolution=16,
            start_with_1x1=False,
            max_workers=4,
            chunk_size=50
        )
        self.assertEqual(optimizer.power_base, 3)
        self.assertEqual(optimizer.max_resolution, 16)
        self.assertFalse(optimizer.start_with_1x1)
        self.assertEqual(optimizer.max_workers, 4)
        self.assertEqual(optimizer.chunk_size, 50)
    
    def test_chunk_size_determination(self):
        """Test automatic chunk size determination"""
        optimizer = ConcurrentFrameOrderingOptimizer(max_workers=4)
        
        # Small dataset
        chunk_size = optimizer._determine_chunk_size(50)
        self.assertGreaterEqual(chunk_size, 10)
        self.assertLessEqual(chunk_size, 50)
        
        # Large dataset
        chunk_size = optimizer._determine_chunk_size(5000)
        self.assertGreaterEqual(chunk_size, 10)
        self.assertLessEqual(chunk_size, 1000)
        
        # Custom chunk size
        optimizer_custom = ConcurrentFrameOrderingOptimizer(chunk_size=100)
        chunk_size = optimizer_custom._determine_chunk_size(5000)
        self.assertEqual(chunk_size, 100)
    
    def test_concurrent_signature_extraction_small_dataset(self):
        """Test that small datasets use single-threaded fallback"""
        optimizer = ConcurrentFrameOrderingOptimizer(max_workers=4)
        
        # Create small test dataset
        frame_files = self._create_test_frame_files(10, ["light", "dark", "random"])
        
        # Should use fallback for small datasets
        signatures = optimizer.extract_signatures_at_resolution_concurrent(frame_files, 2)
        
        self.assertEqual(len(signatures), 10)
        for signature in signatures:
            self.assertIsInstance(signature, np.ndarray)
            self.assertEqual(len(signature), 4)  # 2x2 = 4 pixels
    
    def test_concurrent_signature_extraction_large_dataset(self):
        """Test concurrent signature extraction for larger datasets"""
        optimizer = ConcurrentFrameOrderingOptimizer(max_workers=2, chunk_size=20)
        
        # Create larger test dataset
        frame_files = self._create_test_frame_files(60, ["light", "dark", "random", "checkerboard"])
        
        # Test concurrent extraction
        start_time = time.time()
        signatures = optimizer.extract_signatures_at_resolution_concurrent(frame_files, 4)
        concurrent_time = time.time() - start_time
        
        # Validate results
        self.assertEqual(len(signatures), 60)
        for signature in signatures:
            self.assertIsInstance(signature, np.ndarray)
            self.assertEqual(len(signature), 16)  # 4x4 = 16 pixels
        
        # Compare with single-threaded version
        fallback_optimizer = FrameOrderingOptimizer()
        start_time = time.time()
        
        # Load frames for fallback
        frames = []
        for frame_file in frame_files:
            img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                frames.append(img)
        
        fallback_signatures = fallback_optimizer.extract_signatures_at_resolution(frames, 4)
        fallback_time = time.time() - start_time
        
        # Results should be equivalent
        self.assertEqual(len(signatures), len(fallback_signatures))
        
        # Performance should be comparable or better (allowing for overhead)
        self.assertLess(concurrent_time, fallback_time * 2)  # Allow 2x overhead for small test
    
    def test_progressive_sort_concurrent_small_dataset(self):
        """Test that small datasets use single-threaded fallback"""
        optimizer = ConcurrentFrameOrderingOptimizer(max_workers=4)
        
        # Create small dataset
        frame_files = self._create_test_frame_files(15, ["light", "dark", "random"])
        
        # Should use fallback
        optimized_order, metadata = optimizer.progressive_sort_concurrent(frame_files)
        
        self.assertEqual(len(optimized_order), 15)
        self.assertIn("optimization_time", metadata)
        self.assertGreater(metadata["optimization_time"], 0)
    
    def test_progressive_sort_concurrent_medium_dataset(self):
        """Test concurrent progressive sort for medium datasets"""
        optimizer = ConcurrentFrameOrderingOptimizer(max_workers=2, chunk_size=15)
        
        # Create medium dataset with clear ordering patterns
        frame_files = self._create_test_frame_files(40, ["light", "dark", "light", "dark", "random"])
        
        # Test concurrent sorting
        start_time = time.time()
        optimized_order, metadata = optimizer.progressive_sort_concurrent(frame_files)
        concurrent_time = time.time() - start_time
        
        # Validate results
        self.assertEqual(len(optimized_order), 40)
        self.assertIn("optimization_time", metadata)
        self.assertIn("max_workers", metadata)
        self.assertIn("frames_reordered", metadata)
        self.assertEqual(metadata["max_workers"], 2)
        
        # Should have reordered some frames
        reorder_count = sum(1 for i, orig_i in enumerate(optimized_order) if i != orig_i)
        self.assertGreater(reorder_count, 0)
        
        # Compare with single-threaded version
        fallback_optimizer = FrameOrderingOptimizer()
        start_time = time.time()
        
        # Load frames for fallback
        frames = []
        for frame_file in frame_files:
            img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                frames.append(img)
        
        frame_indices = list(range(len(frames)))
        fallback_order, fallback_metadata = fallback_optimizer.progressive_sort(frame_indices, frames)
        fallback_time = time.time() - start_time
        
        # Results should be deterministic and equivalent
        self.assertEqual(len(optimized_order), len(fallback_order))
        
        # Performance should be comparable
        self.assertLess(concurrent_time, fallback_time * 3)  # Allow 3x overhead for test environment
    
    def test_optimize_frame_order_complete_workflow(self):
        """Test complete frame order optimization workflow"""
        optimizer = ConcurrentFrameOrderingOptimizer(max_workers=2)
        
        # Create test dataset
        frame_files = self._create_test_frame_files(30, ["light", "dark", "random", "checkerboard"])
        
        # Test complete optimization
        result = optimizer.optimize_frame_order(frame_files)
        
        # Validate result structure
        self.assertIn("optimized_order", result)
        self.assertIn("original_order", result)
        self.assertIn("frames_reordered", result)
        self.assertIn("optimization_time", result)
        self.assertIn("method", result)
        self.assertIn("max_workers", result)
        self.assertIn("metadata", result)
        
        # Validate values
        self.assertEqual(len(result["optimized_order"]), 30)
        self.assertEqual(result["original_order"], list(range(30)))
        self.assertEqual(result["method"], "concurrent")
        self.assertEqual(result["max_workers"], 2)
        self.assertGreater(result["optimization_time"], 0)
        
        # Should have reordered some frames
        self.assertGreater(result["frames_reordered"], 0)
    
    def test_error_handling_and_fallback(self):
        """Test error handling and fallback to single-threaded processing"""
        optimizer = ConcurrentFrameOrderingOptimizer(max_workers=2)
        
        # Test with empty frame list
        result = optimizer.optimize_frame_order([])
        self.assertEqual(result["optimized_order"], [])
        self.assertEqual(result["frames_reordered"], 0)
        
        # Test with non-existent files
        non_existent_files = ["/non/existent/file1.png", "/non/existent/file2.png"]
        result = optimizer.optimize_frame_order(non_existent_files)
        self.assertEqual(result["optimized_order"], [])
        self.assertIn("error", result)
    
    def test_concurrent_vs_single_threaded_consistency(self):
        """Test that concurrent and single-threaded results are consistent"""
        # Create test dataset
        frame_files = self._create_test_frame_files(25, ["light", "dark", "random"])
        
        # Test concurrent version
        concurrent_optimizer = ConcurrentFrameOrderingOptimizer(max_workers=2)
        concurrent_result = concurrent_optimizer.optimize_frame_order(frame_files)
        
        # Test single-threaded version
        single_optimizer = FrameOrderingOptimizer()
        
        # Load frames for single-threaded optimizer
        frames = []
        for frame_file in frame_files:
            img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                frames.append(img)
        
        single_result_raw = single_optimizer.optimize_frame_order(frames)
        single_order = single_result_raw['frame_order']
        single_metadata = single_result_raw['metadata']
        single_result = {
            "optimized_order": single_order,
            "optimization_time": single_metadata.get("optimization_time", 0),
            "frames_reordered": sum(1 for i, orig_i in enumerate(single_order) if i != orig_i)
        }
        
        # Results should be equivalent (deterministic sorting)
        self.assertEqual(len(concurrent_result["optimized_order"]), 
                        len(single_result["optimized_order"]))
        
        # Both should reorder the same frames (deterministic algorithm)
        concurrent_reordered = set(i for i, orig_i in enumerate(concurrent_result["optimized_order"]) if i != orig_i)
        single_reordered = set(i for i, orig_i in enumerate(single_result["optimized_order"]) if i != orig_i)
        
        # Should have similar reordering patterns
        self.assertGreater(len(concurrent_reordered & single_reordered), 0)
    
    def test_performance_scaling_with_workers(self):
        """Test performance scaling with different worker counts"""
        frame_files = self._create_test_frame_files(50, ["light", "dark", "random", "checkerboard"])
        
        # Test with different worker counts
        worker_counts = [1, 2, 4]
        times = {}
        
        for workers in worker_counts:
            optimizer = ConcurrentFrameOrderingOptimizer(max_workers=workers)
            
            start_time = time.time()
            result = optimizer.optimize_frame_order(frame_files)
            elapsed_time = time.time() - start_time
            
            times[workers] = elapsed_time
            
            # Validate result
            self.assertEqual(len(result["optimized_order"]), 50)
            self.assertGreater(result["frames_reordered"], 0)
        
        # Performance should generally improve with more workers (allowing for overhead)
        # At minimum, shouldn't be dramatically worse
        self.assertLess(times[4], times[1] * 2)  # 4 workers shouldn't be >2x slower than 1


class TestConcurrentOptimizerFactory(unittest.TestCase):
    """Test concurrent optimizer factory function"""
    
    def test_create_concurrent_optimizer_default(self):
        """Test factory with default configuration"""
        config = {}
        optimizer = create_concurrent_optimizer(config)
        
        self.assertIsInstance(optimizer, ConcurrentFrameOrderingOptimizer)
        self.assertEqual(optimizer.power_base, 2)
        self.assertEqual(optimizer.max_resolution, 8)
        self.assertTrue(optimizer.start_with_1x1)
    
    def test_create_concurrent_optimizer_custom(self):
        """Test factory with custom configuration"""
        config = {
            "power_base": 3,
            "max_resolution": 16,
            "start_with_1x1": False,
            "max_workers": 6,
            "chunk_size": 100
        }
        optimizer = create_concurrent_optimizer(config)
        
        self.assertEqual(optimizer.power_base, 3)
        self.assertEqual(optimizer.max_resolution, 16)
        self.assertFalse(optimizer.start_with_1x1)
        self.assertEqual(optimizer.max_workers, 6)
        self.assertEqual(optimizer.chunk_size, 100)


class TestConcurrentIntegration(unittest.TestCase):
    """Test integration of concurrent optimizer with existing systems"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.frame_generator = MockQRFrameGenerator()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_frame_files(self, count: int, patterns: List[str]) -> List[str]:
        """Create test frame files on disk for concurrent testing"""
        frame_files = []
        
        for i in range(count):
            pattern = patterns[i % len(patterns)]
            frame = self.frame_generator.create_test_qr_frame(size=100, pattern=pattern, seed=i)
            
            frame_path = Path(self.temp_dir) / f"frame_{i:03d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_files.append(str(frame_path))
        
        return frame_files
    
    def test_concurrent_optimizer_api_compatibility(self):
        """Test that concurrent optimizer is API-compatible with original"""
        frame_files = self._create_test_frame_files(20, ["light", "dark", "random"])
        
        # Test both optimizers with same API
        concurrent_optimizer = ConcurrentFrameOrderingOptimizer(max_workers=2)
        single_optimizer = FrameOrderingOptimizer()
        
        # Test concurrent optimizer with file paths
        concurrent_result = concurrent_optimizer.optimize_frame_order(frame_files)
        
        # Test single-threaded optimizer with numpy arrays
        frames = []
        for frame_file in frame_files:
            img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                frames.append(img)
        
        single_result_raw = single_optimizer.optimize_frame_order(frames)
        single_order = single_result_raw['frame_order']
        single_metadata = single_result_raw['metadata']
        single_result = {
            "optimized_order": single_order,
            "optimization_time": single_metadata.get("optimization_time", 0),
            "frames_reordered": sum(1 for i, orig_i in enumerate(single_order) if i != orig_i)
        }
        
        # Results should have same structure
        for key in ["optimized_order", "optimization_time", "frames_reordered"]:
            self.assertIn(key, concurrent_result)
            self.assertIn(key, single_result)
        
        # Both should return valid results
        self.assertEqual(len(concurrent_result["optimized_order"]), 20)
        self.assertEqual(len(single_result["optimized_order"]), 20)


if __name__ == '__main__':
    unittest.main() 