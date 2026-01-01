#!/usr/bin/env python3
"""
Test-Driven Development tests for Phase 3: Video Encoder Integration

Tests the integration of frame ordering optimization with the MemvidEncoder.build_video() method.
Follows TDD approach: Write failing tests first, then implement minimal code to pass.
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
import cv2
import numpy as np
import sys
from unittest.mock import Mock, patch, MagicMock

# Add memvid to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.encoder import MemvidEncoder
from memvid.retriever import MemvidRetriever
from memvid.frame_ordering import FrameOrderingOptimizer


class TestFrameOrderingIntegration(unittest.TestCase):
    """Test integration of frame ordering with MemvidEncoder (Phase 3)"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_chunks = [
            "First chunk of test data for video encoding",
            "Second chunk with different content patterns", 
            "Third chunk containing various information",
            "Fourth chunk for comprehensive testing",
            "Fifth chunk to complete the test dataset"
        ]
        
    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_encoder_has_frame_ordering_parameter(self):
        """Test that MemvidEncoder.build_video() accepts enable_frame_ordering parameter"""
        encoder = MemvidEncoder()
        encoder.add_chunks(self.test_chunks)
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        index_path = os.path.join(self.temp_dir, "test_index.json")
        
        # This should NOT raise an error when enable_frame_ordering is passed
        try:
            result = encoder.build_video(
                video_path, 
                index_path, 
                enable_frame_ordering=True,  # New parameter
                show_progress=False
            )
            self.assertIsInstance(result, dict)
        except TypeError as e:
            if "enable_frame_ordering" in str(e):
                self.fail("build_video() should accept enable_frame_ordering parameter")
            else:
                raise
    
    def test_frame_ordering_configuration_parameters(self):
        """Test that frame ordering configuration parameters are accepted"""
        encoder = MemvidEncoder()
        encoder.add_chunks(self.test_chunks)
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        index_path = os.path.join(self.temp_dir, "test_index.json")
        
        # Should accept frame ordering configuration
        try:
            result = encoder.build_video(
                video_path,
                index_path,
                enable_frame_ordering=True,
                frame_ordering_config={
                    "power_base": 2,
                    "max_resolution": 32,
                    "start_with_1x1": True
                },
                show_progress=False
            )
            self.assertIsInstance(result, dict)
        except TypeError as e:
            if "frame_ordering_config" in str(e):
                self.fail("build_video() should accept frame_ordering_config parameter")
            else:
                raise
    
    def test_frame_ordering_disabled_by_default(self):
        """Test that frame ordering is disabled by default (backward compatibility)"""
        encoder = MemvidEncoder()
        encoder.add_chunks(self.test_chunks)
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        index_path = os.path.join(self.temp_dir, "test_index.json")
        
        # Default behavior should work unchanged
        result = encoder.build_video(video_path, index_path, show_progress=False)
        
        # Should complete successfully
        self.assertIsInstance(result, dict)
        self.assertTrue(os.path.exists(video_path))
        self.assertTrue(os.path.exists(index_path))
        
        # Should NOT have frame ordering metadata by default
        self.assertNotIn("frame_ordering", result)
    
    def test_frame_ordering_generates_metadata(self):
        """Test that enabling frame ordering generates optimization metadata"""
        encoder = MemvidEncoder()
        encoder.add_chunks(self.test_chunks)
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        index_path = os.path.join(self.temp_dir, "test_index.json")
        
        result = encoder.build_video(
            video_path,
            index_path,
            enable_frame_ordering=True,
            show_progress=False
        )
        
        # Should include frame ordering metadata
        self.assertIn("frame_ordering", result)
        
        frame_ordering_meta = result["frame_ordering"]
        self.assertIn("optimized_order", frame_ordering_meta)
        self.assertIn("original_order", frame_ordering_meta)
        self.assertIn("optimization_config", frame_ordering_meta)
        self.assertIn("optimization_time", frame_ordering_meta)
        
        # Orders should be valid permutations
        optimized_order = frame_ordering_meta["optimized_order"]
        original_order = frame_ordering_meta["original_order"]
        
        self.assertEqual(len(optimized_order), len(self.test_chunks))
        self.assertEqual(set(optimized_order), set(range(len(self.test_chunks))))
        self.assertEqual(original_order, list(range(len(self.test_chunks))))
    
    def test_frame_ordering_actually_reorders_frames(self):
        """Test that frame ordering actually changes the frame sequence"""
        encoder = MemvidEncoder()
        
        # Create chunks with very different characteristics that should sort differently
        diverse_chunks = [
            "AAAAAAAA" * 100,  # Very repetitive (light QR)  
            "ZZZZZZZZ" * 100,  # Different repetitive (light QR)
            "The quick brown fox jumps over the lazy dog. " * 20,  # Mixed content (medium QR)
            "1234567890!@#$%^&*()_+{}|:<>?[]\\;'\",./" * 15,  # Dense symbols (dark QR)
            json.dumps({"key": "value", "array": [1,2,3,4,5]}) * 30  # JSON structure (dark QR)
        ]
        
        encoder.add_chunks(diverse_chunks)
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        index_path = os.path.join(self.temp_dir, "test_index.json")
        
        result = encoder.build_video(
            video_path,
            index_path,
            enable_frame_ordering=True,
            show_progress=False
        )
        
        # Should actually reorder frames (not stay in original order)
        frame_ordering_meta = result["frame_ordering"]
        optimized_order = frame_ordering_meta["optimized_order"]
        original_order = frame_ordering_meta["original_order"]
        
        self.assertNotEqual(optimized_order, original_order)
    
    def test_frame_ordering_preserves_retrieval_accuracy(self):
        """Test that frame ordering doesn't break content retrieval"""
        encoder = MemvidEncoder()
        encoder.add_chunks(self.test_chunks)
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        index_path = os.path.join(self.temp_dir, "test_index.json")
        
        # Build video with frame ordering
        result = encoder.build_video(
            video_path,
            index_path,
            enable_frame_ordering=True,
            show_progress=False
        )
        
        # Should be able to retrieve content correctly
        retriever = MemvidRetriever(video_path, index_path)
        
        # Test semantic search still works
        search_results = retriever.search("test data", top_k=3)
        self.assertGreater(len(search_results), 0)
        
        # Test that we can retrieve specific chunks
        for i, original_chunk in enumerate(self.test_chunks):
            # Find this chunk in search results
            found = False
            for chunk_text in search_results:
                if original_chunk in chunk_text or chunk_text in original_chunk:
                    found = True
                    break
            
            if not found:
                # Try a more targeted search
                words = original_chunk.split()[:3]  # Use first few words
                if words:
                    targeted_results = retriever.search(" ".join(words), top_k=5)
                    for chunk_text in targeted_results:
                        if original_chunk in chunk_text or chunk_text in original_chunk:
                            found = True
                            break
            
            self.assertTrue(found, f"Could not retrieve chunk {i}: {original_chunk[:50]}...")


class TestFrameOrderingIndexMapping(unittest.TestCase):
    """Test the frame-to-chunk index mapping with frame ordering (Phase 3)"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_chunks = [
            "First test chunk",
            "Second test chunk", 
            "Third test chunk",
            "Fourth test chunk"
        ]
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_index_contains_frame_order_mapping(self):
        """Test that index file contains frame order mapping when ordering is enabled"""
        encoder = MemvidEncoder()
        encoder.add_chunks(self.test_chunks)
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        index_path = os.path.join(self.temp_dir, "test_index.json")
        
        encoder.build_video(
            video_path,
            index_path,
            enable_frame_ordering=True,
            show_progress=False
        )
        
        # Load index file and check for frame mapping
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Should contain frame order mapping
        self.assertIn("frame_order_map", index_data)
        
        frame_order_map = index_data["frame_order_map"]
        self.assertIn("original_to_video", frame_order_map)
        self.assertIn("video_to_original", frame_order_map)
        
        # Mappings should be valid
        original_to_video = frame_order_map["original_to_video"]
        video_to_original = frame_order_map["video_to_original"]
        
        self.assertEqual(len(original_to_video), len(self.test_chunks))
        self.assertEqual(len(video_to_original), len(self.test_chunks))
        
        # Should be inverse mappings
        for orig_idx, video_idx in enumerate(original_to_video):
            self.assertEqual(video_to_original[video_idx], orig_idx)
    
    def test_index_without_frame_ordering_no_mapping(self):
        """Test that index file doesn't contain mapping when ordering is disabled"""
        encoder = MemvidEncoder()
        encoder.add_chunks(self.test_chunks)
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        index_path = os.path.join(self.temp_dir, "test_index.json")
        
        encoder.build_video(video_path, index_path, show_progress=False)
        
        # Load index file
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Should NOT contain frame order mapping
        self.assertNotIn("frame_order_map", index_data)


class TestFrameOrderingPerformance(unittest.TestCase):
    """Test performance aspects of frame ordering integration (Phase 3)"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_frame_ordering_time_measurement(self):
        """Test that frame ordering time is measured and reported"""
        encoder = MemvidEncoder()
        
        # Create larger dataset for meaningful timing
        large_chunks = [f"Test chunk {i} with some content to encode" * 10 for i in range(20)]
        encoder.add_chunks(large_chunks)
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        index_path = os.path.join(self.temp_dir, "test_index.json")
        
        result = encoder.build_video(
            video_path,
            index_path,
            enable_frame_ordering=True,
            show_progress=False
        )
        
        # Should measure and report timing
        self.assertIn("frame_ordering", result)
        frame_ordering_meta = result["frame_ordering"]
        self.assertIn("optimization_time", frame_ordering_meta)
        
        optimization_time = frame_ordering_meta["optimization_time"]
        self.assertIsInstance(optimization_time, (int, float))
        self.assertGreater(optimization_time, 0)
        self.assertLess(optimization_time, 60)  # Should be reasonable
    
    def test_frame_ordering_no_significant_performance_impact(self):
        """Test that frame ordering doesn't significantly slow down encoding"""
        encoder1 = MemvidEncoder()
        encoder2 = MemvidEncoder()
        
        test_chunks = [f"Performance test chunk {i}" * 5 for i in range(10)]
        encoder1.add_chunks(test_chunks)
        encoder2.add_chunks(test_chunks)
        
        video_path1 = os.path.join(self.temp_dir, "test_video1.mp4")
        index_path1 = os.path.join(self.temp_dir, "test_index1.json")
        video_path2 = os.path.join(self.temp_dir, "test_video2.mp4")
        index_path2 = os.path.join(self.temp_dir, "test_index2.json")
        
        import time
        
        # Time without frame ordering
        start1 = time.time()
        result1 = encoder1.build_video(video_path1, index_path1, show_progress=False)
        time1 = time.time() - start1
        
        # Time with frame ordering
        start2 = time.time()
        result2 = encoder2.build_video(
            video_path2, 
            index_path2, 
            enable_frame_ordering=True,
            show_progress=False
        )
        time2 = time.time() - start2
        
        # Frame ordering shouldn't add more than 5x the base time
        self.assertLess(time2, time1 * 5)
        
        # Both should complete successfully
        self.assertTrue(os.path.exists(video_path1))
        self.assertTrue(os.path.exists(video_path2))


class TestFrameOrderingErrorHandling(unittest.TestCase):
    """Test error handling for frame ordering integration (Phase 3)"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_frame_ordering_config_raises_error(self):
        """Test that invalid frame ordering config raises appropriate error"""
        encoder = MemvidEncoder()
        encoder.add_chunks(["test chunk"])
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        index_path = os.path.join(self.temp_dir, "test_index.json")
        
        # Invalid power_base should raise error
        with self.assertRaises(ValueError):
            encoder.build_video(
                video_path,
                index_path,
                enable_frame_ordering=True,
                frame_ordering_config={"power_base": 1},  # Invalid
                show_progress=False
            )
        
        # Invalid max_resolution should raise error
        with self.assertRaises(ValueError):
            encoder.build_video(
                video_path,
                index_path,
                enable_frame_ordering=True,
                frame_ordering_config={"max_resolution": -5},  # Invalid
                show_progress=False
            )
    
    def test_frame_ordering_graceful_fallback(self):
        """Test graceful fallback when frame ordering fails"""
        encoder = MemvidEncoder()
        encoder.add_chunks(["test chunk"])
        
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        index_path = os.path.join(self.temp_dir, "test_index.json")
        
        # Mock frame ordering to fail
        with patch('memvid.frame_ordering.FrameOrderingOptimizer.optimize_frame_order') as mock_optimize:
            mock_optimize.side_effect = Exception("Simulated frame ordering failure")
            
            # Should still complete video encoding (with warning)
            result = encoder.build_video(
                video_path,
                index_path,
                enable_frame_ordering=True,
                show_progress=False
            )
            
            # Video should still be created
            self.assertTrue(os.path.exists(video_path))
            self.assertIsInstance(result, dict)
            
            # Should indicate frame ordering was skipped
            if "frame_ordering" in result:
                self.assertIn("error", result["frame_ordering"])


if __name__ == '__main__':
    unittest.main() 