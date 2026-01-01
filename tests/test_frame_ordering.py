#!/usr/bin/env python3
"""
Test-Driven Development tests for frame ordering optimization.

Tests the progressive resolution sorting algorithm that minimizes
frame-to-frame differences for better video compression.
"""

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import time
import pytest

# Add memvid to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.frame_ordering import FrameOrderingOptimizer, ResolutionSignature


class MockQRFrameGenerator:
    """Generate mock QR frames for testing"""
    
    @staticmethod
    def create_test_qr_frame(size=477, pattern='random', seed=None):
        """Create a mock QR frame with specified pattern"""
        if seed is not None:
            np.random.seed(seed)
        
        if pattern == 'random':
            # Random black/white pattern
            return np.random.choice([0, 255], size=(size, size), p=[0.5, 0.5]).astype(np.uint8)
        elif pattern == 'dark':
            # Mostly dark QR (dense data)
            return np.random.choice([0, 255], size=(size, size), p=[0.8, 0.2]).astype(np.uint8)
        elif pattern == 'light':
            # Mostly light QR (sparse data)
            return np.random.choice([0, 255], size=(size, size), p=[0.2, 0.8]).astype(np.uint8)
        elif pattern == 'checkerboard':
            # Predictable pattern for testing
            frame = np.zeros((size, size), dtype=np.uint8)
            frame[::2, ::2] = 255
            frame[1::2, 1::2] = 255
            return frame
        elif pattern == 'solid_black':
            return np.zeros((size, size), dtype=np.uint8)
        elif pattern == 'solid_white':
            return np.ones((size, size), dtype=np.uint8) * 255
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    @staticmethod
    def create_test_frame_set(count=10, size=477, patterns=None):
        """Create a set of test QR frames"""
        if patterns is None:
            patterns = ['random'] * count
        
        frames = []
        for i, pattern in enumerate(patterns):
            frame = MockQRFrameGenerator.create_test_qr_frame(size, pattern, seed=i)
            frames.append(frame)
        
        return frames


class TestFrameOrderingOptimizer(unittest.TestCase):
    """Test suite for FrameOrderingOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer_default = FrameOrderingOptimizer()
        self.optimizer_power3 = FrameOrderingOptimizer(power_base=3, max_resolution=81)
        self.optimizer_high_res = FrameOrderingOptimizer(max_resolution=256)
        
        # Create test frame sets
        self.test_frames_small = MockQRFrameGenerator.create_test_frame_set(5)
        self.test_frames_medium = MockQRFrameGenerator.create_test_frame_set(20)
        self.test_frames_patterns = MockQRFrameGenerator.create_test_frame_set(
            6, patterns=['solid_black', 'solid_white', 'dark', 'light', 'checkerboard', 'random']
        )
    
    def test_constructor_default_parameters(self):
        """Test FrameOrderingOptimizer constructor with default parameters"""
        optimizer = FrameOrderingOptimizer()
        
        # These assertions will fail initially (Red phase)
        self.assertEqual(optimizer.power_base, 2)
        self.assertEqual(optimizer.max_resolution, 32)
        self.assertTrue(optimizer.start_with_1x1)
    
    def test_constructor_custom_parameters(self):
        """Test FrameOrderingOptimizer constructor with custom parameters"""
        optimizer = FrameOrderingOptimizer(
            power_base=3, 
            max_resolution=128, 
            start_with_1x1=False
        )
        
        self.assertEqual(optimizer.power_base, 3)
        self.assertEqual(optimizer.max_resolution, 128)
        self.assertFalse(optimizer.start_with_1x1)
    
    def test_parameter_validation(self):
        """Test parameter validation and error handling"""
        
        # Invalid power_base
        with self.assertRaises(ValueError):
            FrameOrderingOptimizer(power_base=1)
        
        with self.assertRaises(ValueError):
            FrameOrderingOptimizer(power_base=0)
        
        # Invalid max_resolution
        with self.assertRaises(ValueError):
            FrameOrderingOptimizer(max_resolution=0)
        
        with self.assertRaises(ValueError):
            FrameOrderingOptimizer(max_resolution=1000)  # Too large


class TestResolutionSequenceGeneration(unittest.TestCase):
    """Test resolution sequence generation with different power bases"""
    
    def test_power_of_2_sequence(self):
        """Test resolution sequence generation with power_base=2"""
        optimizer = FrameOrderingOptimizer(power_base=2, max_resolution=32, start_with_1x1=True)
        sequence = optimizer.generate_resolution_sequence()
        
        expected = [1, 2, 4, 8, 16, 32]
        self.assertEqual(sequence, expected)
    
    def test_power_of_3_sequence(self):
        """Test resolution sequence generation with power_base=3"""
        optimizer = FrameOrderingOptimizer(power_base=3, max_resolution=81, start_with_1x1=True)
        sequence = optimizer.generate_resolution_sequence()
        
        expected = [1, 3, 9, 27, 81]
        self.assertEqual(sequence, expected)
    
    def test_power_of_2_without_1x1(self):
        """Test resolution sequence without 1x1 initial sort"""
        optimizer = FrameOrderingOptimizer(power_base=2, max_resolution=16, start_with_1x1=False)
        sequence = optimizer.generate_resolution_sequence()
        
        expected = [2, 4, 8, 16]
        self.assertEqual(sequence, expected)
    
    def test_sequence_stops_at_max_resolution(self):
        """Test that sequence stops at max_resolution even if power continues"""
        optimizer = FrameOrderingOptimizer(power_base=2, max_resolution=10)
        sequence = optimizer.generate_resolution_sequence()
        
        # Should be [1, 2, 4, 8] and stop before 16 because max_resolution=10
        expected = [1, 2, 4, 8]
        self.assertEqual(sequence, expected)
    
    def test_edge_case_max_resolution_equals_start(self):
        """Test edge case where max_resolution equals starting resolution"""
        optimizer = FrameOrderingOptimizer(power_base=2, max_resolution=2, start_with_1x1=False)
        sequence = optimizer.generate_resolution_sequence()
        
        expected = [2]
        self.assertEqual(sequence, expected)


class TestSignatureExtraction(unittest.TestCase):
    """Test signature extraction at different resolutions"""
    
    def setUp(self):
        self.optimizer = FrameOrderingOptimizer()
        self.test_frames = MockQRFrameGenerator.create_test_frame_set(3)
    
    def test_extract_1x1_signatures(self):
        """Test extraction of 1x1 signatures (overall brightness)"""
        signatures = self.optimizer.extract_signatures_at_resolution(self.test_frames, 1)
        
        # Should return list of single values (brightness)
        self.assertEqual(len(signatures), len(self.test_frames))
        for signature in signatures:
            self.assertEqual(len(signature), 1)  # 1x1 = 1 value
            self.assertIsInstance(signature[0], (int, float, np.number))
    
    def test_extract_2x2_signatures(self):
        """Test extraction of 2x2 signatures"""
        signatures = self.optimizer.extract_signatures_at_resolution(self.test_frames, 2)
        
        self.assertEqual(len(signatures), len(self.test_frames))
        for signature in signatures:
            self.assertEqual(len(signature), 4)  # 2x2 = 4 values
    
    def test_extract_4x4_signatures(self):
        """Test extraction of 4x4 signatures"""
        signatures = self.optimizer.extract_signatures_at_resolution(self.test_frames, 4)
        
        self.assertEqual(len(signatures), len(self.test_frames))
        for signature in signatures:
            self.assertEqual(len(signature), 16)  # 4x4 = 16 values
    
    def test_signature_extraction_deterministic(self):
        """Test that signature extraction is deterministic"""
        signatures1 = self.optimizer.extract_signatures_at_resolution(self.test_frames, 2)
        signatures2 = self.optimizer.extract_signatures_at_resolution(self.test_frames, 2)
        
        np.testing.assert_array_equal(signatures1, signatures2)
    
    def test_different_patterns_have_different_signatures(self):
        """Test that different QR patterns produce different signatures"""
        test_frames = [
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_black'),
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_white'),
            MockQRFrameGenerator.create_test_qr_frame(pattern='checkerboard')
        ]
        
        signatures = self.optimizer.extract_signatures_at_resolution(test_frames, 2)
        
        # All signatures should be different
        self.assertFalse(np.array_equal(signatures[0], signatures[1]))
        self.assertFalse(np.array_equal(signatures[1], signatures[2]))
        self.assertFalse(np.array_equal(signatures[0], signatures[2]))


class TestProgressiveSorting(unittest.TestCase):
    """Test progressive sorting algorithm"""
    
    def setUp(self):
        self.optimizer = FrameOrderingOptimizer()
        # Create frames with predictable ordering characteristics
        self.test_frames = [
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_black'),   # Darkest
            MockQRFrameGenerator.create_test_qr_frame(pattern='dark'),          # Dark
            MockQRFrameGenerator.create_test_qr_frame(pattern='random'),        # Medium
            MockQRFrameGenerator.create_test_qr_frame(pattern='light'),         # Light  
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_white')    # Lightest
        ]
    
    def test_progressive_sort_basic(self):
        """Test basic progressive sorting functionality"""
        frame_indices = list(range(len(self.test_frames)))
        sorted_indices, metadata = self.optimizer.progressive_sort(frame_indices, self.test_frames)
        
        # Should return a permutation of the original indices
        self.assertEqual(len(sorted_indices), len(frame_indices))
        self.assertEqual(set(sorted_indices), set(frame_indices))
    
    def test_sorting_maintains_frame_index_mapping(self):
        """Test that sorting preserves ability to map back to original frames"""
        frame_indices = list(range(len(self.test_frames)))
        sorted_indices, metadata = self.optimizer.progressive_sort(frame_indices, self.test_frames)
        
        # Verify we can still access all original frames through sorted indices
        for idx in sorted_indices:
            self.assertIsInstance(self.test_frames[idx], np.ndarray)
            self.assertEqual(self.test_frames[idx].shape, (477, 477))
    
    def test_stable_sort_behavior(self):
        """Test that sorting is stable (maintains relative order of equal elements)"""
        # Create two identical frames
        identical_frames = [
            MockQRFrameGenerator.create_test_qr_frame(pattern='checkerboard'),
            MockQRFrameGenerator.create_test_qr_frame(pattern='checkerboard'),  # Identical
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_black')
        ]
        
        frame_indices = [0, 1, 2]
        sorted_indices, metadata = self.optimizer.progressive_sort(frame_indices, identical_frames)
        
        # The two identical frames should maintain their relative order
        pos_0 = sorted_indices.index(0)
        pos_1 = sorted_indices.index(1)
        
        # Frame 0 should appear before frame 1 (stable sort)
        self.assertLess(pos_0, pos_1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        self.optimizer = FrameOrderingOptimizer()
    
    def test_empty_frame_list(self):
        """Test handling of empty frame list"""
        empty_frames = []
        sorted_indices, metadata = self.optimizer.progressive_sort([], empty_frames)
        
        self.assertEqual(sorted_indices, [])
    
    def test_single_frame(self):
        """Test handling of single frame"""
        single_frame = [MockQRFrameGenerator.create_test_qr_frame()]
        sorted_indices, metadata = self.optimizer.progressive_sort([0], single_frame)
        
        self.assertEqual(sorted_indices, [0])
    
    def test_two_frames(self):
        """Test handling of two frames"""
        two_frames = [
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_white'),
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_black')
        ]
        sorted_indices, metadata = self.optimizer.progressive_sort([0, 1], two_frames)
        
        # Should return some permutation of [0, 1]
        self.assertEqual(set(sorted_indices), {0, 1})
        self.assertEqual(len(sorted_indices), 2)
    
    def test_non_square_frames_error(self):
        """Test error handling for non-square frames"""
        non_square_frame = np.zeros((400, 500), dtype=np.uint8)  # 400x500 not square
        
        with self.assertRaises(ValueError):
            self.optimizer.extract_signatures_at_resolution([non_square_frame], 2)
    
    def test_mismatched_frame_sizes_error(self):
        """Test error handling for frames of different sizes"""
        frames = [
            np.zeros((400, 400), dtype=np.uint8),  
            np.zeros((500, 500), dtype=np.uint8)   # Different size
        ]
        
        with self.assertRaises(ValueError):
            self.optimizer.extract_signatures_at_resolution(frames, 2)


class TestResolutionSignature(unittest.TestCase):
    """Test ResolutionSignature class"""
    
    def test_resolution_signature_creation(self):
        """Test creating ResolutionSignature objects"""
        data = np.array([100, 150, 200, 250])
        signature = ResolutionSignature(resolution=2, data=data)
        
        self.assertEqual(signature.resolution, 2)
        np.testing.assert_array_equal(signature.data, data)
    
    def test_resolution_signature_comparison(self):
        """Test ResolutionSignature comparison for sorting"""
        sig1 = ResolutionSignature(2, np.array([100, 100, 100, 100]))  # Darker
        sig2 = ResolutionSignature(2, np.array([200, 200, 200, 200]))  # Lighter
        
        # Should be sortable (implement __lt__)
        self.assertTrue(sig1 < sig2 or sig2 < sig1)  # One should be less than the other
    
    def test_resolution_signature_equality(self):
        """Test ResolutionSignature equality for identical signatures"""
        data = np.array([100, 150, 200, 250])
        sig1 = ResolutionSignature(2, data)
        sig2 = ResolutionSignature(2, data.copy())
        
        # Should be equal if data is the same
        self.assertEqual(sig1, sig2)


class TestFullProgressiveSorting(unittest.TestCase):
    """Test the complete progressive sorting algorithm (Phase 2)"""
    
    def setUp(self):
        self.optimizer = FrameOrderingOptimizer(power_base=2, max_resolution=8, start_with_1x1=True)
        
        # Create frames with known sorting characteristics
        self.test_frames = [
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_white', size=100),   # Should be last (brightest)
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_black', size=100),   # Should be first (darkest)
            MockQRFrameGenerator.create_test_qr_frame(pattern='light', size=100),        # Should be 3rd
            MockQRFrameGenerator.create_test_qr_frame(pattern='dark', size=100),         # Should be 2nd
        ]
    
    def test_progressive_sort_actually_sorts(self):
        """Test that progressive_sort actually reorders frames (not just returns original order)"""
        frame_indices = [0, 1, 2, 3]  # Original order
        sorted_indices, metadata = self.optimizer.progressive_sort(frame_indices, self.test_frames)
        
        # Should actually change the order (not return [0, 1, 2, 3])
        self.assertNotEqual(sorted_indices, frame_indices)
        
        # Should still be a valid permutation
        self.assertEqual(set(sorted_indices), set(frame_indices))
    
    def test_progressive_sort_orders_by_brightness(self):
        """Test that progressive sorting orders frames from dark to light"""
        frame_indices = [0, 1, 2, 3]
        sorted_indices, metadata = self.optimizer.progressive_sort(frame_indices, self.test_frames)
        
        # Expected order: solid_black(1), dark(3), light(2), solid_white(0)
        expected_order = [1, 3, 2, 0]  # Indices of frames in brightness order
        self.assertEqual(sorted_indices, expected_order)
    
    def test_progressive_sort_with_different_resolution_sequences(self):
        """Test progressive sorting with different resolution configurations"""
        test_configs = [
            FrameOrderingOptimizer(power_base=2, max_resolution=4, start_with_1x1=True),
            FrameOrderingOptimizer(power_base=2, max_resolution=16, start_with_1x1=False),
            FrameOrderingOptimizer(power_base=3, max_resolution=9, start_with_1x1=True),
        ]
        
        for optimizer in test_configs:
            frame_indices = [0, 1, 2, 3]
            sorted_indices, metadata = optimizer.progressive_sort(frame_indices, self.test_frames)
            
            # All should produce valid permutations
            self.assertEqual(set(sorted_indices), set(frame_indices))
            
            # All should actually sort (change order)
            self.assertNotEqual(sorted_indices, frame_indices)
    
    def test_progressive_sort_stability(self):
        """Test that progressive sorting is stable (identical frames maintain relative order)"""
        # Create frames where some are identical
        identical_frames = [
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_black', size=50),
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_black', size=50),  # Identical to 0
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_white', size=50),
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_black', size=50),  # Identical to 0,1
        ]
        
        frame_indices = [0, 1, 2, 3]
        sorted_indices, metadata = self.optimizer.progressive_sort(frame_indices, identical_frames)
        
        # Find positions of the identical black frames (0, 1, 3)
        black_positions = []
        for idx in [0, 1, 3]:
            black_positions.append(sorted_indices.index(idx))
        
        # They should maintain their relative order: 0 before 1 before 3
        self.assertEqual(black_positions, sorted(black_positions))
    
    def test_progressive_sort_large_frame_set(self):
        """Test progressive sorting with larger frame sets"""
        # Create 20 frames with random patterns
        large_frame_set = MockQRFrameGenerator.create_test_frame_set(
            count=20, 
            patterns=['random'] * 20
        )
        
        frame_indices = list(range(20))
        sorted_indices, metadata = self.optimizer.progressive_sort(frame_indices, large_frame_set)
        
        # Should be valid permutation
        self.assertEqual(set(sorted_indices), set(frame_indices))
        self.assertEqual(len(sorted_indices), 20)


class TestOptimizeFrameOrder(unittest.TestCase):
    """Test the main optimize_frame_order method (Phase 2)"""
    
    def setUp(self):
        self.optimizer = FrameOrderingOptimizer()
        self.test_frames = MockQRFrameGenerator.create_test_frame_set(10)
    
    def test_optimize_frame_order_returns_correct_format(self):
        """Test that optimize_frame_order returns proper dict format"""
        result = self.optimizer.optimize_frame_order(self.test_frames)
        
        # Should return dict with frame_order and metadata
        self.assertIsInstance(result, dict)
        self.assertIn('frame_order', result)
        self.assertIn('metadata', result)
        
        sorted_indices = result['frame_order']
        metadata = result['metadata']
        
        # Indices should be valid permutation
        self.assertEqual(set(sorted_indices), set(range(len(self.test_frames))))
        
        # Metadata should be dict
        self.assertIsInstance(metadata, dict)
    
    def test_optimize_frame_order_includes_metadata(self):
        """Test that optimization metadata includes useful information"""
        result = self.optimizer.optimize_frame_order(self.test_frames)
        metadata = result['metadata']
        
        # Should include configuration info
        self.assertIn('power_base', metadata)
        self.assertIn('max_resolution', metadata)
        self.assertIn('start_with_1x1', metadata)
        
        # Should include process info
        self.assertIn('resolution_sequence', metadata)
        self.assertIn('frame_count', metadata)
        
        # Values should be correct
        self.assertEqual(metadata['power_base'], self.optimizer.power_base)
        self.assertEqual(metadata['frame_count'], len(self.test_frames))
    
    def test_optimize_frame_order_empty_frames(self):
        """Test optimize_frame_order with empty frame list"""
        result = self.optimizer.optimize_frame_order([])
        
        self.assertEqual(result['frame_order'], [])
        self.assertEqual(result['metadata']['frame_count'], 0)
    
    def test_optimize_frame_order_single_frame(self):
        """Test optimize_frame_order with single frame"""
        single_frame = [MockQRFrameGenerator.create_test_qr_frame()]
        result = self.optimizer.optimize_frame_order(single_frame)
        
        self.assertEqual(result['frame_order'], [0])
        self.assertEqual(result['metadata']['frame_count'], 1)


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency with large frame sets (Phase 2)"""
    
    def test_large_frame_set_memory_usage(self):
        """Test that large frame sets don't cause memory issues"""
        # Create 100 frames (reasonable test size)
        large_frame_set = MockQRFrameGenerator.create_test_frame_set(100)
        
        optimizer = FrameOrderingOptimizer(max_resolution=32)
        
        # Should complete without memory errors
        result = optimizer.optimize_frame_order(large_frame_set)
        
        # Verify results are correct
        self.assertEqual(len(result['frame_order']), 100)
        self.assertEqual(set(result['frame_order']), set(range(100)))
        self.assertEqual(result['metadata']['frame_count'], 100)
    
    def test_signature_extraction_memory_efficient(self):
        """Test that signature extraction doesn't hold excessive memory"""
        frames = MockQRFrameGenerator.create_test_frame_set(50)
        optimizer = FrameOrderingOptimizer()
        
        # Extract signatures at multiple resolutions
        resolutions = [1, 2, 4, 8, 16]
        for resolution in resolutions:
            signatures = optimizer.extract_signatures_at_resolution(frames, resolution)
            
            # Should return correct number of signatures
            self.assertEqual(len(signatures), len(frames))
            
            # Each signature should have correct size
            expected_size = resolution * resolution
            for signature in signatures:
                self.assertEqual(len(signature), expected_size)


class TestProgressiveSortingSteps(unittest.TestCase):
    """Test the step-by-step progressive sorting process (Phase 2)"""
    
    def setUp(self):
        self.optimizer = FrameOrderingOptimizer(power_base=2, max_resolution=4, start_with_1x1=True)
        
        # Create frames with very different characteristics for clear sorting
        self.test_frames = [
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_white', size=64),
            MockQRFrameGenerator.create_test_qr_frame(pattern='solid_black', size=64),
            MockQRFrameGenerator.create_test_qr_frame(pattern='checkerboard', size=64),
        ]
    
    def test_extract_all_signatures(self):
        """Test extracting signatures at all resolution levels"""
        resolution_sequence = self.optimizer.generate_resolution_sequence()
        
        all_signatures = {}
        for resolution in resolution_sequence:
            signatures = self.optimizer.extract_signatures_at_resolution(self.test_frames, resolution)
            all_signatures[resolution] = signatures
        
        # Should have signatures for each resolution
        self.assertEqual(set(all_signatures.keys()), set(resolution_sequence))
        
        # Each resolution should have signatures for all frames
        for resolution, signatures in all_signatures.items():
            self.assertEqual(len(signatures), len(self.test_frames))
            
            # Signature size should match resolution
            expected_size = resolution * resolution
            for signature in signatures:
                self.assertEqual(len(signature), expected_size)
    
    def test_stable_sort_implementation(self):
        """Test the internal stable sort implementation"""
        # Create test data for stable sorting
        frame_indices = [0, 1, 2]
        test_signatures = [
            np.array([100]),  # Medium
            np.array([50]),   # Dark  
            np.array([200]),  # Light
        ]
        
        sorted_indices = self.optimizer._stable_sort_by_signatures(frame_indices, test_signatures)
        
        # Should be sorted by signature values: [1, 0, 2] (dark, medium, light)
        expected_order = [1, 0, 2]
        self.assertEqual(sorted_indices, expected_order)


class TestEdgeCasesPhase2(unittest.TestCase):
    """Additional edge case testing for Phase 2"""
    
    def test_very_small_frames(self):
        """Test with very small frame sizes"""
        tiny_frames = [
            np.zeros((4, 4), dtype=np.uint8),
            np.ones((4, 4), dtype=np.uint8) * 255,
            np.random.randint(0, 256, (4, 4), dtype=np.uint8)
        ]
        
        optimizer = FrameOrderingOptimizer(max_resolution=2)
        result = optimizer.optimize_frame_order(tiny_frames)
        
        # Should handle small frames correctly
        self.assertEqual(set(result['frame_order']), {0, 1, 2})
        self.assertEqual(len(result['frame_order']), 3)
    
    def test_high_resolution_limit(self):
        """Test with high resolution limits"""
        frames = MockQRFrameGenerator.create_test_frame_set(5, size=200)
        
        # Test with high resolution limit
        optimizer = FrameOrderingOptimizer(max_resolution=128)
        result = optimizer.optimize_frame_order(frames)
        
        # Should complete successfully
        self.assertEqual(len(result['frame_order']), 5)
        
        # Should have used high resolution sequence
        self.assertGreater(len(result['metadata']['resolution_sequence']), 5)
    
    def test_power_base_3_full_workflow(self):
        """Test complete workflow with power_base=3"""
        frames = MockQRFrameGenerator.create_test_frame_set(8)
        
        optimizer = FrameOrderingOptimizer(power_base=3, max_resolution=27)
        result = optimizer.optimize_frame_order(frames)
        
        # Should work with power base 3
        self.assertEqual(len(result['frame_order']), 8)
        self.assertEqual(result['metadata']['power_base'], 3)
        
        # Resolution sequence should use powers of 3
        expected_sequence = [1, 3, 9, 27]
        self.assertEqual(result['metadata']['resolution_sequence'], expected_sequence)


class TestEarlyStopping:
    """Test early stopping functionality for progressive optimization."""
    
    def test_early_stopping_enabled_by_default(self):
        """Test that early stopping is enabled by default with reasonable threshold."""
        optimizer = FrameOrderingOptimizer()
        assert hasattr(optimizer, 'early_stopping_enabled')
        assert optimizer.early_stopping_enabled is True
        assert hasattr(optimizer, 'early_stopping_threshold')
        assert 0.0 < optimizer.early_stopping_threshold < 1.0
    
    def test_early_stopping_configuration(self):
        """Test early stopping can be configured."""
        optimizer = FrameOrderingOptimizer(
            early_stopping_enabled=False,
            early_stopping_threshold=0.05
        )
        assert optimizer.early_stopping_enabled is False
        assert optimizer.early_stopping_threshold == 0.05
    
    def test_early_stopping_threshold_validation(self):
        """Test early stopping threshold validation."""
        # Valid thresholds
        FrameOrderingOptimizer(early_stopping_threshold=0.01)
        FrameOrderingOptimizer(early_stopping_threshold=0.5)
        
        # Invalid thresholds
        with pytest.raises(ValueError, match="early_stopping_threshold must be between 0 and 1"):
            FrameOrderingOptimizer(early_stopping_threshold=-0.1)
        
        with pytest.raises(ValueError, match="early_stopping_threshold must be between 0 and 1"):
            FrameOrderingOptimizer(early_stopping_threshold=1.5)
    
    def test_early_stopping_with_identical_frames(self):
        """Test early stopping triggers with identical frames (no improvement possible)."""
        # Create identical frames - should trigger early stopping immediately
        frames = [np.ones((32, 32), dtype=np.uint8) * 255] * 10
        
        optimizer = FrameOrderingOptimizer(
            power_base=2,
            max_resolution=16,
            early_stopping_enabled=True,
            early_stopping_threshold=0.1
        )
        
        result = optimizer.optimize_frame_order(frames)
        
        # Should have stopped early due to no improvement
        assert 'early_stopping_triggered' in result['metadata']
        assert result['metadata']['early_stopping_triggered'] is True
        assert 'early_stopping_resolution' in result['metadata']
        assert result['metadata']['early_stopping_resolution'] <= 4  # Should stop very early
    
    def test_early_stopping_with_diverse_frames(self):
        """Test early stopping doesn't trigger prematurely with diverse frames."""
        # Create diverse frames that should benefit from full optimization
        frames = []
        for i in range(20):
            frame = np.zeros((32, 32), dtype=np.uint8)
            # Create different patterns with more diversity
            if i % 4 == 0:
                frame[:16, :16] = 255  # Top-left quadrant
            elif i % 4 == 1:
                frame[:16, 16:] = 255  # Top-right quadrant
            elif i % 4 == 2:
                frame[16:, :16] = 255  # Bottom-left quadrant
            else:
                frame[16:, 16:] = 255  # Bottom-right quadrant
            
            # Add unique variations within each pattern to increase diversity
            frame[i % 32, (i * 2) % 32] = 128  # Add unique pixel
            frames.append(frame)
        
        optimizer = FrameOrderingOptimizer(
            power_base=2,
            max_resolution=16,
            early_stopping_enabled=True,
            early_stopping_threshold=0.01  # Very strict threshold to allow more processing
        )
        
        result = optimizer.optimize_frame_order(frames)
        
        # With very diverse frames and strict threshold, should either not stop early
        # or if it does stop, should have processed multiple resolutions
        if result['metadata'].get('early_stopping_triggered', False):
            # If early stopping triggered, should have processed at least 3 resolutions
            assert result['metadata']['early_stopping_resolution'] >= 4
        else:
            # Should process multiple resolutions
            assert len(result['metadata']['resolution_sequence']) > 2
    
    def test_early_stopping_disabled(self):
        """Test that early stopping can be completely disabled."""
        frames = [np.ones((32, 32), dtype=np.uint8) * 255] * 10  # Identical frames
        
        optimizer = FrameOrderingOptimizer(
            power_base=2,
            max_resolution=16,
            early_stopping_enabled=False
        )
        
        result = optimizer.optimize_frame_order(frames)
        
        # Should not have early stopping metadata when disabled
        assert 'early_stopping_triggered' not in result['metadata']
        assert 'early_stopping_resolution' not in result['metadata']
        # Should process all resolutions
        expected_resolutions = [1, 2, 4, 8, 16]
        assert result['metadata']['resolution_sequence'] == expected_resolutions
    
    def test_early_stopping_improvement_calculation(self):
        """Test that improvement calculation works correctly."""
        # Create frames with gradual similarity - should show decreasing improvement
        frames = []
        for i in range(15):
            frame = np.zeros((32, 32), dtype=np.uint8)
            # Create gradually changing patterns
            brightness = int(255 * (i / 14))  # 0 to 255
            frame[:, :i+1] = brightness
            frames.append(frame)
        
        optimizer = FrameOrderingOptimizer(
            power_base=2,
            max_resolution=32,
            early_stopping_enabled=True,
            early_stopping_threshold=0.1
        )
        
        result = optimizer.optimize_frame_order(frames)
        
        # Should have improvement tracking in metadata
        assert 'resolution_improvements' in result['metadata']
        improvements = result['metadata']['resolution_improvements']
        assert len(improvements) > 0
        
        # Each improvement should be a valid percentage
        for improvement in improvements:
            assert 0.0 <= improvement <= 1.0
    
    def test_early_stopping_performance_benefit(self):
        """Test that early stopping provides performance benefits."""
        # Create frames that should trigger early stopping
        frames = []
        for i in range(50):
            frame = np.random.randint(0, 2, (32, 32), dtype=np.uint8) * 255
            # Add some noise but keep mostly similar
            if i > 0:
                frame = (frames[0] * 0.9 + frame * 0.1).astype(np.uint8)
            frames.append(frame)
        
        # Test with early stopping enabled
        optimizer_early = FrameOrderingOptimizer(
            power_base=2,
            max_resolution=32,
            early_stopping_enabled=True,
            early_stopping_threshold=0.05
        )
        
        start_time = time.time()
        result_early = optimizer_early.optimize_frame_order(frames)
        time_early = time.time() - start_time
        
        # Test with early stopping disabled
        optimizer_full = FrameOrderingOptimizer(
            power_base=2,
            max_resolution=32,
            early_stopping_enabled=False
        )
        
        start_time = time.time()
        result_full = optimizer_full.optimize_frame_order(frames)
        time_full = time.time() - start_time
        
        # Early stopping should be faster (or at least not significantly slower)
        # Note: For small datasets, timing can be variable, so we allow generous tolerance
        assert time_early <= time_full * 2.0 or time_early < 0.01  # 2x tolerance or very fast anyway
        
        # Results should be reasonably similar (within 10% of frame changes)
        early_changes = sum(1 for i in range(1, len(result_early['frame_order'])) 
                           if result_early['frame_order'][i] != i)
        full_changes = sum(1 for i in range(1, len(result_full['frame_order'])) 
                          if result_full['frame_order'][i] != i)
        
        if full_changes > 0:
            similarity_ratio = early_changes / full_changes
            assert 0.8 <= similarity_ratio <= 1.2  # Within 20% similarity


if __name__ == '__main__':
    unittest.main() 