#!/usr/bin/env python3
"""
Phase 4.1: Performance Analysis & Benchmarking Tests

Comprehensive benchmarking suite to measure:
- Real video compression benefits with/without frame ordering
- Performance scaling with different dataset sizes
- Memory usage profiling
- Video codec compatibility
- Compression ratio analysis

This follows the TDD approach for Phase 4 development.
"""

import unittest
import tempfile
import os
import time
import psutil
from pathlib import Path
import json
import cv2
import numpy as np
import sys
from unittest.mock import patch
import subprocess

# Add memvid to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.encoder import MemvidEncoder
from memvid.retriever import MemvidRetriever


class TestCompressionBenchmarks(unittest.TestCase):
    """Test real compression benefits of frame ordering (Phase 4.1.1)"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test datasets of different characteristics
        self.diverse_chunks = [
            "Bright content with lots of whitespace and minimal density",
            "██████████████████████████████████████████████████████████",  # Very dark
            "Mixed content with 50% density ██████     ██████     ██████",
            "Another bright chunk with sparse patterns and light content",
            "████████████████████████████████████████████████████████████",  # Very dark
            "Medium density content with moderate patterns and structure",
            "Sparse whitespace content with minimal black pixels present",
            "█████████████████████ DENSE █████████████████████████████",
        ]
        
        # Similar chunks (should compress well with ordering)
        self.similar_chunks = [
            "Similar content pattern A with consistent structure",
            "Similar content pattern B with consistent structure", 
            "Similar content pattern C with consistent structure",
            "Similar content pattern D with consistent structure",
        ]
        
        # Random chunks (ordering should have minimal benefit)
        self.random_chunks = [
            f"Random content {i} with unpredictable patterns {hash(str(i)) % 1000}"
            for i in range(8)
        ]
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_compression_benefit_measurement(self):
        """Test that we can measure compression benefits accurately"""
        encoder1 = MemvidEncoder()
        encoder2 = MemvidEncoder()
        encoder1.add_chunks(self.diverse_chunks)
        encoder2.add_chunks(self.diverse_chunks)
        
        video_no_ordering = Path(self.temp_dir) / "no_ordering.mp4"
        index_no_ordering = Path(self.temp_dir) / "no_ordering.json"
        video_with_ordering = Path(self.temp_dir) / "with_ordering.mp4"
        index_with_ordering = Path(self.temp_dir) / "with_ordering.json"
        
        # Build without frame ordering
        result1 = encoder1.build_video(
            str(video_no_ordering),
            str(index_no_ordering),
            enable_frame_ordering=False,
            show_progress=False
        )
        
        # Build with frame ordering
        result2 = encoder2.build_video(
            str(video_with_ordering),
            str(index_with_ordering),
            enable_frame_ordering=True,
            show_progress=False
        )
        
        # Measure file sizes
        size_no_ordering = video_no_ordering.stat().st_size
        size_with_ordering = video_with_ordering.stat().st_size
        compression_ratio = size_no_ordering / size_with_ordering
        
        # Should be able to measure compression
        self.assertGreater(size_no_ordering, 0)
        self.assertGreater(size_with_ordering, 0)
        self.assertIsInstance(compression_ratio, float)
        
        # Results should include file size metadata
        self.assertIn("video_file", result1)
        self.assertIn("video_file", result2)
        
        # Store results for analysis
        benchmark_data = {
            "size_no_ordering": size_no_ordering,
            "size_with_ordering": size_with_ordering,
            "compression_ratio": compression_ratio,
            "frame_count": len(self.diverse_chunks),
            "frame_ordering_metadata": result2.get("frame_ordering", {})
        }
        
        # Should be able to save benchmark data
        benchmark_file = Path(self.temp_dir) / "benchmark.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        self.assertTrue(benchmark_file.exists())
    
    def test_compression_benefits_different_content_types(self):
        """Test compression benefits vary by content type"""
        content_types = {
            "diverse": self.diverse_chunks,
            "similar": self.similar_chunks,
            "random": self.random_chunks
        }
        
        compression_ratios = {}
        
        for content_type, chunks in content_types.items():
            encoder1 = MemvidEncoder()
            encoder2 = MemvidEncoder()
            encoder1.add_chunks(chunks)
            encoder2.add_chunks(chunks)
            
            video_no = Path(self.temp_dir) / f"{content_type}_no.mp4"
            index_no = Path(self.temp_dir) / f"{content_type}_no.json"
            video_yes = Path(self.temp_dir) / f"{content_type}_yes.mp4"
            index_yes = Path(self.temp_dir) / f"{content_type}_yes.json"
            
            # Build both versions
            encoder1.build_video(str(video_no), str(index_no), 
                               enable_frame_ordering=False, show_progress=False)
            encoder2.build_video(str(video_yes), str(index_yes), 
                               enable_frame_ordering=True, show_progress=False)
            
            # Calculate compression ratio
            size_no = video_no.stat().st_size
            size_yes = video_yes.stat().st_size
            compression_ratios[content_type] = size_no / size_yes
        
        # All content types should have measurable compression ratios
        for content_type, ratio in compression_ratios.items():
            self.assertGreater(ratio, 0.5)  # At least some compression
            self.assertLess(ratio, 2.0)     # Not unrealistic
        
        # Should be able to compare different content types
        self.assertEqual(len(compression_ratios), 3)


class TestPerformanceScaling(unittest.TestCase):
    """Test performance scaling with different dataset sizes (Phase 4.1.2)"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_scaling_with_small_datasets(self):
        """Test frame ordering performance with small datasets (10-100 frames)"""
        dataset_sizes = [10, 25, 50, 100]
        performance_data = {}
        
        for size in dataset_sizes:
            chunks = [f"Test chunk {i} with content" for i in range(size)]
            encoder = MemvidEncoder()
            encoder.add_chunks(chunks)
            
            video_path = Path(self.temp_dir) / f"test_{size}.mp4"
            index_path = Path(self.temp_dir) / f"test_{size}.json"
            
            start_time = time.time()
            result = encoder.build_video(
                str(video_path),
                str(index_path),
                enable_frame_ordering=True,
                show_progress=False
            )
            total_time = time.time() - start_time
            
            if "frame_ordering" in result:
                optimization_time = result["frame_ordering"]["optimization_time"]
                performance_data[size] = {
                    "total_time": total_time,
                    "optimization_time": optimization_time,
                    "optimization_percentage": optimization_time / total_time * 100
                }
        
        # Should complete all dataset sizes
        self.assertEqual(len(performance_data), len(dataset_sizes))
        
        # Performance should scale reasonably
        for size, data in performance_data.items():
            self.assertLess(data["optimization_percentage"], 20)  # <20% of total time
            self.assertGreater(data["optimization_time"], 0)     # Actually did work
    
    def test_performance_measurement_accuracy(self):
        """Test that performance measurements are accurate and consistent"""
        chunks = [f"Performance test chunk {i}" for i in range(20)]
        
        # Run multiple times to check consistency
        times = []
        for run in range(3):
            encoder = MemvidEncoder()
            encoder.add_chunks(chunks)
            
            video_path = Path(self.temp_dir) / f"perf_test_{run}.mp4"
            index_path = Path(self.temp_dir) / f"perf_test_{run}.json"
            
            start_time = time.time()
            result = encoder.build_video(
                str(video_path),
                str(index_path),
                enable_frame_ordering=True,
                show_progress=False
            )
            total_time = time.time() - start_time
            
            if "frame_ordering" in result:
                times.append(result["frame_ordering"]["optimization_time"])
        
        # Should have consistent measurements
        self.assertEqual(len(times), 3)
        
        # Times should be reasonably consistent (within 50% variance)
        avg_time = sum(times) / len(times)
        for time_val in times:
            variance = abs(time_val - avg_time) / avg_time
            self.assertLess(variance, 0.5)  # <50% variance


class TestMemoryProfiling(unittest.TestCase):
    """Test memory usage profiling and optimization (Phase 4.1.3)"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_usage_measurement(self):
        """Test that we can measure memory usage during frame ordering"""
        chunks = [f"Memory test chunk {i}" * 10 for i in range(50)]
        encoder = MemvidEncoder()
        encoder.add_chunks(chunks)
        
        video_path = Path(self.temp_dir) / "memory_test.mp4"
        index_path = Path(self.temp_dir) / "memory_test.json"
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = encoder.build_video(
            str(video_path),
            str(index_path),
            enable_frame_ordering=True,
            show_progress=False
        )
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Should be able to measure memory usage
        self.assertIsInstance(memory_used, float)
        self.assertGreater(memory_after, 0)
        
        # Memory usage should be reasonable (less than 500MB for test)
        self.assertLess(memory_used, 500)
    
    def test_memory_efficiency_with_large_frames(self):
        """Test memory efficiency doesn't grow exponentially with frame count"""
        dataset_sizes = [10, 20, 40]  # Doubling sizes
        memory_usage = {}
        
        for size in dataset_sizes:
            chunks = [f"Large test chunk {i}" * 20 for i in range(size)]
            encoder = MemvidEncoder()
            encoder.add_chunks(chunks)
            
            video_path = Path(self.temp_dir) / f"mem_test_{size}.mp4"
            index_path = Path(self.temp_dir) / f"mem_test_{size}.json"
            
            # Measure peak memory during encoding
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024
            
            result = encoder.build_video(
                str(video_path),
                str(index_path),
                enable_frame_ordering=True,
                show_progress=False
            )
            
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_usage[size] = memory_after - memory_before
        
        # Memory usage should scale sub-quadratically
        # (quadratic would be bad for large datasets)
        if len(memory_usage) >= 2:
            sizes = sorted(memory_usage.keys())
            for i in range(1, len(sizes)):
                size_ratio = sizes[i] / sizes[i-1]
                memory_ratio = memory_usage[sizes[i]] / max(memory_usage[sizes[i-1]], 1)
                
                # Memory growth should be less than quadratic
                self.assertLess(memory_ratio, size_ratio ** 1.5)


class TestCodecCompatibility(unittest.TestCase):
    """Test video codec compatibility and compression (Phase 4.1.4)"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_chunks = [f"Codec test chunk {i}" for i in range(10)]
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_mp4v_codec_compatibility(self):
        """Test frame ordering works with MP4V codec"""
        encoder = MemvidEncoder()
        encoder.add_chunks(self.test_chunks)
        
        video_path = Path(self.temp_dir) / "mp4v_test.mp4"
        index_path = Path(self.temp_dir) / "mp4v_test.json"
        
        result = encoder.build_video(
            str(video_path),
            str(index_path),
            codec="mp4v",
            enable_frame_ordering=True,
            show_progress=False
        )
        
        # Should complete successfully
        self.assertTrue(video_path.exists())
        self.assertTrue(index_path.exists())
        self.assertIn("frame_ordering", result)
        
        # Should be able to retrieve from the video
        retriever = MemvidRetriever(str(video_path), str(index_path))
        search_results = retriever.search("test chunk", top_k=3)
        self.assertGreater(len(search_results), 0)
    
    def test_h265_codec_fallback_handling(self):
        """Test graceful handling when H.265 codec is not available"""
        encoder = MemvidEncoder()
        encoder.add_chunks(self.test_chunks)
        
        video_path = Path(self.temp_dir) / "h265_test.mp4"
        index_path = Path(self.temp_dir) / "h265_test.json"
        
        # This may fall back to MP4V if H.265 not available
        result = encoder.build_video(
            str(video_path),
            str(index_path),
            codec="h265",
            enable_frame_ordering=True,
            allow_fallback=True,
            show_progress=False
        )
        
        # Should complete successfully (possibly with fallback)
        self.assertTrue(video_path.exists())
        self.assertTrue(index_path.exists())
        
        # Frame ordering should still work regardless of codec
        if "frame_ordering" in result:
            self.assertIn("optimized_order", result["frame_ordering"])
    
    def test_codec_compression_comparison(self):
        """Test compression differences between codecs with frame ordering"""
        encoder1 = MemvidEncoder()
        encoder2 = MemvidEncoder()
        encoder1.add_chunks(self.test_chunks)
        encoder2.add_chunks(self.test_chunks)
        
        mp4v_video = Path(self.temp_dir) / "mp4v.mp4"
        mp4v_index = Path(self.temp_dir) / "mp4v.json"
        h265_video = Path(self.temp_dir) / "h265.mp4"
        h265_index = Path(self.temp_dir) / "h265.json"
        
        # Build with both codecs
        result_mp4v = encoder1.build_video(
            str(mp4v_video), str(mp4v_index),
            codec="mp4v", enable_frame_ordering=True, show_progress=False
        )
        
        result_h265 = encoder2.build_video(
            str(h265_video), str(h265_index),
            codec="h265", enable_frame_ordering=True, 
            allow_fallback=True, show_progress=False
        )
        
        # Both should complete
        self.assertTrue(mp4v_video.exists())
        
        # If H.265 succeeded (didn't fall back), compare sizes
        if h265_video.exists():
            mp4v_size = mp4v_video.stat().st_size
            h265_size = h265_video.stat().st_size
            
            # Both should be reasonable sizes
            self.assertGreater(mp4v_size, 1000)  # At least 1KB
            if h265_size > 1000:  # If H.265 actually worked
                # H.265 might be smaller (better compression)
                compression_improvement = mp4v_size / h265_size
                self.assertGreater(compression_improvement, 0.5)  # Reasonable range


class TestCompressionRatioAnalysis(unittest.TestCase):
    """Test detailed compression ratio measurement and reporting (Phase 4.1.5)"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_compression_ratio_calculation(self):
        """Test accurate compression ratio calculation"""
        # Create content with known compression characteristics
        highly_compressible = ["Same content repeated"] * 10
        poorly_compressible = [f"Random {hash(i)} content {i}" for i in range(10)]
        
        test_cases = {
            "highly_compressible": highly_compressible,
            "poorly_compressible": poorly_compressible
        }
        
        compression_results = {}
        
        for case_name, chunks in test_cases.items():
            encoder_no = MemvidEncoder()
            encoder_yes = MemvidEncoder()
            encoder_no.add_chunks(chunks)
            encoder_yes.add_chunks(chunks)
            
            video_no = Path(self.temp_dir) / f"{case_name}_no.mp4"
            index_no = Path(self.temp_dir) / f"{case_name}_no.json"
            video_yes = Path(self.temp_dir) / f"{case_name}_yes.mp4"
            index_yes = Path(self.temp_dir) / f"{case_name}_yes.json"
            
            # Build both versions
            result_no = encoder_no.build_video(
                str(video_no), str(index_no),
                enable_frame_ordering=False, show_progress=False
            )
            
            result_yes = encoder_yes.build_video(
                str(video_yes), str(index_yes),
                enable_frame_ordering=True, show_progress=False
            )
            
            # Calculate detailed compression metrics
            size_no = video_no.stat().st_size
            size_yes = video_yes.stat().st_size
            
            compression_results[case_name] = {
                "size_without_ordering": size_no,
                "size_with_ordering": size_yes,
                "compression_ratio": size_no / size_yes,
                "bytes_saved": size_no - size_yes,
                "percentage_reduction": (size_no - size_yes) / size_no * 100,
                "frame_count": len(chunks)
            }
        
        # Should have results for both test cases
        self.assertEqual(len(compression_results), 2)
        
        # All metrics should be calculable
        for case_name, results in compression_results.items():
            self.assertGreater(results["size_without_ordering"], 0)
            self.assertGreater(results["size_with_ordering"], 0)
            self.assertIsInstance(results["compression_ratio"], float)
            self.assertIsInstance(results["percentage_reduction"], float)
    
    def test_compression_benefit_reporting(self):
        """Test comprehensive compression benefit reporting"""
        chunks = [
            "Bright content with minimal density",
            "██████████████████████████████████",  # Dark content
            "Mixed content ████     ████     ██",
            "Another bright content sample",
            "██████████████████████████████████"   # Dark content
        ]
        
        encoder_baseline = MemvidEncoder()
        encoder_optimized = MemvidEncoder()
        encoder_baseline.add_chunks(chunks)
        encoder_optimized.add_chunks(chunks)
        
        baseline_video = Path(self.temp_dir) / "baseline.mp4"
        baseline_index = Path(self.temp_dir) / "baseline.json"
        optimized_video = Path(self.temp_dir) / "optimized.mp4"
        optimized_index = Path(self.temp_dir) / "optimized.json"
        
        # Build baseline
        baseline_result = encoder_baseline.build_video(
            str(baseline_video), str(baseline_index),
            enable_frame_ordering=False, show_progress=False
        )
        
        # Build optimized
        optimized_result = encoder_optimized.build_video(
            str(optimized_video), str(optimized_index),
            enable_frame_ordering=True, show_progress=False
        )
        
        # Generate comprehensive report
        baseline_size = baseline_video.stat().st_size
        optimized_size = optimized_video.stat().st_size
        
        report = {
            "test_metadata": {
                "frame_count": len(chunks),
                "chunk_avg_length": sum(len(c) for c in chunks) / len(chunks),
                "content_type": "mixed_density"
            },
            "baseline_metrics": {
                "file_size_bytes": baseline_size,
                "build_time": baseline_result.get("build_time", 0)
            },
            "optimized_metrics": {
                "file_size_bytes": optimized_size,
                "build_time": optimized_result.get("build_time", 0),
                "frame_ordering_time": optimized_result.get("frame_ordering", {}).get("optimization_time", 0),
                "frame_reordering": optimized_result.get("frame_ordering", {}).get("optimized_order", [])
            },
            "compression_analysis": {
                "compression_ratio": baseline_size / optimized_size,
                "bytes_saved": baseline_size - optimized_size,
                "percentage_improvement": (baseline_size - optimized_size) / baseline_size * 100,
                "size_reduction_per_frame": (baseline_size - optimized_size) / len(chunks)
            }
        }
        
        # Should generate complete report
        self.assertIn("test_metadata", report)
        self.assertIn("baseline_metrics", report)
        self.assertIn("optimized_metrics", report)
        self.assertIn("compression_analysis", report)
        
        # Report should be serializable
        report_file = Path(self.temp_dir) / "compression_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.assertTrue(report_file.exists())


if __name__ == '__main__':
    unittest.main() 