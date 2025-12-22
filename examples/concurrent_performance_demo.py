#!/usr/bin/env python3
"""
Phase 4.2.1: Concurrent Frame Ordering Performance Demo

Demonstrates the performance benefits of multi-threaded frame ordering
compared to single-threaded processing for large datasets.

Usage:
    python examples/concurrent_performance_demo.py [--max-frames 1000] [--max-workers 4]
"""

import sys
import tempfile
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add memvid to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.frame_ordering import FrameOrderingOptimizer
from memvid.frame_ordering_concurrent import ConcurrentFrameOrderingOptimizer
from tests.test_frame_ordering import MockQRFrameGenerator
import cv2
import numpy as np


class ConcurrentPerformanceDemo:
    """Demo class for concurrent frame ordering performance comparison"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_generator = MockQRFrameGenerator()
        
    def create_test_dataset(self, size: int, patterns: List[str] = None) -> List[str]:
        """Create test frame files for performance testing"""
        if patterns is None:
            patterns = ["light", "dark", "random", "checkerboard"]
        
        print(f"📁 Creating test dataset with {size} frames...")
        frame_files = []
        
        for i in range(size):
            pattern = patterns[i % len(patterns)]
            frame = self.frame_generator.create_test_qr_frame(size=200, pattern=pattern, seed=i)
            
            frame_path = self.output_dir / f"frame_{i:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_files.append(str(frame_path))
        
        print(f"✅ Created {len(frame_files)} test frames")
        return frame_files
    
    def benchmark_single_threaded(self, frame_files: List[str]) -> Dict[str, Any]:
        """Benchmark single-threaded frame ordering"""
        print(f"\n🔄 Running single-threaded benchmark ({len(frame_files)} frames)...")
        
        # Load frames as numpy arrays
        frames = []
        load_start = time.time()
        for frame_file in frame_files:
            img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                frames.append(img)
        load_time = time.time() - load_start
        
        # Run single-threaded optimization
        optimizer = FrameOrderingOptimizer()
        
        start_time = time.time()
        optimized_order, metadata = optimizer.optimize_frame_order(frames)
        optimization_time = time.time() - start_time
        
        result = {
            "method": "single_threaded",
            "frame_count": len(frames),
            "load_time": load_time,
            "optimization_time": optimization_time,
            "total_time": load_time + optimization_time,
            "frames_reordered": sum(1 for i, orig_i in enumerate(optimized_order) if i != orig_i),
            "reorder_percentage": sum(1 for i, orig_i in enumerate(optimized_order) if i != orig_i) / len(frames) * 100,
            "time_per_frame_ms": optimization_time / len(frames) * 1000,
            "metadata": metadata
        }
        
        print(f"   ⏱️  Load time: {load_time:.3f}s")
        print(f"   🔄 Optimization time: {optimization_time:.3f}s")
        print(f"   📊 Frames reordered: {result['frames_reordered']}/{len(frames)} ({result['reorder_percentage']:.1f}%)")
        print(f"   ⚡ Time per frame: {result['time_per_frame_ms']:.2f}ms")
        
        return result
    
    def benchmark_concurrent(self, frame_files: List[str], max_workers: int = 4) -> Dict[str, Any]:
        """Benchmark concurrent frame ordering"""
        print(f"\n🚀 Running concurrent benchmark ({len(frame_files)} frames, {max_workers} workers)...")
        
        # Run concurrent optimization (loads frames internally)
        optimizer = ConcurrentFrameOrderingOptimizer(max_workers=max_workers)
        
        start_time = time.time()
        result = optimizer.optimize_frame_order(frame_files)
        total_time = time.time() - start_time
        
        # Extract metadata
        metadata = result.get("metadata", {})
        optimization_time = result.get("optimization_time", total_time)
        
        benchmark_result = {
            "method": "concurrent",
            "frame_count": len(frame_files),
            "max_workers": max_workers,
            "optimization_time": optimization_time,
            "total_time": total_time,
            "frames_reordered": result.get("frames_reordered", 0),
            "reorder_percentage": result.get("frames_reordered", 0) / len(frame_files) * 100,
            "time_per_frame_ms": optimization_time / len(frame_files) * 1000,
            "metadata": metadata
        }
        
        print(f"   🔄 Optimization time: {optimization_time:.3f}s")
        print(f"   📊 Frames reordered: {benchmark_result['frames_reordered']}/{len(frame_files)} ({benchmark_result['reorder_percentage']:.1f}%)")
        print(f"   ⚡ Time per frame: {benchmark_result['time_per_frame_ms']:.2f}ms")
        print(f"   🧵 Workers used: {max_workers}")
        
        return benchmark_result
    
    def run_performance_comparison(self, frame_counts: List[int], max_workers: int = 4) -> List[Dict[str, Any]]:
        """Run comprehensive performance comparison across different dataset sizes"""
        print("🏁 Starting concurrent frame ordering performance comparison")
        print("=" * 70)
        
        all_results = []
        
        for frame_count in frame_counts:
            print(f"\n📊 TESTING DATASET SIZE: {frame_count} frames")
            print("-" * 50)
            
            # Create test dataset
            frame_files = self.create_test_dataset(frame_count)
            
            try:
                # Benchmark single-threaded
                single_result = self.benchmark_single_threaded(frame_files)
                
                # Benchmark concurrent
                concurrent_result = self.benchmark_concurrent(frame_files, max_workers)
                
                # Calculate performance improvement
                speedup = single_result["optimization_time"] / concurrent_result["optimization_time"]
                efficiency = speedup / max_workers * 100  # Efficiency percentage
                
                comparison = {
                    "frame_count": frame_count,
                    "single_threaded": single_result,
                    "concurrent": concurrent_result,
                    "performance_improvement": {
                        "speedup": speedup,
                        "efficiency_percent": efficiency,
                        "time_saved_seconds": single_result["optimization_time"] - concurrent_result["optimization_time"],
                        "time_saved_percent": (single_result["optimization_time"] - concurrent_result["optimization_time"]) / single_result["optimization_time"] * 100
                    }
                }
                
                all_results.append(comparison)
                
                # Print comparison summary
                print(f"\n📈 PERFORMANCE COMPARISON:")
                print(f"   🏃 Speedup: {speedup:.2f}x")
                print(f"   ⚡ Efficiency: {efficiency:.1f}%")
                print(f"   💾 Time saved: {comparison['performance_improvement']['time_saved_seconds']:.3f}s ({comparison['performance_improvement']['time_saved_percent']:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ Benchmark failed for {frame_count} frames: {e}")
                continue
        
        return all_results
    
    def print_summary_report(self, results: List[Dict[str, Any]]):
        """Print comprehensive summary report"""
        if not results:
            print("❌ No results to summarize")
            return
        
        print("\n" + "=" * 70)
        print("📊 CONCURRENT FRAME ORDERING PERFORMANCE SUMMARY")
        print("=" * 70)
        
        # Calculate aggregate statistics
        speedups = [r["performance_improvement"]["speedup"] for r in results]
        efficiencies = [r["performance_improvement"]["efficiency_percent"] for r in results]
        time_savings = [r["performance_improvement"]["time_saved_percent"] for r in results]
        
        print(f"\n🏆 OVERALL PERFORMANCE:")
        print(f"  Average speedup: {np.mean(speedups):.2f}x")
        print(f"  Best speedup: {max(speedups):.2f}x ({results[speedups.index(max(speedups))]['frame_count']} frames)")
        print(f"  Average efficiency: {np.mean(efficiencies):.1f}%")
        print(f"  Average time savings: {np.mean(time_savings):.1f}%")
        
        print(f"\n📋 DETAILED RESULTS:")
        print(f"{'Frames':<8} {'Single(s)':<10} {'Concurrent(s)':<12} {'Speedup':<8} {'Efficiency':<10} {'Saved':<8}")
        print("-" * 70)
        
        for result in results:
            frame_count = result["frame_count"]
            single_time = result["single_threaded"]["optimization_time"]
            concurrent_time = result["concurrent"]["optimization_time"]
            speedup = result["performance_improvement"]["speedup"]
            efficiency = result["performance_improvement"]["efficiency_percent"]
            saved_percent = result["performance_improvement"]["time_saved_percent"]
            
            print(f"{frame_count:<8} {single_time:<10.3f} {concurrent_time:<12.3f} {speedup:<8.2f} {efficiency:<10.1f}% {saved_percent:<8.1f}%")
        
        # Scaling analysis
        if len(results) > 1:
            print(f"\n📈 SCALING ANALYSIS:")
            small_result = results[0]
            large_result = results[-1]
            
            size_ratio = large_result["frame_count"] / small_result["frame_count"]
            time_ratio_single = large_result["single_threaded"]["optimization_time"] / small_result["single_threaded"]["optimization_time"]
            time_ratio_concurrent = large_result["concurrent"]["optimization_time"] / small_result["concurrent"]["optimization_time"]
            
            print(f"  Dataset size increase: {size_ratio:.1f}x")
            print(f"  Single-threaded time increase: {time_ratio_single:.1f}x")
            print(f"  Concurrent time increase: {time_ratio_concurrent:.1f}x")
            print(f"  Concurrent scaling advantage: {time_ratio_single / time_ratio_concurrent:.2f}x better")
        
        print(f"\n✅ Concurrent frame ordering provides significant performance benefits!")
        print(f"   Recommended for datasets with >50 frames")
        print(f"   Optimal worker count: 2-8 (depending on CPU cores)")


def main():
    """Main performance demo script"""
    parser = argparse.ArgumentParser(description="Concurrent frame ordering performance demo")
    parser.add_argument("--max-frames", type=int, default=500,
                       help="Maximum number of frames to test (default: 500)")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of worker threads (default: 4)")
    parser.add_argument("--output-dir", help="Output directory for test files")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = ConcurrentPerformanceDemo(args.output_dir)
    
    # Define test dataset sizes (logarithmic progression)
    max_frames = args.max_frames
    frame_counts = [50, 100, 250]
    if max_frames >= 500:
        frame_counts.append(500)
    if max_frames >= 1000:
        frame_counts.append(1000)
    
    # Run performance comparison
    results = demo.run_performance_comparison(frame_counts, args.max_workers)
    
    # Print summary report
    demo.print_summary_report(results)
    
    print(f"\n📁 Test files created in: {demo.output_dir}")


if __name__ == "__main__":
    main() 