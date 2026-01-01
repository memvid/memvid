#!/usr/bin/env python3
"""
Phase 4.1.2: Performance Scaling Analysis

Test frame ordering performance with different dataset sizes:
- Small: 10-50 frames
- Medium: 100-500 frames  
- Large: 1000+ frames

Measures:
- Optimization time scaling
- Memory usage patterns
- Compression benefit consistency
- Performance overhead at scale

Usage:
    python examples/scaling_benchmark.py [--max-size 1000] [--output scaling_results.json]
"""

import sys
import tempfile
import time
import json
import os
from pathlib import Path
import argparse
from typing import Dict, List, Any
import gc

# Add memvid to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.encoder import MemvidEncoder


class ScalingBenchmark:
    """Performance scaling analysis for frame ordering"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def generate_test_content(self, size: int, content_type: str = "mixed") -> List[str]:
        """Generate test content of specified size and type"""
        
        if content_type == "mixed":
            # Mixed content with varying density - should show good compression benefits
            patterns = [
                "Bright content with lots of whitespace and minimal density",
                "██████████████████████████████████████████████████████████",  # Very dark
                "Mixed content ██████     ██████     ██████     ██████     ",
                "Another bright chunk with sparse patterns and light content",
                "████████████████████████████████████████████████████████████",  # Very dark
                "Medium density ████     ████     ████     ████     ████   ",
                "Sparse content with minimal black pixels and lots of space",
                "█████████████████████ DENSE █████████████████████████████",
            ]
            
        elif content_type == "similar":
            # Similar content - moderate compression benefits
            patterns = [
                "Similar content pattern with consistent structure and formatting",
                "Similar content pattern with consistent structure and layout",
                "Similar content pattern with consistent structure and design",
                "Similar content pattern with consistent structure and style",
            ]
            
        elif content_type == "random":
            # Random content - minimal compression benefits
            patterns = [
                f"Random content {i} with unpredictable patterns {hash(str(i*7+13)) % 10000} unique"
                for i in range(20)
            ]
        
        # Generate content by cycling through patterns
        chunks = []
        for i in range(size):
            pattern = patterns[i % len(patterns)]
            # Add some variation while keeping pattern recognizable
            chunk = f"{pattern} [chunk_{i}]"
            chunks.append(chunk)
            
        return chunks
    
    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            # Fallback if psutil not available
            return 0.0
    
    def benchmark_dataset_size(self, size: int, content_type: str = "mixed") -> Dict[str, Any]:
        """Benchmark frame ordering performance for specific dataset size"""
        print(f"📊 Testing dataset size: {size} frames ({content_type} content)")
        
        # Generate test content
        chunks = self.generate_test_content(size, content_type)
        
        # Create encoders
        encoder_baseline = MemvidEncoder()
        encoder_optimized = MemvidEncoder()
        encoder_baseline.add_chunks(chunks)
        encoder_optimized.add_chunks(chunks)
        
        # Setup file paths
        baseline_video = self.output_dir / f"scale_{size}_{content_type}_baseline.mp4"
        baseline_index = self.output_dir / f"scale_{size}_{content_type}_baseline.json"
        optimized_video = self.output_dir / f"scale_{size}_{content_type}_optimized.mp4"
        optimized_index = self.output_dir / f"scale_{size}_{content_type}_optimized.json"
        
        # Measure memory before
        memory_before = self.measure_memory_usage()
        gc.collect()  # Clean up before measurement
        
        # Benchmark baseline encoding
        print(f"   📹 Building baseline ({size} frames)...")
        start_time = time.time()
        baseline_result = encoder_baseline.build_video(
            str(baseline_video),
            str(baseline_index),
            enable_frame_ordering=False,
            show_progress=False
        )
        baseline_time = time.time() - start_time
        baseline_size = baseline_video.stat().st_size
        
        # Clean up and measure memory
        del encoder_baseline
        gc.collect()
        memory_after_baseline = self.measure_memory_usage()
        
        # Benchmark optimized encoding
        print(f"   🔄 Building optimized ({size} frames)...")
        start_time = time.time()
        optimized_result = encoder_optimized.build_video(
            str(optimized_video),
            str(optimized_index),
            enable_frame_ordering=True,
            show_progress=False
        )
        optimized_time = time.time() - start_time
        optimized_size = optimized_video.stat().st_size
        
        # Measure memory after optimization
        memory_after_optimized = self.measure_memory_usage()
        
        # Extract frame ordering metadata
        frame_ordering_time = 0
        frame_reordering = []
        if "frame_ordering" in optimized_result:
            frame_ordering_time = optimized_result["frame_ordering"].get("optimization_time", 0)
            frame_reordering = optimized_result["frame_ordering"].get("optimized_order", [])
        
        # Calculate performance metrics
        compression_ratio = baseline_size / optimized_size if optimized_size > 0 else 1.0
        percentage_improvement = ((baseline_size - optimized_size) / baseline_size * 100) if baseline_size > 0 else 0
        overhead_percentage = (frame_ordering_time / optimized_time * 100) if optimized_time > 0 else 0
        
        # Calculate performance per frame
        time_per_frame = frame_ordering_time / size if size > 0 else 0
        
        result = {
            "dataset_size": size,
            "content_type": content_type,
            "baseline_size_bytes": baseline_size,
            "optimized_size_bytes": optimized_size,
            "compression_ratio": compression_ratio,
            "percentage_improvement": percentage_improvement,
            "baseline_build_time": baseline_time,
            "optimized_build_time": optimized_time,
            "frame_ordering_time": frame_ordering_time,
            "overhead_percentage": overhead_percentage,
            "time_per_frame_ms": time_per_frame * 1000,  # Convert to milliseconds
            "frames_reordered": frame_reordering != list(range(size)),
            "memory_usage": {
                "before_mb": memory_before,
                "after_baseline_mb": memory_after_baseline,
                "after_optimized_mb": memory_after_optimized,
                "peak_usage_mb": max(memory_after_baseline, memory_after_optimized),
                "memory_per_frame_kb": (memory_after_optimized - memory_before) * 1024 / size if size > 0 else 0
            }
        }
        
        print(f"   ✅ Compression: {compression_ratio:.3f}x ({percentage_improvement:.1f}% improvement)")
        print(f"   ⏱️  Ordering time: {frame_ordering_time:.3f}s ({time_per_frame*1000:.2f}ms/frame)")
        print(f"   💾 Memory usage: {memory_after_optimized - memory_before:.1f}MB ({result['memory_usage']['memory_per_frame_kb']:.2f}KB/frame)")
        
        # Clean up for next test
        del encoder_optimized
        gc.collect()
        
        return result
    
    def run_scaling_analysis(self, max_size: int = 1000, content_types: List[str] = ["mixed"]) -> List[Dict[str, Any]]:
        """Run comprehensive scaling analysis"""
        print("📈 Starting performance scaling analysis...")
        print("=" * 60)
        
        # Define test sizes - logarithmic scaling to cover wide range efficiently
        test_sizes = [10, 25, 50, 100, 250, 500]
        if max_size >= 1000:
            test_sizes.append(1000)
        if max_size >= 2000:
            test_sizes.append(2000)
        
        all_results = []
        
        for content_type in content_types:
            print(f"\n🎯 Testing content type: {content_type}")
            print("-" * 40)
            
            for size in test_sizes:
                if size > max_size:
                    continue
                    
                try:
                    result = self.benchmark_dataset_size(size, content_type)
                    all_results.append(result)
                    self.results.append(result)
                except Exception as e:
                    print(f"   ❌ Failed for size {size}: {e}")
                    continue
        
        print(f"\n🎉 Scaling analysis complete! Tested {len(all_results)} configurations")
        return all_results
    
    def analyze_scaling_trends(self) -> Dict[str, Any]:
        """Analyze scaling trends and performance characteristics"""
        if len(self.results) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Sort results by dataset size
        sorted_results = sorted(self.results, key=lambda x: x["dataset_size"])
        
        sizes = [r["dataset_size"] for r in sorted_results]
        times = [r["frame_ordering_time"] for r in sorted_results]
        times_per_frame = [r["time_per_frame_ms"] for r in sorted_results]
        compression_ratios = [r["compression_ratio"] for r in sorted_results]
        memory_usage = [r["memory_usage"]["peak_usage_mb"] for r in sorted_results]
        
        # Calculate scaling efficiency
        def calculate_scaling_factor(sizes, times):
            """Calculate how close to linear/quadratic scaling we are"""
            if len(sizes) < 2:
                return 0
            
            size_ratios = [sizes[i] / sizes[i-1] for i in range(1, len(sizes))]
            time_ratios = [times[i] / times[i-1] for i in range(1, len(times))]
            
            # Linear scaling would have time_ratio ≈ size_ratio
            # Quadratic scaling would have time_ratio ≈ size_ratio^2
            scaling_factors = [time_ratios[i] / size_ratios[i] for i in range(len(time_ratios))]
            return sum(scaling_factors) / len(scaling_factors)
        
        scaling_factor = calculate_scaling_factor(sizes, times)
        
        # Determine scaling behavior
        if scaling_factor < 1.2:
            scaling_behavior = "sub-linear (excellent)"
        elif scaling_factor < 1.8:
            scaling_behavior = "near-linear (good)"
        elif scaling_factor < 3.0:
            scaling_behavior = "super-linear (acceptable)"
        else:
            scaling_behavior = "quadratic+ (concerning)"
        
        analysis = {
            "scaling_summary": {
                "min_size": min(sizes),
                "max_size": max(sizes),
                "size_range_factor": max(sizes) / min(sizes),
                "scaling_factor": scaling_factor,
                "scaling_behavior": scaling_behavior
            },
            "performance_trends": {
                "min_time_per_frame_ms": min(times_per_frame),
                "max_time_per_frame_ms": max(times_per_frame),
                "avg_time_per_frame_ms": sum(times_per_frame) / len(times_per_frame),
                "time_per_frame_stability": max(times_per_frame) / min(times_per_frame)
            },
            "compression_consistency": {
                "min_compression_ratio": min(compression_ratios),
                "max_compression_ratio": max(compression_ratios),
                "avg_compression_ratio": sum(compression_ratios) / len(compression_ratios),
                "compression_stability": max(compression_ratios) / min(compression_ratios)
            },
            "memory_scaling": {
                "min_memory_mb": min(memory_usage),
                "max_memory_mb": max(memory_usage),
                "memory_growth_factor": max(memory_usage) / min(memory_usage) if min(memory_usage) > 0 else 0
            },
            "largest_test": {
                "size": sorted_results[-1]["dataset_size"],
                "time": sorted_results[-1]["frame_ordering_time"],
                "compression_ratio": sorted_results[-1]["compression_ratio"],
                "memory_mb": sorted_results[-1]["memory_usage"]["peak_usage_mb"]
            }
        }
        
        return analysis
    
    def save_results(self, output_file: str = None):
        """Save scaling analysis results"""
        if not output_file:
            output_file = self.output_dir / "scaling_results.json"
        
        output_path = Path(output_file)
        
        full_report = {
            "scaling_analysis": self.analyze_scaling_trends(),
            "detailed_results": self.results,
            "metadata": {
                "total_tests": len(self.results),
                "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "output_directory": str(self.output_dir)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        print(f"📁 Scaling results saved to: {output_path}")
        return output_path
    
    def print_scaling_summary(self):
        """Print formatted scaling analysis summary"""
        analysis = self.analyze_scaling_trends()
        
        print("\n📈 SCALING ANALYSIS SUMMARY")
        print("=" * 50)
        
        if "error" in analysis:
            print(f"❌ {analysis['error']}")
            return
        
        scaling = analysis["scaling_summary"]
        performance = analysis["performance_trends"]
        compression = analysis["compression_consistency"]
        memory = analysis["memory_scaling"]
        largest = analysis["largest_test"]
        
        print(f"Dataset size range: {scaling['min_size']} → {scaling['max_size']} frames ({scaling['size_range_factor']:.1f}x range)")
        print(f"Scaling behavior: {scaling['scaling_behavior']} (factor: {scaling['scaling_factor']:.2f})")
        
        print(f"\n⏱️  PERFORMANCE SCALING:")
        print(f"  Time per frame: {performance['min_time_per_frame_ms']:.2f} → {performance['max_time_per_frame_ms']:.2f}ms")
        print(f"  Average time per frame: {performance['avg_time_per_frame_ms']:.2f}ms")
        print(f"  Performance stability: {performance['time_per_frame_stability']:.2f}x")
        
        print(f"\n🗜️  COMPRESSION CONSISTENCY:")
        print(f"  Compression ratio range: {compression['min_compression_ratio']:.3f}x → {compression['max_compression_ratio']:.3f}x")
        print(f"  Average compression: {compression['avg_compression_ratio']:.3f}x")
        print(f"  Compression stability: {compression['compression_stability']:.2f}x")
        
        print(f"\n💾 MEMORY SCALING:")
        print(f"  Memory usage: {memory['min_memory_mb']:.1f}MB → {memory['max_memory_mb']:.1f}MB")
        print(f"  Memory growth factor: {memory['memory_growth_factor']:.2f}x")
        
        print(f"\n🏆 LARGEST DATASET TESTED:")
        print(f"  Size: {largest['size']} frames")
        print(f"  Optimization time: {largest['time']:.3f}s")
        print(f"  Compression ratio: {largest['compression_ratio']:.3f}x")
        print(f"  Memory usage: {largest['memory_mb']:.1f}MB")


def main():
    """Main scaling benchmark script"""
    parser = argparse.ArgumentParser(description="Analyze frame ordering performance scaling")
    parser.add_argument("--max-size", type=int, default=1000,
                       help="Maximum dataset size to test (default: 1000)")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--output-dir", help="Output directory for test files")
    parser.add_argument("--content-types", nargs="+", default=["mixed"],
                       choices=["mixed", "similar", "random"],
                       help="Content types to test")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = ScalingBenchmark(args.output_dir)
    
    # Run scaling analysis
    results = benchmark.run_scaling_analysis(args.max_size, args.content_types)
    
    # Print summary
    benchmark.print_scaling_summary()
    
    # Save results
    benchmark.save_results(args.output)
    
    print(f"\n✅ Scaling analysis complete! Check output directory: {benchmark.output_dir}")


if __name__ == "__main__":
    main() 