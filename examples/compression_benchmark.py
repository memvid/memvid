#!/usr/bin/env python3
"""
Phase 4.1: Real Compression Benchmarking Script

This script measures the actual compression benefits of frame ordering
across different content types and dataset sizes.

Features:
- Real video compression analysis with/without frame ordering
- Multiple content types and codec testing
- Performance scaling analysis  
- Detailed compression ratio reporting
- CSV/JSON output for analysis

Usage:
    python examples/compression_benchmark.py [--output results.json] [--csv]
"""

import sys
import tempfile
import time
import json
import csv
from pathlib import Path
import argparse
from typing import Dict, List, Any, Tuple

# Add memvid to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.encoder import MemvidEncoder
from memvid.retriever import MemvidRetriever


class CompressionBenchmark:
    """Comprehensive compression benchmarking system for frame ordering"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def create_test_datasets(self) -> Dict[str, List[str]]:
        """Create diverse test datasets with different compression characteristics"""
        datasets = {
            # High contrast content (should benefit most from ordering)
            "high_contrast": [
                "Bright content with lots of whitespace and minimal density",
                "██████████████████████████████████████████████████████████",  # Very dark
                "Another bright chunk with sparse patterns and light content",
                "████████████████████████████████████████████████████████████",  # Very dark
                "Sparse whitespace content with minimal black pixels present",
                "█████████████████████ DENSE █████████████████████████████",
                "Medium density content with moderate patterns and structure",
                "Mixed content with 50% density ██████     ██████     ██████",
            ],
            
            # Similar content (moderate ordering benefit)
            "similar_content": [
                "Similar content pattern A with consistent structure and formatting",
                "Similar content pattern B with consistent structure and formatting", 
                "Similar content pattern C with consistent structure and formatting",
                "Similar content pattern D with consistent structure and formatting",
                "Similar content pattern E with consistent structure and formatting",
                "Similar content pattern F with consistent structure and formatting",
            ],
            
            # Random content (minimal ordering benefit)
            "random_content": [
                f"Random content {i} with unpredictable patterns {hash(str(i*3+7)) % 1000} and unique data"
                for i in range(8)
            ],
            
            # Gradually changing density (should show clear ordering benefit)
            "gradient_density": [
                "████████████████████████████████████████████████████████████",  # 100% dense
                "████████████████████████████████████████████████  ████████",  # ~90% dense
                "████████████████████████████████████████     ████████████",  # ~80% dense  
                "████████████████████████████████     ████████████████████",  # ~70% dense
                "████████████████████████     ████████████████████████████",  # ~60% dense
                "████████████████     ████████████████████████████████████",  # ~50% dense
                "████████     ████████████████████████████████████████████",  # ~40% dense
                "     ████████████████████████████████████████████████████",  # ~30% dense
            ],
            
            # Wikipedia-like content (realistic test case)
            "wikipedia_like": [
                "Machine learning is a subset of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.",
                "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
                "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with interactions between computers and human language.",
                "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
                "Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, and computer science.",
                "Quantum computing is a type of computation that harnesses the collective properties of quantum states to perform calculations.",
            ]
        }
        
        return datasets
    
    def benchmark_compression(self, dataset_name: str, chunks: List[str], 
                            codec: str = "mp4v") -> Dict[str, Any]:
        """Benchmark compression for a specific dataset"""
        print(f"🔬 Benchmarking {dataset_name} with {len(chunks)} chunks ({codec} codec)")
        
        # Create encoders
        encoder_baseline = MemvidEncoder()
        encoder_optimized = MemvidEncoder()
        encoder_baseline.add_chunks(chunks)
        encoder_optimized.add_chunks(chunks)
        
        # Setup file paths
        baseline_video = self.output_dir / f"{dataset_name}_baseline.mp4"
        baseline_index = self.output_dir / f"{dataset_name}_baseline.json"
        optimized_video = self.output_dir / f"{dataset_name}_optimized.mp4"
        optimized_index = self.output_dir / f"{dataset_name}_optimized.json"
        
        # Benchmark baseline (no frame ordering)
        print(f"   📹 Building baseline video...")
        start_time = time.time()
        baseline_result = encoder_baseline.build_video(
            str(baseline_video),
            str(baseline_index),
            codec=codec,
            enable_frame_ordering=False,
            show_progress=False
        )
        baseline_time = time.time() - start_time
        
        # Benchmark optimized (with frame ordering)
        print(f"   🔄 Building optimized video...")
        start_time = time.time()
        optimized_result = encoder_optimized.build_video(
            str(optimized_video),
            str(optimized_index),
            codec=codec,
            enable_frame_ordering=True,
            show_progress=False
        )
        optimized_time = time.time() - start_time
        
        # Gather file size data
        baseline_size = baseline_video.stat().st_size
        optimized_size = optimized_video.stat().st_size
        
        # Calculate metrics
        compression_ratio = baseline_size / optimized_size if optimized_size > 0 else 1.0
        bytes_saved = baseline_size - optimized_size
        percentage_improvement = (bytes_saved / baseline_size * 100) if baseline_size > 0 else 0
        
        # Extract frame ordering metadata
        frame_ordering_time = 0
        frame_reordering = []
        if "frame_ordering" in optimized_result:
            frame_ordering_time = optimized_result["frame_ordering"].get("optimization_time", 0)
            frame_reordering = optimized_result["frame_ordering"].get("optimized_order", [])
        
        # Test retrieval accuracy
        retrieval_test = self.test_retrieval_accuracy(
            str(optimized_video), str(optimized_index), chunks
        )
        
        result = {
            "dataset_name": dataset_name,
            "codec": codec,
            "frame_count": len(chunks),
            "avg_chunk_length": sum(len(c) for c in chunks) / len(chunks),
            "baseline_size_bytes": baseline_size,
            "optimized_size_bytes": optimized_size,
            "compression_ratio": compression_ratio,
            "bytes_saved": bytes_saved,
            "percentage_improvement": percentage_improvement,
            "baseline_build_time": baseline_time,
            "optimized_build_time": optimized_time,
            "frame_ordering_time": frame_ordering_time,
            "frame_ordering_overhead_percent": (frame_ordering_time / optimized_time * 100) if optimized_time > 0 else 0,
            "frame_reordering": frame_reordering,
            "frames_reordered": frame_reordering != list(range(len(chunks))),
            "retrieval_accuracy": retrieval_test["accuracy"],
            "retrieval_time": retrieval_test["avg_retrieval_time"]
        }
        
        print(f"   ✅ Compression ratio: {compression_ratio:.3f}x")
        print(f"   📊 Size reduction: {percentage_improvement:.1f}% ({bytes_saved:,} bytes)")
        print(f"   ⏱️  Ordering overhead: {frame_ordering_time:.3f}s ({result['frame_ordering_overhead_percent']:.1f}%)")
        
        return result
    
    def test_retrieval_accuracy(self, video_path: str, index_path: str, 
                               original_chunks: List[str]) -> Dict[str, Any]:
        """Test that retrieval accuracy is preserved after frame ordering"""
        try:
            retriever = MemvidRetriever(video_path, index_path)
            
            # Test search accuracy with multiple queries
            test_queries = [
                "content",
                "pattern", 
                "machine learning",
                "dense"
            ]
            
            retrieval_times = []
            successful_retrievals = 0
            
            for query in test_queries:
                start_time = time.time()
                results = retriever.search(query, top_k=3)
                retrieval_time = time.time() - start_time
                retrieval_times.append(retrieval_time)
                
                if results:
                    successful_retrievals += 1
            
            accuracy = successful_retrievals / len(test_queries) if test_queries else 0
            avg_retrieval_time = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
            
            return {
                "accuracy": accuracy,
                "avg_retrieval_time": avg_retrieval_time,
                "total_queries": len(test_queries),
                "successful_retrievals": successful_retrievals
            }
            
        except Exception as e:
            print(f"   ⚠️  Retrieval test failed: {e}")
            return {"accuracy": 0, "avg_retrieval_time": 0}
    
    def run_full_benchmark(self, codecs: List[str] = ["mp4v"]) -> List[Dict[str, Any]]:
        """Run comprehensive benchmarking across all datasets and codecs"""
        print("🚀 Starting comprehensive compression benchmarking...")
        print("=" * 60)
        
        datasets = self.create_test_datasets()
        all_results = []
        
        for codec in codecs:
            print(f"\n🎥 Testing codec: {codec}")
            print("-" * 40)
            
            for dataset_name, chunks in datasets.items():
                try:
                    result = self.benchmark_compression(dataset_name, chunks, codec)
                    all_results.append(result)
                    self.results.append(result)
                except Exception as e:
                    print(f"   ❌ Benchmark failed for {dataset_name}: {e}")
                    continue
        
        print(f"\n🎉 Benchmarking complete! Tested {len(all_results)} configurations")
        return all_results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary analysis of benchmark results"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Calculate aggregate statistics
        compression_ratios = [r["compression_ratio"] for r in self.results]
        percentage_improvements = [r["percentage_improvement"] for r in self.results]
        frame_ordering_times = [r["frame_ordering_time"] for r in self.results]
        
        # Find best and worst performing datasets
        best_result = max(self.results, key=lambda x: x["compression_ratio"])
        worst_result = min(self.results, key=lambda x: x["compression_ratio"])
        
        # Calculate frame reordering statistics
        reordered_count = sum(1 for r in self.results if r["frames_reordered"])
        
        summary = {
            "total_tests": len(self.results),
            "compression_ratio_stats": {
                "mean": sum(compression_ratios) / len(compression_ratios),
                "min": min(compression_ratios),
                "max": max(compression_ratios),
                "median": sorted(compression_ratios)[len(compression_ratios)//2]
            },
            "percentage_improvement_stats": {
                "mean": sum(percentage_improvements) / len(percentage_improvements),
                "min": min(percentage_improvements),
                "max": max(percentage_improvements)
            },
            "performance_stats": {
                "avg_frame_ordering_time": sum(frame_ordering_times) / len(frame_ordering_times),
                "max_frame_ordering_time": max(frame_ordering_times),
                "min_frame_ordering_time": min(frame_ordering_times)
            },
            "frame_reordering_stats": {
                "tests_with_reordering": reordered_count,
                "reordering_rate": reordered_count / len(self.results) * 100
            },
            "best_performing_dataset": {
                "name": best_result["dataset_name"],
                "compression_ratio": best_result["compression_ratio"],
                "percentage_improvement": best_result["percentage_improvement"]
            },
            "worst_performing_dataset": {
                "name": worst_result["dataset_name"],
                "compression_ratio": worst_result["compression_ratio"],
                "percentage_improvement": worst_result["percentage_improvement"]
            }
        }
        
        return summary
    
    def save_results(self, output_file: str = None, format: str = "json"):
        """Save benchmark results to file"""
        if not output_file:
            output_file = self.output_dir / f"benchmark_results.{format}"
        
        output_path = Path(output_file)
        
        if format == "json":
            full_report = {
                "summary": self.generate_summary_report(),
                "detailed_results": self.results,
                "metadata": {
                    "total_tests": len(self.results),
                    "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "output_directory": str(self.output_dir)
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(full_report, f, indent=2)
                
        elif format == "csv":
            with open(output_path, 'w', newline='') as csvfile:
                if self.results:
                    fieldnames = self.results[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.results)
        
        print(f"📁 Results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print formatted summary of benchmark results"""
        summary = self.generate_summary_report()
        
        print("\n📊 COMPRESSION BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"Total tests conducted: {summary['total_tests']}")
        print(f"Tests with frame reordering: {summary['frame_reordering_stats']['tests_with_reordering']} ({summary['frame_reordering_stats']['reordering_rate']:.1f}%)")
        
        print(f"\n🏆 COMPRESSION PERFORMANCE:")
        print(f"  Average compression ratio: {summary['compression_ratio_stats']['mean']:.3f}x")
        print(f"  Best compression ratio: {summary['compression_ratio_stats']['max']:.3f}x ({summary['best_performing_dataset']['name']})")
        print(f"  Worst compression ratio: {summary['compression_ratio_stats']['min']:.3f}x ({summary['worst_performing_dataset']['name']})")
        
        print(f"\n📈 SIZE REDUCTION:")
        print(f"  Average improvement: {summary['percentage_improvement_stats']['mean']:.1f}%")
        print(f"  Best improvement: {summary['percentage_improvement_stats']['max']:.1f}% ({summary['best_performing_dataset']['name']})")
        print(f"  Worst improvement: {summary['percentage_improvement_stats']['min']:.1f}% ({summary['worst_performing_dataset']['name']})")
        
        print(f"\n⏱️  PERFORMANCE OVERHEAD:")
        print(f"  Average frame ordering time: {summary['performance_stats']['avg_frame_ordering_time']:.3f}s")
        print(f"  Max frame ordering time: {summary['performance_stats']['max_frame_ordering_time']:.3f}s")
        print(f"  Min frame ordering time: {summary['performance_stats']['min_frame_ordering_time']:.3f}s")


def main():
    """Main benchmarking script"""
    parser = argparse.ArgumentParser(description="Benchmark frame ordering compression benefits")
    parser.add_argument("--output", help="Output file for results (default: benchmark_results.json)")
    parser.add_argument("--csv", action="store_true", help="Also save results as CSV")
    parser.add_argument("--output-dir", help="Output directory for test files")
    parser.add_argument("--codecs", nargs="+", default=["mp4v"], 
                       help="Video codecs to test (default: mp4v)")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = CompressionBenchmark(args.output_dir)
    
    # Run comprehensive benchmarks
    results = benchmark.run_full_benchmark(args.codecs)
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    if args.output:
        benchmark.save_results(args.output, "json")
    else:
        benchmark.save_results(format="json")
    
    if args.csv:
        csv_file = args.output.replace(".json", ".csv") if args.output else None
        benchmark.save_results(csv_file, "csv")
    
    print(f"\n✅ Benchmarking complete! Check output directory: {benchmark.output_dir}")


if __name__ == "__main__":
    main() 