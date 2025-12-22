#!/usr/bin/env python3
"""
Demo: Early Stopping in Frame Ordering Optimization

This demo showcases the early stopping functionality that was implemented in Phase 4.2.2.
Early stopping allows the frame ordering algorithm to terminate early when the improvement
between resolution levels becomes minimal, saving computation time while maintaining
most of the optimization benefits.
"""

import numpy as np
import time
import sys
import os

# Add memvid to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memvid.frame_ordering import FrameOrderingOptimizer


def create_test_frames(count=50, pattern_type='mixed'):
    """Create test frames with different characteristics for early stopping demo."""
    frames = []
    
    if pattern_type == 'identical':
        # Create identical frames - should trigger early stopping immediately
        base_frame = np.ones((64, 64), dtype=np.uint8) * 128
        frames = [base_frame.copy() for _ in range(count)]
        
    elif pattern_type == 'similar':
        # Create very similar frames - should trigger early stopping quickly
        for i in range(count):
            frame = np.ones((64, 64), dtype=np.uint8) * 128
            # Add minimal variation
            frame[i % 64, (i * 2) % 64] = 255
            frames.append(frame)
            
    elif pattern_type == 'diverse':
        # Create diverse frames - should not trigger early stopping
        for i in range(count):
            frame = np.zeros((64, 64), dtype=np.uint8)
            if i % 4 == 0:
                frame[:32, :32] = 255  # Top-left quadrant
            elif i % 4 == 1:
                frame[:32, 32:] = 255  # Top-right quadrant
            elif i % 4 == 2:
                frame[32:, :32] = 255  # Bottom-left quadrant
            else:
                frame[32:, 32:] = 255  # Bottom-right quadrant
            
            # Add unique variations
            frame[i % 64, (i * 3) % 64] = 128
            frames.append(frame)
            
    elif pattern_type == 'mixed':
        # Create mixed frames - some similar, some diverse
        for i in range(count):
            if i < count // 3:
                # First third: similar frames
                frame = np.ones((64, 64), dtype=np.uint8) * 100
                frame[i % 32, i % 32] = 255
            elif i < 2 * count // 3:
                # Second third: diverse frames
                frame = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            else:
                # Last third: gradient frames
                frame = np.zeros((64, 64), dtype=np.uint8)
                brightness = int(255 * ((i - 2 * count // 3) / (count // 3)))
                frame[:, :i-2*count//3+1] = brightness
            frames.append(frame)
    
    return frames


def run_early_stopping_comparison(frames, frame_type, threshold=0.02):
    """Compare optimization with and without early stopping."""
    print(f"\n🔍 Testing {frame_type} frames ({len(frames)} frames)")
    print("=" * 60)
    
    # Test with early stopping enabled
    optimizer_early = FrameOrderingOptimizer(
        power_base=2,
        max_resolution=32,
        early_stopping_enabled=True,
        early_stopping_threshold=threshold
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
    
    # Analyze results
    early_meta = result_early['metadata']
    full_meta = result_full['metadata']
    
    print(f"⚡ Early Stopping Results:")
    print(f"   Time: {time_early:.3f}s")
    print(f"   Triggered: {early_meta.get('early_stopping_triggered', False)}")
    if early_meta.get('early_stopping_triggered'):
        print(f"   Stopped at resolution: {early_meta.get('early_stopping_resolution')}")
        print(f"   Resolutions processed: {len(early_meta.get('resolution_improvements', []))+1}")
    else:
        print(f"   Resolutions processed: {len(early_meta['resolution_sequence'])}")
    
    print(f"\n🔄 Full Optimization Results:")
    print(f"   Time: {time_full:.3f}s")
    print(f"   Resolutions processed: {len(full_meta['resolution_sequence'])}")
    
    # Performance comparison
    speedup = time_full / time_early if time_early > 0 else 1.0
    time_saved = time_full - time_early
    
    print(f"\n📊 Performance Comparison:")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Time saved: {time_saved:.3f}s ({(time_saved/time_full)*100:.1f}%)")
    
    # Quality comparison
    early_order = result_early['frame_order']
    full_order = result_full['frame_order']
    
    # Calculate how many positions are different
    differences = sum(1 for i in range(len(early_order)) if early_order[i] != full_order[i])
    similarity = (len(early_order) - differences) / len(early_order) * 100
    
    print(f"   Order similarity: {similarity:.1f}%")
    print(f"   Different positions: {differences}/{len(early_order)}")
    
    # Show improvement tracking if available
    if 'resolution_improvements' in early_meta:
        improvements = early_meta['resolution_improvements']
        print(f"\n📈 Improvement Tracking:")
        for i, improvement in enumerate(improvements):
            resolution = early_meta['resolution_sequence'][i+1] if i+1 < len(early_meta['resolution_sequence']) else 'N/A'
            print(f"   Resolution {resolution}: {improvement:.3f} ({improvement*100:.1f}% change)")
    
    return {
        'speedup': speedup,
        'time_saved': time_saved,
        'similarity': similarity,
        'early_stopped': early_meta.get('early_stopping_triggered', False)
    }


def main():
    """Run the early stopping demonstration."""
    print("🚀 Early Stopping Frame Ordering Optimization Demo")
    print("=" * 60)
    print("This demo showcases the early stopping functionality that saves")
    print("computation time by stopping when improvements become minimal.")
    
    # Test different frame types
    test_cases = [
        ('identical', 30),
        ('similar', 40),
        ('diverse', 50),
        ('mixed', 60)
    ]
    
    results = {}
    
    for frame_type, count in test_cases:
        frames = create_test_frames(count, frame_type)
        results[frame_type] = run_early_stopping_comparison(frames, frame_type)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 EARLY STOPPING SUMMARY")
    print("=" * 60)
    
    total_speedup = 0
    total_time_saved = 0
    early_stopped_count = 0
    
    for frame_type, result in results.items():
        print(f"{frame_type.capitalize():>10}: {result['speedup']:.2f}x speedup, "
              f"{result['similarity']:.1f}% similarity, "
              f"{'✅ stopped early' if result['early_stopped'] else '❌ full optimization'}")
        
        total_speedup += result['speedup']
        total_time_saved += result['time_saved']
        if result['early_stopped']:
            early_stopped_count += 1
    
    avg_speedup = total_speedup / len(results)
    
    print(f"\n🏆 Overall Performance:")
    print(f"   Average speedup: {avg_speedup:.2f}x")
    print(f"   Total time saved: {total_time_saved:.3f}s")
    print(f"   Early stopping triggered: {early_stopped_count}/{len(results)} cases")
    
    print(f"\n✅ Early stopping provides significant performance benefits!")
    print(f"   Recommended for production use with threshold 0.01-0.05")
    print(f"   Best results with similar/repetitive frame content")


if __name__ == "__main__":
    main() 