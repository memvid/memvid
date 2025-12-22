#!/usr/bin/env python3
"""
Simple frame content verification for the comparison videos
"""

import sys
from pathlib import Path
import cv2

# Add memvid to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.utils import decode_qr

def verify_videos():
    """Verify that both comparison videos contain identical content"""
    
    video1_path = "original_comparison_video.mp4"
    video2_path = "optimized_comparison_video.mp4"
    
    print("🔍 Manual Frame Content Verification")
    print("=" * 50)
    
    # Open both videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Could not open videos")
        return
    
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Original video: {frame_count1} frames")
    print(f"📹 Optimized video: {frame_count2} frames")
    
    if frame_count1 != frame_count2:
        print("❌ Frame count mismatch!")
        return
    
    # Test first 10 frames
    test_frames = min(10, frame_count1)
    print(f"\n🔍 Testing first {test_frames} frames...")
    
    matches = 0
    decode_failures = 0
    mismatches = 0
    
    for i in range(test_frames):
        # Read frames
        cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print(f"  Frame {i}: Failed to read")
            continue
        
        # Decode QR codes
        try:
            data1 = decode_qr(frame1)
            data2 = decode_qr(frame2)
            
            if data1 is None and data2 is None:
                decode_failures += 1
                print(f"  Frame {i}: Both decode failed")
            elif data1 is None or data2 is None:
                mismatches += 1
                print(f"  Frame {i}: One decode failed ({data1 is not None} vs {data2 is not None})")
            elif data1 == data2:
                matches += 1
                print(f"  Frame {i}: ✅ Identical ({len(data1)} chars)")
            else:
                mismatches += 1
                print(f"  Frame {i}: ❌ Content differs ({len(data1)} vs {len(data2)} chars)")
                print(f"    Original: {repr(data1[:50])}")
                print(f"    Optimized: {repr(data2[:50])}")
        
        except Exception as e:
            decode_failures += 1
            print(f"  Frame {i}: Decode error: {e}")
    
    cap1.release()
    cap2.release()
    
    print(f"\n📊 Verification Results:")
    print(f"  ✅ Matches: {matches}")
    print(f"  ❌ Mismatches: {mismatches}")
    print(f"  ⚠️ Decode failures: {decode_failures}")
    
    if matches > 0 and mismatches == 0:
        print(f"\n✅ VERIFICATION PASSED: Frames contain identical content")
        print(f"✅ Optimized QR settings produce identical results")
    elif mismatches > 0:
        print(f"\n❌ VERIFICATION FAILED: Content differences detected")
    else:
        print(f"\n⚠️ VERIFICATION INCONCLUSIVE: All decodes failed")

if __name__ == "__main__":
    verify_videos() 