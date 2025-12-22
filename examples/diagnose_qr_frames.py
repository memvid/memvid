#!/usr/bin/env python3
"""
Diagnose QR frame issues
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add memvid to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.utils import decode_qr

def diagnose_frame(video_path, frame_idx=0, save_image=False):
    """Diagnose a specific frame from the video"""
    
    print(f"🔍 Diagnosing frame {frame_idx} from {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video")
        return
    
    # Get frame info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video info: {total_frames} frames, {width}x{height}")
    
    # Extract frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame {frame_idx}")
        return
    
    print(f"📷 Frame shape: {frame.shape}")
    print(f"📷 Frame dtype: {frame.dtype}")
    print(f"📷 Frame range: {frame.min()} - {frame.max()}")
    
    # Save frame for inspection
    if save_image:
        filename = f"frame_{frame_idx}_{video_path.replace('.mp4', '.png')}"
        cv2.imwrite(filename, frame)
        print(f"💾 Saved frame to {filename}")
    
    # Try QR decode
    try:
        data = decode_qr(frame)
        if data:
            print(f"✅ QR decoded successfully: {len(data)} chars")
            print(f"📝 Content preview: {repr(data[:100])}")
        else:
            print("❌ QR decode returned None")
            
            # Try some diagnostics
            print("🔧 Trying diagnostics...")
            
            # Check if it's mostly black/white (QR pattern)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            unique_values = np.unique(gray)
            print(f"   Gray values: {len(unique_values)} unique ({unique_values[:10]}...)")
            
            # Check for QR-like patterns
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            print(f"   Edge density: {edge_ratio:.3f}")
            
            # Try different thresholds
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            thresh_data = decode_qr(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))
            if thresh_data:
                print(f"✅ Threshold decode worked: {len(thresh_data)} chars")
            else:
                print("❌ Threshold decode also failed")
    
    except Exception as e:
        print(f"❌ QR decode exception: {e}")

def main():
    """Run diagnostics on both comparison videos"""
    
    videos = [
        "original_comparison_video.mp4",
        "optimized_comparison_video.mp4"
    ]
    
    print("🔍 QR Frame Diagnostics")
    print("=" * 50)
    
    for video in videos:
        print(f"\n📹 Analyzing {video}")
        print("-" * 30)
        
        # Check first few frames
        for frame_idx in [0, 1, 2]:
            diagnose_frame(video, frame_idx, save_image=(frame_idx == 0))
            print()

if __name__ == "__main__":
    main() 