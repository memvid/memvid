#!/usr/bin/env python3
"""
QR Optimization Verification

This script demonstrates the improvements from the QR rendering optimizations:
- Reduced box size from 5 to 3 pixels per module
- Reduced border from 3 to 1 module
- Shows before/after size calculations
"""

import qrcode
from PIL import Image
import os
import sys

def create_test_qr(box_size, border, label):
    """Create a test QR code with specified parameters"""
    qr = qrcode.QRCode(
        version=35,  # Same as Memvid config
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    
    qr.add_data("This is a test QR code for Memvid optimization verification. " * 10)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    filename = f"test_qr_{label}.png"
    img.save(filename)
    
    return img.size, filename

def main():
    print("🔍 QR Code Optimization Verification")
    print("=" * 50)
    
    # Original settings
    print("\n📊 Original Settings (box_size=5, border=3):")
    old_size, old_file = create_test_qr(5, 3, "original")
    print(f"  Size: {old_size[0]}x{old_size[1]} pixels")
    print(f"  File: {old_file}")
    
    # Optimized settings
    print("\n✨ Optimized Settings (box_size=3, border=1):")
    new_size, new_file = create_test_qr(3, 1, "optimized")
    print(f"  Size: {new_size[0]}x{new_size[1]} pixels")
    print(f"  File: {new_file}")
    
    # Calculate improvements
    old_pixels = old_size[0] * old_size[1]
    new_pixels = new_size[0] * new_size[1]
    
    print("\n📈 Optimization Results:")
    print(f"  Original: {old_pixels:,} pixels")
    print(f"  Optimized: {new_pixels:,} pixels")
    print(f"  Reduction: {old_pixels - new_pixels:,} pixels ({(1 - new_pixels/old_pixels)*100:.1f}%)")
    
    # Frame utilization (assuming 256x256 video frames)
    frame_pixels = 256 * 256
    old_utilization = old_pixels / frame_pixels * 100
    new_utilization = new_pixels / frame_pixels * 100
    
    print("\n🎬 Video Frame Utilization (256x256 frames):")
    print(f"  Original: {old_utilization:.1f}% ({old_size[0]}x{old_size[1]} in 256x256)")
    print(f"  Optimized: {new_utilization:.1f}% ({new_size[0]}x{new_size[1]} in 256x256)")
    
    if old_utilization > 100:
        print(f"  ⚠️  Original QR is {old_utilization/100:.1f}x larger than video frame!")
    if new_utilization > 100:
        print(f"  ⚠️  Optimized QR is {new_utilization/100:.1f}x larger than video frame!")
    
    print("\n💡 Theoretical Capacity:")
    print("  Version 35 QR: 157x157 modules")
    print("  With border: 159x159 effective area")
    print("  Minimum size: 159x159 pixels (1 pixel per module)")
    print(f"  Current optimized: {new_size[0]}x{new_size[1]} pixels ({new_size[0]/159:.1f}x minimum)")
    
    # Cleanup
    try:
        os.remove(old_file)
        os.remove(new_file)
        print(f"\n🧹 Cleaned up test files")
    except:
        pass

if __name__ == "__main__":
    main() 