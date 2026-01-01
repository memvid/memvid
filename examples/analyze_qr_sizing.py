#!/usr/bin/env python3
"""
Analyze QR code rendering dimensions and optimization opportunities
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for memvid imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.utils import encode_to_qr
from memvid.config import get_default_config, get_codec_parameters
import qrcode

def analyze_qr_dimensions():
    """Analyze QR code dimensions and rendering"""
    
    print("🔍 QR Code Dimension Analysis")
    print("=" * 50)
    
    config = get_default_config()["qr"]
    
    # QR Version to module count mapping
    # Formula: (Version - 1) * 4 + 21
    version = config["version"]
    modules_per_side = (version - 1) * 4 + 21
    
    print(f"📐 QR Code Structure:")
    print(f"  Version: {version}")
    print(f"  Modules per side: {modules_per_side} x {modules_per_side}")
    print(f"  Total modules: {modules_per_side * modules_per_side:,}")
    print(f"  Box size (pixels per module): {config['box_size']}")
    print(f"  Border: {config['border']} modules")
    
    # Current rendering dimensions
    effective_modules = modules_per_side + (2 * config['border'])
    current_pixel_size = effective_modules * config['box_size']
    
    print(f"\n📏 Current Rendering:")
    print(f"  Effective size with border: {effective_modules} x {effective_modules} modules")
    print(f"  Rendered pixel size: {current_pixel_size} x {current_pixel_size} pixels")
    print(f"  Pixels per module: {config['box_size']} x {config['box_size']}")
    
    # Video frame dimensions for different codecs
    print(f"\n🎬 Video Frame Dimensions:")
    codecs = ['mp4v', 'h265', 'h264', 'av1']
    
    for codec in codecs:
        try:
            params = get_codec_parameters(codec)
            frame_width = params['frame_width']
            frame_height = params['frame_height']
            
            # Calculate utilization
            width_utilization = current_pixel_size / frame_width
            height_utilization = current_pixel_size / frame_height
            area_utilization = (current_pixel_size * current_pixel_size) / (frame_width * frame_height)
            
            print(f"  {codec.upper()}:")
            print(f"    Frame: {frame_width} x {frame_height} pixels")
            print(f"    QR utilization: {width_utilization:.1%} x {height_utilization:.1%}")
            print(f"    Area utilization: {area_utilization:.1%}")
            
            # Calculate optimal box size for this frame
            max_box_size_width = frame_width // effective_modules
            max_box_size_height = frame_height // effective_modules
            optimal_box_size = min(max_box_size_width, max_box_size_height)
            
            print(f"    Max possible box size: {optimal_box_size}")
            if optimal_box_size >= 1:
                optimal_pixel_size = effective_modules * optimal_box_size
                optimal_utilization = (optimal_pixel_size * optimal_pixel_size) / (frame_width * frame_height)
                print(f"    Optimal QR size: {optimal_pixel_size} x {optimal_pixel_size}")
                print(f"    Optimal utilization: {optimal_utilization:.1%}")
            print()
            
        except Exception as e:
            print(f"    {codec.upper()}: Error - {e}")

def test_minimal_qr_rendering():
    """Test QR rendering at different box sizes"""
    
    print("🧪 Testing Minimal QR Rendering")
    print("=" * 50)
    
    # Test data
    test_text = "Hello World! This is a test of QR code rendering at different sizes."
    
    # Test different box sizes
    box_sizes = [1, 2, 3, 4, 5, 10]
    
    for box_size in box_sizes:
        try:
            # Create QR with custom box size
            qr = qrcode.QRCode(
                version=35,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=box_size,
                border=3,
            )
            qr.add_data(test_text)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            width, height = img.size
            
            print(f"Box size {box_size}: {width} x {height} pixels")
            
            # Test if it's readable
            import cv2
            img_array = np.array(img.convert('RGB'))
            cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            detector = cv2.QRCodeDetector()
            data, bbox, straight_qrcode = detector.detectAndDecode(cv_img)
            
            readable = "✅" if data else "❌"
            print(f"  Readability: {readable}")
            
            if data and len(data) != len(test_text):
                print(f"  ⚠️  Data mismatch: {len(data)} vs {len(test_text)} chars")
                
        except Exception as e:
            print(f"Box size {box_size}: ❌ Failed - {e}")
    
    print()

def calculate_theoretical_density():
    """Calculate theoretical data density limits"""
    
    print("📊 Theoretical Data Density Analysis")
    print("=" * 50)
    
    # QR capacity analysis for different versions
    qr_versions = [20, 25, 30, 35, 40]
    
    print("QR Version Analysis:")
    for version in qr_versions:
        modules = (version - 1) * 4 + 21
        
        # Theoretical capacities (Medium error correction)
        capacities_m = {
            20: 293, 25: 415, 30: 511, 35: 666, 40: 985
        }
        
        capacity = capacities_m.get(version, "Unknown")
        
        # With minimal rendering (box_size=1, border=0)
        min_pixels = modules * modules
        
        # With current config (box_size=5, border=3)
        current_modules = modules + 6  # 3 border on each side
        current_pixels = current_modules * current_modules * 25  # 5x5 box size
        
        print(f"  Version {version}:")
        print(f"    Modules: {modules} x {modules}")
        print(f"    Capacity: ~{capacity} chars")
        print(f"    Min pixels (1x1): {min_pixels}")
        print(f"    Current pixels (5x5+border): {current_pixels}")
        print(f"    Efficiency ratio: {current_pixels/min_pixels:.1f}x larger")
        
        if isinstance(capacity, int):
            density_min = capacity / min_pixels
            density_current = capacity / current_pixels
            print(f"    Density (min): {density_min:.3f} chars/pixel")
            print(f"    Density (current): {density_current:.3f} chars/pixel")
        print()

def optimization_recommendations():
    """Provide optimization recommendations"""
    
    print("🚀 Optimization Recommendations")
    print("=" * 50)
    
    # Current configuration analysis
    config = get_default_config()["qr"]
    version = config["version"]
    modules = (version - 1) * 4 + 21
    
    print("Current Configuration Issues:")
    print(f"  📐 QR modules: {modules} x {modules}")
    print(f"  📦 Box size: {config['box_size']} (5x5 pixels per module)")
    print(f"  🖼️  Border: {config['border']} modules")
    
    effective_size = (modules + 2 * config['border']) * config['box_size']
    print(f"  📏 Effective QR size: {effective_size} x {effective_size} pixels")
    
    # Frame size analysis
    h265_params = get_codec_parameters('h265')
    frame_size = h265_params['frame_width']
    utilization = (effective_size * effective_size) / (frame_size * frame_size)
    
    print(f"  🎬 Frame size: {frame_size} x {frame_size}")
    print(f"  📊 Current utilization: {utilization:.1%}")
    
    print("\nOptimization Opportunities:")
    
    # 1. Reduce box size
    optimal_box_size = frame_size // (modules + 2 * config['border'])
    if optimal_box_size < config['box_size']:
        print(f"  1️⃣  Reduce box size to {optimal_box_size} (vs current {config['box_size']})")
        new_size = (modules + 2 * config['border']) * optimal_box_size
        new_utilization = (new_size * new_size) / (frame_size * frame_size)
        print(f"     -> QR size: {new_size} x {new_size}")
        print(f"     -> Utilization: {new_utilization:.1%}")
    
    # 2. Reduce border
    print(f"  2️⃣  Reduce border from {config['border']} to 1 module")
    border_1_size = (modules + 2) * config['box_size']
    border_1_util = (border_1_size * border_1_size) / (frame_size * frame_size)
    print(f"     -> QR size: {border_1_size} x {border_1_size}")
    print(f"     -> Utilization: {border_1_util:.1%}")
    
    # 3. Combine optimizations
    print(f"  3️⃣  Combined: box_size=1, border=1")
    combined_size = modules + 2
    if combined_size <= frame_size:
        combined_util = (combined_size * combined_size) / (frame_size * frame_size)
        print(f"     -> QR size: {combined_size} x {combined_size}")
        print(f"     -> Utilization: {combined_util:.1%}")
        space_saved = effective_size * effective_size - combined_size * combined_size
        print(f"     -> Space saved: {space_saved:,} pixels ({space_saved/(frame_size*frame_size):.1%} of frame)")
    else:
        print(f"     -> ❌ Too large: {combined_size} > {frame_size}")
    
    # 4. Larger frame recommendation
    print(f"  4️⃣  Use larger frames for more data density")
    av1_params = get_codec_parameters('av1')
    av1_frame_size = av1_params['frame_width']
    print(f"     -> AV1 codec: {av1_frame_size} x {av1_frame_size} frames")
    
    # Calculate how much more data we could fit
    av1_utilization = (effective_size * effective_size) / (av1_frame_size * av1_frame_size)
    print(f"     -> Current QR utilization in AV1: {av1_utilization:.1%}")
    
    # Could fit multiple QRs or larger QR
    qrs_that_fit = (av1_frame_size // effective_size) ** 2
    if qrs_that_fit > 1:
        print(f"     -> Could fit {qrs_that_fit} QR codes per frame")
        print(f"     -> Data capacity increase: {qrs_that_fit}x")

if __name__ == "__main__":
    analyze_qr_dimensions()
    test_minimal_qr_rendering()
    calculate_theoretical_density()
    optimization_recommendations() 