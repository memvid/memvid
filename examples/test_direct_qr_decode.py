#!/usr/bin/env python3
"""
Direct QR Decode Test

Tests QR code decoding accuracy at the frame level using Memvid's actual QR creation and decoding functions.
Compares original vs optimized settings for direct QR decode reliability.
"""

import tempfile
import os
import sys
from pathlib import Path
import time
import cv2

# Add memvid to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.utils import encode_to_qr, qr_to_frame, decode_qr
from memvid import config

def test_qr_encode_decode(test_chunks, box_size, border, test_name):
    """Test QR encoding and decoding with specific settings"""
    
    # Temporarily modify config
    original_box_size = config.QR_BOX_SIZE
    original_border = config.QR_BORDER
    
    try:
        # Set test configuration
        config.QR_BOX_SIZE = box_size
        config.QR_BORDER = border
        
        successful_decodes = 0
        total_chunks = len(test_chunks)
        decode_errors = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, chunk in enumerate(test_chunks):
                try:
                    # Create QR code
                    qr_img = encode_to_qr(chunk)
                    
                    # Convert to video frame format
                    frame = qr_to_frame(qr_img, (256, 256))
                    
                    # Save frame as image
                    frame_path = os.path.join(temp_dir, f"frame_{i}.png")
                    cv2.imwrite(frame_path, frame)
                    
                    # Try to decode the frame
                    try:
                        decoded_data = decode_qr(frame)
                        
                        if decoded_data == chunk:
                            successful_decodes += 1
                        else:
                            decode_errors.append(f"Chunk {i}: Data mismatch (expected {len(chunk)} chars, got {len(decoded_data)} chars)")
                            
                    except Exception as decode_error:
                        decode_errors.append(f"Chunk {i}: Decode failed - {str(decode_error)}")
                        
                except Exception as e:
                    decode_errors.append(f"Chunk {i}: QR creation failed - {str(e)}")
        
        success_rate = successful_decodes / total_chunks if total_chunks > 0 else 0
        
        return {
            "success_rate": success_rate,
            "successful_decodes": successful_decodes,
            "total_chunks": total_chunks,
            "errors": decode_errors[:5],  # First 5 errors
            "qr_settings": f"box_size={box_size}, border={border}"
        }
        
    finally:
        # Restore original config
        config.QR_BOX_SIZE = original_box_size
        config.QR_BORDER = original_border

def generate_test_data():
    """Generate test data of various types and lengths"""
    test_data = [
        # Short text
        "Hello world! This is a test of QR encoding.",
        
        # Medium text with special characters
        "The quick brown fox jumps over the lazy dog. Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
        
        # Technical content
        "Machine learning algorithms use statistical techniques to enable computer systems to improve performance on a specific task.",
        
        # JSON-like content
        '{"name": "test", "value": 123, "array": [1, 2, 3], "nested": {"key": "value"}}',
        
        # Longer text (near capacity)
        "This is a longer test string designed to test QR code capacity and decode reliability. " * 8,
        
        # Unicode content
        "Unicode test: αβγδε 中文测试 🌟🔍📊 Ñiño résumé café naïve",
        
        # Code-like content
        "def encode_text(data):\n    return qr.make(data)\n\nresult = encode_text('test')",
        
        # Mixed content with numbers
        "Data analysis: 42% improvement, $1,234.56 revenue, dates: 2024-01-15 to 2024-12-31",
        
        # Repeated patterns
        "ABCD" * 50,
        
        # Random-ish content
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt."
    ]
    
    return test_data

def main():
    """Run direct QR decode accuracy test"""
    print("🔍 Direct QR Decode Accuracy Test")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {"box_size": 5, "border": 3, "name": "Original"},
        {"box_size": 3, "border": 1, "name": "Optimized"},
        {"box_size": 2, "border": 1, "name": "Ultra-Dense"},
        {"box_size": 1, "border": 1, "name": "Minimal"},
    ]
    
    test_data = generate_test_data()
    
    print(f"Testing {len(test_data)} chunks with various QR settings")
    print()
    
    results = {}
    
    for config_info in test_configs:
        config_name = config_info["name"]
        box_size = config_info["box_size"]
        border = config_info["border"]
        
        print(f"🔧 Testing {config_name} (box_size={box_size}, border={border})")
        
        try:
            result = test_qr_encode_decode(test_data, box_size, border, config_name)
            results[config_name] = result
            
            print(f"  ✅ Success Rate: {result['success_rate']*100:.1f}% ({result['successful_decodes']}/{result['total_chunks']})")
            
            if result['errors']:
                print(f"  ⚠️  Decode Issues:")
                for error in result['errors']:
                    print(f"    - {error}")
            
        except Exception as e:
            print(f"  ❌ Test Failed: {str(e)}")
            results[config_name] = {"success_rate": 0, "error": str(e)}
        
        print()
    
    # Summary
    print("=" * 60)
    print("📈 DECODE ACCURACY SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Configuration':<15} | {'Success Rate':<12} | {'Status':<20}")
    print("-" * 55)
    
    for config_name in ["Original", "Optimized", "Ultra-Dense", "Minimal"]:
        if config_name in results:
            result = results[config_name]
            if isinstance(result, dict) and "success_rate" in result:
                rate = f"{result['success_rate']*100:.1f}%"
                if result['success_rate'] >= 0.95:
                    status = "✅ Excellent"
                elif result['success_rate'] >= 0.8:
                    status = "🟡 Good"
                elif result['success_rate'] >= 0.5:
                    status = "🟠 Fair"
                else:
                    status = "❌ Poor"
            else:
                rate = "FAIL"
                status = "❌ Failed"
        else:
            rate = "N/A"
            status = "N/A"
        
        print(f"{config_name:<15} | {rate:<12} | {status:<20}")
    
    # Analysis
    print(f"\n💡 ANALYSIS:")
    print("-" * 15)
    
    if "Original" in results and "Optimized" in results:
        orig_rate = results["Original"].get("success_rate", 0) * 100
        opt_rate = results["Optimized"].get("success_rate", 0) * 100
        
        print(f"Original Settings:  {orig_rate:.1f}% success rate")
        print(f"Optimized Settings: {opt_rate:.1f}% success rate")
        
        diff = opt_rate - orig_rate
        if abs(diff) < 5:
            print("✅ No significant difference in decode accuracy")
            print("✅ Optimized settings are safe to use")
        elif diff < 0:
            print(f"⚠️  Optimized settings show {abs(diff):.1f}% lower accuracy")
            print("🤔 Consider keeping original settings for reliability")
        else:
            print(f"🎉 Optimized settings show {diff:.1f}% better accuracy")
            print("✅ Optimized settings are recommended")
    
    return results

if __name__ == "__main__":
    results = main() 