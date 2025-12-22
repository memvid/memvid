#!/usr/bin/env python3
"""
QR Decode Accuracy Test

Tests the decode reliability of QR codes with different box_size and border settings.
Compares original settings (box_size=5, border=3) vs optimized (box_size=3, border=1).
Tests under various conditions including compression, scaling, and noise.
"""

import qrcode
from PIL import Image, ImageFilter
import pyzbar.pyzbar as pyzbar
import io
import random
import string
import numpy as np
from pathlib import Path

def generate_test_data(length=500):
    """Generate random test data of specified length"""
    return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=length))

def create_qr_with_settings(data, box_size, border, label):
    """Create QR code with specific settings"""
    qr = qrcode.QRCode(
        version=35,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    return img

def test_decode_accuracy(img, original_data, test_name):
    """Test if QR code can be decoded accurately"""
    try:
        # Convert PIL image to format pyzbar can read
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Decode QR code
        decoded_objects = pyzbar.decode(Image.open(img_bytes))
        
        if not decoded_objects:
            return False, "No QR code detected"
        
        decoded_data = decoded_objects[0].data.decode('utf-8')
        
        if decoded_data == original_data:
            return True, "Perfect match"
        else:
            return False, f"Data mismatch (got {len(decoded_data)} chars vs {len(original_data)} expected)"
    
    except Exception as e:
        return False, f"Decode error: {str(e)}"

def apply_compression(img, quality):
    """Apply JPEG compression to simulate video compression"""
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=quality)
    img_bytes.seek(0)
    return Image.open(img_bytes)

def apply_scaling(img, scale_factor):
    """Scale image up and down to simulate video processing"""
    original_size = img.size
    # Scale down then back up
    small_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
    img_small = img.resize(small_size, Image.LANCZOS)
    img_restored = img_small.resize(original_size, Image.LANCZOS)
    return img_restored

def apply_blur(img, radius):
    """Apply gaussian blur to simulate focus issues"""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def add_noise(img, noise_level):
    """Add random noise to image"""
    img_array = np.array(img)
    noise = np.random.randint(-noise_level, noise_level + 1, img_array.shape, dtype=np.int16)
    noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def run_comprehensive_test():
    """Run comprehensive decode accuracy tests"""
    print("🔍 QR Code Decode Accuracy Test")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {"box_size": 5, "border": 3, "name": "Original"},
        {"box_size": 3, "border": 1, "name": "Optimized"},
    ]
    
    # Generate test data of various lengths
    test_datasets = [
        ("Short", generate_test_data(100)),
        ("Medium", generate_test_data(300)),
        ("Long", generate_test_data(500)),
        ("Max Capacity", generate_test_data(666)),  # Near Version 35 capacity
    ]
    
    # Test conditions
    test_conditions = [
        ("Perfect", lambda img: img),
        ("JPEG 95%", lambda img: apply_compression(img, 95)),
        ("JPEG 80%", lambda img: apply_compression(img, 80)),
        ("JPEG 60%", lambda img: apply_compression(img, 60)),
        ("Scale 50%", lambda img: apply_scaling(img, 0.5)),
        ("Scale 25%", lambda img: apply_scaling(img, 0.25)),
        ("Blur 0.5", lambda img: apply_blur(img, 0.5)),
        ("Blur 1.0", lambda img: apply_blur(img, 1.0)),
        ("Noise 5", lambda img: add_noise(img, 5)),
        ("Noise 10", lambda img: add_noise(img, 10)),
    ]
    
    results = {}
    
    for config in configs:
        config_name = config["name"]
        results[config_name] = {}
        
        print(f"\n📊 Testing {config_name} Settings (box_size={config['box_size']}, border={config['border']})")
        print("-" * 60)
        
        for data_name, test_data in test_datasets:
            print(f"\n📝 Data Length: {data_name} ({len(test_data)} chars)")
            
            # Create QR code
            try:
                qr_img = create_qr_with_settings(test_data, config["box_size"], config["border"], config_name)
                print(f"  QR Size: {qr_img.size[0]}×{qr_img.size[1]} pixels")
                
                results[config_name][data_name] = {}
                
                # Test under various conditions
                for condition_name, condition_func in test_conditions:
                    try:
                        processed_img = condition_func(qr_img.copy())
                        success, message = test_decode_accuracy(processed_img, test_data, condition_name)
                        
                        results[config_name][data_name][condition_name] = success
                        
                        status = "✅" if success else "❌"
                        print(f"    {status} {condition_name:12}: {message}")
                        
                    except Exception as e:
                        results[config_name][data_name][condition_name] = False
                        print(f"    ❌ {condition_name:12}: Exception - {str(e)}")
                        
            except Exception as e:
                print(f"  ❌ Failed to create QR: {str(e)}")
                results[config_name][data_name] = {}
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("📈 DECODE ACCURACY SUMMARY")
    print("=" * 60)
    
    for data_name, _ in test_datasets:
        print(f"\n📝 {data_name} Data:")
        print("  Condition        | Original | Optimized | Difference")
        print("  -----------------|----------|-----------|----------")
        
        for condition_name, _ in test_conditions:
            try:
                orig_success = results["Original"].get(data_name, {}).get(condition_name, False)
                opt_success = results["Optimized"].get(data_name, {}).get(condition_name, False)
                
                orig_symbol = "✅" if orig_success else "❌"
                opt_symbol = "✅" if opt_success else "❌"
                
                if orig_success == opt_success:
                    diff = "Same"
                elif orig_success and not opt_success:
                    diff = "⚠️ Worse"
                else:  # not orig_success and opt_success
                    diff = "Better"
                
                print(f"  {condition_name:15} | {orig_symbol:>8} | {opt_symbol:>9} | {diff}")
                
            except KeyError:
                print(f"  {condition_name:15} | {'N/A':>8} | {'N/A':>9} | N/A")
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print("-" * 30)
    
    for config_name in ["Original", "Optimized"]:
        total_tests = 0
        successful_tests = 0
        
        for data_name in results[config_name]:
            for condition_name in results[config_name][data_name]:
                total_tests += 1
                if results[config_name][data_name][condition_name]:
                    successful_tests += 1
        
        if total_tests > 0:
            success_rate = (successful_tests / total_tests) * 100
            print(f"{config_name:10}: {successful_tests:3}/{total_tests:3} ({success_rate:5.1f}%)")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_test() 