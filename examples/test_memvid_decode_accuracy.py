#!/usr/bin/env python3
"""
Memvid QR Decode Accuracy Test

Tests decode reliability using Memvid's actual QR encoding and decoding pipeline.
Compares original settings (box_size=5, border=3) vs optimized (box_size=3, border=1).
"""

import tempfile
import os
import sys
from pathlib import Path
import time
import json

# Add memvid to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.encoder import MemvidEncoder
from memvid.retriever import MemvidRetriever
from memvid import config

def test_encode_decode_cycle(test_data, box_size, border, test_name):
    """Test full encode->decode cycle with specific QR settings"""
    
    # Temporarily modify config
    original_box_size = config.QR_BOX_SIZE
    original_border = config.QR_BORDER
    
    try:
        # Set test configuration
        config.QR_BOX_SIZE = box_size
        config.QR_BORDER = border
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, f"test_{test_name}.mp4")
            index_path = os.path.join(temp_dir, f"test_{test_name}_index.json")
            
            # Encode
            start_time = time.time()
            encoder = MemvidEncoder()
            
            for i, chunk in enumerate(test_data):
                encoder.add_text(chunk, chunk_size=512, overlap=32)
            
            encoder.build_video(video_path, index_path)
            encode_time = time.time() - start_time
            
            # Get video stats
            video_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            
            # Decode
            start_time = time.time()
            retriever = MemvidRetriever(video_path, index_path)
            
            # Test retrieving each chunk
            successful_retrievals = 0
            total_chunks = len(test_data)
            decode_errors = []
            
            for i, original_chunk in enumerate(test_data):
                try:
                    # Search for chunk content
                    search_query = original_chunk[:50]  # Use first 50 chars as search
                    results = retriever.search(search_query, top_k=5)
                    
                    # Check if we can find the original chunk
                    found = False
                    for retrieved_chunk, score in results:
                        if original_chunk.strip() in retrieved_chunk or retrieved_chunk.strip() in original_chunk:
                            found = True
                            break
                    
                    if found:
                        successful_retrievals += 1
                    else:
                        decode_errors.append(f"Chunk {i}: Not found in search results")
                        
                except Exception as e:
                    decode_errors.append(f"Chunk {i}: {str(e)}")
            
            decode_time = time.time() - start_time
            
            return {
                "success": successful_retrievals == total_chunks,
                "success_rate": successful_retrievals / total_chunks if total_chunks > 0 else 0,
                "successful_retrievals": successful_retrievals,
                "total_chunks": total_chunks,
                "encode_time": encode_time,
                "decode_time": decode_time,
                "video_size": video_size,
                "errors": decode_errors[:5],  # First 5 errors
                "qr_size": f"{config.QR_BOX_SIZE}×{config.QR_BORDER}"
            }
            
    finally:
        # Restore original config
        config.QR_BOX_SIZE = original_box_size
        config.QR_BORDER = original_border

def generate_test_chunks(num_chunks=50, chunk_size=400):
    """Generate test chunks of various content types"""
    import random
    import string
    
    chunks = []
    
    # Different types of content to test
    content_types = [
        # Technical text
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        
        # Scientific content
        "The theory of relativity fundamentally changed our understanding of space, time, and gravity, showing that massive objects warp spacetime.",
        
        # Mixed content with punctuation
        "Data science combines statistics, mathematics, and computer science to extract insights from structured and unstructured data sets.",
        
        # Content with special characters
        "JSON format uses key-value pairs: {\"name\": \"value\", \"number\": 42, \"array\": [1, 2, 3]}",
        
        # Repeated patterns
        "AAAAAABBBBBBCCCCCCDDDDDDEEEEEEFFFFFFGGGGGG" * 10,
    ]
    
    for i in range(num_chunks):
        if i < len(content_types):
            base_content = content_types[i]
        else:
            # Generate random content
            base_content = ''.join(random.choices(
                string.ascii_letters + string.digits + string.punctuation + ' ', 
                k=200
            ))
        
        # Pad or truncate to desired size
        if len(base_content) < chunk_size:
            base_content = base_content * (chunk_size // len(base_content) + 1)
        
        chunks.append(base_content[:chunk_size])
    
    return chunks

def main():
    """Run comprehensive decode accuracy test"""
    print("🔍 Memvid QR Decode Accuracy Test")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {"box_size": 5, "border": 3, "name": "Original"},
        {"box_size": 3, "border": 1, "name": "Optimized"},
        {"box_size": 2, "border": 1, "name": "Ultra-Dense"},
        {"box_size": 1, "border": 1, "name": "Minimal"},
    ]
    
    # Test datasets
    test_datasets = [
        {"name": "Small", "chunks": generate_test_chunks(10, 300)},
        {"name": "Medium", "chunks": generate_test_chunks(25, 400)},
        {"name": "Large", "chunks": generate_test_chunks(50, 500)},
    ]
    
    results = {}
    
    for dataset in test_datasets:
        dataset_name = dataset["name"]
        test_chunks = dataset["chunks"]
        
        print(f"\n📊 Testing {dataset_name} Dataset ({len(test_chunks)} chunks)")
        print("-" * 50)
        
        results[dataset_name] = {}
        
        for config_info in test_configs:
            config_name = config_info["name"]
            box_size = config_info["box_size"]
            border = config_info["border"]
            
            print(f"\n🔧 {config_name} (box_size={box_size}, border={border})")
            
            try:
                result = test_encode_decode_cycle(
                    test_chunks, box_size, border, f"{dataset_name}_{config_name}"
                )
                
                results[dataset_name][config_name] = result
                
                print(f"  ✅ Success Rate: {result['success_rate']*100:.1f}% ({result['successful_retrievals']}/{result['total_chunks']})")
                print(f"  ⏱️  Encode Time: {result['encode_time']:.2f}s")
                print(f"  ⏱️  Decode Time: {result['decode_time']:.2f}s") 
                print(f"  💾 Video Size: {result['video_size']/1024:.1f} KB")
                
                if result['errors']:
                    print(f"  ⚠️  Sample Errors:")
                    for error in result['errors']:
                        print(f"    - {error}")
                
            except Exception as e:
                print(f"  ❌ Test Failed: {str(e)}")
                results[dataset_name][config_name] = {"success": False, "error": str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 DECODE ACCURACY SUMMARY")  
    print("=" * 60)
    
    print(f"\n{'Dataset':<10} | {'Original':<10} | {'Optimized':<11} | {'Ultra-Dense':<12} | {'Minimal':<10}")
    print("-" * 70)
    
    for dataset_name in results:
        row = f"{dataset_name:<10} |"
        
        for config_name in ["Original", "Optimized", "Ultra-Dense", "Minimal"]:
            if config_name in results[dataset_name]:
                result = results[dataset_name][config_name]
                if isinstance(result, dict) and "success_rate" in result:
                    rate = f"{result['success_rate']*100:.1f}%"
                else:
                    rate = "FAIL"
            else:
                rate = "N/A"
            
            if config_name == "Original":
                row += f" {rate:<10} |"
            elif config_name == "Optimized":
                row += f" {rate:<11} |"
            elif config_name == "Ultra-Dense":
                row += f" {rate:<12} |"
            else:  # Minimal
                row += f" {rate:<10}"
        
        print(row)
    
    # Detailed analysis
    print(f"\n📊 DETAILED ANALYSIS:")
    print("-" * 30)
    
    for dataset_name in results:
        print(f"\n{dataset_name} Dataset:")
        
        for config_name in ["Original", "Optimized"]:
            if config_name in results[dataset_name]:
                result = results[dataset_name][config_name]
                if isinstance(result, dict) and "success_rate" in result:
                    print(f"  {config_name:10}: {result['success_rate']*100:5.1f}% success, {result['video_size']/1024:6.1f} KB")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    original_avg = []
    optimized_avg = []
    
    for dataset_name in results:
        if "Original" in results[dataset_name] and isinstance(results[dataset_name]["Original"], dict):
            original_avg.append(results[dataset_name]["Original"].get("success_rate", 0))
        if "Optimized" in results[dataset_name] and isinstance(results[dataset_name]["Optimized"], dict):
            optimized_avg.append(results[dataset_name]["Optimized"].get("success_rate", 0))
    
    if original_avg and optimized_avg:
        orig_mean = sum(original_avg) / len(original_avg) * 100
        opt_mean = sum(optimized_avg) / len(optimized_avg) * 100
        
        print(f"  Original Average: {orig_mean:.1f}% success rate")
        print(f"  Optimized Average: {opt_mean:.1f}% success rate")
        
        if abs(orig_mean - opt_mean) < 5:
            print("  ✅ Optimized settings are safe to use (similar accuracy)")
        elif opt_mean < orig_mean:
            print(f"  ⚠️  Optimized settings show {orig_mean - opt_mean:.1f}% lower accuracy")
        else:
            print(f"  🎉 Optimized settings show {opt_mean - orig_mean:.1f}% better accuracy")

if __name__ == "__main__":
    main() 