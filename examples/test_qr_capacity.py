#!/usr/bin/env python3
"""
Test QR capacity and chunking behavior
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for memvid imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.utils import encode_to_qr, decode_qr, chunk_text
from memvid.config import get_default_config
import qrcode

def test_qr_capacity():
    """Test QR code data capacity limits"""
    
    print("🧪 Testing QR Code Capacity Limits")
    print("=" * 50)
    
    config = get_default_config()["qr"]
    
    print(f"📋 QR Configuration:")
    print(f"  Version: {config['version']} (1-40 scale)")
    print(f"  Error Correction: {config['error_correction']} (L/M/Q/H)")
    print(f"  Box Size: {config['box_size']}")
    print(f"  Border: {config['border']}")
    
    # QR version 35 theoretical capacity
    # See: https://www.qrcode.com/en/about/version.html
    version_capacities = {
        1: {"L": 25, "M": 20, "Q": 16, "H": 10},
        10: {"L": 174, "M": 135, "Q": 87, "H": 62},
        20: {"L": 370, "M": 293, "Q": 207, "H": 154},
        30: {"L": 666, "M": 511, "Q": 367, "H": 288},
        35: {"L": 858, "M": 666, "Q": 482, "H": 367},
        40: {"L": 1273, "M": 985, "Q": 711, "H": 535}
    }
    
    version = config['version']
    error_level = config['error_correction']
    theoretical_capacity = version_capacities.get(version, {}).get(error_level, "Unknown")
    
    print(f"  Theoretical capacity (alphanumeric): {theoretical_capacity} characters")
    
    # Test actual capacity with different data types
    print(f"\n🔍 Testing actual capacity:")
    
    # Test with simple text
    test_cases = [
        ("Simple text", "A" * 100),
        ("Lorem ipsum", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 10),
        ("JSON data", json.dumps({"text": "A" * 500, "metadata": {"source": "test"}})),
        ("Long text", "The quick brown fox jumps over the lazy dog. " * 20)
    ]
    
    for name, test_data in test_cases:
        try:
            # Try encoding
            qr_img = encode_to_qr(test_data)
            
            # Convert to numpy for decoding test
            import numpy as np
            import cv2
            from PIL import Image
            
            # Convert PIL to OpenCV format for decode test
            img_array = np.array(qr_img.convert('RGB'))
            cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Test decoding
            decoded = decode_qr(cv_img)
            success = decoded == test_data
            
            print(f"  ✅ {name}: {len(test_data)} chars -> {'✓' if success else '✗'}")
            if not success and decoded:
                print(f"     Decoded length: {len(decoded)} chars")
                
        except Exception as e:
            print(f"  ❌ {name}: {len(test_data)} chars -> Failed: {e}")
    
    # Test compression effectiveness
    print(f"\n📦 Testing compression:")
    
    # Create realistic text content
    realistic_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural 
    intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 
    "intelligent agents": any device that perceives its environment and takes actions that maximize its 
    chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is 
    often used to describe machines that mimic "cognitive" functions that humans associate with the 
    human mind, such as "learning" and "problem solving".
    """
    
    uncompressed_size = len(realistic_text)
    
    try:
        qr_img = encode_to_qr(realistic_text)
        print(f"  Original text: {uncompressed_size} characters")
        print(f"  ✅ Successfully encoded with compression")
        
        # Test decoding
        img_array = np.array(qr_img.convert('RGB'))
        cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        decoded = decode_qr(cv_img)
        
        if decoded == realistic_text:
            print(f"  ✅ Successfully decoded")
        else:
            print(f"  ❌ Decode mismatch")
            
    except Exception as e:
        print(f"  ❌ Failed: {e}")

def test_chunking_behavior():
    """Test document chunking behavior"""
    
    print(f"\n📝 Testing Document Chunking")
    print("=" * 50)
    
    # Test with different chunk sizes
    sample_text = """
    Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 
    'learn', that is, methods that leverage data to improve performance on some set of tasks. It is 
    seen as a part of artificial intelligence. Machine learning algorithms build a model based on 
    training data in order to make predictions or decisions without being explicitly programmed to 
    do so. Machine learning algorithms are used in a wide variety of applications, such as in 
    medicine, email filtering, speech recognition, and computer vision, where it is difficult or 
    unfeasible to develop conventional algorithms to perform the needed tasks.
    """ * 3  # Make it longer
    
    chunk_sizes = [256, 512, 1024]
    overlaps = [32, 50, 100]
    
    print(f"Sample text: {len(sample_text)} characters")
    print(f"Sample words: {len(sample_text.split())} words")
    print()
    
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            chunks = chunk_text(sample_text, chunk_size, overlap)
            
            avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
            min_chunk_size = min(len(chunk) for chunk in chunks)
            max_chunk_size = max(len(chunk) for chunk in chunks)
            
            print(f"Chunk size {chunk_size}, overlap {overlap}:")
            print(f"  Chunks created: {len(chunks)}")
            print(f"  Avg chunk size: {avg_chunk_size:.1f} chars")
            print(f"  Min/Max: {min_chunk_size}/{max_chunk_size} chars")
            print(f"  Efficiency: {(sum(len(chunk) for chunk in chunks) / len(sample_text)):.2f}x (due to overlap)")
            print()

def analyze_wikipedia_results():
    """Analyze actual results from Wikipedia testing"""
    
    print(f"📊 Wikipedia Test Results Analysis")
    print("=" * 50)
    
    # Data from our previous tests
    test_results = [
        {
            "name": "AI Articles (3)",
            "articles": 3,
            "total_chars": 10494,
            "chunks": 26,
            "chunk_size": 512,
            "overlap": 50
        },
        {
            "name": "Science Articles (10)",
            "articles": 10,
            "total_chars": 43540,
            "chunks": 105,
            "chunk_size": 512,
            "overlap": 50
        }
    ]
    
    for result in test_results:
        print(f"\n{result['name']}:")
        
        # Calculate metrics
        avg_chunk_size = result['total_chars'] / result['chunks']
        chars_per_article = result['total_chars'] / result['articles']
        chunks_per_article = result['chunks'] / result['articles']
        
        print(f"  📄 Articles: {result['articles']}")
        print(f"  📝 Total characters: {result['total_chars']:,}")
        print(f"  🧩 Total chunks: {result['chunks']}")
        print(f"  📏 Average chunk size: {avg_chunk_size:.1f} chars")
        print(f"  📑 Chars per article: {chars_per_article:.0f}")
        print(f"  🔢 Chunks per article: {chunks_per_article:.1f}")
        
        # Estimate words (rough approximation: 5 chars per word)
        avg_words_per_chunk = avg_chunk_size / 5
        print(f"  💬 Estimated words per chunk: {avg_words_per_chunk:.0f}")
        
        # Frame/time calculations
        fps = 30  # From H265 config
        chunks_per_second = fps
        duration_seconds = result['chunks'] / fps
        
        print(f"  🎬 Video duration: {duration_seconds:.1f} seconds")
        print(f"  ⚡ Chunks per second: {chunks_per_second}")

if __name__ == "__main__":
    test_qr_capacity()
    test_chunking_behavior()
    analyze_wikipedia_results() 