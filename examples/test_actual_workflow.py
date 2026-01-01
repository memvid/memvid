#!/usr/bin/env python3
"""
Test actual Memvid workflow with optimized QR settings
"""

import tempfile
import os
import sys
from pathlib import Path

# Add memvid to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.encoder import MemvidEncoder
from memvid.retriever import MemvidRetriever

def test_workflow():
    """Test the actual encode/decode workflow"""
    print("🔍 Testing Actual Memvid Workflow")
    print("=" * 40)
    
    test_texts = [
        "This is a test of QR decode accuracy with optimized settings.",
        "Machine learning enables computers to learn without explicit programming.",
        "The quick brown fox jumps over the lazy dog with special chars: @#$%^&*()",
        "JSON example: {\"name\": \"test\", \"value\": 123, \"array\": [1, 2, 3]}",
        "Unicode test: αβγδε 中文测试 🌟🔍📊 Ñiño résumé café naïve"
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, 'test.mp4')
        index_path = os.path.join(temp_dir, 'test_index.json')
        
        print(f"Creating memory with {len(test_texts)} chunks...")
        
        # Create memory
        encoder = MemvidEncoder()
        for text in test_texts:
            encoder.add_text(text)
        
        result = encoder.build_video(video_path, index_path)
        print(f"✅ Video created: {os.path.getsize(video_path)/1024:.1f} KB")
        
        # Test retrieval
        retriever = MemvidRetriever(video_path, index_path)
        
        print(f"\nTesting retrieval...")
        successful_retrievals = 0
        
        for i, original_text in enumerate(test_texts):
            # Search using first part of text
            search_query = original_text.split('.')[0] if '.' in original_text else original_text[:30]
            results = retriever.search(search_query, top_k=3)
            
            found = False
            if results:
                for chunk in results:
                    # Check if we found the original text (allowing for chunking differences)
                    if original_text[:50] in chunk or chunk[:50] in original_text:
                        found = True
                        print(f"  ✅ Chunk {i}: Found")
                        break
            
            if not found:
                print(f"  ❌ Chunk {i}: Not found")
                if results:
                    print(f"    Top result: {repr(results[0][:50])}")
            else:
                successful_retrievals += 1
        
        success_rate = successful_retrievals / len(test_texts) * 100
        print(f"\n📊 Results:")
        print(f"  Success Rate: {success_rate:.1f}% ({successful_retrievals}/{len(test_texts)})")
        
        if success_rate >= 80:
            print("  ✅ Optimized QR settings work well!")
        elif success_rate >= 60:
            print("  🟡 Optimized QR settings work reasonably")
        else:
            print("  ⚠️  Potential issues with optimized QR settings")
        
        return success_rate

if __name__ == "__main__":
    test_workflow() 