#!/usr/bin/env python3
"""
Test retrieval functionality without requiring LLM API keys
"""

import sys
from pathlib import Path

# Add parent directory to path for memvid imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid import MemvidRetriever

def test_retrieval():
    """Test semantic search and retrieval"""
    
    # Find the most recent Wikipedia demo files
    demo_dir = Path("output/wikipedia_demo")
    if not demo_dir.exists():
        print("❌ No Wikipedia demo files found. Run the demo first:")
        print("   python examples/wikipedia_demo.py --num-articles 3 --no-chat")
        return False
    
    # Find the most recent video and index files
    video_files = list(demo_dir.glob("*.mp4"))
    if not video_files:
        print("❌ No video files found in output/wikipedia_demo/")
        return False
    
    # Use the most recent file
    video_file = max(video_files, key=lambda p: p.stat().st_mtime)
    # Construct index file name by replacing .mp4 with _index.json
    index_file = video_file.parent / (video_file.stem + '_index.json')
    
    if not index_file.exists():
        print(f"❌ Index file not found: {index_file}")
        return False
    
    print(f"🔍 Testing retrieval with:")
    print(f"  Video: {video_file.name}")
    print(f"  Index: {index_file.name}")
    print("=" * 60)
    
    # Initialize retriever
    try:
        retriever = MemvidRetriever(str(video_file), str(index_file))
        print("✅ Retriever initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize retriever: {e}")
        return False
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "machine learning algorithms",
        "neural networks and deep learning",
        "computer vision applications",
        "natural language processing"
    ]
    
    print(f"\n🎯 Testing semantic search with {len(test_queries)} queries:")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Query: '{query}'")
        
        try:
            # Search for relevant chunks
            results = retriever.search(query, top_k=3)
            
            if results:
                print(f"   Found {len(results)} results:")
                for j, chunk in enumerate(results, 1):
                    # Truncate long chunks for display
                    preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                    print(f"   [{j}] {preview}")
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    
    # Test search with metadata for more detailed results
    print(f"\n📊 Testing search with metadata:")
    print("-" * 60)
    
    try:
        results = retriever.search_with_metadata("machine learning", top_k=2)
        if results:
            for i, result in enumerate(results, 1):
                text = result.get('text', 'No text')[:100] + "..."
                score = result.get('score', 0)
                print(f"   [{i}] Score: {score:.3f} | {text}")
        else:
            print("   No results found")
    except Exception as e:
        print(f"   ❌ Metadata search failed: {e}")
    
    # Performance test
    print(f"\n⚡ Performance test:")
    print("-" * 60)
    
    import time
    
    query = "artificial intelligence and machine learning"
    num_tests = 5
    
    times = []
    for i in range(num_tests):
        start_time = time.time()
        results = retriever.search(query, top_k=5)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"Average search time: {avg_time:.3f} seconds ({num_tests} tests)")
    
    if avg_time < 2.0:
        print("✅ Performance: Excellent (< 2s)")
    elif avg_time < 5.0:
        print("✅ Performance: Good (< 5s)")
    else:
        print("⚠️ Performance: Slow (> 5s)")
    
    return True

if __name__ == "__main__":
    success = test_retrieval()
    if success:
        print(f"\n🎉 Retrieval testing completed successfully!")
    else:
        print(f"\n❌ Retrieval testing failed")
        sys.exit(1) 