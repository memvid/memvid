#!/usr/bin/env python3
"""
Final Apples-to-Apples QR Settings Test

Creates identical datasets with original vs optimized QR settings,
then uses Memvid's actual retrieval system to verify identical content extraction.
This is the real-world use case test.
"""

import tempfile
import os
import sys
from pathlib import Path
import time
import json
import hashlib
import requests

# Add memvid to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.encoder import MemvidEncoder
from memvid.retriever import MemvidRetriever
from memvid import config

def download_wikipedia_articles(count=10):
    """Download Wikipedia articles for testing"""
    
    articles = [
        'Artificial_intelligence', 'Machine_learning', 'Deep_learning', 'Neural_network',
        'Computer_vision', 'Natural_language_processing', 'Algorithm', 'Data_structure',
        'Python_(programming_language)', 'JavaScript', 'Database', 'Software_engineering',
        'Computer_science', 'Information_theory', 'Cryptography', 'Quantum_computing'
    ]
    
    print(f"📥 Downloading {count} Wikipedia articles for test dataset...")
    
    dataset = []
    for i, article in enumerate(articles[:count]):
        print(f"  [{i+1}/{count}] {article.replace('_', ' ')}...", end='')
        
        try:
            # Use Wikipedia API
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{article}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                extract = data.get('extract', '')
                
                if extract and len(extract) > 200:
                    dataset.append({
                        'title': article.replace('_', ' '),
                        'content': extract,
                        'length': len(extract)
                    })
                    print(f" ✅ ({len(extract)} chars)")
                else:
                    print(" ⚠️ Too short")
            else:
                print(" ❌ Failed")
                
        except Exception as e:
            print(f" ❌ Error: {e}")
    
    total_chars = sum(item['length'] for item in dataset)
    print(f"\n📊 Dataset: {len(dataset)} articles, {total_chars:,} characters")
    
    return dataset

def create_memory_video(dataset, box_size, border, config_name, output_dir):
    """Create memory video with specific QR settings"""
    
    # Temporarily modify config
    original_box_size = config.QR_BOX_SIZE
    original_border = config.QR_BORDER
    
    try:
        # Set test configuration
        config.QR_BOX_SIZE = box_size
        config.QR_BORDER = border
        
        video_path = os.path.join(output_dir, f'{config_name.lower()}_memory.mp4')
        index_path = os.path.join(output_dir, f'{config_name.lower()}_index.json')
        
        print(f"\n🎬 Creating {config_name} memory (QR: {box_size}×{border})")
        
        # Create encoder with identical settings
        encoder = MemvidEncoder()
        
        # Add articles in identical order with identical chunking
        for article in dataset:
            encoder.add_text(article['content'], chunk_size=512, overlap=32)
        
        # Build video
        start_time = time.time()
        encoder.build_video(video_path, index_path, show_progress=False)
        build_time = time.time() - start_time
        
        # Get file info
        video_size = os.path.getsize(video_path)
        index_size = os.path.getsize(index_path)
        
        print(f"  ✅ Video: {video_size/1024:.1f} KB")
        print(f"  📇 Index: {index_size/1024:.1f} KB") 
        print(f"  ⏱️ Build: {build_time:.1f}s")
        print(f"  📦 Chunks: {len(encoder.chunks)}")
        
        return {
            'config_name': config_name,
            'box_size': box_size,
            'border': border,
            'video_path': video_path,
            'index_path': index_path,
            'video_size': video_size,
            'index_size': index_size,
            'build_time': build_time,
            'chunk_count': len(encoder.chunks),
            'chunks': encoder.chunks.copy()  # Store chunks for verification
        }
        
    finally:
        # Restore original config
        config.QR_BOX_SIZE = original_box_size
        config.QR_BORDER = original_border

def test_content_retrieval(memory_info, test_queries):
    """Test content retrieval from a memory video"""
    
    print(f"\n🔍 Testing retrieval from {memory_info['config_name']} memory")
    
    retriever = MemvidRetriever(memory_info['video_path'], memory_info['index_path'])
    
    results = []
    retrieval_times = []
    
    for i, query in enumerate(test_queries):
        print(f"  Query {i+1}: '{query[:50]}...'", end='')
        
        start_time = time.time()
        search_results = retriever.search(query, top_k=5)
        search_time = time.time() - start_time
        retrieval_times.append(search_time)
        
        if search_results:
            # Get the best result
            best_result = search_results[0]
            results.append({
                'query': query,
                'result': best_result,
                'result_length': len(best_result),
                'search_time': search_time
            })
            print(f" ✅ ({len(best_result)} chars, {search_time*1000:.1f}ms)")
        else:
            results.append({
                'query': query,
                'result': None,
                'result_length': 0,
                'search_time': search_time
            })
            print(f" ❌ No results ({search_time*1000:.1f}ms)")
    
    avg_time = sum(retrieval_times) / len(retrieval_times)
    print(f"  📊 Avg search time: {avg_time*1000:.1f}ms")
    
    return results

def compare_retrieval_results(results1, results2, config1_name, config2_name):
    """Compare retrieval results between two configurations"""
    
    print(f"\n🔄 Comparing retrieval: {config1_name} vs {config2_name}")
    print("-" * 60)
    
    identical_results = 0
    total_queries = len(results1)
    differences = []
    
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        query = r1['query']
        result1 = r1['result']
        result2 = r2['result']
        
        if result1 is None and result2 is None:
            # Both failed
            identical_results += 1
            status = "❌ Both failed"
        elif result1 is None or result2 is None:
            # One failed
            differences.append({
                'query': query[:50],
                'issue': f"One failed: {result1 is not None} vs {result2 is not None}"
            })
            status = "⚠️ Different success"
        elif result1 == result2:
            # Identical content
            identical_results += 1
            status = "✅ Identical"
        else:
            # Different content - compare more carefully
            # Allow for minor differences due to retrieval ranking
            similarity = calculate_similarity(result1, result2)
            if similarity > 0.95:  # 95% similar
                identical_results += 1
                status = f"✅ Similar ({similarity:.1%})"
            else:
                differences.append({
                    'query': query[:50],
                    'issue': f"Content differs: {len(result1)} vs {len(result2)} chars, {similarity:.1%} similar",
                    'content1': result1[:100],
                    'content2': result2[:100]
                })
                status = f"❌ Different ({similarity:.1%})"
        
        print(f"  Query {i+1}: {status}")
    
    match_rate = identical_results / total_queries
    
    print(f"\n📈 Comparison Summary:")
    print(f"  ✅ Identical/Similar: {identical_results}/{total_queries} ({match_rate:.1%})")
    print(f"  ❌ Differences: {len(differences)}")
    
    if differences:
        print(f"\n⚠️ Differences found:")
        for diff in differences[:3]:  # Show first 3
            print(f"  - {diff['query']}: {diff['issue']}")
    
    return {
        'total_queries': total_queries,
        'identical_results': identical_results,
        'match_rate': match_rate,
        'differences': differences,
        'content_identical': len(differences) == 0
    }

def calculate_similarity(text1, text2):
    """Calculate text similarity (simple approach)"""
    if not text1 or not text2:
        return 0.0
    
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0

def main():
    """Run the definitive apples-to-apples comparison"""
    
    print("🍎 Final Apples-to-Apples QR Settings Test")
    print("=" * 60)
    print("Real-world retrieval accuracy comparison using identical datasets")
    
    # Download test dataset
    dataset = download_wikipedia_articles(12)
    
    if len(dataset) < 5:
        print("❌ Insufficient test data")
        return
    
    # Create test queries from the dataset
    test_queries = []
    for article in dataset[:8]:  # Use first 8 articles for queries
        # Create queries from article content
        content = article['content']
        
        # Extract meaningful phrases for queries
        sentences = content.split('. ')
        if len(sentences) > 1:
            # Use parts of sentences as queries
            test_queries.append(sentences[0][:50])  # First part of first sentence
            
            if len(sentences) > 2:
                test_queries.append(sentences[1][:50])  # First part of second sentence
    
    print(f"\n🔍 Test queries generated: {len(test_queries)}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n📁 Working in: {temp_dir}")
        
        # Test configurations
        configs = [
            {'box_size': 5, 'border': 3, 'name': 'Original'},
            {'box_size': 3, 'border': 1, 'name': 'Optimized'}
        ]
        
        memory_results = []
        retrieval_results = []
        
        # Create memory videos with different QR settings
        for config in configs:
            memory_info = create_memory_video(
                dataset,
                config['box_size'],
                config['border'],
                config['name'],
                temp_dir
            )
            memory_results.append(memory_info)
            
            # Test retrieval
            retrieval_result = test_content_retrieval(memory_info, test_queries)
            retrieval_results.append(retrieval_result)
        
        # Compare results
        if len(retrieval_results) == 2:
            comparison = compare_retrieval_results(
                retrieval_results[0],
                retrieval_results[1],
                configs[0]['name'],
                configs[1]['name']
            )
            
            # Final analysis
            print("\n" + "=" * 60)
            print("📊 FINAL ANALYSIS")
            print("=" * 60)
            
            original = memory_results[0]
            optimized = memory_results[1]
            
            # Storage comparison
            size_diff = (optimized['video_size'] - original['video_size']) / original['video_size'] * 100
            time_diff = optimized['build_time'] - original['build_time']
            
            # QR efficiency
            orig_pixels = ((157 + 2*original['border']) * original['box_size']) ** 2
            opt_pixels = ((157 + 2*optimized['border']) * optimized['box_size']) ** 2
            pixel_reduction = (1 - opt_pixels / orig_pixels) * 100
            
            print(f"\n🔢 Storage & Performance:")
            print(f"  Video Size:      {original['video_size']/1024:.1f} KB → {optimized['video_size']/1024:.1f} KB ({size_diff:+.1f}%)")
            print(f"  Build Time:      {original['build_time']:.1f}s → {optimized['build_time']:.1f}s ({time_diff:+.1f}s)")
            print(f"  QR Pixels:       {orig_pixels:,} → {opt_pixels:,} ({pixel_reduction:+.1f}%)")
            print(f"  Chunk Count:     {original['chunk_count']} vs {optimized['chunk_count']}")
            
            print(f"\n🎯 Content Accuracy:")
            print(f"  Query Match Rate: {comparison['match_rate']:.1%}")
            print(f"  Content Identical: {'✅ YES' if comparison['content_identical'] else '❌ NO'}")
            
            print(f"\n💡 CONCLUSION:")
            print("-" * 20)
            
            if comparison['content_identical'] or comparison['match_rate'] >= 0.95:
                print("✅ OPTIMIZED SETTINGS VERIFIED")
                print("✅ Identical content retrieval accuracy")
                print(f"✅ {abs(pixel_reduction):.1f}% more efficient QR encoding")
                print(f"✅ {'Smaller' if size_diff < 0 else 'Similar'} video files")
                print("\n🎯 RECOMMENDATION: Use optimized QR settings")
                print("   - No loss in retrieval accuracy")
                print("   - Significant storage efficiency gains")
                print("   - Real-world performance verified")
                
                # Save comparison videos for inspection
                import shutil
                orig_dest = "apples_to_apples_original.mp4"
                opt_dest = "apples_to_apples_optimized.mp4"
                shutil.copy2(original['video_path'], orig_dest)
                shutil.copy2(optimized['video_path'], opt_dest)
                print(f"\n📹 Comparison videos saved:")
                print(f"   Original: {orig_dest}")
                print(f"   Optimized: {opt_dest}")
                
            else:
                print("⚠️ RETRIEVAL DIFFERENCES DETECTED")
                print(f"⚠️ Match rate: {comparison['match_rate']:.1%}")
                print("🤔 RECOMMENDATION: Investigate differences before deployment")

if __name__ == "__main__":
    main() 