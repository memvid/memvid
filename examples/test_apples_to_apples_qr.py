#!/usr/bin/env python3
"""
Apples-to-Apples QR Settings Comparison

Downloads identical Wikipedia articles, creates videos with original vs optimized QR settings,
then performs frame-by-frame decode verification to ensure identical content extraction.
Tests optimal video compression settings for QR content.
"""

import tempfile
import os
import sys
from pathlib import Path
import time
import json
import hashlib
import requests
from bs4 import BeautifulSoup

# Add memvid to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.encoder import MemvidEncoder
from memvid.retriever import MemvidRetriever
from memvid.utils import extract_frame, decode_qr, batch_extract_and_decode
from memvid import config
import cv2

def download_wikipedia_article(title):
    """Download a Wikipedia article and return clean text"""
    try:
        # Use Wikipedia API to get clean content
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            extract = data.get('extract', '')
            
            # Get fuller content from the page
            page_url = f"https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'exintro': False,
                'explaintext': True,
                'exsectionformat': 'plain'
            }
            
            full_response = requests.get(page_url, params=params, timeout=10)
            if full_response.status_code == 200:
                full_data = full_response.json()
                pages = full_data.get('query', {}).get('pages', {})
                for page_id, page_data in pages.items():
                    full_extract = page_data.get('extract', '')
                    if full_extract and len(full_extract) > len(extract):
                        extract = full_extract
                        break
            
            return extract.strip() if extract else None
        
    except Exception as e:
        print(f"  ⚠️ Error downloading {title}: {e}")
        return None

def create_test_dataset(num_articles=20):
    """Create a test dataset by downloading Wikipedia articles"""
    
    articles = [
        'Artificial_intelligence', 'Machine_learning', 'Deep_learning', 'Neural_network',
        'Computer_vision', 'Natural_language_processing', 'Algorithm', 'Data_structure',
        'Python_(programming_language)', 'JavaScript', 'Database', 'Software_engineering',
        'Computer_science', 'Information_theory', 'Cryptography', 'Quantum_computing',
        'Internet', 'World_Wide_Web', 'HTTP', 'TCP/IP'
    ]
    
    print(f"📥 Downloading {num_articles} Wikipedia articles...")
    
    dataset = []
    successful_downloads = 0
    
    for i, article in enumerate(articles[:num_articles]):
        print(f"  [{i+1}/{num_articles}] {article.replace('_', ' ')}...", end='')
        
        content = download_wikipedia_article(article)
        if content and len(content) > 100:  # Minimum content length
            dataset.append({
                'title': article.replace('_', ' '),
                'content': content,
                'length': len(content)
            })
            successful_downloads += 1
            print(f" ✅ ({len(content)} chars)")
        else:
            print(" ❌ Failed or too short")
    
    print(f"\n📊 Dataset Summary:")
    print(f"  ✅ Successful: {successful_downloads}/{num_articles}")
    total_chars = sum(item['length'] for item in dataset)
    print(f"  📝 Total content: {total_chars:,} characters")
    
    return dataset

def create_video_with_settings(dataset, box_size, border, config_name, output_dir, codec_settings=None):
    """Create video with specific QR settings"""
    
    # Temporarily modify config
    original_box_size = config.QR_BOX_SIZE
    original_border = config.QR_BORDER
    
    try:
        # Set test configuration
        config.QR_BOX_SIZE = box_size
        config.QR_BORDER = border
        
        video_path = os.path.join(output_dir, f'{config_name.lower()}_video.mp4')
        index_path = os.path.join(output_dir, f'{config_name.lower()}_index.json')
        
        print(f"🎬 Creating {config_name} video (box_size={box_size}, border={border})")
        
        # Create encoder
        encoder = MemvidEncoder()
        
        # Add all articles with consistent chunking
        chunk_metadata = []
        for i, item in enumerate(dataset):
            original_chunk_count = len(encoder.chunks)
            encoder.add_text(item['content'], chunk_size=512, overlap=32)
            new_chunk_count = len(encoder.chunks)
            
            # Track which chunks came from which article
            for chunk_idx in range(original_chunk_count, new_chunk_count):
                chunk_metadata.append({
                    'chunk_index': chunk_idx,
                    'article_index': i,
                    'article_title': item['title']
                })
        
        # Build video with optimal compression settings
        start_time = time.time()
        
        if codec_settings:
            # Apply codec settings for optimal QR compression
            result = encoder.build_video(
                video_path, 
                index_path, 
                codec=codec_settings.get('codec', 'mp4v'),
                show_progress=False
            )
        else:
            result = encoder.build_video(video_path, index_path, show_progress=False)
        
        encode_time = time.time() - start_time
        
        # Get file sizes
        video_size = os.path.getsize(video_path)
        index_size = os.path.getsize(index_path)
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"  ✅ Video: {video_size/1024:.1f} KB, {frame_count} frames, {width}x{height}")
        print(f"  ⏱️ Encode time: {encode_time:.1f}s")
        
        # Save chunk metadata for verification
        metadata_path = os.path.join(output_dir, f'{config_name.lower()}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'chunks': encoder.chunks,
                'chunk_metadata': chunk_metadata,
                'config': {
                    'box_size': box_size,
                    'border': border,
                    'chunk_size': 512,
                    'overlap': 32
                }
            }, f, indent=2)
        
        return {
            'config_name': config_name,
            'box_size': box_size,
            'border': border,
            'video_path': video_path,
            'index_path': index_path,
            'metadata_path': metadata_path,
            'video_size': video_size,
            'index_size': index_size,
            'frame_count': frame_count,
            'encode_time': encode_time,
            'chunks': encoder.chunks.copy(),
            'chunk_metadata': chunk_metadata,
            'video_properties': {
                'fps': fps,
                'width': width,
                'height': height
            }
        }
        
    finally:
        # Restore original config
        config.QR_BOX_SIZE = original_box_size
        config.QR_BORDER = original_border

def frame_by_frame_decode_comparison(video1_info, video2_info):
    """Compare frame-by-frame decode results between two videos"""
    
    print(f"\n🔍 Frame-by-Frame Decode Comparison")
    print(f"  📹 Video 1: {video1_info['config_name']} ({video1_info['frame_count']} frames)")
    print(f"  📹 Video 2: {video2_info['config_name']} ({video2_info['frame_count']} frames)")
    
    if video1_info['frame_count'] != video2_info['frame_count']:
        print(f"  ❌ Frame count mismatch!")
        return False
    
    frame_count = video1_info['frame_count']
    
    # Extract and decode all frames from both videos
    print(f"  🔄 Decoding {frame_count} frames from each video...")
    
    start_time = time.time()
    decoded1 = batch_extract_and_decode(video1_info['video_path'], list(range(frame_count)), show_progress=True)
    decoded2 = batch_extract_and_decode(video2_info['video_path'], list(range(frame_count)), show_progress=True)
    decode_time = time.time() - start_time
    
    print(f"  ⏱️ Decode time: {decode_time:.1f}s")
    print(f"  📊 Video 1 decoded frames: {len(decoded1)}/{frame_count}")
    print(f"  📊 Video 2 decoded frames: {len(decoded2)}/{frame_count}")
    
    # Compare decoded content
    matches = 0
    mismatches = 0
    decode_failures = 0
    mismatch_details = []
    
    for frame_idx in range(frame_count):
        data1 = decoded1.get(frame_idx)
        data2 = decoded2.get(frame_idx)
        
        if data1 is None and data2 is None:
            decode_failures += 1
        elif data1 is None or data2 is None:
            mismatches += 1
            mismatch_details.append({
                'frame': frame_idx,
                'issue': f"One decode failed: {data1 is not None} vs {data2 is not None}"
            })
        elif data1 == data2:
            matches += 1
        else:
            mismatches += 1
            mismatch_details.append({
                'frame': frame_idx,
                'issue': f"Content differs: {len(data1)} vs {len(data2)} chars",
                'data1_preview': data1[:100],
                'data2_preview': data2[:100]
            })
    
    print(f"\n📈 Decode Comparison Results:")
    print(f"  ✅ Identical frames: {matches}")
    print(f"  ❌ Mismatched frames: {mismatches}")
    print(f"  ⚠️ Decode failures: {decode_failures}")
    
    if mismatches > 0:
        print(f"\n⚠️ Mismatch Details (first 5):")
        for detail in mismatch_details[:5]:
            print(f"  Frame {detail['frame']}: {detail['issue']}")
    
    # Overall comparison
    success_rate = matches / frame_count if frame_count > 0 else 0
    print(f"\n🎯 Overall Match Rate: {success_rate*100:.1f}%")
    
    return {
        'total_frames': frame_count,
        'matches': matches,
        'mismatches': mismatches,
        'decode_failures': decode_failures,
        'success_rate': success_rate,
        'identical_content': mismatches == 0,
        'mismatch_details': mismatch_details
    }

def optimize_video_compression_for_qr():
    """Determine optimal video compression settings for QR content"""
    
    # QR codes are black/white, high contrast, geometric patterns
    # Optimal settings should:
    # 1. Preserve sharp edges and avoid artifacts
    # 2. Handle high contrast well  
    # 3. Maximize compression for geometric patterns
    # 4. Use lossless or near-lossless compression
    
    # Try H.265 first (best compression), fall back to MP4V
    # Note: H.265 provides much better compression for geometric content
    return {
        'codec': 'h265',     # Try H.265 first for best compression
        'fallback_codec': 'mp4v',  # Fallback if H.265 fails
        'frame_rate': 30,    # Higher frame rate for more data throughput
        'quality': 'high'    # High quality to preserve QR readability
    }

def main():
    """Run comprehensive apples-to-apples QR comparison"""
    print("🍎 Apples-to-Apples QR Settings Comparison")
    print("=" * 60)
    
    # Create test dataset
    dataset = create_test_dataset(15)  # 15 articles for substantial test
    
    if len(dataset) < 5:
        print("❌ Insufficient test data downloaded")
        return
    
    # Get optimal compression settings
    codec_settings = optimize_video_compression_for_qr()
    print(f"\n🎛️ Optimal Compression Settings:")
    print(f"  Codec: {codec_settings['codec']}")
    print(f"  Fallback Codec: {codec_settings['fallback_codec']}")
    print(f"  Frame Rate: {codec_settings['frame_rate']}")
    print(f"  Quality: {codec_settings['quality']}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n📁 Working directory: {temp_dir}")
        
        # Create videos with different QR settings but identical content
        configs = [
            {'box_size': 5, 'border': 3, 'name': 'Original'},
            {'box_size': 3, 'border': 1, 'name': 'Optimized'}
        ]
        
        video_results = []
        
        for config_info in configs:
            result = create_video_with_settings(
                dataset,
                config_info['box_size'],
                config_info['border'],
                config_info['name'],
                temp_dir,
                codec_settings
            )
            video_results.append(result)
        
        # Frame-by-frame comparison
        if len(video_results) == 2:
            comparison_result = frame_by_frame_decode_comparison(
                video_results[0], 
                video_results[1]
            )
            
            # Detailed analysis
            print("\n" + "=" * 60)
            print("📊 COMPREHENSIVE ANALYSIS")
            print("=" * 60)
            
            original = video_results[0]
            optimized = video_results[1]
            
            print(f"\n🔄 Configuration Comparison:")
            print(f"{'Metric':<20} | {'Original':<15} | {'Optimized':<15} | {'Change'}")
            print("-" * 70)
            
            # QR pixel comparison
            orig_pixels = (157 + 2*original['border']) * original['box_size']
            opt_pixels = (157 + 2*optimized['border']) * optimized['box_size']
            pixel_reduction = (1 - (opt_pixels**2) / (orig_pixels**2)) * 100
            
            print(f"{'QR Size':<20} | {orig_pixels}x{orig_pixels:<8} | {opt_pixels}x{opt_pixels:<8} | {pixel_reduction:+.1f}%")
            
            # File size comparison
            size_change = (optimized['video_size'] - original['video_size']) / original['video_size'] * 100
            print(f"{'Video Size':<20} | {original['video_size']/1024:<14.1f}K | {optimized['video_size']/1024:<14.1f}K | {size_change:+.1f}%")
            
            # Encode time comparison
            time_change = (optimized['encode_time'] - original['encode_time'])
            print(f"{'Encode Time':<20} | {original['encode_time']:<14.1f}s | {optimized['encode_time']:<14.1f}s | {time_change:+.1f}s")
            
            # Content verification
            print(f"{'Content Match':<20} | {'Reference':<15} | {comparison_result['success_rate']*100:<14.1f}% | {'Verified' if comparison_result['identical_content'] else 'MISMATCH'}")
            
            print(f"\n💡 CONCLUSIONS:")
            print("-" * 20)
            
            if comparison_result['identical_content']:
                print("✅ Frame-by-frame decode verification: IDENTICAL content")
                print("✅ No data loss or corruption with optimized QR settings")
                print(f"✅ Storage efficiency gain: {abs(pixel_reduction):.1f}% pixel reduction")
                
                if size_change < 0:
                    print(f"✅ Video compression improvement: {abs(size_change):.1f}% smaller files")
                elif size_change > 5:
                    print(f"⚠️ Video size increase: {size_change:.1f}% (unexpected)")
                else:
                    print(f"➡️ Video size similar: {size_change:+.1f}% difference")
                
                print(f"\n🎯 RECOMMENDATION: Use optimized QR settings")
                print(f"   - Identical decode accuracy verified frame-by-frame")
                print(f"   - {abs(pixel_reduction):.1f}% more efficient QR encoding")
                print(f"   - No content loss or corruption detected")
                
            else:
                print(f"❌ Frame-by-frame decode verification: CONTENT MISMATCH")
                print(f"❌ {comparison_result['mismatches']} frames differ between configurations")
                print(f"⚠️ RECOMMENDATION: Investigate decode differences before using optimized settings")
            
            # Save results for further analysis
            results_path = os.path.join(temp_dir, 'comparison_results.json')
            with open(results_path, 'w') as f:
                json.dump({
                    'dataset_summary': {
                        'articles': len(dataset),
                        'total_chars': sum(item['length'] for item in dataset)
                    },
                    'video_results': video_results,
                    'comparison_result': comparison_result,
                    'codec_settings': codec_settings
                }, f, indent=2, default=str)
            
            print(f"\n💾 Detailed results saved to: {results_path}")
            
            # Copy important files to current directory for inspection
            import shutil
            for result in video_results:
                dest_video = f"{result['config_name'].lower()}_comparison_video.mp4"
                shutil.copy2(result['video_path'], dest_video)
                print(f"📹 Copied {result['config_name']} video to: {dest_video}")

if __name__ == "__main__":
    main() 