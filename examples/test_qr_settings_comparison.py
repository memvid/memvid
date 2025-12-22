#!/usr/bin/env python3
"""
QR Settings Comparison Test

Direct comparison of original vs optimized QR settings using identical test data
to determine if the optimization impacts decode accuracy.
"""

import tempfile
import os
import sys
from pathlib import Path
import time

# Add memvid to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid.encoder import MemvidEncoder
from memvid.retriever import MemvidRetriever
from memvid import config

def test_qr_configuration(test_texts, box_size, border, config_name):
    """Test a specific QR configuration"""
    
    # Temporarily modify config
    original_box_size = config.QR_BOX_SIZE
    original_border = config.QR_BORDER
    
    try:
        # Set test configuration
        config.QR_BOX_SIZE = box_size
        config.QR_BORDER = border
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, f'{config_name}_test.mp4')
            index_path = os.path.join(temp_dir, f'{config_name}_test_index.json')
            
            # Encode
            start_time = time.time()
            encoder = MemvidEncoder()
            for text in test_texts:
                encoder.add_text(text)
            
            encoder.build_video(video_path, index_path, show_progress=False)
            encode_time = time.time() - start_time
            
            # Get file sizes
            video_size = os.path.getsize(video_path)
            index_size = os.path.getsize(index_path)
            
            # Test retrieval
            retriever = MemvidRetriever(video_path, index_path)
            
            successful_retrievals = 0
            total_tests = len(test_texts)
            retrieval_times = []
            failed_searches = []
            
            for i, original_text in enumerate(test_texts):
                # Create search query from the text
                search_query = original_text.split('.')[0] if '.' in original_text else original_text[:50]
                
                start_time = time.time()
                results = retriever.search(search_query, top_k=5)
                search_time = time.time() - start_time
                retrieval_times.append(search_time)
                
                # Check if we found the original text
                found = False
                if results:
                    for chunk in results:
                        # Allow for chunking differences
                        if (original_text[:100] in chunk or 
                            chunk[:100] in original_text or
                            any(word in chunk for word in original_text.split()[:5])):
                            found = True
                            break
                
                if found:
                    successful_retrievals += 1
                else:
                    failed_searches.append({
                        'query': search_query,
                        'original': original_text[:100],
                        'top_result': results[0][:100] if results else 'No results'
                    })
            
            return {
                'config_name': config_name,
                'box_size': box_size,
                'border': border,
                'success_rate': successful_retrievals / total_tests,
                'successful_retrievals': successful_retrievals,
                'total_tests': total_tests,
                'encode_time': encode_time,
                'avg_search_time': sum(retrieval_times) / len(retrieval_times),
                'video_size': video_size,
                'index_size': index_size,
                'failed_searches': failed_searches[:3],  # First 3 failures
                'qr_pixel_count': calculate_qr_pixels(box_size, border)
            }
            
    finally:
        # Restore original config
        config.QR_BOX_SIZE = original_box_size
        config.QR_BORDER = original_border

def calculate_qr_pixels(box_size, border):
    """Calculate total pixels for QR code with given settings"""
    # Version 35 QR: 157x157 modules
    modules = 157
    with_border = modules + (2 * border)
    total_pixels = (with_border * box_size) ** 2
    return total_pixels

def generate_comprehensive_test_data():
    """Generate comprehensive test data covering various scenarios"""
    return [
        # Short, simple text
        "Hello world! This is a basic test.",
        
        # Technical documentation
        "The MemvidEncoder class handles text chunking and QR video creation with configurable parameters for optimization.",
        
        # Code snippet
        "def encode_text(data, chunk_size=512):\n    encoder = MemvidEncoder()\n    encoder.add_text(data, chunk_size)\n    return encoder",
        
        # JSON data
        '{"user": "test", "settings": {"chunk_size": 512, "overlap": 32}, "metadata": {"version": "1.0", "timestamp": "2024-01-01"}}',
        
        # Special characters and punctuation
        "Testing special chars: @#$%^&*()_+-=[]{}|;':\",./<>? and various punctuation marks!!!",
        
        # Unicode content
        "Unicode test: αβγδε ελληνικά, 中文测试, русский текст, العربية, हिन्दी, 日本語, 한국어 🌟🔍📊",
        
        # Long technical content
        "Machine learning algorithms utilize statistical techniques to enable computer systems to improve their performance on specific tasks through experience. Deep learning, a subset of machine learning, uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
        
        # Structured data
        "RESEARCH_DATA: Subject=AI_Performance, Date=2024-01-15, Metrics=[accuracy:95.2%, precision:94.8%, recall:96.1%], Notes='Excellent results with optimized parameters'",
        
        # Repeated patterns (stress test)
        "PATTERN_TEST: " + "ABC123" * 30 + " END_PATTERN",
        
        # Mixed content with numbers and dates
        "Financial Report Q4 2023: Revenue $1,234,567.89, Growth +15.3%, Transactions: 45,678 processed between 2023-10-01 and 2023-12-31.",
        
        # Scientific notation and formulas
        "Einstein's mass-energy equation E=mc² where c≈3×10⁸ m/s, demonstrates the relationship between mass and energy in physics.",
        
        # HTML-like content
        "<document><title>Test Document</title><content>This is a test of HTML-like content with &lt;tags&gt; and &amp; entities.</content></document>",
        
        # Multi-line content
        "Line 1: Introduction to the document\nLine 2: Main content section\nLine 3: Additional details\nLine 4: Conclusion and summary",
        
        # Very long text (near QR capacity)
        "Extended content test: " + "This sentence is repeated to create a longer text chunk for testing QR capacity limits. " * 8,
        
        # Edge case: minimal content
        "A",
        
        # Edge case: numbers only
        "1234567890" * 20,
        
        # Base64-like content
        "Base64Test: aGVsbG93b3JsZA==ABC123def456GHI789jkl012MNO345pqr678STU901vwx234YZ",
        
        # SQL-like content
        "SELECT user_id, name, email FROM users WHERE active = 1 AND created_date > '2024-01-01' ORDER BY name;",
        
        # Configuration text
        "CONFIG: server.host=localhost, server.port=8080, database.url=postgresql://localhost:5432/mydb, cache.enabled=true",
        
        # Error message
        "ERROR: FileNotFoundError at line 123 in /path/to/file.py: [Errno 2] No such file or directory: 'missing_file.txt'"
    ]

def main():
    """Run comprehensive QR settings comparison"""
    print("🔍 QR Settings Comparison Test")
    print("=" * 50)
    print("Testing original vs optimized QR settings with identical data")
    print()
    
    test_data = generate_comprehensive_test_data()
    print(f"📊 Test Dataset: {len(test_data)} diverse text chunks")
    print()
    
    # Test configurations
    configs = [
        {'box_size': 5, 'border': 3, 'name': 'Original'},
        {'box_size': 3, 'border': 1, 'name': 'Optimized'},
        {'box_size': 2, 'border': 1, 'name': 'Ultra-Dense'},
        {'box_size': 1, 'border': 1, 'name': 'Minimal'}
    ]
    
    results = []
    
    for config_info in configs:
        print(f"🔧 Testing {config_info['name']} (box_size={config_info['box_size']}, border={config_info['border']})")
        
        try:
            result = test_qr_configuration(
                test_data, 
                config_info['box_size'], 
                config_info['border'], 
                config_info['name']
            )
            results.append(result)
            
            print(f"  ✅ Success Rate: {result['success_rate']*100:.1f}% ({result['successful_retrievals']}/{result['total_tests']})")
            print(f"  ⏱️  Encode Time: {result['encode_time']:.2f}s")
            print(f"  🔍 Avg Search: {result['avg_search_time']*1000:.1f}ms")
            print(f"  💾 Video Size: {result['video_size']/1024:.1f} KB")
            print(f"  📐 QR Pixels: {result['qr_pixel_count']:,}")
            
            if result['failed_searches']:
                print(f"  ⚠️  Failed Searches: {len(result['failed_searches'])}")
            
        except Exception as e:
            print(f"  ❌ Test Failed: {str(e)}")
        
        print()
    
    # Comparison Analysis
    print("=" * 60)
    print("📈 DETAILED COMPARISON")
    print("=" * 60)
    
    if len(results) >= 2:
        original = next((r for r in results if r['config_name'] == 'Original'), None)
        optimized = next((r for r in results if r['config_name'] == 'Optimized'), None)
        
        if original and optimized:
            print(f"\n🔄 Original vs Optimized Comparison:")
            print(f"{'Metric':<20} | {'Original':<15} | {'Optimized':<15} | {'Change':<15}")
            print("-" * 70)
            
            # Success rate
            orig_success = original['success_rate'] * 100
            opt_success = optimized['success_rate'] * 100
            success_diff = opt_success - orig_success
            print(f"{'Success Rate':<20} | {orig_success:<14.1f}% | {opt_success:<14.1f}% | {success_diff:+.1f}%")
            
            # File sizes
            size_reduction = (1 - optimized['video_size'] / original['video_size']) * 100
            print(f"{'Video Size':<20} | {original['video_size']/1024:<14.1f}K | {optimized['video_size']/1024:<14.1f}K | {size_reduction:+.1f}%")
            
            # QR pixel count
            pixel_reduction = (1 - optimized['qr_pixel_count'] / original['qr_pixel_count']) * 100
            print(f"{'QR Pixels':<20} | {original['qr_pixel_count']:<14,} | {optimized['qr_pixel_count']:<14,} | {pixel_reduction:+.1f}%")
            
            # Performance
            time_diff = optimized['encode_time'] - original['encode_time']
            search_diff = (optimized['avg_search_time'] - original['avg_search_time']) * 1000
            print(f"{'Encode Time':<20} | {original['encode_time']:<14.2f}s | {optimized['encode_time']:<14.2f}s | {time_diff:+.2f}s")
            print(f"{'Search Time':<20} | {original['avg_search_time']*1000:<14.1f}ms | {optimized['avg_search_time']*1000:<14.1f}ms | {search_diff:+.1f}ms")
    
    # Summary table
    print(f"\n📊 SUMMARY TABLE:")
    print(f"{'Config':<12} | {'Success':<8} | {'Video KB':<9} | {'QR Pixels':<10} | {'Status':<15}")
    print("-" * 65)
    
    for result in results:
        status = "✅ Excellent" if result['success_rate'] >= 0.95 else \
                "🟡 Good" if result['success_rate'] >= 0.80 else \
                "🟠 Fair" if result['success_rate'] >= 0.60 else \
                "❌ Poor"
        
        print(f"{result['config_name']:<12} | {result['success_rate']*100:<7.1f}% | {result['video_size']/1024:<8.1f}K | {result['qr_pixel_count']:<9,} | {status}")
    
    # Conclusion
    print(f"\n💡 CONCLUSION:")
    print("-" * 15)
    
    if original and optimized:
        if abs(orig_success - opt_success) < 5:
            print("✅ No significant difference in decode accuracy between original and optimized settings")
            print(f"✅ Optimized settings provide {pixel_reduction:.1f}% pixel reduction with same reliability")
            print("✅ RECOMMENDATION: Use optimized settings for better storage efficiency")
        elif opt_success < orig_success:
            accuracy_loss = orig_success - opt_success
            print(f"⚠️  Optimized settings show {accuracy_loss:.1f}% lower accuracy")
            if accuracy_loss > 10:
                print("🤔 RECOMMENDATION: Consider keeping original settings for critical applications")
            else:
                print("🤔 RECOMMENDATION: Optimized settings acceptable for most use cases")
        else:
            print(f"🎉 Optimized settings show {opt_success - orig_success:.1f}% better accuracy!")
            print("✅ RECOMMENDATION: Use optimized settings - they're both more efficient and more reliable")

if __name__ == "__main__":
    main() 