#!/usr/bin/env python3
"""
Multi-Codec Comparison Tool - Compare ALL available codecs on YOUR data
"""

import sys
import argparse
import time
from pathlib import Path
import json

# Add project root to Python path - works from anywhere
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Also try adding current directory if script is in project root
sys.path.insert(0, str(Path.cwd()))

try:
    from memvid.encoder import MemvidEncoder
    from memvid.config import codec_parameters, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
    print(f"✅ Imported MemvidEncoder from: {project_root}")
except ImportError as e:
    print(f"❌ Could not import MemvidEncoder from {project_root}")
    print(f"   Error: {e}")
    print(f"   Current working directory: {Path.cwd()}")
    print("   Python path includes:")
    for p in sys.path[:5]:  # Show first 5 paths
        print(f"     {p}")
    print()
    print("Solutions:")
    print("1. Run from project root: cd memvid && python examples/codec_comparison.py ...")
    print("2. Install package: pip install -e .")
    print("3. Set PYTHONPATH: export PYTHONPATH=/path/to/memvid:$PYTHONPATH")
    sys.exit(1)


def format_size(bytes_size):
    """Format file size in human readable format."""
    if bytes_size == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def get_available_codecs(encoder):
    """Get list of available codecs and their backends - FIXED"""
    # Import the full codec mapping, not just the default config
    from memvid.config import codec_parameters

    available_codecs = {}

    for codec in codec_parameters.keys():
        if codec == "mp4v":
            available_codecs[codec] = "native"
        else:
            if encoder.dcker_mngr and encoder.dcker_mngr.should_use_docker(codec):
                available_codecs[codec] = "docker"
            else:
                available_codecs[codec] = "native_ffmpeg"

    return available_codecs

def load_user_data(input_path, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Load data from user's file or directory - FIXED VERSION"""
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"❌ Path not found: {input_path}")
        return None, None

    encoder = MemvidEncoder()

    # Handle directory vs single file
    if input_path.is_dir():
        print(f"📂 Loading directory: {input_path}")

        # Find supported files in directory
        supported_extensions = ['.pdf', '.epub', '.txt', '.md', '.json']
        files = []

        for ext in supported_extensions:
            found = list(input_path.rglob(f"*{ext}"))
            files.extend(found)

        if not files:
            print(f"❌ No supported files found in {input_path}")
            print(f"   Looking for: {', '.join(supported_extensions)}")
            return None, None

        print(f"📁 Found {len(files)} supported files")

        # Load all files
        total_files_processed = 0
        total_files_failed = 0

        for file_path in files:
            try:
                if file_path.suffix.lower() == '.pdf':
                    encoder.add_pdf(str(file_path), chunk_size=chunk_size, overlap=overlap)
                elif file_path.suffix.lower() == '.epub':
                    encoder.add_epub(str(file_path), chunk_size=chunk_size, overlap=overlap)
                elif file_path.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    if isinstance(chunks, list):
                        encoder.add_chunks(chunks)
                else:
                    # Text file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    encoder.add_text(text, chunk_size=chunk_size, overlap=overlap)

                total_files_processed += 1
                print(f"   ✅ Processed: {file_path.name}")

            except Exception as e:
                print(f"   ❌ Failed to process {file_path.name}: {e}")
                total_files_failed += 1

        if not encoder.chunks:
            print("❌ No content extracted from any files")
            return None, None

        total_chars = sum(len(chunk) for chunk in encoder.chunks)

        info = {
            'type': 'directory',
            'path': str(input_path),
            'files_processed': total_files_processed,
            'files_failed': total_files_failed,
            'chunks': len(encoder.chunks),
            'total_chars': total_chars,
            'avg_chunk_size': total_chars / len(encoder.chunks)
        }

        print(f"📊 Summary: {total_files_processed} files processed, {len(encoder.chunks)} chunks extracted")

        return encoder, info

    else:
        # Single file - your existing logic
        print(f"📄 Loading file: {input_path.name}")

        try:
            if input_path.suffix.lower() == '.pdf':
                encoder.add_pdf(str(input_path), chunk_size=chunk_size, overlap=overlap)
            elif input_path.suffix.lower() == '.epub':
                encoder.add_epub(str(input_path), chunk_size=chunk_size, overlap=overlap)
            elif input_path.suffix.lower() == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                if isinstance(chunks, list):
                    encoder.add_chunks(chunks)
                else:
                    print("❌ JSON file must contain a list of text chunks")
                    return None, None
            else:
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                encoder.add_text(text, chunk_size=chunk_size, overlap=overlap)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None, None

        if not encoder.chunks:
            print("❌ No content extracted from file")
            return None, None

        total_chars = sum(len(chunk) for chunk in encoder.chunks)

        info = {
            'type': 'file',
            'file_name': input_path.name,
            'file_size': input_path.stat().st_size,
            'chunks': len(encoder.chunks),
            'total_chars': total_chars,
            'avg_chunk_size': total_chars / len(encoder.chunks)
        }

        return encoder, info

def test_codec(encoder, codec, name_stem, output_dir):
    """Test encoding with a specific codec - FIXED"""
    print(f"\n🎬 Testing {codec.upper()} encoding...")

    # Create fresh encoder copy to avoid state issues
    test_encoder = MemvidEncoder()
    test_encoder.chunks = encoder.chunks.copy()

    # Determine file extension from full codec mapping - FIXED
    file_ext = codec_parameters[codec]["video_file_type"]

    # FIX: Add missing dot before extension
    output_path = output_dir / f"{name_stem}_{codec}.{file_ext}"
    index_path = output_dir / f"{name_stem}_{codec}.json"

    start_time = time.time()

    try:
        stats = test_encoder.build_video(
            str(output_path),
            str(index_path),
            codec=codec,
            show_progress=False,  # Keep output clean
            auto_build_docker=True,
            allow_fallback=False  # Fail rather than fallback for comparison
        )

        encoding_time = time.time() - start_time

        # Check if file was actually created and has content
        if output_path.exists():
            file_size = output_path.stat().st_size
        else:
            print(f"   ❌ Output file was not created: {output_path}")
            return {
                'success': False,
                'codec': codec,
                'error': "Output file was not created"
            }

        # Check for zero-byte files
        if file_size == 0:
            print("   ❌ Zero-byte file created - encoding failed")
            return {
                'success': False,
                'codec': codec,
                'error': "Zero-byte file created"
            }

        result = {
            'success': True,
            'codec': codec,
            'backend': stats.get('backend', 'unknown'),
            'file_size': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'encoding_time': encoding_time,
            'chunks_per_mb': len(encoder.chunks) / (file_size / (1024 * 1024)) if file_size > 0 else 0,
            'path': output_path,
            'file_ext': file_ext
        }

        print(f"   ✅ Success: {format_size(file_size)} in {encoding_time:.1f}s via {result['backend']}")
        return result

    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return {
            'success': False,
            'codec': codec,
            'error': str(e)
        }

def run_multi_codec_comparison(encoder, data_info, codecs, output_dir="output"):
    """Run comparison across multiple codecs - FIXED for directories"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate output name based on input type - FIXED
    timestamp = int(time.time())

    if data_info['type'] == 'directory':
        # Use directory name for collections
        dir_name = Path(data_info['path']).name
        name_stem = f"{dir_name}_{data_info['files_processed']}files_{timestamp}"
    else:
        # Use file name for single files
        name_stem = f"{Path(data_info['file_name']).stem}_{timestamp}"

    print("\n🏁 Multi-Codec Comparison Starting")
    print("=" * 60)
    print(f"📁 Output prefix: {name_stem}")

    results = {}

    for codec in codecs:
        results[codec] = test_codec(encoder, codec, name_stem, output_path)

    return results

def print_comparison_table(data_info, results, codecs):
    """Print detailed comparison table - FIXED zero division"""

    print("\n📊 MULTI-CODEC COMPARISON RESULTS")
    print("=" * 80)

    if data_info['type'] == 'directory':
        print(f"📁 Source: {data_info['path']} ({data_info['files_processed']} files)")
    else:
        print(f"📄 Source: {data_info['file_name']}")

    print(f"📋 Content: {data_info['chunks']} chunks, {format_size(data_info['total_chars'])} characters")
    print()

    # Prepare table data
    successful_results = [(codec, result) for codec, result in results.items() if result['success']]
    failed_results = [(codec, result) for codec, result in results.items() if not result['success']]

    if successful_results:
        print("✅ SUCCESSFUL ENCODINGS:")
        print("-" * 80)
        print(f"{'Codec':<8} {'Backend':<12} {'Size':<12} {'Chunks/MB':<10} {'Time':<8} {'Ratio':<8}")
        print("-" * 80)

        # Sort by file size (smallest first)
        successful_results.sort(key=lambda x: x[1]['file_size'])

        # FIX: Handle zero baseline size
        baseline_size = successful_results[0][1]['file_size'] if successful_results else 1
        if baseline_size == 0:
            baseline_size = 1  # Avoid division by zero

        for codec, result in successful_results:
            size_str = format_size(result['file_size'])
            chunks_per_mb = f"{result['chunks_per_mb']:.0f}" if result['chunks_per_mb'] > 0 else "N/A"
            time_str = f"{result['encoding_time']:.1f}s"
            backend = result['backend']
            ratio = result['file_size'] / baseline_size if baseline_size > 0 else 1.0
            ratio_str = f"{ratio:.1f}x" if ratio != 1.0 else "baseline"

            print(f"{codec:<8} {backend:<12} {size_str:<12} {chunks_per_mb:<10} {time_str:<8} {ratio_str:<8}")

        # Find the best compression ratio and best speed - with safety checks
        if successful_results:
            best_compression = min(successful_results, key=lambda x: x[1]['file_size'])
            fastest = min(successful_results, key=lambda x: x[1]['encoding_time'])

            print()
            print(f"🏆 Best Compression: {best_compression[0].upper()} ({format_size(best_compression[1]['file_size'])})")
            print(f"⚡ Fastest Encoding: {fastest[0].upper()} ({fastest[1]['encoding_time']:.1f}s)")

            # Calculate storage efficiency
            total_chunks = data_info['chunks']
            print("\n💾 Storage Efficiency (chunks per MB):")
            print(f"📦 Total chunks in dataset: {total_chunks}")

            storage_efficiency = []
            for codec, result in successful_results:
                # Get file size in MB
                file_size_mb = result['file_size'] / (1024 * 1024)  # Convert bytes to MB

                # Calculate chunks per MB
                chunks_per_mb = total_chunks / file_size_mb if file_size_mb > 0 else 0

                storage_efficiency.append((codec, chunks_per_mb, file_size_mb))
                print(f"   {codec.upper()}: {chunks_per_mb:.1f} chunks/MB ({file_size_mb:.1f} MB total)")

            # Sort by efficiency (highest chunks per MB first)
            storage_efficiency.sort(key=lambda x: x[1], reverse=True)

            print("\n🏆 Storage Efficiency Ranking:")
            for i, (codec, chunks_per_mb, file_size_mb) in enumerate(storage_efficiency, 1):
                efficiency_vs_best = chunks_per_mb / storage_efficiency[0][1] if storage_efficiency[0][1] > 0 else 0
                print(f"   {i}. {codec.upper()}: {chunks_per_mb:.1f} chunks/MB ({efficiency_vs_best:.1%} of best)")

    if failed_results:
        print("\n❌ FAILED ENCODINGS:")
        print("-" * 40)
        for codec, result in failed_results:
            print(f"   {codec.upper()}: {result['error']}")

    print(f"\n📁 Output files saved to: {Path('output').absolute()}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple video codecs on YOUR data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input_path', help='Path to your file (PDF, EPUB, TXT, JSON)')
    parser.add_argument('--codecs', nargs='+', default=['mp4v', 'h265'],
                        help='Codecs to test (default: mp4v h265). Use "all" for all available.')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help='Chunk size for text splitting')
    parser.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP,
                        help='Overlap for text splitting')
    parser.add_argument('--output-dir', default='output',
                        help='Output directory (default: output)')

    args = parser.parse_args()

    print("🎥 Memvid Multi-Codec Comparison Tool")
    print("=" * 50)

    # Load user data
    encoder, data_info = load_user_data(args.input_path, args.chunk_size)
    if not encoder:
        sys.exit(1)

    # Determine available codecs
    available_codecs = get_available_codecs(encoder)

    print("\n🎛️  Available Codecs:")
    for codec, backend in available_codecs.items():
        print(f"   {codec.upper()}: {backend}")

    # Parse codec selection
    if args.codecs == ['all']:
        test_codecs = list(available_codecs.keys())
    else:
        test_codecs = []
        for codec in args.codecs:
            if codec in available_codecs:
                test_codecs.append(codec)
            else:
                print(f"⚠️  Codec '{codec}' not available, skipping")

        if not test_codecs:
            print("❌ No valid codecs specified")
            sys.exit(1)

    print(f"\n🧪 Testing Codecs: {', '.join(test_codecs)}")

    # Show Docker status
    docker_status = encoder.get_docker_status()
    print(f"🐳 Docker Status: {docker_status}")

    # Run comparison
    results = run_multi_codec_comparison(encoder, data_info, test_codecs, args.output_dir)

    # Show results
    print_comparison_table(data_info, results, test_codecs)

    print("\n🎉 Multi-codec comparison complete!")

if __name__ == '__main__':
    main()