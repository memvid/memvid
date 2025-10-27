#!/usr/bin/env python3
"""
Example: Create video memory and index from text data
"""

import sys
import os

from memvid.config import VIDEO_FILE_TYPE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memvid import MemvidEncoder
import time


def main():
    # Example data - could be from files, databases, etc.
    chunks = [
        "The quantum computer achieved 100 qubits of processing power in March 2024.",
        "Machine learning models can now process over 1 trillion parameters efficiently.",
        "The new GPU architecture delivers 5x performance improvement for AI workloads.",
        "Cloud storage costs have decreased by 80% over the past five years.",
        "Quantum encryption methods are becoming standard for secure communications.",
        "Edge computing reduces latency to under 1ms for critical applications.",
        "Neural networks can now generate photorealistic images in real-time.",
        "Blockchain technology processes over 100,000 transactions per second.",
        "5G networks provide speeds up to 10 Gbps in urban areas.",
        "Autonomous vehicles have logged over 50 million miles of testing.",
        "Natural language processing accuracy has reached 98% for major languages.",
        "Robotic process automation saves companies millions in operational costs.",
        "Augmented reality glasses now have 8-hour battery life.",
        "Biometric authentication systems have false positive rates below 0.001%.",
        "Distributed computing networks utilize idle resources from millions of devices.",
        "Green data centers run entirely on renewable energy sources.",
        "AI assistants can understand context across multiple conversation turns.",
        "Cybersecurity AI detects threats 50x faster than traditional methods.",
        "Digital twins simulate entire cities for urban planning.",
        "Voice cloning technology requires only 3 seconds of audio sample.",
    ]
    
    print("Memvid Example: Building Video Memory")
    print("=" * 50)
    
    # Create encoder
    encoder = MemvidEncoder()
    
    # Add chunks
    print(f"\nAdding {len(chunks)} chunks to encoder...")
    encoder.add_chunks(chunks)
    
    # You can also add from text with automatic chunking
    additional_text = """
    The future of computing lies in the convergence of multiple technologies.
    Quantum computing will solve problems that are intractable for classical computers.
    AI and machine learning will become embedded in every application.
    The edge and cloud will work together seamlessly to process data where it makes most sense.
    Privacy-preserving technologies will enable collaboration without exposing sensitive data.
    """
    
    print("\nAdding additional text with automatic chunking...")
    encoder.add_text(additional_text, chunk_size=100, overlap=20)
    
    # Get stats
    stats = encoder.get_stats()
    print(f"\nEncoder stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total characters: {stats['total_characters']}")
    print(f"  Average chunk size: {stats['avg_chunk_size']:.1f} chars")
    
    # Build video and index
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    video_file = os.path.join(output_dir, f"memory.{VIDEO_FILE_TYPE}]")
    index_file = os.path.join(output_dir, "memory_index.json")
    
    print(f"\nBuilding video: {video_file}")
    print(f"Building index: {index_file}")
    
    start_time = time.time()
    build_stats = encoder.build_video(video_file, index_file, show_progress=True)
    elapsed = time.time() - start_time
    
    print(f"\nBuild completed in {elapsed:.2f} seconds")
    print(f"\nVideo stats:")
    print(f"  Duration: {build_stats['duration_seconds']:.1f} seconds")
    print(f"  Size: {build_stats['video_size_mb']:.2f} MB")
    print(f"  FPS: {build_stats['fps']}")
    print(f"  Chunks per second: {build_stats['total_chunks'] / elapsed:.1f}")
    
    print("\nIndex stats:")
    for key, value in build_stats['index_stats'].items():
        print(f"  {key}: {value}")
    
    print("\nSuccess! Video memory created.")
    print(f"\nYou can now use this memory with:")
    print(f"  python examples/chat_memory.py")


if __name__ == "__main__":
    main()