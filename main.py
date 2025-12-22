#!/usr/bin/env python3
"""
Book memory example using chat_with_memory
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from memvid import MemvidEncoder, chat_with_memory

# from memvid.config import VIDEO_FILE_TYPE


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VIDEO_FILE_TYPE = "mp4"

# Book PDF path - Memvid will handle PDF parsing automatically
book_pdf = "data/bitcoin.pdf"  # Replace with your PDF path

# Build memory video from PDF
video_path = f"output/book_memory.{VIDEO_FILE_TYPE}"
index_path = "output/book_memory_index.json"

# Create output directory with subdirectory for sessions
os.makedirs("output/book_chat", exist_ok=True)

config = {
    "qr": {
        "version": 35,
        "error_correction": "M",
        "box_size": 5,
        "border": 3,
        "fill_color": "black",
        "back_color": "white",
    },
    "codec": "h265",
    "chunking": {"chunk_size": 1024, "overlap": 32},
    "retrieval": {"top_k": 5, "batch_size": 100, "max_workers": 4, "cache_size": 1000},
    "embedding": {"model": "all-MiniLM-L6-v2", "dimension": 384},
    "index": {"type": "Flat", "nlist": 100},
    "llm": {
        "model": os.getenv("OPENAI_MODEL"),
        "max_tokens": 8192,
        "temperature": 0.1,
        "context_window": 32000,
    },
    "chat": {"max_history": 10, "context_chunks": 5},
    "performance": {"prefetch_frames": 50, "decode_timeout": 10},
}
# Encode PDF to video - Memvid handles all PDF parsing internally
if not os.path.exists(video_path):
    encoder = MemvidEncoder(config=config)
    print(encoder.config)
    encoder.add_pdf(book_pdf)  # Simple one-liner to add PDF content
    encoder.build_video(video_path, index_path)
    print(f"Created book memory video: {video_path}")

# Get API key from environment or use your own
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\nNote: Set OPENAI_API_KEY environment variable for full LLM responses.")
    print("Without it, you'll only see raw context chunks.\n")

# Chat with the book - interactive session
print("\n📚 Chat with your book! Ask questions about the content.")
print("Example questions:")
print("- 'What is this document about?'")
print("- 'What are the key concepts explained?'\n")


chat_with_memory(
    video_path,
    index_path,
    api_key=api_key,
    llm_model=os.getenv("OPENAI_MODEL"),
    session_dir="output/book_chat",
    config=config,
)
