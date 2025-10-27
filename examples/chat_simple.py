#!/usr/bin/env python3
"""
Simple Memvid Chat with Ollama
Loads memory from output/ folder and starts chat with phi3:latest
"""
import time
import sys
from pathlib import Path

# Add memvid to path
sys.path.insert(0, str(Path(__file__).parent / "memvid-main"))

from memvid import MemvidChat

local_model = 'nemotron-mini:4b' # change this line to acces to your preffered model. But this is very good for this task.

def main():
    # Find files in output folder
    output_dir = Path(__file__).parent / "output"
    
    # Get first .mp4 and .json pair
    mp4_files = list(output_dir.glob("*.mp4"))
    
    if not mp4_files:
        print("❌ No .mp4 files found in output/ folder")
        return
    
    # Use first mp4 found
    video_file = mp4_files[0]
    
    # Find matching json (try both patterns)
    json_file = video_file.parent / f"{video_file.stem}.json"
    if not json_file.exists():
        json_file = video_file.parent / f"{video_file.stem}_index.json"
    
    if not json_file.exists():
        print(f"❌ No matching .json found for {video_file.name}")
        return
    
    print(f"📹 Video: {video_file.name}")
    print(f"📄 Index: {json_file.name}")
    print(f"🤖 Model: {local_model}")
    print()
    
    # Create chat with Ollama
    chat = MemvidChat(
        video_file=str(video_file),
        index_file=str(json_file),
        llm_provider='ollama',
        llm_model=local_model
    )
    
    # Start interactive chat
    chat.interactive_chat()

if __name__ == "__main__":
    main()

