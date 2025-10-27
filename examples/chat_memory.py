#!/usr/bin/env python3
"""
Example: Interactive conversation using MemvidChat
"""

import sys
import os

from memvid.config import VIDEO_FILE_TYPE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memvid import MemvidChat
import time


def print_search_results(results):
    """Pretty print search results"""
    print("\nRelevant context found:")
    print("-" * 50)
    for i, result in enumerate(results[:3]):
        print(f"\n[{i+1}] Score: {result['score']:.3f}")
        print(f"Text: {result['text'][:150]}...")
        print(f"Frame: {result['frame']}")


def main():
    print("Memvid Example: Interactive Chat with Memory")
    print("=" * 50)
    
    # Check if memory files exist
    video_file = f"output/memory.{VIDEO_FILE_TYPE}"
    index_file = "output/memory_index.json"
    
    if not os.path.exists(video_file) or not os.path.exists(index_file):
        print("\nError: Memory files not found!")
        print("Please run 'python examples/build_memory.py' first to create the memory.")
        return
    
    # Initialize chat
    print(f"\nLoading memory from: {video_file}")
    
    # You can set OPENAI_API_KEY, GOOGLE_API_KEY or ANTHROPIC_API_KEY as an environment variable or pass it here
    api_key = "your-api-key-here"
    if not api_key:
        print("\nNote: No OpenAI API key found. Chat will work in context-only mode.")
        print("Set OPENAI_API_KEY environment variable to enable full chat capabilities.")
    
    chat = MemvidChat(video_file, index_file, llm_api_key=api_key)
    chat.start_session()
    
    # Get stats
    stats = chat.get_stats()
    print(f"\nMemory loaded successfully!")
    print(f"  Total chunks: {stats['retriever_stats']['index_stats']['total_chunks']}")
    print(f"  LLM available: {stats['llm_available']}")
    if stats['llm_available']:
        print(f"  LLM model: {stats['llm_model']}")
    
    print("\nInstructions:")
    print("- Type your questions to search the memory")
    print("- Type 'search <query>' to see raw search results")
    print("- Type 'stats' to see system statistics")
    print("- Type 'export' to save conversation")
    print("- Type 'exit' or 'quit' to end the session")
    print("-" * 50)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
                
            elif user_input.lower() == 'stats':
                stats = chat.get_stats()
                print("\nSystem Statistics:")
                print(f"  Messages: {stats['message_count']}")
                print(f"  Cache size: {stats['retriever_stats']['cache_size']}")
                print(f"  Video frames: {stats['retriever_stats']['total_frames']}")
                continue
                
            elif user_input.lower() == 'export':
                export_file = f"output/session_{chat.session_id}.json"
                chat.export_session(export_file)
                print(f"Session exported to: {export_file}")
                continue
                
            elif user_input.lower().startswith('search '):
                query = user_input[7:]
                print(f"\nSearching for: '{query}'")
                start_time = time.time()
                results = chat.search_context(query, top_k=5)
                elapsed = time.time() - start_time
                print(f"Search completed in {elapsed:.3f} seconds")
                print_search_results(results)
                continue
            
            # Regular chat
            print("\nAssistant: ", end="", flush=True)
            start_time = time.time()
            response = chat.chat(user_input)
            elapsed = time.time() - start_time
            
            print(response)
            print(f"\n[Response time: {elapsed:.2f}s]")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    # Export session on exit
    if chat.get_history():
        export_file = f"output/session_{chat.session_id}.json"
        chat.export_session(export_file)
        print(f"\nSession saved to: {export_file}")


if __name__ == "__main__":
    main()