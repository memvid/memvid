#!/usr/bin/env python3
"""
Ollama Local LLM Chat Example

This example demonstrates how to use Memvid with local LLM models via Ollama.
Ollama allows running open-source models like Llama, Mistral, and others locally
without requiring API keys or internet connectivity after initial model download.

Prerequisites:
1. Install Ollama:
   - Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh
   - Windows: Download from https://ollama.com/download
   
2. Pull a model (choose one):
   - ollama pull llama3.2:3b    # Small, fast (3GB)
   - ollama pull llama3.2:latest # Latest Llama (several GB)
   - ollama pull mistral:latest  # Mistral 7B model
   - ollama pull phi3:latest     # Microsoft Phi-3
   
3. Start Ollama server (usually auto-starts):
   - Check status: ollama list
   - Manual start: ollama serve
"""

import sys
import os

# Add parent directory to path to import memvid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memvid import MemvidChat, MemvidEncoder


def create_sample_memory():
    """
    Create a sample knowledge base about space exploration.
    
    This function demonstrates building a video memory from text chunks
    that can be queried later using semantic search.
    """
    print("=" * 60)
    print("Creating Sample Knowledge Base")
    print("=" * 60)
    
    # Initialize encoder to convert text into video format
    encoder = MemvidEncoder()
    
    # Sample chunks about space exploration and technology
    chunks = [
        "NASA was established on July 29, 1958, succeeding the National Advisory Committee for Aeronautics (NACA).",
        "The Apollo 11 mission successfully landed humans on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin were the first astronauts to walk on the lunar surface.",
        "The International Space Station (ISS) is a modular space station in low Earth orbit. It has been continuously inhabited since November 2000.",
        "SpaceX was founded by Elon Musk in 2002 with the goal of reducing space transportation costs and enabling the colonization of Mars.",
        "The James Webb Space Telescope, launched in December 2021, is the most powerful space telescope ever built, designed to observe the universe in infrared.",
        "Mars rovers like Curiosity and Perseverance are exploring the Martian surface to search for signs of past microbial life and collect samples.",
        "The Voyager 1 spacecraft, launched in 1977, is the most distant human-made object from Earth and has entered interstellar space.",
        "Quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously, potentially solving problems intractable for classical computers.",
        "Artificial Intelligence has advanced rapidly with deep learning models like GPT and Claude that can understand and generate human-like text.",
        "CRISPR gene editing technology allows precise modifications to DNA sequences, opening new possibilities in medicine and biotechnology."
    ]
    
    # Add chunks to encoder
    encoder.add_chunks(chunks)
    
    # Build video memory file and search index
    # Using mp4v codec for fast encoding (can use h265 for better compression)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    encoder.build_video(
        output_file=f"{output_dir}/ollama_memory.mp4",
        index_file=f"{output_dir}/ollama_memory_index.json",
        codec="mp4v",  # Fast encoding, good compatibility
        show_progress=True
    )
    
    print("\n✓ Knowledge base created successfully!")
    print(f"  - Video: {output_dir}/ollama_memory.mp4")
    print(f"  - Index: {output_dir}/ollama_memory_index.json")
    print()


def interactive_chat_example():
    """
    Run an interactive chat session using Ollama for local inference.
    
    This demonstrates the main use case: conversational AI with retrieval
    from the video memory, powered by a local LLM model.
    """
    print("=" * 60)
    print("Ollama Interactive Chat Example")
    print("=" * 60)
    print()
    
    # Define paths to memory files
    video_file = "output/ollama_memory.mp4"
    index_file = "output/ollama_memory_index.json"
    
    # Check if memory exists, create if not
    if not os.path.exists(video_file):
        print("Memory not found. Creating sample knowledge base...")
        create_sample_memory()
    
    # Initialize chat with Ollama provider
    # No API key needed - Ollama runs locally!
    print("Initializing chat with Ollama...")
    print("(Make sure Ollama is running: 'ollama serve')")
    print()
    
    try:
        # Create chat instance with Ollama provider
        # model parameter can be changed to any model you have pulled
        chat = MemvidChat(
            video_file=video_file,
            index_file=index_file,
            llm_provider='ollama',
            llm_model='phi3:latest'  # Change to your preferred model
        )
        
        # Start interactive session
        # This will enter a loop where user can ask questions
        chat.interactive_chat()
        
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install Ollama: https://ollama.com/download")
        print("2. Pull a model: ollama pull llama3.2:3b")
        print("3. Start server: ollama serve")
        print("4. Check status: ollama list")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def quick_query_example():
    """
    Demonstrate a quick one-off query without interactive session.
    
    This is useful for programmatic access where you need a single answer
    without maintaining a conversation history.
    """
    print("=" * 60)
    print("Quick Query Example")
    print("=" * 60)
    print()
    
    # Define paths to memory files
    video_file = "output/ollama_memory.mp4"
    index_file = "output/ollama_memory_index.json"
    
    # Check if memory exists
    if not os.path.exists(video_file):
        print("Memory not found. Creating sample knowledge base...")
        create_sample_memory()
    
    try:
        # Initialize chat with Ollama
        chat = MemvidChat(
            video_file=video_file,
            index_file=index_file,
            llm_provider='ollama',
            llm_model='phi3:latest'
        )
        
        # Perform a single query
        query = "When did humans first land on the Moon?"
        print(f"Query: {query}")
        print("\nResponse:")
        print("-" * 60)
        
        # Get response (non-streaming)
        response = chat.chat(query, stream=False)
        print(response)
        print("-" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def list_available_models():
    """
    List all Ollama models available locally.
    
    This helps users see which models they have already pulled
    and can use with Memvid.
    """
    print("=" * 60)
    print("Available Ollama Models")
    print("=" * 60)
    print()
    
    try:
        import ollama
        
        # Get list of available models
        client = ollama.Client()
        models = client.list()
        
        if models and 'models' in models and models['models']:
            print("Models installed locally:")
            for model in models['models']:
                name = model.get('name', 'Unknown')
                size = model.get('size', 0) / (1024**3)  # Convert to GB
                print(f"  • {name} ({size:.2f} GB)")
        else:
            print("No models found. Pull a model with: ollama pull llama3.2:3b")
            
    except ImportError:
        print("❌ Ollama library not installed. Install with: pip install ollama")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("\nMake sure Ollama is running: ollama serve")
    
    print()


def main():
    """
    Main entry point with example menu.
    
    Provides different examples to demonstrate various use cases
    of Ollama integration with Memvid.
    """
    print("\n" + "=" * 60)
    print("Memvid + Ollama: Local LLM Integration Examples")
    print("=" * 60)
    print()
    print("Choose an example:")
    print("  1. Create sample knowledge base")
    print("  2. Interactive chat (recommended)")
    print("  3. Quick single query")
    print("  4. List available models")
    print("  5. Run all examples")
    print()
    
    choice = input("Enter choice (1-5): ").strip()
    print()
    
    if choice == "1":
        create_sample_memory()
    elif choice == "2":
        interactive_chat_example()
    elif choice == "3":
        quick_query_example()
    elif choice == "4":
        list_available_models()
    elif choice == "5":
        # Run all examples in sequence
        list_available_models()
        create_sample_memory()
        quick_query_example()
        print("\nStarting interactive chat...")
        input("Press Enter to continue...")
        interactive_chat_example()
    else:
        print("Invalid choice. Running interactive chat by default...")
        interactive_chat_example()


if __name__ == "__main__":
    main()

