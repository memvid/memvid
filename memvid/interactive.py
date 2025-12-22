"""
Interactive chat interface for Memvid
"""

import os
import time
from typing import Any, Dict, Optional

from .chat import MemvidChat


def chat_with_memory(
    video_file: str,
    index_file: str,
    api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    show_stats: bool = True,
    export_on_exit: bool = True,
    session_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Start an interactive chat session with a video memory.

    Args:
        video_file: Path to QR code video
        index_file: Path to index file
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        llm_model: LLM model to use (default: gpt-3.5-turbo)
        show_stats: Show memory stats on startup
        export_on_exit: Auto-export conversation on exit
        session_dir: Directory to save session files (default: "output")
        config: Optional configuration

    Commands:
        - 'search <query>': Show raw search results
        - 'stats': Show system statistics
        - 'export': Save conversation
        - 'clear': Clear conversation history
        - 'help': Show commands
        - 'exit' or 'quit': End session
    """
    # Set tokenizers parallelism to avoid warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set default session directory
    if session_dir is None:
        session_dir = "output"
    os.makedirs(session_dir, exist_ok=True)

    # Check if files exist
    if not os.path.exists(video_file):
        print(f"Error: Video file not found: {video_file}")
        return
    if not os.path.exists(index_file):
        print(f"Error: Index file not found: {index_file}")
        return

    # Initialize chat
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    print("Initializing Memvid Chat...")

    chat = MemvidChat(
        video_file,
        index_file,
        llm_api_key=api_key,
        llm_model=llm_model,
        config=config,
    )
    chat.start_session()

    # Show stats if requested
    if show_stats:
        stats = chat.get_stats()
        # print(
        #     f"\nMemory loaded: {stats['retriever_stats']['index_stats']['total_chunks']} chunks"
        # )
        if not stats["llm_available"]:
            #     print(f"LLM: {stats['llm_model']}")
            # else:
            print("LLM: Not available (context-only mode)")

    print("\nType 'help' for commands, 'exit' to quit")
    print("-" * 50)

    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
            lower_input = user_input.lower()

            if lower_input in ["exit", "quit", "q"]:
                break

            elif lower_input == "help":
                print("\nCommands:")
                print("  search <query> - Show raw search results")
                print("  stats         - Show system statistics")
                print("  export        - Save conversation")
                print("  clear         - Clear conversation history")
                print("  help          - Show this help")
                print("  exit/quit     - End session")
                continue

            elif lower_input == "stats":
                stats = chat.get_stats()
                print(f"\nMessages: {stats['message_count']}")
                print(f"Cache size: {stats['retriever_stats']['cache_size']}")
                print(f"Video frames: {stats['retriever_stats']['total_frames']}")
                continue

            elif lower_input == "export":
                export_file = os.path.join(
                    session_dir, f"memvid_session_{chat.session_id}.json"
                )
                chat.export_session(export_file)
                print(f"Exported to: {export_file}")
                continue

            elif lower_input == "clear":
                chat.reset_session()
                chat.start_session()
                print("Conversation cleared.")
                continue

            elif lower_input.startswith("search "):
                query = user_input[7:]
                print(f"\nSearching: '{query}'")
                start_time = time.time()
                results = chat.search_context(query, top_k=5)
                elapsed = time.time() - start_time
                print(f"Found {len(results)} results in {elapsed:.3f}s:\n")
                for i, result in enumerate(results[:3]):
                    print(
                        f"{i + 1}. [Score: {result['score']:.3f}] {result['text'][:100]}..."
                    )
                continue

            # Regular chat
            print("\nAssistant: ", end="", flush=True)
            start_time = time.time()
            response = chat.chat(user_input)
            elapsed = time.time() - start_time

            print(response)
            print(f"\n[{elapsed:.1f}s]", end="")

        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

    # Export on exit if requested
    if export_on_exit and chat.get_history():
        export_file = os.path.join(
            session_dir, f"memvid_session_{chat.session_id}.json"
        )
        chat.export_session(export_file)
        print(f"\nSession saved to: {export_file}")

    print("Goodbye!")


def quick_chat(
    video_file: str, index_file: str, query: str, api_key: Optional[str] = None
) -> str:
    """
    Quick one-off query without interactive loop.

    Args:
        video_file: Path to QR code video
        index_file: Path to index file
        query: Question to ask
        api_key: OpenAI API key (optional)

    Returns:
        Response string

    Example:
        >>> from memvid import quick_chat
        >>> response = quick_chat(f"knowledge.{VIDEO_FILE_TYPE}", "knowledge_index.json", "What is quantum computing?")
        >>> print(response)
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    chat = MemvidChat(video_file, index_file, llm_api_key=api_key)
    return chat.chat(query)
