"""
Tests for MemvidEncoder
"""

import tempfile
import os

from memvid import MemvidEncoder


def test_encoder_initialization():
    """Test encoder initialization"""
    encoder = MemvidEncoder()
    assert encoder.chunks == []
    assert encoder.index_manager is not None


def test_add_chunks():
    """Test adding chunks"""
    encoder = MemvidEncoder()
    chunks = ["chunk1", "chunk2", "chunk3"]
    
    encoder.add_chunks(chunks)
    assert len(encoder.chunks) == 3
    assert encoder.chunks == chunks


def test_add_text():
    """Test adding text with auto-chunking"""
    encoder = MemvidEncoder()
    text = "This is a test. " * 50  # 800 characters
    
    encoder.add_text(text, chunk_size=100, overlap=20)
    assert len(encoder.chunks) > 1
    assert all(chunk for chunk in encoder.chunks)  # No empty chunks


def test_build_video():
    """Test video building (integration test)"""
    encoder = MemvidEncoder()
    chunks = [
        "Test chunk 1: Important information",
        "Test chunk 2: More data here",
        "Test chunk 3: Final piece of info"
    ]
    encoder.add_chunks(chunks)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        video_file = os.path.join(temp_dir, "test.mp4")
        index_file = os.path.join(temp_dir, "test_index.json")
        
        # Build video
        stats = encoder.build_video(video_file, index_file, show_progress=False)
        
        # Check files exist
        assert os.path.exists(video_file)
        assert os.path.exists(index_file)
        assert os.path.exists(index_file.replace('.json', '.faiss'))
        
        # Check stats
        assert stats["total_chunks"] == 3
        assert stats["total_frames"] == 3
        assert stats["video_size_mb"] > 0
        assert stats["duration_seconds"] > 0


def test_encoder_stats():
    """Test encoder statistics"""
    encoder = MemvidEncoder()
    chunks = ["short", "medium length chunk", "this is a longer chunk with more text"]
    encoder.add_chunks(chunks)
    
    stats = encoder.get_stats()
    assert stats["total_chunks"] == 3
    assert stats["total_characters"] == sum(len(c) for c in chunks)
    assert stats["avg_chunk_size"] > 0


def test_clear():
    """Test clearing encoder"""
    encoder = MemvidEncoder()
    encoder.add_chunks(["test1", "test2"])
    
    encoder.clear()
    assert encoder.chunks == []
    assert encoder.get_stats()["total_chunks"] == 0