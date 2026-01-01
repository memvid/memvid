"""
Tests for utility functions
"""

from PIL import Image
import tempfile
import os

from memvid.utils import (
    encode_to_qr, decode_qr, chunk_text, qr_to_frame,
    save_index, load_index
)


def test_qr_encode_decode():
    """Test QR code encoding and decoding"""
    test_data = "Hello, Memvid!"
    
    # Encode to QR
    qr_image = encode_to_qr(test_data)
    assert isinstance(qr_image, Image.Image)
    
    # Convert to frame
    frame = qr_to_frame(qr_image, (512, 512))
    assert frame.shape == (512, 512, 3)
    
    # Decode
    decoded = decode_qr(frame)
    assert decoded == test_data


def test_qr_encode_decode_large_data():
    """Test QR with compression for large data"""
    test_data = "x" * 1000  # Large data that will be compressed
    
    # Encode
    qr_image = encode_to_qr(test_data)
    frame = qr_to_frame(qr_image, (512, 512))
    
    # Decode
    decoded = decode_qr(frame)
    assert decoded == test_data


def test_chunk_text():
    """Test text chunking"""
    text = "This is a test. " * 50  # 800 characters
    
    # Test basic chunking
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)  # Allow for sentence boundaries
    
    # Test overlap
    for i in range(len(chunks) - 1):
        # Check that there's some overlap
        assert any(word in chunks[i+1] for word in chunks[i].split()[-5:])


def test_save_load_index():
    """Test index save and load"""
    test_data = {
        "metadata": [{"id": 1, "text": "test"}],
        "config": {"test": True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        # Save
        save_index(test_data, temp_file)
        assert os.path.exists(temp_file)
        
        # Load
        loaded_data = load_index(temp_file)
        assert loaded_data == test_data
    finally:
        os.unlink(temp_file)