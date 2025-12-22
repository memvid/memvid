"""Test that memvid encoding works without Unicode errors on Windows"""
import tempfile
from pathlib import Path
from memvid import MemvidEncoder

# Create test data
test_chunks = [
    "This is test chunk 1",
    "This is test chunk 2",
    "This is test chunk 3"
]

# Create encoder
encoder = MemvidEncoder(enable_docker=False)
encoder.add_chunks(test_chunks)

# Try to encode to video
with tempfile.TemporaryDirectory() as tmpdir:
    video_path = Path(tmpdir) / "test.mp4"
    index_path = Path(tmpdir) / "test.json"

    print("Encoding to video...")
    try:
        encoder.build_video(str(video_path), str(index_path))
        print(f"✓ SUCCESS: Video created at {video_path}")
        print(f"  Video size: {video_path.stat().st_size} bytes")
        print(f"  Index exists: {index_path.exists()}")
    except UnicodeEncodeError as e:
        print(f"✗ FAILED: Unicode error - {e}")
        raise
    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise

print("\n✓ Unicode fix verified - no encoding errors!")
