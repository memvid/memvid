"""
MemvidEncoder - Unified encoding with native OpenCV and FFmpeg (Docker/native) support
"""

import json
import logging
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import cv2
import numpy as np

from .utils import encode_to_qr, qr_to_frame, chunk_text
from .index import IndexManager
from .config import get_default_config, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, VIDEO_CODEC, get_codec_parameters
from .docker_manager import DockerManager

logger = logging.getLogger(__name__)

class MemvidEncoder:
    """
    Unified MemvidEncoder with clean separation between encoding logic and Docker management.
    Supports both native OpenCV encoding and FFmpeg encoding (native or Docker-based).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_docker=True):
        self.config = config or get_default_config()
        self.chunks = []
        self.index_manager = IndexManager()

        # Docker backend (optional)
        self.dcker_mngr = DockerManager() if enable_docker else None

        if self.dcker_mngr and not self.dcker_mngr.is_available():
            logger.info("Docker backend not available - using native encoding only")

    def add_chunks(self, chunks: List[str]):
        """
        Add text chunks to be encoded

        Args:
            chunks: List of text chunks
        """
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks. Total: {len(self.chunks)}")

    def add_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        """
        Add text and automatically chunk it

        Args:
            text: Text to chunk and add
            chunk_size: Target chunk size
            overlap: Overlap between chunks
        """
        chunks = chunk_text(text, chunk_size, overlap)
        self.add_chunks(chunks)

    def add_pdf(self, pdf_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        """
        Extract text from PDF and add as chunks

        Args:
            pdf_path: Path to PDF file
            chunk_size: Target chunk size
            overlap: Overlap between chunks
        """
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")

        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            logger.info(f"Extracting text from {num_pages} pages of {Path(pdf_path).name}")

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n\n"

        if text.strip():
            self.add_text(text, chunk_size, overlap)
            logger.info(f"Added PDF content: {len(text)} characters from {Path(pdf_path).name}")
        else:
            logger.warning(f"No text extracted from PDF: {pdf_path}")

    def add_epub(self, epub_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        """
        Extract text from EPUB and add as chunks

        Args:
            epub_path: Path to EPUB file
            chunk_size: Target chunk size
            overlap: Overlap between chunks
        """
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("ebooklib and beautifulsoup4 are required for EPUB support. Install with: pip install ebooklib beautifulsoup4")

        if not Path(epub_path).exists():
            raise FileNotFoundError(f"EPUB file not found: {epub_path}")

        try:
            book = epub.read_epub(epub_path)
            text_content = []

            logger.info(f"Extracting text from EPUB: {Path(epub_path).name}")

            # Extract text from all document items
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Parse HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Get text and clean it up
                    text = soup.get_text()

                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)

                    if text.strip():
                        text_content.append(text)

            # Combine all text
            full_text = "\n\n".join(text_content)

            if full_text.strip():
                self.add_text(full_text, chunk_size, overlap)
                logger.info(f"Added EPUB content: {len(full_text)} characters from {Path(epub_path).name}")
            else:
                logger.warning(f"No text extracted from EPUB: {epub_path}")

        except Exception as e:
            logger.error(f"Error processing EPUB {epub_path}: {e}")
            raise

    def create_video_writer(self, output_path: str, codec: str = VIDEO_CODEC) -> cv2.VideoWriter:
        """
        Create OpenCV video writer for native encoding

        Args:
            output_path: Path to output video file
            codec: Video codec for OpenCV

        Returns:
            cv2.VideoWriter instance
        """
        from .config import codec_parameters

        if codec not in codec_parameters:  # FIXED
            raise ValueError(f"Unsupported codec: {codec}")

        codec_config = codec_parameters[codec]  # FIXED

        # OpenCV codec mapping
        opencv_codec_map = {
            "mp4v": "mp4v",
            "xvid": "XVID",
            "mjpg": "MJPG"
        }

        opencv_codec = opencv_codec_map.get(codec, codec)
        fourcc = cv2.VideoWriter_fourcc(*opencv_codec)

        return cv2.VideoWriter(
            output_path,
            fourcc,
            codec_config["video_fps"],
            (codec_config["frame_width"], codec_config["frame_height"])
        )

    def _generate_qr_frames(self, temp_dir: Path, show_progress: bool = True) -> Path:
        """
        Generate QR code frames to temporary directory
        
        Args:
            temp_dir: Temporary directory for frame storage
            show_progress: Show progress bar
            
        Returns:
            Path to frames directory
        """
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()

        chunks_iter = enumerate(self.chunks)
        if show_progress:
            chunks_iter = tqdm(chunks_iter, total=len(self.chunks), desc="Generating QR frames")

        for frame_num, chunk in chunks_iter:
            chunk_data = {"id": frame_num, "text": chunk, "frame": frame_num}
            qr_image = encode_to_qr(json.dumps(chunk_data))
            frame_path = frames_dir / f"frame_{frame_num:06d}.png"
            qr_image.save(frame_path)

        created_frames = list(frames_dir.glob("frame_*.png"))
        print(f"🐛 FRAMES: {len(created_frames)} files in {frames_dir}")

        logger.info(f"Generated {len(self.chunks)} QR frames in {frames_dir}")
        return frames_dir

    def _build_ffmpeg_command(self, frames_dir: Path, output_file: Path, codec: str) -> List[str]:
        """Build optimized FFmpeg command using codec configuration"""

        # Get codec-specific configuration
        codec_config = get_codec_parameters(codec.lower())

        # FFmpeg codec mapping
        ffmpeg_codec_map = {
            "h265": "libx265", "hevc": "libx265",
            "h264": "libx264", "avc": "libx264",
            "av1": "libaom-av1", "vp9": "libvpx-vp9"
        }

        ffmpeg_codec = ffmpeg_codec_map.get(codec, codec)

        # Ensure output file has correct extension
        expected_ext = codec_config["video_file_type"]
        if not str(output_file).endswith(expected_ext):
            output_file = output_file.with_suffix(expected_ext)

        # Build base command using config
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(codec_config["video_fps"]),
            '-i', str(frames_dir / 'frame_%06d.png'),
            '-c:v', ffmpeg_codec,
            '-preset', codec_config["video_preset"],
            '-crf', str(codec_config["video_crf"]),
        ]

        # Apply scaling and pixel format based on codec
        if ffmpeg_codec in ['libx265', 'libx264']:
            # Scale to config dimensions for advanced codecs
            target_width = codec_config["frame_width"]
            target_height = codec_config["frame_height"]
            cmd.extend(['-vf', f'scale={target_width}:{target_height}'])
            cmd.extend(['-pix_fmt', codec_config["pix_fmt"]])

            # Add profile if specified in config
            if codec_config.get("video_profile"):
                cmd.extend(['-profile:v', codec_config["video_profile"]])
        else:
            # Use pixel format from config for other codecs
            cmd.extend(['-pix_fmt', codec_config["pix_fmt"]])

        # Threading (limit to 16 max)
        import os
        thread_count = min(os.cpu_count() or 4, 16)
        cmd.extend(['-threads', str(thread_count)])

        print(f"🎬 FFMPEG ENCODING SUMMARY:")
        print(f"   🎥 Codec Config:")
        print(f"      • codec: {codec}")
        print(f"      • file_type: {codec_config.get('video_file_type', 'unknown')}")
        print(f"      • fps: {codec_config.get('fps', 'default')}")
        print(f"      • crf: {codec_config.get('crf', 'default')}")
        print(f"      • height: {codec_config.get('frame_height', 'default')}")
        print(f"      • width: {codec_config.get('frame_width', 'default')}")
        print(f"      • preset: {codec_config.get('video_preset', 'default')}")
        print(f"      • pix_fmt: {codec_config.get('pix_fmt', 'default')}")
        print(f"      • extra_ffmpeg_args: {codec_config.get('extra_ffmpeg_args', 'default')}")

        # Add codec-specific parameters from config
        if codec_config.get("extra_ffmpeg_args"):
            extra_args = codec_config["extra_ffmpeg_args"]
            if isinstance(extra_args, str):
                # Parse string args and add thread count for x264/x265
                if ffmpeg_codec == 'libx265':
                    extra_args = f"{extra_args}:threads={thread_count}"
                    cmd.extend(['-x265-params', extra_args])
                elif ffmpeg_codec == 'libx264':
                    extra_args = f"{extra_args}:threads={thread_count}"
                    cmd.extend(['-x264-params', extra_args])
            else:
                # Direct args list
                cmd.extend(extra_args)

        # General optimizations
        cmd.extend(['-movflags', '+faststart', '-avoid_negative_ts', 'make_zero'])

        cmd.append(str(output_file))
        return cmd

    def _encode_with_opencv(self, frames_dir: Path, output_file: Path, codec: str,
                            show_progress: bool = True) -> Dict[str, Any]:
        """
        Encode video using native OpenCV
        
        Args:
            frames_dir: Directory containing PNG frames
            output_file: Output video file path
            codec: Video codec
            show_progress: Show progress bar
            
        Returns:
            Encoding statistics
        """
        from .config import codec_parameters

        if codec not in codec_parameters:
            raise ValueError(f"Unsupported codec: {codec}")

        codec_config = codec_parameters[codec]  # FIXED: Get specific codec config

        if show_progress:
            logger.info(f"Encoding with OpenCV using {codec} codec...")

        # Create video writer
        writer = self.create_video_writer(str(output_file), codec)
        frame_numbers = []

        try:
            # Load and write frames
            frame_files = sorted(frames_dir.glob("frame_*.png"))
            frame_iter = enumerate(frame_files)

            if show_progress:
                frame_iter = tqdm(frame_iter, total=len(frame_files), desc="Writing video frames")

            for frame_num, frame_file in frame_iter:
                # Load frame
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    logger.warning(f"Could not load frame: {frame_file}")
                    continue

                # Resize if needed
                target_size = (codec_config["frame_width"], codec_config["frame_height"])
                if frame.shape[:2][::-1] != target_size:
                    frame = cv2.resize(frame, target_size)

                # Write frame
                writer.write(frame)
                frame_numbers.append(frame_num)

            return {
                "backend": "opencv",
                "codec": codec,
                "total_frames": len(frame_numbers),
                "video_size_mb": output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0,
                "fps": codec_config["video_fps"],
                "duration_seconds": len(frame_numbers) / codec_config["video_fps"]
            }

        finally:
            writer.release()

    def _encode_with_ffmpeg(self, frames_dir: Path, output_file: Path, codec: str,
                            show_progress: bool = True, auto_build_docker: bool = True) -> Dict[str, Any]:
        """
        Encode video using FFmpeg (native or Docker)
        
        Args:
            frames_dir: Directory containing PNG frames
            output_file: Output video file path
            codec: Video codec
            show_progress: Show progress bar
            auto_build_docker: Whether to auto-build Docker container if needed
            
        Returns:
            Encoding statistics
        """
        # Use full codec mapping
        from .config import codec_parameters

        print(f"🐛 FFMPEG: frames={frames_dir} → docker_mount={frames_dir.parent}")

        cmd = self._build_ffmpeg_command(frames_dir, output_file, codec)

        if self.dcker_mngr and self.dcker_mngr.should_use_docker(codec):
            if show_progress:
                logger.info(f"Encoding with Docker FFmpeg using {codec} codec...")

            result = self.dcker_mngr.execute_ffmpeg(
                cmd, frames_dir.parent, output_file, auto_build=auto_build_docker
            )

            frame_count = len(list(frames_dir.glob("frame_*.png")))
            result.update({
                "codec": codec,
                "total_frames": frame_count,
                "fps": codec_parameters[codec]["video_fps"],
                "duration_seconds": frame_count / codec_parameters[codec]["video_fps"]
            })

            return result

        else:
            if show_progress:
                logger.info(f"Encoding with native FFmpeg using {codec} codec...")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Native FFmpeg failed: {result.stderr}")

            frame_count = len(list(frames_dir.glob("frame_*.png")))
            return {
                "backend": "native_ffmpeg",
                "codec": codec,
                "total_frames": frame_count,
                "video_size_mb": output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0,
                "fps": codec_parameters[codec]["video_fps"],
                "duration_seconds": frame_count / codec_parameters[codec]["video_fps"]
            }

    def _validate_frame_ordering_config(self, config: Optional[Dict[str, Any]]):
        """
        Validate frame ordering configuration parameters
        
        Args:
            config: Frame ordering configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        if config is None:
            return  # Default config will be used
        
        # Validate power_base
        if "power_base" in config:
            power_base = config["power_base"]
            if not isinstance(power_base, int) or power_base < 2:
                raise ValueError(f"power_base must be an integer >= 2, got: {power_base}")
        
        # Validate max_resolution
        if "max_resolution" in config:
            max_resolution = config["max_resolution"]
            if not isinstance(max_resolution, int) or max_resolution < 1:
                raise ValueError(f"max_resolution must be a positive integer, got: {max_resolution}")
        
        # Validate start_with_1x1
        if "start_with_1x1" in config:
            start_with_1x1 = config["start_with_1x1"]
            if not isinstance(start_with_1x1, bool):
                raise ValueError(f"start_with_1x1 must be a boolean, got: {type(start_with_1x1)}")

    def _apply_frame_ordering(self, frames_dir: Path, config: Optional[Dict[str, Any]], 
                             show_progress: bool = True) -> Dict[str, Any]:
        """
        Apply frame ordering optimization to QR frames for better video compression
        
        Args:
            frames_dir: Directory containing generated QR frames
            config: Frame ordering configuration options
            show_progress: Show progress indicator
            
        Returns:
            Dictionary with frame ordering metadata
        """
        import time
        from .frame_ordering import FrameOrderingOptimizer
        
        if show_progress:
            logger.info("Applying frame ordering optimization...")
        
        start_time = time.time()
        
        try:
            # Set up frame ordering configuration
            ordering_config = config or {}
            optimizer = FrameOrderingOptimizer(
                power_base=ordering_config.get("power_base", 2),
                max_resolution=ordering_config.get("max_resolution", 32),
                start_with_1x1=ordering_config.get("start_with_1x1", True)
            )
            
            # Load all QR frames for analysis
            frame_files = sorted(frames_dir.glob("frame_*.png"))
            if not frame_files:
                raise ValueError("No QR frames found for ordering")
            
            frames = []
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    logger.warning(f"Could not load frame: {frame_file}")
                    continue
                
                # Convert to grayscale for frame ordering analysis
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray_frame)
            
            if len(frames) != len(frame_files):
                logger.warning(f"Frame loading mismatch: {len(frames)} loaded vs {len(frame_files)} files")
            
            # Optimize frame order
            optimized_indices, optimization_metadata = optimizer.optimize_frame_order(frames)
            
            # Reorder the actual frame files
            self._reorder_frame_files(frames_dir, optimized_indices)
            
            optimization_time = time.time() - start_time
            
            # Create frame ordering metadata
            frame_ordering_metadata = {
                "optimized_order": optimized_indices,
                "original_order": list(range(len(frames))),
                "optimization_config": {
                    "power_base": optimizer.power_base,
                    "max_resolution": optimizer.max_resolution,
                    "start_with_1x1": optimizer.start_with_1x1
                },
                "optimization_time": optimization_time,
                "frame_count": len(frames)
            }
            frame_ordering_metadata.update(optimization_metadata)
            
            if show_progress:
                logger.info(f"Frame ordering completed in {optimization_time:.2f}s")
                logger.info(f"Reordered {len(frames)} frames using resolution sequence: {optimization_metadata.get('resolution_sequence', [])}")
            
            return frame_ordering_metadata
            
        except Exception as e:
            optimization_time = time.time() - start_time
            logger.warning(f"Frame ordering failed: {e}")
            
            # Return error metadata for graceful degradation
            return {
                "error": str(e),
                "optimization_time": optimization_time,
                "original_order": list(range(len(self.chunks))),
                "optimized_order": list(range(len(self.chunks)))  # No reordering due to error
            }

    def _reorder_frame_files(self, frames_dir: Path, optimized_indices: List[int]):
        """
        Reorder frame files according to optimized indices
        
        Args:
            frames_dir: Directory containing frame files
            optimized_indices: List of indices in optimized order
        """
        import shutil
        
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        
        if len(optimized_indices) != len(frame_files):
            raise ValueError(f"Index count mismatch: {len(optimized_indices)} vs {len(frame_files)}")
        
        # Create temporary directory for reordering
        temp_reorder_dir = frames_dir.parent / "frames_reordered"
        temp_reorder_dir.mkdir()
        
        try:
            # Copy files in optimized order
            for new_position, original_index in enumerate(optimized_indices):
                original_file = frame_files[original_index]
                new_file = temp_reorder_dir / f"frame_{new_position:06d}.png"
                shutil.copy2(original_file, new_file)
            
            # Remove original frames and replace with reordered ones
            for frame_file in frame_files:
                frame_file.unlink()
            
            # Move reordered frames back
            for reordered_file in temp_reorder_dir.glob("frame_*.png"):
                final_file = frames_dir / reordered_file.name
                shutil.move(str(reordered_file), str(final_file))
                
        finally:
            # Clean up temporary directory
            if temp_reorder_dir.exists():
                shutil.rmtree(temp_reorder_dir)

    def build_video(self, output_file: str, index_file: str,
                    codec: str = VIDEO_CODEC, show_progress: bool = True,
                    auto_build_docker: bool = True, allow_fallback: bool = True,
                    enable_frame_ordering: bool = False,
                    frame_ordering_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build QR code video and index from chunks with unified codec handling

        Args:
            output_file: Path to output video file
            index_file: Path to output index file
            codec: Video codec ('mp4v', 'h265', 'h264', etc.)
            show_progress: Show progress bar
            auto_build_docker: Whether to auto-build Docker if needed
            allow_fallback: Whether to fall back to MP4V if advanced codec fails
            enable_frame_ordering: Enable frame ordering optimization for better compression
            frame_ordering_config: Configuration for frame ordering (power_base, max_resolution, etc.)

        Returns:
            Dictionary with build statistics
        """
        if not self.chunks:
            raise ValueError("No chunks to encode. Use add_chunks() first.")

        output_path = Path(output_file)
        index_path = Path(index_file)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Building video with {len(self.chunks)} chunks using {codec} codec")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Generate QR frames (always local)
            frames_dir = self._generate_qr_frames(temp_path, show_progress)

            # Apply frame ordering optimization if enabled
            frame_ordering_metadata = None
            if enable_frame_ordering:
                # Validate frame ordering configuration upfront
                self._validate_frame_ordering_config(frame_ordering_config)
                
                frame_ordering_metadata = self._apply_frame_ordering(
                    frames_dir, frame_ordering_config, show_progress
                )

            try:
                from .config import codec_parameters
                # Choose encoding method based on codec
                if codec == "mp4v":
                    # Always use OpenCV for MP4V
                    stats = self._encode_with_opencv(frames_dir, output_path, codec, show_progress)
                else:
                    # Use FFmpeg for advanced codecs
                    stats = self._encode_with_ffmpeg(frames_dir, output_path, codec,
                                                     show_progress, auto_build_docker)

            except Exception as e:
                if allow_fallback and codec != "mp4v":
                    warnings.warn(f"{codec} encoding failed: {e}. Falling back to MP4V.", UserWarning)
                    stats = self._encode_with_opencv(frames_dir, output_path, "mp4v", show_progress)
                else:
                    raise

            # Build search index
            if show_progress:
                logger.info("Building search index...")

            frame_numbers = list(range(len(self.chunks)))
            self.index_manager.add_chunks(self.chunks, frame_numbers, show_progress)

            # Add frame order mapping to index if frame ordering was enabled
            if frame_ordering_metadata and "optimized_order" in frame_ordering_metadata:
                self._add_frame_order_mapping_to_index(frame_ordering_metadata)

            # Save index
            self.index_manager.save(str(index_path.with_suffix('')))

            # Finalize statistics
            stats.update({
                "total_chunks": len(self.chunks),
                "video_file": str(output_path),
                "index_file": str(index_path),
                "index_stats": self.index_manager.get_stats()
            })

            # Add frame ordering metadata if enabled
            if frame_ordering_metadata:
                stats["frame_ordering"] = frame_ordering_metadata

            if show_progress:
                logger.info(f"Successfully built video: {output_path}")
                logger.info(f"Video duration: {stats.get('duration_seconds', 0):.1f} seconds")
                logger.info(f"Video size: {stats.get('video_size_mb', 0):.1f} MB")

            return stats

    def clear(self):
        """Clear all chunks"""
        self.chunks = []
        self.index_manager = IndexManager()
        logger.info("Cleared all chunks")

    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics"""
        from .config import codec_parameters
        
        docker_status = "disabled"
        if self.dcker_mngr:
            docker_status = "available" if self.dcker_mngr.is_available() else "unavailable"

        return {
            "total_chunks": len(self.chunks),
            "total_characters": sum(len(chunk) for chunk in self.chunks),
            "avg_chunk_size": np.mean([len(chunk) for chunk in self.chunks]) if self.chunks else 0,
            "docker_status": docker_status,
            "supported_codecs": list(codec_parameters.keys()),
            "config": self.config
        }

    def get_docker_status(self) -> str:
        """Get Docker backend status message"""
        if not self.dcker_mngr:
            return "Docker backend disabled"
        return self.dcker_mngr.get_status_message()

    @classmethod
    def from_file(cls, file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP,
                  config: Optional[Dict[str, Any]] = None) -> 'MemvidEncoder':
        """
        Create encoder from text file

        Args:
            file_path: Path to text file
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            config: Optional configuration

        Returns:
            MemvidEncoder instance with chunks loaded
        """
        encoder = cls(config)

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        encoder.add_text(text, chunk_size, overlap)
        return encoder

    @classmethod
    def from_documents(cls, documents: List[str], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP,
                       config: Optional[Dict[str, Any]] = None) -> 'MemvidEncoder':
        """
        Create encoder from list of documents

        Args:
            documents: List of document strings
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            config: Optional configuration

        Returns:
            MemvidEncoder instance with chunks loaded
        """
        encoder = cls(config)

        for doc in documents:
            encoder.add_text(doc, chunk_size, overlap)

        return encoder

    def _add_frame_order_mapping_to_index(self, frame_ordering_metadata: Dict[str, Any]):
        """
        Add frame order mapping to index
        
        Args:
            frame_ordering_metadata: Dictionary with frame ordering metadata
        """
        optimized_order = frame_ordering_metadata["optimized_order"]
        
        self.index_manager.add_frame_order_mapping(optimized_order)