"""
Index management for embeddings and vector search
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path

from .config import get_default_config

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages embeddings, FAISS index, and metadata for fast retrieval"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize IndexManager
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_default_config()
        self.embedding_model = SentenceTransformer(self.config["embedding"]["model"])
        self.dimension = self.config["embedding"]["dimension"]
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Metadata storage
        self.metadata = []
        self.chunk_to_frame = {}  # Maps chunk ID to frame number
        self.frame_to_chunks = {}  # Maps frame number to chunk IDs
        
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration"""
        index_type = self.config["index"]["type"]
        
        if index_type == "Flat":
            # Exact search - best quality, slower for large datasets
            index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IVF":
            # Inverted file index - faster for large datasets
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.config["index"]["nlist"])
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
        # Add ID mapping for retrieval
        index = faiss.IndexIDMap(index)
        return index

    def add_chunks(self, chunks: List[str], frame_numbers: List[int],
                   show_progress: bool = True) -> List[int]:
        """
        Add chunks to index with robust error handling and validation

        Args:
            chunks: List of text chunks
            frame_numbers: Corresponding frame numbers for each chunk
            show_progress: Show progress bar

        Returns:
            List of successfully added chunk IDs
        """
        if len(chunks) != len(frame_numbers):
            raise ValueError("Number of chunks must match number of frame numbers")

        logger.info(f"Processing {len(chunks)} chunks for indexing...")

        # Phase 1: Validate and filter chunks
        valid_chunks = []
        valid_frames = []
        skipped_count = 0

        for chunk, frame_num in zip(chunks, frame_numbers):
            if self._is_valid_chunk(chunk):
                valid_chunks.append(chunk)
                valid_frames.append(frame_num)
            else:
                skipped_count += 1
                logger.warning(f"Skipping invalid chunk at frame {frame_num}: length={len(chunk) if chunk else 0}")

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} invalid chunks out of {len(chunks)} total")

        if not valid_chunks:
            logger.error("No valid chunks to process")
            return []

        logger.info(f"Processing {len(valid_chunks)} valid chunks")

        # Phase 2: Generate embeddings with batch processing and error recovery
        try:
            embeddings = self._generate_embeddings(valid_chunks, show_progress)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []

        if embeddings is None or len(embeddings) == 0:
            logger.error("No embeddings generated")
            return []

        # Phase 3: Add to FAISS index
        try:
            chunk_ids = self._add_to_index(embeddings, valid_chunks, valid_frames)
            logger.info(f"Successfully added {len(chunk_ids)} chunks to index")
            return chunk_ids
        except Exception as e:
            logger.error(f"Failed to add chunks to index: {e}")
            return []

    def _is_valid_chunk(self, chunk: str) -> bool:
        """Validate chunk for SentenceTransformer processing - SIMPLIFIED"""
        if not isinstance(chunk, str):
            return False

        chunk = chunk.strip()

        # Basic checks only
        if len(chunk) == 0:
            return False

        if len(chunk) > 8192:  # SentenceTransformer limit
            return False

        # Remove the harsh alphanumeric requirement - academic text has lots of punctuation!
        # Just ensure it's not binary data
        try:
            chunk.encode('utf-8')  # Can be encoded as UTF-8
            return True
        except UnicodeEncodeError:
            return False

    def _generate_embeddings(self, chunks: List[str], show_progress: bool) -> np.ndarray:
        """Generate embeddings with error handling and batch processing"""

        # Try full batch first
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks (full batch)")
            embeddings = self.embedding_model.encode(
                chunks,
                show_progress_bar=show_progress,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True  # Helps with numerical stability
            )
            return np.array(embeddings).astype('float32')

        except Exception as e:
            logger.warning(f"Full batch embedding failed: {e}. Trying batch processing...")

            # Fall back to smaller batches
            return self._generate_embeddings_batched(chunks, show_progress)

    def _generate_embeddings_batched(self, chunks: List[str], show_progress: bool) -> np.ndarray:
        """Generate embeddings in smaller batches with individual error handling"""

        all_embeddings = []
        valid_chunks = []
        batch_size = 100  # Smaller batches

        total_batches = (len(chunks) + batch_size - 1) // batch_size

        if show_progress:
            from tqdm import tqdm
            batch_iter = tqdm(range(0, len(chunks), batch_size),
                              desc="Processing chunks in batches",
                              total=total_batches)
        else:
            batch_iter = range(0, len(chunks), batch_size)

        for i in batch_iter:
            batch_chunks = chunks[i:i + batch_size]

            try:
                # Try batch
                batch_embeddings = self.embedding_model.encode(
                    batch_chunks,
                    show_progress_bar=False,
                    batch_size=16,  # Even smaller internal batch
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                all_embeddings.extend(batch_embeddings)
                valid_chunks.extend(batch_chunks)

            except Exception as e:
                logger.warning(f"Batch {i//batch_size} failed: {e}. Processing individually...")

                # Process individually
                for chunk in batch_chunks:
                    try:
                        embedding = self.embedding_model.encode(
                            [chunk],
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        all_embeddings.extend(embedding)
                        valid_chunks.append(chunk)

                    except Exception as chunk_error:
                        logger.error(f"Failed to embed individual chunk (length={len(chunk)}): {chunk_error}")
                        # Skip this chunk entirely
                        continue

        if not all_embeddings:
            raise RuntimeError("No embeddings could be generated")

        logger.info(f"Generated embeddings for {len(valid_chunks)} out of {len(chunks)} chunks")
        return np.array(all_embeddings).astype('float32')

    def _add_to_index(self, embeddings: np.ndarray, chunks: List[str], frame_numbers: List[int]) -> List[int]:
        """Add embeddings to FAISS index with error handling"""

        if len(embeddings) != len(chunks) or len(embeddings) != len(frame_numbers):
            # This can happen if some chunks were skipped during embedding
            min_len = min(len(embeddings), len(chunks), len(frame_numbers))
            embeddings = embeddings[:min_len]
            chunks = chunks[:min_len]
            frame_numbers = frame_numbers[:min_len]
            logger.warning(f"Trimmed to {min_len} items due to length mismatch")

        # Assign IDs
        start_id = len(self.metadata)
        chunk_ids = list(range(start_id, start_id + len(chunks)))

        # Train index if needed (for IVF)
        try:
            underlying_index = self.index.index  # Get the actual index from IndexIDMap wrapper

            if isinstance(underlying_index, faiss.IndexIVFFlat):
                nlist = underlying_index.nlist

                if not underlying_index.is_trained:
                    logger.info(f"🧠 FAISS IVF index requires training (nlist={nlist})")
                    logger.info(f"📊 Available embeddings: {len(embeddings)}")

                    # Check if we have enough data for training
                    if len(embeddings) < nlist:
                        logger.warning(f"❌ Insufficient training data: need at least {nlist} embeddings, got {len(embeddings)}")
                        logger.warning("💡 IVF indexes require more data. For single documents, consider using 'Flat' index type in config.")
                        logger.info("🔄 Auto-switching to IndexFlatL2 for reliable operation")
                        # Replace with flat index
                        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                        logger.info("✅ Switched to Flat index (exact search, slower but works with any dataset size)")
                    else:
                        recommended_min = nlist * 10  # IVF works better with 10x+ the nlist size
                        if len(embeddings) < recommended_min:
                            logger.warning(f"⚠️ Suboptimal training data: {len(embeddings)} embeddings (recommended: {recommended_min}+)")
                            logger.warning("💡 Consider using larger dataset or 'Flat' index for better results")

                        logger.info("🏋️ Training FAISS IVF index...")
                        logger.info(f"   - Training vectors: {len(embeddings)}")
                        logger.info(f"   - Clusters (nlist): {nlist}")
                        logger.info(f"   - Expected memory: ~{(len(embeddings) * self.dimension * 4) / 1024 / 1024:.1f} MB")

                        # Use sufficient training data
                        training_data = embeddings[:min(50000, len(embeddings))]
                        underlying_index.train(training_data)
                        logger.info("✅ FAISS IVF training completed successfully")
                else:
                    logger.info(f"✅ FAISS IVF index already trained (nlist={nlist})")
            else:
                logger.info(f"ℹ️ Using {type(underlying_index).__name__} (no training required)")

        except Exception as e:
            logger.error(f"❌ Index training failed with error: {e}")
            logger.error(f"🔍 Error type: {type(e).__name__}")
            logger.info("🔄 Falling back to IndexFlatL2 for reliability")
            logger.info("💡 To avoid this fallback, use 'Flat' index type in config for small datasets")
            # Fallback to simple flat index
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
            logger.info("✅ Fallback complete - using exact search")

        # Add to index
        try:
            self.index.add_with_ids(embeddings, np.array(chunk_ids, dtype=np.int64))
        except Exception as e:
            logger.error(f"Failed to add embeddings to FAISS index: {e}")
            raise

        # Store metadata
        for i, (chunk, frame_num, chunk_id) in enumerate(zip(chunks, frame_numbers, chunk_ids)):
            try:
                metadata = {
                    "id": chunk_id,
                    "text": chunk,
                    "frame": frame_num,
                    "length": len(chunk)
                }
                self.metadata.append(metadata)

                # Update mappings
                self.chunk_to_frame[chunk_id] = frame_num
                if frame_num not in self.frame_to_chunks:
                    self.frame_to_chunks[frame_num] = []
                self.frame_to_chunks[frame_num].append(chunk_id)

            except Exception as e:
                logger.error(f"Failed to store metadata for chunk {chunk_id}: {e}")
                # Continue with other chunks
                continue

        return chunk_ids
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search for similar chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, distance, metadata) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Gather results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:  # Valid result
                metadata = self.metadata[idx]
                results.append((idx, float(dist), metadata))
        
        return results
    
    def get_chunks_by_frame(self, frame_number: int) -> List[Dict[str, Any]]:
        """Get all chunks associated with a frame"""
        chunk_ids = self.frame_to_chunks.get(frame_number, [])
        return [self.metadata[chunk_id] for chunk_id in chunk_ids]
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get chunk metadata by ID"""
        if 0 <= chunk_id < len(self.metadata):
            return self.metadata[chunk_id]
        return None
    
    def save(self, path: str):
        """
        Save index to disk
        
        Args:
            path: Path to save index (without extension)
        """
        path = Path(path)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path.with_suffix('.faiss')))
        
        # Save metadata and mappings
        data = {
            "metadata": self.metadata,
            "chunk_to_frame": self.chunk_to_frame,
            "frame_to_chunks": self.frame_to_chunks,
            "config": self.config
        }
        
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved index to {path}")
    
    def load(self, path: str):
        """
        Load index from disk
        
        Args:
            path: Path to load index from (without extension)
        """
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path.with_suffix('.faiss')))
        
        # Load metadata and mappings
        with open(path.with_suffix('.json'), 'r') as f:
            data = json.load(f)
        
        self.metadata = data["metadata"]
        self.chunk_to_frame = {int(k): v for k, v in data["chunk_to_frame"].items()}
        self.frame_to_chunks = {int(k): v for k, v in data["frame_to_chunks"].items()}
        
        # Update config if available
        if "config" in data:
            self.config.update(data["config"])
        
        logger.info(f"Loaded index from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_chunks": len(self.metadata),
            "total_frames": len(self.frame_to_chunks),
            "index_type": self.config["index"]["type"],
            "embedding_model": self.config["embedding"]["model"],
            "dimension": self.dimension,
            "avg_chunks_per_frame": np.mean([len(chunks) for chunks in self.frame_to_chunks.values()]) if self.frame_to_chunks else 0
        }