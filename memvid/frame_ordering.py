#!/usr/bin/env python3
"""
Frame ordering optimization for video compression.

Implements progressive resolution sorting to minimize frame-to-frame differences
in QR code videos, improving compression efficiency.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


class ResolutionSignature:
    """Represents a QR frame signature at a specific resolution"""
    
    def __init__(self, resolution: int, data: np.ndarray):
        """
        Initialize a resolution signature.
        
        Args:
            resolution: The resolution this signature represents (e.g., 2 for 2x2)
            data: Flattened array of pixel values at this resolution
        """
        self.resolution = resolution
        self.data = data.flatten() if data.ndim > 1 else data
    
    def __lt__(self, other) -> bool:
        """Compare signatures for sorting (lexicographic comparison of data)"""
        if not isinstance(other, ResolutionSignature):
            return NotImplemented
        
        # Compare by resolution first, then by data lexicographically
        if self.resolution != other.resolution:
            return self.resolution < other.resolution
        
        # Lexicographic comparison of data arrays
        for a, b in zip(self.data, other.data):
            if a < b:
                return True
            elif a > b:
                return False
        return len(self.data) < len(other.data)
    
    def __eq__(self, other) -> bool:
        """Check equality of signatures"""
        if not isinstance(other, ResolutionSignature):
            return NotImplemented
        
        return (self.resolution == other.resolution and 
                np.array_equal(self.data, other.data))
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"ResolutionSignature(resolution={self.resolution}, data_shape={self.data.shape})"


class FrameOrderingOptimizer:
    """
    Optimizes the ordering of QR frames to minimize frame-to-frame differences.
    
    Uses progressive resolution sorting: starts with low-resolution signatures (1x1, 2x2)
    and progressively refines with higher resolutions (4x4, 8x8, 16x16, etc.).
    """
    
    def __init__(self, 
                 power_base: int = 2, 
                 max_resolution: int = 32, 
                 start_with_1x1: bool = True,
                 early_stopping_enabled: bool = True,
                 early_stopping_threshold: float = 0.02):
        """
        Initialize the frame ordering optimizer.
        
        Args:
            power_base: Base for resolution progression (2: 1→2→4→8, 3: 1→3→9→27)
            max_resolution: Maximum resolution to use (32, 256, or 477 for full)
            start_with_1x1: Whether to start with 1x1 global brightness sort
            early_stopping_enabled: Whether to enable early stopping optimization
            early_stopping_threshold: Minimum improvement threshold to continue (0.0-1.0)
        """
        # Parameter validation
        if power_base < 2:
            raise ValueError("power_base must be >= 2")
        
        if max_resolution <= 0:
            raise ValueError("max_resolution must be > 0")
        
        if max_resolution > 500:  # Reasonable upper limit
            raise ValueError("max_resolution too large (max 500)")
        
        if not (0.0 <= early_stopping_threshold <= 1.0):
            raise ValueError("early_stopping_threshold must be between 0 and 1")
        
        self.power_base = power_base
        self.max_resolution = max_resolution
        self.start_with_1x1 = start_with_1x1
        self.early_stopping_enabled = early_stopping_enabled
        self.early_stopping_threshold = early_stopping_threshold
    
    def generate_resolution_sequence(self) -> List[int]:
        """
        Generate the sequence of resolutions for progressive sorting.
        
        Returns:
            List of resolutions (e.g., [1, 2, 4, 8, 16, 32])
        """
        sequence = []
        
        # Start with 1x1 if requested
        if self.start_with_1x1:
            sequence.append(1)
        
        # Generate power-based sequence
        current = self.power_base
        while current <= self.max_resolution:
            sequence.append(current)
            current *= self.power_base
        
        return sequence
    
    def extract_signatures_at_resolution(self, frames, resolution):
        """
        Extract multi-resolution signatures for all frames at a specific resolution.
        
        Args:
            frames: List of QR frame images (numpy arrays)
            resolution: Target resolution (e.g. 1, 2, 4, 8, 16)
            
        Returns:
            List of signatures (flattened numpy arrays) for each frame
        """
        if not frames:
            return []
            
        # Validate frames
        self._validate_frames(frames)
            
        signatures = []
        for frame in frames:
            # Resize frame to target resolution using area averaging
            resized = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
            
            # Flatten to create signature vector
            signature = resized.flatten()
            signatures.append(signature)
            
        return signatures
    
    def progressive_sort(self, frame_indices, frames):
        """
        Sort frame indices using progressive multi-resolution algorithm.
        
        Args:
            frame_indices: List of frame indices to sort
            frames: List of actual frame images
            
        Returns:
            Tuple of (sorted_indices, early_stopping_metadata)
        """
        if not frame_indices or len(frame_indices) <= 1:
            return frame_indices[:], {}
            
        # Get resolution sequence
        resolution_sequence = self.generate_resolution_sequence()
        
        # Start with original indices
        sorted_indices = frame_indices[:]
        previous_order = None
        improvements = []
        early_stopping_triggered = False
        early_stopping_resolution = None
        
        # Progressive sorting: sort by each resolution level
        for i, resolution in enumerate(resolution_sequence):
            # Extract signatures at this resolution
            signatures = self.extract_signatures_at_resolution(frames, resolution)
            
            # Get signatures for current sorted order
            current_signatures = [signatures[idx] for idx in sorted_indices]
            
            # Sort indices by signatures (stable sort)
            new_sorted_indices = self._stable_sort_by_signatures(sorted_indices, current_signatures)
            
            # Calculate improvement if early stopping is enabled
            if self.early_stopping_enabled and previous_order is not None:
                improvement = self._calculate_improvement(previous_order, new_sorted_indices)
                improvements.append(improvement)
                
                # Check if improvement is below threshold (skip first resolution)
                if i > 1 and improvement < self.early_stopping_threshold:
                    early_stopping_triggered = True
                    early_stopping_resolution = resolution
                    break
            
            previous_order = sorted_indices[:]
            sorted_indices = new_sorted_indices
            
        # Prepare early stopping metadata
        early_stopping_metadata = {}
        if self.early_stopping_enabled:
            early_stopping_metadata = {
                'early_stopping_triggered': early_stopping_triggered,
                'resolution_improvements': improvements
            }
            if early_stopping_triggered:
                early_stopping_metadata['early_stopping_resolution'] = early_stopping_resolution
            
        return sorted_indices, early_stopping_metadata
    
    def _stable_sort_by_signatures(self, indices, signatures):
        """
        Stable sort of indices based on signature lexicographic comparison.
        
        Args:
            indices: List of indices to sort
            signatures: List of signature arrays corresponding to indices
            
        Returns:
            List of indices sorted by signatures
        """
        # Create pairs of (index, signature) for sorting
        pairs = list(zip(indices, signatures))
        
        # Sort by signature using lexicographic comparison
        # Use tuple conversion for lexicographic comparison
        def signature_key(pair):
            idx, sig = pair
            # Convert numpy array to tuple for lexicographic comparison
            return tuple(sig.flatten())
        
        sorted_pairs = sorted(pairs, key=signature_key)
        
        # Extract sorted indices
        return [idx for idx, sig in sorted_pairs]
    
    def _calculate_improvement(self, previous_order, new_order):
        """
        Calculate the improvement between two frame orderings.
        
        Args:
            previous_order: Previous frame ordering (list of indices)
            new_order: New frame ordering (list of indices)
            
        Returns:
            Improvement ratio (0.0 = no change, 1.0 = completely different)
        """
        if len(previous_order) != len(new_order):
            return 1.0  # Complete change if different lengths
        
        if len(previous_order) == 0:
            return 0.0  # No change if empty
        
        # Count how many positions changed
        changes = sum(1 for i in range(len(previous_order)) 
                     if previous_order[i] != new_order[i])
        
        # Return ratio of changes
        return changes / len(previous_order)
    
    def optimize_frame_order(self, frames):
        """
        Complete frame ordering optimization using progressive resolution sorting.
        
        Args:
            frames: List of QR frame images (numpy arrays)
            
        Returns:
            Dict with optimization results:
            - frame_order: List of frame indices in optimized order
            - metadata: Dict with optimization details including early stopping info
        """
        if not frames:
            return {
                'frame_order': [],
                'metadata': {
                    'frame_count': 0, 
                    'power_base': self.power_base, 
                    'max_resolution': self.max_resolution, 
                    'start_with_1x1': self.start_with_1x1,
                    'resolution_sequence': [],
                    'early_stopping_enabled': self.early_stopping_enabled
                }
            }
        
        # Create initial frame indices
        frame_indices = list(range(len(frames)))
        
        # Get resolution sequence for metadata
        resolution_sequence = self.generate_resolution_sequence()
        
        # Perform progressive sorting
        sorted_indices, early_stopping_metadata = self.progressive_sort(frame_indices, frames)
        
        # Create metadata
        metadata = {
            'frame_count': len(frames),
            'power_base': self.power_base,
            'max_resolution': self.max_resolution,
            'start_with_1x1': self.start_with_1x1,
            'resolution_sequence': resolution_sequence,
            'early_stopping_enabled': self.early_stopping_enabled,
            'early_stopping_threshold': self.early_stopping_threshold,
        }
        
        # Add early stopping metadata
        metadata.update(early_stopping_metadata)
        
        return {
            'frame_order': sorted_indices,
            'metadata': metadata
        }
    
    def _validate_parameters(self):
        """Validate constructor parameters"""
        # TODO: Implement parameter validation - stub will fail tests
        pass
    
    def _validate_frames(self, frames: List[np.ndarray]):
        """Validate that frames are properly formatted"""
        if not frames:
            return
        
        first_frame = frames[0]
        
        # Check that first frame is square
        if len(first_frame.shape) != 2 or first_frame.shape[0] != first_frame.shape[1]:
            raise ValueError("Frames must be square (height == width)")
        
        expected_shape = first_frame.shape
        
        # Check all frames have same shape
        for i, frame in enumerate(frames):
            if frame.shape != expected_shape:
                raise ValueError(f"Frame {i} has shape {frame.shape}, expected {expected_shape}")
            
            if len(frame.shape) != 2:
                raise ValueError(f"Frame {i} must be 2D array")
    
    def _downsample_frame(self, frame: np.ndarray, target_resolution: int) -> np.ndarray:
        """
        Downsample a frame to target resolution using area averaging.
        
        Args:
            frame: Input QR frame
            target_resolution: Target resolution (e.g., 2 for 2x2)
            
        Returns:
            Downsampled frame as target_resolution x target_resolution array
        """
        if target_resolution == 1:
            # Special case: 1x1 means overall brightness/average
            return np.array([frame.mean()])
        
        # Use cv2.resize with INTER_AREA for proper averaging
        downsampled = cv2.resize(
            frame, 
            (target_resolution, target_resolution), 
            interpolation=cv2.INTER_AREA
        )
        
        return downsampled 