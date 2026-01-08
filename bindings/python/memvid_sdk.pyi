# Python type stubs for memvid_sdk

from typing import Optional, List

class Memvid:
    """
    A portable AI memory file (.mv2).
    
    Use `Memvid.create()` to create a new memory file or `Memvid.open()` to open an existing one.
    """
    
    @staticmethod
    def create(path: str) -> "Memvid":
        """Create a new memory file at the given path."""
        ...
    
    @staticmethod
    def open(path: str) -> "Memvid":
        """Open an existing memory file."""
        ...
    
    @staticmethod
    def open_read_only(path: str) -> "Memvid":
        """Open a memory file in read-only mode."""
        ...
    
    def put(
        self,
        text: str,
        title: Optional[str] = None,
        uri: Optional[str] = None,
        tags: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        timestamp: Optional[int] = None,
    ) -> int:
        """
        Store content and return the frame ID.
        
        Args:
            text: The text content to store.
            title: Optional title for the document.
            uri: Optional URI identifier.
            tags: Optional list of tags.
            labels: Optional list of labels.
            timestamp: Optional Unix timestamp.
        
        Returns:
            The frame ID (int) that can be used with remove().
        """
        ...
    
    def remove(self, frame_id: int) -> int:
        """
        Remove a frame by its ID.
        
        This is a soft delete - the frame is marked as deleted and removed from indexes.
        
        Args:
            frame_id: The frame ID returned by put().
        
        Returns:
            The WAL sequence number of the delete operation.
        """
        ...
    
    def commit(self) -> None:
        """Commit pending changes to disk."""
        ...
    
    def seal(self) -> None:
        """Seal the memory file (commit and close)."""
        ...
    
    def frame_count(self) -> int:
        """Get the number of frames in the memory."""
        ...
    
    def is_read_only(self) -> bool:
        """Check if the memory is read-only."""
        ...
