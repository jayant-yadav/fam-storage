"""Base connector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class MediaItem:
    """Represents a single media item from a cloud source."""

    id: str
    filename: str
    mime_type: str
    download_url: str
    created_at: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)


class BaseConnector(ABC):
    """Abstract base class for cloud photo source connectors."""

    @abstractmethod
    def authenticate(self) -> None:
        """Authenticate with the cloud service.

        Raises:
            RuntimeError: If authentication fails.
        """

    @abstractmethod
    def list_media_items(self) -> Iterator[MediaItem]:
        """Yield :class:`MediaItem` objects for all photos in the source.

        Yields:
            :class:`MediaItem` for each photo in the source.
        """

    @abstractmethod
    def download_image(self, item: MediaItem, timeout: int = 30) -> bytes:
        """Download the raw image bytes for a media item.

        Args:
            item: The media item to download.
            timeout: HTTP timeout in seconds.

        Returns:
            Raw image bytes.

        Raises:
            IOError: If the download fails.
        """
