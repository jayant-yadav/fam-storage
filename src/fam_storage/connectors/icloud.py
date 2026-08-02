"""iCloud connector.

Uses the ``pyicloud`` library to access the iCloud Photos library.

Authentication supports:
- Standard username + password
- Two-factor authentication (2FA) via interactive prompt in non-headless mode
- Session cookie caching for repeated runs (``cookie_dir``)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import requests

from fam_storage.config import ICloudConfig
from fam_storage.connectors import BaseConnector, MediaItem

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".gif", ".webp", ".tiff"}


class ICloudConnector(BaseConnector):
    """Connector for Apple iCloud Photos."""

    def __init__(self, config: ICloudConfig) -> None:
        self._config = config
        self._api = None  # pyicloud.PyiCloudService instance

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> None:
        """Authenticate with iCloud.

        Opens an interactive 2FA prompt if required.

        Raises:
            ValueError: If username or password are not configured.
            RuntimeError: If authentication fails.
        """
        username = self._config.username
        password = self._config.password

        if not username or not password:
            raise ValueError(
                "iCloud username and password must be set in config or via "
                "ICLOUD_USERNAME / ICLOUD_PASSWORD environment variables."
            )

        try:
            from pyicloud import PyiCloudService
            from pyicloud.exceptions import (
                PyiCloudFailedLoginException,
                PyiCloudAPIResponseException,
            )
        except ImportError as exc:
            raise ImportError(
                "pyicloud is required for iCloud support. "
                "Install it with: pip install pyicloud"
            ) from exc

        cookie_dir = Path(self._config.cookie_dir)
        cookie_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._api = PyiCloudService(
                username,
                password,
                cookie_directory=str(cookie_dir),
            )
        except Exception as exc:
            raise RuntimeError(f"iCloud authentication failed: {exc}") from exc

        if self._api.requires_2fa:
            logger.info("iCloud 2FA required.")
            code = input("Enter the 2FA code received on your Apple device: ")
            result = self._api.validate_2fa_code(code)
            if not result:
                raise RuntimeError("Invalid iCloud 2FA code.")
            logger.info("iCloud 2FA verified.")

        elif self._api.requires_2sa:
            logger.info("iCloud two-step authentication required.")
            devices = self._api.trusted_devices
            for i, device in enumerate(devices):
                print(
                    f"  [{i}] {device.get('deviceName', 'Unknown')} "
                    f"({device.get('phoneNumber', '')})"
                )
            device_index = int(input("Select trusted device: "))
            device = devices[device_index]
            if not self._api.send_verification_code(device):
                raise RuntimeError("Could not send iCloud verification code.")
            code = input("Enter verification code: ")
            if not self._api.validate_verification_code(device, code):
                raise RuntimeError("Invalid iCloud verification code.")

        logger.info("iCloud authentication successful.")

    # ------------------------------------------------------------------
    # Media listing
    # ------------------------------------------------------------------

    def list_media_items(self) -> Iterator[MediaItem]:
        """Yield image :class:`MediaItem` objects from iCloud Photos.

        Yields:
            :class:`MediaItem` for each photo.

        Raises:
            RuntimeError: If :meth:`authenticate` has not been called.
        """
        if self._api is None:
            raise RuntimeError("Call authenticate() before listing media items.")

        photos = self._api.photos
        library = photos.all

        for idx, photo in enumerate(library):
            filename: str = getattr(photo, "filename", f"photo_{idx}.jpg")
            ext = Path(filename).suffix.lower()
            if ext not in _IMAGE_EXTENSIONS:
                continue

            # Build a stable ID from the asset GUID if available
            item_id = getattr(photo, "id", None) or f"icloud_{idx}"

            yield MediaItem(
                id=item_id,
                filename=filename,
                mime_type=_ext_to_mime(ext),
                download_url="",  # resolved at download time via pyicloud
                created_at=str(getattr(photo, "created", "")),
                source="icloud",
                metadata={"_photo_obj": photo},
            )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_image(self, item: MediaItem, timeout: int = 30) -> bytes:
        """Download image bytes from iCloud.

        Args:
            item: The media item (must contain ``_photo_obj`` in metadata).
            timeout: HTTP timeout in seconds.

        Returns:
            Raw image bytes.

        Raises:
            IOError: If the download fails.
        """
        photo = item.metadata.get("_photo_obj")
        if photo is None:
            raise IOError(
                f"Cannot download {item.filename}: no photo object in metadata."
            )

        try:
            download = photo.download()
            return download.raw.read()
        except Exception as exc:
            raise IOError(
                f"Failed to download iCloud photo {item.filename}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ext_to_mime(ext: str) -> str:
    """Map a file extension to a MIME type."""
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".heic": "image/heic",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
    }.get(ext, "image/jpeg")
