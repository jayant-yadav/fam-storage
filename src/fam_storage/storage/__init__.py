"""Nextcloud storage backend using WebDAV.

Organises photos on the remote server as::

    {remote_base_path}/{person_name}/{YYYY}/{MM}/{filename}

Uses the ``webdavclient3`` library for all remote operations.
Before uploading, checks for an existing file (deduplication via HEAD).
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import PurePosixPath
from typing import Sequence

from fam_storage.config import NextcloudConfig

logger = logging.getLogger(__name__)


class NextcloudStorage:
    """Upload matched photos to a Nextcloud instance over WebDAV.

    Args:
        config: Nextcloud configuration.
    """

    def __init__(self, config: NextcloudConfig) -> None:
        self._config = config
        self._client = None  # webdavclient3 Client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialise the WebDAV client.

        Raises:
            ImportError: If ``webdavclient3`` is not installed.
            RuntimeError: If the connection cannot be established.
        """
        try:
            from webdav3.client import Client  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "webdavclient3 is required. Install: pip install webdavclient3"
            ) from exc

        # Nextcloud WebDAV endpoint is at /remote.php/dav/files/{username}/
        webdav_url = (
            self._config.url.rstrip("/")
            + f"/remote.php/dav/files/{self._config.username}"
        )

        options = {
            "webdav_hostname": webdav_url,
            "webdav_login": self._config.username,
            "webdav_password": self._config.password,
        }

        self._client = Client(options)
        self._client.verify = self._config.verify_ssl
        logger.info("Nextcloud WebDAV client initialised for %s.", webdav_url)

    def upload_image(
        self,
        image_bytes: bytes,
        filename: str,
        matched_people: Sequence[str],
        created_at: str = "",
    ) -> list[str]:
        """Upload an image for each matched person.

        The file is stored at::

            {base_path}/{person}/{YYYY}/{MM}/{filename}

        Existing files with the same path are skipped (deduplication).

        Args:
            image_bytes: Raw image bytes to upload.
            filename: Original filename of the image.
            matched_people: Names of people identified in the image.
            created_at: ISO 8601 timestamp string (used for directory tree).

        Returns:
            List of remote paths where the file was uploaded.
        """
        if self._client is None:
            raise RuntimeError("Call connect() before uploading images.")

        date = _parse_date(created_at)
        year = date.strftime("%Y")
        month = date.strftime("%m")

        uploaded: list[str] = []
        for person in matched_people:
            remote_path = str(
                PurePosixPath(self._config.remote_base_path)
                / _sanitise_name(person)
                / year
                / month
                / filename
            )

            if self._file_exists(remote_path):
                logger.debug("Skipping (already exists): %s", remote_path)
                continue

            self._ensure_remote_dirs(
                str(
                    PurePosixPath(self._config.remote_base_path)
                    / _sanitise_name(person)
                    / year
                    / month
                )
            )

            self._client.upload_to(
                io.BytesIO(image_bytes),
                remote_path,
            )
            logger.info("Uploaded: %s", remote_path)
            uploaded.append(remote_path)

        return uploaded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _file_exists(self, remote_path: str) -> bool:
        """Return ``True`` if *remote_path* already exists on the server."""
        try:
            return self._client.check(remote_path)
        except Exception:  # noqa: BLE001
            return False

    def _ensure_remote_dirs(self, remote_dir: str) -> None:
        """Recursively create directories on the remote server."""
        parts = PurePosixPath(remote_dir).parts
        for i in range(1, len(parts) + 1):
            path = str(PurePosixPath(*parts[:i]))
            try:
                if not self._client.check(path):
                    self._client.mkdir(path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("mkdir %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp, falling back to *now* on error."""
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts, fmt)  # noqa: DTZ007
        except (ValueError, TypeError):
            continue
    from datetime import timezone

    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def _sanitise_name(name: str) -> str:
    """Replace characters that are invalid in remote path segments.

    Only ASCII letters, digits, hyphens, underscores and dots are kept.
    All other characters (including accented letters) are replaced with ``_``.
    """
    return "".join(
        c if (c.isascii() and c.isalnum()) or c in ("-", "_", ".") else "_"
        for c in name
    )
