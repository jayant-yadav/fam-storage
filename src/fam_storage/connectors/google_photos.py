"""Google Photos connector.

Uses the Google Photos Library REST API v1 with OAuth 2.0.

Required OAuth scope: https://www.googleapis.com/auth/photoslibrary.readonly
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from fam_storage.config import GooglePhotosConfig
from fam_storage.connectors import BaseConnector, MediaItem

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/photoslibrary.readonly"]
_API_BASE = "https://photoslibrary.googleapis.com/v1"
_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/heic", "image/webp"}


class GooglePhotosConnector(BaseConnector):
    """Connector for Google Photos Library API."""

    def __init__(self, config: GooglePhotosConfig) -> None:
        self._config = config
        self._session: requests.Session | None = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> None:
        """Authenticate with Google using OAuth 2.0.

        On first run this opens a browser window for user consent.
        On subsequent runs the cached token file is used.

        Raises:
            FileNotFoundError: If the credentials file does not exist.
            RuntimeError: If authentication fails.
        """
        token_path = Path(self._config.token_file)
        credentials_path = Path(self._config.credentials_file)

        creds: Credentials | None = None

        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(
                    str(token_path), _SCOPES
                )
                logger.debug("Loaded Google OAuth token from %s", token_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load cached token: %s", exc)
                creds = None

        if creds is None or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                from google.auth.transport.requests import Request

                creds.refresh(Request())
                logger.info("Refreshed Google OAuth token.")
            else:
                if not credentials_path.exists():
                    raise FileNotFoundError(
                        f"Google credentials file not found: {credentials_path}\n"
                        "Download it from https://console.cloud.google.com/apis/credentials"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), _SCOPES
                )
                creds = flow.run_local_server(port=0)
                logger.info("Google OAuth authentication completed.")

            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")

        self._session = requests.Session()
        self._session.headers["Authorization"] = f"******"

    # ------------------------------------------------------------------
    # Media listing
    # ------------------------------------------------------------------

    def list_media_items(self) -> Iterator[MediaItem]:
        """Yield image :class:`MediaItem` objects from Google Photos.

        Only JPEG, PNG, GIF, HEIC and WebP mime types are yielded.

        Yields:
            :class:`MediaItem` for each photo.

        Raises:
            RuntimeError: If :meth:`authenticate` has not been called.
        """
        if self._session is None:
            raise RuntimeError("Call authenticate() before listing media items.")

        page_token: str | None = None
        while True:
            params: dict = {"pageSize": self._config.page_size}
            if page_token:
                params["pageToken"] = page_token

            resp = self._session.get(f"{_API_BASE}/mediaItems", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for raw in data.get("mediaItems", []):
                mime = raw.get("mimeType", "")
                if mime not in _IMAGE_MIME_TYPES:
                    continue
                yield MediaItem(
                    id=raw["id"],
                    filename=raw.get("filename", raw["id"]),
                    mime_type=mime,
                    # Append =d to get the full original download URL
                    download_url=raw["baseUrl"] + "=d",
                    created_at=raw.get("mediaMetadata", {}).get("creationTime", ""),
                    source="google_photos",
                    metadata=raw,
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_image(self, item: MediaItem, timeout: int = 30) -> bytes:
        """Download a single image from Google Photos.

        Args:
            item: The media item to download.
            timeout: HTTP timeout in seconds.

        Returns:
            Raw image bytes.

        Raises:
            IOError: If the HTTP request fails.
        """
        if self._session is None:
            raise RuntimeError("Call authenticate() before downloading images.")

        resp = self._session.get(item.download_url, timeout=timeout)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise IOError(f"Failed to download {item.filename}: {exc}") from exc

        return resp.content
