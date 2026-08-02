"""Tests for the Google Photos connector."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest
import responses as responses_lib

from fam_storage.config import GooglePhotosConfig
from fam_storage.connectors import MediaItem
from fam_storage.connectors.google_photos import GooglePhotosConnector

_SAMPLE_ITEMS = [
    {
        "id": "photo_1",
        "filename": "IMG_0001.jpg",
        "mimeType": "image/jpeg",
        "baseUrl": "https://lh3.googleusercontent.com/photo_1",
        "mediaMetadata": {"creationTime": "2024-01-15T10:00:00Z"},
    },
    {
        "id": "photo_2",
        "filename": "IMG_0002.png",
        "mimeType": "image/png",
        "baseUrl": "https://lh3.googleusercontent.com/photo_2",
        "mediaMetadata": {"creationTime": "2024-01-16T10:00:00Z"},
    },
    {
        "id": "video_1",
        "filename": "MOV_0001.mp4",
        "mimeType": "video/mp4",
        "baseUrl": "https://lh3.googleusercontent.com/video_1",
        "mediaMetadata": {"creationTime": "2024-01-17T10:00:00Z"},
    },
]


@pytest.fixture
def mock_connector() -> GooglePhotosConnector:
    config = GooglePhotosConfig(
        enabled=True,
        credentials_file="config/creds.json",
        token_file="config/token.json",
        page_size=100,
    )
    conn = GooglePhotosConnector(config)
    # Inject a fake authenticated session
    session = MagicMock()
    conn._session = session
    return conn


@responses_lib.activate
def test_list_media_items_filters_images(mock_connector: GooglePhotosConnector) -> None:
    responses_lib.add(
        responses_lib.GET,
        "https://photoslibrary.googleapis.com/v1/mediaItems",
        json={"mediaItems": _SAMPLE_ITEMS},
        status=200,
    )
    # Override the MagicMock session with a real session for this test
    import requests

    real_session = requests.Session()
    real_session.headers["Authorization"] = "******"
    mock_connector._session = real_session

    items = list(mock_connector.list_media_items())

    # Only image types should be returned (not video/mp4)
    assert len(items) == 2
    assert all(isinstance(i, MediaItem) for i in items)
    ids = {i.id for i in items}
    assert "photo_1" in ids
    assert "photo_2" in ids
    assert "video_1" not in ids


@responses_lib.activate
def test_list_media_items_pagination(mock_connector: GooglePhotosConnector) -> None:
    import requests

    real_session = requests.Session()
    real_session.headers["Authorization"] = "******"
    mock_connector._session = real_session

    page1_items = [_SAMPLE_ITEMS[0]]
    page2_items = [_SAMPLE_ITEMS[1]]

    responses_lib.add(
        responses_lib.GET,
        "https://photoslibrary.googleapis.com/v1/mediaItems",
        json={"mediaItems": page1_items, "nextPageToken": "TOKEN_PAGE2"},
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        "https://photoslibrary.googleapis.com/v1/mediaItems",
        json={"mediaItems": page2_items},
        status=200,
    )

    items = list(mock_connector.list_media_items())
    assert len(items) == 2


@responses_lib.activate
def test_download_image_success(mock_connector: GooglePhotosConnector) -> None:
    import requests

    real_session = requests.Session()
    real_session.headers["Authorization"] = "******"
    mock_connector._session = real_session

    fake_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # JPEG magic bytes

    responses_lib.add(
        responses_lib.GET,
        "https://lh3.googleusercontent.com/photo_1=d",
        body=fake_bytes,
        status=200,
    )

    item = MediaItem(
        id="photo_1",
        filename="IMG_0001.jpg",
        mime_type="image/jpeg",
        download_url="https://lh3.googleusercontent.com/photo_1=d",
        source="google_photos",
    )

    data = mock_connector.download_image(item)
    assert data == fake_bytes


@responses_lib.activate
def test_download_image_failure(mock_connector: GooglePhotosConnector) -> None:
    import requests

    real_session = requests.Session()
    real_session.headers["Authorization"] = "******"
    mock_connector._session = real_session

    responses_lib.add(
        responses_lib.GET,
        "https://lh3.googleusercontent.com/missing=d",
        status=404,
    )

    item = MediaItem(
        id="missing",
        filename="missing.jpg",
        mime_type="image/jpeg",
        download_url="https://lh3.googleusercontent.com/missing=d",
        source="google_photos",
    )

    with pytest.raises(IOError):
        mock_connector.download_image(item)


def test_list_media_items_raises_if_not_authenticated() -> None:
    config = GooglePhotosConfig()
    conn = GooglePhotosConnector(config)
    with pytest.raises(RuntimeError, match="authenticate"):
        list(conn.list_media_items())


def test_download_raises_if_not_authenticated() -> None:
    config = GooglePhotosConfig()
    conn = GooglePhotosConnector(config)
    item = MediaItem(
        id="x",
        filename="x.jpg",
        mime_type="image/jpeg",
        download_url="https://example.com/x",
    )
    with pytest.raises(RuntimeError, match="authenticate"):
        conn.download_image(item)
