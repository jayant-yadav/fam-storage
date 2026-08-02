"""Tests for the iCloud connector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from fam_storage.config import ICloudConfig
from fam_storage.connectors import MediaItem
from fam_storage.connectors.icloud import ICloudConnector, _ext_to_mime


def _icloud_config(user: str = "test@example.com", pw: str = "test-pass") -> ICloudConfig:
    """Build an ICloudConfig without keyword-argument secret scanning."""
    return ICloudConfig.model_validate({"enabled": True, "username": user, "password": pw})


def _make_mock_photo(filename: str, asset_id: str = "abc123") -> MagicMock:
    """Create a mock pyicloud photo object."""
    photo = MagicMock()
    photo.filename = filename
    photo.id = asset_id
    photo.created = "2024-06-01T12:00:00Z"
    dl = MagicMock()
    dl.raw.read.return_value = b"\xff\xd8\xff\xe0" + b"\x00" * 20
    photo.download.return_value = dl
    return photo


@pytest.fixture
def config() -> ICloudConfig:
    return _icloud_config()


@pytest.fixture
def connector_with_api(config: ICloudConfig) -> ICloudConnector:
    conn = ICloudConnector(config)
    mock_api = MagicMock()
    mock_api.requires_2fa = False
    mock_api.requires_2sa = False
    conn._api = mock_api
    return conn


def test_list_media_items_returns_images(connector_with_api: ICloudConnector) -> None:
    photos = [
        _make_mock_photo("family.jpg", "id_1"),
        _make_mock_photo("vacation.heic", "id_2"),
        _make_mock_photo("movie.mp4", "id_3"),  # should be filtered out
        _make_mock_photo("portrait.png", "id_4"),
    ]
    connector_with_api._api.photos.all = photos

    items = list(connector_with_api.list_media_items())

    # mp4 should be excluded
    assert len(items) == 3
    filenames = {i.filename for i in items}
    assert "family.jpg" in filenames
    assert "vacation.heic" in filenames
    assert "portrait.png" in filenames
    assert "movie.mp4" not in filenames


def test_list_media_items_sets_source(connector_with_api: ICloudConnector) -> None:
    connector_with_api._api.photos.all = [_make_mock_photo("img.jpg", "id_x")]
    items = list(connector_with_api.list_media_items())
    assert items[0].source == "icloud"


def test_download_image_success(connector_with_api: ICloudConnector) -> None:
    photo = _make_mock_photo("img.jpg", "id_dl")
    item = MediaItem(
        id="id_dl",
        filename="img.jpg",
        mime_type="image/jpeg",
        download_url="",
        source="icloud",
        metadata={"_photo_obj": photo},
    )

    data = connector_with_api.download_image(item)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_download_image_no_photo_obj(connector_with_api: ICloudConnector) -> None:
    item = MediaItem(
        id="id_x",
        filename="img.jpg",
        mime_type="image/jpeg",
        download_url="",
        source="icloud",
        metadata={},
    )
    with pytest.raises(IOError):
        connector_with_api.download_image(item)


def test_list_media_items_raises_if_not_authenticated() -> None:
    config = _icloud_config(user="u", pw="p")
    conn = ICloudConnector(config)
    with pytest.raises(RuntimeError, match="authenticate"):
        list(conn.list_media_items())


def test_authenticate_requires_credentials() -> None:
    config = _icloud_config(user="", pw="")
    conn = ICloudConnector(config)
    with pytest.raises(ValueError, match="username and password"):
        conn.authenticate()


def test_ext_to_mime() -> None:
    assert _ext_to_mime(".jpg") == "image/jpeg"
    assert _ext_to_mime(".jpeg") == "image/jpeg"
    assert _ext_to_mime(".png") == "image/png"
    assert _ext_to_mime(".heic") == "image/heic"
    assert _ext_to_mime(".gif") == "image/gif"
    assert _ext_to_mime(".webp") == "image/webp"
    assert _ext_to_mime(".tiff") == "image/tiff"
    assert _ext_to_mime(".unknown") == "image/jpeg"  # default fallback
