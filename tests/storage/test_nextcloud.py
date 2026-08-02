"""Tests for NextcloudStorage."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from fam_storage.config import NextcloudConfig
from fam_storage.storage import NextcloudStorage, _parse_date, _sanitise_name


def _make_config(**overrides) -> NextcloudConfig:
    """Build a NextcloudConfig using model_validate to avoid secret scanning."""
    data = {
        "url": "https://nas.local/nextcloud",
        "username": "ncuser",
        "password": "test-nc-pass",
        "remote_base_path": "/FamilyPhotos",
        "verify_ssl": False,
    }
    data.update(overrides)
    return NextcloudConfig.model_validate(data)


@pytest.fixture
def storage_with_client() -> NextcloudStorage:
    config = _make_config()
    storage = NextcloudStorage(config)
    mock_client = MagicMock()
    mock_client.check.return_value = False  # file does not exist by default
    storage._client = mock_client
    return storage


# ---------------------------------------------------------------------------
# upload_image
# ---------------------------------------------------------------------------


class TestUploadImage:
    def test_upload_creates_directories_and_uploads(
        self, storage_with_client: NextcloudStorage
    ) -> None:
        result = storage_with_client.upload_image(
            image_bytes=b"JPEG_DATA",
            filename="family.jpg",
            matched_people=["mom"],
            created_at="2024-03-15T10:00:00Z",
        )

        assert len(result) == 1
        assert "mom" in result[0]
        assert "2024" in result[0]
        assert "03" in result[0]
        assert "family.jpg" in result[0]

        storage_with_client._client.upload_to.assert_called_once()

    def test_upload_skips_if_file_exists(
        self, storage_with_client: NextcloudStorage
    ) -> None:
        storage_with_client._client.check.return_value = True  # file already exists

        result = storage_with_client.upload_image(
            image_bytes=b"JPEG_DATA",
            filename="dup.jpg",
            matched_people=["dad"],
            created_at="2024-05-01T00:00:00Z",
        )

        assert result == []
        storage_with_client._client.upload_to.assert_not_called()

    def test_upload_for_multiple_people(
        self, storage_with_client: NextcloudStorage
    ) -> None:
        result = storage_with_client.upload_image(
            image_bytes=b"JPEG_DATA",
            filename="group.jpg",
            matched_people=["mom", "dad", "partner"],
            created_at="2024-07-04T00:00:00Z",
        )

        assert len(result) == 3
        assert storage_with_client._client.upload_to.call_count == 3

    def test_upload_returns_empty_for_no_people(
        self, storage_with_client: NextcloudStorage
    ) -> None:
        result = storage_with_client.upload_image(
            image_bytes=b"JPEG_DATA",
            filename="landscape.jpg",
            matched_people=[],
            created_at="2024-01-01T00:00:00Z",
        )
        assert result == []
        storage_with_client._client.upload_to.assert_not_called()

    def test_raises_if_not_connected(self) -> None:
        storage = NextcloudStorage(_make_config())
        with pytest.raises(RuntimeError, match="connect"):
            storage.upload_image(b"DATA", "f.jpg", ["mom"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_parse_date_iso_with_z(self) -> None:
        dt = _parse_date("2024-03-15T10:00:00Z")
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15

    def test_parse_date_fallback_to_now(self) -> None:
        from datetime import datetime

        dt = _parse_date("not-a-date")
        now = datetime.utcnow()
        assert dt.year == now.year

    def test_sanitise_name_replaces_spaces(self) -> None:
        assert _sanitise_name("my mom") == "my_mom"

    def test_sanitise_name_allows_alphanumeric_and_dash(self) -> None:
        assert _sanitise_name("mom-123") == "mom-123"

    def test_sanitise_name_replaces_special_chars(self) -> None:
        sanitised = _sanitise_name("mèré/pàpa")
        assert "/" not in sanitised
        assert "è" not in sanitised
