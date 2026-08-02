"""Tests for the pipeline orchestrator."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from fam_storage.config import AppConfig, load_config
from fam_storage.connectors import MediaItem
from fam_storage.detection import DetectionResult
from fam_storage.pipeline import Pipeline
from fam_storage.registry import FaceRegistry, PersonEncodings


def _make_encoding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(128).astype(np.float64)


def _minimal_config(tmp_path: Path, **overrides) -> AppConfig:
    """Build a minimal AppConfig suitable for pipeline tests."""
    import textwrap
    import yaml

    cfg_data = {
        "google_photos": {"enabled": False},
        "icloud": {"enabled": False},
        "references_dir": str(tmp_path / "refs"),
        "detection": {
            "max_workers": 1,
            "batch_size": 2,
            "encodings_cache": str(tmp_path / "enc.pkl"),
        },
        "nextcloud": {
            "url": "https://nas.local/nextcloud",
            "username": "u",
            "password": "p",
        },
        "pipeline": {
            "checkpoint_file": str(tmp_path / "state.json"),
            "dry_run": True,
        },
    }
    cfg_data.update(overrides)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg_data), encoding="utf-8")
    return load_config(cfg_path)


def _populated_registry(tmp_path: Path) -> FaceRegistry:
    cache_path = tmp_path / "enc.pkl"
    people = [PersonEncodings(name="mom", encodings=[_make_encoding()])]
    cache_path.write_bytes(pickle.dumps(people))

    registry = FaceRegistry(
        references_dir=tmp_path / "refs",
        cache_path=cache_path,
    )
    registry.load()
    return registry


class TestPipelineCheckpoint:
    def test_loads_existing_checkpoint(self, tmp_path: Path) -> None:
        config = _minimal_config(tmp_path)
        checkpoint = Path(config.pipeline.checkpoint_file)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(
            json.dumps({"processed_ids": ["id_1", "id_2"]}), encoding="utf-8"
        )

        pipeline = Pipeline(config)
        pipeline._load_checkpoint()

        assert "id_1" in pipeline._processed_ids
        assert "id_2" in pipeline._processed_ids

    def test_creates_checkpoint_on_mark_processed(self, tmp_path: Path) -> None:
        config = _minimal_config(tmp_path)
        pipeline = Pipeline(config)
        pipeline._load_checkpoint()

        pipeline._mark_processed("new_id")

        checkpoint = Path(config.pipeline.checkpoint_file)
        assert checkpoint.exists()
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert "new_id" in data["processed_ids"]

    def test_skips_already_processed_ids(self, tmp_path: Path) -> None:
        config = _minimal_config(tmp_path)
        checkpoint = Path(config.pipeline.checkpoint_file)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(
            json.dumps({"processed_ids": ["already_done"]}), encoding="utf-8"
        )

        registry = _populated_registry(tmp_path)

        pipeline = Pipeline(config)
        pipeline._load_checkpoint()
        pipeline._registry = registry
        pipeline._engine = MagicMock()
        pipeline._engine.process_batch.return_value = []

        mock_connector = MagicMock()
        mock_connector.list_media_items.return_value = [
            MediaItem(
                id="already_done",
                filename="skip.jpg",
                mime_type="image/jpeg",
                download_url="",
                source="test",
            )
        ]
        pipeline._connectors = [mock_connector]

        pipeline.run()

        # process_batch should not have been called since the ID was already processed
        pipeline._engine.process_batch.assert_not_called()


class TestPipelineRun:
    def test_dry_run_does_not_upload(self, tmp_path: Path) -> None:
        config = _minimal_config(tmp_path)
        assert config.pipeline.dry_run is True

        registry = _populated_registry(tmp_path)
        pipeline = Pipeline(config)
        pipeline._load_checkpoint()
        pipeline._registry = registry

        mock_engine = MagicMock()
        mock_engine.process_batch.return_value = [
            DetectionResult(
                item_id="id_1",
                filename="family.jpg",
                matched=True,
                matched_people=["mom"],
            )
        ]
        pipeline._engine = mock_engine
        pipeline._storage = MagicMock()

        image_bytes = b"\xff\xd8\xff" + b"\x00" * 10

        mock_connector = MagicMock()
        mock_connector.list_media_items.return_value = [
            MediaItem(
                id="id_1",
                filename="family.jpg",
                mime_type="image/jpeg",
                download_url="",
                source="test",
            )
        ]
        mock_connector.download_image.return_value = image_bytes
        pipeline._connectors = [mock_connector]

        pipeline.run()

        # Storage should NOT be called in dry_run mode
        pipeline._storage.upload_image.assert_not_called()

    def test_matched_image_is_uploaded(self, tmp_path: Path) -> None:
        config = _minimal_config(tmp_path)
        config.pipeline.dry_run = False

        registry = _populated_registry(tmp_path)
        pipeline = Pipeline(config)
        pipeline._load_checkpoint()
        pipeline._registry = registry

        image_bytes = b"\xff\xd8\xff" + b"\x00" * 10

        mock_engine = MagicMock()
        mock_engine.process_batch.return_value = [
            DetectionResult(
                item_id="id_upload",
                filename="happy.jpg",
                matched=True,
                matched_people=["mom"],
            )
        ]
        pipeline._engine = mock_engine

        mock_storage = MagicMock()
        mock_storage.upload_image.return_value = ["/FamilyPhotos/mom/2024/01/happy.jpg"]
        pipeline._storage = mock_storage

        mock_connector = MagicMock()
        mock_connector.list_media_items.return_value = [
            MediaItem(
                id="id_upload",
                filename="happy.jpg",
                mime_type="image/jpeg",
                download_url="",
                created_at="2024-01-10T08:00:00Z",
                source="test",
            )
        ]
        mock_connector.download_image.return_value = image_bytes
        pipeline._connectors = [mock_connector]

        pipeline.run()

        mock_storage.upload_image.assert_called_once()
        call_kwargs = mock_storage.upload_image.call_args
        assert call_kwargs.kwargs["filename"] == "happy.jpg"
        assert "mom" in call_kwargs.kwargs["matched_people"]

    def test_download_error_is_logged_and_continues(self, tmp_path: Path) -> None:
        config = _minimal_config(tmp_path)
        registry = _populated_registry(tmp_path)

        pipeline = Pipeline(config)
        pipeline._load_checkpoint()
        pipeline._registry = registry
        pipeline._engine = MagicMock()

        mock_connector = MagicMock()
        mock_connector.list_media_items.return_value = [
            MediaItem(
                id="bad_id",
                filename="corrupt.jpg",
                mime_type="image/jpeg",
                download_url="",
                source="test",
            )
        ]
        mock_connector.download_image.side_effect = IOError("network error")
        pipeline._connectors = [mock_connector]

        # Should not raise
        pipeline.run()

        # The item should still be marked as processed
        assert "bad_id" in pipeline._processed_ids
