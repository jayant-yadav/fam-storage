"""Tests for the DetectionEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fam_storage.config import DetectionConfig
from fam_storage.detection import DetectionEngine, DetectionResult, _detect_worker
from fam_storage.registry import FaceRegistry, PersonEncodings


def _make_encoding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(128).astype(np.float64)


def _make_registry_with_people(names: list[str]) -> FaceRegistry:
    """Return a FaceRegistry with fake encodings for each person."""
    registry = MagicMock(spec=FaceRegistry)
    people = [
        PersonEncodings(name=n, encodings=[_make_encoding(i)])
        for i, n in enumerate(names)
    ]
    registry.people = people
    return registry


def _make_config(max_workers: int = 2, tolerance: float = 0.6) -> DetectionConfig:
    return DetectionConfig(
        max_workers=max_workers,
        tolerance=tolerance,
        model="hog",
        batch_size=8,
    )


# ---------------------------------------------------------------------------
# Unit tests for _detect_worker
# ---------------------------------------------------------------------------


class TestDetectWorker:
    def _init(self, people: list[PersonEncodings], tolerance: float = 0.6) -> None:
        import fam_storage.detection as det_module

        det_module._worker_people = people
        det_module._worker_tolerance = tolerance
        det_module._worker_model = "hog"

    def teardown_method(self) -> None:
        import fam_storage.detection as det_module

        det_module._worker_people = None

    def test_returns_no_match_when_no_faces(self, tmp_path) -> None:
        """Image with no detected faces should return matched=False."""
        from PIL import Image
        import io

        # 1×1 white pixel
        img = Image.new("RGB", (1, 1), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        self._init(
            [PersonEncodings(name="mom", encodings=[_make_encoding()])],
        )

        with patch("face_recognition.face_locations", return_value=[]):
            result = _detect_worker(("id1", "img.jpg", image_bytes))

        assert result.matched is False
        assert result.matched_people == []

    def test_returns_match_when_face_recognized(self, tmp_path) -> None:
        from PIL import Image
        import io

        img = Image.new("RGB", (100, 100), color=(200, 180, 160))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        ref_enc = _make_encoding(seed=1)
        close_enc = ref_enc + 0.01  # very close to reference

        self._init(
            [PersonEncodings(name="dad", encodings=[ref_enc])],
            tolerance=0.6,
        )

        fake_location = [(10, 90, 90, 10)]
        with patch("face_recognition.face_locations", return_value=fake_location):
            with patch("face_recognition.face_encodings", return_value=[close_enc]):
                with patch(
                    "face_recognition.face_distance",
                    return_value=np.array([0.2]),
                ):
                    result = _detect_worker(("id2", "photo.jpg", image_bytes))

        assert result.matched is True
        assert "dad" in result.matched_people

    def test_returns_no_match_when_distance_above_tolerance(self) -> None:
        from PIL import Image
        import io

        img = Image.new("RGB", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        ref_enc = _make_encoding(seed=1)
        far_enc = _make_encoding(seed=99)

        self._init(
            [PersonEncodings(name="mom", encodings=[ref_enc])],
            tolerance=0.6,
        )

        fake_location = [(10, 90, 90, 10)]
        with patch("face_recognition.face_locations", return_value=fake_location):
            with patch("face_recognition.face_encodings", return_value=[far_enc]):
                with patch(
                    "face_recognition.face_distance",
                    return_value=np.array([0.9]),  # above tolerance
                ):
                    result = _detect_worker(("id3", "stranger.jpg", image_bytes))

        assert result.matched is False

    def test_handles_decode_error(self) -> None:
        self._init([PersonEncodings(name="mom", encodings=[_make_encoding()])])

        result = _detect_worker(("id_bad", "corrupt.jpg", b"not_an_image"))

        assert result.matched is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# Integration-style tests for DetectionEngine (mocked workers)
# ---------------------------------------------------------------------------


class TestDetectionEngine:
    def test_process_batch_returns_results_for_all_tasks(self, tmp_path) -> None:
        from PIL import Image
        import io

        img = Image.new("RGB", (10, 10))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        registry = _make_registry_with_people(["mom"])
        config = _make_config(max_workers=1)
        engine = DetectionEngine(config, registry)

        tasks = [
            ("id1", "a.jpg", image_bytes),
            ("id2", "b.jpg", image_bytes),
        ]

        with patch("face_recognition.face_locations", return_value=[]):
            results = engine.process_batch(tasks)

        assert len(results) == 2
        assert all(isinstance(r, DetectionResult) for r in results)

    def test_process_batch_empty_returns_empty(self) -> None:
        registry = _make_registry_with_people(["mom"])
        config = _make_config()
        engine = DetectionEngine(config, registry)

        results = engine.process_batch([])
        assert results == []
