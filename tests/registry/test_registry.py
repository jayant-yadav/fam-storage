"""Tests for the FaceRegistry."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fam_storage.registry import FaceRegistry, PersonEncodings


def _make_encoding() -> np.ndarray:
    """Return a fake 128-d face encoding."""
    rng = np.random.default_rng(42)
    return rng.random(128).astype(np.float64)


def _write_reference_images(base: Path, people: dict[str, int]) -> None:
    """Create dummy reference image files in `base/person/` directories."""
    for name, count in people.items():
        person_dir = base / name
        person_dir.mkdir(parents=True)
        for i in range(count):
            # Create an empty file with a jpg extension (mocked encoding later)
            (person_dir / f"photo_{i}.jpg").write_bytes(b"")


class TestFaceRegistryCache:
    def test_load_from_cache(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "encodings.pkl"
        people = [
            PersonEncodings(name="mom", encodings=[_make_encoding()]),
            PersonEncodings(name="dad", encodings=[_make_encoding()]),
        ]
        cache_path.write_bytes(pickle.dumps(people))

        registry = FaceRegistry(
            references_dir=tmp_path / "refs",
            cache_path=cache_path,
        )
        registry.load()

        assert len(registry.people) == 2
        assert registry.names() == ["mom", "dad"]

    def test_force_rebuild_ignores_cache(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "encodings.pkl"
        stale = [PersonEncodings(name="stale", encodings=[_make_encoding()])]
        cache_path.write_bytes(pickle.dumps(stale))

        refs_dir = tmp_path / "refs"
        _write_reference_images(refs_dir, {"mom": 1})

        enc = _make_encoding()

        with patch("face_recognition.load_image_file", return_value=MagicMock()):
            with patch("face_recognition.face_encodings", return_value=[enc]):
                registry = FaceRegistry(
                    references_dir=refs_dir,
                    cache_path=cache_path,
                )
                registry.load(force_rebuild=True)

        assert len(registry.people) == 1
        assert registry.people[0].name == "mom"

    def test_is_empty_true_when_no_people(self, tmp_path: Path) -> None:
        registry = FaceRegistry(
            references_dir=tmp_path / "refs",
            cache_path=tmp_path / "enc.pkl",
        )
        assert registry.is_empty() is True

    def test_is_empty_false_when_loaded(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "enc.pkl"
        cache_path.write_bytes(
            pickle.dumps([PersonEncodings(name="x", encodings=[_make_encoding()])])
        )
        registry = FaceRegistry(
            references_dir=tmp_path / "refs",
            cache_path=cache_path,
        )
        registry.load()
        assert registry.is_empty() is False


class TestFaceRegistryBuild:
    def test_build_creates_people_with_encodings(self, tmp_path: Path) -> None:
        refs_dir = tmp_path / "refs"
        _write_reference_images(refs_dir, {"mom": 2, "dad": 1})

        enc = _make_encoding()

        with patch("face_recognition.load_image_file", return_value=MagicMock()):
            with patch("face_recognition.face_encodings", return_value=[enc]):
                registry = FaceRegistry(
                    references_dir=refs_dir,
                    cache_path=tmp_path / "enc.pkl",
                )
                registry.load(force_rebuild=True)

        assert len(registry.people) == 2
        names = registry.names()
        assert "mom" in names
        assert "dad" in names

    def test_build_skips_person_with_no_faces(self, tmp_path: Path) -> None:
        refs_dir = tmp_path / "refs"
        _write_reference_images(refs_dir, {"mom": 1, "no_face": 1})

        enc = _make_encoding()

        def fake_encodings(img, *args, **kwargs):
            # Return empty list for second call (no_face person)
            if not hasattr(fake_encodings, "_calls"):
                fake_encodings._calls = 0
            fake_encodings._calls += 1
            return [enc] if fake_encodings._calls == 1 else []

        with patch("face_recognition.load_image_file", return_value=MagicMock()):
            with patch("face_recognition.face_encodings", side_effect=fake_encodings):
                registry = FaceRegistry(
                    references_dir=refs_dir,
                    cache_path=tmp_path / "enc.pkl",
                )
                registry.load(force_rebuild=True)

        assert len(registry.people) == 1
        assert registry.people[0].name == "mom"

    def test_build_cache_is_saved(self, tmp_path: Path) -> None:
        refs_dir = tmp_path / "refs"
        _write_reference_images(refs_dir, {"partner": 1})
        cache_path = tmp_path / "enc.pkl"

        enc = _make_encoding()

        with patch("face_recognition.load_image_file", return_value=MagicMock()):
            with patch("face_recognition.face_encodings", return_value=[enc]):
                registry = FaceRegistry(
                    references_dir=refs_dir,
                    cache_path=cache_path,
                )
                registry.load(force_rebuild=True)

        assert cache_path.exists()
        loaded = pickle.loads(cache_path.read_bytes())  # noqa: S301
        assert len(loaded) == 1
        assert loaded[0].name == "partner"

    def test_build_raises_if_refs_dir_missing(self, tmp_path: Path) -> None:
        registry = FaceRegistry(
            references_dir=tmp_path / "nonexistent",
            cache_path=tmp_path / "enc.pkl",
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            registry.load(force_rebuild=True)
