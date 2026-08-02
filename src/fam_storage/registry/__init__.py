"""Person registry: loads and caches face encodings for registered people.

The registry reads reference photos from a directory tree::

    references/
        mom/
            photo1.jpg
            photo2.jpg
        dad/
            portrait.png
        partner/
            img1.jpg

Each sub-directory name becomes the person's label.

Face encodings are computed once and cached in a pickle file so that
subsequent runs do not pay the encoding cost again.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


@dataclass
class PersonEncodings:
    """Face encodings for a single registered person."""

    name: str
    encodings: list[np.ndarray] = field(default_factory=list)


class FaceRegistry:
    """Manages reference face encodings for all registered people.

    Args:
        references_dir: Root directory containing one sub-directory per person.
        cache_path: Path to the pickle cache file for encodings.
    """

    def __init__(self, references_dir: str | Path, cache_path: str | Path) -> None:
        self._references_dir = Path(references_dir)
        self._cache_path = Path(cache_path)
        self._people: list[PersonEncodings] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def people(self) -> list[PersonEncodings]:
        """Return the list of registered :class:`PersonEncodings`."""
        return self._people

    def is_empty(self) -> bool:
        """Return ``True`` if no people have been registered."""
        return len(self._people) == 0 or all(
            len(p.encodings) == 0 for p in self._people
        )

    def load(self, force_rebuild: bool = False) -> None:
        """Load encodings from cache or rebuild from reference images.

        Args:
            force_rebuild: If ``True``, ignore the cache and re-encode.
        """
        if not force_rebuild and self._cache_path.exists():
            self._load_from_cache()
            if not self.is_empty():
                logger.info(
                    "Loaded face encodings for %d people from cache (%s).",
                    len(self._people),
                    self._cache_path,
                )
                return

        logger.info("Building face encodings from reference images …")
        self._build_from_references()
        self._save_to_cache()

    def names(self) -> list[str]:
        """Return the list of registered person names."""
        return [p.name for p in self._people]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_from_references(self) -> None:
        """Encode all reference images and populate :attr:`_people`."""
        try:
            import face_recognition  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "face-recognition is required. Install: pip install face-recognition"
            ) from exc

        if not self._references_dir.is_dir():
            raise FileNotFoundError(
                f"References directory not found: {self._references_dir}"
            )

        self._people = []
        for person_dir in sorted(self._references_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            name = person_dir.name
            person = PersonEncodings(name=name)

            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                    continue
                try:
                    image = face_recognition.load_image_file(str(img_path))
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        person.encodings.append(encodings[0])
                        logger.debug(
                            "Encoded face from %s for '%s'.", img_path.name, name
                        )
                    else:
                        logger.warning(
                            "No face detected in reference image: %s", img_path
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error encoding %s: %s", img_path, exc)

            if person.encodings:
                self._people.append(person)
                logger.info(
                    "Registered '%s' with %d encoding(s).",
                    name,
                    len(person.encodings),
                )
            else:
                logger.warning(
                    "No valid encodings found for '%s' — skipping.", name
                )

    def _save_to_cache(self) -> None:
        """Persist encodings to the pickle cache file."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("wb") as fh:
            pickle.dump(self._people, fh)
        logger.debug("Saved face encodings to %s.", self._cache_path)

    def _load_from_cache(self) -> None:
        """Load encodings from the pickle cache file."""
        try:
            with self._cache_path.open("rb") as fh:
                self._people = pickle.load(fh)  # noqa: S301
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load encoding cache: %s", exc)
            self._people = []
