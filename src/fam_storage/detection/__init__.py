"""On-device face detection and recognition engine.

Runs ``face_recognition`` (dlib-based) in a multi-process pool so that
all available CPU cores are utilised for inference.

Typical usage::

    registry = FaceRegistry(references_dir, cache_path)
    registry.load()

    engine = DetectionEngine(config.detection, registry)
    results = engine.process_batch([(item, image_bytes), ...])
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from fam_storage.config import DetectionConfig
from fam_storage.registry import FaceRegistry, PersonEncodings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    """Outcome of running detection on a single image."""

    item_id: str
    filename: str
    matched: bool
    matched_people: list[str] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Module-level worker state (populated in each worker process)
# ---------------------------------------------------------------------------

_worker_people: list[PersonEncodings] | None = None
_worker_tolerance: float = 0.6
_worker_model: str = "hog"


def _init_worker(
    people: list[PersonEncodings],
    tolerance: float,
    model: str,
) -> None:
    """Initialiser for each worker process in the pool."""
    global _worker_people, _worker_tolerance, _worker_model  # noqa: PLW0603
    _worker_people = people
    _worker_tolerance = tolerance
    _worker_model = model


def _detect_worker(task: tuple[str, str, bytes]) -> DetectionResult:
    """Worker function: detect and recognise faces in one image.

    Args:
        task: ``(item_id, filename, image_bytes)`` tuple.

    Returns:
        :class:`DetectionResult` with match details.
    """
    import io

    import face_recognition  # type: ignore[import]
    from PIL import Image

    item_id, filename, image_bytes = task

    if _worker_people is None:
        return DetectionResult(
            item_id=item_id,
            filename=filename,
            matched=False,
            error="Worker not initialised.",
        )

    try:
        # Decode image; convert to RGB (face_recognition requires RGB numpy array)
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb_array = np.array(img.convert("RGB"))

        locations = face_recognition.face_locations(
            rgb_array, model=_worker_model
        )
        if not locations:
            return DetectionResult(
                item_id=item_id, filename=filename, matched=False
            )

        encodings = face_recognition.face_encodings(rgb_array, locations)

        matched_people: list[str] = []
        for person in _worker_people:
            if not person.encodings:
                continue
            for enc in encodings:
                distances = face_recognition.face_distance(
                    person.encodings, enc
                )
                if np.any(distances <= _worker_tolerance):
                    matched_people.append(person.name)
                    break  # Person matched — no need to check more faces for them

        return DetectionResult(
            item_id=item_id,
            filename=filename,
            matched=bool(matched_people),
            matched_people=matched_people,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Detection error for %s: %s", filename, exc)
        return DetectionResult(
            item_id=item_id,
            filename=filename,
            matched=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DetectionEngine:
    """Multi-process face detection engine.

    Args:
        config: Detection configuration.
        registry: Populated :class:`FaceRegistry` instance.
    """

    def __init__(self, config: DetectionConfig, registry: FaceRegistry) -> None:
        self._config = config
        self._registry = registry

    def process_batch(
        self,
        tasks: Sequence[tuple[str, str, bytes]],
    ) -> list[DetectionResult]:
        """Process a batch of images in parallel worker processes.

        Args:
            tasks: Sequence of ``(item_id, filename, image_bytes)`` tuples.

        Returns:
            List of :class:`DetectionResult` objects, one per task.
        """
        from concurrent.futures import ProcessPoolExecutor

        if not tasks:
            return []

        people = self._registry.people
        tolerance = self._config.tolerance
        model = self._config.model
        workers = min(self._config.max_workers, len(tasks), os.cpu_count() or 1)

        logger.debug(
            "Processing %d images with %d worker(s).", len(tasks), workers
        )

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(people, tolerance, model),
        ) as executor:
            results = list(executor.map(_detect_worker, tasks))

        matched = sum(1 for r in results if r.matched)
        logger.info(
            "Batch complete: %d/%d images matched.", matched, len(tasks)
        )
        return results
