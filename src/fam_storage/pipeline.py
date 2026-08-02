"""Pipeline orchestrator for fam-storage.

Ties together the source connectors, detection engine, and storage backend.
Supports resumable runs via a checkpoint file that tracks processed IDs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

from fam_storage.config import AppConfig
from fam_storage.connectors import BaseConnector, MediaItem
from fam_storage.detection import DetectionEngine, DetectionResult
from fam_storage.registry import FaceRegistry
from fam_storage.storage import NextcloudStorage

logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline: fetch → detect → store.

    Args:
        config: Application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._registry: FaceRegistry | None = None
        self._engine: DetectionEngine | None = None
        self._storage: NextcloudStorage | None = None
        self._connectors: list[BaseConnector] = []
        self._processed_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise all pipeline components.

        - Loads the face registry (from cache or by encoding reference images).
        - Connects to the Nextcloud storage backend.
        - Authenticates with enabled cloud connectors.
        - Loads the checkpoint file.
        """
        self._load_checkpoint()
        self._setup_registry()
        if not self._config.pipeline.dry_run:
            self._setup_storage()
        self._setup_connectors()

    def _setup_registry(self) -> None:
        cfg = self._config
        self._registry = FaceRegistry(
            references_dir=cfg.references_dir,
            cache_path=cfg.detection.encodings_cache,
        )
        self._registry.load()

        if self._registry.is_empty():
            raise RuntimeError(
                f"No face encodings found. "
                f"Add reference images to '{cfg.references_dir}/' "
                f"(one sub-directory per person)."
            )

        self._engine = DetectionEngine(cfg.detection, self._registry)
        logger.info(
            "Registry loaded with %d person(s): %s",
            len(self._registry.people),
            ", ".join(self._registry.names()),
        )

    def _setup_storage(self) -> None:
        self._storage = NextcloudStorage(self._config.nextcloud)
        self._storage.connect()

    def _setup_connectors(self) -> None:
        self._connectors = []

        if self._config.google_photos.enabled:
            from fam_storage.connectors.google_photos import GooglePhotosConnector

            conn = GooglePhotosConnector(self._config.google_photos)
            conn.authenticate()
            self._connectors.append(conn)
            logger.info("Google Photos connector ready.")

        if self._config.icloud.enabled:
            from fam_storage.connectors.icloud import ICloudConnector

            conn = ICloudConnector(self._config.icloud)
            conn.authenticate()
            self._connectors.append(conn)
            logger.info("iCloud connector ready.")

        if not self._connectors:
            logger.warning("No cloud connectors enabled in config.")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full pipeline end-to-end."""
        if self._engine is None:
            raise RuntimeError("Call setup() before run().")

        dry_run = self._config.pipeline.dry_run
        batch_size = self._config.detection.batch_size
        timeout = self._config.pipeline.download_timeout

        total_seen = 0
        total_matched = 0
        total_uploaded = 0
        total_errors = 0

        for connector in self._connectors:
            logger.info("Fetching from %s …", connector.__class__.__name__)

            batch: list[tuple[str, str, bytes]] = []
            batch_items: list[MediaItem] = []

            for item in connector.list_media_items():
                if item.id in self._processed_ids:
                    logger.debug("Skipping (already processed): %s", item.id)
                    continue

                total_seen += 1

                try:
                    image_bytes = connector.download_image(item, timeout=timeout)
                except IOError as exc:
                    logger.error("Download failed for %s: %s", item.filename, exc)
                    total_errors += 1
                    self._mark_processed(item.id)
                    continue

                batch.append((item.id, item.filename, image_bytes))
                batch_items.append(item)

                if len(batch) >= batch_size:
                    matched, uploaded, errors = self._process_batch(
                        batch, batch_items, dry_run
                    )
                    total_matched += matched
                    total_uploaded += uploaded
                    total_errors += errors
                    batch = []
                    batch_items = []

            # Flush remaining items
            if batch:
                matched, uploaded, errors = self._process_batch(
                    batch, batch_items, dry_run
                )
                total_matched += matched
                total_uploaded += uploaded
                total_errors += errors

        logger.info(
            "Pipeline complete — seen: %d, matched: %d, uploaded: %d, errors: %d",
            total_seen,
            total_matched,
            total_uploaded,
            total_errors,
        )

    def _process_batch(
        self,
        batch: list[tuple[str, str, bytes]],
        items: list[MediaItem],
        dry_run: bool,
    ) -> tuple[int, int, int]:
        """Run detection on a batch and upload matches.

        Returns:
            ``(matched_count, uploaded_count, error_count)``
        """
        results: list[DetectionResult] = self._engine.process_batch(batch)

        matched = 0
        uploaded = 0
        errors = 0

        for result, item, (_, _, image_bytes) in zip(results, items, batch):
            self._mark_processed(result.item_id)

            if result.error:
                errors += 1
                continue

            if result.matched:
                matched += 1
                logger.info(
                    "Match found in '%s' — people: %s",
                    result.filename,
                    ", ".join(result.matched_people),
                )

                if not dry_run and self._storage is not None:
                    try:
                        paths = self._storage.upload_image(
                            image_bytes=image_bytes,
                            filename=result.filename,
                            matched_people=result.matched_people,
                            created_at=item.created_at,
                        )
                        uploaded += len(paths)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "Upload failed for %s: %s", result.filename, exc
                        )
                        errors += 1

        return matched, uploaded, errors

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> None:
        checkpoint_path = Path(self._config.pipeline.checkpoint_file)
        if checkpoint_path.exists():
            try:
                data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                self._processed_ids = set(data.get("processed_ids", []))
                logger.info(
                    "Checkpoint loaded: %d already-processed IDs.",
                    len(self._processed_ids),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load checkpoint: %s", exc)
                self._processed_ids = set()
        else:
            self._processed_ids = set()

    def _mark_processed(self, item_id: str) -> None:
        """Mark an item as processed and persist the checkpoint."""
        self._processed_ids.add(item_id)
        checkpoint_path = Path(self._config.pipeline.checkpoint_file)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps({"processed_ids": list(self._processed_ids)}, indent=2),
            encoding="utf-8",
        )
