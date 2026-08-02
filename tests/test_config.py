"""Tests for configuration loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from fam_storage.config import load_config, AppConfig, DetectionConfig


def test_load_config_basic(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """\
            google_photos:
              enabled: false
            icloud:
              enabled: false
            references_dir: "refs"
            detection:
              max_workers: 8
              tolerance: 0.5
            nextcloud:
              url: "https://nas.local/nextcloud"
              username: "admin"
              password: "secret"
            pipeline:
              log_level: "DEBUG"
            """
        ),
        encoding="utf-8",
    )

    config = load_config(cfg_file)

    assert isinstance(config, AppConfig)
    assert config.google_photos.enabled is False
    assert config.icloud.enabled is False
    assert config.references_dir == "refs"
    assert config.detection.max_workers == 8
    assert config.detection.tolerance == 0.5
    assert config.nextcloud.url == "https://nas.local/nextcloud"
    assert config.nextcloud.username == "admin"
    assert config.pipeline.log_level == "DEBUG"


def test_load_config_defaults(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("{}", encoding="utf-8")

    config = load_config(cfg_file)

    assert config.detection.max_workers == 16
    assert config.detection.tolerance == 0.6
    assert config.detection.model == "hog"
    assert config.detection.batch_size == 8
    assert config.pipeline.dry_run is False
    assert config.pipeline.checkpoint_file == "state/processed_ids.json"


def test_load_config_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_invalid_model(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("detection:\n  model: invalid\n", encoding="utf-8")

    with pytest.raises(Exception):
        load_config(cfg_file)


def test_load_config_env_override_icloud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ICLOUD_USERNAME", "myappleid@example.com")
    monkeypatch.setenv("ICLOUD_PASSWORD", "app-specific-pw")

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("icloud:\n  enabled: true\n", encoding="utf-8")

    config = load_config(cfg_file)

    assert config.icloud.username == "myappleid@example.com"
    assert config.icloud.password == "app-specific-pw"


def test_load_config_env_override_nextcloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEXTCLOUD_USERNAME", "ncuser")
    monkeypatch.setenv("NEXTCLOUD_PASSWORD", "ncpass")

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("nextcloud:\n  url: 'https://nc.local'\n", encoding="utf-8")

    config = load_config(cfg_file)

    assert config.nextcloud.username == "ncuser"
    assert config.nextcloud.password == "ncpass"
