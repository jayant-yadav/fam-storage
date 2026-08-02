"""Configuration management for fam-storage.

Reads settings from a YAML config file and overrides values from
environment variables (prefixed with FAM_STORAGE_).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class GooglePhotosConfig(BaseModel):
    enabled: bool = True
    credentials_file: str = "config/google_oauth_credentials.json"
    token_file: str = "config/google_token.json"
    page_size: int = Field(default=100, ge=1, le=100)


class ICloudConfig(BaseModel):
    enabled: bool = True
    username: str = ""
    password: str = ""
    cookie_dir: str = "config/icloud_cookies"

    @model_validator(mode="after")
    def _apply_env_vars(self) -> "ICloudConfig":
        """Override username/password from environment variables if set."""
        env_user = os.environ.get("ICLOUD_USERNAME")
        if env_user:
            self.username = env_user
        env_pass = os.environ.get("ICLOUD_PASSWORD")
        if env_pass:
            self.password = env_pass
        return self


class DetectionConfig(BaseModel):
    tolerance: float = Field(default=0.6, ge=0.0, le=1.0)
    model: str = Field(default="hog", pattern="^(hog|cnn)$")
    max_workers: int = Field(default=16, ge=1, le=256)
    batch_size: int = Field(default=8, ge=1)
    encodings_cache: str = "state/face_encodings.pkl"


class NextcloudConfig(BaseModel):
    url: str = ""
    username: str = ""
    password: str = ""
    remote_base_path: str = "/FamilyPhotos"
    verify_ssl: bool = True

    @model_validator(mode="after")
    def _apply_env_vars(self) -> "NextcloudConfig":
        """Override username/password from environment variables if set."""
        env_user = os.environ.get("NEXTCLOUD_USERNAME")
        if env_user:
            self.username = env_user
        env_pass = os.environ.get("NEXTCLOUD_PASSWORD")
        if env_pass:
            self.password = env_pass
        return self


class PipelineConfig(BaseModel):
    checkpoint_file: str = "state/processed_ids.json"
    download_timeout: int = Field(default=30, ge=1)
    dry_run: bool = False
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    google_photos: GooglePhotosConfig = GooglePhotosConfig()
    icloud: ICloudConfig = ICloudConfig()
    references_dir: str = "references"
    detection: DetectionConfig = DetectionConfig()
    nextcloud: NextcloudConfig = NextcloudConfig()
    pipeline: PipelineConfig = PipelineConfig()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> AppConfig:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated :class:`AppConfig` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file contains invalid values.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return AppConfig.model_validate(raw)
