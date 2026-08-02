# fam-storage: Family Photo Extraction Pipeline — Design Plan

## Overview

`fam-storage` is a local-first pipeline that pulls photos from cloud sources (Google Photos and iCloud), identifies pictures of pre-registered loved ones using an **on-device** Vision Language Model (VLM), and stores the matched photos into a self-hosted **Nextcloud** instance backed by a NAS drive.

All AI inference runs **on-device** on a 16–32 core CPU. No images or biometric data leave the local network.

---

## Goals

1. Connect to Google Photos and iCloud and enumerate photos.
2. Detect and recognize faces of pre-registered people (family members) in each photo using an on-device model.
3. Store matched photos into a Nextcloud (WebDAV) folder, organized by person.
4. Support incremental runs (skip already-processed photos).
5. Be configurable via a single YAML file and environment variables.

---

## Non-Goals

- Cloud-based AI/ML inference
- Real-time streaming (this is a batch pipeline)
- Video support (images only in this version)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       fam-storage pipeline                      │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐                           │
│  │ Google Photos│   │    iCloud    │  ← Source Connectors      │
│  │  Connector   │   │  Connector   │                           │
│  └──────┬───────┘   └──────┬───────┘                           │
│         │                  │                                    │
│         └────────┬─────────┘                                    │
│                  ▼                                              │
│        ┌──────────────────┐                                     │
│        │  Image Download  │  (streaming / chunked)             │
│        │     Queue        │                                     │
│        └────────┬─────────┘                                     │
│                 │                                               │
│                 ▼                                               │
│   ┌─────────────────────────────┐                              │
│   │  On-Device Face Detection   │  face_recognition (dlib)     │
│   │  & Recognition Engine       │  + multiprocessing pool      │
│   │  (16–32 cores, CPU-parallel)│  (configurable workers)      │
│   └─────────────┬───────────────┘                              │
│                 │ matched photos                                │
│                 ▼                                               │
│        ┌─────────────────┐                                      │
│        │  Nextcloud NAS  │  WebDAV upload                      │
│        │  Storage Layer  │  organized by person / date         │
│        └─────────────────┘                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. Source Connectors (`src/fam_storage/connectors/`)

#### Google Photos (`google_photos.py`)
- Uses the [Google Photos Library REST API](https://developers.google.com/photos/library/reference/rest) with OAuth 2.0.
- Supports incremental sync via `pageToken`.
- Downloads images at original resolution using `baseUrl` + `=d` suffix.

**Scopes required:**
- `https://www.googleapis.com/auth/photoslibrary.readonly`

#### iCloud (`icloud.py`)
- Uses the [`pyicloud`](https://github.com/picklepete/pyicloud) library.
- Authenticates with Apple ID + app-specific password.
- Iterates over the `Photos` service albums.

**Credentials:**
- `ICLOUD_USERNAME` / `ICLOUD_PASSWORD` env vars (or config file)
- Supports 2FA prompt in interactive mode; stores session cookie for non-interactive re-use

---

### 2. Reference Person Registry (`src/fam_storage/registry/`)

- User provides a folder of reference images per person (e.g. `references/mom/`, `references/dad/`).
- At startup, the detector encodes all reference images into 128-d face embeddings using `face_recognition`.
- Embeddings are cached to a `.pkl` file to avoid re-encoding on every run.

---

### 3. On-Device VLM Detection Engine (`src/fam_storage/detection/`)

#### Technology choices (CPU-only, 16–32 cores)

| Component | Library | Rationale |
|-----------|---------|-----------|
| Face detection | `face_recognition` (dlib HOG) | Fast CPU inference; no GPU required |
| Face embedding | `face_recognition` 128-d embeddings | Proven accuracy on family-scale datasets |
| Parallelism | `concurrent.futures.ProcessPoolExecutor` | Uses all cores; avoids Python GIL |

#### Detection flow per image
1. Decode the image from bytes.
2. Detect face bounding boxes (HOG model, CPU-fast).
3. Compute 128-d embeddings for each detected face.
4. Compare each embedding against all registered reference embeddings using L2 distance.
5. If any face matches within `tolerance` (default 0.6), mark the photo as a match and record which person(s) were found.

#### Parallelism

```
worker_count = min(cpu_count, max_workers_config)   # e.g. 16
ProcessPoolExecutor(max_workers=worker_count)
```

Each worker receives a batch of downloaded image bytes and returns a list of `DetectionResult` objects. The main process aggregates results and enqueues matched images for upload.

---

### 4. Nextcloud Storage Layer (`src/fam_storage/storage/`)

- Uses **WebDAV** (standard HTTP `PUT`) — no proprietary Nextcloud client library required.
- Library: `webdavclient3` (pure Python, no native deps).
- Uploads are organized as:
  ```
  /FamilyPhotos/{person_name}/{YYYY}/{MM}/{original_filename}
  ```
- Deduplication: before uploading, the layer checks whether the file already exists on the remote (HTTP `HEAD` request). Skipped if it does.

---

### 5. Pipeline Orchestrator (`src/fam_storage/pipeline.py`)

The orchestrator ties everything together:

1. Load configuration.
2. Build face registry from reference images.
3. For each enabled source connector, stream media items in pages.
4. Download each image into memory (no disk I/O by default).
5. Submit batches to the detection pool.
6. For each matched image, upload to Nextcloud.
7. Persist a checkpoint file (`processed_ids.json`) to support resumable runs.

---

## Configuration (`config/config.yaml`)

```yaml
google_photos:
  enabled: true
  credentials_file: "config/google_oauth_credentials.json"
  token_file: "config/google_token.json"

icloud:
  enabled: true
  username: ""          # or set ICLOUD_USERNAME env var
  password: ""          # or set ICLOUD_PASSWORD env var
  cookie_dir: "config/icloud_cookies"

references_dir: "references"    # sub-dirs named after each person

detection:
  tolerance: 0.6        # face match tolerance (lower = stricter)
  model: "hog"          # "hog" (fast) or "cnn" (more accurate, slower)
  max_workers: 16       # parallel detection workers
  batch_size: 8         # images per worker batch

nextcloud:
  url: "https://nas.local/nextcloud"
  username: ""          # or set NEXTCLOUD_USERNAME env var
  password: ""          # or set NEXTCLOUD_PASSWORD env var
  remote_base_path: "/FamilyPhotos"

pipeline:
  checkpoint_file: "state/processed_ids.json"
  download_timeout: 30  # seconds per image download
  log_level: "INFO"
```

---

## Security & Privacy

- **No cloud AI**: All face recognition runs on the local CPU.
- **Credentials**: OAuth tokens, Apple ID credentials, and Nextcloud passwords are stored only locally (config file or env vars). Never committed to source control.
- **Reference images**: Biometric face encodings are stored in a local `.pkl` cache only.
- **TLS**: All connections to Google, Apple, and Nextcloud use HTTPS.
- **`.gitignore`**: `config/*.json`, `config/*.yaml`, `state/`, `references/`, `*.pkl` are excluded from git.

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- A Linux/macOS device with 16–32 CPU cores
- Google Cloud project with Photos Library API enabled
- Apple ID with app-specific password
- Nextcloud instance accessible via WebDAV

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure

```bash
cp config/config.yaml.example config/config.yaml
# Edit config/config.yaml with your credentials
```

### Register Reference People

```bash
mkdir -p references/mom references/dad references/partner
# Copy 3–10 clear face photos into each sub-directory
```

### Run

```bash
# First run: authenticate Google Photos (opens browser)
fam-storage run --config config/config.yaml

# Subsequent runs use saved tokens
fam-storage run --config config/config.yaml

# Dry-run (no upload, just detect)
fam-storage run --config config/config.yaml --dry-run
```

---

## Testing Strategy

- **Unit tests** for each component with mocked external calls.
- **Integration test** (`tests/integration/`) runs the full pipeline against a mock HTTP server (using `pytest-httpserver` or `responses` library).
- Run tests: `pytest tests/ -v`

---

## Future Improvements

- Add support for more cloud sources (Dropbox, OneDrive, Amazon Photos).
- Support GPU-accelerated CNN face models when a CUDA GPU is available.
- Add a web UI dashboard for reviewing matches before upload.
- Support video thumbnails.
- Incremental delta sync using Google Photos change tokens.
