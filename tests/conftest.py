"""Shared pytest fixtures and conftest for the fam-storage test suite.

Injects a lightweight stub for optional heavy native dependencies
(``face_recognition``, ``pyicloud``) so that all unit tests can run
without requiring compiled native extensions (dlib, etc.).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub face_recognition module
# ---------------------------------------------------------------------------

def _make_face_recognition_stub() -> types.ModuleType:
    """Build a minimal stub that satisfies import statements."""
    stub = types.ModuleType("face_recognition")

    stub.load_image_file = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))
    stub.face_locations = MagicMock(return_value=[])
    stub.face_encodings = MagicMock(return_value=[])
    stub.face_distance = MagicMock(return_value=np.array([0.5]))

    return stub


def _make_pyicloud_stub() -> types.ModuleType:
    """Build a minimal stub for pyicloud."""
    stub = types.ModuleType("pyicloud")

    exceptions_mod = types.ModuleType("pyicloud.exceptions")
    exceptions_mod.PyiCloudFailedLoginException = Exception
    exceptions_mod.PyiCloudAPIResponseException = Exception
    sys.modules["pyicloud.exceptions"] = exceptions_mod

    stub.PyiCloudService = MagicMock()
    stub.exceptions = exceptions_mod

    return stub


@pytest.fixture(autouse=True, scope="session")
def inject_optional_stubs() -> None:
    """Inject stubs for optional native modules before any tests run."""
    if "face_recognition" not in sys.modules:
        sys.modules["face_recognition"] = _make_face_recognition_stub()
    if "pyicloud" not in sys.modules:
        sys.modules["pyicloud"] = _make_pyicloud_stub()
