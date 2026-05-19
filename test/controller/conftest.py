"""Pytest bootstrap for `test/controller/`.

The controller modules pull in hardware drivers (hid, oculus_reader, zmq),
network policy clients (openpi_client), and ML stacks (transformers,
sklearn.decomposition, eva.detectors.dinox_detector) at *import time*.
None of those are available in a clean CI/laptop checkout.

This conftest installs lightweight stand-ins into ``sys.modules`` BEFORE
any test file imports a controller, plus a couple of small patches to
``eva.utils.parameters`` so module-level dataclasses construct cleanly.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


# ---------------------------------------------------------------------------
# Make repo root importable so `import eva.controllers...` works when pytest
# is run from anywhere.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# cv2.aruco compatibility shim. The repo uses the legacy ``Dictionary_get`` API
# (removed in OpenCV >= 4.7); patch it back so eva.utils.parameters imports.
# ---------------------------------------------------------------------------
try:
    import cv2  # noqa: E402

    if hasattr(cv2, "aruco") and not hasattr(cv2.aruco, "Dictionary_get"):
        cv2.aruco.Dictionary_get = lambda d: cv2.aruco.getPredefinedDictionary(d)  # type: ignore[attr-defined]
except Exception:
    pass


# ---------------------------------------------------------------------------
# eva.utils.geometry_utils.rotation_matrix is referenced by spacemouse.py but
# isn't defined in the module. Inject a minimal implementation so the import
# resolves; the SpaceMouse controller only uses it inside the HID read loop
# which we never trigger in tests.
# ---------------------------------------------------------------------------
import eva.utils.geometry_utils as _geom  # noqa: E402

if not hasattr(_geom, "rotation_matrix"):
    def _rotation_matrix(angle, direction):
        d = np.asarray(direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)
        c, s = np.cos(angle), np.sin(angle)
        C = 1.0 - c
        x, y, z = d
        R = np.array(
            [
                [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C    ],
            ]
        )
        T = np.eye(4)
        T[:3, :3] = R
        return T

    _geom.rotation_matrix = _rotation_matrix


# ---------------------------------------------------------------------------
# sys.modules shims for optional / hardware deps.
# ---------------------------------------------------------------------------

def _install(name: str, module: types.ModuleType) -> None:
    sys.modules.setdefault(name, module)


def _make_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# --- HID (SpaceMouse) ------------------------------------------------------
class _FakeHidDevice:
    def __init__(self):
        self._opened = False

    def open(self, vendor_id, product_id):
        self._opened = True

    def get_manufacturer_string(self):
        return "fake-manufacturer"

    def get_product_string(self):
        return "fake-spacemouse"

    def set_nonblocking(self, flag):
        pass

    def read(self, n):
        return []

    def close(self):
        self._opened = False


_hid = _make_module(
    "hid",
    device=lambda: _FakeHidDevice(),
    enumerate=lambda *a, **k: [],
)
_install("hid", _hid)


# --- oculus_reader ---------------------------------------------------------
class _FakeOculusReader:
    def __init__(self, *a, **k):
        pass

    def get_transformations_and_buttons(self):
        return {}, {}

    def stop(self):
        pass


_oculus_pkg = _make_module("oculus_reader")
_oculus_reader_mod = _make_module("oculus_reader.reader", OculusReader=_FakeOculusReader)
_install("oculus_reader", _oculus_pkg)
_install("oculus_reader.reader", _oculus_reader_mod)


# --- zmq (Gello) -----------------------------------------------------------
class _FakeZmqSocket:
    def connect(self, *a, **k):
        pass

    def send(self, *a, **k):
        pass

    def recv(self):
        return np.zeros(8, dtype=np.float32).tobytes()


class _FakeZmqContext:
    def socket(self, *a, **k):
        return _FakeZmqSocket()


_zmq = _make_module("zmq", Context=_FakeZmqContext, REQ=1)
_install("zmq", _zmq)


# --- openpi_client (pi0 family) -------------------------------------------
class _FakeWebsocketClientPolicy:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def get_server_metadata(self):
        return {"policy_name": "fake-pi0"}

    def infer(self, payload):
        # Return a chunk of 8-DoF actions long enough to fill any open-loop horizon.
        return {"actions": np.zeros((32, 8), dtype=np.float32)}


_openpi = _make_module("openpi_client")
_openpi_image = _make_module(
    "openpi_client.image_tools",
    resize_with_pad=lambda img, w, h: img,
    convert_to_uint8=lambda img: img.astype(np.uint8),
)
_openpi_ws = _make_module(
    "openpi_client.websocket_client_policy",
    WebsocketClientPolicy=_FakeWebsocketClientPolicy,
)
_install("openpi_client", _openpi)
_install("openpi_client.image_tools", _openpi_image)
_install("openpi_client.websocket_client_policy", _openpi_ws)


# --- json_numpy (molmoact2) ------------------------------------------------
_json_numpy = _make_module(
    "json_numpy",
    patch=lambda: None,
    dumps=lambda obj: "{}",
    loads=lambda s: {},
)
_install("json_numpy", _json_numpy)


# --- transformers / sklearn.decomposition (aawr) ---------------------------
# These are heavy and only needed for aawr/policy. Stub them so the imports
# at the top of those modules succeed; the tests for those controllers skip
# anything that actually invokes them.
if "transformers" not in sys.modules:
    try:
        import transformers  # noqa: F401
    except Exception:
        _install(
            "transformers",
            _make_module(
                "transformers",
                AutoImageProcessor=MagicMock(),
                AutoModel=MagicMock(),
            ),
        )

if "sklearn" not in sys.modules:
    try:
        import sklearn  # noqa: F401
    except Exception:
        _install("sklearn", _make_module("sklearn"))
        _install(
            "sklearn.decomposition",
            _make_module("sklearn.decomposition", PCA=MagicMock()),
        )
else:
    try:
        import sklearn.decomposition  # noqa: F401
    except Exception:
        _install(
            "sklearn.decomposition",
            _make_module("sklearn.decomposition", PCA=MagicMock()),
        )


# --- eva.detectors.dinox_detector -----------------------------------------
# Importing the real one needs `dds_cloudapi_sdk` + a live API token. Replace
# the module entirely so controllers that import DINOX get a no-op class.
class _FakeDINOX:
    def __init__(self, *a, **k):
        pass

    def get_dinox(self, image_path, input_prompts=None, **k):
        return {"objects": [], "boxes": [], "scores": []}


_install(
    "eva.detectors.dinox_detector",
    _make_module("eva.detectors.dinox_detector", DINOX=_FakeDINOX),
)


# --- ably (used transitively via eva.remote_timer) -------------------------
class _FakeAblyRestSync:
    def __init__(self, *a, **k):
        self.channels = MagicMock()


if "ably" not in sys.modules:
    _install("ably", _make_module("ably"))
if "ably.sync" not in sys.modules:
    _install(
        "ably.sync",
        _make_module("ably.sync", AblyRestSync=_FakeAblyRestSync),
    )


# ---------------------------------------------------------------------------
# Patch eva.utils.parameters for fields that are referenced but missing.
# ---------------------------------------------------------------------------
import eva.utils.parameters as _params  # noqa: E402

if not hasattr(_params, "SPACEMOUSE_OVERRIDE_CONFIG"):
    _params.SPACEMOUSE_OVERRIDE_CONFIG = False
if not hasattr(_params, "spacemouse_config"):
    _params.spacemouse_config = {}


# ---------------------------------------------------------------------------
# Silence run_threaded_command so controller __init__ doesn't spawn busy
# background loops that hammer the shimmed sockets / readers.
# ---------------------------------------------------------------------------
import eva.utils.misc_utils as _misc_utils  # noqa: E402


def _noop_run_threaded_command(command, args=(), daemon=True):
    fake = MagicMock()
    fake.is_alive.return_value = False
    return fake


_misc_utils.run_threaded_command = _noop_run_threaded_command
