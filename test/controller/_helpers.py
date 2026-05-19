"""Shared helpers for controller tests.

Provides:
- ``make_obs`` — a synthetic observation dict matching what controllers expect
  from ``FrankaEnv`` (robot_state + image dict keyed by ``f"{cam_id}_{view}"``).
- ``assert_controller_interface`` — checks the small contract every controller
  must implement to plug into ``eva.runner.Runner``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

import eva.utils.parameters as params


# Standard EVA action-space names; used to sanity-check controller.action_space.
VALID_ACTION_SPACES = {
    "cartesian_position",
    "cartesian_velocity",
    "joint_position",
    "joint_velocity",
}
VALID_GRIPPER_SPACES = {"position", "velocity"}


def make_obs(
    cartesian_position: Iterable[float] = (0.4, 0.0, 0.3, 0.0, 0.0, 0.0),
    joint_positions: Iterable[float] = (0.0, -0.2, 0.0, -1.8, 0.0, 1.6, 0.7),
    gripper_position: float = 0.0,
    image_h: int = 180,
    image_w: int = 320,
    cam_ids: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """Build a fake observation dict in FrankaEnv's shape.

    The image dict keys follow ``"<serial>_<view>"`` where view is ``left`` /
    ``right`` (controllers like Pi0/MolmoAct2 filter on ``"left"`` in the key).
    Images are BGRA uint8, matching ZED output.
    """
    if cam_ids is None:
        cam_ids = (
            params.varied_camera_1_id,
            params.varied_camera_2_id,
            params.hand_camera_id,
        )

    def _bgra():
        return np.zeros((image_h, image_w, 4), dtype=np.uint8)

    image = {}
    for cam_id in cam_ids:
        image[f"{cam_id}_left"] = _bgra()
        image[f"{cam_id}_right"] = _bgra()

    return {
        "robot_state": {
            "cartesian_position": np.asarray(cartesian_position, dtype=np.float32),
            "joint_positions": np.asarray(joint_positions, dtype=np.float32),
            "gripper_position": float(gripper_position),
        },
        "image": image,
        "timestamp": {"skip_action": False, "robot_timestamp_seconds": 0.0},
    }


def assert_controller_interface(ctrl) -> None:
    """Assert the controller exposes the contract Runner depends on.

    See ``Runner.set_controller`` / ``Runner.run_trajectory`` for the consumers.
    """
    # Spaces ----------------------------------------------------------------
    assert hasattr(ctrl, "action_space"), "controller missing .action_space"
    assert hasattr(ctrl, "gripper_action_space"), "controller missing .gripper_action_space"
    assert ctrl.action_space in VALID_ACTION_SPACES, (
        f"unexpected action_space: {ctrl.action_space!r}"
    )
    assert ctrl.gripper_action_space in VALID_GRIPPER_SPACES, (
        f"unexpected gripper_action_space: {ctrl.gripper_action_space!r}"
    )

    # Methods ---------------------------------------------------------------
    for method in ("get_name", "reset_state", "register_key", "get_info", "forward"):
        assert callable(getattr(ctrl, method, None)), f"controller missing .{method}()"

    name = ctrl.get_name()
    assert isinstance(name, str) and name, "get_name() must return non-empty str"

    # reset_state / get_info round-trip ------------------------------------
    ctrl.reset_state()
    info = ctrl.get_info()
    assert isinstance(info, dict), "get_info() must return a dict"


def assert_action_pair(result, expected_dim: int | None = None) -> None:
    """Validate a ``forward()`` return value: ``(action_array, info_dict)``."""
    assert isinstance(result, tuple) and len(result) == 2, (
        f"forward() must return (action, info_dict), got {type(result)!r}"
    )
    action, info = result
    action_arr = np.asarray(action)
    assert action_arr.ndim == 1, f"action must be 1-D, got shape {action_arr.shape}"
    if expected_dim is not None:
        assert action_arr.shape[0] == expected_dim, (
            f"expected action dim {expected_dim}, got {action_arr.shape[0]}"
        )
    assert isinstance(info, dict), "info must be a dict"
