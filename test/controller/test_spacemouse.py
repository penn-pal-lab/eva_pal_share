"""Smoke tests for ``eva.controllers.spacemouse.SpaceMouse``.

The real device is replaced via the ``hid`` shim in ``conftest.py``. We bypass
the interactive ``input()`` call in ``forward`` by flipping ``first_action``.
"""
from __future__ import annotations

import numpy as np
import pytest

from eva.controllers.spacemouse import SpaceMouse
from _helpers import (  # type: ignore[import]
    assert_action_pair,
    assert_controller_interface,
    make_obs,
)


@pytest.fixture()
def sm() -> SpaceMouse:
    ctrl = SpaceMouse()
    ctrl.first_action = False  # skip the input() prompt in forward()

    def _fake_state():
        return {
            "dpos": np.zeros(3),
            "raw_drotation": np.zeros(3),
            "grasp": False,
        }

    ctrl.interface.get_controller_state = _fake_state  # type: ignore[assignment]
    return ctrl


def test_interface(sm: SpaceMouse) -> None:
    assert_controller_interface(sm)
    assert sm.action_space == "cartesian_velocity"
    assert sm.gripper_action_space == "position"
    assert sm.get_name() == "spacemouse"
    assert sm.get_policy_name() == "spacemouse-data"


def test_forward_returns_7d_action(sm: SpaceMouse) -> None:
    result = sm.forward(make_obs())
    assert_action_pair(result, expected_dim=7)


def test_macro_queue_via_register_key(sm: SpaceMouse) -> None:
    sm.register_key(ord("w"))  # move_forward macro
    assert len(sm._macro_queue) == 1
    # On the next forward, the macro should be popped and applied.
    action, _ = sm.forward(make_obs())
    assert np.linalg.norm(np.asarray(action)) > 0, "macro should drive a non-zero action"


def test_success_failure_keys(sm: SpaceMouse) -> None:
    sm.register_key(ord("y"))
    assert sm.get_info()["success"] is True
    sm.reset_state()
    sm.register_key(ord("n"))
    assert sm.get_info()["failure"] is True
