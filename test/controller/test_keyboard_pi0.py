"""Tests for ``eva.controllers.keyboard_pi0.KeyboardPi0``.

This is a runtime-switch wrapper. We exercise both branches via its
``_state["current_controller"]`` flag while the PI0 client is stubbed by
the openpi_client shim in ``conftest.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from eva.controllers.keyboard_pi0 import KeyboardPi0
from _helpers import (  # type: ignore[import]
    assert_action_pair,
    assert_controller_interface,
    make_obs,
)


@pytest.fixture()
def kbp() -> KeyboardPi0:
    ctrl = KeyboardPi0()
    ctrl.START_FLAG = False  # bypass the interactive prompt in forward()
    return ctrl


def test_interface_delegates_to_active_controller(kbp: KeyboardPi0) -> None:
    assert_controller_interface(kbp)
    assert kbp.get_name() == "keyboard-pi0-controller"
    # Defaults to keyboard subcontroller (current_controller=1).
    assert kbp.action_space == kbp.keyboard_controller.action_space
    assert kbp.gripper_action_space == kbp.keyboard_controller.gripper_action_space


def test_forward_via_keyboard_branch_returns_7d(kbp: KeyboardPi0) -> None:
    assert kbp._state["current_controller"] == 1  # keyboard
    result = kbp.forward(make_obs())
    assert_action_pair(result, expected_dim=7)


def test_forward_via_pi0_branch_stopped_returns_8d_zero(kbp: KeyboardPi0) -> None:
    # Switch to PI0 branch but keep the policy stopped so forward returns zeros.
    kbp._state["current_controller"] = 0
    kbp.pi0_controller.stop_policy()
    result = kbp.forward(make_obs())
    assert_action_pair(result, expected_dim=8)
    action, _ = result
    assert np.allclose(np.asarray(action), 0.0)


def test_set_instruction_routes_to_pi0(kbp: KeyboardPi0) -> None:
    kbp.set_instruction("grab the marker")
    assert kbp.pi0_controller.current_instruction == "grab the marker"
