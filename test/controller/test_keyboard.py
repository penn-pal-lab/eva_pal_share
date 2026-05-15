"""Smoke tests for ``eva.controllers.keyboard.Keyboard``."""
from __future__ import annotations

import numpy as np
import pytest

from eva.controllers.keyboard import Keyboard
from _helpers import (  # type: ignore[import]
    assert_action_pair,
    assert_controller_interface,
    make_obs,
)


@pytest.fixture()
def kb() -> Keyboard:
    return Keyboard()


def test_interface(kb: Keyboard) -> None:
    assert_controller_interface(kb)
    assert kb.action_space == "cartesian_velocity"
    assert kb.gripper_action_space == "velocity"
    assert kb.get_name() == "keyboard"


def test_forward_returns_7d_cartesian_velocity(kb: Keyboard) -> None:
    result = kb.forward(make_obs())
    assert_action_pair(result, expected_dim=7)
    action, _ = result
    assert np.all(np.abs(np.asarray(action)) <= 1.0), "actions must be clipped to [-1, 1]"


def test_register_key_toggles_movement(kb: Keyboard) -> None:
    assert kb.get_info()["movement_enabled"] is False
    kb.register_key(ord(" "))
    assert kb.get_info()["movement_enabled"] is True
    kb.register_key(ord(" "))
    assert kb.get_info()["movement_enabled"] is False


def test_register_key_success_and_failure(kb: Keyboard) -> None:
    kb.register_key(13)  # Enter -> success
    assert kb.get_info()["success"] is True

    kb.reset_state()
    kb.register_key(8)  # Backspace -> failure
    assert kb.get_info()["failure"] is True


def test_close_runs(kb: Keyboard) -> None:
    kb.close()  # should not raise
