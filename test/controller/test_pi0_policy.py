"""Tests for ``eva.controllers.pi0_policy.Pi0Policy``.

The policy server is stubbed via ``openpi_client.websocket_client_policy`` in
``conftest.py``. We exercise the controller in stopped mode (no server call
at all) and via the running path with the websocket policy returning a
synthetic action chunk.
"""
from __future__ import annotations

import numpy as np
import pytest

from eva.controllers.pi0_policy import Pi0Policy
from _helpers import (  # type: ignore[import]
    assert_action_pair,
    assert_controller_interface,
    make_obs,
)


@pytest.fixture()
def pi0() -> Pi0Policy:
    return Pi0Policy()


def test_interface(pi0: Pi0Policy) -> None:
    assert_controller_interface(pi0)
    assert pi0.action_space == "joint_velocity"
    assert pi0.gripper_action_space == "position"
    assert pi0.open_loop_horizon == 8


def test_set_instruction(pi0: Pi0Policy) -> None:
    pi0.set_instruction("hello world")
    assert pi0.current_instruction == "hello world"


def test_set_horizon(pi0: Pi0Policy) -> None:
    pi0.set_horizon(16)
    assert pi0.open_loop_horizon == 16


def test_forward_when_stopped_returns_zero(pi0: Pi0Policy) -> None:
    pi0.stop_policy()
    result = pi0.forward(make_obs())
    assert_action_pair(result, expected_dim=8)
    action, _ = result
    assert np.allclose(np.asarray(action), 0.0)


def test_register_key_success_failure(pi0: Pi0Policy) -> None:
    pi0.register_key(ord("y"))
    assert pi0.get_info()["success"] is True
    pi0.reset_state()
    pi0.register_key(ord("n"))
    assert pi0.get_info()["failure"] is True
