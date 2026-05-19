"""Smoke tests for ``eva.controllers.replayer.Replayer``.

Replayer loads a trajectory from disk. We feed it a synthetic .npy by
monkeypatching ``numpy.load``.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from eva.controllers.replayer import Replayer
from _helpers import (  # type: ignore[import]
    assert_action_pair,
    assert_controller_interface,
    make_obs,
)


@pytest.fixture()
def replayer() -> Replayer:
    # 7-DoF trajectory: cartesian_position(6) + gripper(1)
    traj = np.tile(
        np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        (5, 1),
    )
    with patch("numpy.load", return_value=traj):
        return Replayer("/tmp/synthetic_traj.npy")


def test_interface(replayer: Replayer) -> None:
    assert_controller_interface(replayer)
    assert replayer.action_space == "cartesian_position"
    assert replayer.gripper_action_space == "position"
    assert replayer.get_name() == "Replayer"
    assert replayer.get_policy_name() == "/tmp/synthetic_traj.npy"


def test_forward_replays_then_holds(replayer: Replayer) -> None:
    obs = make_obs()
    for _ in range(replayer.traj_len):
        result = replayer.forward(obs)
        assert_action_pair(result, expected_dim=7)
    # Trajectory exhausted -> success flips True.
    assert replayer.get_info()["success"] is True


def test_register_key(replayer: Replayer) -> None:
    replayer.register_key(ord("y"))
    assert replayer.get_info()["success"] is True
    replayer.reset_state()
    replayer.register_key(ord("n"))
    assert replayer.get_info()["failure"] is True
