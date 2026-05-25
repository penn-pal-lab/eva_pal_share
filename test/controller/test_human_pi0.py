"""Tests for ``eva.controllers.human_pi0.DemoDiffusionPolicy``.

The constructor reads a trajectory from disk via ``load_trajectory_data``.
We patch that to return synthetic data so the test does not depend on
the human-data directory (``$EVA_HUMAN_DATA``).
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import eva.controllers.human_pi0 as human_pi0
from _helpers import (  # type: ignore[import]
    assert_action_pair,
    assert_controller_interface,
    make_obs,
)


_FAKE_TRAJ_LEN = 32


def _fake_load_trajectory_data(config):
    eef = np.zeros((_FAKE_TRAJ_LEN, 7), dtype=np.float32)
    # Quaternion identity so quat_to_rmat doesn't blow up.
    eef[:, 6] = 1.0
    grip = np.zeros(_FAKE_TRAJ_LEN, dtype=np.float32)
    return "/tmp/fake", eef, grip


@pytest.fixture()
def policy():
    with patch.object(human_pi0, "load_trajectory_data", _fake_load_trajectory_data):
        yield human_pi0.DemoDiffusionPolicy()


def test_interface(policy) -> None:
    assert_controller_interface(policy)
    assert policy.action_space == "joint_velocity"
    assert policy.gripper_action_space == "position"
    assert policy.get_name() == "pi0-droid"


def test_set_instruction(policy) -> None:
    policy.set_instruction("open the drawer")
    assert policy.current_instruction == "open the drawer"


def test_forward_when_stopped_returns_8d_zero(policy) -> None:
    policy.stop_policy()
    result = policy.forward(make_obs())
    assert_action_pair(result, expected_dim=8)
    action, _ = result
    assert np.allclose(np.asarray(action), 0.0)
