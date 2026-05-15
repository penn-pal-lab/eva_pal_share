"""Tests for ``eva.controllers.replay_pi0.ReplayPi0Controller``.

Combines a stubbed PI0 client (openpi shim in conftest) with a Replayer that
loads a trajectory from disk. We patch ``numpy.load`` so the Replayer gets a
synthetic .npz, and DINOX is already stubbed in conftest.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture()
def replay_pi0():
    # Replayer with .npz path checks self.action_space; default is cartesian_position
    # which loads key "actions_pos".
    fake_actions = np.tile(
        np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        (5, 1),
    )

    class _FakeNpz:
        def __getitem__(self, k):
            return fake_actions

    with patch("numpy.load", return_value=_FakeNpz()):
        from eva.controllers.replay_pi0 import ReplayPi0Controller
        # Use a synthetic .npz path so the Replayer branch triggers.
        from eva.controllers.replay_pi0 import ReplayConfig
        cfg = ReplayConfig(traj_path="/tmp/synthetic.npz")
        yield ReplayPi0Controller(config=cfg)


def test_construction_succeeds(replay_pi0) -> None:
    assert replay_pi0 is not None


def test_delegates_action_space_to_active_controller(replay_pi0) -> None:
    current = replay_pi0.get_current_controller()
    assert replay_pi0.action_space == current.action_space
    assert replay_pi0.gripper_action_space == current.gripper_action_space


def test_reset_state_returns_dict(replay_pi0) -> None:
    replay_pi0.reset_state()
    info = replay_pi0.get_info()
    assert isinstance(info, dict)
