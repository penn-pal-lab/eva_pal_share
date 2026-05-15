"""Tests for ``eva.controllers.aawr_pi0.AAWRPi0Controller``.

The constructor builds both a ``Pi0Policy`` (server stubbed in conftest) and
an ``AAWRPolicy`` (loads ML weights from disk). We patch ``AAWRPolicy`` to a
``MagicMock`` so construction succeeds, then exercise the wiring.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def aawr_pi0():
    fake_aawr = MagicMock()
    fake_aawr.action_space = "cartesian_velocity"
    fake_aawr.gripper_action_space = "velocity"
    fake_aawr.get_name.return_value = "fake-aawr"
    fake_aawr.get_policy_name.return_value = "fake-aawr-path"
    fake_aawr.get_info.return_value = {
        "success": False,
        "failure": False,
        "movement_enabled": True,
        "controller_on": True,
    }

    with patch(
        "eva.controllers.aawr_pi0.AAWRPolicy",
        return_value=fake_aawr,
    ):
        from eva.controllers.aawr_pi0 import AAWRPi0Controller
        yield AAWRPi0Controller()


def test_construction_succeeds(aawr_pi0) -> None:
    assert aawr_pi0 is not None
    assert aawr_pi0.get_name() == "aawr-pi0"


def test_delegates_action_space_to_active_controller(aawr_pi0) -> None:
    current = aawr_pi0.get_current_controller()
    assert aawr_pi0.action_space == current.action_space
    assert aawr_pi0.gripper_action_space == current.gripper_action_space


def test_reset_state_returns_dict(aawr_pi0) -> None:
    aawr_pi0.reset_state()
    info = aawr_pi0.get_info()
    assert isinstance(info, dict)
