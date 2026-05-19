"""Tests for ``eva.controllers.pi0_spacemouse.SpaceMousePi0``.

The mixed controller wraps a stubbed PI0 client (openpi_client shim) and a
stubbed SpaceMouse (HID shim). We only exercise the wiring + the spaces.
"""
from __future__ import annotations

import numpy as np
import pytest

from eva.controllers.pi0_spacemouse import SpaceMousePi0
from _helpers import (  # type: ignore[import]
    assert_action_pair,
    assert_controller_interface,
    make_obs,
)


@pytest.fixture()
def smp() -> SpaceMousePi0:
    ctrl = SpaceMousePi0()
    # Make the spacemouse interface deterministic.
    ctrl.spacemouse_controller.first_action = False

    def _fake_state():
        return {
            "dpos": np.zeros(3),
            "raw_drotation": np.zeros(3),
            "grasp": False,
        }

    ctrl.spacemouse_controller.interface.get_controller_state = _fake_state  # type: ignore[assignment]
    return ctrl


def test_interface(smp: SpaceMousePi0) -> None:
    assert_controller_interface(smp)
    assert smp.get_name() == "SpaceMousePi0"


def test_forward_spacemouse_branch_returns_7d(smp: SpaceMousePi0) -> None:
    # The mixed controller's default state should route to one subcontroller.
    result = smp.forward(make_obs())
    assert isinstance(result, tuple) and len(result) == 2
    action, _ = result
    assert np.asarray(action).ndim == 1


def test_pi0_branch_stopped_returns_8d_zero(smp: SpaceMousePi0) -> None:
    # Force the wrapper into the PI0 branch (current_controller == 0).
    smp._state["current_controller"] = 0
    smp.pi0_controller.stop_policy()
    result = smp.forward(make_obs())
    assert_action_pair(result, expected_dim=8)
    action, _ = result
    assert np.allclose(np.asarray(action), 0.0)
