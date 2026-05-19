"""Smoke tests for ``eva.controllers.gello.Gello``.

The ZMQ GELLODevice is replaced via the ``zmq`` shim in ``conftest.py``.
``run_threaded_command`` is also patched out, so ``_update_internal_state``
does not run as a background thread — the controller's ``_state`` therefore
stays in its post-``reset_state`` shape, which is what we exercise here.
"""
from __future__ import annotations

import pytest

from eva.controllers.gello import Gello
from _helpers import assert_controller_interface  # type: ignore[import]


@pytest.fixture()
def gello() -> Gello:
    return Gello()


def test_interface(gello: Gello) -> None:
    assert_controller_interface(gello)
    assert gello.action_space == "joint_position"
    assert gello.gripper_action_space == "velocity"
    assert gello.get_name() == "gello"


def test_reset_state_populates_buttons(gello: Gello) -> None:
    gello.reset_state()
    info = gello.get_info()
    assert "movement_enabled" in info
    assert info["movement_enabled"] is False  # disabled until user enables


def test_register_key_does_not_raise(gello: Gello) -> None:
    gello.register_key(ord("y"))
    gello.register_key(ord("n"))


def test_close_stops_running(gello: Gello) -> None:
    gello.close()
    assert gello.running is False
