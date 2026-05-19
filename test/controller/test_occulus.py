"""Smoke tests for ``eva.controllers.occulus.Occulus``.

The Quest VR reader is replaced via the ``oculus_reader.reader`` shim in
``conftest.py`` and the background thread is suppressed by the patched
``run_threaded_command``. We can therefore only validate the static interface;
``forward()`` requires a populated ``_state["poses"]`` which only the live
reader produces.
"""
from __future__ import annotations

import pytest

from eva.controllers.occulus import Occulus
from _helpers import assert_controller_interface  # type: ignore[import]


@pytest.fixture()
def occulus() -> Occulus:
    return Occulus()


def test_interface(occulus: Occulus) -> None:
    assert_controller_interface(occulus)
    assert occulus.action_space == "cartesian_velocity"
    assert occulus.gripper_action_space == "velocity"
    assert occulus.get_name() == "occulus"


def test_reset_state_shape(occulus: Occulus) -> None:
    occulus.reset_state()
    info = occulus.get_info()
    # get_info() pulls success/failure from _state["buttons"], which reset_state
    # initialises as a dict with A/B keys.
    assert "success" in info
    assert "failure" in info
    assert "movement_enabled" in info
    assert "controller_on" in info


def test_register_key_does_not_raise(occulus: Occulus) -> None:
    occulus.register_key(ord("y"))
    occulus.register_key(ord("n"))


def test_close_stops_running(occulus: Occulus) -> None:
    occulus.close()
    assert occulus.running is False
