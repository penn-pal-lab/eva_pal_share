"""Tests for ``eva.controllers.molmoact2.MolmoAct2Policy``.

Construction does not require the server. ``forward()`` is exercised in two
modes:
- stopped (returns the hold action),
- running with the HTTP call mocked to return a synthetic action chunk.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eva.controllers.molmoact2 import MolmoAct2Policy
from _helpers import (  # type: ignore[import]
    assert_action_pair,
    assert_controller_interface,
    make_obs,
)


@pytest.fixture()
def molmo() -> MolmoAct2Policy:
    return MolmoAct2Policy()


def test_interface(molmo: MolmoAct2Policy) -> None:
    assert_controller_interface(molmo)
    assert molmo.action_space == "joint_position"
    assert molmo.gripper_action_space == "position"
    assert molmo.open_loop_horizon == 15
    # current_instruction is what Runner saves to instruction.txt.
    assert isinstance(molmo.current_instruction, str)


def test_set_instruction(molmo: MolmoAct2Policy) -> None:
    molmo.set_instruction("pick the red cup")
    assert molmo.current_instruction == "pick the red cup"


def test_forward_when_stopped_returns_hold(molmo: MolmoAct2Policy) -> None:
    molmo.stop_policy()
    result = molmo.forward(make_obs())
    assert_action_pair(result, expected_dim=8)


def test_forward_with_mocked_server(molmo: MolmoAct2Policy) -> None:
    fake_chunk = np.zeros((molmo.open_loop_horizon, 8), dtype=np.float32)
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"actions": fake_chunk.tolist()}

    with patch("eva.controllers.molmoact2.requests.post", return_value=fake_resp):
        result = molmo.forward(make_obs())

    assert_action_pair(result, expected_dim=8)
    # After a successful query, the chunk should be populated.
    assert molmo.pred_action_chunk is not None
    assert molmo.policy_query_count == 1


def test_register_key_success_failure(molmo: MolmoAct2Policy) -> None:
    molmo.register_key(ord("y"))
    assert molmo.get_info()["success"] is True
    molmo.reset_state()
    molmo.register_key(ord("n"))
    assert molmo.get_info()["failure"] is True
