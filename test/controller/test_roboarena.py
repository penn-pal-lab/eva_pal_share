"""Smoke tests for ``eva.controllers.roboarena.RoboArenaController``.

The vendored websocket client is replaced with an in-memory fake passed via
``policy_client_cls`` so these tests don't need msgpack/websockets or a live
policy server.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from eva.controllers.roboarena import RoboArenaConfig, RoboArenaController
from _helpers import (  # type: ignore[import]
    assert_action_pair,
    assert_controller_interface,
    make_obs,
)


class _FakeWS:
    last: "List[_FakeWS]" = []

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        _FakeWS.last.append(self)

    def get_server_metadata(self) -> Dict[str, Any]:
        return {
            "policy_name": "fake-roboarena",
            "image_resolution": [224, 224],
            "needs_wrist_camera": True,
            "n_external_cameras": 1,
            "needs_stereo_camera": False,
            "needs_session_id": True,
            "action_space": "joint_velocity",
        }

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(("infer", payload))
        return {"actions": np.zeros((4, 8), dtype=np.float32)}

    def reset(self, info: Dict[str, Any]) -> None:
        self.calls.append(("reset", info))


@pytest.fixture(autouse=True)
def _clear_fake_clients() -> None:
    _FakeWS.last.clear()


@pytest.fixture()
def ctrl() -> RoboArenaController:
    return RoboArenaController(policy_client_cls=_FakeWS)


def test_interface(ctrl: RoboArenaController) -> None:
    assert_controller_interface(ctrl)
    assert ctrl.action_space == "joint_velocity"
    assert ctrl.gripper_action_space == "position"
    assert ctrl.open_loop_horizon == 8
    assert ctrl.get_name() == "roboarena"


def test_get_policy_name_is_json_with_session(ctrl: RoboArenaController) -> None:
    import json

    payload = json.loads(ctrl.get_policy_name())
    assert payload["controller"] == "roboarena"
    assert payload["session_id"] == ctrl.session_id
    assert payload["instruction"] == ""


def test_overrides_in_constructor() -> None:
    c = RoboArenaController(
        policy_client_cls=_FakeWS,
        remote_host="10.0.0.1",
        remote_port=9000,
        instruction="pick the red block",
    )
    assert c.config.remote_host == "10.0.0.1"
    assert c.config.remote_port == 9000
    assert c.current_instruction == "pick the red block"


def test_unknown_kwarg_raises() -> None:
    with pytest.raises(TypeError):
        RoboArenaController(policy_client_cls=_FakeWS, nonsense_field=True)


def test_set_instruction(ctrl: RoboArenaController) -> None:
    ctrl.set_instruction("place the mug in the sink")
    assert ctrl.current_instruction == "place the mug in the sink"


def test_set_horizon(ctrl: RoboArenaController) -> None:
    ctrl.set_horizon(16)
    assert ctrl.open_loop_horizon == 16


def test_register_key_success_failure(ctrl: RoboArenaController) -> None:
    ctrl.register_key(ord("y"))
    assert ctrl.get_info()["success"] is True
    ctrl.reset_state()
    ctrl.register_key(ord("n"))
    assert ctrl.get_info()["failure"] is True


def test_register_key_space_toggles_movement(ctrl: RoboArenaController) -> None:
    initial = ctrl.get_info()["movement_enabled"]
    ctrl.register_key(ord(" "))
    assert ctrl.get_info()["movement_enabled"] is (not initial)


def test_forward_when_stopped_returns_safe_action(ctrl: RoboArenaController) -> None:
    ctrl.stop_policy()
    result = ctrl.forward(make_obs())
    assert_action_pair(result, expected_dim=8)
    action, _ = result
    assert np.allclose(action[:-1], 0.0)


def test_forward_queries_then_chunks(ctrl: RoboArenaController) -> None:
    obs = make_obs()
    queries: List[int] = []
    for _ in range(5):
        _, info = ctrl.forward(obs)
        queries.append(info["policy"]["queried_this_step"])
    # Chunk has 4 actions; queries should happen on step 0 and step 4.
    assert queries[0] == 1
    assert queries[1] == 0
    assert queries[2] == 0
    assert queries[3] == 0
    assert queries[4] == 1


def test_forward_records_policy_diagnostics(ctrl: RoboArenaController) -> None:
    _, info = ctrl.forward(make_obs())
    assert "policy" in info
    diag = info["policy"]
    assert diag["chunk_index"] == 0
    assert diag["chunk_query_count"] == 1
    assert diag["queried_this_step"] == 1
    assert diag["infer_failed"] == 0
    assert isinstance(diag["inference_latency_ms"], float)
    assert diag["inference_latency_ms"] >= 0.0


def test_request_payload_respects_server_metadata(ctrl: RoboArenaController) -> None:
    ctrl.set_instruction("do something")
    ctrl.forward(make_obs())

    assert len(_FakeWS.last) == 1
    client = _FakeWS.last[0]
    infer_calls = [c for c in client.calls if c[0] == "infer"]
    assert infer_calls, "controller should have called infer"
    _, payload = infer_calls[0]

    assert payload["prompt"] == "do something"
    assert payload["session_id"] == ctrl.session_id
    assert "observation/joint_position" in payload
    assert "observation/cartesian_position" in payload
    assert "observation/gripper_position" in payload
    assert "observation/exterior_image_1_left" in payload
    assert "observation/wrist_image_left" in payload
    # n_external_cameras=1, so the secondary external must NOT be present.
    assert "observation/exterior_image_2_left" not in payload
    # needs_stereo_camera=False, so no right-eye stereo keys.
    assert "observation/exterior_image_1_right" not in payload
    assert "observation/wrist_image_right" not in payload

    img = payload["observation/exterior_image_1_left"]
    assert isinstance(img, np.ndarray) and img.shape == (224, 224, 3)


def test_n_external_two_sends_both_cameras() -> None:
    class _FakeTwoCam(_FakeWS):
        def get_server_metadata(self):
            md = super().get_server_metadata()
            md["n_external_cameras"] = 2
            return md

    c = RoboArenaController(policy_client_cls=_FakeTwoCam)
    c.forward(make_obs())
    payload = _FakeTwoCam.last[-1].calls[0][1]
    assert "observation/exterior_image_1_left" in payload
    assert "observation/exterior_image_2_left" in payload


def test_reset_state_calls_server_reset(ctrl: RoboArenaController) -> None:
    ctrl.forward(make_obs())  # triggers connect
    assert ctrl._policy_client is not None
    ctrl.reset_state()

    endpoints = [c[0] for c in ctrl._policy_client.calls]
    assert "reset" in endpoints
    reset_call = next(c for c in ctrl._policy_client.calls if c[0] == "reset")
    assert reset_call[1]["session_id"] == ctrl.session_id


def test_infer_failure_returns_safe_action_and_logs_failure() -> None:
    class _FlakyWS(_FakeWS):
        def infer(self, payload):
            self.calls.append(("infer", payload))
            raise RuntimeError("simulated server crash")

    c = RoboArenaController(policy_client_cls=_FlakyWS)
    action, info = c.forward(make_obs())
    assert action.shape == (8,)
    assert np.allclose(action[:-1], 0.0)
    assert info["policy"]["infer_failed"] == 1
    assert info["policy"]["chunk_index"] == -1


def test_picks_up_policy_name_from_server(ctrl: RoboArenaController) -> None:
    ctrl.forward(make_obs())
    assert ctrl.policy_name == "fake-roboarena"


def test_session_id_is_stable_across_steps(ctrl: RoboArenaController) -> None:
    sid_before = ctrl.session_id
    ctrl.forward(make_obs())
    ctrl.forward(make_obs())
    assert ctrl.session_id == sid_before


def test_explicit_session_id_overrides_uuid() -> None:
    c = RoboArenaController(
        config=RoboArenaConfig(session_id="my-session-42"),
        policy_client_cls=_FakeWS,
    )
    assert c.session_id == "my-session-42"
