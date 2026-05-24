"""Tests for ``eva.controllers.molmoact2.MolmoAct2Policy``.

Covers:
- Runner-facing interface and config setters.
- State lifecycle (reset_state, start/stop, register_key).
- ``_extract_images`` camera-key routing.
- ``forward()`` fallback paths (stopped, missing camera, server error, bad shape).
- ``forward()`` happy path: chunk consumption, re-query at horizon, EMA
  smoothing, per-step max_dq safety clip, gripper clip in info_dict.
- ``_query_server`` shape handling.

Every test passes an explicit ``MolmoAct2Config`` to avoid the controller's
mutable-default-arg trap (a single ``MolmoAct2Config()`` instance is otherwise
shared across all default-constructed policies).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eva.controllers.molmoact2 import MolmoAct2Config, MolmoAct2Policy
from _helpers import (  # type: ignore[import]
    assert_action_pair,
    assert_controller_interface,
    make_obs,
)


def _cfg(**overrides) -> MolmoAct2Config:
    """Fresh config with safe defaults for tests."""
    defaults = dict(
        open_loop_horizon=4,   # small so we can exhaust the chunk fast
        use_ema_smoothing=False,
        ema_alpha=0.7,
        max_dq=10.0,           # large so clip does not interfere unless asked
        request_timeout=0.1,
    )
    defaults.update(overrides)
    return MolmoAct2Config(**defaults)


@pytest.fixture()
def molmo() -> MolmoAct2Policy:
    return MolmoAct2Policy(_cfg())


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
def test_interface(molmo: MolmoAct2Policy) -> None:
    assert_controller_interface(molmo)
    assert molmo.action_space == "joint_position"
    assert molmo.gripper_action_space == "position"
    assert molmo.open_loop_horizon == 4
    assert isinstance(molmo.current_instruction, str)


def test_get_name_and_policy_name_are_model_name(molmo: MolmoAct2Policy) -> None:
    assert molmo.get_name() == "molmoact2"
    assert molmo.get_policy_name() == "molmoact2"


# ---------------------------------------------------------------------------
# Config setters
# ---------------------------------------------------------------------------
def test_set_instruction(molmo: MolmoAct2Policy) -> None:
    molmo.set_instruction("pick the red cup")
    assert molmo.current_instruction == "pick the red cup"


def test_set_horizon_coerces_to_int(molmo: MolmoAct2Policy) -> None:
    molmo.set_horizon(8.0)
    assert molmo.open_loop_horizon == 8
    assert isinstance(molmo.open_loop_horizon, int)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
def test_reset_state_clears_chunk_and_counters(molmo: MolmoAct2Policy) -> None:
    molmo.pred_action_chunk = np.zeros((3, 8), dtype=np.float32)
    molmo.actions_from_chunk_completed = 2
    molmo.policy_query_count = 5
    molmo.inference_steps = [1, 2, 3]
    molmo._q_target_smoothed = np.ones(7, dtype=np.float32)

    molmo.reset_state()

    assert molmo.pred_action_chunk is None
    assert molmo.actions_from_chunk_completed == 0
    assert molmo.policy_query_count == 0
    assert molmo.inference_steps == []
    assert molmo._q_target_smoothed is None
    info = molmo.get_info()
    assert info["success"] is False
    assert info["failure"] is False
    assert info["t_step"] == 0


def test_stop_policy_resets_state_and_close_alias(molmo: MolmoAct2Policy) -> None:
    molmo.pred_action_chunk = np.zeros((3, 8), dtype=np.float32)
    molmo.policy_query_count = 7

    molmo.stop_policy()
    assert molmo._is_running is False
    assert molmo.pred_action_chunk is None
    assert molmo.policy_query_count == 0

    molmo.start_policy()
    assert molmo._is_running is True

    molmo.close()
    assert molmo._is_running is False


def test_register_key_success_failure(molmo: MolmoAct2Policy) -> None:
    molmo.register_key(ord("y"))
    assert molmo.get_info()["success"] is True
    molmo.reset_state()
    molmo.register_key(ord("n"))
    assert molmo.get_info()["failure"] is True


def test_register_key_ignores_unknown(molmo: MolmoAct2Policy) -> None:
    molmo.register_key(ord("z"))
    info = molmo.get_info()
    assert info["success"] is False
    assert info["failure"] is False


# ---------------------------------------------------------------------------
# _extract_images
# ---------------------------------------------------------------------------
def test_extract_images_picks_left_view_for_configured_cams(molmo: MolmoAct2Policy) -> None:
    obs = make_obs()
    ext, wri = molmo._extract_images(obs)
    assert ext is not None and wri is not None
    # BGRA -> RGB drop alpha + reverse channels: shape stays HxWx3.
    assert ext.shape[-1] == 3
    assert wri.shape[-1] == 3


def test_extract_images_returns_none_when_camera_missing(molmo: MolmoAct2Policy) -> None:
    # Build obs with only a random unrelated camera id present.
    obs = make_obs(cam_ids=("99999999",))
    ext, wri = molmo._extract_images(obs)
    assert ext is None
    assert wri is None


def test_extract_images_ignores_right_view() -> None:
    # Make a config where only "right" keys would match; ensure we skip them.
    cfg = _cfg()
    pol = MolmoAct2Policy(cfg)
    obs = make_obs()
    # Wipe out left-view keys, leaving only right ones.
    obs["image"] = {k: v for k, v in obs["image"].items() if "left" not in k}
    ext, wri = pol._extract_images(obs)
    assert ext is None and wri is None


# ---------------------------------------------------------------------------
# forward() fallback paths
# ---------------------------------------------------------------------------
def test_forward_when_stopped_returns_hold(molmo: MolmoAct2Policy) -> None:
    molmo.stop_policy()
    obs = make_obs()
    action, info = molmo.forward(obs)
    assert_action_pair((action, info), expected_dim=8)
    # Hold action = [joint_positions, gripper_position]
    expected = np.concatenate(
        [obs["robot_state"]["joint_positions"], [obs["robot_state"]["gripper_position"]]]
    )
    np.testing.assert_allclose(action, expected)


def test_forward_holds_when_external_camera_missing(molmo: MolmoAct2Policy) -> None:
    # Drop the external camera (varied_camera_1) from the obs.
    import eva.utils.parameters as params
    obs = make_obs(cam_ids=(params.hand_camera_id,))  # only wrist present
    with patch("eva.controllers.molmoact2.requests.post") as post:
        action, _ = molmo.forward(obs)
    post.assert_not_called()
    assert_action_pair((action, _), expected_dim=8)


def test_forward_holds_when_wrist_camera_missing(molmo: MolmoAct2Policy) -> None:
    import eva.utils.parameters as params
    obs = make_obs(cam_ids=(params.varied_camera_1_id,))  # only external present
    with patch("eva.controllers.molmoact2.requests.post") as post:
        action, _ = molmo.forward(obs)
    post.assert_not_called()
    assert_action_pair((action, _), expected_dim=8)


def test_forward_holds_on_server_exception(molmo: MolmoAct2Policy) -> None:
    with patch(
        "eva.controllers.molmoact2.requests.post",
        side_effect=RuntimeError("connection refused"),
    ):
        action, _ = molmo.forward(make_obs())
    assert_action_pair((action, _), expected_dim=8)
    # Hold path -> no chunk was populated.
    assert molmo.pred_action_chunk is None


def test_forward_holds_on_non_200(molmo: MolmoAct2Policy) -> None:
    fake_resp = MagicMock(status_code=500, text="boom")
    with patch("eva.controllers.molmoact2.requests.post", return_value=fake_resp):
        action, _ = molmo.forward(make_obs())
    assert_action_pair((action, _), expected_dim=8)
    assert molmo.pred_action_chunk is None


def test_forward_holds_on_bad_action_shape(molmo: MolmoAct2Policy) -> None:
    # Server returns a chunk of wrong width -> _query_server raises -> hold.
    bad = np.zeros((4, 6), dtype=np.float32).tolist()
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"actions": bad}
    with patch("eva.controllers.molmoact2.requests.post", return_value=fake_resp):
        action, _ = molmo.forward(make_obs())
    assert_action_pair((action, _), expected_dim=8)
    assert molmo.pred_action_chunk is None


# ---------------------------------------------------------------------------
# forward() happy path: chunking, smoothing, clipping
# ---------------------------------------------------------------------------
def _fake_post_factory(chunk: np.ndarray):
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"actions": chunk.tolist()}
    return MagicMock(return_value=fake_resp)


def test_forward_populates_chunk_and_query_count(molmo: MolmoAct2Policy) -> None:
    chunk = np.zeros((molmo.open_loop_horizon, 8), dtype=np.float32)
    with patch("eva.controllers.molmoact2.requests.post", _fake_post_factory(chunk)):
        molmo.forward(make_obs())
    assert molmo.pred_action_chunk is not None
    assert molmo.policy_query_count == 1
    assert molmo.actions_from_chunk_completed == 1


def test_forward_consumes_chunk_then_requeries_at_horizon(molmo: MolmoAct2Policy) -> None:
    # open_loop_horizon=4 from the fixture config.
    chunk = np.zeros((molmo.open_loop_horizon, 8), dtype=np.float32)
    post = _fake_post_factory(chunk)
    with patch("eva.controllers.molmoact2.requests.post", post):
        # First `open_loop_horizon` forwards consume one chunk; the next one re-queries.
        for _ in range(molmo.open_loop_horizon):
            molmo.forward(make_obs())
        assert molmo.policy_query_count == 1
        molmo.forward(make_obs())
        assert molmo.policy_query_count == 2


def test_forward_clips_per_step_joint_delta_to_max_dq() -> None:
    pol = MolmoAct2Policy(_cfg(max_dq=0.05))
    # Server returns absolute joint targets far away from current state.
    big_target = np.tile(
        np.concatenate([np.full(7, 5.0, dtype=np.float32), [0.0]]),
        (pol.open_loop_horizon, 1),
    )
    obs = make_obs(joint_positions=(0.0,) * 7)
    with patch("eva.controllers.molmoact2.requests.post", _fake_post_factory(big_target)):
        action, _ = pol.forward(obs)
    # Each joint should move by at most max_dq from 0.
    np.testing.assert_allclose(action[:7], np.full(7, 0.05), atol=1e-6)


def test_forward_ema_smoothing_blends_first_two_calls() -> None:
    pol = MolmoAct2Policy(_cfg(use_ema_smoothing=True, ema_alpha=0.7, max_dq=10.0))
    # Two queries-worth of targets: first all 0, second all 1.
    # Make horizon=1 so each forward triggers a new query.
    pol.open_loop_horizon = 1
    obs = make_obs(joint_positions=(0.0,) * 7)

    zeros = np.zeros((1, 8), dtype=np.float32)
    ones = np.concatenate([np.ones((1, 7), dtype=np.float32),
                           np.zeros((1, 1), dtype=np.float32)], axis=1)

    post = MagicMock(side_effect=[
        MagicMock(status_code=200, **{"json.return_value": {"actions": zeros.tolist()}}),
        MagicMock(status_code=200, **{"json.return_value": {"actions": ones.tolist()}}),
    ])
    with patch("eva.controllers.molmoact2.requests.post", post):
        # First call: smoothed = raw = 0.
        pol.forward(obs)
        np.testing.assert_allclose(pol._q_target_smoothed, np.zeros(7))
        # Second call: smoothed = 0.7*1 + 0.3*0 = 0.7.
        pol.forward(obs)
        np.testing.assert_allclose(pol._q_target_smoothed, np.full(7, 0.7), atol=1e-6)


def test_forward_clips_gripper_in_info_to_unit_range() -> None:
    pol = MolmoAct2Policy(_cfg())
    # Gripper command beyond [0,1] should be clipped in info_dict.
    chunk = np.tile(
        np.concatenate([np.zeros(7, dtype=np.float32), [5.0]]),
        (pol.open_loop_horizon, 1),
    )
    with patch("eva.controllers.molmoact2.requests.post", _fake_post_factory(chunk)):
        _, info = pol.forward(make_obs())
    assert info["gripper_position"] == 1.0


# ---------------------------------------------------------------------------
# _query_server direct: 1-D response auto-reshape
# ---------------------------------------------------------------------------
def test_query_server_reshapes_1d_response(molmo: MolmoAct2Policy) -> None:
    # 16 elements -> reshape to (2, 8)
    flat = np.zeros(16, dtype=np.float32).tolist()
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"actions": flat}
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    with patch("eva.controllers.molmoact2.requests.post", return_value=fake_resp):
        actions = molmo._query_server(img, img, state)
    assert actions.shape == (2, 8)


def test_query_server_raises_on_missing_actions_key(molmo: MolmoAct2Policy) -> None:
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"oops": []}
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    with patch("eva.controllers.molmoact2.requests.post", return_value=fake_resp):
        with pytest.raises(RuntimeError, match="Bad response keys"):
            molmo._query_server(img, img, state)


# ---------------------------------------------------------------------------
# server_url override + endpoint resolution
# ---------------------------------------------------------------------------
def test_endpoint_kwarg_overrides_url_and_norm_tag() -> None:
    pol = MolmoAct2Policy(
        _cfg(server_url="http://default.example/act", norm_tag=None),
        endpoint={"url": "https://override.example/act", "norm_tag": "franka_droid"},
    )
    assert pol.cfg.server_url == "https://override.example/act"
    assert pol.cfg.norm_tag == "franka_droid"


def test_default_norm_tag_is_none_for_legacy_lan_compat() -> None:
    """LAN server rejects unknown fields; default must not include norm_tag."""
    pol = MolmoAct2Policy(_cfg())
    assert pol.cfg.norm_tag is None


def test_payload_omits_norm_tag_when_none() -> None:
    pol = MolmoAct2Policy(_cfg(norm_tag=None))
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"actions": np.zeros((1, 8)).tolist()}
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    with patch("eva.controllers.molmoact2.requests.post", return_value=fake_resp), \
         patch("eva.controllers.molmoact2.json_numpy.dumps", side_effect=lambda d: d) as dumps:
        pol._query_server(img, img, state)
    assert "norm_tag" not in dumps.call_args[0][0]


def test_payload_includes_norm_tag_when_set() -> None:
    pol = MolmoAct2Policy(_cfg(norm_tag="franka_droid"))
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"actions": np.zeros((1, 8)).tolist()}
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    with patch("eva.controllers.molmoact2.requests.post", return_value=fake_resp), \
         patch("eva.controllers.molmoact2.json_numpy.dumps", side_effect=lambda d: d) as dumps:
        pol._query_server(img, img, state)
    assert dumps.call_args[0][0]["norm_tag"] == "franka_droid"


def test_default_construction_is_isolated_per_instance() -> None:
    """Regression: mutable default arg used to share one cfg across instances."""
    a = MolmoAct2Policy()
    b = MolmoAct2Policy()
    assert a.cfg is not b.cfg, "each policy must get its own MolmoAct2Config"


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------
def test_resolve_endpoint_returns_dict() -> None:
    from eva.controllers.molmoact2 import resolve_molmoact2_endpoint
    ep = resolve_molmoact2_endpoint("lan")
    assert set(ep) == {"url", "norm_tag"}


def test_lan_preset_has_no_norm_tag() -> None:
    """LAN compat: built-in 'lan' must default to norm_tag=None."""
    from eva.utils.parameters import MOLMOACT2_ENDPOINTS
    assert MOLMOACT2_ENDPOINTS["lan"]["norm_tag"] is None


def test_resolve_endpoint_raw_url_has_no_norm_tag() -> None:
    from eva.controllers.molmoact2 import resolve_molmoact2_endpoint
    ep = resolve_molmoact2_endpoint("https://abc.ngrok-free.app/act")
    assert ep["url"] == "https://abc.ngrok-free.app/act"
    assert ep["norm_tag"] is None


def test_resolve_endpoint_rejects_unknown() -> None:
    from eva.controllers.molmoact2 import resolve_molmoact2_endpoint
    with pytest.raises(ValueError, match="Unknown endpoint"):
        resolve_molmoact2_endpoint("not-a-preset-and-not-a-url")
