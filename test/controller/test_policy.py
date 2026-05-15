"""Import-level tests for ``eva.controllers.policy.AAWRAvgPoolPolicy``.

The wrapped policy ("``policy``" in ``Runner.set_controller``) is hard-coded
to load weights from a path passed at construction. We do not own those
weights in this checkpoint, so we only assert that the module imports and
the class advertises the EVA controller interface.
"""
from __future__ import annotations

import inspect

import pytest

from eva.controllers import policy as policy_mod


def test_module_exposes_AAWRAvgPoolPolicy() -> None:
    assert hasattr(policy_mod, "AAWRAvgPoolPolicy")
    cls = policy_mod.AAWRAvgPoolPolicy
    for method in ("get_name", "get_policy_name", "forward", "reset_state",
                   "register_key", "get_info"):
        assert callable(getattr(cls, method, None)), f"missing method: {method}"


def test_AAWRAvgPoolPolicy_init_signature() -> None:
    sig = inspect.signature(policy_mod.AAWRAvgPoolPolicy.__init__)
    assert "policy_path" in sig.parameters
    assert "action_space" in sig.parameters
    assert "gripper_action_space" in sig.parameters


@pytest.mark.skip(reason="requires torch checkpoint + DinoV2 model download")
def test_AAWRAvgPoolPolicy_construction() -> None:
    policy_mod.AAWRAvgPoolPolicy(policy_path="models/dummy.pt")
