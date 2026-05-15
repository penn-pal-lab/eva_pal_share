"""Import-level tests for ``eva.controllers.aawr.AAWRPolicy``.

Full instantiation requires:
- a torch checkpoint at ``models/AWR/.../*.pt``,
- a DINOv2 model download from ``facebook/dinov2-small`` (and its processor),
- a PCA pickle at ``models/dinov2/pca_model_pca_16.pkl``.

None of these are available in a clean checkout, so the smoke test only
checks that the module imports and the class advertises the expected
interface attributes via its source.
"""
from __future__ import annotations

import inspect

import pytest

from eva.controllers import aawr as aawr_mod


def test_module_exposes_AAWRPolicy() -> None:
    assert hasattr(aawr_mod, "AAWRPolicy")
    cls = aawr_mod.AAWRPolicy
    # Required EVA controller methods on the class itself.
    for method in ("get_name", "get_policy_name", "forward", "reset_state",
                   "register_key", "get_info"):
        assert callable(getattr(cls, method, None)), f"missing method: {method}"


def test_AAWRPolicy_init_signature() -> None:
    sig = inspect.signature(aawr_mod.AAWRPolicy.__init__)
    assert "policy_path" in sig.parameters
    assert "pca_path" in sig.parameters


@pytest.mark.skip(reason="requires torch checkpoint, DINOv2 weights, and PCA pickle")
def test_AAWRPolicy_construction() -> None:
    aawr_mod.AAWRPolicy(policy_path="models/AWR/vert/1.pt")
