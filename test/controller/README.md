# Controller smoke tests

Per-controller smoke tests for everything in `eva/controllers/`. They cover the
contract that `eva.runner.Runner` expects:

- `action_space` and `gripper_action_space` attributes
- `get_name()`, `reset_state()`, `register_key(key)`, `get_info()`, `forward(obs)`
- `forward()` returns `(action: np.ndarray, info: dict)`

All hardware (HID, ZED, Quest), policy servers (PI0 websocket, MolmoAct2 HTTP)
and heavy ML deps (DINOX, DINOv2) are shimmed in `conftest.py` so the suite
runs on a laptop with no robot connected.

## Running

```bash
pip install pytest numpy opencv-python requests scipy h5py torch torchvision
pytest test/controller -v
```

Or a single file:

```bash
pytest test/controller/test_keyboard.py -v
```

## What each file covers

| File | Controller | Forward exercised |
| --- | --- | --- |
| `test_keyboard.py` | `Keyboard` | yes (7d cart. vel) |
| `test_spacemouse.py` | `SpaceMouse` | yes (7d cart. vel, HID stubbed) |
| `test_gello.py` | `Gello` | no (needs live ZMQ stream) |
| `test_occulus.py` | `Occulus` | no (needs live VR poses) |
| `test_replayer.py` | `Replayer` | yes (synthetic npy trajectory) |
| `test_molmoact2.py` | `MolmoAct2Policy` | yes (stopped + mocked HTTP) |
| `test_pi0_policy.py` | `Pi0Policy` | yes (stopped path) |
| `test_keyboard_pi0.py` | `KeyboardPi0` | yes (both branches) |
| `test_pi0_spacemouse.py` | `SpaceMousePi0` | yes (both branches) |
| `test_human_pi0.py` | `DemoDiffusionPolicy` | yes (stopped path; traj patched) |
| `test_aawr_pi0.py` | `AAWRPi0Controller` | no (AAWR sub-controller mocked) |
| `test_replay_pi0.py` | `ReplayPi0Controller` | no (DINOX stubbed, traj patched) |
| `test_aawr.py` | `AAWRPolicy` | n/a (interface check only — needs ML weights) |
| `test_policy.py` | `AAWRAvgPoolPolicy` | n/a (interface check only — needs ML weights) |

## Adding a new controller test

1. Drop a `test_<name>.py` next to the others.
2. Use `from _helpers import make_obs, assert_controller_interface, assert_action_pair`.
3. If the controller needs new hardware/network deps at import time, extend
   the `sys.modules` shims in `conftest.py`.
