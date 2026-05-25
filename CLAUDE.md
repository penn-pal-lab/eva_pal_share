# Eva — Claude notes

Eva is the Franka **DROID platform inference/eval stack for MolmoAct2** (teleop, data collection,
closed-loop eval). It is consumed by [`allenai/molmoact2`](https://github.com/allenai/molmoact2) as
the `EVA_DROID` git submodule and pairs with that repo's `examples/droid/host_server_droid.py`.

**Config is secret-free:** machine-specific values (camera serials, IPs, server URLs, sudo
password, cluster/scp targets, data dirs) live in `eva/utils/parameters.py` as
`os.environ.get(VAR, <placeholder>)`. Fill them in via env vars or edit the placeholders locally —
never commit real values. See the Configuration table in `README.md`.

## Repo layout
- `eva/` — Python package: env, robot, controllers, utils.
- `eva/controllers/` — teleop + policy controllers (each exposes `action_space`, `gripper_action_space`, `forward(obs)`, `reset_state`, `register_key`, etc.).
- `eva/runner.py` — `Runner` dispatches by controller name in `set_controller` and runs trajectories.
- `scripts/` — entry points (`start_runner.py`, `collect_trajectory.py`, `run_molmoact2.py`, ...).

## Adding / running a policy

A policy controller plugs into `Runner.set_controller` with a string name and is instantiated there. See `eva/controllers/molmoact2.py` (`MolmoAct2Policy`) and the `elif controller == "molmoact2":` branch in `eva/runner.py`.

## MolmoAct2 policy

- Script: `python scripts/run_molmoact2.py -n <N> [-s lan|ngrok|<url>]`.
- Controller name: `"molmoact2"` (`Runner.set_controller`). Construct with `endpoint={"url": ..., "norm_tag": ...}`.
- Action space: `joint_position` (q1..q7) + gripper `position` in `[0, 1]`. Server returns absolute joint targets; controller applies EMA smoothing (`ema_alpha=0.7`) and clips per-step joint delta (`max_dq=0.15` rad).
- Cameras: LEFT view only — external = `params.varied_camera_1_id` (ZED 2 shoulder), wrist = `params.hand_camera_id` (ZED Mini wrist). Images resized to 320x180 before send.
- Open-loop chunk: 15 actions per server query (`open_loop_horizon`).

### Endpoints + server protocol

- Endpoints are a plain dict `MOLMOACT2_ENDPOINTS` in `eva/utils/parameters.py` (data only): `name -> {"url", "norm_tag"}`. Edit URLs directly there (including a rotated ngrok tunnel) — there is no separate config file. Built-in: `lan` + `ngrok`.
- Each endpoint pairs a URL with a `norm_tag`. **Two server protocols coexist**:
  - **Legacy LAN**: `norm_tag=None` → field is omitted from the payload entirely. The LAN FastAPI server uses its own implicit default and may reject unknown fields, so this is required.
  - **New (ngrok build)**: requires `norm_tag="franka_droid"`; rejects unknown tags including the old `"droid"`.
- `resolve_molmoact2_endpoint(spec)` in `eva/controllers/molmoact2.py` maps a preset name or raw `http(s)://` URL to an `{"url", "norm_tag"}` dict; passed to `runner.set_controller("molmoact2", endpoint=...)` → `MolmoAct2Policy.__init__`.
- Wire-level payload: `POST {url}` with `json_numpy` body `{external_cam, wrist_cam, state(8,), instruction, timestamp[, norm_tag]}`; response `{"actions": (N, 8)}`.

## Conventions

- `yellow_print` from `eva.utils.misc_utils` is the standard way to log to the runner console.
- `Runner.run_trajectory(mode=...)` modes: `"collect"` (saves to failure/, promoted to success/ on `y`), `"evaluate"` (saves to eval/), `"practice"` (no save).
- Data collection requires all 6 camera streams (`len(self.full_cam_ids) == 6`) — otherwise `run_trajectory` raises.

## Keyboard handling

The cv2 display thread in `Runner.display_camera_feed` is the central keylistener.

- **Runner-owned keys** (intercepted before reaching the controller): `o` overlay mode, `l` cycle step scale, `i` prompt instruction.
- **Controller keys** (forwarded via `controller.register_key`): motion macros `w/s/a/d/1/2/5/6/7/8/j/k`, plus `y` success / `n` failure / `space` reset.
- `l` cycles `Runner.step_sizes` and calls `controller.set_step_scale(scale)` if implemented (e.g. SpaceMouse multiplies macro magnitudes by this scale).
- `i` runs `input()` in a daemon thread (so the cv2 window stays responsive) and routes the result through `Runner.set_controller_instruction` → `controller.set_instruction(text)`.
- The HUD overlay (yellow text in the top-left of the camera grid) renders the current step scale and, if the controller has `current_instruction`, the current instruction.
