# Eva (eva_pal_share) — Claude notes

## Repo layout
- `eva/` — Python package: env, robot, controllers, utils.
- `eva/controllers/` — teleop + policy controllers (each exposes `action_space`, `gripper_action_space`, `forward(obs)`, `reset_state`, `register_key`, etc.).
- `eva/runner.py` — `Runner` dispatches by controller name in `set_controller` and runs trajectories.
- `scripts/` — entry points (`start_runner.py`, `collect_trajectory.py`, `run_molmoact2.py`, ...).

## Adding / running a policy

A policy controller plugs into `Runner.set_controller` with a string name and is instantiated there. See `eva/controllers/molmoact2.py` (`MolmoAct2Policy`) and the `elif controller == "molmoact2":` branch in `eva/runner.py`.

## MolmoAct2 policy

- Script: `python scripts/run_molmoact2.py -n <N>` — runs `N` trajectories, prompts for a new instruction each run (Enter keeps the previous one), and waits for `y`/`n` after each rollout.
- Controller name: `"molmoact2"` (in `Runner.set_controller`).
- Action space: `joint_position` (q1..q7) + gripper `position` in `[0, 1]`. Server returns absolute joint targets; controller applies EMA smoothing (`ema_alpha=0.7`) and clips per-step joint delta (`max_dq=0.15` rad).
- Cameras: LEFT view only — external = `params.varied_camera_1_id` (ZED 2 shoulder), wrist = `params.hand_camera_id` (ZED Mini wrist). Images resized to 320x180 before send.
- Open-loop chunk: 15 actions per server query (`open_loop_horizon`).
- Server protocol: HTTP `POST {server_url}` with `json_numpy` payload `{external_cam, wrist_cam, state(8,), instruction, timestamp}`, response `{"actions": (N, 8)}`.
- Server URL: single source of truth is `molmoact2_server_url` in `eva/utils/parameters.py`. The controller default (`MolmoAct2Config.server_url`) and the launcher script both import it.
- First-step debug images are written to `debug/molmoact2_<instruction>_{external,wrist}.jpg`.

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
