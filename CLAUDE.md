# EVA Codebase Notes

## Architecture

- `eva/runner.py` — main orchestrator; owns the env, controller, camera display, and trajectory lifecycle
- `eva/manager.py` — multiprocessing wrapper; `load_runner(manager=False)` for standalone mode
- `eva/env.py` — FrankaEnv gym environment; wraps robot + cameras
- `eva/utils/trajectory_utils.py` — HDF5 writer/reader, `run_trajectory()` control loop
- `eva/controllers/pi0_policy.py` — Pi0 VLA policy; connects to remote server lazily on first `forward()`
- `scripts/collect_trajectory.py` — human/policy data collection
- `scripts/run_pi0.py` — quick single-policy evaluation
- `scripts/run_eval.py` — full evaluation session manager (see below)

## Camera Setup

3 ZED stereo cameras → 6 raw feeds (`{serial}_left`, `{serial}_right`).  
Policy only uses 3 feeds: `hand_camera_left`, `varied_camera_1_left`, `varied_camera_2_left`.  
Camera IDs defined in `eva/utils/parameters.py`.

## Trajectory Modes

| mode | saves to | 6-cam check | dir split on success |
|------|----------|-------------|----------------------|
| `"collect"` | `data/success/` or `data/failure/` | yes | yes |
| `"evaluate"` | `data/eval/` | no | no |
| `"practice"` | nothing | no | no |

Use `mode="evaluate"` for all policy evaluation runs. `mode="collect"` is for human demos only.

## Per-Trajectory Artifacts

Every completed trajectory (collect or evaluate) produces:
```
data/{success|failure|eval}/YYYY-MM-DD/{timestamp}/
  trajectory.h5       — robot state, actions, images (all 6 feeds)
  metadata.json       — auto-written by runner; see schema below
  instruction.txt     — task instruction string
  policy.md           — policy name
  calibration.json    — camera extrinsics
  recordings/         — MP4 + JPEG frames (if post_process=True)
```

### metadata.json schema
```json
{
  "outcome":       "success" | "failure",
  "num_steps":     int,
  "duration_sec":  float,
  "policy":        "pi05-fm",
  "robot":         "fr3-XXXXX",
  "timestamp":     "YYYY-MM-DD_HH-MM-SS",
  "data_source":   "human_teleop" | "policy_rollout" | "replay" | "unknown",
  "instruction":   "pick up the pineapple",
  "quality_grade": "A" | "B" | "C" | "D",   // collect only, operator-supplied
  "notes":         "arm slipped at step 40", // optional
  "scene_setup":   "pineapple on shelf left side", // optional
  "session_id":    "YYYY-MM-DD_HH-MM-SS"    // eval only, set by run_eval.py
}
```

Quality grades: A = excellent, B = good, C = acceptable, D = discard.

## Evaluation Sessions

`scripts/run_eval.py` manages a full evaluation session:
- Prompts for task, evaluator, scene description at start
- Allows per-episode instruction change
- Records outcome, steps, duration, notes per episode
- Saves `data/eval_sessions/eval_session_{timestamp}.json` incrementally
- Writes `session_id` back into each episode's `metadata.json`

```bash
python scripts/run_eval.py -n 10 --instruction "pick up the pineapple"
python scripts/run_eval.py -n 10 --host 10.102.212.31 --external right
```

## run_pi0.py CLI

```bash
python scripts/run_pi0.py -n 5
python scripts/run_pi0.py -n 5 --instruction "pick up the pineapple" --horizon 10
python scripts/run_pi0.py --host 10.102.212.31 --port 8000 --external left
```

## Runner API Notes

- `runner.run_trajectory(mode)` now returns `controller_info`
- `runner.last_traj_dir` — path to the most recently completed trajectory directory
- `runner.update_traj_metadata(dict)` — merges fields into `last_traj_dir/metadata.json`
- `runner.controller` fields can be updated directly after `load_runner()` since Pi0Policy connects lazily

---

## Needs Testing

The following changes were implemented 2026-04-24 and have **not yet been run on hardware**:

### Item 1–2: mode="evaluate" + 6-cam check moved
- `run_pi0.py` now uses `mode="evaluate"` instead of `mode="collect"`
- The 6-camera assertion in `runner.run_trajectory()` is now inside the `collect` branch only
- **Test**: run `run_pi0.py` with fewer than 6 cameras and confirm no crash; confirm eval trajectories land in `data/eval/` not `data/success/failure/`

### Item 3: Camera display trimmed to 3 feeds
- `display_camera_feed()` now filters to `_left` feeds only before rendering
- **Test**: confirm the CV window shows 3 images side by side, not 6; confirm `camera_id=` override still works

### Item 4: run_pi0.py CLI args
- `--host`, `--port`, `--instruction`, `--horizon`, `--external` now accepted
- Fields are set directly on `runner.controller` after `load_runner()`
- **Test**: run with `--instruction "new task"` and confirm `Pi0Policy.current_instruction` reflects the new value before first inference; confirm `--host` override reaches `_ensure_connected()`

### Item 5: metadata.json auto-written by runner
- `_write_traj_metadata()` runs at end of every collect/evaluate trajectory
- `run_trajectory()` now returns `controller_info` (previously returned nothing in collect/evaluate)
- **Test**: confirm `metadata.json` exists after a trajectory; confirm fields are correct; confirm existing callers of `run_trajectory()` that discarded the return value still work

### Item 6: Quality grade prompt in collect_trajectory.py
- After each trajectory, operator is prompted for grade (A/B/C/D) + optional notes/scene
- `update_traj_metadata()` reads and re-writes `metadata.json`
- **Test**: run `collect_trajectory.py`, enter a grade, and inspect the resulting `metadata.json`; test pressing Enter to skip (should write no grade field)

### Item 7–8: run_eval.py session manager
- New script; creates `data/eval_sessions/eval_session_{timestamp}.json`
- Session JSON is written after every episode (incremental — safe if interrupted)
- `session_id` is written back into each episode's `metadata.json`
- **Test**: run a 2-episode session, inspect session JSON and both `metadata.json` files for correct `session_id` linkage; test Ctrl-C mid-session to confirm partial JSON is valid
