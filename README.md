# Eva — Franka DROID inference stack for MolmoAct2

<div align="center">
  <img src="https://github.com/user-attachments/assets/1e36909c-62d8-4fd1-aa3d-333b98d5065e" width="480" />
</div>

Eva is a simple, modular, and extendable Franka infrastructure built on [DROID](https://github.com/droid-dataset/droid). It is the **Franka DROID platform inference/eval stack** for [MolmoAct2](https://github.com/allenai/molmoact2): teleoperation, data collection, and the MolmoAct2 closed-loop eval controller. It is the client that pairs with the MolmoAct2 host server in [`examples/droid/host_server_droid.py`](https://github.com/allenai/molmoact2/tree/main/examples/droid).

Key features:
- Modular design with atomic components, making it configurable and extendable.
- Lightweight and simple interface via the terminal and a camera feed window.
- Clean and organized code, streamlining future development.

> This repo is consumed by `allenai/molmoact2` as the `EVA_DROID` git submodule. To work on it standalone, clone it directly and follow the steps below.

## Installation

1. **DROID base.** The DROID software/hardware setup is the foundation for Eva. Install it following the instructions [here](https://droid-dataset.github.io/droid/).
2. **Environment.** This repo uses [pixi](https://pixi.sh):
   ```bash
   pixi install
   pixi run install-local   # editable installs of openpi-client, oculus_reader, droid
   ```
   `install-local` expects `openpi-main`, `oculus_reader`, and `droid` under `$HOME` (override with `OPENPI_CLIENT_PATH`, `OCULUS_READER_PATH`, `DROID_PATH`). It also needs the [ZED SDK](https://www.stereolabs.com/developers/) + `pyzed`, and [Polymetis](https://facebookresearch.github.io/fairo/polymetis/) on the NUC.

## Configuration

No secrets or machine-specific values are committed. Fill in your own setup in `eva/utils/parameters.py` or via environment variables:

| Variable | Used for | Default |
|---|---|---|
| `HAND_CAMERA_ID`, `VARIED_CAMERA_1_ID`, `VARIED_CAMERA_2_ID` | ZED camera serial numbers (find via `ZED_Explorer`) | placeholder serials |
| `NUC_IP`, `ROBOT_IP` | Franka NUC / robot IPs | `172.16.0.4` / `172.16.0.2` (DROID defaults) |
| `EVA_SUDO_PASSWORD` | sudo password used by the robot launcher | empty |
| `POLICY_SERVER_IP`, `POLICY_SERVER_PORT` | MolmoAct2 LAN host server | `127.0.0.1` / `8101` |
| `MOLMOACT2_NGROK_URL` | MolmoAct2 tunnel URL (the `ngrok` preset) | placeholder |
| `EVA_CLUSTER_DEST` / `EVA_CLUSTER_HOST` | `scp` target for shipping data to a training cluster | placeholder |
| `EVA_HUMAN_DATA` | local human-demo data directory | `~/eva_data/human_data` |
| `ABLY_API_KEY` | optional remote timer ([Ably](https://ably.com)); also exposed as `ably_api_key` in `parameters.py` | unset (timer disabled) |

## Usage

Following the DROID setup, Eva runs on two machines:
- NUC: Handles low-level control of the Franka Emika with a server built on [Polymetis](https://facebookresearch.github.io/fairo/polymetis/).
- Laptop: Handles high-level logic (policy inference, teleoperation, etc) with a runner that executes user scripts.

We recommend the following tmux setup:
```
+-------------------------+-------------------------+
|                         |                         |
|      Server (NUC)       |     Runner (Laptop)     |
+-------------------------+                         |
|    Scripts (Laptop)     |                         |
|                         |                         |
+-------------------------+-------------------------+
```

### Startup

1. On the NUC, run
```bash
cd eva/eva/robot
./launch_server.sh
```
2. On the laptop, run
```bash
conda activate eva
cd eva/scripts
python start_runner.py
```

### Scripts

After the server and runner are started, you can execute scripts found in `eva/scripts/`. Some of the main functions include:
- `collect_trajectory.py`: Collects teleoperated trajectories saved in `eva/data/`.
- `play_trajectory.py`: Replays a selected trajectory.
- `process_trajectory.py`: Processes the compressed trajectory data into a more usable format.
- `calibrate_camera.py`: Calibrates a camera using the Charuco board.
- `check_calibration.py`: Overlays a gripper annotation on the camera feed.
- `take_pictures.py`: Saves camera pictures to `eva/data/images`.
- `reset_robot.py`: Resets the robot pose to default.


### Controller Support

Eva supports the following controllers:
#### Oculus Quest 2 VR
- Classic DROID controller, map actions to the left controller.
- 
#### Keyboard


#### 3Dconnexion SpaceMouse

**Hardware:** 3Dconnexion SpaceMouse Compact (VID `0x256f`, PID `0xc635`). Verify with `lsusb`.

**Dependency:** `hidapi` (included in pixi.toml). The HID device requires exclusive access — only one process can open it at a time.

**Controls:**
| Input | Action |
|---|---|
| Move mouse | Translate end-effector |
| Twist mouse | Rotate end-effector |
| Left button (hold >0.5s) | Toggle gripper |
| Right button | Reset robot |
| `y` key | Mark trajectory as success |
| `n` key | Mark trajectory as failure |
| `space` key | Reset SpaceMouse origin |

**Keyboard macros** (available in both teleop and data collection):
| Key | Macro |
|---|---|
| `w` / `s` | Forward / backward |
| `a` / `d` | Left / right |
| `1` / `2` | Tilt up / down |
| `5` / `6` | Roll CCW / CW |
| `7` / `8` | Rotate left / right |
| `k` | Raise gripper (+Z) |
| `j` | Close gripper |

**Runner-level keys** (handled by `eva/runner.py`, work with any controller):
| Key | Action |
|---|---|
| `l` | Cycle motion step scale (0.5x / 1.0x / 2.0x). Current scale is rendered on the camera-feed window and applied to any controller exposing `set_step_scale`. |
| `i` | Prompt in the terminal for a new task instruction; sent to the controller via `set_instruction` if supported. |
| `o` | Cycle overlay mode. |

**Tuning parameters with the teleop script:**

Use `scripts/spacemouse_teleop.py` to interactively tune SpaceMouse hyperparameters before committing them to `parameters.py`. The script shows a live GUI with wrist + external camera feeds. Notice this script is only used for tuning, not for data collection.

```bash
# Use default config from parameters.py (when SPACEMOUSE_OVERRIDE_CONFIG=True)
python scripts/spacemouse_teleop.py

# All available tuning flags
python scripts/spacemouse_teleop.py \
    --pos_sensitivity 8.0 \
    --rot_sensitivity 8.0 \
    --action_scale 0.1 \
    --deadzone 0.05 \
    --smoothing 0.3 \
    --max_lin_vel 5.0 \
    --max_rot_vel 5.0 \
    --max_gripper_vel 5.0 \
    --external_camera right
```

Additional teleop keys: `g` toggle gripper, `r` reset robot, `i` set instruction overlay, `q`/`ESC` quit.

Config priority: `parameters.py` base (if `SPACEMOUSE_OVERRIDE_CONFIG=True`) → CLI args override on top.

**Collecting data with SpaceMouse + Pi0 mixed controller:**

`scripts/collect_pi0_spacemouse.py` runs a mixed-mode controller that supports runtime switching between a Pi0 policy and SpaceMouse teleoperation.

```bash
# Collect 10 trajectories (default)
python scripts/collect_pi0_spacemouse.py

# Collect N trajectories
python scripts/collect_pi0_spacemouse.py --n 20
```

Controls: `=` switch between Pi0 and SpaceMouse, `space` toggle movement, `y` success, `n` failure.

### Policies

#### MolmoAct2

`scripts/run_molmoact2.py` runs the MolmoAct2 VLA policy via a remote policy server.

```bash
# LAN FastAPI box (default)
python scripts/run_molmoact2.py -n 10

# Named preset (e.g. the ngrok tunnel defined in MOLMOACT2_ENDPOINTS)
python scripts/run_molmoact2.py -n 10 --server ngrok

# Ad-hoc raw URL
python scripts/run_molmoact2.py -n 10 --server https://abc.ngrok-free.app/act
```

**Action interface:** `action_space = joint_position` (q1..q7) + `gripper_action_space = position` (gripper in `[0, 1]`). The server returns absolute joint targets; the controller applies EMA smoothing and clips per-step joint deltas (`max_dq = 0.15` rad) for safety.

**Cameras:** uses the LEFT view of two ZEDs — external = `varied_camera_1_id` (shoulder ZED 2), wrist = `hand_camera_id` (ZED Mini). Images are resized to 320x180 before being sent to the server.

**Server endpoints.** Presets live in the `MOLMOACT2_ENDPOINTS` dict in `eva/utils/parameters.py` — `name -> {"url", "norm_tag"}`. Edit URLs directly there (including a rotated ngrok tunnel); there is no separate config file. `--server` accepts a preset name or a raw `http(s)://` URL.

**Two server protocols.** The LAN FastAPI box uses an implicit default normalization tag, so the client must **not** send a `norm_tag` field (`norm_tag: None`). The newer ngrok-served build **requires** `norm_tag: "franka_droid"`. Each endpoint carries its own `norm_tag`, so a single `--server` flag picks both URL and protocol.


### Development

Code development should be entirely done on the laptop, and to sync the codebase with the NUC, run `./sync_infra.sh`. Remember to restart the server or runner if code changes affect them.

If you are using Eva and plan to make significant changes, **please work in a copy of this directory** (eg, `eva_<yourname>`).

### Data Transfer between Franka Laptop and GPU Cluster

To train model on data collected from the Franka laptop, run the following command:
```bash
./send_data_to_cluster.sh /path/to/data
```


