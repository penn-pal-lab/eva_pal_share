"""MolmoAct2 policy controller for EVA.

Wraps the standalone bridge in run_molmoact2.py as an EVA controller, so
MolmoAct2 plugs into Runner / FrankaEnv like any other policy.

Action interface:
  - action_space         = "joint_position"  (8 DoF: q1..q7 + gripper)
  - gripper_action_space = "position"        (gripper in [0, 1])

Server protocol (HTTP + json_numpy):
  POST {MOLMOACT2_URL}/act with {external_cam, wrist_cam, state(8,), instruction, timestamp}
  Returns {"actions": (N, 8)} of ABSOLUTE joint positions [q1..q7, gripper].
"""
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import requests
import json_numpy

from eva.utils.misc_utils import create_info_dict
import eva.utils.parameters as params

json_numpy.patch()


@dataclass
class MolmoAct2Config:
    server_url: str = params.molmoact2_server_url
    instruction: str = "pick up the blue block."
    model_name: str = "molmoact2"

    # Cameras (sourced from eva.parameters)
    external_camera_id: str = params.varied_camera_1_id  # ZED 2 shoulder
    wrist_camera_id: str = params.hand_camera_id          # ZED Mini wrist

    # Image size MolmoAct2 expects
    input_width: int = 320
    input_height: int = 180

    # Open-loop chunking
    open_loop_horizon: int = 15

    # Safety / smoothing for ABSOLUTE joint position output
    max_dq: float = 0.15        # rad clip per step
    use_ema_smoothing: bool = True
    ema_alpha: float = 0.7      # higher = more responsive, less lag

    # HTTP
    request_timeout: float = 10.0


class MolmoAct2Policy:
    def __init__(self, config: MolmoAct2Config = MolmoAct2Config()):
        print(f"MolmoAct2 init -> {config.server_url}")
        self.cfg = config

        self.action_space = "joint_position"
        self.gripper_action_space = "position"

        self.current_instruction = config.instruction
        self.model_name = config.model_name
        self.open_loop_horizon = config.open_loop_horizon

        self._is_running = True
        self.reset_state()

    # -------- EVA controller interface --------
    def get_name(self):
        return self.model_name

    def get_policy_name(self):
        return self.model_name

    def reset_state(self):
        self.pred_action_chunk = None
        self.actions_from_chunk_completed = 0
        self.policy_query_count = 0
        self.inference_steps = []
        self._q_target_smoothed = None
        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
            "switch_label": 0,
            "t_step": 0,
            "policy_inference": False,
        }

    def set_state(self, key):
        self._state["success"] = key == "y"
        self._state["failure"] = key == "n"

    def get_info(self):
        return self._state.copy()

    def register_key(self, key):
        if key == ord("y"):
            self._state["success"] = True
            print("Success!")
        elif key == ord("n"):
            self._state["failure"] = True
            print("Failure!")

    def set_instruction(self, instruction):
        self.current_instruction = instruction
        print(f"[MolmoAct2] Set instruction: {self.current_instruction}")

    def set_horizon(self, horizon):
        self.open_loop_horizon = int(horizon)
        print(f"[MolmoAct2] Set open loop horizon: {self.open_loop_horizon}")

    def start_policy(self):
        self._is_running = True

    def stop_policy(self):
        self._is_running = False
        self.reset_state()

    def close(self):
        self.stop_policy()

    # -------- Internals --------
    def _extract_images(self, obs_dict):
        """Pull external + wrist images out of observation (matches Pi0Policy style).

        ZED images come in as BGRA. We drop alpha and convert to RGB via [..., ::-1].
        """
        image_observations = obs_dict["image"]
        ext_image, wri_image = None, None

        for key, img in image_observations.items():
            # Only use the LEFT view of each stereo pair (model is monocular)
            if "left" not in key:
                continue
            if self.cfg.external_camera_id in key:
                ext_image = img
            elif self.cfg.wrist_camera_id in key:
                wri_image = img

        def to_rgb(img):
            if img is None:
                return None
            img = img[..., :3]      # BGRA -> BGR
            return img[..., ::-1]   # BGR  -> RGB

        return to_rgb(ext_image), to_rgb(wri_image)

    def _query_server(self, ext_rgb, wri_rgb, state_8d):
        ext_resized = cv2.resize(ext_rgb, (self.cfg.input_width, self.cfg.input_height))
        wri_resized = cv2.resize(wri_rgb, (self.cfg.input_width, self.cfg.input_height))

        payload = {
            "external_cam": ext_resized,
            "wrist_cam": wri_resized,
            "timestamp": time.time(),
            "instruction": self.current_instruction,
            "state": state_8d,
        }
        serialized = json_numpy.dumps(payload)
        resp = requests.post(
            self.cfg.server_url,
            headers={"Content-Type": "application/json"},
            data=serialized,
            timeout=self.cfg.request_timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"MolmoAct2 server {resp.status_code}: {resp.text[:200]}")
        out = resp.json()
        if "actions" not in out:
            raise RuntimeError(f"Bad response keys: {list(out.keys())}")

        actions = np.array(out["actions"])
        if actions.ndim == 1:
            actions = actions.reshape(-1, 8)
        if actions.shape[1] != 8:
            raise RuntimeError(f"Unexpected action shape {actions.shape}, expected (N, 8)")
        return actions

    def _hold_action(self, joint_position, gripper_position):
        """Safe action that holds current pose."""
        return np.concatenate([joint_position, [float(gripper_position)]])

    # -------- Main step --------
    def forward(self, observation):
        joint_position = np.array(observation["robot_state"]["joint_positions"])
        gripper_position = float(observation["robot_state"]["gripper_position"])
        cartesian_position = np.array(observation["robot_state"]["cartesian_position"])

        info_dict = create_info_dict(observation, self._state, gripper_action=gripper_position)
        info_dict["cartesian_position"] = cartesian_position

        if not self._is_running:
            return self._hold_action(joint_position, gripper_position), info_dict

        self._state["t_step"] += 1
        self._state["policy_inference"] = False

        # Refill chunk when empty / exhausted
        if (self.pred_action_chunk is None
                or self.actions_from_chunk_completed >= self.open_loop_horizon
                or self.actions_from_chunk_completed >= len(self.pred_action_chunk)):

            ext_rgb, wri_rgb = self._extract_images(observation)
            if ext_rgb is None:
                print(f"[MolmoAct2] External camera (SN {self.cfg.external_camera_id}) image missing")
                return self._hold_action(joint_position, gripper_position), info_dict
            if wri_rgb is None:
                print(f"[MolmoAct2] Wrist camera (SN {self.cfg.wrist_camera_id}) image missing")
                return self._hold_action(joint_position, gripper_position), info_dict

            # Save first-step debug images (once per session)
            if self.policy_query_count == 0:
                os.makedirs("debug", exist_ok=True)
                tag = self.current_instruction.replace(" ", "_").replace("/", "_")
                cv2.imwrite(f"debug/molmoact2_{tag}_external.jpg", ext_rgb[..., ::-1])
                cv2.imwrite(f"debug/molmoact2_{tag}_wrist.jpg", wri_rgb[..., ::-1])

            state_8d = np.concatenate([joint_position, [gripper_position]])
            t0 = time.time()
            try:
                self.policy_query_count += 1
                print(f"[MolmoAct2] query #{self.policy_query_count}: '{self.current_instruction}'")
                actions = self._query_server(ext_rgb, wri_rgb, state_8d)
            except Exception as e:
                print(f"[MolmoAct2] inference error: {e}")
                return self._hold_action(joint_position, gripper_position), info_dict

            horizon = min(self.open_loop_horizon, len(actions))
            self.pred_action_chunk = actions[:horizon]
            self.actions_from_chunk_completed = 0
            self._state["policy_inference"] = True
            self.inference_steps.append(self._state["t_step"] - 1)
            print(f"[MolmoAct2] chunk shape={actions.shape}, using first {horizon} "
                  f"(took {time.time() - t0:.3f}s)")

        # Pop next absolute joint target + gripper from chunk
        action_raw = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        q_target_raw = action_raw[:7]   # absolute joint targets from MolmoAct2
        gripper_cmd = float(action_raw[7])

        # EMA smoothing on absolute targets
        if self.cfg.use_ema_smoothing:
            if self._q_target_smoothed is None:
                self._q_target_smoothed = q_target_raw.copy()
            else:
                a = self.cfg.ema_alpha
                self._q_target_smoothed = a * q_target_raw + (1 - a) * self._q_target_smoothed
            q_target = self._q_target_smoothed
        else:
            q_target = q_target_raw

        # Clip per-step joint delta for safety
        dq = q_target - joint_position
        dq_clipped = np.clip(dq, -self.cfg.max_dq, self.cfg.max_dq)
        q_safe = joint_position + dq_clipped

        action = np.concatenate([q_safe, [gripper_cmd]])

        info_dict["joint_position"] = q_safe
        info_dict["gripper_position"] = float(np.clip(gripper_cmd, 0.0, 1.0))
        return action, info_dict
