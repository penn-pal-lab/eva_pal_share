"""RoboArena-style controller for EVA.

Wire protocol matches https://github.com/pranavatreya/roboarena_evaluator
(server-driven metadata, msgpack-numpy, websocket with `endpoint` dispatch,
open-loop action chunks). Local storage uses EVA's existing trajectory
pipeline — videos via `MultiCameraWrapper`, HDF5 via `TrajectoryWriter`.

Per-step policy diagnostics (inference latency, chunk index, query count,
infer-failed flag) are returned in the action info dict so they land in
`trajectory.h5` alongside robot state. This lets you align behavior with
policy inference data when analyzing a session.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np

import eva.utils.parameters as params
from eva.controllers._roboarena import image_tools
from eva.controllers._roboarena.base_policy import BasePolicy
from eva.utils.misc_utils import create_info_dict, yellow_print


_VALID_ACTION_SPACES = {
    "joint_position",
    "joint_velocity",
    "cartesian_position",
    "cartesian_velocity",
}
_VALID_EXTERNAL_CAMS = {"left", "right"}
_VALID_GRIPPER_SPACES = {"position", "velocity"}


@dataclass
class RoboArenaConfig:
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000
    instruction: str = ""
    action_space: str = "joint_velocity"
    gripper_action_space: str = "position"
    external_camera: str = "left"
    left_camera_id: str = field(default_factory=lambda: params.varied_camera_1_id)
    right_camera_id: str = field(default_factory=lambda: params.varied_camera_2_id)
    wrist_camera_id: str = field(default_factory=lambda: params.hand_camera_id)
    session_id: Optional[str] = None
    fallback_open_loop_horizon: int = 8


class RoboArenaController:
    def __init__(
        self,
        config: Optional[RoboArenaConfig] = None,
        policy_client_cls: Optional[Type[BasePolicy]] = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            config = RoboArenaConfig()
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise TypeError(f"Unknown RoboArenaController kwarg: {key!r}")
            setattr(config, key, value)

        if policy_client_cls is None:
            # Lazy import: keeps msgpack/websockets out of test envs that inject a fake.
            from eva.controllers._roboarena.websocket_client_policy import (
                WebsocketClientPolicy,
            )

            policy_client_cls = WebsocketClientPolicy

        assert config.action_space in _VALID_ACTION_SPACES, (
            f"action_space must be one of {_VALID_ACTION_SPACES}, got {config.action_space!r}"
        )
        assert config.external_camera in _VALID_EXTERNAL_CAMS, (
            f"external_camera must be one of {_VALID_EXTERNAL_CAMS}, got {config.external_camera!r}"
        )
        assert config.gripper_action_space in _VALID_GRIPPER_SPACES, (
            f"gripper_action_space must be one of {_VALID_GRIPPER_SPACES}, got {config.gripper_action_space!r}"
        )

        self.config = config
        self.action_space = config.action_space
        self.gripper_action_space = config.gripper_action_space
        self.current_instruction = config.instruction
        self.open_loop_horizon = config.fallback_open_loop_horizon

        self._policy_client_cls = policy_client_cls
        self._policy_client: Optional[BasePolicy] = None
        self._server_metadata: Dict[str, Any] = {}
        self._is_running = True

        self.session_id = config.session_id or f"eva-{uuid.uuid4().hex[:8]}"
        self.policy_name = "roboarena-policy"

        self._pred_action_chunk: Optional[np.ndarray] = None
        self._actions_from_chunk_completed = 0
        self._policy_query_count = 0

        self.reset_state()

    # ------------------------------------------------------------------ #
    # EVA controller contract                                            #
    # ------------------------------------------------------------------ #

    def get_name(self) -> str:
        return "roboarena"

    def get_policy_name(self) -> str:
        return json.dumps(
            {
                "controller": "roboarena",
                "policy_name": self.policy_name,
                "server": f"{self.config.remote_host}:{self.config.remote_port}",
                "session_id": self.session_id,
                "server_metadata": _jsonable(self._server_metadata),
                "external_camera": self.config.external_camera,
                "instruction": self.current_instruction,
                "action_space": self.action_space,
                "gripper_action_space": self.gripper_action_space,
            },
            indent=2,
            default=str,
        )

    def reset_state(self) -> None:
        self._pred_action_chunk = None
        self._actions_from_chunk_completed = 0
        self._policy_query_count = 0
        self._state = {
            "success": False,
            "failure": False,
            "movement_enabled": True,
            "controller_on": True,
            "t_step": 0,
        }
        if self._policy_client is not None:
            try:
                self._policy_client.reset({"session_id": self.session_id})
            except Exception as exc:
                yellow_print(f"RoboArena reset() failed: {exc}")

    def register_key(self, key: int) -> None:
        if key == ord("y"):
            self._state["success"] = True
            yellow_print("RoboArena: success")
        elif key == ord("n"):
            self._state["failure"] = True
            yellow_print("RoboArena: failure")
        elif key == ord(" "):
            self._state["movement_enabled"] = not self._state["movement_enabled"]
            yellow_print(f"RoboArena movement_enabled={self._state['movement_enabled']}")
        elif key == ord("r"):
            self.reset_state()
            yellow_print("RoboArena state reset")

    def get_info(self) -> Dict[str, Any]:
        return dict(self._state)

    def set_instruction(self, instruction: str) -> None:
        self.current_instruction = instruction
        yellow_print(f"RoboArena instruction set: {instruction!r}")

    def set_horizon(self, horizon: int) -> None:
        self.open_loop_horizon = int(horizon)
        self.config.fallback_open_loop_horizon = int(horizon)

    def stop_policy(self) -> None:
        self._is_running = False

    def start_policy(self) -> None:
        self._is_running = True
        self._ensure_connected()

    def close(self) -> None:
        self._is_running = False
        self._policy_client = None

    def forward(self, observation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        joint_pos = np.asarray(observation["robot_state"]["joint_positions"], dtype=np.float32)
        cart_pos = np.asarray(observation["robot_state"]["cartesian_position"], dtype=np.float32)
        gripper_pos = float(observation["robot_state"]["gripper_position"])

        info = create_info_dict(observation, self._state)

        if not self._is_running:
            return self._safe_action(joint_pos, cart_pos, gripper_pos), info

        self._ensure_connected()
        self._state["t_step"] += 1

        need_new_chunk = (
            self._pred_action_chunk is None
            or self._actions_from_chunk_completed >= len(self._pred_action_chunk)
        )
        latency_ms = 0.0
        queried_this_step = 0
        infer_failed = 0

        if need_new_chunk:
            self._actions_from_chunk_completed = 0
            request = self._build_request(joint_pos, cart_pos, gripper_pos, observation)
            t0 = time.time()
            try:
                result = self._policy_client.infer(request)
                latency_ms = (time.time() - t0) * 1000.0
                queried_this_step = 1
                chunk = np.asarray(result["actions"])
                if chunk.ndim == 1:
                    chunk = chunk[None, ...]
                self._pred_action_chunk = chunk
                self._policy_query_count += 1
                self.open_loop_horizon = int(chunk.shape[0])
            except Exception as exc:
                latency_ms = (time.time() - t0) * 1000.0
                infer_failed = 1
                yellow_print(f"RoboArena infer() failed: {exc}; emitting safe action")
                info["policy"] = {
                    "inference_latency_ms": float(latency_ms),
                    "chunk_index": -1,
                    "chunk_query_count": int(self._policy_query_count),
                    "queried_this_step": 1,
                    "infer_failed": 1,
                }
                return self._safe_action(joint_pos, cart_pos, gripper_pos), info

        idx = self._actions_from_chunk_completed
        action = np.asarray(self._pred_action_chunk[idx], dtype=np.float32).copy()
        self._actions_from_chunk_completed += 1

        if action.size >= 1:
            action[-1] = 1.0 if action[-1] > 0.5 else 0.0
        if "velocity" in self.action_space and action.size > 1:
            action[:-1] = np.clip(action[:-1], -1.0, 1.0)

        info["policy"] = {
            "inference_latency_ms": float(latency_ms),
            "chunk_index": int(idx),
            "chunk_query_count": int(self._policy_query_count),
            "queried_this_step": int(queried_this_step),
            "infer_failed": int(infer_failed),
        }
        info["cartesian_position"] = cart_pos
        info["gripper_position"] = float(action[-1])
        if self.action_space == "joint_velocity":
            info["joint_velocity"] = action[:-1]
        elif self.action_space == "joint_position":
            info["joint_position"] = action[:-1]
        elif self.action_space == "cartesian_velocity":
            info["cartesian_velocity"] = action[:-1]
        elif self.action_space == "cartesian_position":
            info["cartesian_position"] = action[:-1]

        return action, info

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _ensure_connected(self) -> None:
        if self._policy_client is not None:
            return
        yellow_print(
            f"RoboArena connecting to {self.config.remote_host}:{self.config.remote_port} "
            f"(session_id={self.session_id})"
        )
        self._policy_client = self._policy_client_cls(
            self.config.remote_host, self.config.remote_port
        )
        try:
            self._server_metadata = dict(self._policy_client.get_server_metadata() or {})
        except Exception as exc:
            yellow_print(f"RoboArena get_server_metadata() failed: {exc}")
            self._server_metadata = {}

        if "policy_name" in self._server_metadata:
            self.policy_name = str(self._server_metadata["policy_name"])

        server_action_space = self._server_metadata.get("action_space")
        if server_action_space and server_action_space != self.action_space:
            # Mismatch is a deployment error: FrankaEnv was already configured with
            # self.action_space; sending the server's actions would yield wrong dims.
            yellow_print(
                f"WARNING: RoboArena server action_space={server_action_space!r} "
                f"differs from controller action_space={self.action_space!r}. "
                f"Reconfigure the controller before running a trajectory."
            )

    def _build_request(
        self,
        joint_pos: np.ndarray,
        cart_pos: np.ndarray,
        gripper_pos: float,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        meta = self._server_metadata
        n_external = int(meta.get("n_external_cameras", 1))
        needs_wrist = bool(meta.get("needs_wrist_camera", True))
        needs_stereo = bool(meta.get("needs_stereo_camera", False))
        needs_session_id = bool(meta.get("needs_session_id", False))
        img_res = meta.get("image_resolution")
        if img_res is not None:
            img_res = tuple(img_res)

        images = self._extract_images(obs, needs_stereo=needs_stereo)

        def _prep(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if img is None:
                return None
            if img_res is None:
                return image_tools.convert_to_uint8(img)
            h, w = img_res
            return image_tools.convert_to_uint8(image_tools.resize(img, h, w))

        request: Dict[str, Any] = {
            "observation/joint_position": joint_pos,
            "observation/cartesian_position": cart_pos,
            "observation/gripper_position": np.array([gripper_pos], dtype=np.float32),
            "prompt": self.current_instruction,
        }

        if n_external == 1:
            primary_key = "left_image" if self.config.external_camera == "left" else "right_image"
            request["observation/exterior_image_1_left"] = _prep(images[primary_key])
            if needs_stereo:
                request["observation/exterior_image_1_right"] = _prep(images[primary_key + "_stereo"])
        elif n_external == 2:
            request["observation/exterior_image_1_left"] = _prep(images["left_image"])
            request["observation/exterior_image_2_left"] = _prep(images["right_image"])
            if needs_stereo:
                request["observation/exterior_image_1_right"] = _prep(images["left_image_stereo"])
                request["observation/exterior_image_2_right"] = _prep(images["right_image_stereo"])
        else:
            raise ValueError(f"Unsupported n_external_cameras={n_external}")

        if needs_wrist:
            request["observation/wrist_image_left"] = _prep(images["wrist_image"])
            if needs_stereo:
                request["observation/wrist_image_right"] = _prep(images["wrist_image_stereo"])

        if needs_session_id:
            request["session_id"] = self.session_id

        return request

    def _extract_images(
        self, obs: Dict[str, Any], needs_stereo: bool
    ) -> Dict[str, Optional[np.ndarray]]:
        image_dict = obs.get("image", {}) or {}
        cfg = self.config
        out: Dict[str, Optional[np.ndarray]] = {
            "left_image": None,
            "right_image": None,
            "wrist_image": None,
            "left_image_stereo": None,
            "right_image_stereo": None,
            "wrist_image_stereo": None,
        }

        cam_map = (
            ("left", str(cfg.left_camera_id)),
            ("right", str(cfg.right_camera_id)),
            ("wrist", str(cfg.wrist_camera_id)),
        )
        for key, img in image_dict.items():
            if img is None:
                continue
            for cam_name, cam_id in cam_map:
                if cam_id and cam_id in key:
                    if "left" in key:
                        out[f"{cam_name}_image"] = _strip_alpha_bgr2rgb(img)
                    elif "right" in key and needs_stereo:
                        out[f"{cam_name}_image_stereo"] = _strip_alpha_bgr2rgb(img)
                    break
        return out

    def _safe_action(
        self, joint_pos: np.ndarray, cart_pos: np.ndarray, gripper_pos: float
    ) -> np.ndarray:
        gripper = np.array([gripper_pos], dtype=np.float32)
        if self.action_space == "joint_velocity":
            return np.concatenate([np.zeros(7, dtype=np.float32), gripper])
        if self.action_space == "joint_position":
            return np.concatenate([joint_pos.astype(np.float32)[:7], gripper])
        if self.action_space == "cartesian_velocity":
            return np.concatenate([np.zeros(6, dtype=np.float32), gripper])
        if self.action_space == "cartesian_position":
            return np.concatenate([cart_pos.astype(np.float32)[:6], gripper])
        raise ValueError(self.action_space)


def _strip_alpha_bgr2rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    return img[..., ::-1].copy()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj
