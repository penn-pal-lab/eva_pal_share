"""Vendored from roboarena_evaluator/evaluation_client/base_policy.py."""

import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations.

        Interface (the server declares which fields it expects via metadata):
            Observation:
                - observation/wrist_image_left:  (H, W, 3)   if needs_wrist_camera
                - observation/wrist_image_right: (H, W, 3)   if needs_wrist_camera and needs_stereo_camera
                - observation/exterior_image_{i}_left:  (H, W, 3) for i in 1..n_external_cameras
                - observation/exterior_image_{i}_right: (H, W, 3) if needs_stereo_camera
                - session_id: str if needs_session_id
                - observation/joint_position:    (7,)
                - observation/cartesian_position:(6,)
                - observation/gripper_position:  (1,)
                - prompt: str

            Action:
                - action: (N, 8) or (N, 7) — joint (7 + gripper) or cartesian (6 + gripper). All N actions are
                  executed open-loop before the next infer call.
        """

    def reset(self, reset_info: Dict) -> None:
        """Reset the policy to its initial state."""
        pass
