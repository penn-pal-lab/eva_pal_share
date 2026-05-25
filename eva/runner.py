import os
import time
from copy import deepcopy
from datetime import datetime
import cv2
import h5py
from pathlib import Path
import shutil
import threading
import numpy as np
# teleop 
from eva.controllers.occulus import Occulus
from eva.controllers.spacemouse import SpaceMouse
from eva.controllers.keyboard import Keyboard
from eva.controllers.gello import Gello
# policy
from eva.controllers.replayer import Replayer
from eva.controllers.pi0_policy import Pi0Policy
from eva.controllers.human_pi0 import DemoDiffusionPolicy
from eva.controllers.keyboard_pi0 import KeyboardPi0
from eva.controllers.molmoact2 import MolmoAct2Policy

# Active Perception Series
from eva.controllers.policy import Policy # currently fixed as avg pooling aawr policy
from eva.controllers.aawr import AAWRPolicy
from eva.controllers.pi0_spacemouse import SpaceMousePi0
from eva.controllers.aawr_pi0 import AAWRPi0Controller
from eva.controllers.replay_pi0 import ReplayPi0Controller

# writer & utils
from eva.utils.trajectory_utils import run_trajectory
from eva.utils.calibration_utils import calibrate_camera, check_calibration, check_calibration_info, save_calibration_info
from eva.utils.misc_utils import data_dir, run_threaded_command, print_datadict_tree
from eva.utils.parameters import (
    hand_camera_id,
    code_version,
    robot_serial_number,
    robot_type,
    varied_camera_1_id,
    varied_camera_2_id,
)

from eva.utils.misc_utils import yellow_print

class Runner:
    def __init__(self, env, controller, save_data=False, post_process=False, horizon=None):
        yellow_print("RUN === CONTROLLER === ", controller)
        self.env = env
        self.controller = None
        self.set_controller(controller)

        self.traj_running = False
        self.obs_pointer = {}
        self.horizon = horizon
        # Get Camera Info #
        self.cam_ids = list(env.camera_reader.camera_dict.keys())
        self.cam_ids.sort()

        _, full_cam_ids = self.get_camera_feed()
        self.num_cameras = len(full_cam_ids)
        self.full_cam_ids = full_cam_ids
        self.advanced_calibration = False

        self.stop_camera_feed = None
        self.display_thread = None

        # Make Sure Log Directorys Exist #
        self.success_logdir = os.path.join(data_dir, "success", datetime.now().strftime("%Y-%m-%d"))
        self.failure_logdir = os.path.join(data_dir, "failure", datetime.now().strftime("%Y-%m-%d"))
        self.eval_logdir = os.path.join(data_dir, "eval", datetime.now().strftime("%Y-%m-%d"))
        if not os.path.isdir(self.success_logdir):
            os.makedirs(self.success_logdir)
        if not os.path.isdir(self.failure_logdir):
            os.makedirs(self.failure_logdir)
        self.save_data = save_data
        self.post_process = post_process

        # Keyboard HUD state
        self.step_sizes = [0.5, 1.0, 2.0]
        self.step_size_idx = 1
        self._apply_step_scale_to_controller()
        self._instruction_input_active = False

        self.display_camera_feed()

    @property
    def step_scale(self):
        return self.step_sizes[self.step_size_idx]

    def _apply_step_scale_to_controller(self):
        if self.controller is not None and hasattr(self.controller, "set_step_scale"):
            self.controller.set_step_scale(self.step_scale)

    def _cycle_step_size(self):
        self.step_size_idx = (self.step_size_idx + 1) % len(self.step_sizes)
        self._apply_step_scale_to_controller()
        yellow_print(f"[runner] step scale -> {self.step_scale:.2f}x")

    def _prompt_instruction_thread(self):
        try:
            current = getattr(self.controller, "current_instruction", "")
            text = input(f"\n[runner] Enter new instruction [{current}]: ").strip()
            if text:
                if self.set_controller_instruction(text):
                    yellow_print(f"[runner] instruction set: {text}")
                else:
                    yellow_print("[runner] controller does not support set_instruction")
        finally:
            self._instruction_input_active = False

    def _hud_t_step(self):
        """Read the trajectory step counter from the controller, if exposed."""
        state = getattr(self.controller, "_state", None)
        if isinstance(state, dict):
            return state.get("t_step", 0)
        return 0

    def _hud_external_camera_label(self):
        """Return 'left' / 'right' / '' for the controller's external camera.

        Detection order:
          1. controller.external_camera (Pi0-style: "left"/"right")
          2. controller.cfg.external_camera_id mapped via varied_camera_{1,2}_id
        """
        c = self.controller
        if c is None:
            return ""
        label = getattr(c, "external_camera", None)
        if label in ("left", "right"):
            return label
        cam_id = getattr(c, "external_camera_id", None)
        if cam_id is None:
            cfg = getattr(c, "cfg", None)
            if cfg is not None:
                cam_id = getattr(cfg, "external_camera_id", None)
        if cam_id == varied_camera_1_id:
            return "left"
        if cam_id == varied_camera_2_id:
            return "right"
        return ""

    def reset_robot(self):
        self.env._robot.establish_connection() # Why do this?
        self.controller.reset_state()
        self.env.reset()

    def apply_action(self, action): # TODO check with will
        self.env.step(action)

    def get_controller_info(self):
        info = self.controller.get_info()
        return deepcopy(info)

    def enable_advanced_calibration(self):
        self.advanced_calibration = True
        self.env.camera_reader.enable_advanced_calibration()

    def disable_advanced_calibration(self):
        self.advanced_calibration = False
        self.env.camera_reader.disable_advanced_calibration()

    def set_calibration_mode(self, cam_id):
        self.env.camera_reader.set_calibration_mode(cam_id)

    def set_trajectory_mode(self):
        self.env.camera_reader.set_trajectory_mode()

    def run_trajectory(self, mode, reset_robot=True, wait_for_controller=True):
        info = dict(
            time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            robot_serial_number=f"{robot_type}-{robot_serial_number}",
            version_number=code_version,
            controller=self.controller.get_name(),
        )

        if hasattr(self.controller, "current_instruction"):
            info["instruction"] = self.controller.current_instruction

        if hasattr(self.controller, "open_loop_horizon"):
            info["open_loop_horizon"] = self.controller.open_loop_horizon

        traj_name = info["time"]

        if mode == "collect":
            # Assume failure first, move to success post-run
            save_dir = os.path.join(self.failure_logdir, traj_name)
        elif mode == "evaluate":
            save_dir = os.path.join(self.eval_logdir, traj_name)
        elif mode == "practice":
            save_dir, recording_dir, save_filepath = None, None, None
        
        if save_dir is not None:
            if len(self.full_cam_ids) != 6:
                raise ValueError("WARNING: User is trying to collect data without all three cameras running!")
            recording_dir = os.path.join(save_dir, "recordings")
            save_filepath = os.path.join(save_dir, "trajectory.h5")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(recording_dir, exist_ok=True)
            save_calibration_info(os.path.join(save_dir, "calibration.json"))

            # Save instruction to a text file if available
            if hasattr(self.controller, "current_instruction"):
                instr_file = os.path.join(save_dir, "instruction.txt")
                with open(instr_file, "w") as f:
                    f.write(self.controller.current_instruction)
                yellow_print(f"Saved instruction to {instr_file}")

        yellow_print("Saving policy name, if error please add get_policy_name() to the controller!")
        policy_name = self.controller.get_policy_name()
        with open(os.path.join(save_dir, f"policy.md"), "w") as f:
            f.write(f"# Policy\n\n{policy_name}")


        self.traj_running = True
        self.env._robot.establish_connection()
        controller_info = run_trajectory( # This is from trajectory_utils.py
            self.env,
            controller=self.controller,
            horizon=self.horizon,
            metadata=info,
            obs_pointer=self.obs_pointer,
            reset_robot=reset_robot,
            recording_folderpath=recording_dir,
            save_filepath=save_filepath,
            post_process=self.post_process,
            wait_for_controller=wait_for_controller,
        )
        self.traj_running = False
        self.obs_pointer = {}

        if mode == "collect" and save_filepath is not None:
            if controller_info["success"]:
                new_save_dir = os.path.join(self.success_logdir, traj_name)
                shutil.move(save_dir, new_save_dir)
                save_dir = new_save_dir
    
    def calibrate_camera(self, cam_id, reset_robot=True):
        self.traj_running = True
        self.env._robot.establish_connection()
        success = calibrate_camera(
            self.env,
            cam_id,
            controller=self.controller,
            obs_pointer=self.obs_pointer,
            wait_for_controller=True,
            reset_robot=reset_robot,
        )
        self.traj_running = False
        self.obs_pointer = {}
        return success

    def check_calibration(self, reset_robot=True):
        self.traj_running = True
        self.env._robot.establish_connection()
        success = check_calibration(
            self.env,
            controller=self.controller,
            obs_pointer=self.obs_pointer,
            wait_for_controller=True,
            reset_robot=reset_robot
        )
        self.traj_running = False
        self.obs_pointer = {}
        return success

    def check_calibration_info(self, remove_hand_camera=False):
        info_dict = check_calibration_info(self.full_cam_ids)
        if remove_hand_camera:
            info_dict["old"] = [cam_id for cam_id in info_dict["old"] if (hand_camera_id not in cam_id)]
        return info_dict
        
    def display_camera_feed(self, camera_id=None):
        self.stop_camera_feed = threading.Event()

        self.overlay_mode = 1

        def display_thread():
            cv2.namedWindow("eva", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("eva", 1920, 720)
            cv2.setWindowProperty("eva", cv2.WND_PROP_TOPMOST, 1)
            while not self.stop_camera_feed.is_set():
                try:
                    self.camera_feed, self.cam_ids = self.get_camera_feed()
                    if camera_id is not None:
                        self.camera_feed = [feed for i, feed in enumerate(self.camera_feed) if str(camera_id) in self.cam_ids[i] ]
                except Exception as e:
                    # print("Failed to get camera feed:", e)
                    time.sleep(0.1)
                    continue
                                
                from PIL import Image
                overlay_imgs = [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
                for i in range(len(self.camera_feed)):
                    if overlay_imgs[i] is None:
                        continue
                    img = self.camera_feed[i]
                    overlay_img = Image.open(overlay_imgs[i])
                    overlay_img = np.array(overlay_img)
                    if self.overlay_mode == 0:
                        continue
                    elif self.overlay_mode == 1:
                        self.camera_feed[i] = cv2.addWeighted(img, 0.5, overlay_img, 0.5, 0)
                    elif self.overlay_mode == 2:
                        self.camera_feed[i] = overlay_img
                # self.camera_feed = self.camera_feed[2:]

                cols = [np.vstack(self.camera_feed[i:i+2]) for i in range(0, len(self.camera_feed), 2)]
                grid = np.hstack(cols)
                display_img = cv2.cvtColor(
                    cv2.resize(grid, (0, 0), fx=0.5, fy=0.5),
                    cv2.COLOR_RGB2BGR,
                )

                # HUD overlay (BGR yellow): policy / scale / instruction / step counter / ext camera
                hud_lines = []
                if self.controller is not None and hasattr(self.controller, "get_name"):
                    try:
                        hud_lines.append(f"policy: {self.controller.get_name()}")
                    except Exception:
                        pass
                hud_lines.append(f"scale: {self.step_scale:.2f}x  (l to cycle)")
                if hasattr(self.controller, "current_instruction"):
                    instr = (self.controller.current_instruction or "")[:80]
                    if instr:
                        hud_lines.append(f"instr: {instr}")
                hud_lines.append(f"step: {self._hud_t_step()}")
                ext_label = self._hud_external_camera_label()
                if ext_label:
                    hud_lines.append(f"ext: {ext_label}")

                y = 30
                for line in hud_lines:
                    cv2.putText(
                        display_img, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    )
                    y += 28

                cv2.imshow("eva", display_img)

                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    # Runner-owned keys (work for any controller).
                    if key == ord('o'):
                        self.overlay_mode = (self.overlay_mode + 1) % 3
                    elif key == ord('l'):
                        self._cycle_step_size()
                    elif key == ord('i'):
                        if not self._instruction_input_active:
                            self._instruction_input_active = True
                            threading.Thread(
                                target=self._prompt_instruction_thread,
                                daemon=True,
                            ).start()
                    elif self.controller is not None:
                        # Everything else (w/s/a/d/1/2/5/6/7/8/j/k/y/n/space ...) is a controller key.
                        self.controller.register_key(key)

            cv2.destroyAllWindows()
        self.display_thread = run_threaded_command(display_thread)


    def get_gui_imgs(self, obs):
        all_cam_ids = list(obs["image"].keys())
        all_cam_ids.sort()

        gui_images = []
        for cam_id in all_cam_ids:
            img = cv2.cvtColor(obs["image"][cam_id], cv2.COLOR_BGRA2RGB)
            gui_images.append(img)
        
        # Optional: add depth camera feed
        # depth_cam_ids = list(obs["depth"].keys())
        # depth_cam_ids.sort()
        # import numpy as np
        # for cam_id in depth_cam_ids:
        #     depth = np.nan_to_num(obs["depth"][cam_id])
        #     img = cv2.cvtColor(depth, cv2.COLOR_BGRA2RGB)
        #     gui_images.append(img)
        # all_cam_ids.extend([id+"_depth" for id in depth_cam_ids])

        return gui_images, all_cam_ids

    def get_camera_feed(self):
        if self.traj_running:
            if "image" not in self.obs_pointer:
                raise ValueError
            obs = deepcopy(self.obs_pointer)
        else:
            obs = self.env.read_cameras()[0]
        gui_images, cam_ids = self.get_gui_imgs(obs)
        return gui_images, cam_ids
    
    def get_obs(self):
        if self.traj_running:
            yellow_print("Traj mode")
            if "image" not in self.obs_pointer:
                raise ValueError
            obs = deepcopy(self.obs_pointer)
        else:
            yellow_print("Not traj mode")
            obs = self.env.read_cameras()[0]
        return obs

    def get_robot_state(self): # Written by Tony
        state_dict, _ = self.env._robot.get_robot_state()
        return state_dict

    def close_camera_feed(self):
        if self.stop_camera_feed is not None and self.display_thread is not None:
            self.stop_camera_feed.set()
            self.display_thread.join()
            self.stop_camera_feed = None
            self.display_thread = None
    
    def set_action_space(self, action_space):
        self.env.set_action_space(action_space)
    
    def set_controller(self, controller, **kwargs):
        yellow_print("Setting controller:", controller)
        if controller is None:
            return
        # Avoid reopening SpaceMouse HID device (exclusive access)
        # TODO: rewrite `run_eva_policy.py` to avoid this hack
        if controller == "spacemouse" and isinstance(self.controller, SpaceMouse):
            yellow_print("Controller already set to spacemouse\n=================\n")
            return

        def update_action_spaces(action_space, gripper_action_space):
            yellow_print(f"RUNNER == Updating action spaces - Action: {action_space}, Gripper: {gripper_action_space} ==========")
            self.env.set_action_space(action_space)
            self.env.set_gripper_action_space(gripper_action_space)
        
        
        self.prev_controller = self.controller
        if controller == "occulus":
            self.controller = Occulus()
        elif controller == "keyboard":
            self.controller = Keyboard()
        elif controller == "gello":
            self.controller = Gello()
        elif controller == "spacemouse": 
            self.controller = SpaceMouse()
        elif controller == "policy":
            self.controller = Policy(policy_path="XXX",**kwargs)
        elif controller == "replayer":
            self.controller = Replayer(**kwargs)
        elif controller == "pi0_policy":
            self.controller = Pi0Policy(**kwargs)
        elif controller == "molmoact2":
            self.controller = MolmoAct2Policy(**kwargs)
        elif controller == "demodiffusion_pi0":
            kwargs['on_switch_callback'] = update_action_spaces
            self.controller = DemoDiffusionPolicy(**kwargs)
        elif controller == "aawr_pi0":
            yellow_print('using aawr_pi0')
            kwargs['on_switch_callback'] = update_action_spaces
            self.controller = AAWRPi0Controller(**kwargs)
        elif controller == "replay_pi0":
            kwargs['on_switch_callback'] = update_action_spaces
            self.controller = ReplayPi0Controller(**kwargs)
        elif controller == "spacemouse_pi0":
            kwargs['on_switch_callback'] = update_action_spaces
            self.controller = SpaceMousePi0(**kwargs)
        elif controller == "keyboard_pi0":
            kwargs['on_switch_callback'] = update_action_spaces
            self.controller = KeyboardPi0(**kwargs)
        else:
            raise ValueError(f"Controller {controller} not recognized!")
        yellow_print("Controller set to", self.controller.get_name(), "\n=================\n")

        # Pass env to controller if it needs robot access (e.g., for IK/action conversion)
        if hasattr(self.controller, 'set_env'):
            self.controller.set_env(self.env)

        # Re-apply current step scale to the new controller (no-op if unsupported)
        if hasattr(self, "step_sizes"):
            self._apply_step_scale_to_controller()

        self.env.set_action_space(self.controller.action_space)
        self.env.set_gripper_action_space(self.controller.gripper_action_space)
    


    def set_prev_controller(self):
        self.controller = self.prev_controller
        self.env.set_action_space(self.controller.action_space)
        self.env.set_gripper_action_space(self.controller.gripper_action_space)
    
    def reload_calibration(self):
        self.env.reload_calibration()
    
    def yellow_print(self, string):
        # This is used by scripts to yellow_print to the runner console instead of the script console
        # In general, we want to yellow_print everything to the runner console
        yellow_print(string)

    def set_controller_instruction(self, instruction):
        """Set instruction for the controller if it supports it."""
        if hasattr(self.controller, 'set_instruction'):
            self.controller.set_instruction(instruction)
            return True
        return False

    def close(self):
        self.reset_robot()
        self.close_camera_feed()
        self.env.close()
        self.controller.close()#
