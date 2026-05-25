import os

from cv2 import aruco

##### ROBOT #####
# 172.16.0.x are the standard DROID NUC/robot IPs; override via env if your setup differs.
nuc_ip = os.environ.get("NUC_IP", "172.16.0.4")
robot_ip = os.environ.get("ROBOT_IP", "172.16.0.2")
laptop_ip = os.environ.get("LAPTOP_IP", "127.0.0.1")
sudo_password = os.environ.get("EVA_SUDO_PASSWORD", "")  # set in your shell, never commit
robot_type = "fr3"  # 'panda' or 'fr3'
robot_serial_number = ""


##### POLICY SERVER (MolmoAct2) #####
# name -> {url, norm_tag}. norm_tag=None omits the field (legacy LAN protocol);
# the ngrok build requires "franka_droid". See README.md / CLAUDE.md for the
# protocol. Point these at your own MolmoAct2 host server
# (allenai/molmoact2: examples/droid/host_server_droid.py).
policy_server_ip = os.environ.get("POLICY_SERVER_IP", "127.0.0.1")
policy_server_port = int(os.environ.get("POLICY_SERVER_PORT", "8101"))

MOLMOACT2_ENDPOINTS = {
    "lan":   {"url": f"http://{policy_server_ip}:{policy_server_port}/act", "norm_tag": None},
    # Replace with your own tunnel (or pass a raw URL via --server).
    "ngrok": {"url": os.environ.get("MOLMOACT2_NGROK_URL", "https://YOUR-TUNNEL.ngrok-free.dev/act"), "norm_tag": "franka_droid"},
}
molmoact2_server_url = MOLMOACT2_ENDPOINTS["lan"]["url"]  # default for MolmoAct2Config

##### CAMERAS #####
# ZED camera serial numbers — fill in your own (find them with ZED_Explorer or
# `zed.get_camera_information().serial_number`). Override via env if you prefer.
hand_camera_id = os.environ.get("HAND_CAMERA_ID", "00000000")        # ZED Mini (wrist)
varied_camera_1_id = os.environ.get("VARIED_CAMERA_1_ID", "11111111")  # ZED 2 (shoulder)
varied_camera_2_id = os.environ.get("VARIED_CAMERA_2_ID", "22222222")  # ZED 2 (side)


camera_type_dict = {
    hand_camera_id: 0,
    varied_camera_1_id: 1,
    varied_camera_2_id: 2,
}
camera_type_to_string_dict = {
    0: "hand_camera",
    1: "varied_camera_1",
    2: "varied_camera_2",
}
camera_flip_dict = {
    hand_camera_id: False,
    varied_camera_1_id: True,
    varied_camera_2_id: True,
}

def get_camera_type(cam_id):
    if cam_id not in camera_type_dict:
        return None
    type_int = camera_type_dict[cam_id]
    type_str = camera_type_to_string_dict[type_int]
    return type_str


##### SPACEMOUSE #####
SPACEMOUSE_OVERRIDE_CONFIG = True
spacemouse_config = {
    "max_lin_vel": 5.0,
    "max_rot_vel": 5.0,
    "max_gripper_vel": 5.0,
    "pos_sensitivity": 8.0,
    "rot_sensitivity": 8.0,
    "action_scale": 0.1,
    "deadzone": 0.05,
    "smoothing": 0.3
}

##### CHARUCO BOARD #####
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 12
CHARUCOBOARD_CHECKER_SIZE = 0.030
CHARUCOBOARD_MARKER_SIZE = 0.023
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

ubuntu_pro_token = ""

##### CODE VERSION #####
code_version = "2.0"
