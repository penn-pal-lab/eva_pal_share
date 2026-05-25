"""Run MolmoAct2 policy under EVA.

Usage:
    python run_molmoact2.py -n 1
"""
import argparse
import time

from tqdm import tqdm

from eva.manager import load_runner
from eva.runner import Runner
from eva.utils.parameters import molmoact2_server_url


def evaluate_policy(runner: Runner, n_traj: int = 1):
    runner.set_controller("molmoact2")

    for _ in tqdm(range(n_traj), disable=(n_traj == 1)):
        current_instr = getattr(runner.controller, "current_instruction", "None")
        print(f"\nCurrent instruction: {current_instr}")
        new_instruction = input("Enter new instruction (Enter to keep): ").strip()
        if new_instruction:
            runner.controller.set_instruction(new_instruction)

        start = time.time()
        runner.run_trajectory(mode="collect")
        print("\033[91mReady to reset, press y/n to mark success/failure...\033[0m")

        info = runner.get_controller_info()
        if info["success"] or info["failure"]:
            print(f"\n\033[91mEVAL == Total steps: {info['t_step']}\033[0m")

        print(f"\033[91mTime: {time.time() - start:.2f}s\033[0m")
        runner.reset_robot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_traj", type=int, default=1)
    args = parser.parse_args()

    runner = load_runner(
        manager=False,
        controller="molmoact2",
        record_depth=False,
        record_pcd=False,
        post_process=True,
    )
    print(f"[MolmoAct2] server: {molmoact2_server_url}")
    evaluate_policy(runner, n_traj=args.n_traj)
