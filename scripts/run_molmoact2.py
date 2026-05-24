"""Run the MolmoAct2 policy. See README.md for endpoint config + protocol."""
import argparse
import time

from tqdm import tqdm

from eva.manager import load_runner
from eva.runner import Runner
from eva.controllers.molmoact2 import resolve_molmoact2_endpoint
from eva.utils.parameters import MOLMOACT2_ENDPOINTS


def evaluate_policy(runner: Runner, endpoint, n_traj: int = 1):
    runner.set_controller("molmoact2", endpoint=endpoint)

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
    parser.add_argument(
        "-s", "--server", default="lan",
        help=f"Preset name {sorted(MOLMOACT2_ENDPOINTS)} or full http(s):// URL.",
    )
    args = parser.parse_args()

    try:
        endpoint = resolve_molmoact2_endpoint(args.server)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"[MolmoAct2] {args.server!r} -> {endpoint}")

    runner = load_runner(
        manager=False,
        controller="molmoact2",
        record_depth=False,
        record_pcd=False,
        post_process=True,
    )
    evaluate_policy(runner, endpoint=endpoint, n_traj=args.n_traj)
