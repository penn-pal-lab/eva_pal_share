
import argparse
from tqdm import tqdm
import time
from eva.runner import Runner
from eva.manager import load_runner
from eva.controllers.pi0_policy import Pi0PolicyConfig  # for default values


def evaluate_policy(runner: Runner, n_traj=1):
    for _ in tqdm(range(n_traj), disable=(n_traj == 1)):
        current_instr = getattr(runner.controller, 'current_instruction', 'None')
        print(f"\nCurrent instruction: {current_instr}")
        new_instruction = input("Enter new instruction (press Enter to keep current): ").strip()
        if new_instruction:
            runner.controller.set_instruction(new_instruction)

        start_time = time.time()
        runner.run_trajectory(mode="evaluate")
        print("\033[91mReady to reset, press any controller button...\033[0m")

        controller_info = runner.get_controller_info()
        if controller_info["success"] or controller_info["failure"]:
            print("\n\n\033[91mEVAL == Total steps: ", controller_info["t_step"], "\033[0m")

        end_time = time.time()
        print(f"\033[91mTime taken: {end_time - start_time:.2f} seconds\033[0m")
        runner.reset_robot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pi0 policy evaluation")
    parser.add_argument("-n", "--n_traj", type=int, default=10)
    parser.add_argument("--host", default=Pi0PolicyConfig.remote_host, help="Policy server IP")
    parser.add_argument("--port", type=int, default=Pi0PolicyConfig.remote_port, help="Policy server port")
    parser.add_argument("--instruction", default=Pi0PolicyConfig.instruction, help="Task instruction")
    parser.add_argument("--horizon", type=int, default=Pi0PolicyConfig.open_loop_horizon, help="Open-loop horizon")
    parser.add_argument("--external", default=Pi0PolicyConfig.external_camera, choices=["left", "right"],
                        help="Which external camera to use")
    args = parser.parse_args()

    runner = load_runner(manager=False, controller="pi0_policy", record_depth=False,
                         record_pcd=False, post_process=True)

    # Pi0Policy connects lazily on first forward(), so these updates are safe here.
    runner.controller.remote_host = args.host
    runner.controller.remote_port = args.port
    runner.controller.external_camera = args.external
    runner.controller.set_instruction(args.instruction)
    runner.controller.set_horizon(args.horizon)

    evaluate_policy(runner, n_traj=args.n_traj)
