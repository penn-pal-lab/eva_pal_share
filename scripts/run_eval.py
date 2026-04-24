"""
Evaluation session manager for Pi0 policy.

Usage:
  python scripts/run_eval.py -n 10
  python scripts/run_eval.py -n 10 --instruction "pick up the pineapple" --host 10.102.212.31

Each session produces:
  data/eval_sessions/eval_session_{timestamp}.json   — aggregate results
  data/eval/{date}/{traj_id}/metadata.json           — per-trajectory (written by runner)
"""

import argparse
import json
import os
import time
from datetime import datetime

from eva.manager import load_runner
from eva.controllers.pi0_policy import Pi0PolicyConfig
from eva.utils.misc_utils import data_dir
from eva.utils.parameters import robot_serial_number, robot_type


EVAL_SESSIONS_DIR = os.path.join(data_dir, "eval_sessions")


def _prompt(label: str, default: str = "") -> str:
    if default:
        value = input(f"{label} [{default}]: ").strip()
        return value if value else default
    return input(f"{label}: ").strip()


def _prompt_session_header(policy_name: str) -> dict:
    print("\n\033[93m=== Evaluation Session Setup ===\033[0m")
    return {
        "task":              _prompt("Task name (e.g. pick_pineapple)"),
        "policy":            _prompt("Policy name", default=policy_name),
        "evaluator":         _prompt("Evaluator name"),
        "scene_description": _prompt("Scene description (Enter to skip)"),
    }


def _prompt_episode_notes() -> dict:
    notes = input("Episode notes (Enter to skip): ").strip()
    return {"notes": notes} if notes else {}


def _print_summary(episodes: list):
    n = len(episodes)
    if n == 0:
        return
    n_success = sum(1 for e in episodes if e["outcome"] == "success")
    avg_steps = sum(e["steps"] for e in episodes) / n
    avg_dur = sum(e["duration_sec"] for e in episodes) / n
    print(f"\n\033[93m=== Session Summary ===\033[0m")
    print(f"  Episodes  : {n}")
    print(f"  Success   : {n_success}/{n}  ({n_success/n*100:.1f}%)")
    print(f"  Avg steps : {avg_steps:.1f}")
    print(f"  Avg time  : {avg_dur:.1f}s")


def run_eval(runner, n_traj: int):
    policy_name = runner.controller.get_name()
    header = _prompt_session_header(policy_name)

    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    episodes = []

    os.makedirs(EVAL_SESSIONS_DIR, exist_ok=True)
    session_path = os.path.join(EVAL_SESSIONS_DIR, f"eval_session_{session_id}.json")

    for ep_idx in range(n_traj):
        print(f"\n\033[93m--- Episode {ep_idx + 1}/{n_traj} ---\033[0m")

        # Allow updating instruction before each episode
        current_instr = getattr(runner.controller, "current_instruction", "")
        new_instr = input(f"Instruction [{current_instr}]: ").strip()
        if new_instr:
            runner.controller.set_instruction(new_instr)
        instruction = getattr(runner.controller, "current_instruction", "")

        start_time = time.time()
        controller_info = runner.run_trajectory(mode="evaluate")
        duration_sec = round(time.time() - start_time, 2)

        outcome = "success" if controller_info.get("success") else "failure"
        steps = controller_info.get("t_step", 0)

        print(f"\033[91mOutcome: {outcome} | Steps: {steps} | Time: {duration_sec:.1f}s\033[0m")

        notes_update = _prompt_episode_notes()
        # Write session_id + notes into the trajectory's metadata.json
        traj_meta_update = {"session_id": session_id}
        traj_meta_update.update(notes_update)
        runner.update_traj_metadata(traj_meta_update)

        episode = {
            "episode_idx": ep_idx,
            "traj_id":     os.path.basename(runner.last_traj_dir) if runner.last_traj_dir else "",
            "traj_path":   runner.last_traj_dir or "",
            "instruction": instruction,
            "outcome":     outcome,
            "steps":       steps,
            "duration_sec": duration_sec,
        }
        episode.update(notes_update)
        episodes.append(episode)

        # Save session JSON incrementally so partial sessions are recoverable
        _save_session(session_path, session_id, header, episodes)

        runner.reset_robot()

    _print_summary(episodes)
    print(f"\n\033[93mSession saved to: {session_path}\033[0m")


def _save_session(path: str, session_id: str, header: dict, episodes: list):
    n = len(episodes)
    n_success = sum(1 for e in episodes if e["outcome"] == "success")
    session = {
        "session_id": session_id,
        "robot": f"{robot_type}-{robot_serial_number}",
        **header,
        "episodes": episodes,
        "summary": {
            "n_episodes":       n,
            "n_success":        n_success,
            "success_rate":     round(n_success / n, 3) if n else 0.0,
            "avg_steps":        round(sum(e["steps"] for e in episodes) / n, 1) if n else 0,
            "avg_duration_sec": round(sum(e["duration_sec"] for e in episodes) / n, 1) if n else 0.0,
        },
    }
    with open(path, "w") as f:
        json.dump(session, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pi0 evaluation session manager")
    parser.add_argument("-n", "--n_traj", type=int, default=10, help="Number of episodes")
    parser.add_argument("--host",        default=Pi0PolicyConfig.remote_host,       help="Policy server IP")
    parser.add_argument("--port",        type=int, default=Pi0PolicyConfig.remote_port, help="Policy server port")
    parser.add_argument("--instruction", default=Pi0PolicyConfig.instruction,        help="Initial task instruction")
    parser.add_argument("--horizon",     type=int, default=Pi0PolicyConfig.open_loop_horizon, help="Open-loop horizon")
    parser.add_argument("--external",    default=Pi0PolicyConfig.external_camera,   choices=["left", "right"])
    args = parser.parse_args()

    runner = load_runner(manager=False, controller="pi0_policy", record_depth=False,
                         record_pcd=False, post_process=True)

    runner.controller.remote_host = args.host
    runner.controller.remote_port = args.port
    runner.controller.external_camera = args.external
    runner.controller.set_instruction(args.instruction)
    runner.controller.set_horizon(args.horizon)

    run_eval(runner, n_traj=args.n_traj)
