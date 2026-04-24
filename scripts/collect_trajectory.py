
import argparse
from tqdm import tqdm

from eva.runner import Runner
from eva.manager import load_runner

GRADE_CHOICES = {"A", "B", "C", "D"}


def _prompt_quality_grade() -> dict:
    """Prompt operator for quality grade and optional notes. Returns a dict to merge into metadata.json."""
    while True:
        grade = input("Quality grade (A=excellent / B=good / C=acceptable / D=discard, Enter=skip): ").strip().upper()
        if grade == "":
            return {}
        if grade in GRADE_CHOICES:
            break
        print(f"  Invalid grade '{grade}'. Enter A, B, C, D, or press Enter to skip.")

    updates = {"quality_grade": grade}
    notes = input("Notes (Enter to skip): ").strip()
    if notes:
        updates["notes"] = notes
    scene = input("Scene setup (Enter to skip): ").strip()
    if scene:
        updates["scene_setup"] = scene
    return updates


def collect_trajectory(runner: Runner, controller=None, n_traj=10, practice=False):
    if controller is not None and controller != "spacemouse_pi0":
        runner.set_controller(controller)
    runner.reset_robot()

    for _ in tqdm(range(n_traj), disable=(n_traj == 1)):
        runner.run_trajectory(mode="collect")

        updates = _prompt_quality_grade()
        if updates:
            runner.update_traj_metadata(updates)

        runner.print("Ready to reset, press any controller button...")
        for _ in tqdm(range(1000), desc="Collecting rollouts"):
            controller_info = runner.get_controller_info()
            if controller_info["success"] or controller_info["failure"]:
                break
        runner.reset_robot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_traj", type=int, default=10)
    parser.add_argument("-c", "--controller", default=None, choices=["occulus", "keyboard", "gello", "spacemouse",
                                                                      "replay_pi0", "aawr_pi0", "pi0_policy", "spacemouse_pi0", "policy"])
    parser.add_argument("--practice", action="store_true")
    args = parser.parse_args()

    runner = load_runner(manager=False, controller=args.controller, record_depth=False, record_pcd=False, post_process=True)
    collect_trajectory(runner, controller=args.controller, n_traj=args.n_traj, practice=args.practice)
