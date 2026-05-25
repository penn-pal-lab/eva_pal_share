
import argparse
import subprocess
import os

from eva.utils.misc_utils import get_latest_trajectory, get_latest_image

def send_data(source, destination):
    if not os.path.exists(source):
        # Interpret source as data type, and send the latest data of that type
        assert source in ["latest_trajectory", "latest_image"]
        if source == "latest_trajectory":
            source = get_latest_trajectory()
        elif source == "latest_image":
            source = get_latest_image()

    # Set EVA_CLUSTER_HOST to your training cluster (e.g. user@host); destination is a remote path.
    cluster_host = os.environ.get("EVA_CLUSTER_HOST", "user@cluster.example.com")
    subprocess.run([
        "scp", "-r", source, f"{cluster_host}:{destination}"
    ])
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str)
    parser.add_argument("destination", type=str)
    args = parser.parse_args()

    send_data(args.source, args.destination)