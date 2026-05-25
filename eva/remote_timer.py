import os
import time

# Optional remote timer (https://ably.com). Set your own key in the environment:
#   export ABLY_API_KEY="<your-ably-key>"
ABLY_API_KEY = os.environ.get("ABLY_API_KEY", "")

class RemoteTimer:
    """Publishes timer events to an Ably channel. If ABLY_API_KEY is unset (or
    the ably package is missing), it degrades to a no-op so the rest of Eva runs
    without the optional remote timer."""

    def __init__(self):
        self.channel = None
        if not ABLY_API_KEY:
            return
        try:
            from ably.sync import AblyRestSync
        except ImportError:
            print("[RemoteTimer] ably not installed; remote timer disabled (pip install ably).")
            return
        self.ably = AblyRestSync(ABLY_API_KEY)
        self.channel = self.ably.channels.get("timer-control")

    def reset(self):
        if self.channel is None:
            return
        self.channel.publish("command", {"command": "reset", "status": ""})

    def toggle(self, status: str):
        if self.channel is None:
            return
        self.channel.publish("command", {"command": "toggle", "status": status})

    def set_status(self, status: str):
        if self.channel is None:
            return
        self.channel.publish("command", {"command": "noop", "status": status})


if __name__ == "__main__":
    remote_timer = RemoteTimer()
    remote_timer.reset()
    time.sleep(1)
    remote_timer.toggle("on")
    time.sleep(1)
    remote_timer.set_status("running")
    time.sleep(1)
    remote_timer.toggle("off")