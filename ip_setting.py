# Global IP and port configuration.
# Edit the IP addresses here and import this module everywhere else.

# ── Machine IPs ────────────────────────────────────────────────────────────────
UNITY_IP   = "192.168.50.201"   # Quest / Windows machine running Unity
UBUNTU_IP  = "192.168.50.100"   # Ubuntu / Linux machine
WINDOWS_IP = "0.0.0.0"          # bind address on this machine (all interfaces)
LOCALHOST  = "127.0.0.1"        # loopback for same-machine inter-process comms

# ── Hand tracking ports ────────────────────────────────────────────────────────
# Unity PUBs, we SUB
HAND1_PORT_FROM_UNITY  = 5570   # real hand tracking stream
HAND2_PORT_FROM_UNITY  = 5571   # synthetic hand tracking stream (optional)

# ── World transform port ───────────────────────────────────────────────────────
# main.py PUBs T_world_tracking (Open3D, 4×4) here; main_hand.py SUBs
WORLD_TRANSFORM_PORT   = 5008

# ── Helpers ────────────────────────────────────────────────────────────────────
def to_unity(port: int) -> str:
    return f"tcp://{UNITY_IP}:{port}"

def to_ubuntu(port: int) -> str:
    return f"tcp://{UBUNTU_IP}:{port}"

def bind_local(port: int) -> str:
    return f"tcp://{WINDOWS_IP}:{port}"
