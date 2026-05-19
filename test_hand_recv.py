"""Quick diagnostic: listen on the hand tracking port and print what Unity sends."""
import argparse
import time

import zmq
import ip_setting as cfg

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default=cfg.UNITY_IP)
parser.add_argument("--port", type=int, default=cfg.HAND1_PORT_FROM_UNITY)
parser.add_argument("--timeout", type=float, default=5.0)
args = parser.parse_args()

PORT = args.port
IP   = args.ip

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt_string(zmq.SUBSCRIBE, "")
sub.connect(f"tcp://{IP}:{PORT}")

print(f"Listening for hand data on tcp://{IP}:{PORT} ...")
print("Press Ctrl+C to stop.\n")

count = 0
last_msg_time = time.time()
poller = zmq.Poller()
poller.register(sub, zmq.POLLIN)
try:
    while True:
        events = dict(poller.poll(timeout=500))
        if sub not in events:
            if count == 0 and time.time() - last_msg_time >= args.timeout:
                print(f"No messages after {args.timeout:.1f}s.")
                print("Check Quest IP, Unity HandTrackingSenderNetMQ port, and whether the sender is bound in headset logs.")
                last_msg_time = time.time()
            continue

        parts = sub.recv_multipart()
        count += 1
        last_msg_time = time.time()

        if len(parts) == 1:
            msg = parts[0].decode("utf-8", errors="replace")
            framing = "single"
        else:
            msg = parts[-1].decode("utf-8", errors="replace")
            framing = f"multipart({len(parts)})"

        try:
            import json
            d = json.loads(msg)
            hands = d.get("hands") or {}
            present = [k for k, v in hands.items() if v is not None]
            print(f"[#{count}] {framing}  {len(msg)} chars  |  hands present: {present}")
        except Exception:
            preview = msg[:200].replace("\n", " ")
            print(f"[#{count}] {framing}  {len(msg)} chars  |  {preview}")
except KeyboardInterrupt:
    print(f"\nReceived {count} messages total.")
finally:
    sub.close()
    ctx.term()
