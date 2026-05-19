"""Quick diagnostic: listen on the hand tracking port and print what Unity sends."""
import zmq
import ip_setting as cfg

PORT = cfg.HAND1_PORT_FROM_UNITY  # 5570
IP   = cfg.UNITY_IP

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt_string(zmq.SUBSCRIBE, "")
sub.connect(f"tcp://{IP}:{PORT}")

print(f"Listening for hand data on tcp://{IP}:{PORT} ...")
print("Press Ctrl+C to stop.\n")

count = 0
try:
    while True:
        msg = sub.recv_string()
        count += 1
        try:
            import json
            d = json.loads(msg)
            hands = d.get("hands") or {}
            present = [k for k, v in hands.items() if v is not None]
            print(f"[#{count}] {len(msg)} bytes  |  hands present: {present}")
        except Exception:
            preview = msg[:200].replace("\n", " ")
            print(f"[#{count}] {len(msg)} bytes  |  {preview}")
except KeyboardInterrupt:
    print(f"\nReceived {count} messages total.")
finally:
    sub.close()
    ctx.term()
