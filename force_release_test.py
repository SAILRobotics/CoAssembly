"""
force_release_test.py — Standalone proof-of-concept: release the gripper
when a human pulls on the grasped object.

Idea being tested
------------------
Two symmetric force-threshold triggers, both measured as a delta from a
baseline captured at the start of whichever state is currently active:

  - Idle (gripper open): baseline = resting TCP force with nothing in the
    jaws. If a human presses/pushes an object into the open gripper and the
    force delta exceeds GRASP_FORCE_THRESHOLD for several consecutive
    ticks, the gripper closes automatically (auto-grasp).
  - Monitoring (gripper closed): baseline = TCP force right after closing
    (already includes the object's static weight). If a human tugs the
    object and the force delta exceeds RELEASE_FORCE_THRESHOLD for several
    consecutive ticks, the gripper opens automatically (auto-release).

Manual control always works too: ENTER grasps/releases on demand regardless
of the thresholds, useful for testing and for --tune mode.

Deliberately decoupled from the rest of the pipeline — no Unity, no
PyBullet/IK, no motion commands at all. Only reads robot state (RTDE
receive) and drives the gripper directly.

Force is polled continuously for the entire session — including while the
gripper is open and idle, not just while actively monitoring a grasp — so
the graph (and the printed status) never has gaps.

Usage
-----
    python force_release_test.py [--robot-ip 192.168.50.70] [--tune] [--graph]

    Press ENTER to manually grasp/release (toggles with state); press 'q'
    to quit (releasing first if currently grasping).

    --tune disables both automatic triggers and instead prints the live
    force delta every tick (for whichever state is active), so you can find
    sensible GRASP_FORCE_THRESHOLD / RELEASE_FORCE_THRESHOLD values for
    your object/gripper before relying on them.

    --graph opens a live-updating plot of Fx/Fy/Fz vs session time, redrawn
    every poll tick for as long as the script runs (across grasp cycles —
    grasp/release moments are marked with vertical lines).
"""

import argparse
import msvcrt
import sys
import time
from pathlib import Path

import numpy as np

import main_setting as cfg

from rtde_receive import RTDEReceiveInterface

sys.path.insert(0, str(Path(__file__).parent / "robot_controller"))
from robotiq_gripper import RobotiqGripper

# ── Tunables ──────────────────────────────────────────────────────────────────
GRIPPER_PORT            = 63352
GRASP_FORCE_THRESHOLD   = 2.5   # N — PLACEHOLDER. Tune with --tune (idle state) before relying on this.
RELEASE_FORCE_THRESHOLD = 20.0   # N — PLACEHOLDER. Tune with --tune (monitoring state) before relying on this.
DEBOUNCE_HITS_REQUIRED  = 5      # consecutive over-threshold ticks before acting
POLL_PERIOD_S           = 0.05   # 20 Hz
GRIP_SPEED              = 255
GRIP_FORCE_CLOSE        = 100
GRIP_FORCE_OPEN         = 10


def poll_force(rtde: RTDEReceiveInterface) -> "np.ndarray | None":
    """[Fx,Fy,Fz,Mx,Myq,Mz] (N / Nm, base frame), or None on a bad read."""
    try:
        return np.array(rtde.getActualTCPForce(), dtype=float)
    except Exception:
        return None


class LiveForcePlot:
    """A single matplotlib window that keeps growing for the whole session —
    update() is called once per poll tick and redraws immediately."""

    def __init__(self):
        import matplotlib.pyplot as plt   # deferred — only needed when --graph is used
        self._plt = plt
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(9, 5))
        self._t, self._fx, self._fy, self._fz = [], [], [], []
        self._lines = {
            "Fx": self._ax.plot([], [], label="Fx", color="C0")[0],
            "Fy": self._ax.plot([], [], label="Fy", color="C1")[0],
            "Fz": self._ax.plot([], [], label="Fz", color="C2")[0],
        }
        self._ax.set_xlabel("Session time (s)")
        self._ax.set_ylabel("TCP force (N)")
        self._ax.set_title("TCP force vs time")
        self._ax.legend()
        self._fig.tight_layout()

    def update(self, t: float, force_xyz: np.ndarray) -> None:
        self._t.append(t)
        self._fx.append(force_xyz[0])
        self._fy.append(force_xyz[1])
        self._fz.append(force_xyz[2])
        self._lines["Fx"].set_data(self._t, self._fx)
        self._lines["Fy"].set_data(self._t, self._fy)
        self._lines["Fz"].set_data(self._t, self._fz)
        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw_idle()

    def mark_event(self, t: float, label: str, color: str = "gray") -> None:
        self._ax.axvline(t, color=color, linestyle="--", alpha=0.6)
        self._ax.annotate(label, (t, self._ax.get_ylim()[1]),
                          rotation=90, va="top", fontsize=8, color=color)

    def pause(self, period_s: float) -> None:
        """Sleeps AND processes GUI events so the window actually redraws."""
        self._plt.pause(period_s)

    def block_until_closed(self) -> None:
        self._plt.ioff()
        self._plt.show()


def _wait(plot: "LiveForcePlot | None") -> None:
    if plot is not None:
        plot.pause(POLL_PERIOD_S)
    else:
        time.sleep(POLL_PERIOD_S)


def _check_key() -> "str | None":
    """Non-blocking single-keypress check (Windows). Returns the lowercased
    key, or None if nothing was pressed since the last check."""
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        try:
            return ch.decode(errors="ignore").lower()
        except Exception:
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Auto-grasp on contact, auto-release on pull — both via TCP force thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--robot-ip", default=cfg.ROBOT_IP,
                    help="UR robot controller IP for live TCP force via RTDE")
    ap.add_argument("--tune", action="store_true",
                    help="Print live force delta instead of ever triggering — "
                         "use this first to pick GRASP/RELEASE_FORCE_THRESHOLD")
    ap.add_argument("--graph", action="store_true",
                    help="Live-plot Fx/Fy/Fz vs session time for the whole "
                         "run (requires matplotlib)")
    args = ap.parse_args()

    rtde = None
    g = None
    plot = None
    t0 = time.perf_counter()
    try:
        try:
            rtde = RTDEReceiveInterface(args.robot_ip)
            print(f"[RTDE] Connected to {args.robot_ip}")
        except Exception as e:
            print(f"[RTDE] Could not connect to {args.robot_ip}: {e}")
            return

        print(f"[Gripper] Connecting → {args.robot_ip}:{GRIPPER_PORT}")
        g = RobotiqGripper()
        g.connect(args.robot_ip, GRIPPER_PORT)
        g.activate()
        print("[Gripper] Ready.")

        if args.graph:
            plot = LiveForcePlot()

        state    = "idle"      # "idle" | "monitoring"
        baseline = None        # set on entering whichever state is active
        hits     = 0

        def _rebaseline() -> "np.ndarray | None":
            b = poll_force(rtde)
            return b[:3] if b is not None else None

        def do_grasp(reason: str) -> None:
            nonlocal state, baseline, hits
            g.move_and_wait_for_pos(g.get_closed_position(),
                                    GRIP_SPEED, GRIP_FORCE_CLOSE)
            print(f"\n[Grasp:{reason}] Gripper closed (is_closed={g.is_closed()}).")
            if plot is not None:
                plot.mark_event(time.perf_counter() - t0, f"grasp ({reason})", color="green")
            baseline = _rebaseline()
            if baseline is None:
                print("[Grasp] Could not read baseline force — staying idle.")
                return
            print(f"[Monitor] Baseline force (N) = "
                  f"[{baseline[0]:+.2f}, {baseline[1]:+.2f}, {baseline[2]:+.2f}]. "
                  f"Monitoring for pull...")
            state, hits = "monitoring", 0

        def do_release(reason: str) -> None:
            nonlocal state, baseline, hits
            g.move_and_wait_for_pos(g.get_open_position(),
                                    GRIP_SPEED, GRIP_FORCE_OPEN)
            print(f"\n[Release:{reason}] Gripper opened.")
            if plot is not None:
                plot.mark_event(time.perf_counter() - t0, f"release ({reason})",
                               color="red" if reason == "force" else "orange")
            state, hits = "idle", 0
            baseline = _rebaseline()
            print("\n[Idle] Press ENTER to grasp, 'q' to quit.")

        baseline = _rebaseline()
        print("\n[Idle] Force is now being read continuously. Press ENTER to "
              "grasp, 'q' to quit.")

        while True:
            # ── Force is read every tick regardless of state ────────────────
            force = poll_force(rtde)
            if force is not None and plot is not None:
                plot.update(time.perf_counter() - t0, force[:3])

            if force is not None and baseline is None:
                baseline = force[:3]   # opportunistic recapture if the first read failed

            if force is not None and baseline is not None:
                delta = float(np.linalg.norm(force[:3] - baseline))
                threshold = (GRASP_FORCE_THRESHOLD if state == "idle"
                            else RELEASE_FORCE_THRESHOLD)

                if args.tune:
                    print(f"\r[Tune:{state}] delta={delta:6.2f} N "
                          f"(threshold={threshold:.1f})", end="", flush=True)
                else:
                    hits = hits + 1 if delta > threshold else 0
                    if hits >= DEBOUNCE_HITS_REQUIRED:
                        print(f"\n[{state.upper()} TRIGGER] delta={delta:.1f} N exceeded "
                              f"{threshold} N for {DEBOUNCE_HITS_REQUIRED} consecutive ticks.")
                        if state == "idle":
                            do_grasp("force")
                        else:
                            do_release("force")

            # ── Non-blocking keypress handling ───────────────────────────────
            key = _check_key()
            if key in ("\r", "\n"):
                if state == "idle":
                    do_grasp("key")
                else:
                    do_release("key")
            elif key == "q":
                if state == "monitoring":
                    do_release("quit")
                break

            _wait(plot)

    except KeyboardInterrupt:
        pass
    finally:
        if g is not None:
            try:
                g.disconnect()
            except Exception:
                pass
        if rtde is not None:
            try:
                rtde.disconnect()
            except Exception:
                pass
        print("[Done]")
        if plot is not None:
            print("[Graph] Close the plot window to exit.")
            plot.block_until_closed()


if __name__ == "__main__":
    main()
