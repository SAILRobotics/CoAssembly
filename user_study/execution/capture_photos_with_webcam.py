"""Capture a photo from the webcam every 0.3 seconds and save it to a folder.

Usage:
    python capture_photos.py [--output FOLDER] [--interval SECONDS] [--camera INDEX] [--no-preview]

A live preview window is shown. Press 'q' in the window (or Ctrl+C) to stop.
"""

import argparse
import time
from pathlib import Path

import cv2

WINDOW_NAME = "Capture (press 'q' to quit)"


def main():
    parser = argparse.ArgumentParser(description="Take a photo every N seconds.")
    parser.add_argument("--output", default="captures", help="Folder to save photos in.")
    parser.add_argument("--interval", type=float, default=0.3, help="Seconds between photos.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument("--no-preview", action="store_true", help="Disable the live preview window.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    show_preview = not args.no_preview
    print(f"Saving photos to {output_dir.resolve()} every {args.interval}s. Press 'q' or Ctrl+C to stop.")

    count = 0
    next_capture = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame, skipping.")
                continue

            # Save on the interval; the preview keeps refreshing in between.
            now = time.monotonic()
            if now >= next_capture:
                count += 1
                filepath = output_dir / f"{count:05d}.jpg"
                cv2.imwrite(str(filepath), frame)
                print(f"Saved {filepath.name}")
                next_capture += args.interval
                # If we fell behind, don't try to catch up in a burst.
                if next_capture < now:
                    next_capture = now + args.interval

            if show_preview:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nStopped. Captured {count} photos.")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
