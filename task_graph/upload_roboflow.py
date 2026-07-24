#!/usr/bin/env python3
"""Upload every image in a folder to a Roboflow project's annotation queue."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


# Optional: paste your private Roboflow API key between the quotes.
# Keep this empty to use the ROBOFLOW_API_KEY environment variable instead.
ROBOFLOW_API_KEY = "ikVoX6gD1IAFWgMeUx6x"

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}


def find_images(folder: Path, recursive: bool) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def encode_multipart(image_path: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"----roboflow-{secrets.token_hex(16)}"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        body.extend(value.encode())
        body.extend(b"\r\n")

    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            'Content-Disposition: form-data; name="file"; '
            f'filename="{image_path.name}"\r\n'
        ).encode()
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.extend(image_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def upload_image(
    image_path: Path,
    project: str,
    api_key: str,
    split: str,
    batch: str | None,
    tags: list[str],
    retries: int,
    timeout: float,
) -> dict[str, Any]:
    query: dict[str, str | list[str]] = {"api_key": api_key}
    if batch:
        query["batch"] = batch
    if tags:
        query["tag"] = tags

    project_path = urllib.parse.quote(project, safe="")
    url = (
        f"https://api.roboflow.com/dataset/{project_path}/upload?"
        + urllib.parse.urlencode(query, doseq=True)
    )
    fields = {"name": image_path.name, "split": split}
    body, content_type = encode_multipart(image_path, fields)

    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": content_type, "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8", errors="replace")
                return json.loads(payload) if payload else {"success": True}
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code not in RETRYABLE_HTTP_CODES or attempt == retries:
                raise RuntimeError(f"HTTP {error.code}: {detail[:500]}") from None
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == retries:
                reason = getattr(error, "reason", error)
                raise RuntimeError(f"network error: {reason}") from None

        time.sleep(min(2**attempt, 8))

    raise RuntimeError("upload failed")  # Unreachable, for type checkers.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a folder of images to a Roboflow annotation queue.",
    )
    parser.add_argument("folder", type=Path, help="folder containing images")
    parser.add_argument(
        "--project",
        required=True,
        help="Roboflow project ID/slug (the project part of its app.roboflow.com URL)",
    )
    parser.add_argument(
        "--api-key",
        default=ROBOFLOW_API_KEY or os.environ.get("ROBOFLOW_API_KEY"),
        help="private API key; overrides the value in the script/environment",
    )
    parser.add_argument("--batch", help="annotation batch name")
    parser.add_argument(
        "--split", choices=("train", "valid", "test"), default="train"
    )
    parser.add_argument(
        "--tag", action="append", default=[], help="image tag (repeatable)"
    )
    parser.add_argument("--recursive", action="store_true", help="scan subfolders")
    parser.add_argument("--workers", type=int, default=4, help="parallel uploads (default: 4)")
    parser.add_argument("--retries", type=int, default=3, help="retry count (default: 3)")
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="seconds/request (default: 60)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list images without uploading"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = args.folder.expanduser()

    if not folder.is_dir():
        print(f"Error: folder does not exist: {folder}", file=sys.stderr)
        return 2
    if args.workers < 1 or args.retries < 0 or args.timeout <= 0:
        print("Error: workers must be >= 1, retries >= 0, timeout > 0.", file=sys.stderr)
        return 2
    if not args.api_key and not args.dry_run:
        print(
            "Error: set ROBOFLOW_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 2

    images = find_images(folder, args.recursive)
    if not images:
        print(f"No supported images found in {folder}")
        return 0

    duplicate_names = sorted(
        name
        for name in {path.name for path in images}
        if sum(path.name == name for path in images) > 1
    )
    if duplicate_names:
        print(
            "Error: duplicate filenames found (Roboflow upload names must be unique): "
            + ", ".join(duplicate_names),
            file=sys.stderr,
        )
        return 2

    print(f"Found {len(images)} image(s) for project '{args.project}'.")
    if args.dry_run:
        for image in images:
            print(image)
        return 0

    succeeded = 0
    failed: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {
            executor.submit(
                upload_image,
                image,
                args.project,
                args.api_key,
                args.split,
                args.batch,
                args.tag,
                args.retries,
                args.timeout,
            ): image
            for image in images
        }
        for job in as_completed(jobs):
            image = jobs[job]
            try:
                result = job.result()
                if result.get("success", True):
                    succeeded += 1
                    duplicate = " (duplicate)" if result.get("duplicate") else ""
                    print(f"[OK] {image.name}{duplicate}")
                else:
                    failed.append((image, json.dumps(result)))
                    print(f"[FAILED] {image.name}: {result}", file=sys.stderr)
            except Exception as error:
                failed.append((image, str(error)))
                print(f"[FAILED] {image.name}: {error}", file=sys.stderr)

    print(f"Finished: {succeeded} succeeded, {len(failed)} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
