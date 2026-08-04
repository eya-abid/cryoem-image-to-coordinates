#!/usr/bin/env python3
"""Fail when a GitHub candidate contains credentials, local paths, or large files."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
PROHIBITED = {
    "workstation path": re.compile(r"/(?:run/media/guest|home/guest)/"),
    "GitHub token": re.compile(r"gh[oprsu]_[A-Za-z0-9_]{30,}"),
    "Zenodo token assignment": re.compile(r"ZENODO_ACCESS_TOKEN\s*=\s*\S+"),
    "generic bearer token": re.compile(r"Authorization:\s*Bearer\s+\S+", re.IGNORECASE),
}


def main() -> None:
    failures: list[str] = []
    checked = 0
    candidates = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for relative_bytes in candidates:
        if not relative_bytes:
            continue
        path = ROOT / relative_bytes.decode("utf-8")
        if not path.is_file():
            continue
        checked += 1
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            failures.append(f"large file ({size} bytes): {path.relative_to(ROOT)}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PROHIBITED.items():
            if pattern.search(text):
                failures.append(f"{label}: {path.relative_to(ROOT)}")
    print(f"Checked {checked} files")
    if failures:
        print("Public repository audit failed:", file=sys.stderr)
        print("\n".join(f"- {failure}" for failure in failures), file=sys.stderr)
        raise SystemExit(1)
    print("Public repository audit passed")


if __name__ == "__main__":
    main()
