#!/usr/bin/env python3
"""Verify exported source bytes independently of installed ML dependencies."""

from pathlib import Path

from common import ROOT, sha256


def main():
    count = 0
    for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"Checksum mismatch: {relative}")
        count += 1
    print(f"Verified {count} files.")


if __name__ == "__main__":
    main()
