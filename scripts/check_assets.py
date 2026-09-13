#!/usr/bin/env python3
"""Verify the original inference checkpoint against the exported hash manifest."""

import argparse
import json
from pathlib import Path

from common import ROOT, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    args = p.parse_args()
    manifest = json.loads((ROOT / "provenance/checkpoint_assets.json").read_text())
    for row in manifest["files"]:
        path = args.checkpoint / row["name"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise SystemExit(f"Missing or mismatched asset: {path}")
        print(f"OK {row['name']}")


if __name__ == "__main__":
    main()
