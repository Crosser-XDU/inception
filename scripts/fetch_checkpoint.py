#!/usr/bin/env python3
"""Explicitly fetch the eight known inference assets; no base model or optimizer."""

import argparse
import json
from pathlib import Path
import re
import subprocess

from common import ROOT, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="117_jump_vpn", help="An SSH alias on your own machine.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", args.host):
        p.error("Invalid SSH alias.")
    manifest = json.loads((ROOT / "provenance/checkpoint_assets.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for row in manifest["files"]:
        target = args.output / row["name"]
        if target.exists():
            if sha256(target) != row["sha256"]:
                raise SystemExit(f"Existing mismatched asset, not overwritten: {target}")
            print(f"Verified existing {target.name}")
            continue
        temporary = target.with_suffix(target.suffix + ".partial")
        if temporary.exists():
            raise SystemExit(f"Inspect/remove this incomplete download explicitly: {temporary}")
        subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                        f"{args.host}:{manifest['checkpoint']}/{row['name']}", str(temporary)], check=True)
        if temporary.stat().st_size != row["bytes"] or sha256(temporary) != row["sha256"]:
            raise SystemExit(f"Download failed checksum: {temporary}")
        temporary.replace(target)
        print(f"Verified {target.name}")


if __name__ == "__main__":
    main()
