"""Create or verify MANIFEST_SHA256.txt for regular files in this package."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_SHA256.txt"


def hashes() -> dict[str, str]:
    files = [
        p for p in ROOT.rglob("*")
        if p.is_file()
        and p != MANIFEST
        and "__pycache__" not in p.parts
        and not {"data", "download"}.intersection(p.relative_to(ROOT).parts)
    ]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = "".join(f"{digest}  {name}\n" for name, digest in hashes().items())
    if args.check:
        if not MANIFEST.exists() or MANIFEST.read_text(encoding="utf-8") != expected:
            raise SystemExit("Manifest does not match package files; regenerate with python code/make_manifest.py")
        print(f"PASS: {len(hashes())} files match MANIFEST_SHA256.txt")
    else:
        MANIFEST.write_text(expected, encoding="utf-8")
        print(f"Wrote {len(hashes())} entries to MANIFEST_SHA256.txt")


if __name__ == "__main__":
    main()
