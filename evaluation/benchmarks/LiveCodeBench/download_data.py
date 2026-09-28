#!/usr/bin/env python3
"""Fetch and verify the frozen LCB v5/v6 source files from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
REVISION = MANIFEST["revision"]
SOURCE = ROOT / "data/source"
DATASETS = {
    "v5": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"),
    "v6-only": ("test6.jsonl",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, filename: str) -> None:
    expected = MANIFEST["source_files"][filename]["sha256"]
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 mismatch (expected {expected}, got {actual})")


def fetch(filename: str) -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    target = SOURCE / filename
    if target.exists():
        verify(target, filename)
        print(f"verified existing {target}")
        return

    url = (
        "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/"
        f"{REVISION}/{filename}?download=true"
    )
    partial = target.with_suffix(target.suffix + ".partial")
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "opd-reproducibility/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
                digest.update(block)
            output.flush()
            os.fsync(output.fileno())
        expected = MANIFEST["source_files"][filename]["sha256"]
        if digest.hexdigest() != expected:
            raise ValueError(f"downloaded {filename}: SHA-256 mismatch")
        partial.replace(target)
        print(f"downloaded and verified {target}")
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("v5", "v6-only", "all"), default="all")
    parser.add_argument("--check", action="store_true", help="verify already downloaded files only")
    args = parser.parse_args()
    selected = DATASETS[args.dataset] if args.dataset != "all" else tuple(
        filename for files in DATASETS.values() for filename in files
    )
    for filename in selected:
        path = SOURCE / filename
        if args.check:
            if not path.is_file():
                raise FileNotFoundError(f"missing source file: {path}")
            verify(path, filename)
            print(f"verified {path}")
        else:
            fetch(filename)


if __name__ == "__main__":
    main()
