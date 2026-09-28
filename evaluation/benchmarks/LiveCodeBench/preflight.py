#!/usr/bin/env python3
"""Check judge dependencies and the isolated Linux execution backend."""

import argparse
import shutil

import numpy  # noqa: F401
import tqdm  # noqa: F401

from sandbox import self_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="store_true", help="run the sandbox security probe")
    args = parser.parse_args()
    required = ("setpriv", "unshare", "mount", "chroot", "prlimit")
    missing = [name for name in required if not shutil.which(name)]
    if missing:
        raise SystemExit(f"missing required Linux tools: {', '.join(missing)}")
    if not shutil.which("capsh"):
        raise SystemExit("missing capsh (install libcap2-bin or the distribution equivalent)")
    if args.backend:
        self_test()
    print("LiveCodeBench judge preflight: ok")


if __name__ == "__main__":
    main()
