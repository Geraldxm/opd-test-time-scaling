#!/usr/bin/env python3
"""Convert math-eval raw rows, run the frozen EvalPlus judge, and aggregate Pass@K."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path

from raw_to_samples import convert
from sandbox import run as run_sandboxed

ROOT = Path(__file__).resolve().parent
CANONICAL = ROOT / "data/canonical/humaneval_plus.jsonl"
SOURCE = ROOT / "data/source/HumanEvalPlus.jsonl"
DATASET = "humaneval"
DISPLAY_NAME = "HumanEval+"
GOPD_SOURCE_SNAPSHOT_COMMIT = "37371a4c31ad7947746200d234161769191f4748"
METADATA = ROOT / "data/canonical/humaneval_plus.metadata.json"
PINNED_SOURCE_SHA256 = "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"
PINNED_CANONICAL_SHA256 = "7fee5dcd0f1f438df47d4eab083c5b0d4ba4c68881e795f7455a65eb2a80b71b"
VENDOR = ROOT / "vendor/evalplus"
VENDOR_TREE_SHA256 = "e1a5530bae474eade8245f1588901938dd0a06354d42b38c2ed3d6f098480e5d"
JUDGE_MODULE_SHA256 = "2ba7560f74466c500b95fe995b14c509aa75f6c452df10d2583059878d60a07f"
LOCAL_TIMEOUT_DIFF_SHA256 = "4bba455ec37fd2a579a5e7e9347d34341208701ac095908162c1afbb66719e4d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_frozen_vendor() -> None:
    manifest = VENDOR / "SHA256SUMS"
    tree = hashlib.sha256()
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        relative_path = Path(relative)
        path = VENDOR / relative_path
        if relative_path.is_absolute() or ".." in relative_path.parts or path.is_symlink() or not path.is_file():
            raise ValueError("EvalPlus vendor manifest contains an invalid path")
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"EvalPlus vendored file hash mismatch: {relative}")
        entries.append((relative, digest))
    if not entries or len({relative for relative, _ in entries}) != len(entries):
        raise ValueError("EvalPlus vendor manifest is empty or contains duplicate paths")
    expected_files = {relative for relative, _ in entries}
    actual_files = {
        path.relative_to(VENDOR).as_posix()
        for path in VENDOR.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual_files != expected_files or any(path.is_symlink() for path in VENDOR.rglob("*")):
        raise ValueError("EvalPlus vendor file set differs from its frozen manifest")
    for relative, digest in sorted(entries):
        tree.update(relative.encode("utf-8")); tree.update(b"\0")
        tree.update(bytes.fromhex(digest)); tree.update(b"\n")
    if tree.hexdigest() != VENDOR_TREE_SHA256:
        raise ValueError("EvalPlus vendor source-tree SHA-256 mismatch")
    module = VENDOR / "evalplus/eval/__init__.py"
    if sha256_file(module) != JUDGE_MODULE_SHA256:
        raise ValueError("EvalPlus judge module SHA-256 mismatch")
    patch = ROOT / "VENDORED_TIMEOUT_PATCH.diff"
    if sha256_file(patch) != LOCAL_TIMEOUT_DIFF_SHA256:
        raise ValueError("documented local EvalPlus timeout patch SHA-256 mismatch")


def verify_frozen_data() -> None:
    """Check metadata and both byte snapshots against scorer-pinned digests."""
    verify_frozen_vendor()
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    expected_metadata = {
        "dataset": "humaneval_plus",
        "source_path": "data/source/HumanEvalPlus.jsonl",
        "source_sha256": PINNED_SOURCE_SHA256,
        "canonical_path": "data/canonical/humaneval_plus.jsonl",
        "canonical_sha256": PINNED_CANONICAL_SHA256,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"frozen metadata {key} does not match the scorer's pinned value")
    if sha256_file(SOURCE) != PINNED_SOURCE_SHA256:
        raise ValueError("frozen EvalPlus source SHA-256 mismatch")
    if sha256_file(CANONICAL) != PINNED_CANONICAL_SHA256:
        raise ValueError("frozen canonical problem SHA-256 mismatch")
    source_rows, canonical_rows = read_jsonl(SOURCE), read_jsonl(CANONICAL)
    source_ids = [str(row["task_id"]) for row in source_rows]
    canonical_ids = [str(row["id"]) for row in canonical_rows]
    if len(source_ids) != metadata.get("source_row_count") or len(canonical_ids) != metadata.get("canonical_row_count"):
        raise ValueError("frozen dataset row count does not match metadata")
    if len(set(source_ids)) != len(source_ids) or source_ids != canonical_ids:
        raise ValueError("source and canonical problem IDs/order differ")


def expected_ids() -> set[str]:
    verify_frozen_data()
    return {str(row["id"]) for row in read_jsonl(CANONICAL)}


def pass_at_k(n: int, correct: int, k: int) -> float:
    if not 1 <= k <= n or not 0 <= correct <= n:
        raise ValueError(f"invalid pass@K counts: n={n}, correct={correct}, k={k}")
    if n - correct < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / index for index in range(n - correct + 1, n + 1))


def default_ks(depth: int) -> list[int]:
    values = []
    k = 1
    while k <= depth:
        values.append(k)
        k *= 2
    if not values or values[-1] != depth:
        values.append(depth)
    return values


def parse_ks(value: str | None, depth: int) -> list[int]:
    if value is None:
        return default_ks(depth)
    try:
        ks = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--ks must be a comma-separated list of positive integers") from exc
    if not ks or len(ks) != len(set(ks)) or any(k < 1 or k > depth for k in ks):
        raise ValueError(f"--ks must contain unique values in [1, {depth}]")
    return sorted(ks)


def ensure_empty_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"output directory must be empty: {path}")


def site_packages() -> list[Path]:
    candidates = [sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    values = []
    for value in candidates:
        path = Path(value).resolve()
        if path.is_dir() and path not in values:
            values.append(path)
    return values


def run_evalplus(samples: Path, metrics: Path, parallel: int | None, wall_timeout: int) -> None:
    python = Path(sys.executable).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    if base_prefix == Path("/"):
        raise RuntimeError("refusing to expose the filesystem root as the Python runtime")
    sites = site_packages()
    readonly = [[str(base_prefix), str(base_prefix)]]
    readonly.extend([[str(path), str(path)] for path in sites if not path.is_relative_to(base_prefix)])
    readonly.extend(
        [
            [str(VENDOR), "/vendor"],
            [str(SOURCE), "/inputs/evalplus.jsonl"],
            [str(samples), "/inputs/samples.jsonl"],
        ]
    )
    if metrics.exists() or metrics.is_symlink():
        raise FileExistsError(f"refusing to overwrite EvalPlus results: {metrics}")
    command = [
        str(python),
        "/vendor/evalplus/evaluate.py",
        "--dataset",
        DATASET,
        "--samples",
        "/inputs/samples.jsonl",
        "--output-file",
        "/outputs/eval_results.json",
    ]
    if parallel is not None:
        command.extend(["--parallel", str(parallel)])
    pythonpath = os.pathsep.join(["/vendor", *(str(path) for path in sites)])
    metrics.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="evalplus-output-") as scratch:
        scratch_path = Path(scratch)
        run_sandboxed(
            command,
            readonly,
            [[str(scratch_path), "/outputs"]],
            pythonpath=pythonpath,
            env={
                "HUMANEVAL_OVERRIDE_PATH": "/inputs/evalplus.jsonl",
                "EVALPLUS_CACHE_DIR": "/tmp/evalplus-cache",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
            timeout_seconds=wall_timeout,
            allowed_outputs={"eval_results.json"},
        )
        entries = list(scratch_path.iterdir())
        if len(entries) != 1 or entries[0].name != "eval_results.json" or not entries[0].is_file() or entries[0].is_symlink():
            raise RuntimeError("EvalPlus sandbox must produce exactly one regular eval_results.json")
        if metrics.exists() or metrics.is_symlink():
            raise FileExistsError(f"refusing to overwrite EvalPlus results: {metrics}")
        with entries[0].open("rb") as source, metrics.open("xb") as destination:
            shutil.copyfileobj(source, destination)


def aggregate(metrics_path: Path, output_dir: Path, ks_arg: str | None) -> dict:
    verify_frozen_data()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_hash = hashlib.md5(SOURCE.read_bytes()).hexdigest()
    if metrics.get("hash") != source_hash:
        raise ValueError("EvalPlus result dataset hash does not match the bundled snapshot")
    rows = metrics.get("eval")
    if not isinstance(rows, dict) or set(rows) != expected_ids():
        raise ValueError("EvalPlus result task IDs do not match the bundled canonical data")

    counts = {}
    ordered_verdicts = {}
    depths = set()
    for task_id, verdicts in rows.items():
        if not isinstance(verdicts, list) or not verdicts:
            raise ValueError(f"{task_id}: no per-sample verdicts")
        correct = []
        for verdict in verdicts:
            valid_statuses = {"pass", "fail", "timeout"}
            if verdict.get("base_status") not in valid_statuses or verdict.get("plus_status") not in valid_statuses:
                raise ValueError(f"{task_id}: result is missing base/plus status")
            correct.append(verdict["base_status"] == "pass" and verdict["plus_status"] == "pass")
        ordered_verdicts[task_id] = correct
        counts[task_id] = sum(correct)
        depths.add(len(verdicts))
    if len(depths) != 1:
        raise ValueError(f"per-problem sample counts differ: {sorted(depths)}")
    depth = next(iter(depths))
    ks = parse_ks(ks_arg, depth)
    summary = {
        "benchmark": DISPLAY_NAME,
        "num_problems": len(counts),
        "samples_per_problem": depth,
        "source_sha256": PINNED_SOURCE_SHA256,
        "canonical_sha256": PINNED_CANONICAL_SHA256,
        "gopd_source_snapshot_commit": GOPD_SOURCE_SNAPSHOT_COMMIT,
        "evalplus_vendor_tree_sha256": VENDOR_TREE_SHA256,
        "evalplus_judge_module_sha256": JUDGE_MODULE_SHA256,
        "local_timeout_diff_sha256": LOCAL_TIMEOUT_DIFF_SHA256,
        "correct_per_problem": counts,
        "pass_at_k": {
            str(k): sum(pass_at_k(depth, correct, k) for correct in counts.values()) / len(counts)
            for k in ks
        },
        "avg_at_k": {
            str(k): sum(sum(ordered_verdicts[task_id][:k]) / k for task_id in ordered_verdicts) / len(ordered_verdicts)
            for k in ks
        },
        "sample_order_contract": "EvalPlus serializes per-task verdicts by completion_id; converter sorts input samples by sample_idx before judging.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("benchmark", "num_problems", "samples_per_problem", "k", "pass_at_k", "avg_at_k"),
        )
        writer.writeheader()
        for k, value in summary["pass_at_k"].items():
            writer.writerow(
                {
                    "benchmark": DISPLAY_NAME,
                    "num_problems": len(counts),
                    "samples_per_problem": depth,
                    "k": k,
                    "pass_at_k": format(value, ".17g"),
                    "avg_at_k": format(summary["avg_at_k"][k], ".17g"),
                }
            )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Score {DISPLAY_NAME} samples.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw", type=Path, help="math-eval raw JSONL, raw directory, or completed run directory")
    source.add_argument("--eval-results", type=Path, help="reuse an EvalPlus eval_results.json and only recalculate Pass@K")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ks", help="comma-separated K values; default is powers of two up to the sample depth")
    parser.add_argument("--parallel", type=int, default=4, help="EvalPlus worker count (default: 4)")
    parser.add_argument(
        "--wall-timeout", type=int, default=172800,
        help="whole EvalPlus judge wall-clock limit in seconds (default: 172800 / 48 hours)",
    )
    args = parser.parse_args()
    if args.parallel is not None and args.parallel < 1:
        parser.error("--parallel must be at least 1")
    if args.wall_timeout < 1:
        parser.error("--wall-timeout must be at least 1")
    verify_frozen_data()
    output_dir = args.out_dir.resolve()
    ensure_empty_output(output_dir)
    if args.raw is not None:
        samples = output_dir / "samples.jsonl"
        count = convert(args.raw.resolve(), samples)
        if count < 1:
            raise ValueError("raw input contains no samples")
        run_evalplus(samples, output_dir / "eval_results.json", args.parallel, args.wall_timeout)
        metrics_path = output_dir / "eval_results.json"
    else:
        metrics_path = args.eval_results.resolve()
    summary = aggregate(metrics_path, output_dir, args.ks)
    print(json.dumps({key: summary[key] for key in ("benchmark", "num_problems", "samples_per_problem", "pass_at_k", "avg_at_k")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
