#!/usr/bin/env python3
"""Score converted LCB outputs using the frozen judge subset in a Linux sandbox."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sandbox import run as run_sandbox


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
SOURCE_SETS = {
    "v5": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"),
    "v6-only": ("test6.jsonl",),
    "smoke": (),
}
COUNTS = {"v5": 880, "v6-only": 175, "smoke": 1}
CHUNK_BYTES = 128 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_candidates(path: Path) -> tuple[dict[str, list[str]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("custom outputs must be a nonempty JSON list")
    outputs: dict[str, list[str]] = {}
    depths = set()
    for row in payload:
        if not isinstance(row, dict) or not isinstance(row.get("question_id"), (str, int)):
            raise ValueError("each custom output needs question_id and code_list")
        question_id = str(row["question_id"])
        codes = row.get("code_list")
        if not isinstance(codes, list) or not codes or any(not isinstance(code, str) for code in codes):
            raise ValueError(f"{question_id}: code_list must contain strings")
        if question_id in outputs:
            raise ValueError(f"duplicate custom output question_id: {question_id}")
        outputs[question_id] = codes
        depths.add(len(codes))
    if len(depths) != 1:
        raise ValueError(f"inconsistent samples per question: {sorted(depths)}")
    return outputs, next(iter(depths))


def source_files(dataset: str, source_dir: Path) -> tuple[Path, ...]:
    if dataset == "smoke":
        return (ROOT / "examples/smoke/source.jsonl",)
    return tuple(source_dir / name for name in SOURCE_SETS[dataset])


def iter_chunks(dataset: str, files: tuple[Path, ...], outputs: dict[str, list[str]]):
    seen: set[str] = set()
    rows = 0
    for path in files:
        filename = path.name
        spec = MANIFEST["source_files"].get(filename) if dataset != "smoke" else None
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset file: {path}; run download_data.py first")
        digest = hashlib.sha256()
        file_rows = 0
        chunk: list[tuple[bytes, str]] = []
        size = 0
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                question_id = str(row.get("question_id", ""))
                if not question_id or question_id in seen:
                    raise ValueError(f"{filename}: missing or duplicate question_id {question_id!r}")
                if question_id not in outputs:
                    raise ValueError(f"no generated output for source question {question_id}")
                seen.add(question_id)
                file_rows += 1
                rows += 1
                if chunk and size + len(line) > CHUNK_BYTES:
                    yield chunk
                    chunk, size = [], 0
                chunk.append((line, question_id))
                size += len(line)
        if spec is not None:
            if file_rows != spec["questions"]:
                raise ValueError(f"{filename}: expected {spec['questions']} rows, found {file_rows}")
            if digest.hexdigest() != spec["sha256"]:
                raise ValueError(f"{filename}: frozen source SHA-256 mismatch")
        elif file_rows != 1:
            raise ValueError(f"smoke source must have one row, found {file_rows}")
        if chunk:
            yield chunk
    if rows != COUNTS[dataset]:
        raise ValueError(f"{dataset}: expected {COUNTS[dataset]} questions, found {rows}")
    if seen != set(outputs):
        missing = sorted(set(outputs) - seen)[:5]
        raise ValueError(f"source/candidate question IDs differ; unexpected candidate IDs: {missing}")


def validate_sources(dataset: str, files: tuple[Path, ...], outputs: dict[str, list[str]]) -> None:
    """Validate every frozen source byte and ID before running candidate code."""
    seen: set[str] = set()
    total = 0
    for path in files:
        filename = path.name
        spec = MANIFEST["source_files"].get(filename) if dataset != "smoke" else None
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset file: {path}; run download_data.py first")
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                question_id = str(row.get("question_id", ""))
                if not question_id or question_id in seen:
                    raise ValueError(f"{filename}: missing or duplicate question_id {question_id!r}")
                seen.add(question_id)
                count += 1
                total += 1
        if spec is not None:
            if count != spec["questions"] or digest.hexdigest() != spec["sha256"]:
                raise ValueError(f"{filename}: frozen source row count or SHA-256 mismatch")
        elif count != 1:
            raise ValueError(f"smoke source must have one row, found {count}")
    if total != COUNTS[dataset]:
        raise ValueError(f"{dataset}: expected {COUNTS[dataset]} questions, found {total}")
    if seen != set(outputs):
        missing = sorted(seen - set(outputs))[:5]
        extra = sorted(set(outputs) - seen)[:5]
        raise ValueError(f"source/candidate question IDs differ; missing={missing}, extra={extra}")


def child_runtime() -> tuple[str, str, list[tuple[Path, str]]]:
    prefix = Path(sys.prefix).resolve()
    if prefix == Path("/"):
        raise RuntimeError("refusing to mount the filesystem root as the Python environment")
    executable = Path(sys.executable).resolve()
    mounts: list[tuple[Path, str]] = []
    if not executable.is_relative_to(Path("/usr")):
        if not executable.is_relative_to(prefix):
            mounts.append((executable, str(executable)))
    if not prefix.is_relative_to(Path("/usr")):
        mounts.append((prefix, str(prefix)))
    visible_pythonpath = ["/lcb/vendor"]
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry).resolve()
        if path.is_relative_to(Path("/usr")) or path.is_relative_to(prefix):
            visible_pythonpath.append(str(path))
    return str(executable), ":".join(dict.fromkeys(visible_pythonpath)), mounts


def judge_chunk(
    source_chunk: Path,
    candidate_chunk: Path,
    output_dir: Path,
    workers: int,
    timeout: int,
    wall_timeout: int,
) -> list[dict]:
    output_dir.mkdir(mode=0o755)
    executable, pythonpath, runtime_mounts = child_runtime()
    readonly = [
        [str(source_chunk), "/inputs/source.jsonl"],
        [str(candidate_chunk), "/inputs/candidates.json"],
        [str(ROOT / "judge_chunk.py"), "/lcb/judge_chunk.py"],
        [str(ROOT / "vendor"), "/lcb/vendor"],
        *runtime_mounts,
    ]
    run_sandbox(
        [executable, "/lcb/judge_chunk.py", "--source", "/inputs/source.jsonl", "--candidates", "/inputs/candidates.json", "--output", "/outputs/graded.jsonl", "--workers", str(workers), "--timeout", str(timeout)],
        readonly,
        [(output_dir, "/outputs")],
        pythonpath=pythonpath,
        cwd="/lcb",
        timeout_seconds=wall_timeout,
        allowed_outputs={"graded.jsonl"},
    )
    result = output_dir / "graded.jsonl"
    if not result.is_file():
        raise RuntimeError("judge completed without producing graded.jsonl")
    with result.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(not isinstance(row.get("correctness"), list) for row in rows):
        raise RuntimeError("judge returned malformed verdicts")
    return rows


def curves(correctness: list[list[bool]]) -> list[dict]:
    n = len(correctness[0])
    tasks = len(correctness)
    results = []
    for k in range(1, n + 1):
        pass_values = []
        avg_values = []
        for row in correctness:
            correct = sum(row)
            if correct == 0:
                pass_value = 0.0
            elif n - correct < k:
                pass_value = 1.0
            else:
                pass_value = 1.0 - math.comb(n - correct, k) / math.comb(n, k)
            pass_values.append(pass_value)
            avg_values.append(sum(row[:k]) / k)
        results.append({"k": k, "n": n, "pass_at_k": sum(pass_values) / tasks, "avg_at_k": sum(avg_values) / tasks})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SOURCE_SETS), required=True)
    parser.add_argument("--custom-outputs", type=Path, required=True, help="JSON from raw_to_samples.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/source")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--wall-timeout", type=int, default=43200, help="maximum seconds allowed for one judge chunk")
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1 or args.wall_timeout < 1:
        raise SystemExit("--workers, --timeout, and --wall-timeout must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    outputs, sample_count = read_candidates(args.custom_outputs)
    files = source_files(args.dataset, args.data_dir.resolve())
    validate_sources(args.dataset, files, outputs)
    preflight = subprocess.run([sys.executable, str(ROOT / "preflight.py"), "--backend"], check=False)
    if preflight.returncode:
        raise SystemExit(preflight.returncode)

    grades: dict[str, list[bool]] = {}
    with tempfile.TemporaryDirectory(prefix="lcb-score-") as temporary:
        temp_root = Path(temporary)
        temp_root.chmod(0o755)
        work = temp_root / "chunks"
        work.mkdir()
        for number, chunk in enumerate(iter_chunks(args.dataset, files, outputs)):
            chunk_dir = work / f"chunk-{number:04d}"
            chunk_dir.mkdir(mode=0o755)
            source_path = chunk_dir / "source.jsonl"
            source_path.write_bytes(b"".join(line for line, _ in chunk))
            candidates_path = chunk_dir / "candidates.json"
            candidates_path.write_text(
                json.dumps([{"question_id": question_id, "code_list": outputs[question_id]} for _, question_id in chunk]),
                encoding="utf-8",
            )
            source_path.chmod(0o644)
            candidates_path.chmod(0o644)
            result_rows = judge_chunk(source_path, candidates_path, chunk_dir / "judge-output", args.workers, args.timeout, args.wall_timeout)
            expected_ids = {question_id for _, question_id in chunk}
            if {str(row["question_id"]) for row in result_rows} != expected_ids:
                raise RuntimeError(f"chunk {number}: incomplete or mismatched judge output")
            for row in result_rows:
                question_id = str(row["question_id"])
                correctness = row["correctness"]
                if len(correctness) != sample_count or any(type(value) is not bool for value in correctness):
                    raise RuntimeError(f"{question_id}: invalid/incomplete sample verdicts")
                grades[question_id] = correctness
        if len(grades) != COUNTS[args.dataset]:
            raise RuntimeError(f"incomplete result: {len(grades)} of {COUNTS[args.dataset]} questions")
        ordered_ids = sorted(grades)
        correctness = [grades[question_id] for question_id in ordered_ids]
        curve_rows = curves(correctness)
        final_root = temp_root / "result"
        final_root.mkdir()
        with (final_root / "graded.jsonl").open("w", encoding="utf-8") as stream:
            for question_id in ordered_ids:
                row = grades[question_id]
                stream.write(json.dumps({"question_id": question_id, "correctness": row, "num_correct": sum(row)}) + "\n")
        with (final_root / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("k", "n", "pass_at_k", "avg_at_k"))
            writer.writeheader()
            for row in curve_rows:
                writer.writerow({"n": sample_count, **row})
        metrics = {
            "dataset": f"livecodebench_{args.dataset}" if args.dataset != "smoke" else "smoke",
            "question_count": len(ordered_ids),
            "samples_per_question": sample_count,
            "judge_provenance": MANIFEST["judge"],
            "data_revision": MANIFEST["revision"] if args.dataset != "smoke" else "synthetic smoke input",
            "pass_at_k": {str(row["k"]): row["pass_at_k"] for row in curve_rows},
            "avg_at_k": {str(row["k"]): row["avg_at_k"] for row in curve_rows},
        }
        (final_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        if args.dataset != "smoke":
            (final_root / "source_sha256.json").write_text(
                json.dumps({name: MANIFEST["source_files"][name]["sha256"] for name in SOURCE_SETS[args.dataset]}, indent=2) + "\n",
                encoding="utf-8",
            )
        publish = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.inprogress-", dir=output_dir.parent))
        publish.rmdir()
        try:
            shutil.copytree(final_root, publish)
            os.replace(publish, output_dir)
        except BaseException:
            shutil.rmtree(publish, ignore_errors=True)
            raise
    print(output_dir)


if __name__ == "__main__":
    main()
