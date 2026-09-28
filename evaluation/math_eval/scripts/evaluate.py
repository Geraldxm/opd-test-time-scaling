#!/usr/bin/env python3
"""Generate durable math-evaluation rollouts from one explicit run config."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from inference import (
    create_backend,
    load_prompt,
    preflight_model,
    render_prompt,
    resolved_thinking_request,
    sampling_request,
    validate_decode,
    validate_model,
)
from io_utils import atomic_json, read_jsonl, stable_hash, utc_now
from resume import (
    SCHEMA_VERSION,
    RawShardWriter,
    ensure_manifest,
    missing_indices,
    scan_raw,
    target_key,
)


TOP_LEVEL_FIELDS = {"dataset", "model", "prompt", "decode", "output"}
DATASET_FIELDS = {"name", "path", "limit"}
OUTPUT_FIELDS = {"root", "shard_size", "compression", "fsync_every"}


def _exact_fields(value: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{location} has unknown fields: {sorted(unknown)}")


def _reject_nulls(value: dict[str, Any], location: str) -> None:
    nulls = {key for key, item in value.items() if item is None}
    if nulls:
        raise ValueError(f"{location} fields cannot be null: {sorted(nulls)}")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("run config must be an object")
    _exact_fields(config, TOP_LEVEL_FIELDS, "config")
    missing = TOP_LEVEL_FIELDS - config.keys()
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    dataset = config["dataset"]
    output = config["output"]
    if not isinstance(dataset, dict) or not isinstance(output, dict):
        raise ValueError("dataset and output must be objects")
    _exact_fields(dataset, DATASET_FIELDS, "dataset")
    _exact_fields(output, OUTPUT_FIELDS, "output")
    _reject_nulls(dataset, "dataset")
    _reject_nulls(output, "output")
    for key in ("name", "path"):
        if not dataset.get(key):
            raise ValueError(f"dataset requires {key}")
    if dataset.get("limit") is not None and int(dataset["limit"]) < 1:
        raise ValueError("dataset.limit must be >= 1")
    config["model"] = validate_model(config["model"])
    if not isinstance(config["prompt"], str) or not config["prompt"].strip():
        raise ValueError("prompt must be a non-empty path string")
    load_prompt(config["prompt"], config["model"]["input_mode"])
    if not output.get("root"):
        raise ValueError("output requires root")
    config["decode"] = validate_decode(config["decode"])
    if output.get("compression", "none") not in {"none", "gzip"}:
        raise ValueError("output.compression must be none or gzip")
    return config


def load_dataset(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(config["path"]).expanduser()
    rows = []
    ids = set()
    limit = config.get("limit")
    for line_number, row in enumerate(read_jsonl(path), 1):
        missing = {"id", "problem", "answer"} - row.keys()
        if missing:
            raise ValueError(f"{path}:{line_number}: missing canonical fields {sorted(missing)}")
        problem_id = str(row["id"])
        if problem_id in ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {problem_id!r}")
        if not str(row["problem"]).strip() or not str(row["answer"]).strip():
            raise ValueError(f"{path}:{line_number}: problem and answer must be non-empty")
        ids.add(problem_id)
        rows.append(row)
        if limit is not None and len(rows) >= int(limit):
            break
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for package in ("vllm", "torch", "transformers", "math-verify"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    snapshot = {"python": platform.python_version(), "packages": packages}
    try:
        import torch

        snapshot["cuda_runtime"] = torch.version.cuda
        snapshot["gpu_names"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except Exception:
        snapshot["cuda_runtime"] = None
        snapshot["gpu_names"] = []
    return snapshot


def code_revision() -> dict[str, Any] | None:
    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": dirty}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def raw_row(
    run_id: str,
    dataset_config: dict[str, Any],
    model: dict[str, Any],
    decode: dict[str, Any],
    source: dict[str, Any],
    sample_idx: int,
    result,
) -> dict[str, Any]:
    dataset = str(dataset_config["name"])
    problem_id = str(source["id"])
    problem_uid = f"{dataset}:{problem_id}"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model": str(model["name"]),
        "model_path": model.get("path"),
        "backend": model["backend"],
        "input_mode": model["input_mode"],
        "thinking": model.get("thinking") if "thinking" in model else None,
        "dataset": dataset,
        "problem_id": problem_id,
        "problem_uid": problem_uid,
        "problem": str(source["problem"]),
        "gold_answer": str(source["answer"]),
        "gold_hash": stable_hash(str(source["answer"])),
        "sample_idx": sample_idx,
        "decode": decode,
        **result.as_dict(),
        "truncated": result.finish_reason == "length",
        "created_at": utc_now(),
    }


def _expected_keys(
    rows: list[dict[str, Any]], model: str, dataset: str, samples_per_problem: int
) -> set[tuple[str, str, str, int]]:
    return {
        target_key(model, dataset, f"{dataset}:{source['id']}", sample_idx)
        for source in rows
        for sample_idx in range(samples_per_problem)
    }


def run(
    config_path: Path,
    run_id: str,
    resume: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
    sample_band_size: int = 64,
    request_batch_size_override: int | None = None,
    fsync_every_override: int | None = None,
) -> Path:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    if sample_band_size < 1:
        raise ValueError("sample_band_size must be >= 1")
    if request_batch_size_override is not None and request_batch_size_override < 1:
        raise ValueError("request_batch_size_override must be >= 1")
    if fsync_every_override is not None and fsync_every_override < 1:
        raise ValueError("fsync_every_override must be >= 1")
    config_text = config_path.read_text(encoding="utf-8")
    config = load_config(config_path)
    rows = load_dataset(config["dataset"])
    dataset_sha256 = stable_hash(rows)
    partition_rows = rows[shard_index::shard_count]
    prompt = load_prompt(config["prompt"], config["model"]["input_mode"])
    prompt_sha256 = stable_hash(prompt)
    thinking_preflight = preflight_model(config["model"])
    output = config["output"]
    fsync_every = (
        fsync_every_override
        if fsync_every_override is not None
        else int(output.get("fsync_every", 1))
    )
    physical_run_id = (
        run_id
        if shard_count == 1
        else f"{run_id}.part-{shard_index:05d}-of-{shard_count:05d}"
    )
    run_dir = Path(output["root"]).expanduser() / physical_run_id
    generation_spec = {
        "dataset": config["dataset"],
        "dataset_sha256": dataset_sha256,
        "model": config["model"],
        "prompt_sha256": prompt_sha256,
        "decode": {
            key: value
            for key, value in config["decode"].items()
            if key != "samples_per_problem"
        },
        "output": {
            "shard_size": int(output.get("shard_size", 100)),
            "compression": output.get("compression", "none"),
            "fsync_every": int(output.get("fsync_every", 1)),
        },
        "request_template": {
            "input_mode": config["model"]["input_mode"],
            "thinking": (
                config["model"].get("thinking")
                if "thinking" in config["model"]
                else None
            ),
            "resolved_thinking": resolved_thinking_request(config["model"]),
            "thinking_preflight": thinking_preflight,
            "sampling": sampling_request(config["decode"]),
        },
    }
    model_name = str(config["model"]["name"])
    dataset_name = str(config["dataset"]["name"])
    samples_per_problem = int(config["decode"]["samples_per_problem"])
    manifest_path = run_dir / "manifests/run.json"
    if resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot_path = run_dir / "manifests/config.yaml"
        snapshot = (
            yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
            if snapshot_path.exists()
            else None
        )
        if isinstance(snapshot, dict) and isinstance(snapshot.get("decode"), dict):
            legacy_spec = dict(generation_spec)
            legacy_spec["decode"] = dict(config["decode"])
            legacy_spec["decode"]["samples_per_problem"] = int(
                snapshot["decode"]["samples_per_problem"]
            )
            if existing.get("generation_spec_hash") == stable_hash(legacy_spec):
                generation_spec = legacy_spec
    expected_keys = _expected_keys(
        partition_rows, model_name, dataset_name, samples_per_problem
    )
    global_expected_keys = _expected_keys(
        rows, model_name, dataset_name, samples_per_problem
    )
    partition_manifest = (
        {
            "partition": {"index": shard_index, "count": shard_count},
            "expected_sample_count": len(expected_keys),
            "global_expected_sample_count": len(global_expected_keys),
            "expected_sample_keys_sha256": stable_hash(sorted(expected_keys)),
            "global_expected_sample_keys_sha256": stable_hash(
                sorted(global_expected_keys)
            ),
        }
        if shard_count > 1
        else {}
    )
    manifest = ensure_manifest(
        run_dir,
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "model": str(config["model"]["name"]),
            "backend": config["model"]["backend"],
            "dataset": str(config["dataset"]["name"]),
            "status": "running",
            "sample_count": 0,
            "dataset_sha256": dataset_sha256,
            "code_revision": code_revision(),
            "environment": environment_snapshot(),
            "thinking_preflight": thinking_preflight,
            **partition_manifest,
        },
        generation_spec,
        config_text,
        prompt,
        resume,
    )
    operational_overrides = {
        key: value
        for key, value in {
            "request_batch_size": request_batch_size_override,
            "fsync_every": fsync_every_override,
        }.items()
        if value is not None
    }
    if operational_overrides:
        manifest.setdefault("operational_override_history", []).append(
            {"started_at": utc_now(), **operational_overrides}
        )
    completed = scan_raw(run_dir / "raw", recover_tail=True)
    recorded_raw_hash = manifest.get("raw_content_sha256")
    if recorded_raw_hash is not None:
        current_raw_hash = stable_hash(
            [completed[key].row for key in sorted(completed)]
        )
        if current_raw_hash != recorded_raw_hash:
            raise RuntimeError(f"{run_dir}: raw content changed since completion")
    problem_uids = {f"{dataset_name}:{source['id']}" for source in partition_rows}
    existing_indices = [
        key[3]
        for key in completed
        if key[0] == model_name
        and key[1] == dataset_name
        and key[2] in problem_uids
    ]
    if existing_indices and max(existing_indices) >= samples_per_problem:
        raise ValueError(
            f"target samples_per_problem={samples_per_problem} is below existing "
            f"sample_idx={max(existing_indices)}; use replay --sample-limit instead"
        )
    work = []
    for source in partition_rows:
        uid = f"{dataset_name}:{source['id']}"
        for sample_idx in missing_indices(
            completed,
            model_name,
            dataset_name,
            uid,
            samples_per_problem,
        ):
            work.append((source, sample_idx))
    work.sort(key=lambda item: item[1] // sample_band_size)

    raw_dir = run_dir / "raw" / safe_name(model_name) / safe_name(dataset_name)
    indices_by_problem = {
        f"{dataset_name}:{source['id']}": {
            key[3] for key in completed if key[2] == f"{dataset_name}:{source['id']}"
        }
        for source in partition_rows
    }
    depths = {}
    for uid, indices in indices_by_problem.items():
        depth = 0
        while depth in indices:
            depth += 1
        depths[uid] = depth
    sample_count = len(completed)

    def update_progress(status: str) -> None:
        manifest.update({
            **partition_manifest,
            "status": status,
            "sample_count": sample_count,
            "target_samples_per_problem": samples_per_problem,
            "target_sample_count": len(expected_keys),
            "common_sample_depth": min(depths.values(), default=0),
            "sample_band_size": sample_band_size,
            "updated_at": utc_now(),
        })
        if status == "running":
            manifest["process_id"] = os.getpid()
        else:
            manifest.pop("process_id", None)
        atomic_json(manifest_path, manifest)

    for key in (
        "completed_at",
        "stopped_at",
        "failed_at",
        "raw_content_sha256",
        "evaluation_status",
        "evaluated_at",
        "parser_id",
        "parser_config_hash",
        "parsed_rows",
        "parsed_path",
        "metrics_path",
        "sample_limit",
    ):
        manifest.pop(key, None)
    update_progress("running")
    stop_requested = False
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print(
            "Stop requested; finishing the current backend batch and sealing raw data.",
            file=sys.stderr,
            flush=True,
        )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(
        signal.SIGINT,
        signal.SIG_IGN if config["model"]["backend"] == "vllm" else request_stop,
    )
    if work:
        try:
            backend = create_backend(config["model"], thinking_preflight)
            if request_batch_size_override is not None:
                if not hasattr(backend, "request_batch_size"):
                    raise ValueError(
                        "request_batch_size_override requires a batched backend"
                    )
                backend.request_batch_size = request_batch_size_override
            request_batch_size = getattr(backend, "request_batch_size", 1)
            with RawShardWriter(
                raw_dir,
                shard_size=int(output.get("shard_size", 100)),
                compression=output.get("compression", "none"),
                fsync_every=fsync_every,
            ) as writer:
                for offset in range(0, len(work), request_batch_size):
                    if stop_requested:
                        break
                    batch = work[offset : offset + request_batch_size]
                    requests = [
                        {
                            "prompt": render_prompt(prompt, str(source["problem"])),
                            "decode": config["decode"],
                            "sample_idx": sample_idx,
                        }
                        for source, sample_idx in batch
                    ]
                    if request_batch_size > 1:
                        results = backend.generate_batch(requests)
                    else:
                        results = [backend.generate(**requests[0])]
                    for (source, sample_idx), result in zip(
                        batch, results, strict=True
                    ):
                        writer.append(
                            raw_row(
                                run_id,
                                config["dataset"],
                                config["model"],
                                config["decode"],
                                source,
                                sample_idx,
                                result,
                            )
                        )
                        uid = f"{dataset_name}:{source['id']}"
                        indices_by_problem[uid].add(sample_idx)
                        sample_count += 1
                        while depths[uid] in indices_by_problem[uid]:
                            depths[uid] += 1
                    update_progress("running")
                    if stop_requested:
                        break
        except BaseException:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
            manifest.update({"status": "failed", "failed_at": utc_now()})
            manifest.pop("process_id", None)
            atomic_json(manifest_path, manifest)
            raise

    manifest.update({"status": "finalizing", "updated_at": utc_now()})
    manifest.pop("process_id", None)
    atomic_json(manifest_path, manifest)
    signal.signal(signal.SIGINT, previous_sigint)
    signal.signal(signal.SIGTERM, previous_sigterm)

    # Seal a fully written active shard even when resume found no missing work.
    RawShardWriter(
        raw_dir,
        shard_size=int(output.get("shard_size", 100)),
        compression=output.get("compression", "none"),
        fsync_every=fsync_every,
    ).close()
    if list(raw_dir.glob("part-*.jsonl.inprogress")):
        raise RuntimeError(f"{run_dir}: unsealed raw shard remains")
    completed = scan_raw(run_dir / "raw")
    actual_keys = set(completed)
    if stop_requested:
        manifest["stopped_at"] = utc_now()
        update_progress("stopped")
        return run_dir
    invalid_keys = {
        key
        for key in actual_keys
        if key[0] != model_name
        or key[1] != dataset_name
        or key[2] not in problem_uids
        or key[3] < 0
    }
    if actual_keys != expected_keys or invalid_keys:
        missing = len(expected_keys - actual_keys)
        unexpected = len(actual_keys - expected_keys)
        raise RuntimeError(
            f"{run_dir}: raw sample keys do not match partition "
            f"(missing={missing}, unexpected={unexpected}, invalid={len(invalid_keys)})"
        )
    sample_count = len(actual_keys)
    raw_content_sha256 = stable_hash(
        [completed[key].row for key in sorted(completed)]
    )
    if manifest.get("raw_content_sha256") not in (None, raw_content_sha256):
        raise RuntimeError(f"{run_dir}: raw content changed since completion")

    manifest.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "sample_count": sample_count,
            "raw_content_sha256": raw_content_sha256,
        }
    )
    manifest.pop("process_id", None)
    atomic_json(run_dir / "manifests/run.json", manifest)
    return run_dir


def main() -> int:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config", type=Path, required=True)
    argument_parser.add_argument("--run-id", required=True)
    argument_parser.add_argument("--resume", action="store_true")
    argument_parser.add_argument("--shard-index", type=int, default=0)
    argument_parser.add_argument("--shard-count", type=int, default=1)
    argument_parser.add_argument("--sample-band-size", type=int, default=64)
    argument_parser.add_argument("--request-batch-size-override", type=int)
    argument_parser.add_argument("--fsync-every-override", type=int)
    args = argument_parser.parse_args()
    run_dir = run(
        args.config,
        args.run_id,
        args.resume,
        args.shard_index,
        args.shard_count,
        args.sample_band_size,
        args.request_batch_size_override,
        args.fsync_every_override,
    )
    manifest = json.loads((run_dir / "manifests/run.json").read_text(encoding="utf-8"))
    status = manifest["status"]
    print(json.dumps({"run_dir": str(run_dir), "status": status}, ensure_ascii=False))
    return 130 if status == "stopped" else 0


if __name__ == "__main__":
    raise SystemExit(main())
