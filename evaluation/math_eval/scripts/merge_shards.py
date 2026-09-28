#!/usr/bin/env python3
"""Validate and merge independently generated data partitions."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from io_utils import atomic_json, read_jsonl, stable_hash, utc_now
from resume import sample_key


COMMON_FIELDS = (
    "run_id",
    "schema_version",
    "config_sha256",
    "prompt_sha256",
    "dataset_sha256",
    "generation_spec_hash",
    "code_revision",
    "global_expected_sample_count",
    "global_expected_sample_keys_sha256",
)
RAW_PATTERNS = ("**/part-*.jsonl", "**/part-*.jsonl.gz")


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifests/run.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing shard manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest must be an object")
    return manifest


def _verify_snapshot(run_dir: Path, manifest: dict[str, Any]) -> None:
    config_path = run_dir / "manifests/config.yaml"
    prompt_path = run_dir / "manifests/prompt.json"
    if stable_hash(config_path.read_text(encoding="utf-8")) != manifest["config_sha256"]:
        raise ValueError(f"{run_dir}: config snapshot hash mismatch")
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    if stable_hash(prompt) != manifest["prompt_sha256"]:
        raise ValueError(f"{run_dir}: prompt snapshot hash mismatch")


def _environment_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    environment = manifest.get("environment")
    if not isinstance(environment, dict) or not isinstance(
        environment.get("packages"), dict
    ):
        raise ValueError("shard manifest has invalid environment")
    return {"python": environment.get("python"), "packages": environment["packages"]}


def _raw_rows(
    raw_dir: Path, manifest: dict[str, Any]
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    active = list(raw_dir.glob("**/*.jsonl.inprogress"))
    if active:
        raise ValueError(f"{raw_dir}: unsealed raw shards remain")
    rows: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    sources: dict[tuple[str, str, str, int], Path] = {}
    raw_paths = sorted(
        {path for pattern in RAW_PATTERNS for path in raw_dir.glob(pattern)}
    )
    for raw_path in raw_paths:
        for row in read_jsonl(raw_path):
            key = sample_key(row)
            if str(row.get("run_id")) != str(manifest["run_id"]):
                raise ValueError(f"{raw_path}: raw run_id does not match manifest")
            if row.get("schema_version") != manifest["schema_version"]:
                raise ValueError(f"{raw_path}: raw schema does not match manifest")
            if key in rows:
                raise ValueError(f"duplicate sample key {key}: {sources[key]} and {raw_path}")
            rows[key] = row
            sources[key] = raw_path
    return rows


def _raw_content_sha256(
    rows: dict[tuple[str, str, str, int], dict[str, Any]],
) -> str:
    return stable_hash([rows[key] for key in sorted(rows)])


def _common_sample_depth(
    rows: dict[tuple[str, str, str, int], dict[str, Any]],
) -> int:
    indices_by_problem: dict[str, set[int]] = {}
    for _model, _dataset, problem_uid, sample_idx in rows:
        indices_by_problem.setdefault(problem_uid, set()).add(sample_idx)
    depths = []
    for indices in indices_by_problem.values():
        depth = 0
        while depth in indices:
            depth += 1
        depths.append(depth)
    return min(depths, default=0)


def merge(output: Path, shard_dirs: list[Path]) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")

    shards: dict[int, tuple[Path, dict[str, Any]]] = {}
    reference: dict[str, Any] | None = None
    partition_count: int | None = None
    all_keys: set[tuple[str, str, str, int]] = set()
    key_sources: dict[tuple[str, str, str, int], Path] = {}
    all_rows: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    environment_signature: dict[str, Any] | None = None

    for run_dir in shard_dirs:
        manifest = _load_manifest(run_dir)
        required_fields = {
            *COMMON_FIELDS,
            "expected_sample_count",
            "expected_sample_keys_sha256",
            "raw_content_sha256",
            "environment",
        }
        missing_fields = required_fields - manifest.keys()
        if missing_fields:
            raise ValueError(f"{run_dir}: manifest missing fields {sorted(missing_fields)}")
        if manifest.get("status") != "completed":
            raise ValueError(f"{run_dir}: shard is not completed")
        partition = manifest.get("partition")
        if not isinstance(partition, dict) or set(partition) != {"index", "count"}:
            raise ValueError(f"{run_dir}: invalid partition metadata")
        index, count = partition["index"], partition["count"]
        if not isinstance(index, int) or not isinstance(count, int) or not 0 <= index < count:
            raise ValueError(f"{run_dir}: invalid partition index/count")
        if partition_count is None:
            partition_count = count
        elif count != partition_count:
            raise ValueError("shards disagree on partition count")
        if index in shards:
            raise ValueError(f"duplicate partition index: {index}")
        if reference is None:
            reference = manifest
            environment_signature = _environment_signature(manifest)
        else:
            mismatches = [
                field for field in COMMON_FIELDS if manifest[field] != reference[field]
            ]
            if mismatches:
                raise ValueError(f"{run_dir}: shard manifest mismatch: {mismatches}")
            if _environment_signature(manifest) != environment_signature:
                raise ValueError(f"{run_dir}: shard environment mismatch")
        _verify_snapshot(run_dir, manifest)

        local_rows = _raw_rows(run_dir / "raw", manifest)
        local_keys = set(local_rows)
        duplicates = local_keys & all_keys
        if duplicates:
            key = min(duplicates)
            raise ValueError(f"duplicate sample key {key}: {key_sources[key]} and {run_dir}")
        if len(local_keys) != manifest.get("expected_sample_count"):
            raise ValueError(f"{run_dir}: local sample count does not match manifest")
        if stable_hash(sorted(local_keys)) != manifest.get("expected_sample_keys_sha256"):
            raise ValueError(f"{run_dir}: local sample keys do not match manifest")
        if _raw_content_sha256(local_rows) != manifest["raw_content_sha256"]:
            raise ValueError(f"{run_dir}: raw content hash mismatch")
        key_sources.update({key: run_dir for key in local_keys})
        all_keys.update(local_keys)
        all_rows.update(local_rows)
        shards[index] = (run_dir, manifest)

    assert reference is not None and partition_count is not None
    expected_indices = set(range(partition_count))
    if set(shards) != expected_indices:
        raise ValueError(
            f"incomplete partitions: missing={sorted(expected_indices - set(shards))}"
        )
    if len(all_keys) != reference["global_expected_sample_count"]:
        raise ValueError("merged sample count does not match global manifest")
    if stable_hash(sorted(all_keys)) != reference["global_expected_sample_keys_sha256"]:
        raise ValueError("merged sample keys do not match global manifest")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for index, (run_dir, manifest) in sorted(shards.items()):
            copied_raw = temporary / "raw" / f"partition-{index:05d}"
            shutil.copytree(
                run_dir / "raw",
                copied_raw,
            )
            if (
                _raw_content_sha256(_raw_rows(copied_raw, manifest))
                != manifest["raw_content_sha256"]
            ):
                raise ValueError(f"{run_dir}: raw content changed while copying")
        manifest_dir = temporary / "manifests"
        manifest_dir.mkdir(parents=True)
        shutil.copy2(
            shards[0][0] / "manifests/config.yaml", manifest_dir / "config.yaml"
        )
        shutil.copy2(
            shards[0][0] / "manifests/prompt.json", manifest_dir / "prompt.json"
        )
        merged_manifest = dict(reference)
        for field in (
            "partition",
            "expected_sample_count",
            "global_expected_sample_count",
            "expected_sample_keys_sha256",
            "global_expected_sample_keys_sha256",
            "environment",
            "process_id",
        ):
            merged_manifest.pop(field, None)
        merged_manifest.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "sample_count": len(all_keys),
                "target_sample_count": len(all_keys),
                "common_sample_depth": _common_sample_depth(all_rows),
                "raw_content_sha256": _raw_content_sha256(all_rows),
                "worker_environments": [
                    {"partition": index, "environment": manifest["environment"]}
                    for index, (_run_dir, manifest) in sorted(shards.items())
                ],
            }
        )
        atomic_json(manifest_dir / "run.json", merged_manifest)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shard_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    output = merge(args.output, args.shard_dirs)
    print(json.dumps({"run_dir": str(output), "status": "completed"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
