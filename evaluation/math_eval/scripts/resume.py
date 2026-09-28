#!/usr/bin/env python3
"""Durable sample-level JSONL storage and resume helpers."""

from __future__ import annotations

import gzip
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from io_utils import atomic_json, atomic_text, read_jsonl, stable_hash


SCHEMA_VERSION = 3
SampleKey = tuple[str, str, str, int]


def sample_key(row: dict[str, Any]) -> SampleKey:
    return (
        str(row["model"]),
        str(row["dataset"]),
        str(row["problem_uid"]),
        int(row["sample_idx"]),
    )


def target_key(model: str, dataset: str, problem_uid: str, sample_idx: int) -> SampleKey:
    return model, dataset, problem_uid, sample_idx


def _canonical(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _part_index(path: Path) -> int:
    return int(path.name.split("-", 1)[1].split(".", 1)[0])


def _read_active(path: Path, recover_tail: bool) -> list[dict[str, Any]]:
    lines = path.read_bytes().splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    valid_bytes = 0
    unterminated_valid = False
    for index, line in enumerate(lines):
        complete = line.endswith(b"\n")
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            row = None
        if not isinstance(row, dict):
            if recover_tail and index == len(lines) - 1:
                break
            raise ValueError(f"{path}:{index + 1}: invalid JSONL row")
        rows.append(row)
        valid_bytes += len(line)
        unterminated_valid = not complete
    total_bytes = sum(map(len, lines))
    if recover_tail and valid_bytes != total_bytes:
        with path.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    elif recover_tail and unterminated_valid:
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return rows


def read_raw(path: Path, recover_tail: bool = False) -> list[dict[str, Any]]:
    if path.name.endswith(".jsonl.inprogress"):
        return _read_active(path, recover_tail)
    return list(read_jsonl(path))


@dataclass(frozen=True)
class IndexedRow:
    row: dict[str, Any]
    path: Path


def scan_raw(root: Path, recover_tail: bool = False) -> dict[SampleKey, IndexedRow]:
    completed: dict[SampleKey, IndexedRow] = {}
    if not root.exists():
        return completed
    patterns = ("**/part-*.jsonl", "**/part-*.jsonl.gz", "**/part-*.jsonl.inprogress")
    paths = sorted({path for pattern in patterns for path in root.glob(pattern)})
    for path in paths:
        for row in read_raw(path, recover_tail):
            key = sample_key(row)
            previous = completed.get(key)
            if previous and _canonical(previous.row) != _canonical(row):
                raise ValueError(f"conflicting rows for sample key {key}: {previous.path} and {path}")
            completed.setdefault(key, IndexedRow(row, path))
    return completed


def missing_indices(
    completed: dict[SampleKey, IndexedRow],
    model: str,
    dataset: str,
    problem_uid: str,
    samples_per_problem: int,
) -> list[int]:
    return [
        index
        for index in range(samples_per_problem)
        if target_key(model, dataset, problem_uid, index) not in completed
    ]


class RawShardWriter:
    def __init__(
        self,
        directory: Path,
        shard_size: int = 100,
        compression: str = "none",
        fsync_every: int = 1,
    ):
        if shard_size < 1 or fsync_every < 1:
            raise ValueError("shard_size and fsync_every must be >= 1")
        if compression not in {"none", "gzip"}:
            raise ValueError("compression must be 'none' or 'gzip'")
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.compression = compression
        self.fsync_every = fsync_every
        active = sorted(directory.glob("part-*.jsonl.inprogress"))
        if len(active) > 1:
            raise ValueError(f"{directory}: multiple active shards")
        all_parts = [p for pattern in ("part-*.jsonl", "part-*.jsonl.gz", "part-*.jsonl.inprogress") for p in directory.glob(pattern)]
        self.part_index = _part_index(active[0]) if active else max((_part_index(p) for p in all_parts), default=-1) + 1
        self.active_path = directory / f"part-{self.part_index:05d}.jsonl.inprogress"
        self.row_count = len(read_raw(self.active_path, True)) if self.active_path.exists() else 0
        self._handle = None
        self._since_sync = 0
        if self.row_count >= shard_size:
            self._seal()

    def _open(self) -> None:
        if self._handle is None:
            self._handle = self.active_path.open("a", encoding="utf-8")

    def append(self, row: dict[str, Any]) -> None:
        self._open()
        self._handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._handle.flush()
        self._since_sync += 1
        if self._since_sync >= self.fsync_every:
            os.fsync(self._handle.fileno())
            self._since_sync = 0
        self.row_count += 1
        if self.row_count >= self.shard_size:
            self._seal()

    def _seal(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
        if not self.active_path.exists():
            return
        if self.row_count == 0:
            self.active_path.unlink()
            return
        suffix = ".jsonl.gz" if self.compression == "gzip" else ".jsonl"
        sealed = self.directory / f"part-{self.part_index:05d}{suffix}"
        if sealed.exists():
            if self.compression != "gzip":
                raise FileExistsError(f"refusing to overwrite {sealed}")
            with gzip.open(sealed, "rb") as source:
                if source.read() != self.active_path.read_bytes():
                    raise FileExistsError(f"conflicting sealed and active shards: {sealed}")
            self.active_path.unlink()
        elif self.compression == "gzip":
            temporary = sealed.with_suffix(sealed.suffix + ".tmp")
            try:
                with self.active_path.open("rb") as source, gzip.open(temporary, "wb") as target:
                    shutil.copyfileobj(source, target)
                os.replace(temporary, sealed)
            finally:
                temporary.unlink(missing_ok=True)
            self.active_path.unlink()
        else:
            os.replace(self.active_path, sealed)
        self.part_index += 1
        self.active_path = self.directory / f"part-{self.part_index:05d}.jsonl.inprogress"
        self.row_count = 0
        self._since_sync = 0

    def close(self) -> None:
        self._seal()

    def abort(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "RawShardWriter":
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        self.abort()
        if exc_type is None:
            self._seal()


def ensure_manifest(
    run_dir: Path,
    payload: dict[str, Any],
    generation_spec: dict[str, Any],
    config_text: str,
    prompt: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    manifest_dir = run_dir / "manifests"
    manifest_path = manifest_dir / "run.json"
    config_path = manifest_dir / "config.yaml"
    prompt_path = manifest_dir / "prompt.json"
    config_hash = stable_hash(config_text)
    prompt_hash = stable_hash(prompt)
    spec_hash = stable_hash(generation_spec)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"{manifest_path} exists; pass --resume")
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("run schema does not match the current schema")
        if not config_path.exists():
            raise FileNotFoundError(f"cannot resume without {config_path}")
        if not prompt_path.exists():
            raise FileNotFoundError(f"cannot resume without {prompt_path}")
        snapshot_hash = stable_hash(config_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != snapshot_hash:
            raise ValueError("config snapshot does not match run manifest")
        prompt_snapshot_hash = stable_hash(
            json.loads(prompt_path.read_text(encoding="utf-8"))
        )
        if existing.get("prompt_sha256") != prompt_snapshot_hash:
            raise ValueError("prompt snapshot does not match run manifest")
        if existing.get("prompt_sha256") != prompt_hash:
            raise ValueError("prompt does not match existing run")
        if existing.get("generation_spec_hash") != spec_hash:
            raise ValueError("generation spec does not match existing run")
        return existing
    if resume:
        raise FileNotFoundError(f"cannot resume without {manifest_path}")
    if config_path.exists():
        if stable_hash(config_path.read_text(encoding="utf-8")) != config_hash:
            raise FileExistsError(f"refusing to overwrite {config_path}")
    else:
        atomic_text(config_path, config_text)
    if prompt_path.exists():
        if stable_hash(json.loads(prompt_path.read_text(encoding="utf-8"))) != prompt_hash:
            raise FileExistsError(f"refusing to overwrite {prompt_path}")
    else:
        atomic_json(prompt_path, prompt)
    manifest = {
        **payload,
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config_hash,
        "prompt_sha256": prompt_hash,
        "generation_spec_hash": spec_hash,
    }
    atomic_json(manifest_path, manifest)
    return manifest
