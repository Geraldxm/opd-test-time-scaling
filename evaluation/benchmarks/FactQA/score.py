#!/usr/bin/env python3
"""Replay and aggregate frozen FactQA v3 scores from a completed math-eval run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import string
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCORER_ID = "fact-v3"
ANSWER_LINE = re.compile(r"^<answer>([^<>\r\n]*)</answer>$")
HERE = Path(__file__).resolve().parent


def stable_hash(value: Any) -> str:
    serialized = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_answer(text: str) -> str:
    """The frozen FlashRAG ExactMatch normalizer used by fact-v3."""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer(raw_text: Any) -> dict[str, Any]:
    """Extract exactly one ordered, nonempty answer span; preserve v3 diagnostics."""
    text = "" if raw_text is None else str(raw_text)
    result = {
        "parser_valid": False,
        "parser_reason": None,
        "answer": None,
        "has_think_tag": "<think>" in text or "</think>" in text,
        "strict_final_line_valid": False,
    }
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        result["parser_reason"] = "answer_tag_count"
        return result
    final_line = text.rstrip().splitlines()[-1] if text.rstrip() else ""
    match = ANSWER_LINE.fullmatch(final_line)
    if match is not None and match.group(1).strip():
        result["strict_final_line_valid"] = True
    opening, closing = text.index("<answer>"), text.index("</answer>")
    if closing < opening:
        result["parser_reason"] = "answer_tag_order"
        return result
    answer = text[opening + len("<answer>"):closing]
    if not answer.strip():
        result["parser_reason"] = "empty_answer"
        return result
    if "<" in answer or ">" in answer:
        result["parser_reason"] = "nested_answer_tag"
        return result
    result.update(parser_valid=True, answer=answer.strip())
    return result


def get_aliases(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata")
    value = metadata.get("golden_answers") if isinstance(metadata, dict) else None
    if value is None:
        value = row.get("golden_answers")
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"dataset row {row.get('id')!r} has no golden_answers list")
    output = [str(item) for item in value if item is not None and str(item).strip()]
    if not output:
        raise ValueError(f"dataset row {row.get('id')!r} has no non-empty golden_answers")
    return output


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path.name}:{line_number}: expected a JSON object")
                yield row


def load_dataset(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    rows = list(read_jsonl(path))
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        problem_id = str(row.get("id", ""))
        problem = row.get("problem")
        if not problem_id or problem_id in lookup:
            raise ValueError(f"{path.name}: missing or duplicate dataset id {problem_id!r}")
        if not isinstance(problem, str):
            raise ValueError(f"dataset row {problem_id!r} has no string problem")
        metadata = row.get("metadata")
        source_dataset = metadata.get("source_dataset") if isinstance(metadata, dict) else None
        lookup[problem_id] = {
            "aliases": get_aliases(row),
            "problem": problem,
            "source_dataset": str(source_dataset or "unknown"),
        }
    if not lookup:
        raise ValueError(f"{path.name}: empty dataset")
    return lookup, stable_hash(rows)


def raw_paths(run_dir: Path) -> list[Path]:
    root = run_dir / "raw"
    if not root.is_dir():
        raise ValueError("run directory must contain raw/")
    active = sorted(root.rglob("*.inprogress"))
    if active:
        names = [p.relative_to(run_dir).as_posix() for p in active[:3]]
        raise ValueError(f"unsealed .inprogress shards remain: {names}")
    paths = sorted([*root.rglob("*.jsonl"), *root.rglob("*.jsonl.gz")])
    if not paths:
        raise ValueError("raw/: no JSONL files")
    return paths


def validate_generation_manifest(
    run_dir: Path, dataset_sha256: str, expected_samples: int, samples_per_problem: int,
) -> tuple[dict[str, Any], Path]:
    path = run_dir / "manifests" / "run.json"
    if not path.is_file():
        raise ValueError("manifests/run.json is required")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise ValueError("math-eval run manifest must have status=completed")
    if manifest.get("partition") is not None:
        raise ValueError("score a merged math-eval run, not an individual partition")
    if manifest.get("schema_version") != 3:
        raise ValueError("expected math-eval raw schema_version=3")
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("math-eval dataset_sha256 differs from the supplied canonical dataset")
    if manifest.get("sample_count") != expected_samples:
        raise ValueError(f"manifest sample_count must be {expected_samples}")
    if manifest.get("common_sample_depth") != samples_per_problem:
        raise ValueError("manifest common_sample_depth differs from requested sample count")
    raw_content_sha256 = manifest.get("raw_content_sha256")
    if not isinstance(raw_content_sha256, str) or len(raw_content_sha256) != 64:
        raise ValueError("math-eval manifest must record raw_content_sha256")
    return manifest, path


def pass_at_k(sample_count: int, correct_count: int, k: int) -> float:
    if correct_count == 0:
        return 0.0
    if sample_count - correct_count < k:
        return 1.0
    return 1.0 - math.comb(sample_count - correct_count, k) / math.comb(sample_count, k)


def default_k_values(sample_count: int) -> list[int]:
    values = [1]
    while values[-1] * 2 <= sample_count:
        values.append(values[-1] * 2)
    if values[-1] != sample_count:
        values.append(sample_count)
    return values


def aggregate(
    verdicts: list[dict[str, Any]], samples_per_problem: int, k_values: list[int],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in verdicts:
        groups[row["problem_id"]].append(row)
    if not groups:
        raise ValueError("no scored responses")
    for problem_id, rows in groups.items():
        if len(rows) != samples_per_problem or {r["sample_idx"] for r in rows} != set(range(samples_per_problem)):
            raise ValueError(f"problem {problem_id!r} does not have a complete sample prefix")
    by_k: dict[str, Any] = {}
    for k in k_values:
        passes, averages = [], []
        for rows in groups.values():
            ordered = sorted(rows, key=lambda row: row["sample_idx"])
            correct = sum(bool(row["em"]) for row in ordered)
            passes.append(pass_at_k(samples_per_problem, correct, k))
            averages.append(sum(bool(row["em"]) for row in ordered[:k]) / k)
        by_k[str(k)] = {
            "pass_at_k": sum(passes) / len(passes),
            "avg_at_k": sum(averages) / len(averages),
        }
    return {"problem_count": len(groups), "k": by_k}


def score(
    run_dir: Path, dataset_path: Path, samples_per_problem: int,
    k_values: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= samples_per_problem <= 1024:
        raise ValueError("samples_per_problem must be in [1, 1024]")
    k_values = default_k_values(samples_per_problem) if k_values is None else sorted(set(k_values))
    if not k_values or any(isinstance(k, bool) or k < 1 or k > samples_per_problem for k in k_values):
        raise ValueError("each K must satisfy 1 <= K <= samples_per_problem")
    lookup, dataset_sha256 = load_dataset(dataset_path)
    expected_samples = len(lookup) * samples_per_problem
    run_manifest, run_manifest_path = validate_generation_manifest(
        run_dir, dataset_sha256, expected_samples, samples_per_problem,
    )
    raw_files = raw_paths(run_dir)
    expected = {(problem_id, sample_idx) for problem_id in lookup for sample_idx in range(samples_per_problem)}
    seen: set[tuple[str, int]] = set()
    verdicts: list[dict[str, Any]] = []
    models, datasets = set(), set()
    problem_uids: dict[str, str] = {}
    for path in raw_files:
        for line, raw in enumerate(read_jsonl(path), 1):
            problem_id, sample_idx = str(raw.get("problem_id", "")), raw.get("sample_idx")
            if not problem_id or not isinstance(sample_idx, int) or isinstance(sample_idx, bool):
                raise ValueError(f"{path.name}:{line}: requires problem_id and integer sample_idx")
            pair = (problem_id, sample_idx)
            if pair in seen:
                raise ValueError(f"duplicate raw result for {pair}")
            if problem_id not in lookup:
                raise ValueError(f"{path.name}:{line}: unknown problem_id {problem_id!r}")
            if raw.get("problem") != lookup[problem_id]["problem"]:
                raise ValueError(f"{path.name}:{line}: problem text differs for {problem_id!r}")
            if not isinstance(raw.get("raw_text"), str):
                raise ValueError(f"{path.name}:{line}: raw_text must be a string")
            if raw.get("run_id") != run_manifest.get("run_id"):
                raise ValueError(f"{path.name}:{line}: raw run_id differs from manifest")
            if raw.get("schema_version") != run_manifest.get("schema_version"):
                raise ValueError(f"{path.name}:{line}: raw schema_version differs from manifest")
            model, dataset, problem_uid = raw.get("model"), raw.get("dataset"), raw.get("problem_uid")
            if not all(isinstance(v, str) and v for v in (model, dataset, problem_uid)):
                raise ValueError(f"{path.name}:{line}: raw model, dataset and problem_uid are required")
            models.add(model)
            datasets.add(dataset)
            previous_problem_id = problem_uids.setdefault(problem_uid, problem_id)
            if previous_problem_id != problem_id:
                raise ValueError(f"problem_uid {problem_uid!r} maps to multiple problem ids")
            seen.add(pair)
            parsed = extract_answer(raw["raw_text"])
            normalized = normalize_answer(parsed["answer"]) if parsed["parser_valid"] else ""
            golden_answers = lookup[problem_id]["aliases"]
            normalized_aliases = [normalize_answer(item) for item in golden_answers]
            correct = bool(normalized) and normalized in normalized_aliases if parsed["parser_valid"] else False
            verdicts.append({
                "scorer_id": SCORER_ID,
                "problem_id": problem_id,
                "sample_idx": sample_idx,
                "raw_path": path.relative_to(run_dir).as_posix(),
                "raw_line": line,
                "truncated": bool(raw.get("truncated", False)),
                "finish_reason": raw.get("finish_reason"),
                "parser_valid": parsed["parser_valid"],
                "parser_reason": parsed["parser_reason"],
                "has_think_tag": parsed["has_think_tag"],
                "strict_final_line_valid": parsed["strict_final_line_valid"],
                "answer": parsed["answer"],
                "normalized_answer": normalized,
                "golden_answers": golden_answers,
                "normalized_golden_answers": normalized_aliases,
                "source_dataset": lookup[problem_id]["source_dataset"],
                "em": correct,
            })
    if len(models) != 1 or len(datasets) != 1:
        raise ValueError("raw output must contain exactly one model and one dataset")
    if next(iter(datasets)) != run_manifest.get("dataset"):
        raise ValueError("raw dataset name differs from generation manifest")
    missing = expected - seen
    unexpected = seen - expected
    if missing or unexpected:
        raise ValueError(f"raw sample keys mismatch (missing={len(missing)}, unexpected={len(unexpected)})")

    counts = Counter()
    for row in verdicts:
        counts["parser_invalid"] += not row["parser_valid"]
        counts["truncated"] += row["truncated"]
        counts["em_correct"] += row["em"]
        counts["think_tag"] += row["has_think_tag"]
        counts["strict_final_line_invalid"] += not row["strict_final_line_valid"]
    total = len(verdicts)
    source_rates = {}
    for source in sorted({row["source_dataset"] for row in verdicts}):
        source_rows = [row for row in verdicts if row["source_dataset"] == source]
        source_rates[source] = {
            "observed": len(source_rows),
            "em": sum(row["em"] for row in source_rows) / len(source_rows),
            "parser_invalid": sum(not row["parser_valid"] for row in source_rows),
            "truncated": sum(row["truncated"] for row in source_rows),
            "strict_final_line_invalid": sum(not row["strict_final_line_valid"] for row in source_rows),
        }
    raw_sha256 = {
        path.relative_to(run_dir).as_posix(): file_sha256(path) for path in raw_files
    }
    run_manifest_sha256 = file_sha256(run_manifest_path)
    aggregation = aggregate(verdicts, samples_per_problem, k_values)
    metrics = {
        "scorer_id": SCORER_ID,
        "dataset": dataset_path.stem,
        "model": next(iter(models)),
        "run_id": run_manifest.get("run_id"),
        "expected": expected_samples,
        "observed": total,
        "parser_invalid": counts["parser_invalid"],
        "truncated": counts["truncated"],
        "think_tag_diagnostic": counts["think_tag"],
        "strict_final_line_invalid": counts["strict_final_line_invalid"],
        "em": counts["em_correct"] / total,
        "parser_invalid_rate": counts["parser_invalid"] / total,
        "truncation_rate": counts["truncated"] / total,
        "extraction_rate": 1 - counts["parser_invalid"] / total,
        "per_source": source_rates,
        "dataset_sha256": dataset_sha256,
        "generation_manifest_sha256": run_manifest_sha256,
        "raw_content_sha256": run_manifest["raw_content_sha256"],
        "raw_sha256": raw_sha256,
        "samples_per_problem": samples_per_problem,
        "problem_count": aggregation["problem_count"],
        "k": aggregation["k"],
    }
    return verdicts, metrics


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_outputs(run_dir: Path, verdicts: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    output = run_dir / "parsed" / SCORER_ID
    scorer_sha256 = file_sha256(Path(__file__))
    provenance = {
        "scorer_id": SCORER_ID,
        "scorer_sha256": scorer_sha256,
        "dataset_sha256": metrics["dataset_sha256"],
        "generation_manifest_sha256": metrics["generation_manifest_sha256"],
        "raw_content_sha256": metrics["raw_content_sha256"],
        "raw_sha256": metrics["raw_sha256"],
        "samples_per_problem": metrics["samples_per_problem"],
        "k_values": [int(k) for k in metrics["k"]],
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != provenance:
            raise FileExistsError(f"parsed outputs already exist for different inputs: {SCORER_ID}")
    atomic_write(output / "parsed.jsonl", "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in verdicts
    ))
    atomic_write(output / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    atomic_write(manifest_path, json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")


def check_package() -> dict[str, Any]:
    manifest_path = HERE / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("package") != "FactQA" or manifest.get("data_version") != "fact_eval_v2":
        raise ValueError("unexpected FactQA data manifest")
    if manifest.get("scorer_sha256") != file_sha256(Path(__file__)):
        raise ValueError("score.py hash differs from data_manifest.json")
    prompt = HERE / "prompts" / "chat" / "00-user.txt"
    if file_sha256(prompt) != manifest.get("prompt_sha256"):
        raise ValueError("prompt hash differs from data_manifest.json")
    provenance_path = HERE / "data" / "selection_provenance.json"
    if file_sha256(provenance_path) != manifest.get("selection_provenance_sha256"):
        raise ValueError("selection provenance hash differs from data_manifest.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_names = {"2wikimultihopqa.jsonl", "hotpotqa.jsonl", "popqa.jsonl", "triviaqa.jsonl"}
    data_dir = HERE / "data" / "canonical"
    actual_names = {p.name for p in data_dir.glob("*.jsonl")}
    if actual_names != expected_names:
        raise ValueError(f"canonical datasets must be exactly {sorted(expected_names)}")
    if set(provenance.get("datasets", {})) != {Path(name).stem for name in expected_names}:
        raise ValueError("selection provenance must contain only the four paper FactQA datasets")
    for name in sorted(expected_names):
        path = data_dir / name
        entry = manifest["datasets"].get(path.stem)
        if not entry or file_sha256(path) != entry.get("canonical_file_sha256"):
            raise ValueError(f"{name}: frozen canonical file hash mismatch")
        rows = list(read_jsonl(path))
        if len(rows) != 200 or stable_hash(rows) != entry.get("math_eval_dataset_sha256"):
            raise ValueError(f"{name}: expected 200 frozen rows and matching math-eval hash")
        selected_ids = {item["id"] for item in provenance["datasets"][path.stem]["selected_questions"]}
        if selected_ids != {str(row["id"]) for row in rows}:
            raise ValueError(f"{name}: selected question IDs differ from canonical data")
    return {"datasets": sorted(expected_names), "questions": 800, "scorer_id": SCORER_ID}


def self_check() -> None:
    """Run a two-sample CPU-only end-to-end check on temporary synthetic data."""
    problem = {"id": "tiny:one", "problem": "Who wrote Hamlet?", "answer": "William Shakespeare",
              "metadata": {"golden_answers": ["William Shakespeare", "Shakespeare"],
                           "source_dataset": "tiny"}}
    dataset_rows = [problem]
    dataset_hash = stable_hash(dataset_rows)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset_path = root / "tiny.jsonl"
        dataset_path.write_text(json.dumps(problem, ensure_ascii=False) + "\n", encoding="utf-8")
        run_dir = root / "run"
        raw_dir = run_dir / "raw" / "tiny"
        (run_dir / "manifests").mkdir(parents=True)
        raw_dir.mkdir(parents=True)
        raw_rows = []
        for sample_idx, text in enumerate(("Thoughts.\n<answer>William Shakespeare</answer>", "<answer>Wrong</answer>")):
            raw_rows.append({
                "schema_version": 3, "run_id": "tiny-run", "model": "tiny-model",
                "dataset": "tiny", "problem_id": "tiny:one", "problem_uid": "tiny:tiny:one",
                "problem": problem["problem"], "sample_idx": sample_idx,
                "raw_text": text, "truncated": False, "finish_reason": "stop",
            })
        raw_path = raw_dir / "part-00000.jsonl"
        raw_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows), encoding="utf-8")
        (run_dir / "manifests" / "run.json").write_text(json.dumps({
            "status": "completed", "schema_version": 3, "run_id": "tiny-run", "dataset": "tiny",
            "dataset_sha256": dataset_hash, "sample_count": 2, "common_sample_depth": 2,
            "raw_content_sha256": stable_hash(raw_rows),
        }), encoding="utf-8")
        verdicts, metrics = score(run_dir, dataset_path, 2, [1, 2])
        assert [row["em"] for row in verdicts] == [True, False]
        assert metrics["k"]["1"]["avg_at_k"] == 1.0
        assert metrics["k"]["2"]["avg_at_k"] == 0.5
        assert metrics["k"]["1"]["pass_at_k"] == 0.5
        assert metrics["k"]["2"]["pass_at_k"] == 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--samples-per-problem", type=int, default=1024)
    parser.add_argument("--k", type=int, nargs="+")
    parser.add_argument("--check-package", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.check_package:
        print(json.dumps(check_package(), sort_keys=True))
        return
    if args.self_check:
        self_check()
        print("FactQA CPU self-check passed")
        return
    if args.run_dir is None or args.dataset is None:
        parser.error("--run-dir and --dataset are required unless using a check option")
    package = check_package()
    allowed = {Path(name).stem: entry for name, entry in package_dataset_entries().items()}
    dataset_path = args.dataset.resolve()
    if dataset_path.stem not in allowed or file_sha256(dataset_path) != allowed[dataset_path.stem]["canonical_file_sha256"]:
        raise SystemExit("--dataset must be one of the four unchanged canonical files in this package")
    verdicts, metrics = score(args.run_dir.resolve(), dataset_path, args.samples_per_problem, args.k)
    write_outputs(args.run_dir.resolve(), verdicts, metrics)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


def package_dataset_entries() -> dict[str, dict[str, Any]]:
    return json.loads((HERE / "data_manifest.json").read_text(encoding="utf-8"))["datasets"]


if __name__ == "__main__":
    main()
