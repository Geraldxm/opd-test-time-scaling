#!/usr/bin/env python3
"""Run a one-task CPU judge smoke and scorer contract checks in a temp copy."""

import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_jsonl(path: Path, row: dict) -> bytes:
    content = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    path.write_bytes(content)
    return content


def patch_pin(path: Path, key: str, digest: str) -> None:
    text = path.read_text(encoding="utf-8")
    text, count = re.subn(rf'^{key} = "[0-9a-f]+"$', f'{key} = "{digest}"', text, flags=re.MULTILINE)
    require(count == 1, f"could not update test fixture pin {key}")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="evalplus-cpu-smoke-") as temp:
        temp_root = Path(temp)
        temp_root.chmod(0o755)
        package = temp_root / "package"
        package.mkdir()
        for name in ("score.py", "raw_to_samples.py", "sandbox.py", "VENDORED_TIMEOUT_PATCH.diff"):
            shutil.copy2(ROOT / name, package / name)
        shutil.copytree(ROOT / "vendor/evalplus", package / "vendor/evalplus", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        metadata_path = ROOT / "data/canonical" / next(ROOT.joinpath("data/canonical").glob("*.metadata.json")).name
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = package / metadata["source_path"]
        canonical = package / metadata["canonical_path"]
        source.parent.mkdir(parents=True)
        canonical.parent.mkdir(parents=True)
        source_row = json.loads((ROOT / metadata["source_path"]).read_text(encoding="utf-8").splitlines()[0])
        canonical_row = json.loads((ROOT / metadata["canonical_path"]).read_text(encoding="utf-8").splitlines()[0])
        require(str(source_row["task_id"]) == str(canonical_row["id"]), "source/canonical smoke task IDs differ")
        source_bytes = write_jsonl(source, source_row)
        canonical_bytes = write_jsonl(canonical, canonical_row)
        metadata["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        metadata["canonical_sha256"] = hashlib.sha256(canonical_bytes).hexdigest()
        metadata["source_row_count"] = metadata["canonical_row_count"] = 1
        (package / metadata["canonical_path"]).with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        patch_pin(package / "score.py", "PINNED_SOURCE_SHA256", metadata["source_sha256"])
        patch_pin(package / "score.py", "PINNED_CANONICAL_SHA256", metadata["canonical_sha256"])
        patch_pin(package / "raw_to_samples.py", "CANONICAL_SHA256", metadata["canonical_sha256"])

        raw = temp_root / "raw" / "part-00000.jsonl"
        raw.parent.mkdir()
        solution = str(source_row["canonical_solution"])
        if any(line.startswith("def ") for line in str(source_row["prompt"]).splitlines()):
            solution = str(source_row["prompt"]) + solution
        good_response = f"Explanation before code.\n```python\n{solution}\n```\n```python\nraise RuntimeError('second fence must be ignored')\n```"
        bad_response = "```python\nraise RuntimeError('intentional failing sample')\n```"
        raw_rows = [
            {"problem_id": source_row["task_id"], "sample_idx": 1, "input_mode": "chat", "problem": source_row["prompt"], "final_text": good_response},
            {"problem_id": source_row["task_id"], "sample_idx": 0, "input_mode": "chat", "problem": source_row["prompt"], "final_text": bad_response},
        ]
        raw.write_text("".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8")
        output = temp_root / "score"
        command = [sys.executable, str(package / "score.py"), "--raw", str(raw), "--out-dir", str(output), "--ks", "1,2", "--parallel", "1"]
        result = subprocess.run(command, text=True, capture_output=True, timeout=1800)
        if result.returncode:
            raise RuntimeError(f"one-task EvalPlus CPU smoke failed:\n{result.stdout}\n{result.stderr}")
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        require(summary["num_problems"] == 1 and summary["samples_per_problem"] == 2, "smoke result has wrong problem/sample count")
        require(summary["pass_at_k"]["1"] == 0.5 and summary["pass_at_k"]["2"] == 1.0, "base+plus verdicts or converter order are wrong")
        require(summary["avg_at_k"]["1"] == 0.0 and summary["avg_at_k"]["2"] == 0.5, "avg@K did not follow sorted sample_idx order")
        eval_rows = json.loads((output / "eval_results.json").read_text(encoding="utf-8"))["eval"][source_row["task_id"]]
        require(eval_rows[0]["base_status"] == "fail" and eval_rows[0]["plus_status"] == "fail", "sample_idx=0 was not judged first")
        require(eval_rows[1]["base_status"] == "pass" and eval_rows[1]["plus_status"] == "pass", "canonical smoke sample did not pass base+plus")

        sys.path.insert(0, str(package))
        score = importlib.import_module("score")
        synthetic = temp_root / "synthetic_eval_results.json"
        synthetic.write_text(json.dumps({
            "hash": hashlib.md5(source_bytes).hexdigest(),
            "eval": {source_row["task_id"]: [
                {"base_status": "timeout", "plus_status": "pass"},
                {"base_status": "pass", "plus_status": "pass"},
                {"base_status": "fail", "plus_status": "timeout"},
            ]},
        }), encoding="utf-8")
        aggregate_dir = temp_root / "aggregate"
        aggregate_dir.mkdir()
        ordered = score.aggregate(synthetic, aggregate_dir, "1,2,3")
        require(score.default_ks(3) == [1, 2, 3], "default K list omitted a non-power-of-two full-depth K")
        require(ordered["correct_per_problem"][source_row["task_id"]] == 1, "timeout verdict was counted as correct")
        require(ordered["avg_at_k"] == {"1": 0.0, "2": 0.5, "3": 1 / 3}, "avg@K did not use verdict completion order")

        # Both integrity checks must fail before any replay or code execution.
        score_path = package / "score.py"
        del sys.modules["score"]
        for data_path, label in ((source, "source"), (canonical, "canonical")):
            original = data_path.read_bytes()
            data_path.write_bytes(original + b" ")
            bad_out = temp_root / f"bad-{label}"
            rejected = subprocess.run(
                [sys.executable, str(score_path), "--eval-results", str(output / "eval_results.json"), "--out-dir", str(bad_out)],
                text=True, capture_output=True, timeout=30,
            )
            data_path.write_bytes(original)
            require(rejected.returncode != 0 and "SHA-256 mismatch" in rejected.stderr, f"tampered {label} snapshot was not rejected")

    print("EvalPlus CPU smoke and scorer contract checks: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
