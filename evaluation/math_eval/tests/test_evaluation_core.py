import gzip
import json
import os
import signal
import sys
import tempfile
import threading
import unittest
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import yaml
from math_verify.errors import TimeoutException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_runs import compare, load_run
from evaluate import load_config, load_dataset, run
from inference import (
    InferenceResult,
    _retry_delay,
    OpenAIBackend,
    load_prompt,
    normalize_local_text,
    reject_soft_thinking,
    render_prompt,
    resolved_thinking_request,
    sampling_request,
    thinking_template_preflight,
    validate_model,
)
from io_utils import read_jsonl
from merge_shards import merge
from metrics import compute_metrics
from parser import (
    DUAL_PARSER_CONFIG_HASH,
    DUAL_PARSER_ID,
    PARSER_CONFIG_HASH,
    PARSER_ID,
    V5_DUAL_PARSER_CONFIG_HASH,
    V5_DUAL_PARSER_ID,
    V51_DUAL_PARSER_CONFIG_HASH,
    V51_DUAL_PARSER_ID,
    V52_DUAL_PARSER_CONFIG_HASH,
    V52_DUAL_PARSER_ID,
    parse_and_verify,
    parse_dual_and_verify,
    parse_v5_dual_and_verify,
    parse_v51_dual_and_verify,
    parse_v52_dual_and_verify,
)
from replay_evaluation import replay
from resume import RawShardWriter, scan_raw


def decode_config(**updates):
    value = {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_output_tokens": 32,
        "samples_per_problem": 1,
    }
    value.update(updates)
    return value


class MockHandler(BaseHTTPRequestHandler):
    payload = None
    attempts = 0
    retry_once = True
    retry_after = "Wed, 21 Oct 2015 07:28:00 GMT"
    malformed_choice = False
    response_id = 123

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        type(self).payload = json.loads(self.rfile.read(length))
        type(self).attempts += 1
        if type(self).retry_once and type(self).attempts == 1:
            self.send_response(429)
            self.send_header("Retry-After", type(self).retry_after)
            self.end_headers()
            return
        choice = (
            7
            if type(self).malformed_choice
            else {
                "finish_reason": "stop",
                "message": {
                    "reasoning": "two plus two",
                    "content": "\\boxed{4}",
                },
            }
        )
        body = json.dumps(
            {
                "id": type(self).response_id,
                "choices": [choice],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class ContractTest(unittest.TestCase):
    def test_canonical_loader_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                '{"id":"x","problem":"p","answer":"1"}\n'
                '{"id":"x","problem":"q","answer":"2"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                load_dataset({"path": str(path)})

    def test_decode_mapping_and_omission(self):
        request = sampling_request(
            decode_config(
                temperature=0.2,
                top_p=0.9,
                top_k=20,
                stop_token_ids=[151645],
            )
        )
        self.assertEqual(request["max_tokens"], 32)
        self.assertEqual(request["top_k"], 20)
        self.assertEqual(request["stop_token_ids"], [151645])
        self.assertNotIn("min_p", request)
        self.assertNotIn("presence_penalty", request)
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            sampling_request({**decode_config(), "mystery": True})
        with self.assertRaisesRegex(ValueError, "cannot be null"):
            sampling_request(decode_config(min_p=None))
        with self.assertRaisesRegex(ValueError, "missing fields"):
            sampling_request({"max_output_tokens": 1})
        with self.assertRaisesRegex(ValueError, "stop_token_ids"):
            sampling_request(decode_config(stop_token_ids=[True]))

    def test_openai_required_fields_fail_validation(self):
        complete = {
            "name": "mock",
            "backend": "openai",
            "input_mode": "chat",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_model": "mock",
            "api_key_env": "TEST_API_KEY",
        }
        for key in ("endpoint", "api_model", "api_key_env"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_model({name: value for name, value in complete.items() if name != key})

    def test_retry_after_parsing(self):
        self.assertEqual(_retry_delay("2", 9), 2)
        self.assertEqual(_retry_delay("not-a-date", 9), 9)
        self.assertEqual(
            _retry_delay("Wed, 21 Oct 2015 07:28:00 GMT", 9),
            0,
        )

    def test_thinking_output_statuses_and_soft_rejection(self):
        cases = [
            ("answer", None, "stop", "not_applicable"),
            ("work\n</think>\n\\boxed{2}", True, "stop", "thinking_completed"),
            ("unfinished", True, "length", "thinking_truncated"),
            ("unfinished", True, "stop", "thinking_format_error"),
            ("\\boxed{2}", False, "stop", "non_thinking_ok"),
            ("work</think>\\boxed{2}", False, "stop", "non_thinking_violation"),
        ]
        for raw_text, thinking, finish_reason, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    normalize_local_text(raw_text, thinking, finish_reason)[2],
                    expected,
                )
        reasoning, final, _ = normalize_local_text(
            "work\n</think>\n\\boxed{2}", True, "stop"
        )
        self.assertEqual((reasoning, final), ("work", "\\boxed{2}"))
        self.assertEqual(
            resolved_thinking_request({"backend": "vllm", "thinking": True}),
            {"chat_template_kwargs": {"enable_thinking": True}},
        )
        with self.assertRaisesRegex(ValueError, "soft thinking"):
            reject_soft_thinking("solve this /think")

    def test_thinking_template_capability_matrix(self):
        class ToggleTokenizer:
            chat_template = "toggle-template"

            def apply_chat_template(self, _messages, **kwargs):
                return f"prompt:{kwargs['enable_thinking']}"

        class FixedTokenizer:
            chat_template = "fixed-template"

            def apply_chat_template(self, _messages, **_kwargs):
                return "prompt"

        base = {
            "name": "fake",
            "backend": "vllm",
            "path": "unused",
            "input_mode": "chat",
        }
        supported = thinking_template_preflight(
            ToggleTokenizer(), {**base, "thinking": False}
        )
        self.assertTrue(supported["supported"])
        self.assertFalse(supported["requested"])
        with self.assertRaisesRegex(ValueError, "must be explicit"):
            thinking_template_preflight(ToggleTokenizer(), base)
        unsupported = thinking_template_preflight(FixedTokenizer(), base)
        self.assertFalse(unsupported["supported"])
        with self.assertRaisesRegex(ValueError, "omit model.thinking"):
            thinking_template_preflight(FixedTokenizer(), {**base, "thinking": True})
        with self.assertRaisesRegex(ValueError, "completion models cannot set thinking"):
            validate_model({**base, "input_mode": "completion", "thinking": False})
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            validate_model({**base, "profile": "qwen3.5-hybrid"})


class PromptTest(unittest.TestCase):
    def test_file_prompt_loading_rendering_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completion = root / "completion.txt"
            completion.write_bytes(b"before\r\n{{problem}}\r\nafter")
            prompt = load_prompt(str(completion), "completion")
            self.assertEqual(
                render_prompt(prompt, "2+2?"),
                "before\r\n2+2?\r\nafter",
            )

            chat = root / "chat"
            chat.mkdir()
            (chat / "00-system.txt").write_text("rules\n", encoding="utf-8")
            (chat / "10-user.txt").write_text("Q: {{problem}}\n", encoding="utf-8")
            prompt = load_prompt(str(chat), "chat")
            self.assertEqual(
                render_prompt(prompt, "2+2?"),
                [
                    {"role": "system", "content": "rules\n"},
                    {"role": "user", "content": "Q: 2+2?\n"},
                ],
            )

            completion.write_text("no marker", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly once"):
                load_prompt(str(completion), "completion")
            completion.write_text("{{problem}} {{problem}}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly once"):
                load_prompt(str(completion), "completion")

            bad = root / "bad"
            bad.mkdir()
            (bad / "system.txt").write_text("rules", encoding="utf-8")
            (bad / "01-user.txt").write_text("{{problem}}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must match"):
                load_prompt(str(bad), "chat")

            wrong_end = root / "wrong-end"
            wrong_end.mkdir()
            (wrong_end / "00-user.txt").write_text("{{problem}}", encoding="utf-8")
            (wrong_end / "01-assistant.txt").write_text("answer", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "end with a user"):
                load_prompt(str(wrong_end), "chat")


class OpenAIAdapterTest(unittest.TestCase):
    @staticmethod
    def model(port):
        return {
            "name": "mock",
            "backend": "openai",
            "input_mode": "chat",
            "thinking": False,
            "reasoning_parameter": "chat_template_kwargs.enable_thinking",
            "capabilities": ["chat_template_kwargs.enable_thinking"],
            "endpoint": f"http://127.0.0.1:{port}/v1/chat/completions",
            "api_model": "mock",
            "api_key_env": "TEST_API_KEY",
            "runtime": {"timeout_seconds": 5, "max_retries": 1},
        }

    def test_direct_request_and_response_separation(self):
        MockHandler.attempts = 0
        MockHandler.retry_once = True
        MockHandler.malformed_choice = False
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {"TEST_API_KEY": "secret"}):
                result = OpenAIBackend(self.model(server.server_port)).generate(
                    [{"role": "user", "content": "2+2?"}],
                    decode_config(max_output_tokens=16),
                    0,
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(result.reasoning_text, "two plus two")
        self.assertEqual(result.final_text, "\\boxed{4}")
        self.assertEqual(MockHandler.payload["max_tokens"], 16)
        self.assertEqual(
            MockHandler.payload["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertEqual(MockHandler.attempts, 2)
        self.assertEqual(result.response_id, "123")
        self.assertEqual(result.thinking_status, "non_thinking_violation")
        self.assertNotIn("secret", json.dumps(result.as_dict()))

    def test_stop_token_ids_require_provider_capability(self):
        with (
            patch.dict(os.environ, {"TEST_API_KEY": "secret"}),
            self.assertRaisesRegex(ValueError, "stop_token_ids"),
        ):
            OpenAIBackend(self.model(1)).generate(
                [{"role": "user", "content": "2+2?"}],
                decode_config(stop_token_ids=[151645]),
                0,
            )

    def test_malformed_choice_is_rejected(self):
        MockHandler.attempts = 0
        MockHandler.retry_once = False
        MockHandler.malformed_choice = True
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with (
                patch.dict(os.environ, {"TEST_API_KEY": "secret"}),
                self.assertRaisesRegex(ValueError, "choice must be an object"),
            ):
                OpenAIBackend(self.model(server.server_port)).generate(
                    [{"role": "user", "content": "2+2?"}],
                    decode_config(),
                    0,
                )
        finally:
            MockHandler.retry_once = True
            MockHandler.malformed_choice = False
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class ResumeTest(unittest.TestCase):
    @staticmethod
    def row(index):
        return {
            "model": "m",
            "dataset": "d",
            "problem_uid": f"d:{index}",
            "sample_idx": 0,
        }

    def test_none_and_gzip_roundtrip(self):
        for compression in ("none", "gzip"):
            with self.subTest(compression=compression), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with RawShardWriter(root / "m" / "d", shard_size=1, compression=compression) as writer:
                    writer.append(self.row(0))
                    writer.append(self.row(1))
                rows = scan_raw(root)
                self.assertEqual(len(rows), 2)
                suffix = ".jsonl.gz" if compression == "gzip" else ".jsonl"
                self.assertEqual(len(list((root / "m" / "d").glob(f"*{suffix}"))), 2)

    def test_active_tail_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "m" / "d" / "part-00000.jsonl.inprogress"
            active.parent.mkdir(parents=True)
            active.write_bytes(
                (json.dumps(self.row(0)) + "\n").encode() + b'{"model":'
            )
            rows = scan_raw(root, recover_tail=True)
            self.assertEqual(len(rows), 1)
            self.assertTrue(active.read_bytes().endswith(b"\n"))

    def test_complete_tail_without_newline_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "part-00000.jsonl.inprogress"
            active.write_text(json.dumps(self.row(0)), encoding="utf-8")
            rows = scan_raw(root, recover_tail=True)
            self.assertEqual(len(rows), 1)
            self.assertTrue(active.read_bytes().endswith(b"\n"))

    def test_empty_active_shard_is_removed_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "part-00000.jsonl.inprogress"
            active.touch()
            RawShardWriter(root).close()
            self.assertFalse(active.exists())

    def test_gzip_orphan_recovery_and_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "part-00000.jsonl.inprogress"
            sealed = root / "part-00000.jsonl.gz"
            payload = (json.dumps(self.row(0)) + "\n").encode()
            active.write_bytes(payload)
            with gzip.open(sealed, "wb") as handle:
                handle.write(payload)
            writer = RawShardWriter(root, shard_size=1, compression="gzip")
            self.assertFalse(active.exists())
            self.assertEqual(len(scan_raw(root)), 1)
            writer.abort()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "part-00000.jsonl.inprogress"
            sealed = root / "part-00000.jsonl.gz"
            active.write_text(json.dumps(self.row(0)) + "\n", encoding="utf-8")
            with gzip.open(sealed, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(self.row(1)) + "\n")
            with self.assertRaisesRegex(FileExistsError, "conflicting"):
                RawShardWriter(root, shard_size=1, compression="gzip")
            self.assertTrue(active.exists())
            self.assertTrue(sealed.exists())

    def test_gzip_failure_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "part-00000.jsonl.inprogress"
            active.write_text(json.dumps(self.row(0)) + "\n", encoding="utf-8")
            writer = RawShardWriter(root, shard_size=10, compression="gzip")
            with (
                patch("resume.shutil.copyfileobj", side_effect=OSError("boom")),
                self.assertRaisesRegex(OSError, "boom"),
            ):
                writer.close()
            self.assertTrue(active.exists())
            self.assertEqual(list(root.glob("*.tmp")), [])


class ParserAndMetricsTest(unittest.TestCase):
    def test_parser_statuses_and_upstream_defaults(self):
        cases = [
            (r"\boxed{42}", "42", "correct"),
            (r"\fbox{42}", "42", "correct"),
            (r"\boxed{\frac{1}{2}}", "0.5", "correct"),
            (r"\boxed{x^2-1}", "(x-1)(x+1)", "correct"),
            (
                r"\boxed{(3, \frac{\pi}{2})}",
                r"\left(3, \frac{\pi}{2}\right)",
                "correct",
            ),
            (r"\boxed{\frac{5}{9}}", r"\frac 59", "correct"),
            (r"\boxed{10080}", r"10,\!080", "correct"),
            (r"\boxed{6-5i}", "6 - 5i", "correct"),
            (r"\boxed{0.333333}", r"\frac{1}{3}", "correct"),
            (r"\boxed{0.3333}", r"\frac{1}{3}", "incorrect"),
            (r"\boxed{0.34}", r"\frac{1}{3}", "incorrect"),
            (r"\boxed{12^{\mathrm{th}}}", "12", "correct"),
            ("The final answer is $12$.", "12", "correct"),
            (r"\boxed{1}\boxed{2}", "2", "correct"),
            (r"\boxed{1}\boxed{2}\boxed{\frac{3}{", "2", "correct"),
            (
                r"\boxed{\begin{pmatrix}1\\2\end{pmatrix}}",
                "(1,2)",
                "incorrect",
            ),
            (r"\boxed{\{x \mid x > 0\}}", r"(0,\infty)", "incorrect"),
            (r"\boxed{41}", "42", "incorrect"),
            ("", "42", "no_candidate"),
            ("No final answer was produced.", "42", "parse_error"),
        ]
        for final, gold, status in cases:
            with self.subTest(final=final, gold=gold, status=status):
                self.assertEqual(parse_and_verify(final, gold).status, status)
        self.assertEqual(
            parse_and_verify("The final answer is $12$.", "12", truncated=True).status,
            "correct",
        )
        self.assertEqual(PARSER_ID, "math-v3")
        self.assertEqual(
            PARSER_CONFIG_HASH,
            "b303dece0387564391eb0b5433c6f3fb2f6290b57b15424d5041fbfa6a88b16a",
        )

        with patch("parser.verify", side_effect=RuntimeError("boom")):
            result = parse_and_verify(r"\boxed{42}", "42")
        self.assertEqual(result.status, "verification_error")
        self.assertFalse(result.is_correct)

        with (
            patch("parser.verify", side_effect=TimeoutException("boom")),
            self.assertRaises(TimeoutException),
        ):
            parse_and_verify(r"\boxed{42}", "42")

        class Unprintable:
            def __bool__(self):
                return True

            def __str__(self):
                raise ValueError("too large to normalize")

        with (
            patch("parser.parse", side_effect=[[42], Unprintable()]),
            self.assertRaisesRegex(ValueError, "too large to normalize"),
        ):
            parse_and_verify(r"\boxed{42}", "42")

        with self.assertRaisesRegex(ValueError, "gold answer must be non-empty"):
            parse_and_verify(r"\boxed{42}", "")

    def test_dual_strict_soft_contract(self):
        boxed = parse_dual_and_verify(
            r"work \boxed{42} trailing \boxed{\frac{1}{", "42", truncated=True
        )
        self.assertIs(boxed.strict, boxed.soft)
        self.assertEqual(boxed.strict.status, "correct")

        unboxed = parse_dual_and_verify(
            "The final answer is $42$.", "42", truncated=True
        )
        self.assertEqual(unboxed.strict.status, "no_candidate")
        self.assertEqual(unboxed.soft.status, "correct")
        self.assertEqual(unboxed.soft.extraction_rule, "full_final_text")

        wrong_box = parse_dual_and_verify(
            r"\boxed{41}. On reflection the answer is $42$.", "42"
        )
        self.assertIs(wrong_box.strict, wrong_box.soft)
        self.assertEqual(wrong_box.strict.status, "incorrect")

        empty = parse_dual_and_verify("", "42")
        self.assertIs(empty.strict, empty.soft)
        self.assertEqual(empty.strict.status, "no_candidate")

        fbox = parse_dual_and_verify(r"\fbox{42}", "42")
        self.assertEqual(fbox.strict.status, "no_candidate")
        self.assertEqual(fbox.soft.status, "parse_error")
        self.assertEqual(DUAL_PARSER_ID, "math-v4-dual")
        self.assertEqual(
            DUAL_PARSER_CONFIG_HASH,
            "f036179d2bbe55689e5da60878b7665745e881550fa98c07f0febf978c50f26a",
        )

        piecewise = r"f(n)=\left\{\begin{array}{cl}0&n=0\\1&n>0\end{array}\right."
        response = r"\boxed{" + piecewise + "}"
        self.assertEqual(
            parse_dual_and_verify(response, piecewise).strict.status,
            "no_candidate",
        )
        v5 = parse_v5_dual_and_verify(response, piecewise)
        self.assertIs(v5.strict, v5.soft)
        self.assertEqual(v5.strict.status, "correct")
        self.assertEqual(parse_v51_dual_and_verify(response, piecewise), v5)
        self.assertEqual(
            parse_v5_dual_and_verify(r"\boxed{\frac{1}{2}", r"\frac{1}{2}").strict.status,
            "no_candidate",
        )
        self.assertEqual(V5_DUAL_PARSER_ID, "math-v5-dual")
        self.assertTrue(V5_DUAL_PARSER_CONFIG_HASH)

        with patch("parser.verify", side_effect=TimeoutException("boom")):
            timeout = parse_v51_dual_and_verify(r"\boxed{42}", "42")
        self.assertEqual(timeout.strict.status, "verification_error")

        class Unprintable:
            def __bool__(self):
                return True

            def __str__(self):
                raise ValueError("too large to normalize")

        with patch("parser.parse", side_effect=[[42], Unprintable()]):
            unprintable = parse_v51_dual_and_verify(r"\boxed{42}", "42")
        self.assertEqual(unprintable.strict.status, "parse_error")
        self.assertEqual(V51_DUAL_PARSER_ID, "math-v5.1-dual")
        self.assertEqual(
            V51_DUAL_PARSER_CONFIG_HASH,
            "e732b5ad06825d23fdbddd293d69f40b7fa9643c4a8c96b3e4baf53269d4b5e2",
        )
        pathological = (
            "38016000000000000**\n524. **20528640000000000 + 20528"
        )
        guarded = parse_v52_dual_and_verify(pathological, "103", truncated=True)
        self.assertEqual(guarded.strict.status, "no_candidate")
        self.assertEqual(guarded.soft.status, "parse_error")
        self.assertIn("TimeoutException", guarded.soft.error)
        self.assertEqual(V52_DUAL_PARSER_ID, "math-v5.2-dual")
        self.assertEqual(
            V52_DUAL_PARSER_CONFIG_HASH,
            "e0b9e40ca45b1844b4a1d31252988bab15083be80615d3e9649b71ffb1f2cc22",
        )

    def test_metrics_and_compare_compatibility(self):
        rows = []
        statuses = (
            "correct",
            "incorrect",
            "no_candidate",
            "parse_error",
            "verification_error",
        )
        for index, status in enumerate(statuses, 1):
            rows.append(
                {
                    "run_id": "a",
                    "dataset": "d",
                    "model": "a",
                    "problem_uid": f"d:{index}",
                    "sample_idx": 0,
                    "is_correct": status == "correct",
                    "status": status,
                    "parser_id": PARSER_ID,
                    "parser_config_hash": PARSER_CONFIG_HASH,
                    "truncated": False,
                    "output_tokens": 2,
                }
            )
        metrics = compute_metrics(rows, [1])
        self.assertEqual(metrics["schema_version"], 2)
        self.assertEqual(metrics["problem_count"], 5)
        self.assertEqual(metrics["k"]["1"]["pass_at_k"], 0.2)
        self.assertEqual(metrics["k"]["1"]["avg_at_k"], 0.2)
        self.assertEqual(metrics["parser_failure_count"], 3)
        self.assertEqual(metrics["parser_failure_rate"], 0.6)
        self.assertEqual(
            metrics["status_rates"],
            {status: 0.2 for status in sorted(statuses)},
        )
        only_correct = compute_metrics([rows[0]], [1])
        self.assertEqual(only_correct["parser_failure_rate"], 0.0)
        self.assertEqual(only_correct["status_rates"]["parse_error"], 0.0)
        self.assertEqual(only_correct["status_rates"]["no_candidate"], 0.0)

        dual_rows = []
        for index, (final, truncated) in enumerate(
            (
                (r"\boxed{42}", False),
                ("The final answer is $42$.", True),
                ("", True),
            ),
            1,
        ):
            result = parse_dual_and_verify(final, "42", truncated=truncated)
            dual_rows.append(
                {
                    "run_id": "dual",
                    "dataset": "d",
                    "model": "m",
                    "problem_uid": f"d:dual-{index}",
                    "sample_idx": 0,
                    "final_text": final,
                    "truncated": truncated,
                    "output_tokens": 2,
                    **asdict(result.strict),
                    "soft": asdict(result.soft),
                    "parser_id": DUAL_PARSER_ID,
                    "parser_config_hash": DUAL_PARSER_CONFIG_HASH,
                }
            )
        dual_metrics = compute_metrics(dual_rows, [1])
        self.assertEqual(dual_metrics["schema_version"], 3)
        self.assertEqual(dual_metrics["strict"]["accuracy"], 1 / 3)
        self.assertEqual(dual_metrics["soft"]["accuracy"], 2 / 3)
        self.assertEqual(dual_metrics["complete_box_count"], 1)
        self.assertEqual(
            dual_metrics["no_candidate_counts"],
            {"empty_final_text": 1, "no_complete_box": 1},
        )
        self.assertEqual(dual_metrics["soft_recovery_count"], 1)
        self.assertEqual(len(dual_metrics["interaction_counts"]), 3)

        with self.assertRaisesRegex(ValueError, "duplicate parsed sample key"):
            compute_metrics([*dual_rows, dual_rows[0]], [1])
        mixed_hash = {
            **dual_rows[0],
            "problem_uid": "d:mixed-hash",
            "parser_config_hash": "different",
        }
        with self.assertRaisesRegex(ValueError, "mixed parser_config_hash"):
            compute_metrics([dual_rows[0], mixed_hash], [1])
        with self.assertRaisesRegex(ValueError, "positive integers"):
            compute_metrics(dual_rows, [0])

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            def write_run(name, solved, decode=None):
                path = directory / f"{name}.jsonl"
                with path.open("w", encoding="utf-8") as handle:
                    for uid in ("d:1", "d:2"):
                        row = {
                            **rows[0],
                            "run_id": name,
                            "model": name,
                            "problem_uid": uid,
                            "is_correct": uid in solved,
                            "gold_hash": uid,
                            "status": "correct" if uid in solved else "incorrect",
                            "parser_config_hash": "p",
                            "decode": decode or {"temperature": 0},
                        }
                        handle.write(json.dumps(row) + "\n")
                return path
            base = load_run(write_run("base", {"d:1"}), 1)
            target = load_run(write_run("target", {"d:1", "d:2"}), 1)
            result = compare(base, target)
            self.assertEqual(result["acquisition"], 1)
            incompatible = load_run(write_run("bad", {"d:1"}, {"temperature": 1}), 1)
            with self.assertRaisesRegex(ValueError, "decode semantics"):
                compare(base, incompatible)


class ConfigFilesTest(unittest.TestCase):
    def test_repository_configs_are_valid_and_reference_omits_optional_sampling(self):
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "configs").glob("*.yaml")):
            with self.subTest(path=path.name):
                load_config(path)
        request = sampling_request(load_config(root / "configs/reference_vllm.yaml")["decode"])
        for key in (
            "top_k",
            "min_p",
            "min_output_tokens",
            "stop",
            "stop_token_ids",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        ):
            self.assertNotIn(key, request)


class PipelineTest(unittest.TestCase):
    def test_banded_stop_resume_expand_and_sample_limit(self):
        generated = []

        class FakeBackend:
            request_batch_size = 2
            stop_after_batch = False

            def generate(self, prompt, decode, sample_idx):
                generated.append((prompt[-1]["content"], sample_idx))
                return InferenceResult(
                    raw_text="\\boxed{4}",
                    reasoning_text=None,
                    final_text="\\boxed{4}",
                    thinking_status="non_thinking_ok",
                    finish_reason="stop",
                    prompt_tokens=3,
                    output_tokens=2,
                    response_id=str(len(generated)),
                    resolved_request={},
                )

            def generate_batch(self, requests):
                results = [self.generate(**request) for request in requests]
                if self.stop_after_batch:
                    self.stop_after_batch = False
                    signal.raise_signal(signal.SIGTERM)
                return results

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text(
                '{"id":"1","problem":"first","answer":"4"}\n'
                '{"id":"2","problem":"second","answer":"4"}\n',
                encoding="utf-8",
            )
            prompt_dir = root / "prompt"
            prompt_dir.mkdir()
            (prompt_dir / "00-user.txt").write_text("{{problem}}", encoding="utf-8")
            config = {
                "dataset": {"name": "tiny", "path": str(dataset), "limit": 2},
                "model": {
                    "name": "fake",
                    "backend": "vllm",
                    "path": "unused",
                    "input_mode": "chat",
                    "thinking": False,
                },
                "prompt": str(prompt_dir),
                "decode": decode_config(samples_per_problem=2),
                "output": {
                    "root": str(root / "runs"),
                    "shard_size": 100,
                    "compression": "none",
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            preflight = {
                "applicable": True,
                "configured": True,
                "supported": True,
                "requested": False,
                "chat_template_sha256": "template-hash",
            }
            backend = FakeBackend()
            with (
                patch("evaluate.preflight_model", return_value=preflight),
                patch("evaluate.create_backend", return_value=backend),
            ):
                run_dir = run(config_path, "expand", sample_band_size=2)
                self.assertEqual(
                    generated,
                    [("first", 0), ("first", 1), ("second", 0), ("second", 1)],
                )
                config["decode"]["samples_per_problem"] = 4
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                run(config_path, "expand", True, sample_band_size=2)
                self.assertEqual(
                    generated[-4:],
                    [("first", 2), ("first", 3), ("second", 2), ("second", 3)],
                )
                config["decode"]["samples_per_problem"] = 3
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "below existing"):
                    run(config_path, "expand", True, sample_band_size=2)
                config["decode"]["samples_per_problem"] = 4
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

                raw_path = next(run_dir.glob("raw/**/*.jsonl"))
                original = raw_path.read_bytes()
                tampered = list(read_jsonl(raw_path))
                tampered[0]["final_text"] = "\\boxed{5}"
                raw_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in tampered),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "raw content changed"):
                    run(config_path, "expand", True, sample_band_size=2)
                raw_path.write_bytes(original)

                backend.stop_after_batch = True
                stopped = run(config_path, "stopped", sample_band_size=2)
                stopped_manifest = json.loads(
                    (stopped / "manifests/run.json").read_text()
                )
                self.assertEqual(stopped_manifest["status"], "stopped")
                self.assertEqual(stopped_manifest["sample_count"], 2)
                self.assertEqual(stopped_manifest["common_sample_depth"], 0)
                self.assertNotIn("process_id", stopped_manifest)
                self.assertFalse(list(stopped.glob("raw/**/*.inprogress")))
                run(config_path, "stopped", True, sample_band_size=2)

                with patch(
                    "evaluate.create_backend", side_effect=RuntimeError("backend failed")
                ):
                    with self.assertRaisesRegex(RuntimeError, "backend failed"):
                        run(config_path, "failed", sample_band_size=2)
                failed_manifest = json.loads(
                    (root / "runs/failed/manifests/run.json").read_text()
                )
                self.assertEqual(failed_manifest["status"], "failed")
                self.assertNotIn("process_id", failed_manifest)

            manifest = json.loads((run_dir / "manifests/run.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["target_samples_per_problem"], 4)
            self.assertEqual(manifest["common_sample_depth"], 4)
            self.assertEqual(len(scan_raw(run_dir / "raw")), 8)
            parsed, _ = replay(run_dir, [1, 2], sample_limit=2)
            self.assertEqual(len(list(read_jsonl(parsed))), 4)

    def test_generate_resume_and_replay_ready_raw(self):
        batch_calls = []

        class FakeBackend:
            request_batch_size = 2

            def generate(self, prompt, decode, sample_idx):
                return InferenceResult(
                    raw_text="\\boxed{4}",
                    reasoning_text=None,
                    final_text="\\boxed{4}",
                    thinking_status="non_thinking_ok",
                    finish_reason="stop",
                    prompt_tokens=3,
                    output_tokens=2,
                    response_id=f"r{sample_idx}",
                    resolved_request={
                        "messages": prompt,
                        "sampling": sampling_request(decode, sample_idx),
                    },
                )

            def generate_batch(self, requests):
                batch_calls.append(len(requests))
                return [self.generate(**request) for request in requests]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text(
                '{"id":"1","problem":"2+2?","answer":"4"}\n', encoding="utf-8"
            )
            prompt_dir = root / "prompt"
            prompt_dir.mkdir()
            (prompt_dir / "00-system.txt").write_text("rules", encoding="utf-8")
            user_prompt = prompt_dir / "01-user.txt"
            user_prompt.write_text("{{problem}}", encoding="utf-8")
            config = {
                "dataset": {
                    "name": "tiny",
                    "path": str(dataset),
                    "limit": 1,
                },
                "model": {
                    "name": "fake",
                    "backend": "vllm",
                    "path": "unused",
                    "input_mode": "chat",
                    "thinking": False,
                },
                "prompt": str(prompt_dir),
                "decode": decode_config(max_output_tokens=8, samples_per_problem=2),
                "output": {
                    "root": str(root / "runs"),
                    "shard_size": 10,
                    "compression": "none",
                },
            }
            config_path = root / "config.yaml"
            config_text = yaml.safe_dump(config)
            config_path.write_text(config_text, encoding="utf-8")
            preflight = {
                "applicable": True,
                "configured": True,
                "supported": True,
                "requested": False,
                "chat_template_sha256": "template-hash",
            }
            with (
                patch("evaluate.preflight_model", return_value=preflight),
                patch("evaluate.create_backend", return_value=FakeBackend()) as create,
            ):
                run_dir = run(
                    config_path,
                    "run",
                    False,
                    request_batch_size_override=1,
                    fsync_every_override=2,
                )
                raw_dir = run_dir / "raw" / "fake" / "tiny"
                sealed = raw_dir / "part-00000.jsonl"
                active = raw_dir / "part-00000.jsonl.inprogress"
                sealed.rename(active)
                with self.assertRaisesRegex(ValueError, "unsealed raw shards"):
                    replay(run_dir, [1, 2])
                run(config_path, "run", True)
                self.assertFalse(active.exists())
                self.assertTrue(sealed.exists())
                config["decode"]["temperature"] = 0.1
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "generation spec"):
                    run(config_path, "run", True)
                self.assertEqual(create.call_count, 1)
                config_path.write_text(config_text, encoding="utf-8")
                dataset.write_text(
                    '{"id":"1","problem":"2+2?","answer":"5"}\n', encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "generation spec"):
                    run(config_path, "run", True)
                dataset.write_text(
                    '{"id":"1","problem":"2+2?","answer":"4"}\n', encoding="utf-8"
                )
                user_prompt.write_text("Question: {{problem}}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "prompt does not match"):
                    run(config_path, "run", True)
                user_prompt.write_text("{{problem}}", encoding="utf-8")
                snapshot_path = run_dir / "manifests/config.yaml"
                snapshot_path.write_text("# tampered\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "config snapshot"):
                    run(config_path, "run", True)
                snapshot_path.write_text(config_text, encoding="utf-8")
                prompt_snapshot = run_dir / "manifests/prompt.json"
                prompt_snapshot_text = prompt_snapshot.read_text(encoding="utf-8")
                prompt_snapshot.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "prompt snapshot"):
                    run(config_path, "run", True)
                prompt_snapshot.write_text(prompt_snapshot_text, encoding="utf-8")

            self.assertEqual(batch_calls, [])
            self.assertEqual(create.return_value.request_batch_size, 1)
            manifest = json.loads((run_dir / "manifests/run.json").read_text())
            self.assertEqual(manifest["schema_version"], 3)
            self.assertEqual(manifest["thinking_preflight"], preflight)
            self.assertEqual(
                manifest["operational_override_history"][0],
                {
                    "started_at": manifest["operational_override_history"][0][
                        "started_at"
                    ],
                    "request_batch_size": 1,
                    "fsync_every": 2,
                },
            )
            self.assertTrue(manifest["prompt_sha256"])
            self.assertNotIn("generation_spec", manifest)
            self.assertNotIn("config_path", manifest)
            self.assertEqual(
                (run_dir / "manifests/config.yaml").read_text(encoding="utf-8"),
                config_text,
            )
            prompt_snapshot = json.loads(
                (run_dir / "manifests/prompt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(prompt_snapshot["messages"][1]["content"], "{{problem}}")
            raw = scan_raw(run_dir / "raw")
            self.assertEqual(len(raw), 2)
            self.assertTrue(all(row.row["final_text"] == "\\boxed{4}" for row in raw.values()))
            self.assertTrue(all(row.row["thinking"] is False for row in raw.values()))
            self.assertTrue(
                all(
                    row.row["resolved_request"]["messages"][1]["content"] == "2+2?"
                    for row in raw.values()
                )
            )
            self.assertTrue(all("profile" not in row.row for row in raw.values()))
            self.assertTrue(
                all("answer_contract" not in row.row for row in raw.values())
            )
            self.assertTrue(all("allow_last_number" not in row.row for row in raw.values()))

            parsed_path, metrics_path = replay(run_dir, [1, 2])
            parsed_rows = list(read_jsonl(parsed_path))
            self.assertEqual(len(parsed_rows), 2)
            self.assertTrue(
                all(row["parser_id"] == V52_DUAL_PARSER_ID for row in parsed_rows)
            )
            self.assertTrue(
                all(
                    row["status"] == row["soft"]["status"] == "correct"
                    for row in parsed_rows
                )
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["schema_version"], 3)
            self.assertEqual(metrics["strict"]["k"]["2"]["pass_at_k"], 1.0)
            self.assertEqual(metrics["soft_recovery_count"], 0)
            self.assertEqual(load_run(parsed_path, 2)["solved"], {"tiny:1"})

            v51_parsed, _ = replay(run_dir, [1, 2], V51_DUAL_PARSER_ID)
            self.assertTrue(
                all(
                    row["parser_id"] == V51_DUAL_PARSER_ID
                    for row in read_jsonl(v51_parsed)
                )
            )

    def test_partition_resume_merge_and_replay(self):
        generated = []

        class FakeBackend:
            def generate(self, prompt, decode, sample_idx):
                generated.append((prompt[-1]["content"], sample_idx))
                return InferenceResult(
                    raw_text="\\boxed{4}",
                    reasoning_text=None,
                    final_text="\\boxed{4}",
                    thinking_status="non_thinking_ok",
                    finish_reason="stop",
                    prompt_tokens=3,
                    output_tokens=2,
                    response_id=str(len(generated)),
                    resolved_request={},
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text(
                "".join(
                    json.dumps({"id": str(index), "problem": "2+2?", "answer": "4"})
                    + "\n"
                    for index in range(4)
                ),
                encoding="utf-8",
            )
            prompt_dir = root / "prompt"
            prompt_dir.mkdir()
            (prompt_dir / "00-user.txt").write_text("{{problem}}", encoding="utf-8")
            config = {
                "dataset": {"name": "tiny", "path": str(dataset), "limit": 4},
                "model": {
                    "name": "fake",
                    "backend": "vllm",
                    "path": "unused",
                    "input_mode": "chat",
                    "thinking": False,
                },
                "prompt": str(prompt_dir),
                "decode": decode_config(samples_per_problem=100),
                "output": {
                    "root": str(root / "runs"),
                    "shard_size": 250,
                    "compression": "none",
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            preflight = {
                "applicable": True,
                "configured": True,
                "supported": True,
                "requested": False,
                "chat_template_sha256": "template-hash",
            }
            with (
                patch("evaluate.preflight_model", return_value=preflight),
                patch("evaluate.create_backend", return_value=FakeBackend()),
            ):
                shard0 = run(config_path, "parallel", shard_index=0, shard_count=2)
                shard1 = run(config_path, "parallel", shard_index=1, shard_count=2)
                sealed = next((shard1 / "raw").glob("**/part-*.jsonl"))
                rows = list(read_jsonl(sealed))
                active = sealed.with_suffix(".jsonl.inprogress")
                active.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows[:-1]),
                    encoding="utf-8",
                )
                sealed.unlink()
                shard_manifest_path = shard1 / "manifests/run.json"
                shard_manifest = json.loads(shard_manifest_path.read_text())
                shard_manifest.pop("raw_content_sha256")
                shard_manifest["status"] = "running"
                shard_manifest_path.write_text(
                    json.dumps(shard_manifest), encoding="utf-8"
                )
                run(config_path, "parallel", True, shard_index=1, shard_count=2)

            self.assertEqual(len(generated), 401)
            keys0 = set(scan_raw(shard0 / "raw"))
            keys1 = set(scan_raw(shard1 / "raw"))
            self.assertFalse(keys0 & keys1)
            self.assertEqual(len(keys0 | keys1), 400)
            self.assertEqual(
                {key[2] for key in keys0}, {"tiny:0", "tiny:2"}
            )
            self.assertEqual(
                {key[2] for key in keys1}, {"tiny:1", "tiny:3"}
            )

            with self.assertRaisesRegex(ValueError, "incomplete partitions"):
                merge(root / "runs/missing", [shard0])
            existing = root / "runs/existing"
            existing.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                merge(existing, [shard0, shard1])

            raw_path = next((shard0 / "raw").glob("**/part-*.jsonl"))
            original = raw_path.read_bytes()
            tampered = list(read_jsonl(raw_path))
            tampered[0]["final_text"] = "\\boxed{5}"
            raw_path.write_text(
                "".join(json.dumps(row) + "\n" for row in tampered),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "raw content hash mismatch"):
                merge(root / "runs/tampered", [shard0, shard1])
            raw_path.write_bytes(original)

            shard_manifest_path = shard1 / "manifests/run.json"
            shard_manifest = json.loads(shard_manifest_path.read_text())
            shard_manifest.pop("common_sample_depth")
            shard_manifest_path.write_text(
                json.dumps(shard_manifest), encoding="utf-8"
            )
            merged = merge(root / "runs/parallel", [shard1, shard0])
            parsed_path, metrics_path = replay(merged, [1, 100])
            self.assertEqual(len(list(read_jsonl(parsed_path))), 400)
            self.assertEqual(json.loads(metrics_path.read_text())["sample_count"], 400)
            manifest = json.loads((merged / "manifests/run.json").read_text())
            self.assertEqual(manifest["run_id"], "parallel")
            self.assertEqual(manifest["sample_count"], 400)
            self.assertEqual(manifest["target_sample_count"], 400)
            self.assertEqual(manifest["common_sample_depth"], 100)
            self.assertNotIn("process_id", manifest)
            self.assertNotIn("partition", manifest)
            self.assertNotIn("environment", manifest)
            self.assertEqual(len(manifest["worker_environments"]), 2)
            self.assertTrue(manifest["raw_content_sha256"])


if __name__ == "__main__":
    unittest.main()
