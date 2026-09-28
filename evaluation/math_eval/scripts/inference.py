#!/usr/bin/env python3
"""vLLM and OpenAI-compatible adapters with one explicit request contract."""

from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from io_utils import stable_hash


DECODE_FIELDS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "max_output_tokens",
    "min_output_tokens",
    "stop",
    "stop_token_ids",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "samples_per_problem",
}
REQUIRED_DECODE_FIELDS = {
    "temperature",
    "top_p",
    "seed",
    "max_output_tokens",
    "samples_per_problem",
}
NON_STANDARD_API_FIELDS = {
    "top_k",
    "min_p",
    "min_output_tokens",
    "stop_token_ids",
    "repetition_penalty",
}
SOFT_THINKING_MARKERS = ("/think", "/no_think", "/nothink")
PROBLEM_MARKER = "{{problem}}"
MESSAGE_FILE = re.compile(r"^(\d+)-(system|user|assistant)\.txt$")
COMMON_MODEL_FIELDS = {"name", "backend", "input_mode", "thinking", "runtime"}
VLLM_MODEL_FIELDS = COMMON_MODEL_FIELDS | {"path", "tokenizer_path"}
OPENAI_MODEL_FIELDS = COMMON_MODEL_FIELDS | {
    "endpoint",
    "api_model",
    "api_key_env",
    "reasoning_parameter",
    "capabilities",
}
VLLM_RUNTIME_FIELDS = {
    "tensor_parallel_size",
    "dtype",
    "max_context_tokens",
    "gpu_memory_utilization",
    "max_num_seqs",
    "enforce_eager",
    "disable_custom_all_reduce",
    "language_model_only",
    "gdn_prefill_backend",
    "request_batch_size",
}
OPENAI_RUNTIME_FIELDS = {"timeout_seconds", "max_retries"}


@dataclass(frozen=True)
class InferenceResult:
    raw_text: str
    reasoning_text: str | None
    final_text: str
    thinking_status: str
    finish_reason: str
    prompt_tokens: int | None
    output_tokens: int | None
    response_id: str | None
    resolved_request: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def reject_soft_thinking(text: str) -> None:
    lowered = text.lower()
    if any(marker in lowered for marker in SOFT_THINKING_MARKERS):
        raise ValueError("soft thinking commands are forbidden; use model.thinking")


def _prompt_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: prompt must be UTF-8") from exc


def load_prompt(path_value: str, input_mode: str) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    if input_mode == "completion":
        if not path.is_file():
            raise ValueError(f"completion prompt must be a file: {path}")
        prompt = {"mode": "completion", "text": _prompt_text(path)}
    else:
        if not path.is_dir():
            raise ValueError(f"chat prompt must be a directory: {path}")
        indexed = []
        seen = set()
        for item in path.iterdir():
            match = MESSAGE_FILE.fullmatch(item.name)
            if not item.is_file() or match is None:
                raise ValueError(
                    f"{item}: chat prompt files must match <index>-<role>.txt"
                )
            index = int(match.group(1))
            if index in seen:
                raise ValueError(f"{path}: duplicate message index {index}")
            seen.add(index)
            indexed.append(
                (index, {"role": match.group(2), "content": _prompt_text(item)})
            )
        messages = [message for _, message in sorted(indexed)]
        if not messages:
            raise ValueError(f"{path}: chat prompt directory is empty")
        if messages[-1]["role"] != "user":
            raise ValueError("chat prompt must end with a user message")
        prompt = {"mode": "chat", "messages": messages}

    contents = (
        [prompt["text"]]
        if prompt["mode"] == "completion"
        else [message["content"] for message in prompt["messages"]]
    )
    marker_count = sum(content.count(PROBLEM_MARKER) for content in contents)
    if marker_count != 1:
        raise ValueError(f"prompt must contain {PROBLEM_MARKER!r} exactly once")
    for content in contents:
        reject_soft_thinking(content)
    return prompt


def render_prompt(prompt: dict[str, Any], problem: str) -> str | list[dict[str, str]]:
    reject_soft_thinking(problem)
    if prompt["mode"] == "completion":
        rendered = prompt["text"].replace(PROBLEM_MARKER, problem)
        reject_soft_thinking(rendered)
        return rendered
    messages = [
        {
            "role": message["role"],
            "content": message["content"].replace(PROBLEM_MARKER, problem),
        }
        for message in prompt["messages"]
    ]
    for message in messages:
        reject_soft_thinking(message["content"])
    return messages


def validate_decode(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("decode must be an object")
    unknown = set(value) - DECODE_FIELDS
    if unknown:
        raise ValueError(f"decode has unknown fields: {sorted(unknown)}")
    missing = REQUIRED_DECODE_FIELDS - value.keys()
    if missing:
        raise ValueError(f"decode is missing fields: {sorted(missing)}")
    nulls = {key for key, item in value.items() if item is None}
    if nulls:
        raise ValueError(f"decode fields cannot be null: {sorted(nulls)}")
    decode = json.loads(json.dumps(value))
    decode["temperature"] = float(decode["temperature"])
    decode["top_p"] = float(decode["top_p"])
    decode["seed"] = int(decode["seed"])
    decode["max_output_tokens"] = int(decode["max_output_tokens"])
    decode["samples_per_problem"] = int(decode["samples_per_problem"])
    if decode["temperature"] < 0:
        raise ValueError("temperature must be >= 0")
    if not 0 < decode["top_p"] <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if decode["max_output_tokens"] < 1:
        raise ValueError("max_output_tokens must be >= 1")
    if decode["samples_per_problem"] < 1:
        raise ValueError("samples_per_problem must be >= 1")
    if "top_k" in decode:
        decode["top_k"] = int(decode["top_k"])
        if decode["top_k"] < 0:
            raise ValueError("top_k must be >= 0")
    if "min_p" in decode:
        decode["min_p"] = float(decode["min_p"])
        if not 0 <= decode["min_p"] <= 1:
            raise ValueError("min_p must be in [0, 1]")
    if "min_output_tokens" in decode:
        decode["min_output_tokens"] = int(decode["min_output_tokens"])
        if decode["min_output_tokens"] < 0:
            raise ValueError("min_output_tokens must be >= 0")
    if "stop" in decode and not (
        isinstance(decode["stop"], str)
        or isinstance(decode["stop"], list)
        and all(isinstance(item, str) for item in decode["stop"])
    ):
        raise ValueError("stop must be a string or list of strings")
    if "stop_token_ids" in decode and not (
        isinstance(decode["stop_token_ids"], list)
        and decode["stop_token_ids"]
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in decode["stop_token_ids"]
        )
    ):
        raise ValueError("stop_token_ids must be a non-empty list of non-negative integers")
    for key in ("repetition_penalty", "presence_penalty", "frequency_penalty"):
        if key in decode:
            decode[key] = float(decode[key])
    return decode


def sampling_request(decode: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    decode = validate_decode(decode)
    request = {
        key: decode[key]
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "min_output_tokens",
            "stop",
            "stop_token_ids",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        )
        if key in decode
    }
    request["seed"] = int(decode["seed"] if seed is None else seed)
    request["max_tokens"] = int(decode["max_output_tokens"])
    return request


def validate_model(model: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    backend = model.get("backend")
    if backend not in {"vllm", "openai"}:
        raise ValueError("model.backend must be vllm or openai")
    allowed = VLLM_MODEL_FIELDS if backend == "vllm" else OPENAI_MODEL_FIELDS
    unknown = set(model) - allowed
    if unknown:
        raise ValueError(f"model has unknown fields: {sorted(unknown)}")
    nulls = {key for key, item in model.items() if item is None}
    if nulls:
        raise ValueError(f"model fields cannot be null: {sorted(nulls)}")
    if not model.get("name"):
        raise ValueError("model requires name")
    if backend == "vllm" and not model.get("path"):
        raise ValueError("vLLM model requires path")
    if backend == "openai":
        for key in ("endpoint", "api_model", "api_key_env"):
            if not model.get(key):
                raise ValueError(f"OpenAI model requires {key}")
    input_mode = model.get("input_mode")
    if input_mode not in {"completion", "chat"}:
        raise ValueError("model.input_mode must be completion or chat")
    if "thinking" in model and not isinstance(model["thinking"], bool):
        raise ValueError("model.thinking must be true or false")
    if input_mode == "completion" and "thinking" in model:
        raise ValueError("completion models cannot set thinking")
    runtime = model.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("model.runtime must be an object")
    runtime_allowed = VLLM_RUNTIME_FIELDS if backend == "vllm" else OPENAI_RUNTIME_FIELDS
    runtime_unknown = set(runtime) - runtime_allowed
    if runtime_unknown:
        raise ValueError(f"model.runtime has unknown fields: {sorted(runtime_unknown)}")
    runtime_nulls = {key for key, item in runtime.items() if item is None}
    if runtime_nulls:
        raise ValueError(f"model.runtime fields cannot be null: {sorted(runtime_nulls)}")
    capabilities = model.get("capabilities", [])
    if not isinstance(capabilities, list) or not all(
        isinstance(item, str) for item in capabilities
    ):
        raise ValueError("model.capabilities must be a list of strings")
    validated = json.loads(json.dumps(model))
    if "thinking" in validated and backend == "openai":
        resolved_thinking_request(validated)
    return validated


def resolved_thinking_request(model: dict[str, Any]) -> dict[str, Any]:
    if "thinking" not in model:
        return {}
    enabled = model["thinking"]
    if model["backend"] == "vllm":
        # `enable_thinking` is a Qwen chat-template convention, not a
        # vLLM-wide reasoning switch. Other protocols need a separate adapter.
        return {"chat_template_kwargs": {"enable_thinking": enabled}}
    parameter = model.get("reasoning_parameter")
    if not parameter or parameter not in set(model.get("capabilities", [])):
        raise ValueError("provider must declare reasoning_parameter capability")
    value: Any = enabled
    for key in reversed(parameter.split(".")):
        value = {key: value}
    return value


def thinking_template_preflight(tokenizer, model: dict[str, Any]) -> dict[str, Any]:
    messages = [{"role": "user", "content": "Reply with one word."}]
    tokenizer_path = model.get("tokenizer_path", model["path"])
    try:
        rendered_true = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        rendered_false = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception as exc:
        raise ValueError(f"cannot render chat template for {tokenizer_path}") from exc
    supported = rendered_true != rendered_false
    configured = "thinking" in model
    if supported and not configured:
        raise ValueError(
            f"{tokenizer_path} supports enable_thinking; model.thinking must be explicit"
        )
    if not supported and configured:
        raise ValueError(
            f"{tokenizer_path} does not support enable_thinking; omit model.thinking"
        )
    template = getattr(tokenizer, "chat_template", "")
    return {
        "applicable": True,
        "configured": configured,
        "supported": supported,
        "requested": model.get("thinking") if configured else None,
        "chat_template_sha256": stable_hash(
            template if isinstance(template, (str, dict, list)) else str(template)
        ),
    }


def preflight_model(model: dict[str, Any]) -> dict[str, Any]:
    model = validate_model(model)
    configured = "thinking" in model
    if model["backend"] != "vllm" or model["input_mode"] != "chat":
        return {
            "applicable": False,
            "configured": configured,
            "supported": None,
            "requested": model.get("thinking") if configured else None,
            "chat_template_sha256": None,
        }
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model.get("tokenizer_path", model["path"]),
        trust_remote_code=True,
    )
    return thinking_template_preflight(tokenizer, model)


# This adapter only supports Qwen-style `<think>...</think>` output.
def normalize_local_text(
    raw_text: str,
    thinking: bool | None,
    finish_reason: str,
) -> tuple[str | None, str, str]:
    if thinking is None:
        return None, raw_text.strip(), "not_applicable"

    before, separator, after = raw_text.partition("</think>")
    if thinking:
        if separator:
            reasoning = before.removeprefix("<think>").strip()
            return reasoning or None, after.strip(), "thinking_completed"
        reasoning = raw_text.removeprefix("<think>").strip() or None
        status = "thinking_truncated" if finish_reason == "length" else "thinking_format_error"
        return reasoning, "", status

    if separator:
        reasoning = before.removeprefix("<think>").strip()
        status = "non_thinking_violation" if reasoning else "non_thinking_ok"
        return reasoning or None, after.strip(), status
    if "<think>" in raw_text:
        _, _, reasoning_text = raw_text.partition("<think>")
        reasoning = reasoning_text.strip() or None
        status = "non_thinking_violation" if reasoning else "non_thinking_ok"
        return reasoning, "", status
    return None, raw_text.strip(), "non_thinking_ok"


class VllmBackend:
    def __init__(
        self,
        model: dict[str, Any],
        thinking_preflight_result: dict[str, Any] | None = None,
    ):
        self.model = validate_model(model)
        self.thinking_preflight = (
            thinking_preflight_result
            if thinking_preflight_result is not None
            else preflight_model(self.model)
        )
        runtime = self.model.get("runtime", {})
        self.request_batch_size = int(runtime.get("request_batch_size", 1))
        if self.request_batch_size < 1:
            raise ValueError("model.runtime.request_batch_size must be >= 1")
        from vllm import LLM

        kwargs: dict[str, Any] = {
            "model": self.model["path"],
            "tokenizer": self.model.get("tokenizer_path", self.model["path"]),
            "trust_remote_code": True,
            "tensor_parallel_size": int(runtime.get("tensor_parallel_size", 1)),
            "dtype": runtime.get("dtype", "auto"),
            "gpu_memory_utilization": float(runtime.get("gpu_memory_utilization", 0.9)),
        }
        mapping = {"max_context_tokens": "max_model_len"}
        for key in ("max_context_tokens", "max_num_seqs", "enforce_eager", "disable_custom_all_reduce", "language_model_only", "gdn_prefill_backend"):
            if key in runtime:
                kwargs[mapping.get(key, key)] = runtime[key]
        self.engine_request = kwargs
        self.llm = LLM(**kwargs)

    def generate(
        self,
        prompt: str | list[dict[str, str]],
        decode: dict[str, Any],
        sample_idx: int,
    ) -> InferenceResult:
        return self.generate_batch(
            [
                {
                    "prompt": prompt,
                    "decode": decode,
                    "sample_idx": sample_idx,
                }
            ]
        )[0]

    def generate_batch(self, requests: list[dict[str, Any]]) -> list[InferenceResult]:
        from vllm import SamplingParams

        resolved = [
            sampling_request(
                request["decode"],
                int(request["decode"]["seed"]) + request["sample_idx"],
            )
            for request in requests
        ]
        params = [SamplingParams(**item) for item in resolved]
        thinking = self.model.get("thinking") if "thinking" in self.model else None
        if self.model["input_mode"] == "completion":
            prompts = [request["prompt"] for request in requests]
            if not all(isinstance(prompt, str) for prompt in prompts):
                raise ValueError("completion backend requires text prompts")
            outputs = self.llm.generate(prompts, params, use_tqdm=False)
            resolved_requests = [
                {"prompt": prompt, "sampling": sampling}
                for prompt, sampling in zip(prompts, resolved, strict=True)
            ]
        else:
            messages = [request["prompt"] for request in requests]
            if not all(isinstance(message, list) for message in messages):
                raise ValueError("chat backend requires message lists")
            template_kwargs = resolved_thinking_request(self.model).get("chat_template_kwargs")
            chat_kwargs = (
                {"chat_template_kwargs": template_kwargs}
                if template_kwargs is not None
                else {}
            )
            outputs = self.llm.chat(
                messages,
                params,
                use_tqdm=False,
                **chat_kwargs,
            )
            resolved_requests = [
                {
                    "messages": message,
                    "sampling": sampling,
                    **chat_kwargs,
                }
                for message, sampling in zip(messages, resolved, strict=True)
            ]
        results = []
        for request_output, resolved_request in zip(outputs, resolved_requests, strict=True):
            output = request_output.outputs[0]
            raw_text = output.text
            finish_reason = str(output.finish_reason or "unknown")
            reasoning_text, final_text, thinking_status = normalize_local_text(
                raw_text,
                thinking,
                finish_reason,
            )
            results.append(
                InferenceResult(
                    raw_text=raw_text,
                    reasoning_text=reasoning_text,
                    final_text=final_text,
                    thinking_status=thinking_status,
                    finish_reason=finish_reason,
                    prompt_tokens=len(request_output.prompt_token_ids or []),
                    output_tokens=len(output.token_ids or []),
                    response_id=str(request_output.request_id),
                    resolved_request=resolved_request,
                )
            )
        return results


def _retry_delay(retry_after: str | None, fallback: float) -> float:
    if retry_after is None:
        return fallback
    try:
        seconds = float(retry_after)
        return max(0.0, seconds) if seconds < float("inf") else fallback
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return fallback


class OpenAIBackend:
    def __init__(self, model: dict[str, Any]):
        self.model = validate_model(model)
        if self.model["input_mode"] != "chat":
            raise ValueError("OpenAI-compatible backend supports chat only in v1")
        if not os.environ.get(self.model["api_key_env"]):
            raise ValueError(f"missing API key environment variable {self.model['api_key_env']}")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def generate(
        self,
        prompt: list[dict[str, str]],
        decode: dict[str, Any],
        sample_idx: int,
    ) -> InferenceResult:
        if not isinstance(prompt, list):
            raise ValueError("OpenAI-compatible backend requires message lists")
        resolved = sampling_request(decode, int(decode["seed"]) + sample_idx)
        capabilities = set(self.model.get("capabilities", []))
        unsupported = NON_STANDARD_API_FIELDS & resolved.keys() - capabilities
        if unsupported:
            raise ValueError(f"provider does not declare support for: {sorted(unsupported)}")
        payload = {
            "model": self.model["api_model"],
            "messages": prompt,
            **resolved,
        }
        payload.update(resolved_thinking_request(self.model))

        request = urllib.request.Request(
            self.model["endpoint"],
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {os.environ[self.model['api_key_env']]}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        runtime = self.model.get("runtime", {})
        retries = int(runtime.get("max_retries", 3))
        timeout = float(runtime.get("timeout_seconds", 180))
        for attempt in range(retries + 1):
            try:
                with self.opener.open(request, timeout=timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if attempt == retries or exc.code < 500 and exc.code != 429:
                    detail = exc.read().decode("utf-8", errors="replace")[:1000]
                    raise RuntimeError(f"OpenAI-compatible HTTP {exc.code}: {detail}") from exc
                retry_after = exc.headers.get("Retry-After")
                time.sleep(_retry_delay(retry_after, min(2**attempt, 30)))
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, json.JSONDecodeError) as exc:
                if attempt == retries:
                    raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
                time.sleep(min(2**attempt, 30))

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenAI-compatible response has no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("OpenAI-compatible response choice must be an object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("OpenAI-compatible response has no message")
        final_text = message.get("content") or ""
        reasoning_text = message.get("reasoning_content") or message.get("reasoning") or None
        if not isinstance(final_text, str) or reasoning_text is not None and not isinstance(reasoning_text, str):
            raise ValueError("OpenAI-compatible response text fields must be strings")
        if not final_text and not reasoning_text:
            raise ValueError("OpenAI-compatible response is empty")
        raw_text = (
            f"<think>{reasoning_text}</think>\n{final_text}"
            if reasoning_text
            else final_text
        )
        finish_reason = str(choice.get("finish_reason") or "unknown")
        thinking = self.model.get("thinking") if "thinking" in self.model else None
        if thinking is None:
            thinking_status = "not_applicable"
        elif thinking:
            if reasoning_text:
                thinking_status = "thinking_completed"
            elif finish_reason == "length":
                thinking_status = "thinking_truncated"
            else:
                thinking_status = "thinking_format_error"
        else:
            thinking_status = (
                "non_thinking_violation" if reasoning_text else "non_thinking_ok"
            )
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return InferenceResult(
            raw_text=raw_text,
            reasoning_text=reasoning_text,
            final_text=final_text,
            thinking_status=thinking_status,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            response_id=str(body["id"]) if body.get("id") is not None else None,
            resolved_request=payload,
        )


def create_backend(
    model: dict[str, Any],
    thinking_preflight_result: dict[str, Any] | None = None,
):
    return (
        VllmBackend(model, thinking_preflight_result)
        if model["backend"] == "vllm"
        else OpenAIBackend(model)
    )
