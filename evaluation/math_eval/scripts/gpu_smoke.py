#!/usr/bin/env python3
"""Minimal real-checkpoint smoke for completion/chat and TP validation."""

from __future__ import annotations

import argparse
import json

from inference import VllmBackend, preflight_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--input-mode", choices=["completion", "chat"], required=True)
    parser.add_argument("--thinking", choices=["true", "false"])
    parser.add_argument("--tp", type=int, choices=[1, 2], required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--gdn-prefill-backend")
    args = parser.parse_args()

    model = {
        "name": args.name,
        "backend": "vllm",
        "path": args.path,
        "input_mode": args.input_mode,
        "runtime": {
            "tensor_parallel_size": args.tp,
            "dtype": "bfloat16",
            "max_context_tokens": 4096,
            "max_num_seqs": 1,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": True,
        },
    }
    if args.thinking is not None:
        model["thinking"] = args.thinking == "true"
    if args.language_model_only:
        model["runtime"]["language_model_only"] = True
    if args.gdn_prefill_backend:
        model["runtime"]["gdn_prefill_backend"] = args.gdn_prefill_backend

    thinking_preflight = preflight_model(model)
    backend = VllmBackend(model, thinking_preflight)
    results = []
    decode = {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_output_tokens": 64,
        "samples_per_problem": 1,
    }
    prompt = (
        [
            {"role": "system", "content": "Put the final answer in \\boxed{}."},
            {"role": "user", "content": "Only answer this: 1+1=?"},
        ]
        if args.input_mode == "chat"
        else "Only answer this: 1+1=?"
    )
    result = backend.generate(prompt, decode, 0)
    results.append(
        {
            "thinking": model.get("thinking"),
            "thinking_preflight": thinking_preflight,
            "thinking_status": result.thinking_status,
            "finish_reason": result.finish_reason,
            "raw_nonempty": bool(result.raw_text),
            "reasoning_present": result.reasoning_text is not None,
            "final_nonempty": bool(result.final_text),
            "output_tokens": result.output_tokens,
            "chat_template_kwargs": result.resolved_request.get("chat_template_kwargs"),
        }
    )
    print(json.dumps({"model": args.name, "tp": args.tp, "results": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
