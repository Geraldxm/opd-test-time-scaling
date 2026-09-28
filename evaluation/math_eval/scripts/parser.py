#!/usr/bin/env python3
"""Shared Math-Verify answer extraction and equivalence checking."""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import asdict, dataclass

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from math_verify.utils import timeout

from io_utils import stable_hash


PARSER_ID = "math-v3"
DUAL_PARSER_ID = "math-v4-dual"
V5_DUAL_PARSER_ID = "math-v5-dual"
V51_DUAL_PARSER_ID = "math-v5.1-dual"
V52_DUAL_PARSER_ID = "math-v5.2-dual"
NORMALIZATION_TIMEOUT_SECONDS = 5
MATH_VERIFY_VERSION = importlib.metadata.version("math-verify")
EXTRACTION_CONFIG = (
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
)
PARSER_CONFIG_HASH = stable_hash(
    {
        "parser_id": PARSER_ID,
        "math_verify_version": MATH_VERIFY_VERSION,
        "extraction_config": [asdict(config) for config in EXTRACTION_CONFIG],
        "candidate_selection": "last_complete_boxed_else_full_final_text",
        "gold_math_environment": True,
        "raise_on_error": True,
    }
)
DUAL_PARSER_CONFIG_HASH = stable_hash(
    {
        "parser_id": DUAL_PARSER_ID,
        "math_verify_version": MATH_VERIFY_VERSION,
        "extraction_config": [asdict(config) for config in EXTRACTION_CONFIG],
        "strict_candidate_selection": "last_complete_boxed_only",
        "strict_markers": ["boxed"],
        "soft_candidate_selection": "strict_candidate_else_full_final_text",
        "truncated_affects_candidate": False,
        "gold_math_environment": True,
        "raise_on_error": True,
    }
)
V5_DUAL_PARSER_CONFIG_HASH = stable_hash(
    {
        "parser_id": V5_DUAL_PARSER_ID,
        "math_verify_version": MATH_VERIFY_VERSION,
        "extraction_config": [asdict(config) for config in EXTRACTION_CONFIG],
        "strict_candidate_selection": "last_complete_boxed_only",
        "strict_markers": ["boxed"],
        "strict_brace_matching": "balanced_unescaped_braces",
        "soft_candidate_selection": "strict_candidate_else_full_final_text",
        "truncated_affects_candidate": False,
        "gold_math_environment": True,
        "raise_on_error": True,
    }
)
V51_DUAL_PARSER_CONFIG_HASH = stable_hash(
    {
        "parser_id": V51_DUAL_PARSER_ID,
        "math_verify_version": MATH_VERIFY_VERSION,
        "extraction_config": [asdict(config) for config in EXTRACTION_CONFIG],
        "strict_candidate_selection": "last_complete_boxed_only",
        "strict_markers": ["boxed"],
        "strict_brace_matching": "balanced_unescaped_braces",
        "soft_candidate_selection": "strict_candidate_else_full_final_text",
        "truncated_affects_candidate": False,
        "gold_math_environment": True,
        "raise_on_error": True,
        "failure_handling": "timeout_and_normalization_errors_to_verdict",
    }
)
V52_DUAL_PARSER_CONFIG_HASH = stable_hash(
    {
        "parser_id": V52_DUAL_PARSER_ID,
        "math_verify_version": MATH_VERIFY_VERSION,
        "extraction_config": [asdict(config) for config in EXTRACTION_CONFIG],
        "strict_candidate_selection": "last_complete_boxed_only",
        "strict_markers": ["boxed"],
        "strict_brace_matching": "balanced_unescaped_braces",
        "soft_candidate_selection": "strict_candidate_else_full_final_text",
        "truncated_affects_candidate": False,
        "gold_math_environment": True,
        "raise_on_error": True,
        "failure_handling": "timeout_and_normalization_errors_to_verdict",
        "prediction_normalization_timeout_seconds": NORMALIZATION_TIMEOUT_SECONDS,
    }
)


@dataclass(frozen=True)
class Verdict:
    status: str
    is_correct: bool
    candidate_text: str | None
    normalized_prediction: str | None
    normalized_gold: str | None
    extraction_rule: str | None
    error: str | None = None


@dataclass(frozen=True)
class ParseResult:
    status: str
    is_correct: bool
    candidate_text: str | None
    normalized_prediction: str | None
    normalized_gold: str | None
    extraction_rule: str | None
    parser_id: str = PARSER_ID
    error: str | None = None


@dataclass(frozen=True)
class DualParseResult:
    strict: Verdict
    soft: Verdict


def _is_escaped(text: str, position: int) -> bool:
    backslashes = 0
    while position > 0 and text[position - 1] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


def _boxed_at(
    text: str,
    marker: int,
    *,
    ignore_escaped_braces: bool = False,
) -> str | None:
    start = marker
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    for position in range(start, len(text)):
        if text[position] == "{" and (
            not ignore_escaped_braces or not _is_escaped(text, position)
        ):
            depth += 1
        elif text[position] == "}" and (
            not ignore_escaped_braces or not _is_escaped(text, position)
        ):
            depth -= 1
            if depth == 0:
                return text[start + 1 : position].strip()
    return None


def _last_boxed(
    text: str,
    pattern: str,
    *,
    ignore_escaped_braces: bool = False,
) -> str | None:
    candidates = []
    for match in re.finditer(pattern, text):
        value = _boxed_at(
            text,
            match.end(),
            ignore_escaped_braces=ignore_escaped_braces,
        )
        if value is not None:
            candidates.append(value)
    return candidates[-1] if candidates else None


def last_boxed(text: str) -> str | None:
    return _last_boxed(text, r"\\(?:boxed|fbox)")


def last_strict_boxed(text: str) -> str | None:
    return _last_boxed(text, r"\\boxed")


def last_v5_strict_boxed(text: str) -> str | None:
    return _last_boxed(text, r"\\boxed", ignore_escaped_braces=True)


def extract_candidate(final_text: str, truncated: bool = False) -> tuple[str | None, str | None]:
    if not final_text.strip():
        return None, None
    boxed = last_boxed(final_text)
    if boxed is not None:
        return boxed, "last_boxed"
    return final_text.strip(), "full_final_text"


def _math_environment(value: str) -> str:
    stripped = value.strip()
    environments = (("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
    if any(
        stripped.startswith(start) and stripped.endswith(end)
        for start, end in environments
    ):
        return stripped
    return f"${stripped}$"


def _parse_gold(gold: str) -> list:
    if not gold.strip():
        raise ValueError("gold answer must be non-empty")
    try:
        parsed_gold = parse(
            _math_environment(gold),
            extraction_config=EXTRACTION_CONFIG,
            raise_on_error=True,
        )
    except Exception as exc:
        raise ValueError(
            f"gold cannot be parsed: {gold!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if not parsed_gold:
        raise ValueError(f"gold cannot be parsed: {gold!r}")
    return parsed_gold


def _no_candidate(normalized_gold: str) -> Verdict:
    return Verdict(
        status="no_candidate",
        is_correct=False,
        candidate_text=None,
        normalized_prediction=None,
        normalized_gold=normalized_gold,
        extraction_rule=None,
    )


def _verify_candidate(
    candidate: str | None,
    rule: str | None,
    parsed_gold: list,
    normalized_gold: str,
    safe: bool = False,
    normalization_timeout_seconds: int | None = None,
) -> Verdict:
    if candidate is None:
        return _no_candidate(normalized_gold)

    prediction_source = _math_environment(candidate) if rule == "last_boxed" else candidate
    try:
        parsed_prediction = parse(
            prediction_source,
            extraction_config=EXTRACTION_CONFIG,
            raise_on_error=True,
        )
    except Exception as exc:
        return Verdict(
            status="parse_error",
            is_correct=False,
            candidate_text=candidate,
            normalized_prediction=None,
            normalized_gold=normalized_gold,
            extraction_rule=rule,
            error=f"{type(exc).__name__}: {exc}",
        )
    except TimeoutException as exc:
        if not safe:
            raise
        return Verdict(
            status="parse_error",
            is_correct=False,
            candidate_text=candidate,
            normalized_prediction=None,
            normalized_gold=normalized_gold,
            extraction_rule=rule,
            error=f"{type(exc).__name__}: {exc}",
        )
    if not parsed_prediction:
        return Verdict(
            status="parse_error",
            is_correct=False,
            candidate_text=candidate,
            normalized_prediction=None,
            normalized_gold=normalized_gold,
            extraction_rule=rule,
            error="candidate could not be parsed",
        )

    try:
        normalized_prediction = (
            timeout(timeout_seconds=normalization_timeout_seconds)(str)(
                parsed_prediction
            )
            if normalization_timeout_seconds is not None
            else str(parsed_prediction)
        )
    except (Exception, TimeoutException) as exc:
        if not safe:
            raise
        return Verdict(
            status="parse_error",
            is_correct=False,
            candidate_text=candidate,
            normalized_prediction=None,
            normalized_gold=normalized_gold,
            extraction_rule=rule,
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        correct = bool(verify(parsed_gold, parsed_prediction, raise_on_error=True))
    except Exception as exc:
        return Verdict(
            status="verification_error",
            is_correct=False,
            candidate_text=candidate,
            normalized_prediction=normalized_prediction,
            normalized_gold=normalized_gold,
            extraction_rule=rule,
            error=f"{type(exc).__name__}: {exc}",
        )
    except TimeoutException as exc:
        if not safe:
            raise
        return Verdict(
            status="verification_error",
            is_correct=False,
            candidate_text=candidate,
            normalized_prediction=normalized_prediction,
            normalized_gold=normalized_gold,
            extraction_rule=rule,
            error=f"{type(exc).__name__}: {exc}",
        )
    return Verdict(
        status="correct" if correct else "incorrect",
        is_correct=correct,
        candidate_text=candidate,
        normalized_prediction=normalized_prediction,
        normalized_gold=normalized_gold,
        extraction_rule=rule,
    )


def parse_and_verify(final_text: str, gold: str, truncated: bool = False) -> ParseResult:
    parsed_gold = _parse_gold(gold)
    verdict = _verify_candidate(
        *extract_candidate(final_text, truncated=truncated),
        parsed_gold,
        str(parsed_gold),
    )
    return ParseResult(**asdict(verdict))


def _parse_dual(
    final_text: str,
    gold: str,
    boxed: str | None,
    safe: bool = False,
    normalization_timeout_seconds: int | None = None,
) -> DualParseResult:
    parsed_gold = _parse_gold(gold)
    normalized_gold = str(parsed_gold)
    if boxed is not None:
        verdict = _verify_candidate(
            boxed,
            "last_boxed",
            parsed_gold,
            normalized_gold,
            safe,
            normalization_timeout_seconds,
        )
        return DualParseResult(strict=verdict, soft=verdict)

    strict = _no_candidate(normalized_gold)
    if not final_text.strip():
        return DualParseResult(strict=strict, soft=strict)
    soft = _verify_candidate(
        final_text.strip(),
        "full_final_text",
        parsed_gold,
        normalized_gold,
        safe,
        normalization_timeout_seconds,
    )
    return DualParseResult(strict=strict, soft=soft)


def parse_dual_and_verify(
    final_text: str,
    gold: str,
    truncated: bool = False,
) -> DualParseResult:
    return _parse_dual(final_text, gold, last_strict_boxed(final_text))


def parse_v5_dual_and_verify(
    final_text: str,
    gold: str,
    truncated: bool = False,
) -> DualParseResult:
    return _parse_dual(final_text, gold, last_v5_strict_boxed(final_text))


def parse_v51_dual_and_verify(
    final_text: str,
    gold: str,
    truncated: bool = False,
) -> DualParseResult:
    return _parse_dual(final_text, gold, last_v5_strict_boxed(final_text), safe=True)


def parse_v52_dual_and_verify(
    final_text: str,
    gold: str,
    truncated: bool = False,
) -> DualParseResult:
    return _parse_dual(
        final_text,
        gold,
        last_v5_strict_boxed(final_text),
        safe=True,
        normalization_timeout_seconds=NORMALIZATION_TIMEOUT_SECONDS,
    )
