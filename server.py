"""
Agent Verifier MCP Server

Tools for testing and verifying another agent's responses/outputs.
Run with:
    pip install "mcp[cli]" httpx
    python server.py
Or mount via streamable HTTP:
    mcp.run(transport="streamable-http")
"""

import json
import re
import time
from difflib import SequenceMatcher
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AgentVerifier", json_response=True)


# ---------------------------------------------------------------------------
# 1. Schema / structure validation
# ---------------------------------------------------------------------------


@mcp.tool()
def validate_json_schema(
    output: str,
    schema: str,
) -> dict[str, Any]:
    """
    Validate that an agent's JSON output conforms to a given JSON Schema.

    Args:
        output: The agent's raw output string (must be valid JSON).
        schema: A JSON Schema object as a JSON string.

    Returns:
        {valid, errors, parsed_output}
    """
    try:
        import jsonschema  # optional dep; graceful error if missing
    except ImportError:
        return {
            "valid": False,
            "errors": ["jsonschema package not installed. Run: pip install jsonschema"],
            "parsed_output": None,
        }

    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "errors": [f"Output is not valid JSON: {e}"],
            "parsed_output": None,
        }

    try:
        schema_obj = json.loads(schema)
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "errors": [f"Schema is not valid JSON: {e}"],
            "parsed_output": None,
        }

    validator = jsonschema.Draft7Validator(schema_obj)
    errors = [e.message for e in validator.iter_errors(data)]
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "parsed_output": data,
    }


@mcp.tool()
def check_required_fields(
    output: str,
    required_fields: list[str],
) -> dict[str, Any]:
    """
    Check that a JSON agent output contains all required top-level keys.

    Args:
        output: The agent's raw output string (must be valid JSON).
        required_fields: List of field names that must be present.

    Returns:
        {valid, missing_fields, present_fields}
    """
    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "missing_fields": required_fields,
            "present_fields": [],
            "error": str(e),
        }

    if not isinstance(data, dict):
        return {
            "valid": False,
            "missing_fields": required_fields,
            "present_fields": [],
            "error": "Output is not a JSON object",
        }

    present = [f for f in required_fields if f in data]
    missing = [f for f in required_fields if f not in data]
    return {
        "valid": len(missing) == 0,
        "missing_fields": missing,
        "present_fields": present,
    }


# ---------------------------------------------------------------------------
# 2. Content / semantic checks
# ---------------------------------------------------------------------------


@mcp.tool()
def check_contains_keywords(
    output: str,
    keywords: list[str],
    require_all: bool = True,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """
    Verify that an agent's output contains expected keywords or phrases.

    Args:
        output: The agent's response text.
        keywords: Words or phrases to look for.
        require_all: If True, ALL keywords must be present. If False, ANY one is enough.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        {valid, found, missing, match_count}
    """
    text = output if case_sensitive else output.lower()
    found = []
    missing = []
    for kw in keywords:
        needle = kw if case_sensitive else kw.lower()
        (found if needle in text else missing).append(kw)

    valid = (len(missing) == 0) if require_all else (len(found) > 0)
    return {
        "valid": valid,
        "found": found,
        "missing": missing,
        "match_count": len(found),
    }


@mcp.tool()
def check_no_forbidden_content(
    output: str,
    forbidden_patterns: list[str],
    use_regex: bool = False,
) -> dict[str, Any]:
    """
    Ensure an agent's output does NOT contain forbidden words, phrases, or regex patterns.

    Args:
        output: The agent's response text.
        forbidden_patterns: Strings or regex patterns that must NOT appear.
        use_regex: If True, treat each pattern as a regular expression.

    Returns:
        {valid, violations}  — violations lists each forbidden match found.
    """
    violations = []
    for pattern in forbidden_patterns:
        if use_regex:
            try:
                if re.search(pattern, output, re.IGNORECASE):
                    violations.append({"pattern": pattern, "type": "regex"})
            except re.error as e:
                violations.append(
                    {"pattern": pattern, "type": "invalid_regex", "error": str(e)}
                )
        else:
            if pattern.lower() in output.lower():
                violations.append({"pattern": pattern, "type": "literal"})

    return {
        "valid": len(violations) == 0,
        "violations": violations,
    }


@mcp.tool()
def measure_similarity(
    output: str,
    reference: str,
    threshold: float = 0.8,
) -> dict[str, Any]:
    """
    Compute the similarity ratio between an agent's output and a reference answer.
    Uses SequenceMatcher (character-level diffing); good for near-duplicate detection.

    Args:
        output: The agent's response.
        reference: The expected / gold-standard response.
        threshold: Minimum ratio (0–1) to be considered "similar enough".

    Returns:
        {valid, similarity_ratio, threshold, diff_summary}
    """
    ratio = SequenceMatcher(None, output.strip(), reference.strip()).ratio()
    # Build a simple diff summary
    matcher = SequenceMatcher(None, output.splitlines(), reference.splitlines())
    opcodes = matcher.get_opcodes()
    diff_summary = [
        {"op": tag, "output_lines": i2 - i1, "reference_lines": j2 - j1}
        for tag, i1, i2, j1, j2 in opcodes
        if tag != "equal"
    ]
    return {
        "valid": ratio >= threshold,
        "similarity_ratio": round(ratio, 4),
        "threshold": threshold,
        "diff_summary": diff_summary,
    }


# ---------------------------------------------------------------------------
# 3. Format / length checks
# ---------------------------------------------------------------------------


@mcp.tool()
def check_output_length(
    output: str,
    min_chars: int | None = None,
    max_chars: int | None = None,
    min_words: int | None = None,
    max_words: int | None = None,
) -> dict[str, Any]:
    """
    Verify an agent's output length is within acceptable character and/or word bounds.

    Args:
        output: The agent's response text.
        min_chars: Minimum character count (inclusive), or None to skip.
        max_chars: Maximum character count (inclusive), or None to skip.
        min_words: Minimum word count (inclusive), or None to skip.
        max_words: Maximum word count (inclusive), or None to skip.

    Returns:
        {valid, char_count, word_count, violations}
    """
    char_count = len(output)
    word_count = len(output.split())
    violations = []

    if min_chars is not None and char_count < min_chars:
        violations.append(f"Too short: {char_count} chars < minimum {min_chars}")
    if max_chars is not None and char_count > max_chars:
        violations.append(f"Too long: {char_count} chars > maximum {max_chars}")
    if min_words is not None and word_count < min_words:
        violations.append(f"Too few words: {word_count} < minimum {min_words}")
    if max_words is not None and word_count > max_words:
        violations.append(f"Too many words: {word_count} > maximum {max_words}")

    return {
        "valid": len(violations) == 0,
        "char_count": char_count,
        "word_count": word_count,
        "violations": violations,
    }


@mcp.tool()
def check_format_rules(
    output: str,
    must_be_json: bool = False,
    must_start_with: str | None = None,
    must_end_with: str | None = None,
    must_match_regex: str | None = None,
) -> dict[str, Any]:
    """
    Check basic formatting rules on an agent's output.

    Args:
        output: The agent's response text.
        must_be_json: If True, the output must parse as valid JSON.
        must_start_with: If set, output must begin with this string.
        must_end_with: If set, output must end with this string.
        must_match_regex: If set, the entire output must match this regex.

    Returns:
        {valid, violations}
    """
    violations = []
    stripped = output.strip()

    if must_be_json:
        try:
            json.loads(output)
        except json.JSONDecodeError as e:
            violations.append(f"Not valid JSON: {e}")

    if must_start_with and not stripped.startswith(must_start_with):
        violations.append(f"Does not start with {must_start_with!r}")

    if must_end_with and not stripped.endswith(must_end_with):
        violations.append(f"Does not end with {must_end_with!r}")

    if must_match_regex:
        try:
            if not re.fullmatch(must_match_regex, stripped, re.DOTALL):
                violations.append(f"Does not match regex: {must_match_regex}")
        except re.error as e:
            violations.append(f"Invalid regex: {e}")

    return {
        "valid": len(violations) == 0,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# 4. LLM-as-judge evaluation
# ---------------------------------------------------------------------------


@mcp.tool()
async def llm_judge(
    agent_output: str,
    evaluation_criteria: str,
    original_prompt: str = "",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 512,
) -> dict[str, Any]:
    """
    Use a separate Claude call to judge an agent's output against given criteria.
    Returns a structured verdict with a score and reasoning.

    Args:
        agent_output: The response produced by the agent under test.
        evaluation_criteria: Natural-language description of what "good" looks like.
        original_prompt: The prompt/task the agent was responding to (for context).
        model: The Claude model to use for judging.
        max_tokens: Token budget for the judge's response.

    Returns:
        {valid, score (1-5), reasoning, raw_judge_response}
    """
    import httpx

    system_prompt = (
        "You are an impartial evaluator. "
        "Given an agent's output and evaluation criteria, return ONLY a JSON object with:\n"
        '  "score": integer 1-5 (1=very poor, 5=excellent),\n'
        '  "reasoning": string (concise explanation),\n'
        '  "pass": boolean (true if score >= 3)\n'
        "No markdown, no extra text — raw JSON only."
    )

    user_message = (
        f"## Evaluation Criteria\n{evaluation_criteria}\n\n"
        + (f"## Original Prompt\n{original_prompt}\n\n" if original_prompt else "")
        + f"## Agent Output\n{agent_output}"
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_message}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["content"][0]["text"].strip()
            parsed = json.loads(raw)
            return {
                "valid": parsed.get("pass", False),
                "score": parsed.get("score"),
                "reasoning": parsed.get("reasoning"),
                "raw_judge_response": raw,
            }
    except Exception as e:
        return {
            "valid": False,
            "score": None,
            "reasoning": None,
            "raw_judge_response": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# 5. Composite / batch verification
# ---------------------------------------------------------------------------


@mcp.tool()
def run_verification_suite(
    output: str,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Run multiple verification checks in a single call and aggregate results.

    Each check is a dict with:
      - "type": one of "keywords", "forbidden", "length", "json_fields", "format", "similarity"
      - Plus the relevant kwargs for that check type.

    Supported check types and their kwargs:
      keywords    → keywords (list[str]), require_all (bool), case_sensitive (bool)
      forbidden   → forbidden_patterns (list[str]), use_regex (bool)
      length      → min_chars, max_chars, min_words, max_words (all optional int)
      json_fields → required_fields (list[str])
      format      → must_be_json, must_start_with, must_end_with, must_match_regex
      similarity  → reference (str), threshold (float)

    Returns:
      {overall_valid, passed, failed, total, results: [{type, valid, ...}]}

    Note: For LLM-as-judge, use the llm_judge tool separately (it requires async).

    Example checks list:
      [
        {"type": "keywords", "keywords": ["summary", "conclusion"]},
        {"type": "length", "min_words": 50, "max_words": 500},
        {"type": "forbidden", "forbidden_patterns": ["I cannot", "I'm unable"]}
      ]
    """
    dispatch = {
        "keywords": lambda c: check_contains_keywords(
            output,
            c["keywords"],
            c.get("require_all", True),
            c.get("case_sensitive", False),
        ),
        "forbidden": lambda c: check_no_forbidden_content(
            output,
            c["forbidden_patterns"],
            c.get("use_regex", False),
        ),
        "length": lambda c: check_output_length(
            output,
            c.get("min_chars"),
            c.get("max_chars"),
            c.get("min_words"),
            c.get("max_words"),
        ),
        "json_fields": lambda c: check_required_fields(output, c["required_fields"]),
        "format": lambda c: check_format_rules(
            output,
            c.get("must_be_json", False),
            c.get("must_start_with"),
            c.get("must_end_with"),
            c.get("must_match_regex"),
        ),
        "similarity": lambda c: measure_similarity(
            output, c["reference"], c.get("threshold", 0.8)
        ),
    }

    results = []
    for check in checks:
        check_type = check.get("type", "unknown")
        fn = dispatch.get(check_type)
        if fn is None:
            result = {
                "type": check_type,
                "valid": False,
                "error": f"Unknown check type: {check_type}",
            }
        else:
            try:
                result = fn(check)
                result["type"] = check_type
            except Exception as e:
                result = {"type": check_type, "valid": False, "error": str(e)}
        results.append(result)

    passed = sum(1 for r in results if r.get("valid"))
    failed = len(results) - passed
    return {
        "overall_valid": failed == 0,
        "passed": passed,
        "failed": failed,
        "total": len(results),
        "results": results,
    }


@mcp.tool()
def diff_outputs(
    output_a: str,
    output_b: str,
    label_a: str = "Agent A",
    label_b: str = "Agent B",
) -> dict[str, Any]:
    """
    Produce a structured diff between two agents' outputs — useful for A/B comparison.

    Args:
        output_a: First agent's response.
        output_b: Second agent's response.
        label_a: Human-readable label for the first output.
        label_b: Human-readable label for the second output.

    Returns:
        {similarity_ratio, identical, only_in_a, only_in_b, common_lines, line_diff}
    """
    lines_a = set(output_a.splitlines())
    lines_b = set(output_b.splitlines())

    matcher = SequenceMatcher(None, output_a, output_b)
    ratio = round(matcher.ratio(), 4)

    line_diff = []
    lm = SequenceMatcher(None, output_a.splitlines(), output_b.splitlines())
    for tag, i1, i2, j1, j2 in lm.get_opcodes():
        if tag == "equal":
            continue
        line_diff.append(
            {
                "operation": tag,
                f"{label_a}_lines": output_a.splitlines()[i1:i2],
                f"{label_b}_lines": output_b.splitlines()[j1:j2],
            }
        )

    return {
        "similarity_ratio": ratio,
        "identical": output_a == output_b,
        f"only_in_{label_a.lower().replace(' ', '_')}": list(lines_a - lines_b),
        f"only_in_{label_b.lower().replace(' ', '_')}": list(lines_b - lines_a),
        "common_lines": list(lines_a & lines_b),
        "line_diff": line_diff,
    }


@mcp.tool()
async def benchmark_agent(
    prompts_and_expected: list[dict[str, str]],
    evaluation_criteria: str,
    agent_outputs: list[str],
) -> dict[str, Any]:
    """
    Score a batch of agent responses across multiple prompts using LLM-as-judge.

    Args:
        prompts_and_expected: List of {"prompt": ..., "expected": ...} dicts.
        evaluation_criteria: Shared criteria applied to every response.
        agent_outputs: List of agent responses, aligned with prompts_and_expected.

    Returns:
        {average_score, pass_rate, per_item_results}
    """
    if len(prompts_and_expected) != len(agent_outputs):
        return {
            "error": f"Mismatch: {len(prompts_and_expected)} prompts vs {len(agent_outputs)} outputs"
        }

    per_item = []
    for i, (pe, out) in enumerate(zip(prompts_and_expected, agent_outputs)):
        # Augment criteria with expected answer for each item
        item_criteria = evaluation_criteria
        if pe.get("expected"):
            item_criteria += f"\n\nExpected answer reference:\n{pe['expected']}"

        result = await llm_judge(
            agent_output=out,
            evaluation_criteria=item_criteria,
            original_prompt=pe.get("prompt", ""),
        )
        per_item.append(
            {
                "index": i,
                "prompt": pe.get("prompt", ""),
                **result,
            }
        )
        time.sleep(0.2)  # gentle rate limiting

    scores = [r["score"] for r in per_item if r.get("score") is not None]
    avg = round(sum(scores) / len(scores), 2) if scores else None
    pass_rate = round(sum(1 for r in per_item if r.get("valid")) / len(per_item), 2)

    return {
        "average_score": avg,
        "pass_rate": pass_rate,
        "total_items": len(per_item),
        "per_item_results": per_item,
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
