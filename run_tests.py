"""
AI Unit Test Runner
====================
1. Sends a prompt to a "subject" AI (the one being tested)
2. Passes the output to a "judge" agent that connects to the AgentVerifier MCP server
3. The judge approves or rejects based on your criteria
4. Prints a test report

Usage:
    uv run python run_tests.py

Make sure the MCP server is running first:
    uv run python server.py
"""

import asyncio
import json
import os
import textwrap
from dataclasses import dataclass, field
from datetime import datetime

import httpx

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MCP_SERVER_URL    = "http://localhost:8000/mcp"
SUBJECT_MODEL     = "claude-sonnet-4-20250514"   # the AI being tested
JUDGE_MODEL       = "claude-sonnet-4-20250514"   # the AI doing the judging


# ── Test case definition ───────────────────────────────────────────────────────

@dataclass
class TestCase:
    """
    Define a test case like a unit test:
      - name:     human-readable label shown in the report
      - prompt:   the exact prompt sent to the subject AI
      - criteria: plain-English description of what a passing response looks like
      - checks:   optional deterministic checks run before the judge even looks
                  (see run_verification_suite in server.py for supported types)
    """
    name:     str
    prompt:   str
    criteria: str
    checks:   list[dict] = field(default_factory=list)


# ── Define your test suite here ───────────────────────────────────────────────

TEST_SUITE: list[TestCase] = [
    TestCase(
        name="Basic arithmetic explanation",
        prompt="Explain why 0.1 + 0.2 does not equal 0.3 in most programming languages.",
        criteria=(
            "Must explain floating point representation. "
            "Must mention binary or IEEE 754. "
            "Should be clear enough for a junior developer. "
            "Must not be condescending or overly academic."
        ),
        checks=[
            {"type": "keywords",  "keywords": ["floating point", "binary"], "require_all": False},
            {"type": "length",    "min_words": 60, "max_words": 600},
            {"type": "forbidden", "forbidden_patterns": ["I cannot", "I'm sorry", "As an AI"]},
        ],
    ),
    TestCase(
        name="Code generation — reverse a string in Python",
        prompt="Write a Python function that reverses a string. Include a docstring and two example calls.",
        criteria=(
            "Must include a working Python function. "
            "Must include a docstring. "
            "Must show at least two example calls or usage examples. "
            "Code must be syntactically correct Python."
        ),
        checks=[
            {"type": "keywords",  "keywords": ["def ", "docstring", '"""'], "require_all": False},
            {"type": "keywords",  "keywords": ["def "],                     "require_all": True},
            {"type": "forbidden", "forbidden_patterns": ["I cannot"]},
        ],
    ),
    TestCase(
        name="Concise factual answer",
        prompt="What is the capital of Australia? Answer in one sentence.",
        criteria=(
            "Must correctly state Canberra as the capital. "
            "Must be a single sentence. "
            "Must not mention Sydney as the capital."
        ),
        checks=[
            {"type": "keywords",  "keywords": ["Canberra"], "require_all": True},
            {"type": "forbidden", "forbidden_patterns": ["Sydney is the capital"]},
            {"type": "length",    "max_words": 40},
        ],
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

async def call_anthropic(prompt: str, system: str, model: str, client: httpx.AsyncClient) -> str:
    """Make a simple single-turn Anthropic API call and return the text response."""
    resp = await client.post(
        ANTHROPIC_API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model":      model,
            "max_tokens": 1024,
            "system":     system,
            "messages":   [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()


async def call_mcp_tool(tool: str, args: dict, client: httpx.AsyncClient) -> dict:
    """
    Call a tool on the AgentVerifier MCP server via its JSON-RPC HTTP endpoint.
    We call it directly here so this script has no extra MCP client dependency.
    """
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "tools/call",
        "params":  {"name": tool, "arguments": args},
    }
    resp = await client.post(MCP_SERVER_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"MCP error: {data['error']}")
    # MCP returns content as a list; first item is the JSON result
    content = data["result"]["content"][0]["text"]
    return json.loads(content)


# ── Core logic ────────────────────────────────────────────────────────────────

async def run_test(tc: TestCase, client: httpx.AsyncClient) -> dict:
    """
    Run a single test case end-to-end:
      1. Prompt the subject AI
      2. Run deterministic checks (if any)
      3. Run the LLM judge
      4. Return a result dict
    """
    print(f"\n{'─'*60}")
    print(f"  TEST: {tc.name}")
    print(f"{'─'*60}")

    # ── Step 1: Get subject AI response ──────────────────────────────────────
    print("  → Prompting subject AI...")
    subject_output = await call_anthropic(
        prompt=tc.prompt,
        system="You are a helpful assistant. Answer clearly and directly.",
        model=SUBJECT_MODEL,
        client=client,
    )
    print(f"  ✉  Subject response ({len(subject_output.split())} words):")
    print(textwrap.indent(textwrap.shorten(subject_output, width=200, placeholder="..."), "     "))

    result = {
        "name":           tc.name,
        "prompt":         tc.prompt,
        "subject_output": subject_output,
        "checks_passed":  None,
        "check_details":  None,
        "judge_score":    None,
        "judge_reasoning": None,
        "verdict":        None,   # "PASS" | "FAIL"
        "fail_reason":    None,
    }

    # ── Step 2: Deterministic checks via MCP ────────────────────────────────
    if tc.checks:
        print("  → Running deterministic checks via MCP...")
        suite = await call_mcp_tool("run_verification_suite", {
            "output": subject_output,
            "checks": tc.checks,
        })
        result["checks_passed"] = suite["overall_valid"]
        result["check_details"] = suite["results"]

        status = "✅ all passed" if suite["overall_valid"] else f"❌ {suite['failed']}/{suite['total']} failed"
        print(f"  ✔  Deterministic checks: {status}")

        if not suite["overall_valid"]:
            failed = [r for r in suite["results"] if not r.get("valid")]
            result["verdict"]     = "FAIL"
            result["fail_reason"] = f"Deterministic checks failed: {failed}"
            print(f"  ✘  Skipping judge — already failed deterministic checks.")
            return result

    # ── Step 3: LLM judge via MCP ────────────────────────────────────────────
    print("  → Asking judge to evaluate...")
    judgment = await call_mcp_tool("llm_judge", {
        "agent_output":        subject_output,
        "evaluation_criteria": tc.criteria,
        "original_prompt":     tc.prompt,
    })

    result["judge_score"]    = judgment.get("score")
    result["judge_reasoning"] = judgment.get("reasoning")

    score   = judgment.get("score", 0)
    verdict = "PASS" if judgment.get("valid") else "FAIL"
    result["verdict"] = verdict

    icon = "✅" if verdict == "PASS" else "❌"
    print(f"  {icon} Judge verdict: {verdict}  (score {score}/5)")
    print(f"     Reasoning: {judgment.get('reasoning', '')}")

    if verdict == "FAIL":
        result["fail_reason"] = f"Judge scored {score}/5 — {judgment.get('reasoning', '')}"

    return result


async def run_suite() -> None:
    """Run all test cases and print a final report."""
    print("\n" + "═"*60)
    print("  AI UNIT TEST RUNNER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Subject model : {SUBJECT_MODEL}")
    print(f"  Judge model   : {JUDGE_MODEL}")
    print(f"  MCP server    : {MCP_SERVER_URL}")
    print("═"*60)

    async with httpx.AsyncClient(timeout=90) as client:
        results = []
        for tc in TEST_SUITE:
            try:
                r = await run_test(tc, client)
            except Exception as e:
                r = {
                    "name":    tc.name,
                    "verdict": "ERROR",
                    "error":   str(e),
                }
                print(f"  💥 ERROR running test: {e}")
            results.append(r)

    # ── Final report ─────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    failed = sum(1 for r in results if r["verdict"] == "FAIL")
    errors = sum(1 for r in results if r["verdict"] == "ERROR")
    total  = len(results)

    print("\n" + "═"*60)
    print("  RESULTS")
    print("═"*60)
    for r in results:
        icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}.get(r["verdict"], "?")
        score_str = f"  score {r.get('judge_score')}/5" if r.get("judge_score") else ""
        print(f"  {icon} {r['name']}{score_str}")
        if r["verdict"] != "PASS":
            reason = r.get("fail_reason") or r.get("error", "")
            print(f"       ↳ {reason}")

    print("─"*60)
    print(f"  {passed}/{total} passed   {failed} failed   {errors} errors")
    print("═"*60 + "\n")

    # Exit with non-zero code if any tests failed (useful in CI)
    if failed or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(run_suite())