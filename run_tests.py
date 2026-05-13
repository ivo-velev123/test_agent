"""
AI Unit Test Runner
====================
Prompts a subject AI then uses the AgentVerifier MCP server to judge and
approve / reject each response — like a smarter unit test.

Usage:
    export GEMINI_API_KEY=your_key_here
    uv run python run_tests.py

Make sure the MCP server is running first in another terminal:
    uv run python server.py
"""

import asyncio
import textwrap
from dataclasses import dataclass, field
from datetime import datetime

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# ── Config ────────────────────────────────────────────────────────────────────

MCP_SERVER_URL = "http://localhost:8000/mcp"   # Streamable HTTP endpoint

# Provider + model used for BOTH the subject agent and the judge.
# Swap these out to test any combination.
SUBJECT_PROVIDER = "gemini"
SUBJECT_MODEL    = "gemini-2.5-flash"

JUDGE_PROVIDER   = "gemini"
JUDGE_MODEL      = "gemini-2.5-flash"

# Other provider examples — just change the strings above:
#   SUBJECT_PROVIDER = "anthropic" / SUBJECT_MODEL = "claude-haiku-4-5-20251001"
#   SUBJECT_PROVIDER = "openai"    / SUBJECT_MODEL = "gpt-4o-mini"
#   SUBJECT_PROVIDER = "ollama"    / SUBJECT_MODEL = "llama3"


# ── Test case definition ──────────────────────────────────────────────────────

@dataclass
class TestCase:
    """
    One AI unit test.
      name     — label shown in the report
      prompt   — exact prompt sent to the subject AI
      criteria — plain-English pass/fail rules for the judge
    """
    name:     str
    prompt:   str
    criteria: str


# ── Your test suite — edit this ───────────────────────────────────────────────

TEST_SUITE: list[TestCase] = [
    TestCase(
        name="Sky colour explanation",
        prompt="Explain why the sky is blue in one sentence.",
        criteria="Must mention Rayleigh scattering. Must be a single sentence.",
    ),
    TestCase(
        name="Basic Python function",
        prompt="Write a Python function that reverses a string. Include a docstring.",
        criteria=(
            "Must include a working Python function definition using 'def'. "
            "Must include a docstring. "
            "Code must be syntactically valid Python."
        ),
    ),
    TestCase(
        name="Capital city — factual accuracy",
        prompt="What is the capital of Australia? Answer in one sentence.",
        criteria=(
            "Must correctly state Canberra as the capital of Australia. "
            "Must NOT say Sydney is the capital. "
            "Must be a single sentence."
        ),
    ),
]


# ── Test runner ───────────────────────────────────────────────────────────────

async def run_test(tc: TestCase, session: ClientSession) -> dict:
    """Run a single test: prompt the subject, then judge the output."""
    print(f"\n{'─'*60}")
    print(f"  TEST: {tc.name}")
    print(f"{'─'*60}")

    # Step 1 — get the subject AI's response
    print(f"  → Prompting subject AI ({SUBJECT_PROVIDER}/{SUBJECT_MODEL})...")
    subject_result = await session.call_tool(
        "prompt_subject_agent",
        arguments={
            "prompt":    tc.prompt,
            "provider":  SUBJECT_PROVIDER,
            "model":     SUBJECT_MODEL,
        },
    )
    subject_output = subject_result.content[0].text
    preview = textwrap.shorten(subject_output, width=160, placeholder="...")
    print(f"  ✉  Response: {preview}")

    # Step 2 — judge the response
    print(f"  → Judging ({JUDGE_PROVIDER}/{JUDGE_MODEL})...")
    judge_result = await session.call_tool(
        "llm_judge",
        arguments={
            "agent_output":        subject_output,
            "evaluation_criteria": tc.criteria,
            "original_prompt":     tc.prompt,
            "provider":            JUDGE_PROVIDER,
            "model":               JUDGE_MODEL,
        },
    )

    # The tool returns a JSON string — parse it
    import json
    judgment = json.loads(judge_result.content[0].text)

    verdict  = "PASS" if judgment.get("valid") else "FAIL"
    score    = judgment.get("score", "?")
    reason   = judgment.get("reasoning", "")
    icon     = "✅" if verdict == "PASS" else "❌"

    print(f"  {icon} {verdict}  (score {score}/10)")
    print(f"     {reason}")

    return {
        "name":           tc.name,
        "verdict":        verdict,
        "score":          score,
        "reasoning":      reason,
        "subject_output": subject_output,
    }


async def main() -> None:
    print("\n" + "═"*60)
    print("  AI UNIT TEST RUNNER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Subject : {SUBJECT_PROVIDER}/{SUBJECT_MODEL}")
    print(f"  Judge   : {JUDGE_PROVIDER}/{JUDGE_MODEL}")
    print(f"  Server  : {MCP_SERVER_URL}")
    print("═"*60)

    results = []

    async with streamable_http_client(MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for tc in TEST_SUITE:
                try:
                    r = await run_test(tc, session)
                except Exception as e:
                    print(f"  💥 ERROR: {e}")
                    r = {"name": tc.name, "verdict": "ERROR", "score": None, "reasoning": str(e)}
                results.append(r)

    # ── Final report ──────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    failed = sum(1 for r in results if r["verdict"] == "FAIL")
    errors = sum(1 for r in results if r["verdict"] == "ERROR")

    print("\n" + "═"*60)
    print("  RESULTS")
    print("═"*60)
    for r in results:
        icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}.get(r["verdict"], "?")
        score_str = f"  score {r['score']}/10" if r.get("score") is not None else ""
        print(f"  {icon} {r['name']}{score_str}")
        if r["verdict"] != "PASS":
            print(f"       ↳ {r.get('reasoning', r.get('error', ''))}")
    print("─"*60)
    print(f"  {passed}/{len(results)} passed   {failed} failed   {errors} errors")
    print("═"*60 + "\n")

    if failed or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())