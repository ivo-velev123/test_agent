# Agent Verifier MCP Server

Smarter unit tests for AI agents. Prompt a subject AI, then have a judge agent approve or reject the output based on plain-English criteria — backed by deterministic checks and LLM-as-judge scoring.

```
your test case
    └── prompt ──→ Subject AI ──→ response
                                    ├── deterministic checks (keywords, length, forbidden, JSON schema...)
                                    └── LLM judge (scores 1–5, pass/fail with reasoning)
                                              └── PASS / FAIL
```

## Setup

```bash
uv init agent-verifier
cd agent-verifier
uv add "mcp[cli]" httpx jsonschema
```

Copy `server.py` and `run_tests.py` into the project directory.

## Quickstart

**Terminal 1** — start the verifier MCP server:
```bash
uv run python server.py
# Listening on http://localhost:8000/mcp
```

**Terminal 2** — run your test suite:
```bash
uv run python run_tests.py
```

You'll see output like:
```
════════════════════════════════════════════════════════════
  AI UNIT TEST RUNNER  2024-01-15 14:32:01
════════════════════════════════════════════════════════════

  TEST: Basic arithmetic explanation
  → Prompting subject AI...
  ✉  Subject response (143 words): Floating point numbers are stored in binary...
  → Running deterministic checks via MCP...
  ✔  Deterministic checks: all passed
  → Asking judge to evaluate...
  ✅ Judge verdict: PASS  (score 4/5)
     Reasoning: Correctly explains IEEE 754, accessible language, good example.

════════════════════════════════════════════════════════════
  RESULTS
════════════════════════════════════════════════════════════
  ✅ Basic arithmetic explanation  score 4/5
  ✅ Code generation — reverse a string  score 5/5
  ❌ Concise factual answer  score 2/5
       ↳ Judge scored 2/5 — Response mentioned Sydney before correcting itself.
──────────────────────────────────────────────────────────
  2/3 passed   1 failed   0 errors
════════════════════════════════════════════════════════════
```

Exits with code `1` if any tests fail — works in CI pipelines.

## Development mode (with MCP Inspector)

```bash
uv run mcp dev server.py
```

---

## Writing test cases

Edit the `TEST_SUITE` list in `run_tests.py`. Each test case has three parts:

```python
TestCase(
    name="What shows up in the report",
    prompt="The exact prompt sent to the subject AI",
    criteria=(
        "Plain-English description of what a passing response looks like. "
        "Be specific — the judge reads this directly."
    ),
    checks=[  # optional fast pre-checks before the judge runs
        {"type": "keywords",  "keywords": ["must", "contain", "these"]},
        {"type": "length",    "min_words": 50, "max_words": 500},
        {"type": "forbidden", "forbidden_patterns": ["I cannot", "As an AI"]},
        {"type": "format",    "must_be_json": True},
    ],
)
```

Deterministic checks run first and are cheap (no LLM call). If they fail, the judge is skipped entirely. This keeps costs low — bad outputs get caught early.

### Supported check types

| type | key args |
|---|---|
| `keywords` | `keywords` (list), `require_all` (bool), `case_sensitive` (bool) |
| `forbidden` | `forbidden_patterns` (list), `use_regex` (bool) |
| `length` | `min_chars`, `max_chars`, `min_words`, `max_words` |
| `json_fields` | `required_fields` (list) |
| `format` | `must_be_json`, `must_start_with`, `must_end_with`, `must_match_regex` |
| `similarity` | `reference` (str), `threshold` (float, default 0.8) |

---

## Tools at a Glance

The MCP server also exposes these tools individually — useful if you want to call them from another agent or script:

| Tool | What it checks |
|---|---|
| `validate_json_schema` | Agent JSON output against a JSON Schema |
| `check_required_fields` | All expected keys present in JSON output |
| `check_contains_keywords` | Required keywords/phrases present in text |
| `check_no_forbidden_content` | Forbidden strings/regex absent from output |
| `measure_similarity` | Fuzzy match ratio vs a reference answer |
| `check_output_length` | Character and word count bounds |
| `check_format_rules` | Starts/ends with, matches regex, valid JSON |
| `llm_judge` | LLM-as-judge scoring (1–5) with reasoning |
| `run_verification_suite` | Run multiple checks in one call |
| `diff_outputs` | Line-level diff between two agents' responses |
| `benchmark_agent` | Batch LLM-judged scoring across many prompts |