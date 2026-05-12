# Agent Verifier MCP Server

An MCP server exposing tools so one agent can test and verify another agent's outputs.

## Installation

Create a uv project and add the dependencies:

```bash
uv init agent-verifier
cd agent-verifier
uv add "mcp[cli]" httpx jsonschema
```

Then copy `server.py` into the project directory.

## Running

```bash
# Stdio (for Claude Desktop / Claude Code)
uv run python server.py

# Streamable HTTP (for remote agents)
uv run python server.py   # runs on http://localhost:8000/mcp

# Or via the mcp CLI (development mode with inspector)
uv run mcp dev server.py
```

Add to Claude Code:
```bash
claude mcp add --transport http agent-verifier http://localhost:8000/mcp
```

---

## Tools at a Glance

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

---

## Usage Examples

### Verify a structured JSON response
```python
result = await session.call_tool("validate_json_schema", {
    "output": '{"name": "Alice", "age": 30}',
    "schema": '{"type": "object", "required": ["name", "age"], "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'
})
# → {valid: true, errors: [], parsed_output: {...}}
```

### Check an agent didn't refuse the task
```python
result = await session.call_tool("check_no_forbidden_content", {
    "output": agent_response,
    "forbidden_patterns": ["I cannot", "I'm unable to", "I don't have access"]
})
```

### Run a full verification suite in one shot
```python
result = await session.call_tool("run_verification_suite", {
    "output": agent_response,
    "checks": [
        {"type": "keywords",  "keywords": ["summary", "recommendation"]},
        {"type": "length",    "min_words": 100, "max_words": 800},
        {"type": "forbidden", "forbidden_patterns": ["I cannot", "sorry"]},
        {"type": "format",    "must_end_with": "."}
    ]
})
# → {overall_valid: true/false, passed: 3, failed: 1, total: 4, results: [...]}
```

### LLM-as-judge
```python
result = await session.call_tool("llm_judge", {
    "agent_output": agent_response,
    "original_prompt": "Summarise the Q3 earnings report in 3 bullet points.",
    "evaluation_criteria": "Response must contain exactly 3 bullets, each covering revenue, profit, and outlook. Tone should be neutral and factual."
})
# → {valid: true, score: 4, reasoning: "...", raw_judge_response: "..."}
```

### Compare two agents (A/B)
```python
result = await session.call_tool("diff_outputs", {
    "output_a": agent_a_response,
    "output_b": agent_b_response,
    "label_a": "GPT-4o",
    "label_b": "Claude"
})
# → {similarity_ratio: 0.73, identical: false, only_in_gpt4o: [...], ...}
```

### Batch benchmark
```python
result = await session.call_tool("benchmark_agent", {
    "prompts_and_expected": [
        {"prompt": "What is 2+2?", "expected": "4"},
        {"prompt": "Capital of France?", "expected": "Paris"},
    ],
    "evaluation_criteria": "Answer must be correct, concise, and not include unnecessary caveats.",
    "agent_outputs": ["The answer is 4.", "It's Paris, the capital city."]
})
# → {average_score: 4.5, pass_rate: 1.0, per_item_results: [...]}
```