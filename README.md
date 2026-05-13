# Agent Verifier MCP Server

Smarter unit tests for AI agents. Prompt a subject AI, then have a judge agent score and approve/reject the output based on plain-English criteria.

```
test case
  └── prompt ──→ Subject AI ──→ response
                                  └── LLM Judge (scores 0–10, pass/fail + reasoning)
                                            └── PASS ✅ / FAIL ❌
```

---

## Prerequisites

You need **Node.js**, **Python**, and **uv** installed before anything else.

- **Node.js** (v18+): https://nodejs.org
- **uv**: https://docs.astral.sh/uv/getting-started/installation/
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

---

## Step 1 — Install the Gemini CLI

```bash
npm install -g @google/gemini-cli
```

Verify it installed:
```bash
gemini --version
```

---

## Step 2 — Authenticate with Gemini

Run Gemini once interactively to log in:
```bash
gemini
```

This opens a browser and saves your credentials locally. Once you see the prompt, type `/quit` and press Enter to exit. You only need to do this once.

> **Alternative:** If you have a Gemini API key, you can skip the browser login and just set it as an environment variable instead:
> ```bash
> export GEMINI_API_KEY=your_key_here
> ```

---

## Step 3 — Create the project

```bash
uv init agent-verifier
cd agent-verifier
uv add "mcp[cli]" httpx
```

---

## Step 4 — Add the files

Copy `server.py` and `run_tests.py` into the `agent-verifier` directory you just created. Your folder should look like:

```
agent-verifier/
├── server.py
├── run_tests.py
└── pyproject.toml    ← created by uv
```

---

## Step 5 — Run it

You need two terminal windows open, both inside the `agent-verifier` directory.

**Terminal 1 — start the MCP server:**
```bash
uv run python server.py
```
You should see something like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```
Leave this running.

**Terminal 2 — run your tests:**
```bash
uv run python run_tests.py
```

You should see output like this:
```
════════════════════════════════════════════════════════════
  AI UNIT TEST RUNNER  2024-01-15 14:32:01
  Subject : gemini/gemini-2.0-flash
  Judge   : gemini/gemini-2.0-flash
════════════════════════════════════════════════════════════

  TEST: Sky colour explanation
  → Prompting subject AI (gemini/gemini-2.0-flash)...
  ✉  Response: The sky is blue due to Rayleigh scattering...
  → Judging (gemini/gemini-2.0-flash)...
  ✅ PASS  (score 9/10)
     Correctly mentions Rayleigh scattering in a single sentence.

════════════════════════════════════════════════════════════
  RESULTS
════════════════════════════════════════════════════════════
  ✅ Sky colour explanation  score 9/10
  ✅ Basic Python function  score 8/10
  ✅ Capital city — factual accuracy  score 10/10
──────────────────────────────────────────────────────────
  3/3 passed   0 failed   0 errors
════════════════════════════════════════════════════════════
```

---

## Writing your own test cases

Open `run_tests.py` and edit the `TEST_SUITE` list. Each test has three parts:

```python
TestCase(
    name="What shows up in the report",
    prompt="The exact prompt sent to the AI being tested",
    criteria="Plain-English rules the judge uses to pass or fail the response.",
),
```

For example:
```python
TestCase(
    name="Haiku about rain",
    prompt="Write a haiku about rain.",
    criteria=(
        "Must follow the 5-7-5 syllable structure. "
        "Must be about rain. "
        "Must not rhyme."
    ),
),
```

Be as specific as you like in `criteria` — the judge reads it directly.

---

## Switching to a different AI provider

At the top of `run_tests.py` you can change which AI is being tested and which AI does the judging:

```python
SUBJECT_PROVIDER = "gemini"        # the AI being tested
SUBJECT_MODEL    = "gemini-2.0-flash"

JUDGE_PROVIDER   = "gemini"        # the AI doing the judging
JUDGE_MODEL      = "gemini-2.0-flash"
```

You can mix and match — for example, test an Ollama model and judge it with Gemini:
```python
SUBJECT_PROVIDER = "ollama"
SUBJECT_MODEL    = "llama3"

JUDGE_PROVIDER   = "gemini"
JUDGE_MODEL      = "gemini-2.0-flash"
```

### Supported providers

| Provider | How to set up |
|---|---|
| `gemini` | Follow Steps 1–2 above |
| `anthropic` | `export ANTHROPIC_API_KEY=your_key` |
| `openai` | `export OPENAI_API_KEY=your_key` |
| `ollama` | Install from [ollama.com](https://ollama.com), run `ollama pull llama3` |

### Model name reference

| Provider | Example models |
|---|---|
| `gemini` | `gemini-2.0-flash`, `gemini-1.5-pro` |
| `anthropic` | `claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001` |
| `openai` | `gpt-4o`, `gpt-4o-mini` |
| `ollama` | `llama3`, `gemma3:27b`, `mistral` |

---

## Troubleshooting

**`gemini: command not found`**
The Gemini CLI isn't on your PATH. Try closing and reopening your terminal after installing, or run `npm install -g @google/gemini-cli` again.

**`Connection refused` when running `run_tests.py`**
The MCP server isn't running. Make sure Terminal 1 is still running `uv run python server.py`.

**`Please set an Auth method...`**
You haven't authenticated the Gemini CLI yet. Go back to Step 2.

**Tests hang for a long time**
Normal for the first run — the Gemini CLI has a small startup cost. Subsequent calls are faster.