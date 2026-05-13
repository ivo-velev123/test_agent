"""
Agent Verifier MCP Server
=========================
Hosts two tools:
  - prompt_subject_agent : send a prompt to any supported AI backend
  - llm_judge            : evaluate the response and return pass/fail + score

Supported backends (set via the `provider` argument):
  - "gemini"   — Google Gemini via the Gemini CLI  (run `gemini` once to auth first)
  - "ollama"   — any local Ollama model      (requires Ollama running locally)
  - "anthropic"— Claude via Anthropic API   (requires ANTHROPIC_API_KEY env var)
  - "openai"   — OpenAI-compatible endpoint  (requires OPENAI_API_KEY env var)

Run:
    uv run python server.py
"""

import asyncio
import json
import os
import re

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AgentVerifier")

# ── Provider helpers ──────────────────────────────────────────────────────────

async def _call_gemini(prompt: str, system: str, model: str) -> str:
    """
    Call Gemini via the official Gemini CLI (gemini -p "..." -o text).
    Auth is handled by the CLI itself — run `gemini` once interactively first
    to set up credentials, or set GEMINI_API_KEY in your environment.

    Model note: when authenticated via Google account (not API key), use
    "gemini-2.5-flash" or "gemini-2.5-pro". Pass an empty string to use
    the CLI default. "gemini-2.0-flash" is an API-only model name and will
    cause a 404 with account-based auth.

    System prompt is prepended to the user prompt since the CLI has no
    separate system flag. We set GEMINI_CLI_NO_IDE=1 to suppress IDE hooks
    and extension noise that appear when running non-interactively.
    """
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    cmd = ["gemini", "-p", full_prompt, "-o", "text"]
    # Only pass -m if a model is explicitly specified; let CLI use its default otherwise
    if model:
        cmd += ["-m", model]

    # Inherit environment but suppress IDE/hook integrations that pollute stderr
    env = os.environ.copy()
    env["GEMINI_CLI_NO_IDE"] = "1"       # disables IDE companion extension attempts
    env["NO_COLOR"] = "1"                # no ANSI escape codes in output

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)

    stderr_text = stderr.decode().strip()

    if process.returncode != 0:
        raise RuntimeError(f"Gemini CLI exited with code {process.returncode}: {stderr_text}")

    output = stdout.decode().strip()
    if not output:
        # Some errors are written to stdout by the CLI instead of stderr
        raise RuntimeError(f"Gemini CLI returned empty output. stderr: {stderr_text}")

    return output


async def _call_anthropic(prompt: str, system: str, model: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 1024,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()


async def _call_openai(prompt: str, system: str, model: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def _call_ollama(prompt: str, system: str, model: str) -> str:
    # Use the Ollama HTTP API instead of subprocess — more reliable and supports system prompts
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


async def _dispatch(prompt: str, system: str, provider: str, model: str) -> str:
    """Route to the correct provider."""
    provider = provider.lower()
    if provider == "gemini":
        return await _call_gemini(prompt, system, model)
    elif provider == "anthropic":
        return await _call_anthropic(prompt, system, model)
    elif provider == "openai":
        return await _call_openai(prompt, system, model)
    elif provider == "ollama":
        return await _call_ollama(prompt, system, model)
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: gemini, anthropic, openai, ollama"
        )


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
async def prompt_subject_agent(
    prompt: str,
    provider: str = "gemini",
    model: str = "gemini-2.5-flash",
) -> str:
    """
    Send a prompt to an AI and return its response.

    Args:
        prompt:   The prompt to send to the subject agent.
        provider: Which AI backend to use — "gemini", "anthropic", "openai", or "ollama".
        model:    The model name for the chosen provider.
                  Gemini examples    : gemini-2.5-flash, gemini-2.5-pro
                  Anthropic examples : claude-sonnet-4-20250514, claude-haiku-4-5-20251001
                  OpenAI examples    : gpt-4o, gpt-4o-mini
                  Ollama examples    : llama3, gemma3:27b, mistral

    Returns:
        The agent's response as a plain string.
    """
    try:
        return await _dispatch(
            prompt=prompt,
            system="You are a helpful assistant. Answer clearly and directly.",
            provider=provider,
            model=model,
        )
    except Exception as e:
        return f"ERROR [{provider}/{model}]: {e}"


@mcp.tool()
async def llm_judge(
    agent_output: str,
    evaluation_criteria: str,
    original_prompt: str = "",
    provider: str = "gemini",
    model: str = "gemini-2.5-flash",
) -> dict:
    """
    Evaluate an agent's output against criteria. Returns a verdict with score and reasoning.

    Args:
        agent_output:         The response produced by the subject agent.
        evaluation_criteria:  Plain-English description of what a passing response looks like.
        original_prompt:      The prompt that produced the output (optional, adds context).
        provider:             Which AI backend to use for judging.
        model:                The model name for the chosen provider.

    Returns:
        {valid (bool), score (0-10), reasoning (str)}
    """
    system = (
        "You are an impartial evaluator. "
        "Given an agent's output and evaluation criteria, return ONLY a JSON object with:\n"
        '  "score": integer 0-10 (0=completely wrong, 10=perfect),\n'
        '  "reasoning": string (concise explanation of the score),\n'
        '  "pass": boolean (true if score >= 6)\n'
        "No markdown fences, no extra text — raw JSON only."
    )

    user_message = (
        f"## Evaluation Criteria\n{evaluation_criteria}\n\n"
        + (f"## Original Prompt\n{original_prompt}\n\n" if original_prompt else "")
        + f"## Agent Output\n{agent_output}"
    )

    try:
        raw = await _dispatch(
            prompt=user_message,
            system=system,
            provider=provider,
            model=model,
        )

        # Strip markdown fences if the model added them anyway
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())

        parsed = json.loads(cleaned)
        return {
            "valid": bool(parsed.get("pass", False)),
            "score": parsed.get("score"),
            "reasoning": parsed.get("reasoning"),
        }
    except json.JSONDecodeError:
        return {
            "valid": False,
            "score": 0,
            "reasoning": f"Judge returned non-JSON output: {raw[:300]}",
        }
    except Exception as e:
        return {
            "valid": False,
            "score": 0,
            "reasoning": f"Judge error: {e}",
        }


if __name__ == "__main__":
    # Streamable HTTP transport — clients connect to http://localhost:8000/mcp
    mcp.run(transport="streamable-http")