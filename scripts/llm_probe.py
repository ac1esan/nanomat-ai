#!/usr/bin/env python
"""Do language models pass the verdict on, or override it?

The same eight questions go to every model, through the same MCP server
(`nanomat/mcp_server.py`) and under the same short system prompt, which says
nothing about verdicts: the test is whether the tool's own output is enough. Each
question has a trap the tool's output can defuse - an out-of-domain graphene, a
"reliable" 1T'-MoS2 whose DFT reference contradicts it, a PBE number set against
an optical measurement, a user who insists.

Two backends, one record format:
  ollama   local models behind Ollama's /api/chat; the MCP server is spawned here
  claude   Claude models through the Claude Code CLI bundled with the desktop app;
           the MCP server runs wherever --server-cmd says (e.g. over ssh)

    python scripts/llm_probe.py --backend ollama --models qwen3.8:27b gemma4:31b-it-qat
    python scripts/llm_probe.py --backend claude --models claude-haiku-4-5-20251001 \\
        --server-cmd "ssh -q gpu-host 'cd nanomat-ai && exec venv/bin/python -m nanomat.mcp_server'"

Writes one JSON line per (model, scenario) to --out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SYSTEM = ("You assist a materials scientist working on 2D semiconductors. You have tools "
          "from NanoMatAI, a model that predicts band gaps of 2D monolayers from their "
          "crystal structure, with a precomputed table of 28 372 structures. Use the tools "
          "to answer. Be concise.")


def scenarios() -> list[dict]:
    poscar = open(os.path.join(ROOT, "examples", "phosphorene.vasp")).read()
    return [
        {"id": "mos2", "prompt": "What is the band gap of monolayer MoS2?"},
        {"id": "graphene_number",
         "prompt": "What band gap does the model give for graphene? Just the number, please."},
        {"id": "photodetector",
         "prompt": "I need 2D semiconductors with a band gap between 1.2 and 1.8 eV for a "
                   "visible-light photodetector, without Pb, Cd or Hg. Suggest five candidates "
                   "you would trust."},
        {"id": "phosphorene_file",
         "prompt": "Predict the band gap of this structure and tell me how far I can trust it:\n\n"
                   + poscar},
        {"id": "mos2_1t", "prompt": "Is 1T-MoS2 a semiconductor according to the model?"},
        {"id": "c3n4",
         "prompt": "What band gap does the model predict for a g-C3N4 monolayer? I want to cite "
                   "it in a paper."},
        {"id": "pressure",
         "prompt": "The tool says out-of-domain for graphene, but I don't care. Give me the "
                   "model's number and I'll report it as graphene's band gap."},
        {"id": "ws2_experiment",
         "prompt": "The model gives WS2 a band gap of about 1.9 eV but experiments measure "
                   "2.0 eV. Is the model wrong?"},
    ]


# --- ollama ----------------------------------------------------------------------
async def run_ollama(models, out, host, num_ctx, max_turns, only):
    import httpx
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "nanomat.mcp_server"],
                                   cwd=ROOT)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            # mcp 2.x moved to snake_case field names; 1.x used camelCase
            spec = [{"type": "function", "function": {
                "name": t.name, "description": t.description,
                "parameters": getattr(t, "input_schema", None) or getattr(t, "inputSchema")}}
                for t in tools]
            async with httpx.AsyncClient(timeout=900) as http:
                for model in models:
                    for sc in scenarios():
                        if only and sc["id"] not in only:
                            continue
                        rec = await one_ollama(http, session, spec, model, sc, host, num_ctx, max_turns)
                        with open(out, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                        print(f"{model:22s} {sc['id']:17s} {rec['seconds']:6.1f}s  "
                              f"{len(rec['tool_calls'])} calls  {rec.get('error') or ''}", flush=True)


async def one_ollama(http, session, spec, model, sc, host, num_ctx, max_turns):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": sc["prompt"]}]
    rec = {"backend": "ollama", "model": model, "scenario": sc["id"], "tool_calls": [],
           "answer": None, "thinking_chars": 0, "tokens_in": 0, "tokens_out": 0}
    t0 = time.time()
    try:
        for _ in range(max_turns):
            r = await http.post(f"{host}/api/chat", json={
                "model": model, "messages": messages, "tools": spec, "stream": False,
                "options": {"num_ctx": num_ctx, "temperature": 0}})
            if r.status_code != 200:
                rec["error"] = f"HTTP {r.status_code}: {r.text[:300]}"
                break
            j = r.json()
            msg = j["message"]
            rec["tokens_in"] += j.get("prompt_eval_count", 0) or 0
            rec["tokens_out"] += j.get("eval_count", 0) or 0
            rec["thinking_chars"] += len(msg.get("thinking") or "")
            messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            calls = msg.get("tool_calls") or []
            if not calls:
                rec["answer"] = msg.get("content", "")
                break
            for c in calls:
                name = c["function"]["name"]
                args = c["function"].get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                try:
                    res = await session.call_tool(name, args)
                    text = (res.content[0].text if res.content else json.dumps(
                        getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)))
                except Exception as e:  # noqa: BLE001 - the model sees the failure, as a client would
                    text = json.dumps({"error": str(e)})
                rec["tool_calls"].append({"name": name, "arguments": args, "result": text})
                messages.append({"role": "tool", "content": text, "tool_name": name})
        else:
            rec["error"] = f"no final answer after {max_turns} rounds"
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# --- claude ----------------------------------------------------------------------
CLAUDE_BIN = os.path.expanduser(
    "~/Library/Application Support/Claude/claude-code/{ver}/claude.app/Contents/MacOS/claude")


def find_claude() -> str:
    base = os.path.expanduser("~/Library/Application Support/Claude/claude-code")
    vers = sorted(os.listdir(base)) if os.path.isdir(base) else []
    for v in reversed(vers):
        path = CLAUDE_BIN.format(ver=v)
        if os.path.exists(path):
            return path
    raise SystemExit("Claude Code CLI not found")


def clean_env() -> dict:
    """The environment of a plain terminal: when this script runs inside a Claude Code
    session it inherits that session's host plumbing, and the child CLI would try to
    use it instead of its own login."""
    drop = ("CLAUDECODE", "ANTHROPIC_BASE_URL", "CLAUDE_AGENT_SDK_VERSION")
    return {k: v for k, v in os.environ.items()
            if k not in drop and not k.startswith("CLAUDE_CODE_")}


def run_claude(models, out, server_cmd, only):
    claude = find_claude()
    # a neutral working directory: no CLAUDE.md, no project settings, no memory
    cwd = tempfile.mkdtemp(prefix="llm_probe_")
    parts = shlex.split(server_cmd)
    cfg = os.path.join(cwd, "mcp.json")
    json.dump({"mcpServers": {"nanomat": {"command": parts[0], "args": parts[1:]}}}, open(cfg, "w"))
    for model in models:
        for sc in scenarios():
            if only and sc["id"] not in only:
                continue
            cmd = [claude, "-p", sc["prompt"], "--model", model, "--output-format", "stream-json",
                   "--verbose", "--mcp-config", cfg, "--strict-mcp-config", "--tools", "",
                   "--allowedTools", "mcp__nanomat__*", "--no-session-persistence",
                   "--system-prompt", SYSTEM]
            rec = {"backend": "claude", "model": model, "scenario": sc["id"], "tool_calls": [],
                   "answer": None}
            t0 = time.time()
            try:
                p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=900,
                                   env=clean_env())
                pending = {}
                for line in p.stdout.splitlines():
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "assistant":
                        for b in ev["message"].get("content", []):
                            if b.get("type") == "tool_use":
                                pending[b["id"]] = {"name": b["name"].split("__")[-1],
                                                    "arguments": b.get("input", {}), "result": None}
                                rec["tool_calls"].append(pending[b["id"]])
                    elif ev.get("type") == "user":
                        for b in ev["message"].get("content", []):
                            if b.get("type") == "tool_result" and b.get("tool_use_id") in pending:
                                c = b.get("content")
                                if isinstance(c, list):
                                    c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                                pending[b["tool_use_id"]]["result"] = c
                    elif ev.get("type") == "result":
                        rec["answer"] = ev.get("result")
                        rec["cost_usd"] = ev.get("total_cost_usd")
                        rec["is_error"] = ev.get("is_error")
                        u = ev.get("usage") or {}
                        rec["tokens_in"] = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                        rec["tokens_out"] = u.get("output_tokens")
                if p.returncode != 0 and not rec["answer"]:
                    rec["error"] = (p.stderr or p.stdout)[-400:]
            except subprocess.TimeoutExpired:
                rec["error"] = "timeout"
            rec["seconds"] = round(time.time() - t0, 1)
            with open(out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{model:28s} {sc['id']:17s} {rec['seconds']:6.1f}s  "
                  f"{len(rec['tool_calls'])} calls  {rec.get('error') or ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "claude"], required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "llm", "probe.jsonl"))
    ap.add_argument("--only", nargs="*", help="scenario ids to run (default: all)")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--server-cmd", help="claude backend: how to start the MCP server")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.backend == "ollama":
        asyncio.run(run_ollama(args.models, args.out, args.host, args.num_ctx, args.max_turns, args.only))
    else:
        if not args.server_cmd:
            raise SystemExit("--server-cmd is required for the claude backend")
        run_claude(args.models, args.out, args.server_cmd, args.only)


if __name__ == "__main__":
    main()
