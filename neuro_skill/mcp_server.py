"""
MCP (Model Context Protocol) server — zero-integration Agent access.

Agents (Claude Code, Cursor, Codex, Hermes) auto-discover this tool via
their MCP configuration. No import, no HTTP, no CLI — the agent calls
neuroskill_query(...) directly.

Add to your Agent's MCP config:
  Claude Code:  ~/.claude/mcp.json -> {"mcpServers": {"neuro-skill": {...}}}
  Cursor:       .cursor/mcp.json
  Codex:        ~/.codex/mcp.json

Architecture (inspired by CodeGraph):
  stdio JSON-RPC server → parse request → router.query() → structured result
"""

from __future__ import annotations

import json, sys, os, time
from pathlib import Path
from typing import Any

# ── MCP protocol constants ──
JSONRPC_VERSION = "2.0"
SERVER_NAME = "neuro-skill"
SERVER_VERSION = "0.4.0"

# ── Tool definitions (exposed to agents) ──
TOOLS = [
    {
        "name": "neuroskill_query",
        "description": (
            "Find the most relevant skills/agents for a user query. "
            "Returns top-ranked skills with confidence scores. "
            "Use this when you need to determine which skill/agent to invoke "
            "for a given task. Input: natural language query. "
            "Output: ranked list of skill names with scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question or task description to route to a skill",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of skills to return (default 5, max 10)",
                },
                "method": {
                    "type": "string",
                    "enum": ["hybrid", "cosine", "graph_spread", "jaccard", "keyword"],
                    "default": "hybrid",
                    "description": "Routing method. hybrid is the default and best general-purpose option",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "neuroskill_compare",
        "description": (
            "Compare two or more skills side-by-side. Given a query, returns "
            "which skill(s) best match and WHY — showing feature overlap. "
            "Useful when multiple skills could apply and you need to decide."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user's question"},
                "skill_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of skill names to compare",
                },
            },
            "required": ["query", "skill_names"],
        },
    },
    {
        "name": "neuroskill_status",
        "description": (
            "Get current neuro-skill index status: skill count, feature count, "
            "graph density, build time, query statistics. "
            "Use this to check if an index rebuild is needed (e.g., after "
            "adding new skills)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "neuroskill_predict",
        "description": (
            "Predict which skills are relevant for a given source file. "
            "Extracts imports, decorators, and class/function names from the file, "
            "then returns pre-loaded skills that match. "
            "Use this at session start to warm up the skill router with "
            "context-relevant skills before the user even asks a question. "
            "Input: absolute path to a source file. "
            "Output: skills matching the code context detected in the file."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the source file to analyze",
                },
            },
            "required": ["file_path"],
        },
    },
]


# ── Lazy router (same as autostart.py) ──

_router: Any = None
_router_stats: dict = {}


def _get_router():
    global _router, _router_stats
    if _router is None:
        from neuro_skill import SkillRouter
        dirs = _resolve_dirs()
        _router = SkillRouter()
        _router_stats = _router.build(dirs)
    return _router


def _resolve_dirs():
    env = os.environ.get("NEURO_SKILL_DIRS", "")
    if env:
        return [d.strip() for d in env.split(":") if d.strip()]
    return [
        os.path.expanduser("~/.claude/skills/"),
        os.path.expanduser("~/.claude/agents/"),
        os.path.expanduser("~/.claude/.agents/skills/"),
    ]


# ── Tool handlers ──

def _handle_query(args: dict) -> dict:
    router = _get_router()
    q = args["query"]
    top_k = min(args.get("top_k", 5), 10)
    method = args.get("method", "hybrid")
    results = router.query(q, top_k=top_k, method=method)

    skills = [{"name": name, "score": round(score, 4)} for name, score in results]
    return {
        "query": q,
        "method": method,
        "top_match": skills[0] if skills else None,
        "skills": skills,
        "n_total": router.skill_count,
    }


def _handle_compare(args: dict) -> dict:
    router = _get_router()
    import neuro_skill.features as fmod

    q = args["query"]
    qf = fmod.extract_query_features(q)
    q_set = fmod.feature_set(qf)

    comparisons = []
    for name in args["skill_names"]:
        skill = router.get_skill(name)
        if not skill:
            comparisons.append({"name": name, "error": "not found"})
            continue

        sf = fmod.extract_skill_features(skill)
        s_set = fmod.feature_set(sf)
        overlap = q_set & s_set
        jac = len(overlap) / max(len(q_set | s_set), 1) if q_set or s_set else 0.0

        # Also get raw score from hybrid
        results = router.query(q, top_k=min(10, router.skill_count))
        score = next((s for n, s in results if n == name), 0.0)

        comparisons.append({
            "name": name,
            "score": round(score, 4),
            "jaccard": round(jac, 4),
            "shared_features": sorted(overlap),
            "skill_features": sorted(s_set),
            "query_features": sorted(q_set),
        })

    comparisons.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {"query": q, "comparisons": comparisons}


def _handle_status(_args: dict) -> dict:
    router = _get_router()
    return {
        "n_skills": router.skill_count,
        "n_features": _router_stats.get("n_features", "?"),
        "n_broad": _router_stats.get("n_broad", "?"),
        "n_precise": _router_stats.get("n_precise", "?"),
        "graph_density": round(_router_stats.get("graph_density", 0), 4),
        "build_time_s": round(_router_stats.get("total_time_s", 0), 2),
    }


def _handle_predict(args: dict) -> dict:
    """Predict relevant skills from a source file's code context."""
    file_path = args["file_path"]
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return {"error": f"File not found: {file_path}", "skills": []}
    if path.suffix not in (".py", ".ts", ".js", ".go", ".rs", ".java", ".kt", ".swift", ".cpp", ".c", ".rb", ".php"):
        return {"file": str(path), "skills": [], "hint": f"File type '{path.suffix}' not supported for AST analysis. Currently supports: py, ts, js, go, rs, java, kt, swift, cpp, c, rb, php."}

    # Extract code features from file content
    import re
    text = path.read_text(encoding="utf-8", errors="ignore")
    keywords = set()

    # Imports
    for m in re.finditer(r'(?:import|from)\s+(\w+)', text):
        kw = m.group(1).lower()
        if len(kw) >= 2: keywords.add(kw)
    # Decorators/annotations
    for m in re.finditer(r'@(\w+)', text):
        keywords.add(m.group(1).lower())
    # Class names
    for m in re.finditer(r'class\s+(\w+)', text):
        parts = re.findall(r'[A-Z][a-z]+', m.group(1))
        keywords.update(p.lower() for p in parts)
    # Function names (public)
    for m in re.finditer(r'def\s+(\w+)', text):
        name = m.group(1)
        if not name.startswith("_"):
            keywords.update(re.findall(r'[a-z]{3,}', name.lower()))

    # Also try CodeGraph if available
    cg_keywords = set()
    try:
        import subprocess
        result = subprocess.run(
            ["codegraph", "query", "--kind", "import,class,function", "--json"],
            cwd=str(path.parent),
            capture_output=True, text=True, timeout=8,
        )
        if result.returncode == 0 and result.stdout.strip():
            cg_data = json.loads(result.stdout)
            for item in (cg_data if isinstance(cg_data, list) else cg_data.get("results", [])):
                name = item.get("name", "") if isinstance(item, dict) else str(item)
                cg_keywords.update(re.findall(r'[a-z]{3,}', name.lower()))
    except Exception:
        pass  # CodeGraph not available is fine

    all_kw = sorted(keywords | cg_keywords)[:30]
    if not all_kw:
        return {"file": str(path), "skills": [], "keywords": [], "hint": "No code features detected."}

    # Route: build a pseudo-query from keywords and find matching skills
    router = _get_router()
    pseudo_query = " ".join(all_kw)
    results = router.query(pseudo_query, top_k=8, method="hybrid")

    return {
        "file": str(path),
        "keywords": all_kw,
        "skills": [{"name": name, "score": round(score, 4)} for name, score in results],
        "hint": "Pre-load these skills for this file's context",
    }


TOOL_HANDLERS = {
    "neuroskill_query": _handle_query,
    "neuroskill_compare": _handle_compare,
    "neuroskill_status": _handle_status,
    "neuroskill_predict": _handle_predict,
}


# ── JSON-RPC server (stdio) ──

def _send_rpc(data: dict):
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run_stdio_server():
    """Main MCP server loop — reads JSON-RPC from stdin, writes to stdout."""
    _send_rpc({
        "jsonrpc": JSONRPC_VERSION,
        "method": "log",
        "params": {"message": f"{SERVER_NAME} v{SERVER_VERSION} MCP server started"},
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = request.get("id")
        method = request.get("method", "")

        # ── Initialize ──
        if method == "initialize":
            _send_rpc({
                "jsonrpc": JSONRPC_VERSION, "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                },
            })

        # ── List tools ──
        elif method == "tools/list":
            _send_rpc({
                "jsonrpc": JSONRPC_VERSION, "id": req_id,
                "result": {"tools": TOOLS},
            })

        # ── Call tool ──
        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})

            handler = TOOL_HANDLERS.get(tool_name)
            if handler:
                try:
                    result = handler(tool_args)
                    _send_rpc({
                        "jsonrpc": JSONRPC_VERSION, "id": req_id,
                        "result": {
                            "content": [{
                                "type": "text",
                                "text": json.dumps(result, ensure_ascii=False),
                            }],
                        },
                    })
                except Exception as e:
                    _send_rpc({
                        "jsonrpc": JSONRPC_VERSION, "id": req_id,
                        "error": {"code": -1, "message": str(e)},
                    })
            else:
                _send_rpc({
                    "jsonrpc": JSONRPC_VERSION, "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                })

        # ── Shutdown ──
        elif method == "shutdown":
            _send_rpc({"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": None})
            break

        # ── Notifications (no id) ──
        elif method == "notifications/initialized":
            pass  # ack only, no response
        elif method == "log":
            pass

        else:
            _send_rpc({
                "jsonrpc": JSONRPC_VERSION, "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            })

    # Clean shutdown
    sys.exit(0)


# ── CLI entry ──

def cmd_mcp():
    """CLI: neuro-skill mcp — start stdio MCP server."""
    import argparse
    p = argparse.ArgumentParser(description="neuro-skill MCP server")
    p.add_argument("--version", action="version", version=f"{SERVER_NAME} v{SERVER_VERSION}")
    args = p.parse_args()

    # Re-open stderr to avoid polluting stdout (which is the JSON-RPC channel)
    sys.stderr = open(os.devnull, "w") if not sys.stderr.isatty() else sys.stderr

    run_stdio_server()


if __name__ == "__main__":
    cmd_mcp()
