#!/usr/bin/env python
"""
neuro-skill query entry — drop-in replacement for trigger-word matching.

Usage:
  python query_skill.py "check Python code for SQL injection"
  echo "飞书发消息给张三" | python query_skill.py --top 5

Loads your extras automatically from extras_template.py.
Edit extras_template.py to add your own domain keywords.
"""

import sys, json, time, argparse, os
from pathlib import Path

# ── Production config ──
# Skill directories are read from NEURO_SKILL_DIRS env var (colon-separated),
# or default to Claude Code standard paths.
_DEFAULT_DIRS = ":".join([
    os.path.expanduser("~/.claude/skills/"),
    os.path.expanduser("~/.claude/agents/"),
    os.path.expanduser("~/.claude/.agents/skills/"),
])


def _load_extras():
    """Load user extras from extras_template.py if available."""
    try:
        from extras_template import EXTRA_BROAD, EXTRA_PRECISE
        return dict(EXTRA_BROAD), dict(EXTRA_PRECISE)
    except ImportError:
        return {}, {}


def get_router():
    from neuro_skill.features import BROAD, PRECISE
    eb, ep = _load_extras()
    BROAD.update(eb)
    PRECISE.update(ep)

    # Resolve skill dirs from env or default
    env_dirs = os.environ.get("NEURO_SKILL_DIRS", _DEFAULT_DIRS)
    prod_dirs = [d for d in env_dirs.split(":") if d.strip()]

    from neuro_skill import SkillRouter
    router = SkillRouter()
    router.build(prod_dirs)
    return router


_router = None


def query_skills(query_text: str, top_k: int = 5, method: str = "hybrid") -> dict:
    global _router
    if _router is None:
        _router = get_router()
    t0 = time.time()
    results = _router.query(query_text, top_k=top_k, method=method)
    elapsed_ms = (time.time() - t0) * 1000
    return {
        "skills": [{"name": name, "score": round(score, 4)} for name, score in results],
        "query": query_text,
        "time_ms": round(elapsed_ms, 1),
        "n_total": _router.skill_count,
    }


def main():
    p = argparse.ArgumentParser(description="neuro-skill query")
    p.add_argument("query", nargs="?", help="Query text (or pipe via stdin)")
    p.add_argument("--top", "-k", type=int, default=5)
    p.add_argument("--method", "-m", default="hybrid",
                   choices=["hybrid", "cosine", "graph_spread", "jaccard", "keyword"])
    p.add_argument("--format", default="json", choices=["json", "text"])
    args = p.parse_args()

    q = args.query or sys.stdin.read().strip()
    if not q:
        print(json.dumps({"error": "No query"}, ensure_ascii=False))
        sys.exit(1)

    r = query_skills(q, top_k=args.top, method=args.method)
    if args.format == "text":
        for s in r["skills"]:
            print(f"{s['name']}: {s['score']:.4f}")
    else:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
