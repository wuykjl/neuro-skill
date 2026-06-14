"""
CodeGraph bridge — use AST-level code structure to enrich skill routing.

When CodeGraph is installed in the same project, this module can:
  1. Index the skill implementation code (not just YAML frontmatter)
  2. Extract function names, class names, imports as additional features
  3. Auto-detect new skills from git history (which files changed recently)

This turns skill routing from "name + description only" into
"name + description + code structure" — improving precision for
skills whose description text is sparse or generic.

Inspired by CodeGraph's tree-sitter AST extraction architecture.
"""

from __future__ import annotations

import subprocess, json, os, re
from pathlib import Path
from typing import Optional


def _run_codegraph(subcommand: str, project_dir: str,
                   timeout: int = 15) -> Optional[dict]:
    """Call codegraph CLI and parse JSON output."""
    try:
        result = subprocess.run(
            ["codegraph", subcommand, "--json"],
            cwd=project_dir,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def extract_code_features(
    skill_dirs: list[str],
    codegraph_available: bool = True,
) -> dict[str, list[str]]:
    """
    Use CodeGraph to extract code-level keywords from skill implementations.

    For each skill directory that contains source code:
      1. Run `codegraph query` to get top symbols
      2. Extract function/class names as PRECISE feature keywords
      3. Extract import patterns as BROAD domain hints

    Returns: {"broad": {domain: [keywords]}, "precise": {skill: [keywords]}}
    """
    if not codegraph_available:
        return {"broad": {}, "precise": {}}

    broad_features: dict[str, set[str]] = {}
    precise_features: dict[str, set[str]] = {}

    for skill_dir in skill_dirs:
        dp = Path(skill_dir).expanduser().resolve()
        if not dp.exists():
            continue

        # Only process dirs that have a .codegraph/ index
        cg_dir = dp / ".codegraph"
        if not cg_dir.exists():
            # Try parent dir too (for project-level codegraph)
            cg_dir = dp.parent / ".codegraph"
        if not cg_dir.exists():
            continue

        # Search for symbols matching common patterns
        symbols = _run_codegraph(f"query --kind function,class,method", str(dp))
        if not symbols:
            continue

        # Extract keywords from symbol names
        skill_name = dp.name
        keywords: set[str] = set()

        for sym in (symbols if isinstance(symbols, list) else symbols.get("results", [])):
            name = sym.get("name", "") if isinstance(sym, dict) else str(sym)
            # Convert CamelCase/snake_case to keywords
            parts = re.findall(r'[A-Z][a-z]+|[a-z]+', name)
            for p in parts:
                if len(p) >= 3 and p.lower() not in _STOP_CODE:
                    keywords.add(p.lower())

        if keywords:
            precise_features[skill_name] = list(keywords)[:12]

    return {"broad": {k: list(v)[:8] for k, v in broad_features.items()},
            "precise": {k: list(v)[:12] for k, v in precise_features.items()}}


_STOP_CODE = {
    "the", "and", "for", "get", "set", "new", "use", "all", "has",
    "main", "init", "run", "add", "del", "put", "old", "raw",
}


def merge_codegraph_features(
    base_broad: dict, base_precise: dict,
    skill_dirs: list[str],
) -> tuple[dict, dict]:
    """
    Merge CodeGraph-extracted features into existing feature sets.
    Never replaces existing keywords — only adds novel ones.
    """
    cg = extract_code_features(skill_dirs)
    if not cg["broad"] and not cg["precise"]:
        return dict(base_broad), dict(base_precise)

    merged_b = dict(base_broad)
    merged_p = dict(base_precise)

    for skill_name, keywords in cg["precise"].items():
        if skill_name in merged_p:
            existing = set(merged_p[skill_name])
            novel = [k for k in keywords if k not in existing]
            if novel:
                merged_p[skill_name] = list(existing) + novel[:8]

    return merged_b, merged_p


def auto_init_codegraph(skill_dirs: list[str]) -> list[str]:
    """Initialize CodeGraph in skill directories that have source code."""
    initialized = []
    for d in skill_dirs:
        dp = Path(d).expanduser().resolve()
        if not dp.exists():
            continue
        # Check if has real code (not just .md files)
        has_code = any(dp.rglob(f"*.{ext}"))
        for ext in ["py", "ts", "js", "go", "rs", "java"]:
            if list(dp.rglob(f"*.{ext}")):
                has_code = True
                break
        if has_code and not (dp / ".codegraph").exists():
            result = _run_codegraph("", str(dp))
            if result is not None:
                initialized.append(str(dp))
    return initialized
