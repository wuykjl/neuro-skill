"""
Typed skill graph — depends_on + complements edges.

Inspired by agent-skill-finder's typed edge model (depends_on, complements).

neuro-skill's original graph G is undirected — all edges mean "similar."
ASF proved two edge types are essential:
  - depends_on:  hard dependency (skill A requires skill B first)
  - complements: soft similarity (skill A is often used with skill B)

This module builds both edge types from skill frontmatter metadata,
then extends the hybrid router to use typed graph diffusion.
"""

from __future__ import annotations

import re, json
import numpy as np
from pathlib import Path
from typing import Optional


def parse_skill_dependencies(skill: dict) -> dict:
    """
    Extract dependency info from a skill's YAML frontmatter.

    Looks for these optional fields in skill metadata:
      requires: [skill_name, ...]        → depends_on edges
      complements: [skill_name, ...]     → complements edges
      provides: [capability, ...]        → for I/O planner
      needs: [capability, ...]           → for I/O planner

    If no explicit edges declared, falls back to heuristic:
      - "auth" in name/desc → depends_on auth-skill
      - skill names containing framework names → depends_on framework skills
    """
    # Check if skill dict has frontmatter metadata
    meta = skill.get("_meta", {})
    if not meta:
        # Try to parse from search_text (name + description + triggers)
        # For agent files with full YAML frontmatter, try the raw body
        body = skill.get("_body", "")
        if body and body.startswith("---"):
            parts = body.split("---", 2)
            if len(parts) >= 3:
                try:
                    import yaml
                    meta = yaml.safe_load(parts[1]) or {}
                except Exception:
                    meta = {}

    return {
        "depends_on": meta.get("requires", meta.get("depends_on", [])),
        "complements": meta.get("complements", []),
        "provides": meta.get("provides", []),
        "needs": meta.get("needs", meta.get("inputs", [])),
    }


def _name_to_idx(skills: list[dict], name: str) -> Optional[int]:
    """Resolve a skill name (or partial match) to index."""
    for i, s in enumerate(skills):
        if s["name"] == name:
            return i
    # Partial match
    for i, s in enumerate(skills):
        if name.lower() in s["name"].lower():
            return i
    return None


def build_typed_matrices(skills: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build G_depends (N×N) and G_complements (N×N) from skill metadata.

    G_depends[i][j] = 1.0 if skill_i requires skill_j (unweighted, hard dep)
    G_complements[i][j] = similarity + explicit complements edges (weighted)

    Returns (G_depends, G_complements).
    """
    N = len(skills)
    G_d = np.zeros((N, N), dtype=np.float64)
    G_c = np.zeros((N, N), dtype=np.float64)

    # Parse each skill's dependencies
    edges_found = 0
    for i, skill in enumerate(skills):
        deps = parse_skill_dependencies(skill)

        # depends_on edges (hard, unweighted)
        for dep_name in deps["depends_on"]:
            j = _name_to_idx(skills, dep_name)
            if j is not None and j != i:
                G_d[i, j] = 1.0
                edges_found += 1

        # complements edges (soft, weighted by keyword overlap)
        for comp_name in deps["complements"]:
            j = _name_to_idx(skills, comp_name)
            if j is not None and j != i:
                # Weight by Jaccard of search_text keywords
                from neuro_skill.features import tokenize
                si = tokenize(skill.get("search_text", ""))
                sj = tokenize(skills[j].get("search_text", ""))
                jac = len(si & sj) / max(len(si | sj), 1) if si or sj else 0.5
                G_c[i, j] = max(G_c[i, j], 0.3 + 0.7 * jac)
                edges_found += 1

    return G_d, G_c


def typed_graph_spread(
    skills: list[dict],
    query: str,
    G_complements: np.ndarray,
    G_depends: Optional[np.ndarray] = None,
    F: Optional[np.ndarray] = None,
    meta: Optional[dict] = None,
    steps: int = 3,
    decay: float = 0.5,
) -> tuple[np.ndarray, list[int]]:
    """
    Typed graph spreading activation.

    - Complements edges: soft diffusion, same as original spreading activation
    - Depends_on edges: hard pull — when a skill scores high, its dependencies
      also get boosted (not diffused, just added)

    Returns (scores, required_dep_indices) where required_dep_indices are
    skill indices that MUST be included regardless of score.
    """
    from neuro_skill.routers import graph_spread, _normalize

    # Phase 1: complement spreading (same as original)
    scores = graph_spread(skills, query, G=G_complements if G_complements.any() else None,
                          steps=steps, decay=decay)
    scores = _normalize(scores)

    # Phase 2: dependency pull — for top-k skills, pull in their deps
    required = set()
    if G_depends is not None and G_depends.any():
        top_k = min(10, len(skills))
        top_idx = np.argsort(scores)[-top_k:]
        for i in top_idx:
            if scores[i] > 0.1:
                for j in range(len(skills)):
                    if G_depends[i, j] > 0.5:
                        required.add(j)
                        scores[j] = max(scores[j], scores[i] * 0.7)

    scores = _normalize(scores)
    return scores, sorted(required)


# ── Edge discovery from existing skills ──

def auto_discover_edges(skills: list[dict]) -> dict:
    """
    Heuristic edge discovery when no explicit edges are declared.

    Rules:
      1. If skill name contains "auth" → depends_on any skill named "*auth*"
      2. If skill name contains a language (python, go, etc.) and another
         skill has the same language in its name → complements edge
      3. "review" skills → complements with "test" skills
    """
    N = len(skills)
    depends_on: dict[str, list[str]] = {}
    complements: dict[str, list[str]] = {}

    lang_patterns = [
        "python", "go", "rust", "java", "kotlin", "swift", "dart",
        "react", "vue", "angular", "django", "fastapi", "spring",
        "flutter", "php", "cpp", "csharp", "typescript",
    ]

    for i, s1 in enumerate(skills):
        n1 = s1["name"]
        deps = []
        comps = []

        for j, s2 in enumerate(skills):
            if i == j:
                continue
            n2 = s2["name"]

            # Rule 1: auth dependency
            if "auth" in n1.lower() and "auth" in n2.lower():
                deps.append(n2)

            # Rule 2: same language family
            for lang in lang_patterns:
                if lang in n1.lower() and lang in n2.lower():
                    comps.append(n2)
                    break

            # Rule 3: review ↔ test
            if ("review" in n1.lower() and "test" in n2.lower()) or \
               ("test" in n1.lower() and "review" in n2.lower()):
                comps.append(n2)

        if deps:
            depends_on[n1] = list(set(deps))
        if comps:
            complements[n1] = list(set(comps))[:5]

    return {"depends_on": depends_on, "complements": complements}
