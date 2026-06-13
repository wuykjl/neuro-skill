"""
SkillRouter — Graph+Feature hybrid skill routing engine.

Usage:
    from neuro_skill import SkillRouter

    router = SkillRouter()
    router.build(["~/.claude/skills/", "~/.claude/agents/"])
    results = router.query("check Python code for SQL injection", top_k=5)
    for name, score in results:
        print(f"  {name}: {score:.3f}")
"""

from __future__ import annotations

import time
from pathlib import Path

from neuro_skill.parser import load_skills
from neuro_skill.index import SkillIndex
from neuro_skill.routers import ROUTERS as _ALL_ROUTERS


class SkillRouter:
    """Hybrid skill routing engine — graph + cosine + keyword fusion."""

    def __init__(self):
        self._skills: list[dict] = []
        self._index = SkillIndex()
        self._built = False

    # ── Build ──

    def build(self, directories: list[str], rank: int = 8) -> dict:
        """Load skills from directories and build the index."""
        t0 = time.time()
        self._skills = load_skills(directories)
        if not self._skills:
            raise RuntimeError(f"No skills found in {directories}")
        stats = self._index.build(self._skills, rank=rank)
        self._built = True
        stats["total_time_s"] = round(time.time() - t0, 3)
        return stats

    def build_from_skills(self, skills: list[dict], rank: int = 8) -> dict:
        """Build index from pre-loaded skill dicts."""
        self._skills = skills
        self._index = SkillIndex()
        stats = self._index.build(self._skills, rank=rank)
        self._built = True
        return stats

    # ── Query ──

    def query(
        self,
        query: str,
        top_k: int = 10,
        method: str = "hybrid",
        **kwargs,
    ) -> list[tuple[str, float]]:
        """Return top-k matching skills for a user query.

        Methods: hybrid, cosine, graph_spread, jaccard, keyword, tfidf
        """
        if not self._built:
            raise RuntimeError("Index not built. Call .build() first.")
        if method not in _ALL_ROUTERS:
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(_ALL_ROUTERS)}"
            )

        scores = _ALL_ROUTERS[method](
            self._skills, query,
            F=self._index.F, G=self._index.G, meta=self._index.meta,
            cp_weights=self._index.cp_weights,
            cp_factors=self._index.cp_factors,
            **kwargs,
        )
        order = scores.argsort()[::-1][:top_k]
        return [(self._skills[i]["name"], float(scores[i])) for i in order]

    # ── Info ──

    @property
    def skill_count(self) -> int:
        return len(self._skills)

    @property
    def skill_names(self) -> list[str]:
        return [s["name"] for s in self._skills]

    def get_skill(self, name: str) -> dict | None:
        for s in self._skills:
            if s["name"] == name:
                return s
        return None

    # ── Persist ──

    def save(self, path: str):
        self._index.save(path)

    @classmethod
    def load(cls, path: str, directories: list[str]) -> "SkillRouter":
        router = cls()
        router._skills = load_skills(directories)
        router._index = SkillIndex.load(path, router._skills)
        router._built = True
        return router
