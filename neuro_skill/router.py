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

import numpy as np

from neuro_skill.parser import load_skills
from neuro_skill.index import SkillIndex
from neuro_skill.routers import ROUTERS as _ALL_ROUTERS


class SkillRouter:
    """Hybrid skill routing engine — graph + cosine + keyword fusion."""

    def __init__(self, feedback_path: str | None = "~/.neuro-skill-feedback.json"):
        self._skills: list[dict] = []
        self._index = SkillIndex()
        self._built = False
        self._feedback: "ErrorBook | None" = None
        self._feedback_path = feedback_path
        self._personalize: "Personalizer | None" = None

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
        enable_feedback: bool = True,
        **kwargs,
    ) -> list[tuple[str, float]]:
        """Return top-k matching skills for a user query.

        Methods: hybrid, cosine, graph_spread, jaccard, keyword, tfidf
        enable_feedback: apply Error Book corrections (default True)
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

        # Apply Error Book feedback adjustments
        if enable_feedback and method == "hybrid":
            fb = self._get_feedback()
            names = [s["name"] for s in self._skills]
            scores = np.array(fb.adjust(query, scores.tolist(), names),
                              dtype=np.float64)

            # Fourth signal: collaborative filtering personalization
            if self._personalize is not None and self._personalize._trained:
                p_boost = self._personalize.personalize(query)
                if p_boost is not None and len(p_boost) == len(scores):
                    # Insert boost as 4th RRF signal:
                    # create a rank array where boosted skills get lower rank
                    rank_orig = np.zeros(len(scores))
                    order = np.argsort(-scores)
                    for r, idx in enumerate(order):
                        rank_orig[idx] = float(r)
                    # Adjusted rank: original rank * (1 - 0.3 * boost)
                    adj_rank = rank_orig * (1.0 - 0.3 * (p_boost - 0.5))
                    # Convert back to pseudo-scores (lower rank = higher score)
                    scores = 1.0 / (60.0 + adj_rank)

        order = scores.argsort()[::-1][:top_k]
        return [(self._skills[i]["name"], float(scores[i])) for i in order]

    def observe(self, query: str, selected_skill: str):
        """Record implicit feedback — user picked this skill from results."""
        p = self._get_personalize()
        p.observe(query, selected_skill)

    def train_personalize(self):
        """Factorize the implicit feedback matrix. Call after accumulating observations."""
        p = self._get_personalize()
        p.train([s["name"] for s in self._skills])

    def personalize_stats(self) -> dict:
        """Get personalization statistics."""
        return self._get_personalize().stats()

    def learn(self, query: str, preferred_skill: str):
        """Record a user correction — explicit + implicit feedback."""
        fb = self._get_feedback()
        fb.correct(query, preferred_skill)
        self.observe(query, preferred_skill)

    def feedback_stats(self) -> dict:
        """Get Error Book statistics."""
        return self._get_feedback().stats()

    def plan(
        self, query: str, top_k: int = 5, enable_feedback: bool = True
    ) -> "PlanResult":
        """
        One-shot: route → infer deps → topo-sort → execution plan.

        Combines hybrid routing, typed graph edge inference, and
        topological sort into a single call. Returns a PlanResult
        with ordered steps, reasoning, and validity.

        Args:
          query: user query text
          top_k: how many skills to consider
          enable_feedback: apply Error Book corrections

        Returns:
          PlanResult with .steps, .reasoning, .valid, .to_prompt()
        """
        from neuro_skill.typed_graph import auto_discover_edges
        from neuro_skill.planner import plan as _plan, PlanResult

        # Step 1: Route
        ranked = self.query(query, top_k=top_k, method="hybrid",
                            enable_feedback=enable_feedback)

        # Step 2: Infer dependency edges
        edges = auto_discover_edges(self._skills)
        graph = edges.get("depends_on", {})

        # Step 3: Build skill index for planner
        skill_index = {
            name: self.get_skill(name) or {"name": name, "search_text": ""}
            for name, _ in ranked
        }

        # Step 4: Plan
        return _plan(ranked, skill_index, dependency_graph=graph)

    def _get_feedback(self):
        if self._feedback is None:
            from neuro_skill.feedback import ErrorBook
            self._feedback = ErrorBook(self._feedback_path)
        return self._feedback

    def _get_personalize(self):
        if self._personalize is None:
            from neuro_skill.personalize import Personalizer
            path = str(self._feedback_path).replace("-feedback", "-personalize") if self._feedback_path else "~/.neuro-skill-personalize.json"
            self._personalize = Personalizer(path)
        return self._personalize

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
