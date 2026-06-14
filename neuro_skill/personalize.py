"""
Implicit feedback + collaborative filtering — fourth RRF signal.

Learns from which skills users actually pick (not just top-ranked ones).
Builds a query→skill preference matrix, factorizes with ALS, and
produces personalized score boosts for each query.

Pattern: context → impute → boost
  - Observe: record (query_hash, skill_name) each time a skill is selected
  - Factorize: ALS on query×skill implicit matrix (CPU, <1s for 1000 entries)
  - Personalize: for a new query, use similar queries' skill preferences as boost

Pure CPU. pip install implicit. Sub-millisecond inference.

Usage:
  from neuro_skill.personalize import Personalizer

  p = Personalizer()
  p.observe("review python code", "python-reviewer")
  p.observe("review python code", "code-reviewer")  # reinforce
  p.train()  # factorize the matrix

  boosts = p.personalize("check python code for bugs")
  # → boosts for skills frequently selected for similar queries
"""

from __future__ import annotations

import os, json, time, threading
from pathlib import Path
from typing import Optional

import numpy as np
from collections import defaultdict

from neuro_skill.features import query_hash as query_key


class Personalizer:
    """Learn user skill preferences from implicit feedback.

    Implements Alternating Least Squares (ALS) via 'implicit' library.
    Falls back to simple co-occurrence counts if library not available.

    State persisted to ~/.neuro-skill-feedback.json (same as ErrorBook).
    """

    def __init__(self, path: str = "~/.neuro-skill-personalize.json"):
        self._path = Path(path).expanduser()
        self._observations: dict[str, dict[str, int]] = {}
        self._skill_names: list[str] = []
        self._query_ids: list[str] = []
        self._model: Optional[object] = None
        self._item_factors: Optional[np.ndarray] = None
        self._thompson_beta: dict = {}
        self._trained = False
        self._loaded = False

    # ── Observe ──────────────────────────

    def _load(self):
        if not self._loaded and self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._observations = data.get("obs", {})
            except (json.JSONDecodeError, IOError):
                pass
        self._loaded = True

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"obs": self._observations}, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self._path)

    def observe(self, query: str, skill_name: str, weight: int = 1):
        """Record a skill selection event. Updates Thompson priors live.

        Thompson Sampling with Beta(alpha, beta) prior:
          - Picked skill: alpha += weight (success evidence)
          - All skills start at uninformed Beta(1, 1)
          - Competitive Beta penalty: other skills get small beta increase
            (they were visible but not picked)
        """
        self._load()
        qk = query_key(query)
        if qk not in self._observations:
            self._observations[qk] = {}
        self._observations[qk][skill_name] = (
            self._observations[qk].get(skill_name, 0) + weight
        )

        # Live Thompson update
        if hasattr(self, '_thompson_beta'):
            if qk not in self._thompson_beta:
                self._thompson_beta[qk] = {}
            alpha_old, beta_old = self._thompson_beta[qk].get(
                skill_name, (1.0, 1.0)
            )
            # Picked skill: boost alpha by weight (success)
            self._thompson_beta[qk][skill_name] = (
                alpha_old + weight, beta_old
            )
            # Competitors: small beta increase (visible but not picked)
            for sn in list(self._thompson_beta[qk].keys()):
                if sn != skill_name:
                    a, b = self._thompson_beta[qk][sn]
                    self._thompson_beta[qk][sn] = (a, b + weight * 0.2)

    # ── Train ────────────────────────────

    def train(self, skill_names: list[str], method: str = "thompson"):
        """Train the model. ALS factorizes a confidence matrix; Thompson
        Sampling (default) learns Beta distributions from every observation.

        skill_names: ordered list of all possible skills (index mapping).
        method: "thompson" (default) or "als".
        """
        self._skill_names = list(skill_names)
        self._load()

        if not self._observations:
            self._trained = False
            return

        if method == "thompson":
            self._trained = self._train_thompson(skill_names)
            return

        n_skills = len(skill_names)
        skill_to_idx = {name: i for i, name in enumerate(skill_names)}

        # Build confidence matrix: rows=query_hashes, cols=skills
        self._query_ids = sorted(self._observations.keys())
        n_queries = len(self._query_ids)
        M = np.zeros((n_queries, n_skills), dtype=np.float64)

        for qi, qk in enumerate(self._query_ids):
            for sn, count in self._observations[qk].items():
                if sn in skill_to_idx:
                    M[qi, skill_to_idx[sn]] = 1.0 + np.log1p(count)

        if M.sum() < 1:
            self._trained = False
            return

        # Try ALS via implicit library
        try:
            import scipy.sparse as sp
            from implicit.als import AlternatingLeastSquares

            sparse_M = sp.csr_matrix(M.astype(np.float32))
            factors = min(64, max(4, n_skills // 2))
            model = AlternatingLeastSquares(factors=factors,
                                            regularization=0.1, iterations=15,
                                            random_state=42)
            model.fit(sparse_M, show_progress=False)
            self._model = model
            # user_factors: (n_queries, factors), item_factors: (n_skills, factors)
            self._user_factors = model.user_factors.copy()    # for query→embedding lookup
            self._item_factors = model.item_factors.copy()    # kept for reference
            self._query_to_embedding = {}
            for qi, emb in enumerate(self._user_factors):
                self._query_to_embedding[self._query_ids[qi]] = emb
            self._trained = True
            return
        except ImportError:
            pass  # fallback

        # Fallback: co-occurrence — M'(n_skills, n_queries) @ preferences → boost
        # For each query, preferences = normalized column of M weighted by query similarity
        cooc = M.T @ M  # (n_skills, n_skills)
        norms = np.linalg.norm(cooc, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        self._cooc = cooc / norms
        self._M = M  # (n_queries, n_skills) — raw confidence matrix
        self._trained = True

    # ── Thompson Sampling (default, learned from every observation) ──

    def _train_thompson(self, skill_names: list[str]) -> bool:
        """Minimal Beta(alpha, beta) priors from observation counts.

        alpha = 1 + count(picked)  — success evidence (starts at 1)
        beta  = 1                   — uninformed prior

        After just 1 observation, expected = (1+1)/(1+1+1) = 2/3 = 0.67.
        An unobserved skill gets alpha=1, beta=1 → expected 0.50.

        This lets Thompson produce differentiated signals from the
        very first data point — no warm-up period needed.
        """
        self._thompson_beta: dict[str, dict[str, tuple[float, float]]] = {}
        self._trained = True

        if not self._observations:
            return True

        for qk, skill_counts in self._observations.items():
            self._thompson_beta[qk] = {}
            for sn, count in skill_counts.items():
                self._thompson_beta[qk][sn] = (1.0 + float(count), 1.0)

        return True

    # ── Personalize ──────────────────────

    def personalize(self, query: str) -> np.ndarray:
        """Produce a per-skill boost vector of length n_skills.

        Thompson path: sample Beta(a,b) for each skill, rank by score.
        ALS path:      user_embedding @ item_factors.T → normalize
        Cooc path:     weighted column of M → normalize
        """
        n = len(self._skill_names)
        if not self._trained or n == 0:
            return np.ones(n) * 0.5

        qk = query_key(query)

        # ── Thompson Sampling path (primary, no external dependency) ──
        if hasattr(self, '_thompson_beta') and self._thompson_beta:
            # Find the closest matching query hash (exact or partial)
            beta_entry = self._thompson_beta.get(qk)
            if beta_entry is None:
                # Aggregate across all observations weighted by query similarity
                boost = np.ones(n) * 0.5
                skill_idx = {name: i for i, name in enumerate(self._skill_names)}
                total_weight = 0.0
                for obs_qk, obs_skills in self._thompson_beta.items():
                    # Simple overlap: shared tokens between queries
                    obs_tokens = set(obs_qk.split())
                    q_tokens = set(qk.split())
                    overlap = len(obs_tokens & q_tokens) / max(len(obs_tokens | q_tokens), 1)
                    if overlap < 0.1:
                        continue
                    w = overlap
                    total_weight += w
                    for sn, (alpha, beta) in obs_skills.items():
                        if sn in skill_idx:
                            # Thompson sample: draw from Beta(alpha, beta)
                            sample = np.random.beta(alpha, beta)
                            boost[skill_idx[sn]] += w * sample
                if total_weight > 0:
                    boost = boost / total_weight
                    boost = np.clip(boost, 0.3, 1.0)
                return boost

            # Exact query hash match: sample Beta for each observed skill
            skill_idx = {name: i for i, name in enumerate(self._skill_names)}
            boost = np.ones(n) * 0.3  # neutral baseline
            for sn, (alpha, beta) in beta_entry.items():
                if sn in skill_idx:
                    # Thompson sample from Beta(alpha, beta)
                    boost[skill_idx[sn]] = np.random.beta(alpha, beta)
            # Clamp to reasonable range
            return np.clip(boost, 0.3, 1.0)

        # ── ALS path ──
        if self._model is not None:
            # Try to get embedding for this query
            emb = None
            if qk in self._query_to_embedding:
                emb = self._query_to_embedding[qk]

            if emb is not None and self._item_factors is not None:
                # emb: (factors,), item_factors: (n_skills, factors)
                scores = self._item_factors @ emb   # → (n_skills,)
                boost = np.where(scores > 0, scores / max(scores.max(), 1), 0.3)
                return boost.astype(np.float64)

        # ── Fallback co-occurrence path ──
        if hasattr(self, '_M') and self._M is not None:
            if qk in self._query_ids:
                qi = self._query_ids.index(qk)
                # This query's observed preferences as a binary vector
                query_vec = self._M[qi]  # (n_skills,)
                if query_vec.max() > 0:
                    # Co-occurrence-weighted boost
                    boost = self._cooc.T @ query_vec  # (n_skills,) — skills similar to this query's picks
                    if boost.max() > boost.min():
                        boost = (boost - boost.min()) / (boost.max() - boost.min())
                    return 0.3 + 0.7 * boost

        # ── Unknown query: look up observations directly ──
        skill_idx = {name: i for i, name in enumerate(self._skill_names)}
        boost = np.zeros(n)
        for sn, count in self._observations.get(qk, {}).items():
            if sn in skill_idx:
                boost[skill_idx[sn]] += count
        if boost.max() > 0:
            return 0.3 + 0.7 * boost / boost.max()
        return np.ones(n) * 0.5

    # ── Info ──

    def stats(self) -> dict:
        """Training and observation statistics."""
        self._load()
        total_obs = sum(sum(v.values()) for v in self._observations.values())
        return {
            "unique_queries": len(self._observations),
            "total_observations": total_obs,
            "trained": self._trained,
            "n_skills": len(self._skill_names),
            "file": str(self._path),
        }

    def clear(self):
        self._observations = {}
        self._model = None
        self._item_factors = None
        self._trained = False
        self._thompson_beta = {}
        self._save()
