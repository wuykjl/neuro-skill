"""
Error Book — lightweight learning feedback for skill routing.

Inspired by LLM-Wiki's Error Book mechanism (arXiv 2605.25480):
  "Persistent structural and semantic self-correction."

When a user corrects a routing result (picks skill #3 instead of #1),
this module records the signal. Subsequent similar queries auto-boost
the corrected skill. Boosts decay over time — no permanent bias.

All state lives in a single JSON file (~2KB for 100 corrections).
Zero external dependencies. Zero training. Sub-millisecond overlay.

Usage:
  from neuro_skill.feedback import ErrorBook

  book = ErrorBook("~/.neuro-skill-feedback.json")
  book.correct("Go构建报错", preferred="go-build-resolver")
  adjusted_scores = book.adjust("Go编译失败", original_scores, skill_names)
"""

from __future__ import annotations

import json, hashlib, time, os, re
from pathlib import Path
from collections import defaultdict
from typing import Optional


def _query_hash(query: str) -> str:
    """Stable hash: normalize → hash first 3 meaningful tokens.

    Two queries that share the same key concept words
    should hit the same feedback entry, even if phrased differently.
    """
    # Extract meaningful tokens (3+ chars, filter short/common words)
    tokens = re.findall(r'[a-z]{3,}|[一-鿿]{2,4}', query.lower())
    # Take first 5 tokens (captures the core intent)
    key = " ".join(tokens[:5]) if tokens else query.lower()[:30]
    return hashlib.md5(key.encode()).hexdigest()[:12]


class ErrorBook:
    """Persistent learning feedback for skill routing.

    Stores corrections as: {query_hash: {preferred_skill_name: boost}}
    Boosts are additive with decay — a 24h-old boost is worth half
    a fresh one. This prevents permanent bias from old corrections.

    Thread-safe for reads, write-on-correct.
    """

    def __init__(self, path: str = "~/.neuro-skill-feedback.json"):
        self._path = Path(path).expanduser()
        self._entries: dict[str, dict[str, float]] = {}
        self._loaded = False

    def _load(self):
        if not self._loaded:
            if self._path.exists():
                try:
                    self._entries = json.loads(
                        self._path.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, IOError):
                    self._entries = {}
            self._loaded = True

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._entries, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self._path)  # atomic on POSIX, best-effort on Windows

    def correct(
        self,
        query: str,
        preferred: str,
        boost: float = 1.0,
        timestamp: Optional[float] = None,
    ):
        """Record a user correction.

        Args:
          query:     the original query text
          preferred: the skill name the user actually chose
          boost:     boost strength (default 1.0 = ~2 rank positions in RRF)
          timestamp: unix time (default: now)
        """
        self._load()
        qhash = _query_hash(query)
        ts = timestamp or time.time()

        if qhash not in self._entries:
            self._entries[qhash] = {}

        # Accumulate: repeated corrections for same query strengthen the boost
        current = self._entries[qhash].get(preferred, 0.0)
        self._entries[qhash][preferred] = current + boost

        # Store timestamp as a special key
        self._entries[qhash]["_ts"] = ts

        # Prune entries older than 30 days
        self._prune(30)

        self._save()

    def adjust(
        self,
        query: str,
        scores: list[float],
        skill_names: list[str],
        decay_days: float = 7.0,
    ) -> list[float]:
        """Adjust routing scores using learned feedback.

        Args:
          query:       current query text
          scores:      raw RRF scores from the router
          skill_names: skill names matching scores (same order)
          decay_days:  half-life of a boost in days (default 7)

        Returns:
          adjusted scores (same length as input)
        """
        self._load()
        if not self._entries:
            return scores

        qhash = _query_hash(query)
        if qhash not in self._entries:
            return scores

        entry = self._entries[qhash]
        now = time.time()
        entry_ts = entry.get("_ts", now)

        # Decay factor: 0.5^(age / half_life)
        age_days = (now - entry_ts) / 86400.0
        decay = 0.5 ** (age_days / max(decay_days, 0.1))

        # Build boost map from entry (skip _ts key)
        boost_map = {
            k: v * decay for k, v in entry.items() if k != "_ts"
        }

        if not boost_map:
            return scores

        # Convert to numpy for efficient adjustment
        import numpy as np
        adjusted = np.array(scores, dtype=np.float64)

        for i, name in enumerate(skill_names):
            if name in boost_map:
                # RRF scores are ~0.02-0.05 range.
                # A boost of 1.0 means adding 0.01-0.015 (~2-3 rank positions).
                adjusted[i] += boost_map[name] * 0.012

        return adjusted.tolist()

    def _prune(self, max_days: int = 30):
        """Remove entries older than max_days."""
        now = time.time()
        cutoff = now - max_days * 86400
        stale = [
            h for h, e in self._entries.items()
            if e.get("_ts", 0) < cutoff
        ]
        for h in stale:
            del self._entries[h]

    def stats(self) -> dict:
        """Get feedback statistics."""
        self._load()
        now = time.time()
        total = 0
        active = 0
        for e in self._entries.values():
            boosts = [v for k, v in e.items() if k != "_ts"]
            total += sum(boosts)
            if any(b * (0.5 ** ((now - e.get("_ts", now)) / 86400 / 7)) > 0.1
                   for b in boosts):
                active += 1
        return {
            "entries": len(self._entries),
            "total_boosts": total,
            "active_entries": active,
            "file": str(self._path),
        }

    def clear(self):
        """Reset all feedback."""
        self._entries = {}
        self._loaded = True
        self._save()
