"""
Hermes pre_llm_call plugin — self-contained skill router.

Zero external dependencies. BM25 + regex rules, ~200 lines.
LLM sees only top-3 skills, not 332. 6-10ms per query.

Architecture (community-verified, 2026-06-15):
  on_session_start → scan SKILL.md files → build in-memory index
  pre_llm_call     → BM25 keyword match + rule check → inject top-3

This file lives in Hermes repo. No dependency on neuro-skill package.
The neuro-skill author can freely modify their code — this plugin is
entirely self-contained.
"""

import json
import math
import os
import re
import sys
import threading
from pathlib import Path

# ── Configuration ──

# Skill directories to scan (auto-detected)
_SKILL_DIRS_CACHE = None

# Routes cache: skill_name → (full_path, name, search_text)
_RouteIndex = None
_RouteLock = threading.Lock()

# Rules cache: pattern → skill_name
_RulesCache = None


def _get_skill_dirs() -> list[str]:
    global _SKILL_DIRS_CACHE
    if _SKILL_DIRS_CACHE is not None:
        return _SKILL_DIRS_CACHE

    home = Path.home()
    local = os.environ.get("LOCALAPPDATA", "") if sys.platform == "win32" else ""
    hermes_home = os.environ.get("HERMES_HOME", "")

    dirs = [
        str(home / ".claude" / ".skills-store" / "skills"),
        str(home / ".claude" / ".skills-store" / "agents"),
        str(home / ".claude" / "skills"),
        str(home / ".claude" / "agents"),
        str(home / ".claude" / ".agents" / "skills"),
        str(home / ".hermes" / "skills"),
        str(home / ".hermes" / "agents"),
    ]
    if local:
        dirs.append(str(Path(local) / "hermes" / "skills"))
        dirs.append(str(Path(local) / "hermes-agent" / "skills"))
    if hermes_home:
        dirs.append(str(Path(hermes_home) / "skills"))
        dirs.append(str(Path(hermes_home) / "agents"))

    _SKILL_DIRS_CACHE = sorted(set(d for d in dirs if Path(d).is_dir()))
    return _SKILL_DIRS_CACHE


# ── Skill File Discovery ──

def _find_skill_files(dirs: list[str]) -> list[tuple[str, str, str]]:
    """Scan directories for SKILL.md and .md agent files.
    Returns list of (name, description, search_text).
    """
    seen = set()
    skills = []

    for d in dirs:
        dp = Path(d)
        if not dp.exists():
            continue
        for item in sorted(dp.iterdir()):
            name = None
            description = ""
            filepath = None

            if item.is_dir():
                smd = item / "SKILL.md"
                if smd.exists():
                    filepath = smd
            elif item.is_file() and item.suffix == ".md":
                filepath = item

            if filepath is None:
                continue

            try:
                text = filepath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            meta, body = _parse_frontmatter(text)
            name = meta.get("name", filepath.stem)
            description = meta.get("description", "")

            if name in seen or len(text.strip()) < 20:
                continue
            seen.add(name)

            # search_text = name + description + first 500 chars of body
            search_text = f"{name} {description} {body[:500]}".lower()
            skills.append((name, description, search_text))

    return skills


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from skill text."""
    text = text.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                import yaml
                meta = yaml.safe_load(parts[1]) or {}
            except Exception:
                meta = {}
            return meta, parts[2].strip()
    return {}, text


# ── BM25 Keyword Router ──

def _tokenize_set(text: str) -> set[str]:
    """Split text into unique tokens: ASCII words + Chinese bigrams."""
    tokens = set()
    tokens.update(re.findall(r"[a-z0-9]{2,}", text.lower()))
    tokens.update(re.findall(r"[一-鿿]{2,6}", text.lower()))
    return tokens


def _tokenize_list(text: str) -> list[str]:
    """Split text into token list (preserves duplicates for TF counting)."""
    text_lower = text.lower()
    tokens = re.findall(r"[a-z0-9]{2,}", text_lower)
    tokens.extend(re.findall(r"[一-鿿]{2,6}", text_lower))
    return tokens


class _KeywordIndex:
    """Minimal BM25 keyword index. No numpy needed."""

    def __init__(self, skills: list[tuple[str, str, str]]):
        self.skills = skills
        self._doc_tokens = [_tokenize_list(st) for _, _, st in skills]
        self._doc_token_sets = [set(t) for t in self._doc_tokens]
        self._doc_lens = [len(t) for t in self._doc_tokens]
        self._avgdl = sum(self._doc_lens) / max(len(skills), 1)
        self._inverted = {}
        for i, token_set in enumerate(self._doc_token_sets):
            for t in token_set:
                self._inverted.setdefault(t, []).append(i)

        # IDF precompute
        N = len(skills)
        self._idf = {}
        for term, docs in self._inverted.items():
            self._idf[term] = math.log(1 + (N - len(docs) + 0.5) / (len(docs) + 0.5))

    def query(self, text: str, top_k: int = 3,
              k1: float = 1.2, b: float = 0.75) -> list[tuple[str, float]]:
        """Rank skills by BM25 relevance to query text."""
        q_tokens = _tokenize_set(text)
        if not q_tokens:
            return []

        N = len(self.skills)
        scores = [0.0] * N
        avgdl = max(self._avgdl, 1)

        for qt in q_tokens:
            idf = self._idf.get(qt, 0)
            if idf == 0:
                continue
            for doc_idx in self._inverted.get(qt, []):
                tf = self._doc_tokens[doc_idx].count(qt)
                dl = self._doc_lens[doc_idx]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avgdl)
                scores[doc_idx] += idf * numerator / max(denominator, 0.01)

        # Rank
        ranked = sorted(
            [(self.skills[i][0], scores[i]) for i in range(N) if scores[i] > 0],
            key=lambda x: -x[1],
        )[:top_k]
        return ranked


# ── Rule Engine ──

def _load_rules() -> list[dict]:
    """Load priority rules from ~/.neuro-skill/rules.json."""
    global _RulesCache
    if _RulesCache is not None:
        return _RulesCache
    rules_path = Path.home() / ".neuro-skill" / "rules.json"
    if rules_path.exists():
        try:
            _RulesCache = json.loads(rules_path.read_text(encoding="utf-8"))
            return _RulesCache
        except Exception:
            pass
    _RulesCache = []
    return _RulesCache


def _check_rules(query: str) -> str | None:
    """Check if query matches a priority rule. Returns skill name or None."""
    rules = _load_rules()
    for rule in rules:
        pattern = rule.get("pattern", "")
        skill = rule.get("skill", "")
        if pattern and skill:
            try:
                if re.search(pattern, query, re.IGNORECASE):
                    return skill
            except re.error:
                pass
    return None


# ── Hook Handlers ──

def on_session_start(**kwargs):
    """Scan skills and build BM25 index. ~500ms for 332 skills, one-time."""
    global _RouteIndex
    with _RouteLock:
        if _RouteIndex is not None:
            return
        try:
            dirs = _get_skill_dirs()
            skills = _find_skill_files(dirs)
            _RouteIndex = _KeywordIndex(skills)
        except Exception:
            import logging
            logging.getLogger("neuro_skill").warning(
                "Failed to build skill index", exc_info=True
            )


def pre_llm_call(**kwargs) -> dict:
    """Route user query, return top-3 skills as context block.

    Called before every LLM invocation by Hermes.
    Injects {"context": str} into the LLM prompt.
    """
    global _RouteIndex

    user_message = kwargs.get("user_message", "")
    if not user_message or not user_message.strip():
        return {}

    # Build index on first call (backup if on_session_start didn't fire)
    if _RouteIndex is None:
        try:
            on_session_start()
        except Exception:
            return {}

    if _RouteIndex is None:
        return {}

    lines = ["[Top 3 skills for this query]"]

    # Rule check (priority)
    rule_match = _check_rules(user_message)
    if rule_match:
        lines.insert(0, f"  (Rule: {rule_match})")
        lines.append(f"  1. {rule_match} (1.000)")
        lines.append("")
        lines.append("Rule-matched — use this skill unless the query clearly needs something else.")
        return {"context": "\n".join(lines)}

    # BM25 routing
    try:
        results = _RouteIndex.query(user_message, top_k=3)
    except Exception:
        return {}

    if not results:
        lines.append("  (no strong match — use built-in tools)")
        return {"context": "\n".join(lines)}

    for i, (name, score) in enumerate(results):
        lines.append(f"  {i+1}. {name} ({score:.3f})")

    lines.append("")
    lines.append("If none match, fall back to built-in tools.")

    return {"context": "\n".join(lines)}
