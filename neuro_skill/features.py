"""
Two-tier feature extraction.

BROAD / PRECISE are module-level dicts initialized from base_features at import.
They can be safely mutated to add user or auto-discovered features.
"""

import re
import hashlib

# Initialize from base — use dict() to create independent copies
from neuro_skill.base_features import BROAD as _BASE_BROAD, PRECISE as _BASE_PRECISE

BROAD = dict(_BASE_BROAD)
PRECISE = dict(_BASE_PRECISE)


# ── Shared tokenizer (used by both _match and routers.keyword) ──

def tokenize(text: str) -> set[str]:
    """Split into tokens: ASCII words + Chinese bigrams + lang keywords.

    Handles mixed-script queries like "检索cs文件安全检查".
    Built-in regex cache avoids repeated compilation.
    """
    text_lower = text.lower()
    tokens = set()

    # ASCII-only word match (avoids Unicode \w matching CJK as word chars)
    tokens.update(re.findall(r"[a-z0-9]{2,}", text_lower))

    # Single-letter language keywords caught separately:
    # c#, f#, go, rs, ts, js — trapped as standalone letter pairs
    for m in re.finditer(r"(?:^|(?<=[^a-z]))[a-z]{1,3}(?:#[a-z]+)?(?:(?=[^a-z])|$)", text_lower):
        tokens.add(m.group())

    # Chinese: overlapping bigrams from pure-CJK spans
    for m in re.finditer(r"[一-鿿]{2,}", text_lower):
        span = m.group()
        for i in range(len(span) - 1):
            tokens.add(span[i:i + 2])

    # Whole Chinese word spans (non-overlapping, for feature matching)
    tokens.update(re.findall(r"[一-鿿]{2,6}", text_lower))

    return tokens


# ── Shared query hash (used by Feedback, Personalize, ErrorBook) ──

def query_hash(query: str) -> str:
    """Stable hash for a query — same intent → same key.

    Two queries that share the same key concept words produce the
    same hash, so feedback and personalization transfer across
    differently-phrased but semantically identical queries.
    """
    tokens = re.findall(r'[a-z]{3,}|[一-鿿]{2,4}', query.lower())
    key = " ".join(tokens[:5]) if tokens else query.lower()[:30]
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ── Precompiled regex cache (compiled once, reused forever) ──

_REGEX_CACHE: dict[tuple[str, str], re.Pattern] = {}

def _compile_regex(pattern: str) -> re.Pattern | None:
    key = ("_compile_regex", pattern)
    if key not in _REGEX_CACHE:
        try:
            _REGEX_CACHE[key] = re.compile(pattern, re.I)
        except re.error:
            _REGEX_CACHE[key] = None
    return _REGEX_CACHE[key]


_CHINESE_PATTERN = re.compile(r'[一-鿿]')
_EN_MIXED_PATTERNS: dict[str, re.Pattern] = {}
for kw_base, feat_name in [
    ("go", "go"), ("c++", "cpp"), ("c#", "csharp"),
    ("cs", "csharp"), ("ts", "javascript_ts"), ("js", "javascript_ts"),
    ("rs", "rust"), ("kt", "kotlin"), ("rb", "ruby"),
]:
    try:
        _EN_MIXED_PATTERNS[kw_base] = (
            re.compile(r'(?<![a-z])' + re.escape(kw_base) + r'[一-鿿文件]', re.I),
            feat_name,
        )
    except re.error:
        pass


def _match(text: str, keyword_map: dict[str, list[str]]) -> set[str]:
    """Match text against predefined keyword categories. One hit per category.

    Supports:
      - substring matching
      - regex patterns (starting with \\b) — precompiled, cached
      - Chinese-English mixed: "Go构建" auto-detects language features
    """
    text_lower = text.lower()
    matched = set()

    # 中英混合扩展: "cs文件" → csharp, "go构建" → go, etc.
    if _CHINESE_PATTERN.search(text_lower):
        for kw_base, (pattern, feat_name) in _EN_MIXED_PATTERNS.items():
            if pattern.search(text_lower):
                matched.add(feat_name)

    for category, keywords in keyword_map.items():
        for kw in keywords:
            kw_lower = kw.lower()
            # Regex patterns (precompiled, cached)
            if kw_lower.startswith('\\b'):
                compiled = _compile_regex(kw_lower)
                if compiled is not None and compiled.search(text_lower):
                    matched.add(category)
                    break
            elif kw_lower in text_lower:
                matched.add(category)
                break

    return matched


def extract_skill_features(skill: dict) -> dict[str, set[str]]:
    """
    从 skill 的 search_text(name + description + triggers)提取特征.
    关键: 不在 full body 上匹配,避免假阳性.
    """
    return {
        "broad": _match(skill["search_text"], BROAD),
        "precise": _match(skill["search_text"], PRECISE),
    }


def extract_query_features(query: str) -> dict[str, set[str]]:
    """从用户查询中提取两级特征"""
    return {
        "broad": _match(query, BROAD),
        "precise": _match(query, PRECISE),
    }


def feature_set(feats: dict[str, set[str]]) -> set[str]:
    """合并 broad + precise 为统一特征集"""
    return feats["broad"] | feats["precise"]
