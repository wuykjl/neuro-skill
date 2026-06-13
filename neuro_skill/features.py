"""
Two-tier feature extraction.

BROAD / PRECISE are module-level dicts initialized from base_features at import.
They can be safely mutated to add user or auto-discovered features.
"""

import re

# Initialize from base — use dict() to create independent copies
from neuro_skill.base_features import BROAD as _BASE_BROAD, PRECISE as _BASE_PRECISE

BROAD = dict(_BASE_BROAD)
PRECISE = dict(_BASE_PRECISE)


def _match(text: str, keyword_map: dict[str, list[str]]) -> set[str]:
    """匹配文本中的关键词，每类别最多命中一个。

    支持:
      - 普通子串匹配
      - 正则模式 (以 \\b 开头的模式)
      - 中英混合: "Go构建" → 自动尝试在 "Go" 后拼接常见中文字符
    """
    text_lower = text.lower()
    matched = set()

    # 中英混合扩展: 如果文本包含中文字符，对短英文名做拼接匹配
    has_chinese = bool(re.search(r'[一-鿿]', text_lower))
    extra_patterns = {}
    if has_chinese:
        for kw_base in ["go", "c++", "c#"]:
            if re.search(r'(?<![a-z])' + re.escape(kw_base) + r'[一-鿿]', text_lower, re.I):
                kw_map = {"go": "go", "c++": "cpp", "c#": "csharp"}
                extra_patterns[kw_map.get(kw_base, kw_base)] = True

    for category, keywords in keyword_map.items():
        for kw in keywords:
            kw_lower = kw.lower()
            # Regex patterns (start with \\b)
            if kw_lower.startswith('\\b'):
                try:
                    if re.search(kw_lower, text_lower, re.I):
                        matched.add(category)
                        break
                except re.error:
                    pass
            elif kw_lower in text_lower:
                matched.add(category)
                break

        # Extra: check Chinese-mixed patterns
        if category not in matched and category in extra_patterns:
            matched.add(category)

    return matched


def extract_skill_features(skill: dict) -> dict[str, set[str]]:
    """
    从 skill 的 search_text（name + description + triggers）提取特征。
    关键: 不在 full body 上匹配，避免假阳性。
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
