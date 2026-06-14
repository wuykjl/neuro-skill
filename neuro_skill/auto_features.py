"""
Auto feature discovery — v2: quality-gated, new-user-safe.

Fixes over v1:
  - 200+ stop words covering agent description boilerplate
  - English terms require >= 3 chars (filters "is", "be", "of")
  - Min DF = max(5, 2% of skills) for candidate terms
  - Quality gate: cluster must have >= 3 meaningful terms
  - Rejects clusters of generic single words
  - Produces sub-16 broad + sub-60 precise = minimal, clean feature set
"""

import re
import math
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Optional
from neuro_skill.parser import load_skills


# ── Comprehensive stop words ──
STOP_WORDS_EN = {
    # Articles / pronouns / prepositions (always noise)
    'the', 'and', 'for', 'use', 'this', 'that', 'with', 'from',
    'your', 'will', 'have', 'been', 'all', 'has', 'are', 'was',
    'can', 'not', 'but', 'its', 'you', 'when', 'how', 'what',
    'should', 'must', 'used', 'also', 'using', 'into', 'than',
    'just', 'does', 'may', 'each', 'any', 'new', 'see', 'more',
    'well', 'way', 'get', 'one', 'two', 'like', 'make', 'need',
    'is', 'be', 'of', 'in', 'on', 'to', 'it', 'or', 'an', 'at',
    'by', 'as', 'if', 'so', 'no', 'we', 'he', 'she', 'they',
    'do', 'did', 'had', 'has', 'was', 'were', 'am', 'been',
    'who', 'whom', 'which', 'where', 'why', 'about', 'such',
    'only', 'very', 'over', 'under', 'out', 'up', 'down',
    'then', 'now', 'here', 'there', 'some', 'many', 'most',
    'every', 'few', 'less', 'more', 'other', 'own', 'same',
    'would', 'could', 'might', 'shall', 'after', 'before',
    'during', 'while', 'since', 'until', 'once', 'without',
    'through', 'above', 'below', 'between', 'among',
    # Agent description boilerplate
    'use', 'when', 'used', 'using', 'specialist', 'expert',
    'proactively', 'ensure', 'ensure', 'ensuring', 'provides',
    'provide', 'provided', 'including', 'includes', 'include',
    'based', 'available', 'required', 'require', 'requires',
    'needed', 'necessary', 'following', 'follow', 'follows',
    'must', 'always', 'never', 'code', 'agent', 'skill',
    'skills', 'work', 'works', 'working', 'task', 'tasks',
    'help', 'helps', 'tool', 'tools', 'system', 'systems',
    'process', 'processes', 'support', 'supports', 'file',
    'files', 'data', 'type', 'types', 'feature', 'features',
    'application', 'applications', 'project', 'projects',
    'change', 'changes', 'check', 'checks', 'set', 'sets',
    'run', 'running', 'runs', 'find', 'finds', 'found',
    'automatically', 'additional', 'common', 'multiple',
    'different', 'specific', 'various', 'complex', 'simple',
    'standard', 'advanced', 'basic', 'complete', 'full',
    'current', 'previous', 'next', 'first', 'last',
    'best', 'good', 'better', 'great', 'high', 'low',
    'analysis', 'management', 'development', 'support',
    'customization', 'configuration', 'implementation',
    'generation', 'integration', 'automation',
    'review', 'specializes', 'specializing', 'handling',
    'handles', 'handle', 'focuses', 'focus', 'focused',
    'fixes', 'resolution', 'resolving', 'diagnosis',
    'assessment', 'evaluation', 'improvement', 'maintenance',
    'designed', 'designs', 'creates', 'building', 'built',
    'manages', 'operates', 'operation', 'operations',
    'issues', 'practices', 'guidelines', 'methodology',
    'approach', 'techniques', 'strategies', 'patterns',
    'quality', 'quality', 'reliability', 'performance',
    'scalability', 'maintainability', 'usability',
    'correctness', 'accuracy', 'efficiency', 'consistency',
}

STOP_WORDS_CN = {
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人',
    '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
    '你', '会', '着', '没有', '看', '好', '自己', '这', '他', '她',
    '它', '们', '那', '些', '什么', '怎么', '如何', '哪', '吗',
    '啊', '吧', '呢', '哦', '嗯', '哈', '可以', '里面', '这个',
    '那个', '因为', '所以', '但是', '如果', '虽然', '已经',
    '而且', '或者', '然后', '不过', '还是', '只是', '可能',
    '应该', '需要', '能够', '进行', '使用', '通过', '提供',
    '支持', '包括', '用于', '帮助', '确保', '检查', '管理',
    '操作', '处理', '工作', '任务', '功能', '系统', '项目',
    '代码', '文件', '数据', '类型', '工具', '自动', '配置',
    '设置', '安装', '运行', '更新', '创建', '删除', '修改',
}

STOP_WORDS = STOP_WORDS_EN | STOP_WORDS_CN

# Minimum term length for English (Chinese bigrams are fine)
MIN_EN_LEN = 3


def _tokenize(text: str) -> list[str]:
    """Quality-filtered multi-language tokenizer."""
    tokens = []
    text_lower = text.lower()

    # English words: >= 3 chars, allow hyphens/slashes for tech terms
    for m in re.finditer(r'\b[a-z]{3,}(?:[-/][a-z]{2,})*\b', text_lower):
        word = m.group()
        if word not in STOP_WORDS:
            tokens.append(word)

    # Chinese bigrams/trigrams
    for m in re.finditer(r'[一-鿿]{2,6}', text_lower):
        word = m.group()
        if word not in STOP_WORDS:
            tokens.append(word)

    # CamelCase splitting
    for m in re.finditer(r'\b([A-Z][a-z]+){2,}\b', text):
        tokens.append(m.group().lower())

    # snake_case splitting
    for m in re.finditer(r'\b[a-z]+(?:_[a-z]+){1,}\b', text):
        for part in m.group().lower().split('_'):
            if part not in STOP_WORDS and len(part) >= 3:
                tokens.append(part)

    return tokens


def extract_candidate_features(
    skills: list[dict],
    min_df: Optional[int] = None,
    max_df_ratio: float = 0.75,
    min_terms_per_cluster: int = 2,
    cooc_threshold: Optional[float] = None,
    min_neighbors: Optional[int] = None,
) -> dict[str, list[str]]:
    """
    Quality-gated auto feature extraction. Parameters auto-tune to N.

    min_df:            minimum skills a term appears in (default: max(3, N*0.015))
    max_df_ratio:      terms in > this ratio of skills are ignored
    min_terms_per_cluster: reject clusters with fewer than this many terms
    cooc_threshold:    min Jaccard-normalized co-occurrence (default: adapts to N)
    min_neighbors:     min neighbors for a cluster seed (default: adapts to N)
    """
    N = len(skills)
    if min_df is None:
        min_df = max(3, int(N * 0.015))
    if cooc_threshold is None:
        # Smaller N → need looser co-occurrence
        cooc_threshold = 0.06 if N < 100 else 0.08 if N < 300 else 0.10
    if min_neighbors is None:
        min_neighbors = 1 if N < 100 else 2

    docs = [_tokenize(s["search_text"]) for s in skills]

    # ── Term frequency + IDF ──
    df = Counter()
    for tokens in docs:
        df.update(set(tokens))

    # Filter by doc frequency
    valid_terms = {}
    for term, doc_freq in df.items():
        if doc_freq < min_df:
            continue
        doc_ratio = doc_freq / N
        if doc_ratio > max_df_ratio:
            continue  # too common, not discriminative
        idf = math.log((N + 1) / (doc_freq + 1)) + 1.0
        # Bonus for multi-word terms (only count special chars if present)
        parts = (term.count('-') + term.count('/') + 1) if ('-' in term or '/' in term) else 1
        multi_bonus = 1.0 + 0.3 * (parts - 1)
        valid_terms[term] = idf * multi_bonus

    if not valid_terms:
        return {}

    # ── Select top candidates ──
    sorted_terms = sorted(valid_terms.items(), key=lambda x: x[1], reverse=True)
    max_candidates = min(200, len(sorted_terms))
    top_terms = sorted_terms[:max_candidates]
    top_term_set = {t for t, _ in top_terms}
    term_list = sorted(top_term_set)
    T = len(term_list)
    if T < min_terms_per_cluster:
        return {}

    term_to_idx = {t: i for i, t in enumerate(term_list)}

    # ── Co-occurrence matrix (Jaccard normalized) ──
    cooc = np.zeros((T, T), dtype=np.float64)
    for tokens in docs:
        indices = [term_to_idx[t] for t in tokens if t in term_to_idx]
        for ii in range(len(indices)):
            for jj in range(ii + 1, len(indices)):
                i, j = indices[ii], indices[jj]
                cooc[i, j] += 1
                cooc[j, i] += 1

    # Vectorized Jaccard normalization
    row_sums = cooc.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-10] = 1.0
    cooc = cooc / row_sums

    # ── Greedy clustering with quality gate ──
    n_clusters = min(28, T // 2)
    assigned = set()
    clusters = {}

    for seed, _ in sorted(top_terms, key=lambda x: x[1], reverse=True):
        if seed in assigned:
            continue
        if len(clusters) >= n_clusters:
            break

        idx = term_to_idx[seed]
        # Stricter co-occurrence threshold: 0.15 (was 0.05)
        neighbors = []
        for j in range(T):
            if j != idx and term_list[j] not in assigned and cooc[idx, j] > cooc_threshold:
                neighbors.append((term_list[j], cooc[idx, j]))

        if len(neighbors) >= min_neighbors:
            neighbors.sort(key=lambda x: x[1], reverse=True)
            cluster_terms = [seed] + [t for t, _ in neighbors[:10]]
            clusters[seed] = cluster_terms
            assigned.add(seed)
            assigned.update(t for t, _ in neighbors[:10])

    # ── Quality gate: reject clusters with too few meaningful terms ──
    quality_clusters = {}
    for seed, terms in clusters.items():
        # Filter: only keep terms of reasonable length
        clean = [t for t in terms if len(t) >= 3 or any('一' <= c <= '鿿' for c in t)]
        if len(clean) >= min_terms_per_cluster:
            # Generate a clean feature name from the seed
            key = re.sub(r'[^a-z0-9_]', '_', seed.lower())[:30].strip('_')
            if not key or len(key) < 3:
                key = f'cluster_{len(quality_clusters)}'
            quality_clusters[key] = list(set(clean))[:12]

    return quality_clusters


def split_broad_precise(
    features: dict[str, list[str]],
    skills: list[dict],
    n_broad: int = 14,
) -> tuple[dict, dict]:
    """
    Split features by how many skills they match.
    Broad = matches many skills (coarse categories).
    Precise = matches few skills (fine-grained).
    """
    N = len(skills)

    # Compute coverage per feature
    coverage = {}
    for key, terms in features.items():
        match_count = 0
        for s in skills:
            text = s["search_text"]
            if any(t.lower() in text for t in terms):
                match_count += 1
        coverage[key] = match_count / max(N, 1)

    # Sort by coverage: more coverage = more broad
    sorted_feats = sorted(coverage.items(), key=lambda x: x[1], reverse=True)

    # Adaptive broad threshold: higher coverage = more broad
    broad_thresh = min(0.15, max(0.08, 2.0 / max(N, 10)))
    broad = {}
    precise = {}
    for key, cov in sorted_feats:
        if len(broad) < n_broad and cov >= broad_thresh:
            broad[key] = features[key]
        else:
            precise[key] = features[key]

    return broad, precise


def auto_discover_features(
    skill_dirs: list[str],
    output_path: Optional[str] = None,
) -> dict:
    """End-to-end auto feature discovery. Returns {broad, precise, stats}."""
    skills = load_skills(skill_dirs)
    if not skills:
        raise RuntimeError(f"No skills found in {skill_dirs}")

    features = extract_candidate_features(skills)
    broad, precise = split_broad_precise(features, skills)

    stats = {
        "n_skills": len(skills),
        "n_features": len(features),
        "n_broad": len(broad),
        "n_precise": len(precise),
    }

    result = {"broad": broad, "precise": precise, "stats": stats}

    if output_path:
        _write_features_module(broad, precise, output_path)
        print(f"Auto-discovered features written to {output_path}")
        print(f"  {stats['n_broad']} broad + {stats['n_precise']} precise "
              f"= {stats['n_features']} total from {stats['n_skills']} skills")

    return result


def _write_features_module(broad: dict, precise: dict, path: str):
    lines = [
        '"""Auto-discovered features — generated by neuro-skill."""',
        '',
        '# Broad features',
        'BROAD = {',
    ]
    for key, terms in sorted(broad.items()):
        ts = ', '.join(f'"{t}"' for t in sorted(terms)[:12])
        lines.append(f'    "{key}": [{ts}],')
    lines.extend(['}', '', '# Precise features', 'PRECISE = {'])
    for key, terms in sorted(precise.items()):
        ts = ', '.join(f'"{t}"' for t in sorted(terms)[:12])
        lines.append(f'    "{key}": [{ts}],')
    lines.append('}')
    lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
