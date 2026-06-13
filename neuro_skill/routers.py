"""
Skill 路由方法

  1. hybrid:       混合路由 (cosine + graph + keyword) — 默认
  2. graph_spread: 图扩散激活
  3. cosine:       余弦相似度
  4. tensor:       张量增强
  5. jaccard:      特征 Jaccard
  6. keyword:      关键词匹配
  7. tfidf:        TF-IDF
"""

import re
import math
import time
import numpy as np
from neuro_skill.features import (
    extract_query_features, extract_skill_features, feature_set, tokenize,
)

# ── 工具 ──

def _normalize(v: np.ndarray) -> np.ndarray:
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn) if mx - mn > 1e-10 else np.zeros_like(v)


def _rank(skills: list[dict], scores: np.ndarray,
          top_k: int = 10) -> list[tuple[str, float, int]]:
    """将分数排序,返回 [(name, score, idx), ...]"""
    order = np.argsort(scores)[::-1][:top_k]
    return [(skills[i]["name"], float(scores[i]), int(i)) for i in order]


# ── 方法 1: 关键词匹配 ──

def keyword(skills: list[dict], query: str, **_kw) -> np.ndarray:
    q_tokens = tokenize(query)
    scores = np.zeros(len(skills))
    for i, s in enumerate(skills):
        s_tokens = tokenize(s["search_text"])
        scores[i] = len(q_tokens & s_tokens) / max(len(q_tokens), 1)
    return scores


# ── 方法 2: 特征 Jaccard ──

def jaccard(skills: list[dict], query: str, **_kw) -> np.ndarray:
    qf = extract_query_features(query)
    q_set = feature_set(qf)
    scores = np.zeros(len(skills))
    for i, s in enumerate(skills):
        sf = extract_skill_features(s)
        s_set = feature_set(sf)
        jac = (len(q_set & s_set) / max(len(q_set | s_set), 1)
               if q_set or s_set else 0.0)
        scores[i] = jac
    return scores


# ── 方法 3: TF-IDF ──

def tfidf(skills: list[dict], query: str, **_kw) -> np.ndarray:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        docs = [query] + [s["search_text"] for s in skills]
        vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        tfidf_m = vec.fit_transform(docs)
        sims = cosine_similarity(tfidf_m[0], tfidf_m[1:])[0]
        return sims
    except ImportError:
        return keyword(skills, query)


# ── 方法 4: 余弦相似度 ──

def cosine(skills: list[dict], query: str,
           F: np.ndarray | None, meta: dict | None, **_kw) -> np.ndarray:
    if F is None or meta is None:
        return np.zeros(len(skills))

    qf = extract_query_features(query)
    qv = np.zeros(F.shape[1])
    for b in qf["broad"]:
        if b in meta["broad"]:
            qv[meta["broad"][b]] = 1.0
    for p in qf["precise"]:
        if p in meta["precise"]:
            qv[meta["precise"][p]] = 1.5

    q_norm = np.linalg.norm(qv)
    if q_norm < 1e-10:
        return np.zeros(len(skills))

    sims = F @ qv / (np.linalg.norm(F, axis=1) * q_norm + 1e-10)
    return sims


# ── 方法 5: 图扩散激活 ──

def graph_spread(skills: list[dict], query: str,
                 G: np.ndarray | None, **kw) -> np.ndarray:
    if G is None:
        return jaccard(skills, query)

    N = len(skills)
    steps = kw.get("graph_steps", 3)
    decay = kw.get("graph_decay", 0.5)

    qf = extract_query_features(query)
    q_set = feature_set(qf)

    # 初始激活
    activation = np.zeros(N)
    for i, s in enumerate(skills):
        s_set = feature_set(extract_skill_features(s))
        if q_set or s_set:
            activation[i] = len(q_set & s_set) / max(len(q_set | s_set), 1)
    if activation.sum() < 1e-10:
        activation = np.ones(N) / N
    activation /= max(activation.sum(), 1.0)

    total = activation.copy()
    current = activation.copy()
    for step in range(steps):
        current = G.T @ current * (decay ** (step + 1))
        total += current

    return total


# ── 方法 6: 张量增强 ──

def tensor(skills: list[dict], query: str,
           F: np.ndarray | None, meta: dict | None,
           cp_weights: np.ndarray | None, cp_factors: list | None,
           **kw) -> np.ndarray:
    if F is None or cp_factors is None:
        return cosine(skills, query, F=F, meta=meta)

    base = cosine(skills, query, F=F, meta=meta)
    A = cp_factors[0]
    N = len(skills)

    top_k = min(30, N)
    top_idx = np.argsort(base)[-top_k:]
    enhanced = base.copy()
    for i in top_idx:
        if base[i] > 0.05:
            a_i = A[i, :]
            a_norm = np.linalg.norm(a_i) + 1e-10
            for j in range(N):
                if i != j:
                    a_j = A[j, :]
                    a_sim = abs(np.dot(a_i, a_j)) / (
                        a_norm * (np.linalg.norm(a_j) + 1e-10)
                    )
                    if a_sim > 0.3:
                        enhanced[j] += base[i] * a_sim * 0.25
    return enhanced


# ── 方法 7: 混合路由 ──

def hybrid(skills: list[dict], query: str,
           F: np.ndarray | None, G: np.ndarray | None,
           meta: dict | None, **kw) -> np.ndarray:
    """
    混合路由: 余弦 + 图扩散 + 关键词 三层融合.

    权重: cos=0.40, graph=0.40, keyword=0.20
    """
    N = len(skills)

    s_cos = cosine(skills, query, F=F, meta=meta)
    s_graph = graph_spread(skills, query, G=G, **kw)
    s_kw = keyword(skills, query)

    w_cos = kw.get("w_cos", 0.40)
    w_graph = kw.get("w_graph", 0.40)
    w_kw = kw.get("w_kw", 0.20)

    fused = (w_cos * _normalize(s_cos)
             + w_graph * _normalize(s_graph)
             + w_kw * _normalize(s_kw))

    return _normalize(fused)


# ── 路由注册表 ──

ROUTERS = {
    "hybrid": hybrid,
    "graph_spread": graph_spread,
    "cosine": cosine,
    "tensor": tensor,
    "jaccard": jaccard,
    "keyword": keyword,
    "tfidf": tfidf,
}
