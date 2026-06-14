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


# ── 方法 1: Okapi BM25 关键词匹配 ──

# Per-skill document stats (computed once, shared across calls)
_bm25_cache: dict = {}  # {id(skills): {"avgdl": float, "doc_lens": np.ndarray, "tf": defaultdict, "df": dict, "N": int}}


def _bm25_precompute(skills: list[dict]):
    """Precompute BM25 document stats for a skill set (lazy, cached by id)."""
    skey = id(skills)
    if skey in _bm25_cache:
        return _bm25_cache[skey]

    from collections import Counter as _Counter
    _idf_counter = _Counter()
    doc_lens = []
    docs = [tokenize(s["search_text"]) for s in skills]
    for tokens in docs:
        doc_lens.append(len(tokens))
        for t in set(tokens):
            _idf_counter[t] += 1

    N = len(skills)
    avgdl = sum(doc_lens) / max(N, 1)

    stat = {"avgdl": avgdl, "doc_lens": np.array(doc_lens, dtype=np.float64),
            "df": dict(_idf_counter), "N": N}
    _bm25_cache[skey] = stat
    # Limit cache size
    if len(_bm25_cache) > 4:
        _bm25_cache.pop(next(iter(_bm25_cache)))
    return stat


def keyword(skills: list[dict], query: str, **_kw) -> np.ndarray:
    """Okapi BM25 keyword scoring (replaces simple intersection/union).

    Parameters: k1=1.2, b=0.75 (standard Okapi defaults).
    IDF computed from the skill search_text corpus.
    Pure numpy vectorization — no per-skill Python loop.
    """
    import math as _math
    N = len(skills)
    q_tokens = tokenize(query)
    if not q_tokens:
        return np.zeros(N)

    # BM25 with standard Okapi parameters
    k1 = 1.2
    b = 0.75
    stat = _bm25_precompute(skills)
    avgdl = stat["avgdl"]
    doc_lens = stat["doc_lens"]
    N_docs = stat["N"]

    # Compute IDF for query terms
    idf = np.zeros(len(q_tokens))
    valid = np.zeros(len(q_tokens), dtype=bool)
    for qi, qt in enumerate(q_tokens):
        df = stat["df"].get(qt, 0)
        if df > 0:
            # Smooth IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            idf[qi] = _math.log((N_docs - df + 0.5) / (df + 0.5) + 1.0)
            valid[qi] = True

    if not valid.any():
        return np.zeros(N)

    # Pre-tokenize all skill docs once
    skill_tokens = [tokenize(s["search_text"]) for s in skills]

    # Compute BM25 scores for all skills in one pass
    # For each query term → compute tf for all docs → accumulate weighted sum
    scores = np.zeros(N, dtype=np.float64)
    for qi, qt in enumerate(q_tokens):
        if not valid[qi]:
            continue
        # Term frequency per document (count occurrences)
        tf = np.zeros(N, dtype=np.float64)
        for i, tokens in enumerate(skill_tokens):
            tf[i] = sum(1 for t in tokens if t == qt)
        if not tf.any():
            continue
        # BM25 scoring
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_lens / max(avgdl, 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            term_scores = np.where(denominator > 0, idf[qi] * numerator / denominator, 0.0)
        scores += term_scores

    return scores


# ── 方法 2: 特征 Jaccard ──

def jaccard(skills: list[dict], query: str, F: np.ndarray | None = None, **_kw) -> np.ndarray:
    """Vectorized Jaccard using pre-computed feature matrix F.

    Without F (cold path): falls back to per-skill Python loop (~25ms).
    With F (hot path): pure numpy, ~0.5ms.
    """
    if F is not None and _kw.get("meta"):
        # Hot path: use pre-computed feature vectors
        meta = _kw["meta"]
        qf = extract_query_features(query)
        qv = np.zeros(F.shape[1])
        for b in qf["broad"]:
            if b in meta["broad"]:
                qv[meta["broad"][b]] = 1.0
        for p in qf["precise"]:
            if p in meta["precise"]:
                qv[meta["precise"][p]] = 1.5
        # Jaccard = intersection / union for binary vectors
        # For each skill i: AND(qv, F[i]) / OR(qv, F[i])
        F_binary = (F > 0).astype(np.float64)
        q_binary = (qv > 0).astype(np.float64)
        intersection = F_binary @ q_binary  # (N,) — number of shared features
        union = F_binary.sum(axis=1) + q_binary.sum() - intersection  # (N,)
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = np.where(union > 0, intersection / union, 0.0)
        return scores.astype(np.float64)

    # Cold path: per-skill loop (fallback)
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

    # Vectorized initial activation: use jaccard with F if available
    F_mat = kw.pop("F", None)
    meta_d = kw.pop("meta", None)
    if F_mat is not None and meta_d is not None:
        activation = jaccard(skills, query, F=F_mat, meta=meta_d, **kw)
    else:
        qf = extract_query_features(query)
        q_set = feature_set(qf)
        activation = np.zeros(N)
        for i, s in enumerate(skills):
            s_set = feature_set(extract_skill_features(s))
            if q_set or s_set:
                activation[i] = len(q_set & s_set) / max(len(q_set | s_set), 1)
    if activation.sum() < 1e-10:
        activation = np.ones(N) / N
    activation = activation / max(activation.sum(), 1.0)

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
    Hybrid routing with adaptive graph weights.

    Weight adaptation (continuous, inspired by Hermes 332-skill test):
      - density ~15% (sweet spot): cos=0.40, graph=0.40, kw=0.20
      - density < 5% (too sparse) or > 60% (too dense): graph decays to ~0
      - Gaussian decay from optimum, sigma=0.10
    """
    import math as _math
    N = len(skills)

    # Compute graph density for adaptive weighting
    if G is not None and G.size > 0:
        nz = float((G > 0).sum())
        density = nz / G.size
    else:
        density = 0.0

    # Gaussian: peak at density=0.15, decay on both sides
    sigma = 0.10
    graph_factor = _math.exp(-((density - 0.15) ** 2) / (2 * sigma * sigma))
    w_graph_auto = 0.40 * graph_factor
    w_cos_auto = 0.40 + (0.40 - w_graph_auto) * 0.5
    w_kw_auto  = 0.20 + (0.40 - w_graph_auto) * 0.5

    w_cos = kw.get("w_cos", w_cos_auto)
    w_graph = kw.get("w_graph", w_graph_auto)
    w_kw = kw.get("w_kw", w_kw_auto)

    s_cos = cosine(skills, query, F=F, meta=meta)
    s_graph = graph_spread(skills, query, G=G, F=F, meta=meta, **kw)
    s_kw = keyword(skills, query)

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
