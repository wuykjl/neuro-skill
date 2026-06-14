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


# ── Shared utilities ──

from functools import lru_cache as _lru_cache

@_lru_cache(maxsize=512)
def _tokenize_cached(text: str) -> frozenset:
    """Cached tokenization for frequently re-tokenized search_text."""
    return frozenset(tokenize(text))


# ── 方法 1: Okapi BM25 关键词匹配 ──

# BM25 precompute cache with LRU eviction
_bm25_cache: dict = {}  # {id(skills): stat}
_bm25_order: list = []  # simple FIFO for LRU


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
    # LRU eviction: keep last 8 entries (doubled from 4)
    _bm25_order.append(skey)
    if len(_bm25_order) > 8:
        old = _bm25_order.pop(0)
        _bm25_cache.pop(old, None)
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

    # Pre-tokenize all skill docs once using cached tokenizer
    skill_tokens = [set(_tokenize_cached(s["search_text"])) for s in skills]

    # Build reverse index for O(1) tf count
    term_to_doc_tf: dict[str, dict[int, int]] = {}
    for i, tokens in enumerate(skill_tokens):
        tf_i = {}
        for t in tokens:
            tf_i[t] = tf_i.get(t, 0) + 1
        for t, cnt in tf_i.items():
            if t not in term_to_doc_tf:
                term_to_doc_tf[t] = {}
            term_to_doc_tf[t][i] = cnt

    # Compute BM25 scores for all skills in one pass
    scores = np.zeros(N, dtype=np.float64)
    for qi, qt in enumerate(q_tokens):
        if not valid[qi]:
            continue
        # O(1) lookup via reverse index instead of O(N*T) scan
        doc_tf = term_to_doc_tf.get(qt, {})
        if not doc_tf:
            continue
        tf = np.zeros(N, dtype=np.float64)
        for di, cnt in doc_tf.items():
            tf[di] = float(cnt)
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
        # Hot path: shared query vector (cached)
        meta = _kw["meta"]
        qv = _build_query_vector(
            query,
            tuple(sorted(meta["broad"].items())),
            tuple(sorted(meta["precise"].items())),
            F.shape[1],
        )
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


# ── 共享查询向量构建 ──

@_lru_cache(maxsize=512)
def _build_query_vector(query: str, broad_keys: tuple, precise_keys: tuple,
                        F_cols: int) -> np.ndarray:
    """Build feature vector for a query. Cacheable — keys are hashable tuples."""
    qf = extract_query_features(query)
    qv = np.zeros(F_cols)
    broad = {b[0]: b[1] for b in broad_keys} if broad_keys else {}
    precise = {p[0]: p[1] for p in precise_keys} if precise_keys else {}
    for b in qf["broad"]:
        if b in broad:
            qv[broad[b]] = 1.0
    for p in qf["precise"]:
        if p in precise:
            qv[precise[p]] = 1.5
    return qv


# ── 方法 4: 余弦相似度 ──

def cosine(skills: list[dict], query: str,
           F: np.ndarray | None, meta: dict | None, **_kw) -> np.ndarray:
    if F is None or meta is None:
        return np.zeros(len(skills))

    qv = _build_query_vector(
        query,
        tuple(sorted(meta["broad"].items())),
        tuple(sorted(meta["precise"].items())),
        F.shape[1],
    )

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


# ── 方法 7: 混合路由 (RRF fusion) ──

def hybrid(skills: list[dict], query: str,
           F: np.ndarray | None, G: np.ndarray | None,
           meta: dict | None, **kw) -> np.ndarray:
    """
    Reciprocal Rank Fusion (RRF) — the standard hybrid fusion algorithm
    used by OpenSearch, Azure AI Search, Weaviate, Pinecone.

    Replaces min-max normalization + weighted sum with rank-based fusion.
    RRF is insensitive to raw score distributions (BM25: 0~15,
    cosine: -1~1, graph: ~0.001) — only rank position matters.

    RRF_score(d) = sum over methods of 1 / (k + rank_i(d))
    where k=60 (standard constant, from OpenSearch/Azure documentation).

    This also eliminates the need for adaptive weight tuning — RRF
    self-adapts to any graph density, skill count, or score distribution.
    """
    N = len(skills)
    K = 60  # RRF constant (OpenSearch/Azure standard)

    # Compute all three signal vectors
    s_bm25 = keyword(skills, query)
    s_cos = cosine(skills, query, F=F, meta=meta)
    s_graph = graph_spread(skills, query, G=G, F=F, meta=meta, **kw)

    # Convert scores to ranks (0 = highest score)
    # argsort ascending → flip to get descending rank
    rank_bm25 = np.zeros(N, dtype=np.float64)
    rank_cos = np.zeros(N, dtype=np.float64)
    rank_graph = np.zeros(N, dtype=np.float64)

    # argsort returns indices sorted by value ascending
    # we want rank 0 = highest score, so we reverse
    for rank_arr, scores in [
        (rank_bm25, s_bm25), (rank_cos, s_cos), (rank_graph, s_graph)
    ]:
        order = np.argsort(-scores)  # descending
        for rank, idx in enumerate(order):
            rank_arr[idx] = float(rank)

    # Filter out methods that returned all zeros (no signal)
    signals = [rank_bm25, rank_cos, rank_graph]

    # 5th signal: LLM semantic rerank (optional, API-required)
    if kw.get("enable_llm"):
        rank_llm = llm_rerank(query, skills, top_n=min(10, N),
                              model=kw.get("llm_model", "claude-haiku-4-5-20251001"),
                              api_key=kw.get("llm_api_key"))
        if rank_llm is not None:
            signals.append(rank_llm)

    active = 0
    rrf = np.zeros(N, dtype=np.float64)
    for rank_arr in signals:
        if rank_arr.max() > rank_arr.min() or not np.allclose(rank_arr, 0):
            rrf += 1.0 / (K + rank_arr)
            active += 1

    # If no signal at all, return uniform
    if active == 0:
        return np.ones(N) / N

    return rrf


# ── 方法 8: LLM Rerank (5th RRF signal) ──

def llm_rerank(query: str, skills: list[dict], top_n: int = 10,
               model: str = "claude-haiku-4-5-20251001",
               api_key: str | None = None) -> np.ndarray | None:
    """
    Let Haiku re-rank the top-N candidates. Returns rank array or None.

    Cost:  ~$0.0003/call (Haiku, top-10), ~200ms latency
    Fallback: returns None silently — hybrid() skips the 5th signal.

    The LLM sees skill name + description for semantic understanding
    that BM25/feature-cosine cannot provide (e.g. "make it faster" vs
    "optimize database queries").
    """
    import os as _os, json as _json

    N = len(skills)
    if N == 0:
        return None

    # Get top-N candidates. Through BM25 if available, otherwise first N.
    try:
        s_bm25 = keyword(skills, query)
        top_idx = np.argsort(-s_bm25)[:min(top_n, N)]
    except Exception:
        top_idx = np.arange(min(top_n, N))

    candidates = []
    for idx in top_idx:
        s = skills[idx]
        desc = s.get("description", "")[:120].replace("\n", " ")
        candidates.append({"name": s["name"], "description": desc})

    c_list = "\n".join(
        f"{i+1}. {c['name']}: {c['description']}"
        for i, c in enumerate(candidates)
    )

    prompt = (
        f"Task: Rank these skills by relevance to the query.\n\n"
        f"Query: {query}\n\n"
        f"Skills:\n{c_list}\n\n"
        f"Return ONLY a JSON array of skill names in order of relevance "
        f"(most relevant first). Include ALL skills in the list:\n"
        f'["skill_name_1", "skill_name_2", ...]'
    )

    try:
        import anthropic
        resolved_key = (
            api_key
            or _os.environ.get("ANTHROPIC_API_KEY")
            or _os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
        if not resolved_key:
            return None
        client = anthropic.Anthropic(
            api_key=resolved_key,
            base_url=_os.environ.get("ANTHROPIC_BASE_URL"),
        )
        actual_model = (
            model
            or _os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL")
            or _os.environ.get("ANTHROPIC_MODEL")
            or "claude-haiku-4-5-20251001"
        )

        ranked_names = None
        for attempt in range(2):  # one retry
            try:
                resp = client.messages.create(
                    model=actual_model, max_tokens=256, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                    thinking={"type": "disabled"},
                )
                text = "".join(
                    block.text for block in resp.content
                    if hasattr(block, "text") and block.text
                ).strip()
                # Clean markdown fences
                if text.startswith("```"):
                    text = "\n".join(text.split("\n")[1:-1])
                # Try to find a JSON array in the response
                import re as _re
                match = _re.search(r'\["[^"]+"(?:,\s*"[^"]+")*\]', text)
                if match:
                    ranked_names = _json.loads(match.group())
                else:
                    ranked_names = _json.loads(text)
                break
            except Exception:
                if attempt == 0:
                    continue
                ranked_names = None
        if ranked_names is None:
            return None
    except Exception:
        return None

    # Convert LLM rank order to rank array (size N)
    rank_llm = np.ones(N, dtype=np.float64) * N  # unranked = last
    for rank, name in enumerate(ranked_names):
        for idx in range(N):
            if skills[idx]["name"] == name:
                rank_llm[idx] = float(rank)
                break

    return rank_llm


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
