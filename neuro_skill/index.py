"""
Skill 索引构建器

构建:
  1. 特征矩阵 F: N × total_features (broad + precise)
  2. 技能相似度图 G: N × N (cosine on F, row-normalized)
  3. 三维张量 X: N × N × 3 (domain/language/action Jaccard)
  4. CP 张量分解因子矩阵 (可选,用于张量增强路由)
"""

import time
import numpy as np
from neuro_skill.features import (
    extract_skill_features, BROAD, PRECISE,
)

# 分类: 将 broad/precise 特征归入 domain/language/action 三个切片
_DOMAIN_KEYS = {
    "security", "frontend", "backend", "database", "devops",
    "network", "mobile", "desktop", "data", "ml", "document",
    "feishu", "firecrawl", "design",
}
_LANG_KEYS = {
    "python", "javascript_ts", "react_specific", "go", "rust",
    "java", "kotlin", "swift", "dart_flutter", "php", "csharp",
    "cpp", "harmonyos", "shell",
}
_ACTION_KEYS = {
    "code_quality", "testing", "build_fix", "security_scan",
    "performance", "architect", "planning", "tdd_testing",
    "refactor_clean", "e2e", "documentation",
}


def _cat_features(feats: dict[str, set[str]], cat: int) -> set[str]:
    """返回指定 category 的特征子集"""
    all_f = feats["broad"] | feats["precise"]
    if cat == 0:
        return all_f & _DOMAIN_KEYS
    elif cat == 1:
        return all_f & _LANG_KEYS
    else:
        return all_f & _ACTION_KEYS


def _compute_feature_weights(F_binary: np.ndarray) -> np.ndarray:
    """IG weighting: rare features (few skills) → high weight. Common → low."""
    N = F_binary.shape[0]
    if N == 0:
        return np.ones(F_binary.shape[1])
    p = np.clip(F_binary.sum(axis=0) / N, 0.0, 1.0)
    # Rare features (p→0) → high weight; Common features (p→1) → low weight
    # factor = 0.3 + 1.5 * (1 - p) yields [0.3, 1.8]
    return 0.3 + 1.5 * (1.0 - p)


def _feature_diagnostics(F_binary, ig, all_broad, all_precise, skill_feats):
    """Per-feature coverage and per-skill feature count."""
    N = F_binary.shape[0]
    coverage = F_binary.sum(axis=0) / max(N, 1)
    all_features = all_broad + all_precise
    high_freq = [(all_features[j], round(float(coverage[j]), 2))
                 for j in range(F_binary.shape[1]) if coverage[j] > 0.80]
    per_skill = F_binary.sum(axis=1).astype(int).tolist()
    low_skill = [(i, int(per_skill[i])) for i in range(N) if per_skill[i] < 2]
    # Also compute zero-feature skills (same as low_skill with count=0)
    zero_skill = [(i, 0) for i in range(N) if per_skill[i] == 0]
    return {
        "high_freq_features": high_freq,
        "low_coverage_skills": low_skill,
        "zero_features_count": len(zero_skill),
    }


def _build_feature_matrix(skills: list[dict]) -> tuple[np.ndarray, dict]:
    """Build F: N x M with IG weights. Broad=1.0, precise=1.5, both * IG."""
    skill_feats = [extract_skill_features(s) for s in skills]
    N = len(skills)

    all_broad = sorted(set().union(*(f["broad"] for f in skill_feats)))
    all_precise = sorted(set().union(*(f["precise"] for f in skill_feats)))

    broad_idx = {n: i for i, n in enumerate(all_broad)}
    precise_idx = {n: i + len(all_broad) for i, n in enumerate(all_precise)}

    M = len(all_broad) + len(all_precise)
    F_binary = np.zeros((N, M), dtype=np.float64)
    for i, f in enumerate(skill_feats):
        for b in f["broad"]:
            F_binary[i, broad_idx[b]] = 1.0
        for p in f["precise"]:
            F_binary[i, precise_idx[p]] = 1.0

    ig = _compute_feature_weights(F_binary)
    F = F_binary.copy()
    for p_idx in precise_idx.values():
        F[:, p_idx] *= 1.5
    F = F * ig[np.newaxis, :]

    meta = {
        "broad": broad_idx,
        "precise": {n: i for n, i in precise_idx.items()},
        "all_broad": all_broad,
        "all_precise": all_precise,
        "M": M,
        "ig_weights": ig,
        "diagnostics": _feature_diagnostics(F_binary, ig, all_broad, all_precise, skill_feats),
    }
    return F, meta


def _build_graph(F: np.ndarray, knn: int | None = None) -> np.ndarray:
    """
    Build skill similarity graph with symmetric degree normalization.

    Two modes:
      - knn=None (default): full cosine adjacency, all edges kept.
        Works well at 150-300 skills (density ~15%).
      - knn=k: k-nearest-neighbor graph. Only top-k edges per node kept,
        then symmetrized. Guarantees consistent edge count regardless of
        skill count. Recommended for N < 100 or N > 300.

    Symmetric normalization: G = D^(-1/2) @ A @ D^(-1/2).
    (Spreading Activation for RAG, 2025)
    """
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    F_norm = F / norms
    A = F_norm @ F_norm.T
    np.fill_diagonal(A, 0.0)  # remove self-loops

    # k-NN sparsification: keep only top-k similar neighbors per node
    if knn is not None and knn < A.shape[0] - 1:
        N = A.shape[0]
        k = min(knn, N - 1)
        # For each row, keep only top-k neighbors (argpartition avoids full sort)
        for i in range(N):
            row = A[i]
            top_k_idx = np.argpartition(-row, k - 1)[:k]
            mask = np.zeros(N, dtype=bool)
            mask[top_k_idx] = True
            A[i, ~mask] = 0.0
        # Symmetrize: union of directed k-NN edges
        A = np.maximum(A, A.T)

    # Symmetric degree normalization: D^(-1/2) @ A @ D^(-1/2)
    deg = A.sum(axis=1)
    deg[deg < 1e-10] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_half = np.diag(d_inv_sqrt)
    G = D_inv_half @ A @ D_inv_half

    return G


def _build_tensor(skills: list[dict], skill_feats: list) -> np.ndarray:
    """构建三阶张量 X[N, N, 3]"""
    N = len(skills)
    D = 3
    X = np.zeros((N, N, D), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            for cat in range(D):
                si = _cat_features(skill_feats[i], cat)
                sj = _cat_features(skill_feats[j], cat)
                if si or sj:
                    X[i, j, cat] = len(si & sj) / len(si | sj)
                elif i == j:
                    X[i, j, cat] = 1.0
    return X


def cp_decomposition(X: np.ndarray, rank: int = 8,
                     max_iter: int = 100, tol: float = 1e-5) -> tuple:
    """
    CP 张量分解 (ALS).

    返回: (weights: (R,), factors: [A(N×R), B(N×R), C(D×R)])
    """
    N, M, D = X.shape
    R = max(3, min(rank, N, M, D * 4))

    A = np.random.randn(N, R).astype(np.float64) * 0.1
    B = np.random.randn(M, R).astype(np.float64) * 0.1
    C = np.random.randn(D, R).astype(np.float64) * 0.1

    reg = 1e-6
    prev_err = float("inf")
    for it in range(max_iter):
        # Update A
        bc = np.zeros((M * D, R))
        for r in range(R):
            bc[:, r] = np.outer(B[:, r], C[:, r]).ravel()
        A = X.reshape(N, M * D) @ bc @ np.linalg.pinv(bc.T @ bc + reg * np.eye(R))

        # Update B
        ac = np.zeros((N * D, R))
        for r in range(R):
            ac[:, r] = np.outer(A[:, r], C[:, r]).ravel()
        B = X.transpose(1, 0, 2).reshape(M, N * D) @ ac @ np.linalg.pinv(ac.T @ ac + reg * np.eye(R))

        # Update C
        ab = np.zeros((N * M, R))
        for r in range(R):
            ab[:, r] = np.outer(A[:, r], B[:, r]).ravel()
        C = X.transpose(2, 0, 1).reshape(D, N * M) @ ab @ np.linalg.pinv(ab.T @ ab + reg * np.eye(R))

        if it % 20 == 0:
            recon = sum(np.einsum("i,j,k->ijk", A[:, r], B[:, r], C[:, r]) for r in range(R))
            err = np.linalg.norm(X - recon) / (np.linalg.norm(X) + 1e-10)
            if abs(prev_err - err) < tol:
                break
            prev_err = err

    # Normalize
    weights = np.ones(R)
    for r in range(R):
        na = np.linalg.norm(A[:, r])
        nb = np.linalg.norm(B[:, r])
        nc = np.linalg.norm(C[:, r])
        w = na * nb * nc
        if w > 1e-10:
            weights[r] = w
            A[:, r] /= na
            B[:, r] /= nb
            C[:, r] /= nc

    return weights, [A, B, C]


class SkillIndex:
    """Skill 索引: 特征矩阵 + 图 + 张量 + 命名查找"""

    def __init__(self):
        self.skills: list[dict] = []
        self.F: np.ndarray | None = None        # 特征矩阵
        self.G: np.ndarray | None = None         # 相似度图
        self.X: np.ndarray | None = None         # 三阶张量
        self.meta: dict = {}                     # 特征名→列号
        self.cp_weights: np.ndarray | None = None
        self.cp_factors: list | None = None
        self._name_to_idx: dict[str, int] = {}

    def build(self, skills: list[dict], rank: int = 8) -> dict:
        """
        构建索引.

        返回 stats dict.
        """
        t0 = time.time()
        self.skills = skills
        self._name_to_idx = {s["name"]: i for i, s in enumerate(skills)}

        # 特征矩阵
        self.F, self.meta = _build_feature_matrix(skills)
        skill_feats = [extract_skill_features(s) for s in skills]

        # 相似度图 — adaptive k-NN for consistent edge density
        N_graph = self.F.shape[0]
        k_adaptive = min(8, max(3, int(np.sqrt(N_graph) / 3)))
        self.G = _build_graph(self.F, knn=k_adaptive)

        # 三阶张量 + CP 分解
        self.X = _build_tensor(skills, skill_feats)
        self.cp_weights, self.cp_factors = cp_decomposition(self.X, rank=rank)

        elapsed = time.time() - t0
        return {
            "n_skills": len(skills),
            "n_features": self.meta["M"],
            "n_broad": len(self.meta["broad"]),
            "n_precise": len(self.meta["all_precise"]),
            "graph_density": float(
                np.count_nonzero(self.G > 0.01) / max(self.G.size, 1)
            ),
            "rank": len(self.cp_weights),
            "build_time_s": round(elapsed, 3),
            "diagnostics": {
                "high_freq_features": [
                    (f, c) for f, c in self.meta.get("diagnostics", {}).get("high_freq_features", [])
                ],
                "low_coverage_skills": [
                    (skills[i]["name"], c) for i, c in
                    self.meta.get("diagnostics", {}).get("low_coverage_skills", [])
                ],
                "zero_features_count": self.meta.get("diagnostics", {}).get("zero_features_count", 0),
            },
        }

    def get_idx(self, name: str) -> int | None:
        return self._name_to_idx.get(name)

    def save(self, path: str):
        """保存索引到 .npz"""
        np.savez(
            path,
            F=self.F, G=self.G, X=self.X,
            cp_weights=self.cp_weights,
            cp_A=self.cp_factors[0],
            cp_B=self.cp_factors[1],
            cp_C=self.cp_factors[2],
        )
        import json
        meta_path = path.replace(".npz", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "skills": [s["name"] for s in self.skills],
                "broad": self.meta.get("broad", {}),
                "precise": self.meta.get("precise", {}),
                "all_broad": self.meta.get("all_broad", []),
                "all_precise": self.meta.get("all_precise", []),
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, skills: list[dict]) -> "SkillIndex":
        """从 .npz 加载索引"""
        data = np.load(path, allow_pickle=True)
        idx = cls()
        idx.skills = skills
        idx._name_to_idx = {s["name"]: i for i, s in enumerate(skills)}
        idx.F = data["F"]
        idx.G = data["G"]
        idx.X = data["X"]
        idx.cp_weights = data["cp_weights"]
        idx.cp_factors = [data["cp_A"], data["cp_B"], data["cp_C"]]

        import json
        meta_path = path.replace(".npz", "_meta.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                idx.meta = json.load(f)
        except FileNotFoundError:
            idx.meta = {}
        return idx
