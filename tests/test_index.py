"""Test index building: feature matrix, graph, tensor, CP decomposition."""
import numpy as np
import pytest
from neuro_skill.index import SkillIndex, cp_decomposition


def _mock_skills(n=20):
    return [
        {
            "name": f"skill-{i}",
            "search_text": (
                f"skill-{i} "
                f"{'security' if i % 3 == 0 else ''} "
                f"{'python' if i % 4 == 0 else ''} "
                f"{'react' if i % 5 == 0 else ''} "
                f"{'build' if i % 6 == 0 else ''} "
                f"code review deploy test documentation"
            ),
        }
        for i in range(n)
    ]


class TestSkillIndex:
    def test_build(self):
        skills = _mock_skills(20)
        idx = SkillIndex()
        stats = idx.build(skills, rank=4)
        assert stats["n_skills"] == 20
        assert stats["n_features"] > 0
        assert stats["graph_density"] >= 0
        assert stats["rank"] == 4
        assert stats["build_time_s"] > 0

    def test_skill_count(self):
        skills = _mock_skills(15)
        idx = SkillIndex()
        idx.build(skills)
        assert len(idx.skills) == 15

    def test_name_lookup(self):
        skills = _mock_skills(10)
        idx = SkillIndex()
        idx.build(skills)
        assert idx.get_idx("skill-0") == 0
        assert idx.get_idx("skill-9") == 9
        assert idx.get_idx("nonexistent") is None

    def test_small_skillset(self):
        """Works with as few as 5 skills."""
        skills = _mock_skills(5)
        idx = SkillIndex()
        stats = idx.build(skills, rank=3)
        assert stats["n_skills"] == 5

    def test_graph_shape(self):
        skills = _mock_skills(20)
        idx = SkillIndex()
        idx.build(skills)
        assert idx.G.shape == (20, 20)
        # Row-stochastic (each row sums to ~1)
        row_sums = idx.G.sum(axis=1)
        assert np.allclose(row_sums[row_sums > 0], 1.0, atol=0.01)

    def test_tensor_shape(self):
        skills = _mock_skills(20)
        idx = SkillIndex()
        idx.build(skills)
        assert idx.X.shape == (20, 20, 3)  # N x N x 3 categories

    def test_save_load(self, tmp_path):
        skills = _mock_skills(15)
        idx = SkillIndex()
        idx.build(skills)

        path = str(tmp_path / "test_index.npz")
        idx.save(path)

        loaded = SkillIndex.load(path, skills)
        assert loaded.F.shape == idx.F.shape
        assert loaded.G.shape == idx.G.shape
        assert np.allclose(loaded.F, idx.F)
        assert np.allclose(loaded.G, idx.G, atol=0.01)


class TestCPDecomposition:
    def test_basic(self):
        X = np.random.rand(20, 20, 3) * 0.5
        np.fill_diagonal(X[:, :, 0], 1.0)  # make it structured
        w, factors = cp_decomposition(X, rank=4, max_iter=60)
        A, B, C = factors
        assert len(w) == 4
        assert A.shape == (20, 4)
        assert B.shape == (20, 4)
        assert C.shape == (3, 4)

    def test_small_tensor(self):
        X = np.ones((5, 5, 3)) * 0.5
        w, factors = cp_decomposition(X, rank=3, max_iter=40)
        assert len(w) == 3  # rank clamped to min(3, N, M, D*4)

    def test_reconstruction_improves(self):
        """ALS should reduce reconstruction error over iterations."""
        X = np.random.rand(10, 10, 3)
        for i in range(10):
            X[i, i, :] = 1.0  # add diagonal structure

        w, factors = cp_decomposition(X, rank=3, max_iter=80)
        A, B, C = factors

        recon = np.zeros_like(X)
        for r in range(len(w)):
            recon += np.einsum("i,j,k->ijk", A[:, r], B[:, r], C[:, r])

        err = np.linalg.norm(X - recon) / np.linalg.norm(X)
        # Reconstruction should be reasonable (not perfect, but < 1.0)
        assert err < 1.0
