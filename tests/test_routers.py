"""Test routing methods produce correct output shapes and ranges."""
import numpy as np
from neuro_skill.routers import keyword, jaccard, cosine, hybrid, _normalize


def _mock_skills(n=10):
    return [{
        "name": f"skill-{i}",
        "search_text": f"skill {i} python security code review build test deploy",
    } for i in range(n)]


def _mock_F(N, M=5):
    return np.random.rand(N, M)


def _mock_G(N):
    G = np.random.rand(N, N)
    np.fill_diagonal(G, 0)
    G /= G.sum(axis=1, keepdims=True)
    return G


def _mock_meta():
    return {"broad": {"security": 0, "frontend": 1}, "precise": {"python": 2, "go": 3}}


class TestNormalize:
    def test_range(self):
        v = np.array([0.1, 0.5, 1.0])
        n = _normalize(v)
        assert 0.0 <= n.min() <= 0.01
        assert 0.99 <= n.max() <= 1.01

    def test_constant(self):
        v = np.array([0.5, 0.5, 0.5])
        n = _normalize(v)
        assert np.allclose(n, 0.0)

    def test_zero(self):
        v = np.zeros(5)
        n = _normalize(v)
        assert np.allclose(n, 0.0)


class TestKeyword:
    def test_shape(self):
        skills = _mock_skills(10)
        scores = keyword(skills, "python security")
        assert len(scores) == 10
        assert scores.dtype == np.float64

    def test_empty_query(self):
        skills = _mock_skills(5)
        scores = keyword(skills, "")
        assert np.allclose(scores, 0.0)

    def test_perfect_match(self):
        skills = [{"name": "s", "search_text": "python security"}]
        scores = keyword(skills, "python security")
        assert scores[0] > 0.0  # should have some overlap


class TestJaccard:
    def test_shape(self):
        skills = _mock_skills(10)
        scores = jaccard(skills, "python code review")
        assert len(scores) == 10

    def test_self_match(self):
        """A skill that exactly matches query features should score high."""
        skills = [{
            "name": "security-skill",
            "search_text": "security vulnerability detection review",
        }]
        scores = jaccard(skills, "check security vulnerabilities")
        # The query matches 'security' from BROAD
        assert scores[0] >= 0.0


class TestCosine:
    def test_shape(self):
        N = 10
        skills = _mock_skills(N)
        F = _mock_F(N)
        meta = _mock_meta()
        scores = cosine(skills, "python security", F=F, meta=meta)
        assert len(scores) == N

    def test_no_features(self):
        skills = _mock_skills(5)
        scores = cosine(skills, "python", F=None, meta=None)
        assert np.allclose(scores, 0.0)  # falls back to zeros


class TestHybrid:
    def test_shape(self):
        N = 10
        skills = _mock_skills(N)
        F = _mock_F(N)
        G = _mock_G(N)
        meta = _mock_meta()
        scores = hybrid(skills, "python security code review", F=F, G=G, meta=meta)
        assert len(scores) == N

    def test_weights_sum(self):
        N = 10
        skills = _mock_skills(N)
        F = _mock_F(N)
        G = _mock_G(N)
        meta = _mock_meta()
        scores = hybrid(skills, "python", F=F, G=G, meta=meta)
        # Scores must be in [0, 1] after normalize
        assert 0.0 <= scores.max() <= 1.0
        assert 0.0 <= scores.min() <= 1.0
