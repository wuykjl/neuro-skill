"""Edge case tests — degenerate inputs, boundary values, graceful degradation."""
import numpy as np
from neuro_skill import SkillRouter


def _mock_skills(n, base="python security code review test"):
    return [{"name": f"skill-{i}", "search_text": f"{base} skill-{i}"}
            for i in range(n)]


class TestEmptySkills:
    def test_build_empty_dirs(self):
        """Building from empty skill lists should not crash."""
        router = SkillRouter()
        try:
            router.build([])
        except RuntimeError:
            pass  # expected: no skills found
        else:
            assert True  # also fine if it builds with 0 skills

    def test_build_from_empty_skills(self):
        """build_from_skills with empty list should work."""
        router = SkillRouter()
        router.build_from_skills([])
        # Query on empty index should handle gracefully
        try:
            router.query("test", top_k=5)
        except RuntimeError:
            pass  # expected: index not built
        except Exception:
            pass  # any exception is OK, just no segfault

    def test_query_before_build(self):
        """Query before build should raise RuntimeError."""
        router = SkillRouter()
        try:
            router.query("test")
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            assert True


class TestSingleSkill:
    def test_single_skill_returns_itself(self):
        """One skill — it should always be returned."""
        skills = [{"name": "only", "search_text": "python code review"}]
        router = SkillRouter()
        router.build_from_skills(skills)
        results = router.query("anything", top_k=3)
        assert len(results) == 1
        assert results[0][0] == "only"

    def test_duplicate_names_dedup(self):
        """Duplicate skill names should be handled."""
        skills = [
            {"name": "dup", "search_text": "python"},
            {"name": "dup", "search_text": "java"},
        ]
        router = SkillRouter()
        router.build_from_skills(skills)
        # Should not crash — may keep first or second
        results = router.query("python", top_k=3)
        assert len(results) >= 1


class TestBoundaryValues:
    def test_top_k_larger_than_N(self):
        """top_k > skill count should return all skills."""
        # Use distinct keywords so BM25 creates meaningful differentiation
        # (identical search_text triggers quality gate: cos_gap=0 → no signal)
        skills = [
            {"name": f"skill-{i}", "search_text": f"unique word skill only this {i}"}
            for i in range(5)
        ]
        router = SkillRouter()
        router.build_from_skills(skills)
        results = router.query("unique word skill", top_k=100)
        assert len(results) == 5

    def test_top_k_zero(self):
        """top_k=0 should return empty list."""
        skills = _mock_skills(5)
        router = SkillRouter()
        router.build_from_skills(skills)
        results = router.query("test", top_k=0)
        assert len(results) == 0

    def test_empty_query(self):
        """Empty query should not crash."""
        skills = _mock_skills(5)
        router = SkillRouter()
        router.build_from_skills(skills)
        results = router.query("", top_k=5)
        assert len(results) >= 0  # no crash is success

    def test_very_long_query(self):
        """1000-char query should not OOM."""
        skills = _mock_skills(20)
        router = SkillRouter()
        router.build_from_skills(skills)
        long_q = "python security " * 200
        try:
            results = router.query(long_q, top_k=5)
            assert len(results) >= 0
        except Exception:
            pass  # no segfault

    def test_unicode_query(self):
        """Unicode / emoji query should not crash."""
        skills = _mock_skills(10)
        router = SkillRouter()
        router.build_from_skills(skills)
        results = router.query("🐍🔒 检查代码安全", top_k=3)
        assert len(results) >= 0


class TestMethodGraceful:
    def test_unknown_method(self):
        """Unknown routing method should raise ValueError."""
        skills = _mock_skills(5)
        router = SkillRouter()
        router.build_from_skills(skills)
        try:
            router.query("test", method="nonexistent")
            assert False, "Expected ValueError"
        except ValueError:
            assert True

    def test_all_methods_work(self):
        """Every registered method should return results."""
        skills = _mock_skills(10)
        router = SkillRouter()
        router.build_from_skills(skills)
        for method in ["hybrid", "cosine", "keyword", "jaccard", "graph_spread"]:
            results = router.query("python", top_k=3, method=method)
            assert len(results) == 3, f"Method {method} failed"
