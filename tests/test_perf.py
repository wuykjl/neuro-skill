"""Performance benchmarks — regression guard for latency and build time."""
import time, pytest
from neuro_skill import SkillRouter


def _mock_skills(n, base="python security code review test build deploy"):
    return [{"name": f"skill-{i}", "search_text": f"{base} skill-{i}"}
            for i in range(n)]


class TestQueryLatency:
    def test_query_under_50ms(self):
        """100-skill query should complete in < 50ms."""
        skills = _mock_skills(100)
        router = SkillRouter()
        router.build_from_skills(skills)
        # Warmup
        for _ in range(3):
            router.query("python code review", top_k=5)
        # Measure
        t0 = time.perf_counter()
        for _ in range(10):
            router.query("python code review", top_k=5)
        avg_ms = (time.perf_counter() - t0) * 1000 / 10
        assert avg_ms < 50, f"Query avg {avg_ms:.0f}ms, expected < 50ms"

    def test_build_under_1s(self):
        """300-skill build should complete in < 1 second."""
        skills = _mock_skills(300)
        t0 = time.perf_counter()
        router = SkillRouter()
        stats = router.build_from_skills(skills)
        elapsed = time.perf_counter() - t0
        assert stats["n_skills"] == 300
        assert elapsed < 1.0, f"Build {elapsed:.1f}s > 1s"

    def test_cold_first_query_ok(self):
        """First query after build should not crash."""
        skills = _mock_skills(50)
        router = SkillRouter()
        router.build_from_skills(skills)
        results = router.query("python", top_k=5)
        assert len(results) == 5


class TestGraphDensity:
    def test_knn_density_in_range(self):
        """k-NN graph density should be bounded."""
        skills = _mock_skills(200)
        router = SkillRouter()
        stats = router.build_from_skills(skills)
        density = stats["graph_density"]
        # k-NN with k ~ sqrt(N)/3 ≈ 5 for 200 skills
        # After symmetrization, density should be 0.02–0.25
        assert 0.01 < density < 0.30, f"Graph density {density:.3f} out of range"
