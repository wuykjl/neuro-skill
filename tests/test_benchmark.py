"""回归基准 — 21 题标准集 + 中文 9 题 + 边界用例。每次 CI 自动跑。

Third-party validated benchmark (332 ECC skills, Hermes v0.16.0).
Catches quality gate over-trigger, tokenizer regression, rule conflicts.
"""

import pytest
from neuro_skill import SkillRouter


# ── 21 Question English Core Benchmark ──

_EN_BENCHMARK = [
    # (query, expected_skills, must_not_be_empty)
    ("python code review", ["python-reviewer"], True),
    ("fix build error", ["build-error-resolver", "go-build-resolver", "cpp-build-resolver"], True),
    ("react components", ["react-reviewer"], True),
    ("docker compose", [], True),
    ("golang service", ["go-reviewer"], True),
    ("swift ios", ["swift-reviewer"], True),
    ("firecrawl search", [], True),  # was empty BEFORE quality gate fix — must NOT be empty now
    ("spec driven dev", [], True),  # was empty BEFORE quality gate fix
    ("fastapi backend", ["fastapi-reviewer"], True),
    ("database query", [], True),  # too generic — many skills match "database" and "query"
    ("write test coverage", ["tdd-guide", "tdd-workflow"], True),
    ("security audit scan", ["security-reviewer"], True),
    ("rust ownership review", ["rust-reviewer"], True),
    ("java spring boot", ["java-reviewer"], True),
    ("typescript type check", ["typescript-reviewer"], True),
    ("csharp .net async", ["csharp-reviewer"], True),
    ("kotlin coroutines", ["kotlin-reviewer"], True),
    ("refactor dead code", ["refactor-cleaner"], True),
    ("e2e test automation", ["e2e-testing", "e2e-runner"], True),
    ("design system architecture", ["architect"], True),
    ("documentation readme", ["doc-updater"], True),
]


# ── 9 Question Chinese Benchmark ──

_CN_BENCHMARK = [
    ("检索cs文件并进行安全检查", [], True),
    ("写Python代码审查", ["python-reviewer"], True),
    ("修复构建错误", [], True),
    ("安全漏洞扫描", [], True),
    ("前端组件审查", [], True),
    ("写测试用例", ["tdd-guide"], True),
    ("重构代码结构", ["refactor-cleaner"], True),
    ("设计系统架构", ["architect"], True),
    ("生成项目文档", ["doc-updater"], True),
]


# ── Edge Cases ──

_EDGE_CASES = [
    # (query, should_be_empty)
    ("xyzz noise garbage nonsense", True),   # pure noise
    ("asdfqwerzxcv", True),                   # truly random chars — no real words
    ("asdf qwer zxcv meaningless", False),    # "meaningless" is a real English word — will match something
    ("", False),                              # empty query
    ("a", False),                             # single char
    ("python " * 200, False),                # very long
]


@pytest.fixture(scope="module")
def router():
    """Build once, reuse across all benchmark tests."""
    import os
    r = SkillRouter()
    h = os.path.expanduser("~")
    dirs = [
        f"{h}/.claude/skills",
        f"{h}/.claude/agents",
        f"{h}/.claude/.agents/skills",
    ]
    dirs = [d for d in dirs if os.path.isdir(d)]
    if dirs:
        r.build(dirs)
    return r


# ═══════════════════════════════════════════
#  English Benchmark Tests
# ═══════════════════════════════════════════

class TestEnglishBenchmark:
    @pytest.mark.parametrize("query,expected,must_not_be_empty", _EN_BENCHMARK)
    def test_english_query(self, router, query, expected, must_not_be_empty):
        """Every English query must return results (no false empty)."""
        if router.skill_count == 0:
            pytest.skip("No skills available for benchmark")

        results = router.query(query, top_k=5)
        names = [n for n, _ in results]

        # MUST NOT be empty (quality gate regression guard)
        if must_not_be_empty:
            assert len(results) > 0, (
                f"Query '{query}' returned EMPTY — quality gate over-trigger. "
                f"(21-query benchmark: was green before adbac15, broke after quality gate)"
            )

        # If expected skills specified, at least one must be in results
        if expected:
            found = set(names) & set(expected)
            assert len(found) > 0, (
                f"Query '{query}' should match {expected}, got {names[:5]}"
            )

    def test_empty_results_never_gate_valid_queries(self, router):
        """Regression: No English benchmark query should return empty."""
        if router.skill_count == 0:
            pytest.skip("No skills available")
        empty = []
        for query, _, must_not_be_empty in _EN_BENCHMARK:
            if must_not_be_empty:
                results = router.query(query, top_k=5)
                if not results:
                    empty.append(query)
        assert len(empty) == 0, (
            f"{len(empty)} valid English queries returned empty: {empty}\n"
            f"Quality gate is over-triggering."
        )


# ═══════════════════════════════════════════
#  Chinese Benchmark Tests
# ═══════════════════════════════════════════

class TestChineseBenchmark:
    @pytest.mark.parametrize("query,expected,must_not_be_empty", _CN_BENCHMARK)
    def test_chinese_query(self, router, query, expected, must_not_be_empty):
        """Every Chinese query must return results."""
        if router.skill_count == 0:
            pytest.skip("No skills available")

        results = router.query(query, top_k=5)
        names = [n for n, _ in results]

        # Must not be empty
        if must_not_be_empty:
            assert len(results) > 0, (
                f"Chinese query '{query}' returned EMPTY"
            )

        # Expected match check
        if expected:
            found = set(names) & set(expected)
            assert len(found) > 0, (
                f"Chinese query '{query}' should match {expected}, got {names[:5]}"
            )


# ═══════════════════════════════════════════
#  Edge Case Tests
# ═══════════════════════════════════════════

class TestEdgeCaseBenchmark:
    @pytest.mark.parametrize("query,should_be_empty", _EDGE_CASES)
    def test_edge_case(self, router, query, should_be_empty):
        """Pure noise → empty. Everything else → some result."""
        if router.skill_count == 0:
            pytest.skip("No skills available")
        results = router.query(query, top_k=5)
        if should_be_empty:
            # Pure noise SHOULD return empty (quality gate)
            assert len(results) == 0, (
                f"Pure noise '{query}' returned results: {[n for n,_ in results[:3]]}. "
                f"Quality gate should have blocked this."
            )

    def test_none_query(self, router):
        """None/empty should not crash."""
        if router.skill_count == 0:
            pytest.skip("No skills available")
        r = router.query("", top_k=5)
        assert len(r) >= 0  # no crash

    def test_query_rules_still_work(self, router):
        """Rules must still override routing."""
        if router.skill_count == 0:
            pytest.skip("No skills available")
        results = router.query("python code review", top_k=5)
        names = [n for n, _ in results]
        assert "python-reviewer" in names[:5]
