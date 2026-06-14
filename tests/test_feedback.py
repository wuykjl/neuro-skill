"""Test Error Book feedback mechanism."""
import time, tempfile, os
from pathlib import Path
from neuro_skill.feedback import ErrorBook, _query_hash


class TestQueryHash:
    def test_same_query_same_hash(self):
        assert _query_hash("go build error fix") == _query_hash("go build error fix")

    def test_similar_queries_same_hash(self):
        """Queries whose first 5 tokens are identical should hash same."""
        h1 = _query_hash("build error fix compile link")
        h2 = _query_hash("build error fix compile link failed again")
        assert h1 == h2  # both produce "build error fix compile link" as key

    def test_different_queries_different_hash(self):
        h1 = _query_hash("Go构建报错排查")
        h2 = _query_hash("Python SQL注入检查")
        assert h1 != h2

    def test_short_query(self):
        h = _query_hash("x")
        assert len(h) == 12


class TestErrorBook:
    def test_correct_and_adjust(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            book = ErrorBook(path)
            book.correct("build error fix compile link", "go-build-resolver")

            # Same query should boost
            scores = [0.044, 0.040, 0.038]
            names = ["find-skills", "go-build-resolver", "docx"]
            adj = book.adjust("build error fix compile link", scores, names)
            assert adj[1] > scores[1]  # go-build-resolver boosted

            # Similar query (same 5-token prefix → same hash) also boosted
            adj2 = book.adjust("build error fix compile link failure", scores, names)
            assert adj2[1] > scores[1]

            # Unrelated query should not boost
            adj3 = book.adjust("python code security review", scores, names)
            assert adj3[1] == scores[1]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_decay(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            book = ErrorBook(path)
            # Create a correction 14 days ago
            old_ts = time.time() - 14 * 86400
            book.correct("go build error fix", "go-build-resolver", timestamp=old_ts)

            scores = [0.044, 0.040, 0.038]
            names = ["find-skills", "go-build-resolver", "docx"]
            # With 7-day half-life, 14-day-old boost = 0.25x strength
            adj = book.adjust("go build error fix", scores, names, decay_days=7.0)

            # Should still boost, but much less
            assert adj[1] > scores[1]  # boost exists
            # Boost should be ~0.003 (0.012 * 1.0 * 0.25)
            assert adj[1] - scores[1] < 0.005  # low boost
        finally:
            Path(path).unlink(missing_ok=True)

    def test_accumulation(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            book = ErrorBook(path)
            book.correct("go build error fix", "go-build-resolver")
            book.correct("go build error fix", "go-build-resolver")  # reinforce

            scores = [0.044, 0.040, 0.038]
            names = ["find-skills", "go-build-resolver", "docx"]
            adj = book.adjust("go build error fix", scores, names)
            # Two corrections = 2x boost
            assert adj[1] - scores[1] > 0.015
        finally:
            Path(path).unlink(missing_ok=True)

    def test_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            book = ErrorBook(path)
            book.correct("q1", "s1")
            book.correct("q2", "s2")
            s = book.stats()
            assert s["entries"] == 2
            assert s["total_boosts"] >= 1.9
        finally:
            Path(path).unlink(missing_ok=True)

    def test_clear(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            book = ErrorBook(path)
            book.correct("q1", "s1")
            book.clear()
            s = book.stats()
            assert s["entries"] == 0
        finally:
            Path(path).unlink(missing_ok=True)


class TestRouterIntegration:
    def test_learn_and_query(self):
        from neuro_skill import SkillRouter
        import os

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fb_path = f.name
        try:
            router = SkillRouter(feedback_path=fb_path)
            dirs = [os.path.expanduser("~/.claude/agents/")]
            router.build(dirs)

            # Query first
            results = router.query("Go build error fix", top_k=5, enable_feedback=True)
            assert len(results) == 5

            # Learn
            router.learn("Go build error fix", "go-build-resolver")
            assert Path(fb_path).exists()

            # Query again — should boost go-build-resolver
            results2 = router.query("Go build error fix", top_k=5, enable_feedback=True)
            names = [r[0] for r in results2]
            assert "go-build-resolver" in names

            # Query with feedback disabled — no boost
            results3 = router.query("Go build error fix", top_k=5, enable_feedback=False)
            assert len(results3) == 5
        finally:
            Path(fb_path).unlink(missing_ok=True)
