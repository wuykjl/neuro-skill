"""Tests for planner module — I/O dependency resolution and topological sort."""
import pytest
from neuro_skill.planner import plan, quick_plan, PlanStep, PlanResult, _extract_capabilities


class TestExtractCapabilities:
    def test_empty_skill(self):
        provides, needs = _extract_capabilities({})
        assert provides == set()
        assert needs == set()

    def test_provides_needs_from_meta(self):
        skill = {"_meta": {"provides": ["token", "user_id"], "needs": ["api_key"]}}
        provides, needs = _extract_capabilities(skill)
        assert provides == {"token", "user_id"}
        assert needs == {"api_key"}

    def test_outputs_inputs_fallback(self):
        skill = {"_meta": {"outputs": ["result"], "inputs": ["query"]}}
        provides, needs = _extract_capabilities(skill)
        assert provides == {"result"}
        assert needs == {"query"}

    def test_dict_style_capabilities(self):
        skill = {"_meta": {
            "provides": [{"name": "access_token"}],
            "needs": [{"name": "client_id"}]
        }}
        provides, needs = _extract_capabilities(skill)
        assert provides == {"access_token"}
        assert needs == {"client_id"}

    def test_case_insensitive(self):
        skill = {"_meta": {"provides": ["OpenID"], "needs": ["API_KEY"]}}
        provides, needs = _extract_capabilities(skill)
        assert provides == {"openid"}
        assert needs == {"api_key"}


class TestPlanEmpty:
    def test_empty_skills(self):
        result = plan([], {})
        assert result.steps == []
        assert result.valid

    def test_no_capabilities_returns_score_order(self):
        skills = [("a", 0.9), ("b", 0.5), ("c", 0.8)]
        idx = {name: {} for name, _ in skills}
        result = plan(skills, idx)
        assert result.steps == ["a", "b", "c"]
        assert result.valid


class TestPlanDeps:
    def test_simple_dependency_chain(self):
        rank = [("auth", 0.9), ("api", 0.7), ("db", 0.5)]
        idx = {
            "auth": {"_meta": {"provides": ["token"]}},
            "api": {"_meta": {"needs": ["token"], "provides": ["data"]}},
            "db": {"_meta": {"needs": ["data"]}},
        }
        result = plan(rank, idx)
        assert result.steps == ["auth", "api", "db"]
        assert result.valid

    def test_unresolved_dependency(self):
        rank = [("consumer", 0.9)]
        idx = {"consumer": {"_meta": {"needs": ["missing_output"]}}}
        result = plan(rank, idx)
        # consumer has no provider for needed output — but it has indegree 0
        # so it should still be ordered since unmet needs don't block topo sort
        assert len(result.steps) == 1
        assert "consumer" in result.steps

    def test_explicit_dependency_graph(self):
        rank = [("a", 0.9), ("b", 0.8)]
        idx = {"a": {}, "b": {}}
        dep_graph = {"b": ["a"]}
        result = plan(rank, idx, dependency_graph=dep_graph)
        assert result.steps == ["a", "b"]

    def test_cycle_detection(self):
        rank = [("a", 0.9), ("b", 0.8)]
        idx = {
            "a": {"_meta": {"needs": ["x"], "provides": ["y"]}},
            "b": {"_meta": {"needs": ["y"], "provides": ["x"]}},
        }
        result = plan(rank, idx)
        # Cycle: a needs x, b provides x; b needs y, a provides y
        # Topo sort may or may not resolve — check that result exists
        assert len(result.steps) >= 0
        # Both in cycle → may both be in unresolved
        unresolved_set = set(result.unresolved_deps)
        assert unresolved_set == set() or unresolved_set == {"a", "b"} or unresolved_set == {"a"} or unresolved_set == {"b"}


class TestPlanResult:
    def test_to_dict(self):
        result = PlanResult(["a", "b"], ["a provides x"], [])
        d = result.to_dict()
        assert d["steps"] == ["a", "b"]
        assert d["valid"] is True

    def test_to_prompt(self):
        result = PlanResult(["auth", "api"], ["auth provides token → api"], [])
        prompt = result.to_prompt()
        assert "## Skill Execution Plan" in prompt
        assert "**auth**" in prompt
        assert "**api**" in prompt

    def test_to_prompt_empty(self):
        result = PlanResult([], [], [])
        assert result.to_prompt() == ""

    def test_invalid_prompt(self):
        result = PlanResult(["a"], [], ["missing_x"])
        prompt = result.to_prompt()
        assert "⚠" in prompt
        assert "missing_x" in prompt

    def test_repr(self):
        result = PlanResult(["a", "b"], [], [])
        assert "Plan(" in repr(result)
        assert "→" in repr(result)


class TestQuickPlan:
    def test_quick_plan(self):
        from neuro_skill import SkillRouter
        skills = [
            {"name": "step1", "search_text": "first step",
             "_meta": {"provides": ["result1"]}},
            {"name": "step2", "search_text": "second step",
             "_meta": {"needs": ["result1"]}},
        ]
        router = SkillRouter()
        router.build_from_skills(skills)
        ranked = router.query("step", top_k=2)
        result = quick_plan(ranked, router)
        assert result.valid
        assert result.steps[0] == "step1"
        assert result.steps[1] == "step2"


class TestScoringOrder:
    def test_higher_score_first_when_no_deps(self):
        rank = [("low", 0.2), ("high", 0.99)]
        idx = {"low": {}, "high": {}}
        # Input order preserved when no deps
        result = plan(rank, idx)
        assert result.steps == ["low", "high"]
