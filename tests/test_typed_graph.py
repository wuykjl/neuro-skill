"""Tests for typed_graph module — skill dependency parsing and edge discovery."""
import numpy as np
from neuro_skill.typed_graph import (
    parse_skill_dependencies,
    _name_to_idx,
    build_typed_matrices,
    auto_discover_edges,
)


class TestParseSkillDependencies:
    def test_empty_skill(self):
        result = parse_skill_dependencies({})
        assert result["depends_on"] == []
        assert result["complements"] == []
        assert result["provides"] == []

    def test_requires_field_in_meta(self):
        skill = {"_meta": {"requires": ["auth", "db"]}}
        result = parse_skill_dependencies(skill)
        assert result["depends_on"] == ["auth", "db"]

    def test_depends_on_field_in_meta(self):
        skill = {"_meta": {"depends_on": ["auth"], "complements": ["logger"]}}
        result = parse_skill_dependencies(skill)
        assert result["depends_on"] == ["auth"]
        assert result["complements"] == ["logger"]

    def test_body_yaml_frontmatter(self):
        skill = {"_body": "---\nrequires:\n  - auth\n  - db\n---\nSome description"}
        result = parse_skill_dependencies(skill)
        assert "auth" in result["depends_on"]
        assert "db" in result["depends_on"]


class TestNameToIdx:
    def test_found(self):
        skills = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        assert _name_to_idx(skills, "b") == 1

    def test_not_found(self):
        skills = [{"name": "a"}]
        assert _name_to_idx(skills, "missing") is None

    def test_empty_list(self):
        assert _name_to_idx([], "a") is None


class TestBuildTypedMatrices:
    def test_empty_skills(self):
        dep, comp = build_typed_matrices([])
        assert dep.shape == (0, 0)
        assert comp.shape == (0, 0)

    def test_single_skill(self):
        skills = [{"name": "only", "search_text": "test"}]
        dep, comp = build_typed_matrices(skills)
        assert dep.shape == (1, 1)
        assert comp.shape == (1, 1)

    def test_dependency_matrix(self):
        skills = [
            {"name": "a", "_meta": {"requires": ["b"]}},
            {"name": "b", "search_text": "base"},
        ]
        dep, comp = build_typed_matrices(skills)
        # a requires b → dep[a_idx][b_idx] = 1 (G_d[i][j]=1 means skill_i requires skill_j)
        assert dep[0, 1] == 1.0  # a → b edge


class TestAutoDiscoverEdges:
    def test_empty_skills(self):
        result = auto_discover_edges([])
        assert "depends_on" in result
        assert "complements" in result

    def test_auth_chain_discovery(self):
        skills = [
            {"name": "auth", "search_text": "authentication login OAuth token"},
            {"name": "api", "search_text": "API call request authenticated requires auth"},
        ]
        result = auto_discover_edges(skills)
        assert isinstance(result, dict)

    def test_review_test_chain(self):
        skills = [
            {"name": "code-reviewer", "search_text": "review code quality PR"},
            {"name": "security-reviewer", "search_text": "review security vulnerabilities CVE scan"},
            {"name": "tdd", "search_text": "test driven development write tests"},
        ]
        result = auto_discover_edges(skills)
        assert isinstance(result, dict)

    def test_same_language_pairing(self):
        skills = [
            {"name": "python-reviewer", "search_text": "review python code PEP8"},
            {"name": "python-tester", "search_text": "test python pytest"},
        ]
        result = auto_discover_edges(skills)
        assert isinstance(result, dict)
