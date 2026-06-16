"""Tests for neuro-skill-router plugin — 5-layer routing pipeline."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from __init__ import (
    _tokenize_set, _parse_frontmatter, _KeywordIndex,
    _check_rules, _is_task_query, _trigger_match,
    _levenshtein, _fuzzy_keyword_correction,
)

def test_tokenize():
    ts = _tokenize_set("python code review")
    assert "python" in ts and "code" in ts and "review" in ts
    ts_cn = _tokenize_set("test file search locate")
    assert len(ts_cn) >= 3

def test_bm25_routing():
    skills = [
        ("python-reviewer", "Python review", "python pep8 security code review"),
        ("go-builder", "Go fix", "go golang build fix error"),
        ("security-scanner", "Audit", "security vulnerability CVE owasp scan"),
    ]
    idx = _KeywordIndex(skills)
    r = idx.query("python code review", top_k=3)
    assert r[0][0] == "python-reviewer"
    r2 = idx.query("xyz no match whatever", top_k=3)
    assert len(r2) == 0

def test_edge_cases():
    idx = _KeywordIndex([])
    assert idx.query("anything", top_k=3) == []
    idx2 = _KeywordIndex([("only", "Only", "only one skill")])
    r = idx2.query("only skill", top_k=3)
    assert len(r) == 1 and r[0][0] == "only"

def test_frontmatter():
    text = """---
name: test-skill
description: Test description
---
# Body"""
    meta, body = _parse_frontmatter(text)
    assert meta["name"] == "test-skill"

def test_rule_matching():
    import __init__ as plugin
    orig = plugin._RulesCache
    plugin._RulesCache = [{"pattern": "search.*cs", "skill": "csharp-reviewer"}]
    assert _check_rules("search cs files") == "csharp-reviewer"
    assert _check_rules("python code") is None
    plugin._RulesCache = orig

# ── Layer 1: Task Gate ──

def test_task_gate():
    for q in ["thanks", "好的", "ok", "got it", "讲得很好", "bye", "good morning"]:
        assert not _is_task_query(q), f"'{q}' should be skipped"
    for q in ["帮我优化网站", "python code review", "fix build error", "部署到生产"]:
        assert _is_task_query(q), f"'{q}' should be a task"
    assert not _is_task_query("")
    assert not _is_task_query("h")

# ── Layer 2: Trigger Match ──

def test_trigger_match():
    import __init__ as plugin
    orig = plugin._TriggerIndex
    plugin._TriggerIndex = {
        "help me optimize": "perf-optimizer",
        "check for security issues": "security-scanner",
    }
    assert _trigger_match("help me optimize") == "perf-optimizer"
    assert _trigger_match("i need help me optimize my website") == "perf-optimizer"
    assert _trigger_match("python code review") is None
    plugin._TriggerIndex = orig

# ── Layer 4: Levenshtein ──

def test_levenshtein():
    assert _levenshtein("pythn", "python") == 1
    assert _levenshtein("go", "go") == 0
    assert _levenshtein("", "abc") == 3

def test_fuzzy_correction():
    known = {"python", "security", "review", "code"}
    c = _fuzzy_keyword_correction("pythn securty revew", known)
    assert "python" in c and "security" in c and "review" in c

# ── pre_llm_call integration ──

def test_pre_llm_call_layers():
    import __init__ as plugin
    plugin._RouteIndex = _KeywordIndex([("test", "Test", "test skill context format")])
    from __init__ import pre_llm_call
    r = pre_llm_call(user_message="test skill")
    assert "context" in r and "Top 3" in r["context"]
    assert pre_llm_call(user_message="") == {}

def test_zero_imports():
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__init__.py")
    src = open(fp, encoding="utf-8").read()
    assert "from neuro_skill" not in src
    assert "import neuro_skill" not in src

if __name__ == "__main__":
    tests = [test_tokenize, test_bm25_routing, test_edge_cases, test_frontmatter,
             test_rule_matching, test_task_gate, test_trigger_match,
             test_levenshtein, test_fuzzy_correction, test_pre_llm_call_layers,
             test_zero_imports]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
    print(f"{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
