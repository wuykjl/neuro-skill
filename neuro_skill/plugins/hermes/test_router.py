"""Tests for neuro-skill-router plugin — run: python test_router.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from __init__ import _tokenize_set, _parse_frontmatter, _KeywordIndex, _check_rules

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

def test_rule_matching():
    import __init__ as plugin
    orig = plugin._RulesCache
    plugin._RulesCache = [{"pattern": "search.*cs", "skill": "csharp-reviewer"}]
    assert _check_rules("search cs files") == "csharp-reviewer"
    assert _check_rules("python code") is None
    plugin._RulesCache = orig

def test_edge_cases():
    idx = _KeywordIndex([])
    assert idx.query("anything", top_k=3) == []
    idx2 = _KeywordIndex([("only", "Only", "only one skill")])
    r = idx2.query("only skill", top_k=3)
    assert len(r) == 1 and r[0][0] == "only"

def test_pre_llm_call():
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
    tests = [test_tokenize, test_bm25_routing, test_rule_matching,
             test_edge_cases, test_pre_llm_call, test_zero_imports]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
    print(f"{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
