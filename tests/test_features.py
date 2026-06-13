"""Test feature extraction and tokenization."""
import pytest
from neuro_skill.features import tokenize, _match, extract_query_features, extract_skill_features, feature_set, BROAD, PRECISE


def test_tokenize_english():
    tokens = tokenize("python security code review")
    assert "python" in tokens
    assert "security" in tokens
    assert "code" in tokens
    assert "review" in tokens


def test_tokenize_chinese():
    tokens = tokenize("Python code security check")
    assert "python" in tokens
    assert "code" in tokens
    assert "security" in tokens
    assert "check" in tokens


def test_tokenize_deduplicates():
    tokens = tokenize("python python python")
    assert len([t for t in tokens if t == "python"]) == 1


def test_tokenize_short_filtered():
    """tokenize() keeps 2-char words; _match() filters stopwords."""
    tokens = tokenize("is be of in on code")
    # tokenize keeps ALL words >= 2 chars
    assert "is" in tokens  # tokenize keeps short words
    assert "code" in tokens


def test_match_security():
    matched = _match("check python code for sql injection vulnerabilities", BROAD)
    assert "security" in matched


def test_match_frontend():
    matched = _match("react component with tailwind css review", BROAD)
    assert "frontend" in matched


def test_match_database():
    matched = _match("optimize postgresql database query performance", BROAD)
    assert "database" in matched


def test_match_python_precise():
    matched = _match("python django fastapi application", PRECISE)
    assert "python" in matched


def test_match_go_precise():
    matched = _match("golang goroutine concurrency patterns", PRECISE)
    assert "go" in matched


def test_match_chinese_security():
    matched = _match("检查代码有没有SQL注入漏洞", BROAD)
    assert "security" in matched


def test_match_chinese_build():
    matched = _match("构建报错排查编译失败", PRECISE)
    assert "build_fix" in matched


def test_feature_set():
    feats = {"broad": {"security", "frontend"}, "precise": {"python"}}
    assert feature_set(feats) == {"security", "frontend", "python"}


def test_empty_match():
    matched = _match("xyzzy quux wibble", BROAD)
    assert len(matched) == 0
    matched = _match("xyzzy quux wibble", PRECISE)
    assert len(matched) == 0


def test_skill_features():
    skill = {"name": "test-skill", "search_text": "python security code review specialist"}
    feats = extract_skill_features(skill)
    assert "security" in feats["broad"]
    assert "python" in feats["precise"]


def test_query_features():
    feats = extract_query_features("检查python代码sql注入")
    assert "security" in feats["broad"]
    assert "python" in feats["precise"]


def test_regex_pattern():
    """\\bgo\\b should match standalone 'go' but not 'golang'."""
    matched = _match("go build command", PRECISE)
    assert "go" in matched


def test_broad_precise_feature_count():
    """Ensure base features exist."""
    assert len(BROAD) == 15, f"Expected 15 broad, got {len(BROAD)}"
    assert len(PRECISE) == 23, f"Expected 23 precise, got {len(PRECISE)}"
