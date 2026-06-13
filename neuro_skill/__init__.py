"""
neuro-skill — Zero-cost hybrid skill router for AI agents.

Graph spreading activation + cosine similarity + keyword fusion.
Pure local, 40ms/query, no API calls.

Usage:
    from neuro_skill import SkillRouter

    router = SkillRouter()
    router.build(["~/.claude/skills/", "~/.claude/agents/"])
    results = router.query("check Python code for SQL injection", top_k=5)
    for name, score in results:
        print(f"  {name}: {score:.3f}")
"""

from neuro_skill.router import SkillRouter

__version__ = "0.2.0"
__all__ = ["SkillRouter"]
