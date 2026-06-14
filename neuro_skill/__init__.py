"""
neuro-skill — Zero-cost hybrid skill router for AI agents.

Two ways to use:

  # Explicit (you control lifecycle):
      from neuro_skill import SkillRouter
      router = SkillRouter()
      router.build(["./skills/", "./agents/"])
      router.query("python security review", top_k=5)

  # Auto-start (first call builds, stays hot):
      from neuro_skill import query
      query("Go build error", top_k=3)
      # No build(), no server commands. Just import and call.
"""

from neuro_skill.router import SkillRouter

# Lazy auto-start query — first call ~500ms, then 5ms
from neuro_skill.autostart import query

__version__ = "0.6.1"
__all__ = ["SkillRouter", "query"]
