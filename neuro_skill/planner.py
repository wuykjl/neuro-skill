"""
Skill I/O planner — topo-sort routed skills into execution order.

Inspired by agent-skill-finder's Stage 4 (capability-typed I/O planner).

After neuro-skill routes a query to the top-3 skills, this module
produces a valid execution order. If skill A's output is skill B's input,
A runs before B. If no I/O types are declared, returns the order unchanged.

Usage:
  from neuro_skill.planner import plan

  skills = router.query("send message to Zhang San and create meeting", top_k=3)
  plan_result = plan(skills)
  # plan_result.steps = ["lark-contact", "lark-im", "lark-calendar"]
  # plan_result.reasoning = "lark-contact outputs open_id → lark-im needs open_id → ..."
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional


class PlanStep:
    def __init__(self, skill_name: str, score: float,
                 provides: list[str], needs: list[str]):
        self.skill_name = skill_name
        self.score = score
        self.provides = set(provides)
        self.needs = set(needs)


class PlanResult:
    def __init__(self, steps: list[str], reasoning: list[str],
                 unresolved_deps: list[str]):
        self.steps = steps
        self.reasoning = reasoning
        self.unresolved_deps = unresolved_deps
        self.valid = len(unresolved_deps) == 0

    def __repr__(self):
        arrow = " → ".join(self.steps) if self.steps else "(empty)"
        status = "✓" if self.valid else f"⚠ unresolved: {self.unresolved_deps}"
        return f"Plan({arrow}, {status})"

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "reasoning": self.reasoning,
            "unresolved_deps": self.unresolved_deps,
            "valid": self.valid,
        }

    def to_prompt(self) -> str:
        """Format as a prompt fragment for the LLM."""
        if not self.steps:
            return ""
        lines = ["## Skill Execution Plan", ""]
        if not self.valid:
            lines.append(f"⚠ Missing dependencies: {', '.join(self.unresolved_deps)}")
            lines.append("")
        for i, step in enumerate(self.steps):
            lines.append(f"{i+1}. **{step}**")
        if self.reasoning:
            lines.append("")
            lines.append("### Reasoning")
            for r in self.reasoning:
                lines.append(f"- {r}")
        return "\n".join(lines)


def _extract_capabilities(skill: dict) -> tuple[set[str], set[str]]:
    """Extract I/O capabilities from a skill's metadata."""
    meta = skill.get("_meta", {})

    provides = set()
    for p in meta.get("provides", meta.get("outputs", [])):
        name = p if isinstance(p, str) else p.get("name", str(p))
        provides.add(name.lower())

    needs = set()
    for n in meta.get("needs", meta.get("inputs", [])):
        name = n if isinstance(n, str) else n.get("name", str(n))
        needs.add(name.lower())

    return provides, needs


def plan(
    ranked_skills: list[tuple[str, float]],
    skill_index: dict[str, dict],
    dependency_graph: Optional[dict[str, list[str]]] = None,
) -> PlanResult:
    """
    Plan execution order for a set of ranked skills.

    Algorithm:
      1. Extract I/O types from each skill's metadata
      2. Build dependency graph: if skill A provides X and skill B needs X,
         B depends on A
      3. Merge with explicit depends_on edges
      4. Topological sort
      5. Skills at same level are sorted by original routing score

    Args:
      ranked_skills:    [(skill_name, score), ...] from router.query()
      skill_index:      {skill_name: skill_dict} for capability lookup
      dependency_graph: optional explicit {skill_name: [dep_names]}

    Returns:
      PlanResult with ordered steps, reasoning, and validity
    """
    if not ranked_skills:
        return PlanResult([], [], [])

    # Build step objects
    steps = {}
    for name, score in ranked_skills:
        skill = skill_index.get(name, {})
        provides, needs = _extract_capabilities(skill)
        steps[name] = PlanStep(name, score, list(provides), list(needs))

    # If no capabilities declared, just return in score order
    has_capabilities = any(
        ps.provides or ps.needs for ps in steps.values()
    )
    if not has_capabilities and not dependency_graph:
        ordered = [name for name, _ in ranked_skills]
        return PlanResult(
            ordered,
            ["No I/O types declared — returned in routing score order"],
            [],
        )

    # Build dependency graph
    # indegree count → topological sort
    graph: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = defaultdict(int)
    for name in steps:
        indegree[name] = 0  # ensure all nodes exist

    reasoning = []

    # Edges from I/O types
    for name_b, ps_b in steps.items():
        for name_a, ps_a in steps.items():
            if name_a == name_b:
                continue
            # If B needs something that A provides → B depends on A
            overlap = ps_b.needs & ps_a.provides
            if overlap:
                if name_b not in graph.get(name_a, []):
                    graph[name_a].append(name_b)
                    indegree[name_b] += 1
                    reasoning.append(
                        f"{name_a} provides {sorted(overlap)} → required by {name_b}"
                    )

    # Edges from explicit dependency graph
    if dependency_graph:
        for name, deps in dependency_graph.items():
            if name not in steps:
                continue
            for dep in deps:
                if dep in steps and name not in graph.get(dep, []):
                    graph[dep].append(name)
                    indegree[name] += 1
                    reasoning.append(f"{name} depends_on {dep} (explicit)")

    # Topological sort (Kahn's algorithm with heapq for O(N log N))
    import heapq
    heap = []
    for name in steps:
        if indegree[name] == 0:
            heapq.heappush(heap, (-steps[name].score, name))

    ordered = []
    while heap:
        _, node = heapq.heappop(heap)
        ordered.append(node)
        for neighbor in graph.get(node, []):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                heapq.heappush(heap, (-steps[neighbor].score, neighbor))

    # Check for cycles
    unresolved = [name for name in steps if name not in ordered]

    return PlanResult(ordered, reasoning, unresolved)


def quick_plan(router_result: list[tuple[str, float]],
               router: "SkillRouter") -> PlanResult:
    """
    Convenience wrapper: plan from router.query() result.

    Usage:
      results = router.query("send message to Zhang San and create meeting", top_k=3)
      plan_result = quick_plan(results, router)
      print(plan_result.to_prompt())
    """
    skill_index = {name: router.get_skill(name) or {} for name, _ in router_result}
    return plan(router_result, skill_index)
