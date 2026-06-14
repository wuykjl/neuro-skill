#!/usr/bin/env python
"""
Tech Research Pipeline — search → structure → save → retrieve.

Prevents the "didn't know it existed" problem by proactively searching
for mature technologies before building, and storing findings in Obsidian
for future retrieval.

Modes:
  research <domain>  — search web, save to Obsidian
  retrieve <query>   — search local Obsidian for prior research
  before <project>   — run before starting a project: search + save + link
  after <project>    — run after finishing: retrospective search for missed tech

Usage:
  python tech_research.py research "skill routing agent tool selection"
  python tech_research.py retrieve "graph knowledge database"
  python tech_research.py before neuro-skill
  python tech_research.py after neuro-skill
"""

from __future__ import annotations

import os, re, json, subprocess, time, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

# ── Config ──

OBSIDIAN_ROOT = Path("E:/内网/Obsidian知识库/内网笔记本")
RESEARCH_DIR = OBSIDIAN_ROOT / "技术研究"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

# Local timezone (UTC+8)
TZ = timezone(timedelta(hours=8))


# ═══════════════════════════════════════════════════════════════
# 1. Research — search web and save to Obsidian
# ═══════════════════════════════════════════════════════════════

def search_github_repos(query: str, n: int = 10) -> list[dict]:
    """Search GitHub for relevant open-source projects."""
    try:
        result = subprocess.run(
            ["gh", "search", "repos", query,
             "--sort", "stars", "--limit", str(n),
             "--json", "name,fullName,url,description,stargazersCount,language,license,updatedAt"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            repos = json.loads(result.stdout)
            return [{
                "name": r.get("fullName", r.get("name", "")),
                "url": r.get("url", ""),
                "description": r.get("description", "") or "",
                "stars": r.get("stargazersCount", 0),
                "language": r.get("language", ""),
                "license": (r.get("license") or {}).get("name", "") if r.get("license") else "",
                "updated": r.get("updatedAt", "")[:10] if r.get("updatedAt") else "",
            } for r in repos]
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return []


def search_web_papers(query: str) -> list[dict]:
    """Search for relevant papers and articles. Returns structured results.

    Note: This is a placeholder. In production, use WebSearch/WebFetch.
    The results should be populated by the caller (agent) using available tools.
    """
    return []


def _format_github_results(repos: list[dict]) -> str:
    if not repos:
        return ""
    lines = ["## GitHub 项目", ""]
    lines.append("| 项目 | Stars | 语言 | 许可证 | 描述 |")
    lines.append("|------|-------|------|--------|------|")
    for r in repos[:12]:
        desc = (r["description"] or "")[:80].replace("|", "\\|")
        lic = (r["license"] or "")[:15]
        lines.append(
            f"| [{r['name']}]({r['url']}) | {r['stars']} | "
            f"{r.get('language','')} | {lic} | {desc} |"
        )
    lines.append("")
    return "\n".join(lines)


def research_and_save(
    domain: str,
    project: Optional[str] = None,
    github_query: Optional[str] = None,
    key_findings: Optional[list[str]] = None,
    related_tech: Optional[list[str]] = None,
    notes: Optional[str] = None,
) -> dict:
    """
    Research a technology domain: search web, save structured findings to Obsidian.

    Args:
      domain:        technology domain to research (e.g. "skill routing agent tool selection")
      project:       optional project name to link to
      github_query:  custom GitHub search query
      key_findings:  manually provided key findings (from LLM analysis)
      related_tech:  related technologies discovered
      notes:         additional notes

    Returns:
      {"saved": <path>, "github_results": [...], "note_content": "..."}
    """
    now = datetime.now(TZ)
    safe_name = re.sub(r'[\\/:*?"<>|]', '-', domain)[:60]
    filename = f"{safe_name}.md"
    filepath = RESEARCH_DIR / filename

    # Search GitHub
    gh_query = github_query or domain
    github_results = search_github_repos(gh_query)

    # Build the note
    lines = [
        "---",
        f"domain: {domain}",
        f"date: {now.strftime('%Y-%m-%d')}",
        f"tags: [技术研究, {domain.split()[0] if domain else '未分类'}]",
        "---",
        "",
        f"# {domain}",
        "",
        f"> 研究日期: {now.strftime('%Y-%m-%d %H:%M')}",
    ]

    if project:
        lines.append(f"> 关联项目: [[{project}]]")
        lines.append("")

    lines.append("")

    # Key findings
    if key_findings:
        lines.append("## 关键发现")
        lines.append("")
        for f in key_findings:
            lines.append(f"- {f}")
        lines.append("")

    # GitHub results
    if github_results:
        lines.append(_format_github_results(github_results))

    # Related technologies
    if related_tech:
        lines.append("## 相关技术")
        lines.append("")
        for t in related_tech:
            lines.append(f"- [[{t}]]" if "[[" not in t else f"- {t}")
        lines.append("")

    # Notes
    if notes:
        lines.append("## 笔记")
        lines.append("")
        lines.append(notes)
        lines.append("")

    content = "\n".join(lines)
    filepath.write_text(content, encoding="utf-8")

    return {
        "saved": str(filepath),
        "domain": domain,
        "github_results": github_results,
        "note_content": content,
    }


# ═══════════════════════════════════════════════════════════════
# 2. Retrieve — search local Obsidian for prior research
# ═══════════════════════════════════════════════════════════════

def retrieve_research(
    query: str,
    max_results: int = 10,
    search_body: bool = True,
) -> list[dict]:
    """
    Search the local Obsidian knowledge base for relevant prior research.

    Searches both the 技术研究 folder and general notes for matching content.
    Returns results sorted by relevance.
    """
    results = []

    # Search file names first (fast)
    for f in RESEARCH_DIR.rglob("*.md"):
        if _match_query(query, f.name):
            results.append(_read_note(f, "filename_match"))
        elif search_body:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                if _match_query(query, content[:2000]):
                    results.append(_read_note(f, "content_match"))
            except Exception as e:
                import logging
                logging.getLogger("neuro_skill.tech").debug("File read error: %s", e)

    # Also search general notes for technology references
    for f in OBSIDIAN_ROOT.rglob("*.md"):
        if "技术研究" in str(f) or f.parent == RESEARCH_DIR:
            continue
        if _match_query(query, f.name):
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                # Only include if it looks like a tech reference
                if "github" in content.lower() or "stars" in content.lower() or \
                   "技术" in content or "开源" in content or "agent" in content.lower():
                    results.append(_read_note(f, "general_note"))
            except Exception as e:
                import logging
                logging.getLogger("neuro_skill.tech").debug("File read error: %s", e)
        if len(results) >= max_results:
            break

    # Sort: filename matches first, then content matches
    results.sort(key=lambda r: 0 if "filename" in r["match_type"] else 1)
    return results[:max_results]


def _match_query(query: str, text: str) -> bool:
    """Simple multi-keyword match."""
    keywords = re.findall(r'\w{2,}', query.lower())
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches >= max(1, len(keywords) // 2)


def _read_note(f: Path, match_type: str) -> dict:
    stat = f.stat()
    content = f.read_text(encoding="utf-8", errors="ignore")
    # Extract frontmatter
    meta = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()

    # Extract summary (first non-heading paragraph after frontmatter)
    body = parts[2] if content.startswith("---") and len(parts) >= 3 else content
    summary = ""
    for line in body.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith(">"):
            summary = line[:200]
            break

    return {
        "file": str(f.relative_to(OBSIDIAN_ROOT)),
        "path": str(f),
        "domain": meta.get("domain", ""),
        "date": meta.get("date", ""),
        "tags": meta.get("tags", ""),
        "summary": summary,
        "match_type": match_type,
        "size_kb": round(stat.st_size / 1024, 1),
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=TZ).strftime("%Y-%m-%d"),
    }


# ═══════════════════════════════════════════════════════════════
# 3. Before/After project hooks
# ═══════════════════════════════════════════════════════════════

def before_project(project_name: str, domain: str) -> dict:
    """Run before starting a project. Searches for prior art and saves findings."""
    result = research_and_save(
        domain=domain,
        project=project_name,
        notes=f"建项前研究 — {project_name}。研究日期: {datetime.now(TZ).strftime('%Y-%m-%d')}",
    )
    # Also check for related prior research
    related = retrieve_research(domain, max_results=5)
    if related:
        with open(result["saved"], "a", encoding="utf-8") as f:
            f.write("\n## 已有的相关研究\n\n")
            for r in related:
                f.write(f"- [[{r['file'].replace('.md','')}]]\n")

    return {
        **result,
        "related_research": [r["file"] for r in related],
    }


def after_project(project_name: str, domain: str,
                  missed_technologies: Optional[list[str]] = None) -> dict:
    """Run after finishing a project. Records what was missed for future reference."""
    notes = f"项目后回顾 — {project_name}。"
    if missed_technologies:
        notes += f"\n\n## 项目中遗漏的技术\n\n"
        for t in missed_technologies:
            notes += f"- {t}\n"

    return research_and_save(
        domain=f"{domain} (项目后回顾)",
        project=project_name,
        key_findings=[
            f"项目 {project_name} 已完成",
            f"发现 {len(missed_technologies or [])} 个遗漏技术",
        ],
        related_tech=missed_technologies or [],
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Tech Research Pipeline — search → save → retrieve",
    )
    sub = p.add_subparsers(dest="command")

    # research
    p_r = sub.add_parser("research", help="Search web and save to Obsidian")
    p_r.add_argument("domain", help="Technology domain to research")
    p_r.add_argument("--project", "-p", help="Project name to link")
    p_r.add_argument("--github", "-g", help="Custom GitHub search query")
    p_r.add_argument("--findings", "-f", nargs="*", help="Key findings (space-separated)")
    p_r.add_argument("--notes", "-n", help="Additional notes")

    # retrieve
    p_ret = sub.add_parser("retrieve", help="Search local Obsidian for prior research")
    p_ret.add_argument("query", help="Search query")
    p_ret.add_argument("--max", "-m", type=int, default=10)
    p_ret.add_argument("--json", action="store_true", help="Output as JSON")

    # before
    p_b = sub.add_parser("before", help="Pre-project research")
    p_b.add_argument("project", help="Project name")
    p_b.add_argument("domain", help="Technology domain")

    # after
    p_a = sub.add_parser("after", help="Post-project retrospective")
    p_a.add_argument("project", help="Project name")
    p_a.add_argument("domain", help="Technology domain")
    p_a.add_argument("--missed", nargs="*", help="Technologies missed during the project")

    # quick
    p_q = sub.add_parser("quick", help="Quick research from piped input")
    p_q.add_argument("domain", nargs="?", help="Domain (optional if piped)")

    args = p.parse_args()

    if args.command == "research":
        result = research_and_save(
            domain=args.domain,
            project=args.project,
            github_query=args.github,
            key_findings=args.findings,
            notes=args.notes,
        )
        print(f"Saved: {result['saved']}")
        print(f"GitHub repos found: {len(result['github_results'])}")
        for r in result["github_results"][:5]:
            print(f"  {r['name']} ({r['stars']}★) — {r['description'][:60]}")

    elif args.command == "retrieve":
        results = retrieve_research(args.query, max_results=args.max)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(f"Found {len(results)} results for '{args.query}':\n")
            for r in results:
                print(f"  [{r['match_type']}] {r['file']}")
                print(f"        {r['summary'][:100]}")
                print()

    elif args.command == "before":
        result = before_project(args.project, args.domain)
        print(f"Pre-project research saved: {result['saved']}")
        print(f"Related prior research: {len(result.get('related_research',[]))} notes")

    elif args.command == "after":
        result = after_project(args.project, args.domain, args.missed)
        print(f"Post-project retrospective saved: {result['saved']}")

    elif args.command == "quick":
        domain = args.domain or sys.stdin.read().strip()
        if not domain:
            print("Error: no domain provided")
            sys.exit(1)
        result = research_and_save(domain=domain)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        p.print_help()


if __name__ == "__main__":
    main()
