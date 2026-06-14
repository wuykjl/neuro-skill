"""
Skill 文件解析器

只提取 name + description + trigger 段(根治假阳性),
不碰完整 body text.
"""

import re
import yaml
from pathlib import Path
from typing import Optional

# ── Precompiled regex (compiled once at import, reused millions of times) ──
_TRIGGER_PATTERN = re.compile(
    r"(?i)(trigger|触发|Triggers?|when to use|use when|use this)"
)
_HEADING_PATTERN = re.compile(r"^##\s+")
_BULLET_PATTERN = re.compile(r"^\s*[-*]")
_WORD_PATTERN = re.compile(r"^\w")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 YAML frontmatter,返回 (meta, body)"""
    text = text.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            return meta, parts[2].strip()
    return {}, text


def _extract_triggers(body: str) -> str:
    """从 body 中提取 trigger/触发 相关段落 + 标题"""
    lines = body.split("\n")
    triggers = []
    in_trigger = False
    for line in lines:
        if _TRIGGER_PATTERN.search(line):
            in_trigger = True
            triggers.append(line)
        elif in_trigger:
            stripped = line.strip()
            if _BULLET_PATTERN.match(stripped):
                triggers.append(line)
            elif _WORD_PATTERN.match(stripped):
                in_trigger = False

    # Fallback: 如果没有显式 trigger 段,提取所有 ## 标题作为语义关键词
    if not triggers:
        for line in lines:
            if _HEADING_PATTERN.match(line.strip()):
                triggers.append(line.strip())

    return " ".join(triggers)


def parse_skill_file(filepath: Path) -> Optional[dict]:
    """
    解析单个 skill/agent .md 文件.

    返回:
        {
            "name": str,
            "description": str,
            "search_text": str,  # name + description + triggers (only!)
            "source": "skill" | "agent",
        }
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    if len(text.strip()) < 20:
        return None

    meta, body = parse_frontmatter(text)

    name = meta.get("name", filepath.stem)
    description = meta.get("description", "")

    # Trigger 段落
    triggers = _extract_triggers(body)

    # 组装 search_text: 只用 name + description + triggers
    # 刻意不包含完整的 body text,避免假阳性
    search_text = f"{name} {description} {triggers}"

    return {
        "name": name,
        "description": description,
        "search_text": search_text.lower(),
        "source": "agent" if "agents" in str(filepath) else "skill",
    }


def load_skills(directories: list[str]) -> list[dict]:
    """
    从目录列表中加载所有 skill/agent 文件.

    每个目录下一次遍历——合并 *.md + */SKILL.md + symlink→SKILL.md 三种模式.
    """
    skills = []
    seen = set()
    for d in directories:
        dp = Path(d).expanduser().resolve()
        if not dp.exists():
            continue

        # Single iterdir: covers flat .md, subdir SKILL.md, and symlink→SKILL.md
        md_files = set()
        for item in dp.iterdir():
            if item.is_symlink() and item.is_dir():
                resolved = item.resolve()
                smd = resolved / "SKILL.md"
                if smd.is_file():
                    md_files.add(smd)
            elif item.is_file() and item.suffix == ".md":
                md_files.add(item)
            elif item.is_dir():
                for sfx in ["SKILL.md", "skill.md"]:
                    smd = item / sfx
                    if smd.is_file():
                        md_files.add(smd)

        for f in sorted(md_files):
            info = parse_skill_file(f)
            if info and info["name"] not in seen:
                seen.add(info["name"])
                skills.append(info)
    return skills
