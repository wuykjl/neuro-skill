# neuro-skill

**Zero-cost hybrid skill router for AI agents. Not an embedding replacement — a prompt-bloat terminator.**

When your agent has 150+ skills, the LLM scans every skill description on every query — burning tokens and making black-box decisions. neuro-skill replaces that: 6ms local pre-ranking, top-3 skills injected into the prompt, zero API cost.

| Before (LLM blind scan) | After (neuro-skill pre-rank) |
|---|---|
| 332 skill descriptions in prompt | Top-3 skills only |
| LLM decides which skill — black box | Transparent scores, debuggable features |
| New skill → restart session | Cron rebuild, seconds |
| Token cost scales with skill count | Token cost flat (top-3) |

## How It Works

```
"Go build error troubleshooting"
               │
     ┌─────────┼─────────┬─────────┬──────────┐
     ▼         ▼         ▼         ▼          ▼
   BM25     Cosine    Graph      CF        LLM
   Keyword  Similarity Spread   Personalize Rerank
     │         │         │         │          │
     └─────────┼─────────┴─────────┼──────────┘
               ▼
        RRF Fusion (Reciprocal Rank Fusion)
               ▼
     [go-build-resolver: 1.000,
      rust-build-resolver: 0.984,
      go-reviewer: 0.956]
```

Up to 5 signals fused via Reciprocal Rank Fusion (RRF): BM25 keyword, feature cosine similarity, graph spreading activation, collaborative filtering personalization, and optional LLM semantic rerank. No embedding API required. No GPU. Fully offline. Haiku rerank available as opt-in 5th signal (~$0.0003/call).

## Quick Start

```bash
pip install neuro-skill
```

```python
from neuro_skill import SkillRouter

router = SkillRouter()
router.build(["./my-skills/", "./my-agents/"])

results = router.query("check Python code for SQL injection", top_k=5)
for name, score in results:
    print(f"  {name}: {score:.3f}")
```

**Your skills need YAML frontmatter** in `.md` files:

```markdown
---
name: my-skill
description: What this skill does and when to use it
---
```

The base model covers 17 programming domains + 32 languages/frameworks/actions out of the box. For domain-specific skills, add keywords via extras files — two real-world examples in [extras/](extras/).

**Aim for features ≥ √(skills)** — 150 skills need ~80 features, 300 need ~120+.

## Performance

### Author benchmark (270 skills, 40 queries, Chinese + English)

| Method | MRR | Hit@5 | Cost per query |
|--------|-----|-------|---------------|
| text-embedding-3 (API) | ~0.70 | ~80% | $0.00002 |
| **neuro-skill hybrid** | **0.547** | **72%** | **$0** |
| TF-IDF | 0.20 | 35% | $0 |
| Keyword matching | 0.097 | 18% | $0 |

**5.5× better than keyword at the same cost. Optional LLM rerank (Haiku, ~$0.0003/call) closes the semantic gap with paid embedding.**

### Third-party validation (neuro-skill v0.7.1, 39 queries × 3 sets, independent tester)

| Skills | Features | Hit@1 (hybrid) | Hit@1 (keyword) | Graph density |
|--------|----------|---------------|-----------------|---------------|
| 54 (homogeneous) | 25 | 42% | **84%** | 83% — too dense |
| 314 ECC (base only) | 37 | 21% | 26% | 0.5% — too sparse |
| 314 ECC + 94 features | 94 | **65%** | 45% | 0.35% — works |
| 332 multi-domain + 112 features | 112 | **61%** | 76% | 0.37% — works |

| Method | Core domains (21q) | Hard domains (10q) | Chinese (8q) | Total (39q) |
|--------|-------------------|-------------------|--------------|-------------|
| hybrid | 95% / 100% | 40% / 60% | 0%→72% / 12%→87% | 61% / 72% |
| + CF personalize | 0.98 boost | same | same | — |
| + LLM rerank (Haiku) | 5th RRF signal | semantic gap closed | — | +~$0.0003/call |

**Threshold: hybrid outperforms keyword when skills ≥ 150 AND features ≥ 80.**

## What It Can Do

neuro-skill is now a full pipeline, not just a router:

| Capability | How | Command / API |
|-----------|-----|--------------|
| **Route:** find the right skill | 5-signal RRF (BM25 + cosine + graph + CF + optional LLM) | `router.query()` |
| **Personalize:** learn what you prefer | ALS collaborative filtering from implicit feedback | `router.observe()` / `router.train_personalize()` |
| **Plan:** orchestrate multiple skills | Topological sort over typed dep edges (depends_on + complements) | `router.plan()` |
| **Predict:** warm up from code context | AST import detection + CodeGraph symbol extraction | `neuroskill_predict(file)` via MCP |
| **Learn:** self-correct over time | Error Book persistent corrections with decay | `router.learn()` |

## Real-World Integration (Hermes Case Study)

A third-party tester integrated neuro-skill into the Hermes agent framework (332 skills). Benefits measured:

**1. Prompt bloat elimination.** Before: all 332 skill descriptions in every prompt. After: top-3 pre-ranked skills only. Input tokens slashed.

**2. Debuggable ranking.** LLM skill selection is a black box — you never know why a skill was (or wasn't) picked. neuro-skill exposes scores, feature overlap, and the `diagnose` CLI command for miss-case investigation.

**3. Offline, second-level updates.** Install new skills → cronjob `--rebuild` (0.5s) → next query uses new features. No session restart needed.

**4. Cost independent of skill count.** Token cost for skill routing stays flat (top-3 only) regardless of whether you have 50 or 500 skills.

## What It's Good At

- **150-300 skills, zero API budget** — the sweet spot
- **Multi-skill queries** — "send a message AND create a calendar event" activates both
- **Chinese queries** — 50 bilingual features covering 常见中文触发词 (扫描安全漏洞、发消息、容器部署)
- **Offline / air-gapped** — no network needed
- **Transparent, debuggable routing** — not a black box
- **Personalization** — learns from which skills you actually pick
- **Orchestration** — plan("security + deploy") → ordered multi-skill pipeline

## What It's Not Good At

- **< 20 skills** — keyword matching is fine, or often better
- **Zero-feature new domains** — you need to add keywords. Two complete extras examples in [extras/](extras/)
- **Semantic nuance (without LLM)** — "make it faster" needs `enable_llm=True` (5th RRF signal)
- **Replacing embeddings** — it doesn't. It replaces prompt bloat and black-box LLM selection

## CLI

```bash
neuro-skill build ./skills/ ./agents/ -o ./index/
neuro-skill query "Rust code review" -i ./index/ -k 5
neuro-skill eval queries.json -d ./skills/ ./agents/
neuro-skill diagnose -d ./skills/ ./agents/  # find miss cases
```

## Supported Editors & Agents

| Tool | Integration | Method |
|------|------------|--------|
| **Claude Code** | `neuro-skill install claude` | MCP (4 tools: query, compare, status, predict) |
| **Cursor** | `neuro-skill install cursor` | MCP |
| **Codex CLI** | `neuro-skill install codex` | MCP |
| **Hermes** | Third-party validated (332 skills) | Adapter script |
| **Windsurf** | Edit `.windsurf/mcp.json` | MCP |
| **Continue** | Edit `~/.continue/mcp.json` | MCP |
| **GitHub Copilot** | `curl localhost:8765/query` | HTTP |
| **JetBrains AI** | `curl localhost:8765/query` | HTTP |
| **Any Python script** | `from neuro_skill import query` | Python API |
| **Any shell / CI** | `neuro-skill query "..."` | CLI |
| **OpenCode / Kiro / Aider** | Edit their MCP config | MCP |

## Feature System

Three-tier design:

| Tier | Source | Quality | Effort |
|------|--------|---------|--------|
| Base (50 features) | Ships with neuro-skill | Hit@1 95% (core domains) | Zero |
| Extras (your domains) | You write, copy from [extras/](extras/) | MRR ~0.45-0.55 | 10-30 min |
| LLM-generated | `llm_extract_features(dirs)` | MRR ~0.48 (87% of hand-crafted) | $0.03 one-time |

Two complete extras examples:
- [extras_template.py](extras_template.py) — lark/firecrawl/VPN (author's own setup)
- [extras/extras_ecc.py](extras/extras_ecc.py) — 16 broad + 45 precise for 314 ECC skills (third-party contributed, Hit@1 65%)

## Requirements

- Python ≥ 3.10
- numpy, pyyaml, scikit-learn
- Zero API, zero GPU, fully offline
- 54 tests, all green

## Ecosystem

Projects that neuro-skill builds on, was tested with, or drew design inspiration from:

| Project | Role |
|---------|------|
| [ECC](https://github.com/wuykjl/ecc) (Enterprise Coding Conventions) | 117-rule coding standard framework. Used as a benchmark skill set during neuro-skill development. An `extras_ecc.py` example is included in [extras/](extras/). |
| [MetaSkill](https://github.com/Dicklesworthstone/meta_skill) | Local-first skill management platform. Thompson Sampling personalization inspired by its bandit-optimized suggestion engine. |
| [CodeGraph](https://github.com/colbymchenry/codegraph) | Pre-indexed code knowledge graph. MCP server architecture and `neuroskill_predict` context analysis inspired by its design. |
| [agent-skill-finder](https://github.com/theshubh007/agent-skill-finder) | Independent validation of the BM25+Jaccard+Graph+zeroLLM architecture. Converged from a different starting point. |

## License

MIT — see [LICENSE](LICENSE).
