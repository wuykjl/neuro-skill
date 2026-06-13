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
     ┌─────────┼─────────┐
     ▼         ▼         ▼
   Graph     Cosine    Keyword
   Spread    Similarity Match
   (40%)     (40%)      (20%)
     │         │         │
     └─────────┼─────────┘
               ▼
     [go-build-resolver: 1.00,
      go-reviewer: 0.89,
      build-error-resolver: 0.72]
```

Three signals fused: graph neighbors of matched skills, feature vector cosine similarity, and precise keyword overlap. No embedding API. No GPU. No network.

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

The base model covers 17 programming domains + 29 languages/frameworks/actions out of the box. For domain-specific skills, add keywords via extras files — two real-world examples in [extras/](extras/).

**Aim for features ≥ √(skills)** — 150 skills need ~80 features, 300 need ~120+.

## Performance

### Author benchmark (270 skills, 40 queries, Chinese + English)

| Method | MRR | Hit@5 | Cost per query |
|--------|-----|-------|---------------|
| text-embedding-3 (API) | ~0.70 | ~80% | $0.00002 |
| **neuro-skill hybrid** | **0.547** | **72%** | **$0** |
| TF-IDF | 0.20 | 35% | $0 |
| Keyword matching | 0.097 | 18% | $0 |

**5.5× better than keyword at the same cost. 80% as good as paid embedding at zero cost.**

### Third-party validation (independent tester, Hermes + ECC + addyosmani skills)

| Skills | Features | Hit@1 (hybrid) | Hit@1 (keyword) | Graph density |
|--------|----------|---------------|-----------------|---------------|
| 54 (homogeneous) | 25 | 42% | **84%** | 83% — too dense |
| 314 ECC (base only) | 37 | 21% | 26% | 0.5% — too sparse |
| 314 ECC + 94 features | 94 | **65%** | 45% | 0.35% — works |
| 332 multi-domain + 112 features | 112 | **61%** | 76% | 0.37% — works |

**Threshold: hybrid outperforms keyword when skills ≥ 150 AND features ≥ 80.**

## Real-World Integration (Hermes Case Study)

A third-party tester integrated neuro-skill into the Hermes agent framework (332 skills). Benefits measured:

**1. Prompt bloat elimination.** Before: all 332 skill descriptions in every prompt. After: top-3 pre-ranked skills only. Input tokens slashed.

**2. Debuggable ranking.** LLM skill selection is a black box — you never know why a skill was (or wasn't) picked. neuro-skill exposes scores, feature overlap, and the `diagnose` CLI command for miss-case investigation.

**3. Offline, second-level updates.** Install new skills → cronjob `--rebuild` (0.5s) → next query uses new features. No session restart needed.

**4. Cost independent of skill count.** Token cost for skill routing stays flat (top-3 only) regardless of whether you have 50 or 500 skills.

## What It's Good At

- **150-300 skills, zero API budget** — the sweet spot
- **Multi-skill queries** — "send a message AND create a calendar event" activates both
- **Chinese queries** — hand-crafted bilingual features in base model
- **Offline / air-gapped** — no network needed
- **Transparent, debuggable routing** — not a black box

## What It's Not Good At

- **< 50 skills** — keyword matching is fine, or often better
- **Semantic nuance** — "make it faster" vs "optimize database queries" needs embeddings
- **Zero-feature new domains** — you need to add keywords. Two complete extras examples in [extras/](extras/)
- **Replacing embeddings** — it doesn't. It replaces prompt bloat and black-box LLM selection

## CLI

```bash
neuro-skill build ./skills/ ./agents/ -o ./index/
neuro-skill query "Rust code review" -i ./index/ -k 5
neuro-skill eval queries.json -d ./skills/ ./agents/
neuro-skill diagnose -d ./skills/ ./agents/  # find miss cases
```

## Feature System

Three-tier design:

| Tier | Source | Quality | Effort |
|------|--------|---------|--------|
| Base (46 features) | Ships with neuro-skill | MRR ~0.20 (generic domains) | Zero |
| Extras (your domains) | You write, copy from [extras/](extras/) | MRR ~0.45-0.55 | 10-30 min |
| LLM-generated | `llm_extract_features(dirs)` | MRR ~0.48 (87% of hand-crafted) | $0.03 one-time |

Two complete extras examples:
- [extras_template.py](extras_template.py) — lark/firecrawl/VPN (author's own setup)
- [extras/extras_ecc.py](extras/extras_ecc.py) — 16 broad + 45 precise for 314 ECC skills (third-party contributed, Hit@1 65%)

## Requirements

- Python ≥ 3.10
- numpy, pyyaml, scikit-learn
- Zero API, zero GPU, fully offline
- 44 tests, all green

## License

MIT — see [LICENSE](LICENSE).
