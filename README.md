# neuro-skill

**Zero-cost hybrid skill router for AI agents. Pure local, 40ms/query, no API calls.**

When your AI agent has 50+ skills and a user asks "check my Python code for SQL injection", keyword matching finds the right skill ~18% of the time. Embedding-based routing (text-embedding-3) gets ~70% but costs money and adds latency. **neuro-skill gets ~55% — same ballpark, zero cost.**

It's not magic. It's graph spreading activation + feature similarity + keyword fusion, running entirely on your machine.

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

Three signals fused: graph neighbors of matched skills, feature vector cosine similarity, and precise keyword overlap.

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

The base model covers 15 general programming domains + 23 languages/frameworks/actions out of the box. For domain-specific skills (lark, firecrawl, medical, finance, etc.), add your own keywords:

```python
router = SkillRouter()
# Add your domain-specific features
import neuro_skill.features as f
f.PRECISE["medical"] = ["fda", "clinical", "drug", "药品", "临床"]
router.build(["./my-skills/"])
```

Or use the [extras template](extras_template.py).

## Performance

Benchmarked on 270 skills, 40 real-world queries (Chinese + English):

| Method | MRR | Hit@5 | Cost per query |
|--------|-----|-------|---------------|
| text-embedding-3 (API) | ~0.70 | ~80% | $0.00002 |
| **neuro-skill** | **0.55** | **72%** | **$0** |
| TF-IDF | 0.20 | 35% | $0 |
| Keyword matching | 0.10 | 18% | $0 |

**neuro-skill is 5.5× better than keyword matching at the same cost, and 80% as good as paid embedding at zero cost.**

## What It's Good At

- **150-300 skills, zero API budget** — the sweet spot
- **Multi-skill queries** — "send a message AND create a calendar event" activates both `lark-im` and `lark-calendar`
- **Chinese queries** — hand-crafted bilingual features
- **Offline / air-gapped** — no network needed

## What It's Not Good At

- **< 20 skills** — keyword matching is fine, you don't need this
- **> 500 skills** — graph becomes too dense, embedding-based routing works better
- **Completely novel domains** — you need to add your own keywords (see [extras](extras_template.py))
- **Semantic nuance** — "make it faster" vs "optimize database queries" — embedding-based routing handles this better

## CLI

```bash
neuro-skill build ./skills/ ./agents/ -o ./index/
neuro-skill query "Rust code review" -i ./index/ -k 5
neuro-skill eval queries.json -d ./skills/ ./agents/
```

## Add Your Own Features

See [extras_template.py](extras_template.py) — copy it, add your keywords, done.

## Requirements

- Python >= 3.10
- numpy, pyyaml, scikit-learn
- Zero API, zero GPU, fully offline

## License

MIT — see [LICENSE](LICENSE).
