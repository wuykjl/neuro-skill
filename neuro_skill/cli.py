"""
CLI — neuro-skill 命令行接口

Usage:
    neuro-skill build ~/.claude/skills/ ~/.claude/agents/
    neuro-skill query "检查Python代码有没有SQL注入漏洞"
    neuro-skill query "React component review" --method graph_spread
    neuro-skill eval --queries queries.json
    neuro-skill stats
"""

import argparse
import json
import sys
import time
from pathlib import Path

from neuro_skill.router import SkillRouter
from neuro_skill.routers import ROUTERS


def cmd_build(args):
    router = SkillRouter()
    dirs = args.directories
    print(f"Building index from {len(dirs)} directories...")
    stats = router.build(dirs, rank=args.rank)
    print(f"  Skills:      {stats['n_skills']}")
    print(f"  Features:    {stats['n_features']} "
          f"(broad={stats['n_broad']}, precise={stats['n_precise']})")
    print(f"  Graph dense: {stats['graph_density']:.1%}")
    print(f"  CP rank:     {stats['rank']}")
    print(f"  Build time:  {stats['build_time_s']}s")

    if args.output:
        router.save(args.output)
        print(f"  Saved to:    {args.output}")


def cmd_query(args):
    # 尝试加载已保存的索引
    if args.index and Path(args.index).exists():
        router = SkillRouter.load(args.index, args.directories or [])
        print(f"Loaded index from {args.index} ({router.skill_count} skills)")
    elif args.directories:
        router = SkillRouter()
        stats = router.build(args.directories, rank=args.rank)
        print(f"Built index: {stats['n_skills']} skills in {stats['total_time_s']}s")
    else:
        print("Error: need --directories or --index", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    results = router.query(
        args.query,
        top_k=args.top_k,
        method=args.method,
        graph_steps=args.graph_steps,
        graph_decay=args.graph_decay,
        w_cos=args.w_cos,
        w_graph=args.w_graph,
        w_kw=args.w_kw,
    )
    elapsed = time.time() - t0

    print(f"\nQuery:  {args.query}")
    print(f"Method: {args.method}  ({elapsed*1000:.1f}ms)\n")
    for i, (name, score) in enumerate(results):
        bar = "█" * int(score * 30)
        print(f"  {i+1:2d}. {name:<40s} {score:.4f}  {bar}")


def cmd_eval(args):
    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"Error: {args.queries} not found", file=sys.stderr)
        sys.exit(1)

    with open(queries_path, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    router = SkillRouter()
    stats = router.build(args.directories, rank=args.rank)
    print(f"Index: {stats['n_skills']} skills, {stats['n_features']} features")

    methods = args.methods.split(",") if args.methods else ["hybrid", "jaccard", "keyword"]

    print(f"\nEvaluating {len(test_queries)} queries with {len(methods)} methods...\n")

    for method in methods:
        if method not in ROUTERS:
            continue
        total_mrr = 0.0
        total_hit = 0
        t0 = time.time()
        for q in test_queries:
            results = router.query(q["query"], top_k=5, method=method)
            top_names = [r[0] for r in results]
            for rank_i, name in enumerate(top_names):
                if name in q.get("expected", []):
                    total_mrr += 1.0 / (rank_i + 1)
                    break
            if set(top_names) & set(q.get("expected", [])):
                total_hit += 1

        n = len(test_queries)
        elapsed = time.time() - t0
        print(f"  {method:<20s}  "
              f"MRR={total_mrr/n:.4f}  "
              f"Hit@5={total_hit/n:.2%}  "
              f"{elapsed*1000/n:.0f}ms/q")


def cmd_stats(args):
    router = SkillRouter()
    if args.directories:
        stats = router.build(args.directories)
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


def cmd_diagnose(args):
    """诊断 miss cases: 显示每个查询为什么漏掉期望 skill"""
    from neuro_skill.features import extract_query_features, extract_skill_features, feature_set
    router = SkillRouter()
    router.build(args.directories)

    # Auto-generate test queries from skill names if no query file
    queries = []
    if args.queries:
        with open(args.queries, encoding='utf-8') as f:
            queries = json.load(f)
    else:
        print("No query file provided. Generating self-test queries from skill names...\n")
        # Take a sample of skills as self-tests
        import random
        sample = random.sample(router._skills, min(20, len(router._skills)))
        for s in sample:
            queries.append({
                "query": s["description"][:80],
                "expected": [s["name"]],
            })

    print(f"Diagnosing {len(queries)} queries against {router.skill_count} skills\n")

    for q in queries:
        results = router.query(q["query"], top_k=5, method="hybrid")
        top_names = [r[0] for r in results]
        expected = set(q.get("expected", []))
        found = set(top_names) & expected
        missed = expected - found

        if missed:
            print(f"  Q: {q['query'][:60]}")
            print(f"  Expected: {', '.join(expected)}")
            print(f"  Got: {', '.join(top_names[:3])}")

            for name in missed:
                skill = router.get_skill(name)
                if skill is None:
                    print(f"    [{name}]: NOT IN INDEX")
                    continue
                sf = extract_skill_features(skill)
                qf = extract_query_features(q["query"])
                s_set = feature_set(sf)
                q_set = feature_set(qf)
                overlap = q_set & s_set

                if not overlap:
                    print(f"    [{name}]: NO FEATURE OVERLAP — 查询特征={q_set}, skill特征={s_set}")
                else:
                    s_only = s_set - q_set
                    print(f"    [{name}]: overlap={overlap}, skill独有={s_only} (被稀释={len(s_only) > len(overlap)*3})")
            print()


def cmd_discover_features(args):
    """自动特征发现"""
    from neuro_skill.auto_features import auto_discover_features
    output = args.output or (str(Path.home() / "neuro-skill/auto_features.py"))
    result = auto_discover_features(args.directories, output_path=output)
    stats = result["stats"]
    print(f"\nDiscovered {stats['n_features']} features "
          f"({stats['n_broad']} broad + {stats['n_precise']} precise) "
          f"from {stats['n_skills']} skills")


def main():
    parser = argparse.ArgumentParser(
        prog="neuro-skill",
        description="Hybrid skill routing engine",
    )
    sub = parser.add_subparsers(dest="command")

    # build
    p_build = sub.add_parser("build", help="Build skill index")
    p_build.add_argument("directories", nargs="+", help="Skill/agent directories")
    p_build.add_argument("--rank", type=int, default=8, help="CP rank")
    p_build.add_argument("-o", "--output", help="Save index to .npz")

    # query
    p_query = sub.add_parser("query", help="Query skills")
    p_query.add_argument("query", help="Search query")
    p_query.add_argument("--directories", "-d", nargs="+",
                         default=["~/.claude/skills/", "~/.claude/agents/"],
                         help="Skill directories")
    p_query.add_argument("--index", "-i", help="Load pre-built index")
    p_query.add_argument("--method", "-m", default="hybrid",
                         choices=list(ROUTERS))
    p_query.add_argument("--top-k", "-k", type=int, default=10)
    p_query.add_argument("--rank", type=int, default=8)
    p_query.add_argument("--graph-steps", type=int, default=3)
    p_query.add_argument("--graph-decay", type=float, default=0.5)
    p_query.add_argument("--w-cos", type=float, default=0.40)
    p_query.add_argument("--w-graph", type=float, default=0.40)
    p_query.add_argument("--w-kw", type=float, default=0.20)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate routing quality")
    p_eval.add_argument("queries", help="JSON file with test queries")
    p_eval.add_argument("--directories", "-d", nargs="+",
                        default=["~/.claude/skills/", "~/.claude/agents/"],
                        help="Skill directories")
    p_eval.add_argument("--methods", default="hybrid,jaccard,keyword",
                        help="Comma-separated methods to test")
    p_eval.add_argument("--rank", type=int, default=8)

    # stats
    p_stats = sub.add_parser("stats", help="Show index statistics")
    p_stats.add_argument("directories", nargs="*",
                         default=["~/.claude/skills/", "~/.claude/agents/"],
                         help="Skill directories")

    # diagnose
    p_diag = sub.add_parser("diagnose", help="Diagnose miss cases")
    p_diag.add_argument("--directories", "-d", nargs="+",
                        default=["~/.claude/skills/", "~/.claude/agents/"],
                        help="Skill directories")
    p_diag.add_argument("--queries", "-q", default=None,
                        help="JSON query file (default: auto-generate from skill names)")

    # discover-features
    p_disc = sub.add_parser("discover-features",
                            help="Auto-discover features from skill set")
    p_disc.add_argument("--directories", "-d", nargs="+",
                        default=["~/.claude/skills/", "~/.claude/agents/"],
                        help="Skill directories")
    p_disc.add_argument("-o", "--output", help="Output Python file path")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "diagnose":
        cmd_diagnose(args)
    elif args.command == "discover-features":
        cmd_discover_features(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
