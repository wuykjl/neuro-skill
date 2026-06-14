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
    import math as _math
    density = stats.get('graph_density', 0)
    n_features = stats['n_features']
    n_skills = stats['n_skills']
    min_feats = int(_math.sqrt(n_skills))

    print(f"  Graph dense: {density:.1%}", end="")
    if density > 0.60:
        print(" (spreading activation offline — too dense)")
    elif density < 0.01:
        print(" (spreading activation offline — too sparse)")
    else:
        print("")

    print(f"  CP rank:     {stats['rank']}")
    print(f"  Build time:  {stats['build_time_s']}s")

    if n_features < min_feats:
        print(f"  *** Warning: {n_features} features < sqrt({n_skills})={min_feats}. "
              f"Consider adding domain-specific extras.")

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


def cmd_serve(args):
    """启动内存常驻服务器"""
    from neuro_skill.server import cmd_serve as _serve_main
    import sys as _sys
    _sys.argv = ["neuro-skill-serve"]
    if args.socket:
        _sys.argv.extend(["--socket", args.socket])
    else:
        _sys.argv.extend(["--port", str(args.port)])
    _sys.argv.extend(["--dirs"] + args.directories)
    from neuro_skill.server import _SkillServer, _QueryHandler
    from http.server import HTTPServer

    dirs = [str(Path(d).expanduser()) for d in args.directories]
    svr = _SkillServer(dirs)
    print(f"neuro-skill server: {svr.stats['n_skills']} skills, {svr.build_ms:.0f}ms build")
    print(f"Listening on http://localhost:{args.port}")
    print(f"  curl 'http://localhost:{args.port}/query?q=Go+build+error&k=5'")
    print(f"  curl 'http://localhost:{args.port}/health'")

    httpd = HTTPServer(("127.0.0.1", args.port), _QueryHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\nShutting down.")
        httpd.shutdown()


def cmd_install(args):
    """Install MCP hook into AI coding agents."""
    import json as _json
    agent = args.agent
    mcp_cmd = args.mcp_command

    mcp_config = {
        "neuro-skill": {
            "command": mcp_cmd,
            "args": [],
            "description": "Zero-cost hybrid skill router — pre-rank skills before LLM sees them",
        }
    }

    targets = []
    if agent in ("claude", "all"):
        targets.append(Path.home() / ".claude" / "mcp.json")
    if agent in ("cursor", "all"):
        targets.append(Path.cwd() / ".cursor" / "mcp.json")
    if agent in ("codex", "all"):
        targets.append(Path.home() / ".codex" / "mcp.json")

    installed = []
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if target.exists():
            try:
                existing = _json.loads(target.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing.setdefault("mcpServers", {}).update(mcp_config)
        target.write_text(_json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        installed.append(str(target))

    print(f"neuro-skill MCP installed for {agent}:")
    for p in installed:
        print(f"  + {p}")
    print()
    print("Next steps:")
    print("  1. Restart your AI agent")
    print("  2. The agent will auto-discover the neuro-skill MCP tools")
    print("  3. Try: 'Find the best skill for checking Python SQL injection'")


def cmd_edges(args):
    """Auto-discover typed edges from skills."""
    from neuro_skill import SkillRouter
    from neuro_skill.parser import load_skills
    from neuro_skill.typed_graph import auto_discover_edges

    skills = load_skills(args.directories)
    edges = auto_discover_edges(skills)

    n_dep = sum(len(v) for v in edges["depends_on"].values())
    n_comp = sum(len(v) for v in edges["complements"].values())
    print(f"Auto-discovered: {n_dep} depends_on edges, {n_comp} complements edges")
    print()

    if edges["depends_on"]:
        print("depends_on edges:")
        for name, deps in sorted(edges["depends_on"].items()):
            print(f"  {name} → {', '.join(deps)}")
    if edges["complements"]:
        print()
        print("complements edges (top 10):")
        shown = 0
        for name, comps in sorted(edges["complements"].items(), key=lambda x: -len(x[1])):
            if shown >= 10:
                break
            print(f"  {name} ↔ {', '.join(comps[:4])}")
            shown += 1


def cmd_plan(args):
    """Plan execution order for top-k skills after routing."""
    from neuro_skill import SkillRouter
    from neuro_skill.planner import quick_plan

    router = SkillRouter()
    router.build(args.directories)

    # Route
    results = router.query(args.query, top_k=args.top_k)
    print(f"Query: {args.query}")
    print(f"Top-{args.top_k} skills:")
    for name, score in results:
        print(f"  {name}: {score:.3f}")

    # Plan
    plan_result = quick_plan(results, router)
    print()
    print(plan_result.to_prompt())


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

    # serve
    p_serve = sub.add_parser("serve", help="Start memory-resident server")
    p_serve.add_argument("--directories", "-d", nargs="+",
                         default=[
                             str(Path.home() / ".claude/skills/"),
                             str(Path.home() / ".claude/agents/"),
                             str(Path.home() / ".claude/.agents/skills/"),
                         ],
                         help="Skill directories")
    p_serve.add_argument("--port", "-p", type=int, default=8765, help="HTTP port")
    p_serve.add_argument("--socket", "-s", help="Unix socket path (faster than HTTP for local IPC)")

    # install
    p_install = sub.add_parser("install", help="Install MCP hook into AI coding agents")
    p_install.add_argument("agent", choices=["claude", "cursor", "codex", "all"],
                           help="Target agent (claude/cursor/codex/all)")
    p_install.add_argument("--mcp-command", default="neuro-skill-mcp",
                           help="MCP server command (default: neuro-skill-mcp)")

    # edges
    p_edges = sub.add_parser("edges", help="Auto-discover typed edges (depends_on + complements)")
    p_edges.add_argument("--directories", "-d", nargs="+",
                         default=[
                             str(Path.home() / ".claude/skills/"),
                             str(Path.home() / ".claude/agents/"),
                             str(Path.home() / ".claude/.agents/skills/"),
                         ],
                         help="Skill directories")

    # plan
    p_plan = sub.add_parser("plan", help="Plan execution order for top-k skills")
    p_plan.add_argument("query", help="User query")
    p_plan.add_argument("--top-k", "-k", type=int, default=5)
    p_plan.add_argument("--directories", "-d", nargs="+",
                        default=[
                            str(Path.home() / ".claude/skills/"),
                            str(Path.home() / ".claude/agents/"),
                            str(Path.home() / ".claude/.agents/skills/"),
                        ],
                        help="Skill directories")

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
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "install":
        cmd_install(args)
    elif args.command == "edges":
        cmd_edges(args)
    elif args.command == "plan":
        cmd_plan(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
