# neuro-skill 开发流程全记录

> 项目: `wuykjl/neuro-skill` | 102 tests green | Python 3.10+
> 时间跨度: 2026-06-13 ~ 2026-06-15
> 审核人: @tonydwb (Hermes), 独立社区测试者 (匿名)

---

## 一、项目启动 (6/13-6/14)

### 1.1 核心路由引擎 (v0.2 → v0.7.2)

**目标**: 用纯 Python 实现零成本技能路由器，不依赖外部 API。

| 版本 | 内容 | commit |
|------|------|--------|
| v0.2 | 零成本混合路由引擎 — graph spreading + cosine + keyword | `a4b501a` |
| v0.4 | MCP server + file watcher + CodeGraph bridge | `d4026a1` |
| v0.5 | 实现 RRF (Reciprocal Rank Fusion) 融合 BM25 + cosine + graph | `dde2a93` |
| v0.5.1 | k-NN graph 自适应密度 + 零特征提示 | `dde2a93` |
| v0.6 | IG 加权特征 + 自诊断 + 两轮 LLM 生成 | `e19cc68` |
| v0.6.1 | plan() 管道 (route → deps → sort) | `cbd392c` |
| v0.7 | CF 个性化 (Thompson Sampling 替换 ALS) | `307c36f` |
| v0.7.2 | 中文关键词覆盖 0%→可行, LLM rerank 5th signal | `eecf5df` |

**关键技术突破**:
- **RRF 5-signal fusion**: BM25 + Cosine + Graph + CF + LLM rerank
- **Thompson Sampling**: 替代 ALS，冷启动友好
- **中英混合分词器**: CJK bigram + `\w` 修正，解决中文 query 只匹配 1 个字的问题
- **O(1) 反向索引**: BM25 查找从 O(N) 降到 O(1)

### 1.2 性能与安全加固

| 改动 | commit |
|------|--------|
| 修复 7 处静默异常吞没 | `94b967e` |
| np.load allow_pickle=False 防止任意代码执行 | `8f6ad3d` |
| 除零 guard (空技能集) | `cb6d37f` |
| BM25 LRU cache + tokenizer 缓存 | `0764ac7` |
| 正则预编译 + 单次遍历 | `9ed27e7` |
| 测试 39→70→102→141 | 多个 commits |

---

## 二、AI 自动调用死结 (6/14-6/15 核心攻坚)

### 2.1 问题发现

**核心矛盾**: neuro-skill 路由精度 95%，但 AI Agent **从来不调 router**。

```
用户: "检索cs文件安全检查"
  ↓
AI 系统提示里有 153 个 skills + neuroskill_query MCP tool
  ↓
AI 决定: "我用 Glob 搜文件 + Read 审查就行" → router 从未被调用
  ↓
56% 跳过率 (Vercel 2026年1月评测证实)
```

### 2.2 9 轮方案迭代 — 每条路都撞天花板

#### 第 1 轮: SKILL.md 自发现
- 方案: 将 neuro-skill 写成普通 skill，放入 skills 目录
- 结果: AI 在 56% 情况下跳过，不读正文

#### 第 2 轮: MCP tool 注入
- 方案: neuro_skill MCP server，4 个工具注册到 AI
- 结果: 工具输出被视为"建议"而非"指令"，AI 可覆盖

#### 第 3 轮: PTY wrapper
- 方案: `neuro-skill wrap claude` — PTY 包装器在 stdin 层拦截并注入
- 结果: 跨 Agent 通用但不方便。`neuro-skill bootstrap` 解决 alias

#### 第 4 轮: hide 隐藏
- 方案: 所有技能移动到 `.skills-store/`，只留 neuro-skill
- 结果: 堵了 Skill 通道但 Glob/Bash 内置工具仍可绕过

#### 第 5 轮: inject CLAUDE.md
- 方案: `neuro-skill inject` — 压缩 153 技能索引注入 CLAUDE.md
- 结果: CLI.md 100% 到达模型但随机遵守 (30-60%)

#### 第 6 轮: PreToolUse hook
- 方案: hook 拦截 Bash/Glob/Skill 调用，exit 2 阻止
- 结果: 能拦但模型可以重试同一工具
- 发现: Anthropic 官方 #19308 承认同类 bug，标记 duplicate 关闭

#### 第 7 轮: hostify
- 方案: 寄生注入路由指令到每个 SKILL.md 正文
- 结果: 失效——正文在 AI 选中技能后才读取，决策阶段只读 frontmatter 的 description

#### 第 8 轮: description 前缀
- 方案: `[NS|先查询]` 前缀注入 description 字段，影响系统提示
- 结果: 前缀不被解析为指令

#### 第 9 轮: Hermes pre_llm_call hook

**架构差异决定成败**:

```
Claude Code:  LLM 先看 332 个技能 → 决定用 Glob → router 从未参与
Hermes:      用户 query 到达 → pre_llm_call hook → router.query() → LLM 只看 3 个
              ^^^^^^^^^^^^^^^^ 路由在 LLM 看到 query 之前完成
```

- 社区测试者独立验证: Hermes v0.16.0 + DeepSeek V4 Flash + 332 skills, 路由 6-10ms
- 这个验证证明了 **神经技能路由不是代码问题——是 Agent 架构层面的路线选择问题**

---

## 三、路由准确性提升 (6/15 会话内持续改进)

### 3.1 规则白名单引擎

**动机**: AI 不调用 router → 用户需要 100% 控制的情况

```bash
neuro-skill rule add "检索.*cs" "csharp-reviewer"
neuro-skill rule add "cs.*安全" "skill-vetter"
```

- 48 条规则覆盖 8 个领域
- 匹配到时 score=1.0，不可覆盖
- 中文 → firecrawl 优先，英文 → exa 优先，WebSearch 兜底

### 3.2 模糊纠正 (correct)

用户不记得技能名 → 自然语言描述 → 自动解析 + Error Book 学习:

```bash
neuro-skill correct "检索cs安全检查" "C# 代码审查工具"
# → 解析为 csharp-reviewer → 记入 Error Book
```

### 3.3 加权 RRF + 稀疏覆盖门

- BM25 权重自适应: coverage <10% 时降低到 0.2
- 质量门: BM25 <5% + cosine gap <0.5% 时返回空 → 让 AI 用自己工具

### 3.4 中英混合分词器

"检索cs文件安全检查" 的分词:
- 之前: token = `检索`
- 之后: token = `检索`, `检索cs`, `cs`, `文件`, `安全检查`, `安全`, `检查`

`cs文件` 被识别为 csharp 特征 → 触发 language detection

### 3.5 return_body

`query("...", return_body=True)` → 返回 `(name, score, body)` 三元组。
AI 直接从路由结果获取完整技能指令，不需要读隐藏文件。

---

## 四、Hermes 插件集成 (6/15 突破)

### 4.1 向 NousResearch 提交 PR

**提交内容**:
1. `plugin.yaml` — name, hooks 声明 (on_session_start + pre_llm_call)
2. `__init__.py` — 插件入口，pre_llm_call 实现
3. `README.md` — 架构文档 + 已验证配置

**审核者**: `@tonydwb` (Hermes contributor)
**审核状态**: Comment（待深度审查）
**审核要求**: 配置文档化、边界情况处理、测试覆盖

### 4.2 零依赖重构

审核后补推: 将 `from neuro_skill import SkillRouter` 替换为 **内嵌 250 行 BM25 + 规则引擎**。

**理由**: 如果 neuro-skill PyPI 变动或不可用，Hermes 插件完全不受影响。"Vendoring lite" 模式 — 在所有维持长期外部依赖的开源项目中是标准做法。

### 4.3 测试补充

审核者要求测试 → 6 个独立测试 (test_router.py):
- BM25 关键词路由 (精确匹配 + 无匹配回退)
- 规则覆盖 (正则规则优先于 BM25)
- 边界场景 (空索引、单技能、空 query)
- pre_llm_call 上下文格式
- 零外部依赖验证 (no import neuro_skill)

### 4.4 插件架构

```
Hermes session start
  ↓
on_session_start: 扫描全部 442 个 SKILL.md → 建立 BM25 索引 (500ms, 一次性)
  ↓
用户 "检索cs安全检查"
  ↓
pre_llm_call: 规则检查 → 匹配 'cs.*安全' → 直接返回 "csharp-reviewer (1.0)"
  未匹配则 BM25 查询 → top-3 注入
  ↓
LLM 收到: [Top 3 skills for this query] + 原始消息
  只看到 3 个技能名称 — 不是 442 个
```

---

## 五、Bug 暴露与修复

### 5.1 质量门过触发 (adbac15 → adac25d)

- **暴露方式**: 第三方测试者用 21 题基准跑全版本
- **表现**: 6/21 有效短 query 被误判为噪声 → 返回空
- **根因**: `bm25_cov < 0.20` 对 2-3 词短 query 过于激进 (天然 <10%)
- **修复**: 阈值收紧到 `bm25_cov < 0.05` + `cos_gap < 0.005`
- **测试缺口**: 边界测试 (`xyzz`) 通过，但 21 题回归未跑

### 5.2 边缘测试污染

mock 技能有相同 search_text → cosine gap=0 → 触发质量门
修复: 每个技能给定独有关键词

### 5.3 39 项回归基准

上述 bug 暴露后立即补充:
- 21 英文核心 + 9 中文 + 9 边界场景
- pure noise → 必须空 (`asdfqwerzxcv`)
- 有效短 query → 绝对不能空 (有 `firecrawl search`, `spec driven dev`)
- 每次 CI 自动跑

---

## 六、项目架构全景

### 6.1 模块组织

```
neuro_skill/
├── router.py          # 核心: SkillRouter class (query, learn, plan, feedback)
├── routers.py         # 7 路由算法 (BM25, cosine, graph, jaccard, tfidf, hybrid, LLM)
├── index.py           # SkillIndex: F 矩阵 + G 图 + CP 张量分解
├── features.py        # 特征提取: 分词器, _match, extract_skill_features
├── base_features.py   # 17 broad + 32 precise 特征定义
├── parser.py          # YAML frontmatter 解析
├── feedback.py        # Error Book — 持久化反馈学习
├── personalize.py     # Thompson Sampling CF 个性化
├── mcp_server.py      # MCP JSON-RPC over stdio (4 tools)
├── autostart.py       # 懒加载单例 (自动构建)
├── planner.py         # I/O topological sort 管道
├── cli.py             # 命令行接口 (17 个 subcommand)
│
plugins/hermes/        # Hermes pre_llm_call 插件
├── __init__.py        # 独立 BM25 + 规则引擎（零外部依赖）
├── plugin.yaml        # 插件 manifest
├── test_router.py     # 6 个独立插件测试
│
extras/
└── extras_ecc.py      # ECC 领域特征 (65 个特征)
```

### 6.2 四层路由流水线

```
query → 1. 规则白名单 (48条/rules.json)    → 命中: score=1.0
        2. RRF 加权路由 (BM25+cosine+graph) → 强信号: 返回排名
            └ 质量门 (<5%+gap<0.005)        → 空结果
        3. 模糊纠正 (neuro-skill correct)    → Error Book 学习
        4. 全部失败 → AI 用内置工具
```

### 6.3 集成矩阵

| Agent | 机制 | 可靠性 |
|-------|------|--------|
| **Hermes** | pre_llm_call hook | 100% host-level (流程内) |
| Claude Code | inject CLI.md | 到达 100%, 遵守 30-60% |
| Cursor / Codex / Gemini | MCP tools | 工具输出可被覆盖 |
| **Python API** | query(), return_body=True, plan() | 直接调用 |
| **CLI** | neuro-skill query/add/rule/correct | 直接调用 |

---

## 七、开发经验教训

### 7.1 测试教训

1. **边界修完必须跑回归**: 质量门修复时只测了 `xyzz` (噪声)，没跑 21 题基准。
2. **外部基准不可替代**: 第三方用例覆盖了作者/CI 都漏掉的短 query 场景。
3. **mock 数据不能太简单**: 5 个技能相同文本 → cosine gap=0 → 触发真实 bug。

### 7.2 架构教训

1. **Vercel 68% 的评测不是寓言——是现实**: 外部研究验证了技能系统的固有局限性。
2. **Host-level 路由 > prompt-level 路由**: Hermes 的 `pre_llm_call` hook 先于 LLM 决策，
   这是 MCP / inject / hide / hook 都无法达到的确定性。
3. **零依赖是 PR 被接的底线**: 外部依赖 = 维护承诺 = 项目方的风险。
   任何集成 PR 必须自带核心逻辑。

### 7.3 部署教训

1. **发布节奏过快暴露了回归**: 质量门实现后立刻推送，跳过了回归测试。
   21 题基准应该作为提交前的必须步骤。
2. **第三方验证的价值**: 独立测试者跑多版本时发现了作者侧漏掉的 bug。
3. **规则引擎是确定性增量的入口**: 规则引擎给用户提供了 100% 控制的回退方案。

---

## 八、审核状态与展望

| 项目 | 状态 |
|------|------|
| Hermes PR #46484 | Open — awaiting @tonydwb 再次审核 |
| GitHub Stars | 8 |
| 测试 | 141 项全部通过 (102 unit + 39 基准) |
| 生产级判定 | **已就绪** — Hermes 集成已验证, 其他 Agent 通过 CLI/API 支持 |

### 下一步

1. Hermes PR merge → Hermes 用户自动获得技能路由
2. PyPI 发布 → 用户 `pip install neuro-skill` 可用
3. 更多 Agent 集成 (Codex/Cursor 的 host-level hook)
4. 领域扩展 (医疗/金融/游戏 extras 配置文件)

---

> 最后更新: 2026-06-15
> 作者: wuykjl (https://github.com/wuykjl)
> 许可证: MIT
