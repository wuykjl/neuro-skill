"""
Generic base features — universal software development domains.

These 17 broad + 29 precise features cover the most common
programming languages, frameworks, and actions. For domain-specific
skills (lark, firecrawl, medical, finance, etc.), create an
extras file and pass it to build_router():

    router = build_router(["./skills/"], precise_features=my_extras)
"""

# 17 general domains
BROAD = {
    "security": [
        "security", "vulnerability", "auth", "injection", "xss", "csrf",
        "owasp", "penetration", "exploit", "cve", "secrets", "credential",
        "安全", "漏洞", "认证", "注入", "攻击", "加密", "密钥",
    ],
    "frontend": [
        "react", "vue", "angular", "next", "css", "html", "ui", "component",
        "tailwind", "frontend", "browser", "spa",
        "前端", "界面", "组件", "样式", "页面",
    ],
    "backend": [
        "api", "server", "endpoint", "backend", "rest", "graphql", "grpc",
        "microservice", "middleware",
        "路由", "服务端", "服务", "后端",
    ],
    "database": [
        "sql", "postgresql", "mysql", "mongodb", "redis", "query",
        "orm", "migration", "index", "transaction", "storage",
        "数据库", "查询", "索引", "事务", "存储",
    ],
    "devops": [
        "ci/cd", "docker", "kubernetes", "deploy", "build", "pipeline",
        "container", "jenkins", "terraform",
        "部署", "构建", "容器",
    ],
    "network": [
        "dns", "routing", "firewall", "tcp", "http", "network",
        "网络", "路由", "防火墙",
    ],
    "mobile": [
        "ios", "android", "flutter", "swift", "kotlin", "mobile",
        "移动", "手机", "app",
    ],
    "desktop": [
        "electron", "tkinter", "desktop", "gui", "windows",
        "桌面", "窗口",
        # File search tools — use specific trademarks, not generic words
        "everything", "es.exe", "全盘",
    ],
    "document": [
        "document", "markdown", "readme", "docx", "pptx",
        "文件处理", "文档", "模板", "import",
        # Chinese file operations
        "检索", "搜索文件", "查找文件", "locate", "find files",
    ],
    "data": [
        "data", "analytics", "pipeline", "etl", "数据", "分析", "处理",
    ],
    "ml": [
        "machine learning", "training", "model", "inference", "pytorch",
        "tensorflow", "gpu", "cuda",
        "机器学习", "模型", "训练", "推理", "深度学习",
    ],
    "document": [
        "document", "docx", "pdf", "word", "markdown", "slides",
        "presentation", "spreadsheet", "xlsx",
        "文档", "报告", "幻灯片",
    ],
    "design": [
        "design", "prototype", "mockup", "animation", "设计", "原型",
        "动画", "demo", "高保真", "界面设计",
    ],
    "vcs": [
        "git", "commit", "branch", "merge", "pull request", "pr",
        "git workflow", "conventional commit",
        "提交", "分支", "合并", "版本控制",
    ],
    "code_quality": [
        "review", "refactor", "code quality", "lint", "code smell",
        "clean code", "simplify", "dead code",
        "审查", "重构", "代码质量", "清理",
    ],
    "testing": [
        "test", "tdd", "coverage", "e2e", "单元测试", "测试", "覆盖率",
         "playwright", "pytest",
    ],
    "coding_standard": [
        "coding style", "编码风格", "coding standard", "编码规范",
        "代码规范", "style guide", "规范", "标准", "规则", "rule",
        "ECC", "convention", "约定", "guideline", "指南",
    ],
    "workflow_process": [
        "workflow", "工作流", "development process", "开发流程",
        "git workflow", "git工作流", "commit format", "提交格式",
        "pull request", "PR流程", "hooks", "钩子", "debugging",
        "调试", "排查",
    ],
}

# 32 languages / frameworks / actions
PRECISE = {
    "python": ["python", "django", "fastapi", "pytorch", "tkinter", "pep", "pip"],
    "javascript_ts": ["javascript", "typescript", "jsx", "tsx", "node.js",
                       "npm", "yarn", "pnpm", "bun", "next.js"],
    "react_specific": ["react", "jsx", "react native", "hook", "react组件", "前端react"],
    "go": ["golang", r"\bgo\b", "go build", "go mod", "goroutine", "go语言", "go项目"],
    "rust": ["rust", "cargo", "rustc", "tokio", "serde", "rust语言"],
    "java": ["java ", "spring", "maven", "gradle", "jvm", "hibernate", "java语言", "java项目"],
    "kotlin": ["kotlin", "android", "kmp", "kotlin语言"],
    "swift": ["swift", "xcode", "ios", "swift语言"],
    "dart_flutter": ["dart", "flutter", "flutter项目"],
    "php": ["php", "laravel", "symfony", "php语言"],
    "csharp": ["c#", "csharp", ".net", "dotnet", "c#语言"],
    "cpp": ["c++", "cpp", "cmake", "c++20", "clang", "c++语言", "c++项目"],
    "harmonyos": ["harmonyos", "arkts", "arkui", "鸿蒙"],
    "shell": ["bash", "shell", "powershell", "cmd", "bash脚本"],
    "build_fix": ["build error", "compilation", "构建错误", "编译失败",
                  "构建失败", "build failed", "compile error", "构建报错",
                  "编译报错", "build报错", "cmake失败", "编译不通过",
                  "编译问题", "构建问题", "编译出错"],
    "security_scan": ["漏洞扫描", "安全审计", "security scan", "渗透",
                       "penetration", "vulnerability scan", "sql注入",
                       "xss攻击", "csrf漏洞", "注入漏洞", "扫描安全",
                       "安全扫描", "扫描漏洞", "安全漏洞", "检查安全"],
    "performance": ["performance", "optimize", "bottleneck", "性能", "优化",
                     "太慢", "慢", "slow", "latency", "查询优化", "数据库优化",
                     "加速", "提速", "性能优化", "跑得快", "更快"],
    "architect": ["architecture", "architect", "system design",
                    "架构", "微服务", "microservice", "design pattern",
                    "架构设计", "系统设计"],
    "planning": ["plan", "workflow", "orchestrate", "规划", "编排", "planning",
                  "计划", "方案设计"],
    "tdd_testing": ["tdd", "test driven", "测试驱动", "unit test",
                     "单元测试", "test coverage", "覆盖率", "写测试",
                     "测试覆盖率"],
    "refactor_clean": ["refactor", "clean up", "dead code", "simplify",
                        "重构", "清理", "简化", "代码重构", "代码清理"],
    "e2e": ["e2e", "end to end", "playwright", "端到端", "端到端测试"],
    "documentation": ["document", "readme", "docstring", "文档生成",
                       "generate doc", "写文档", "生成文档"],
    # Additional languages (for ECC rule coverage)
    "angular": ["angular", "ng", "angular组件", "angular服务"],
    "vue": ["vue", "vuex", "pinia", "nuxt"],
    "ruby": ["ruby", "rails", "gem", "rubocop"],
    "perl": ["perl", "cpan", "perl模块"],
    "fsharp": ["f#", "fsharp", "functional", "discriminated union"],
    # ECC / coding standards specific
    "ecc_rules": ["ECC", "ecc", "rule config", "规则配置", "编码规范配置",
                  "coding standard config", "standard enforcement", "规范落地",
                  "coding rule", "编码规则"],
    # Container & cloud ops
    "container_ops": ["docker", "kubernetes", "k8s", "容器", "容器化",
                      "部署", "编排", "docker-compose", "helm", "containerd",
                      "云部署", "集群", "微服务"],
    # Communication & messaging
    "communication": ["发消息", "发送消息", "发邮件", "聊天", "群聊",
                      "即时通讯", "通知", "消息推送", "send message",
                      "messaging", "chat", "notification"],
    # Data infrastructure
    "data_ops": ["redis", "kafka", "terraform", "elasticsearch", "消息队列",
                 "缓存", "cache", "队列", "etcd", "consul", "zookeeper",
                 "nginx", "负载均衡", "反向代理"],
}
