"""
Domain-specific extras — add YOUR features here.

Copy this file, add your keywords, and pass to build_router():

    from neuro_skill.builder import build_router
    from my_extras import MY_BROAD, MY_PRECISE
    router = build_router(["./skills/"], broad_features=MY_BROAD, precise_features=MY_PRECISE)

Real example (from the neuro-skill author's own setup):
"""

# ── Example: author's domain-specific broad features ──
# Replace with your own domains
EXTRA_BROAD = {
    "communication": [
        "飞书", "lark", "feishu", "消息", "日程", "日历", "通讯录",
        "即时通讯", "云文档", "审批", "考勤", "邮件",
    ],
    "scraping": [
        "firecrawl", "爬虫", "爬取", "抓取", "scrape", "crawl",
        "网页提取", "结构化数据", "网页抓取",
    ],
}

# ── Example: author's domain-specific precise features ──
EXTRA_PRECISE = {
    # Flybook / Lark services
    "lark_im": ["发消息", "即时通讯", "群聊", "聊天", "send message"],
    "lark_calendar_vc": ["日程", "日历", "会议", "视频会议", "calendar", "meeting", "vc"],
    "lark_contact": ["通讯录", "联系人", "员工", "contact"],
    "lark_doc_wiki": ["云文档", "知识库", "wiki", "doc", "docx"],
    "lark_base": ["多维表格", "bitable", "base", "数据表"],
    "lark_approval": ["审批", "approval", "审批流"],
    "lark_drive": ["云盘", "云空间", "上传", "下载", "drive"],
    "lark_task": ["任务", "待办", "todo", "task"],
    # Web scraping
    "firecrawl": ["firecrawl", "scrape", "crawl", "爬取", "抓取"],
    # Unique tools
    "huashu_design": ["花叔", "高保真原型", "HTML原型", "prototype"],
    "sangfor_vpn": ["深信服", "easyconnect", "vpn卸载", "sangfor"],
    "pyinstaller": ["pyinstaller", "打包exe", "tkinter打包"],
    "skill_vetter": ["skill安全", "skill扫描", "skill审查"],
}

# ── Your turn: replace the above with your own features ──
# MY_BROAD = {
#     "my_domain": ["keyword1", "keyword2", "关键词"],
# }
# MY_PRECISE = {
#     "my_service": ["service-name", "服务名"],
# }
