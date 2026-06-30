"""Generate GitHub portfolio PDF — single-page, styled layout.

Personal info is read from .env (see .env.example). Never commit real contact details.
"""

import os
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DESKTOP = Path.home() / "Desktop"
OUTPUT = DESKTOP / "GitHub项目作品集.pdf"
FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")

NAME = os.getenv("PORTFOLIO_NAME", "【请填写姓名】")
PHONE = os.getenv("PORTFOLIO_PHONE", "【请填写手机】")
EMAIL = os.getenv("PORTFOLIO_EMAIL", "【请填写邮箱】")
GITHUB = os.getenv("PORTFOLIO_GITHUB", "https://github.com/hannahcai961-skycatty")

C_PRIMARY = (26, 54, 93)
C_ACCENT1 = (37, 99, 235)
C_ACCENT2 = (5, 150, 105)
C_BG_LIGHT = (241, 245, 249)
C_BG_CARD = (248, 250, 252)
C_TEXT = (30, 41, 59)
C_MUTED = (100, 116, 139)
C_WHITE = (255, 255, 255)
C_TAG_BG = (219, 234, 254)
C_TAG_TX = (30, 64, 175)


class PortfolioPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("msyh", "", str(FONT_REGULAR))
        self.add_font("msyh", "B", str(FONT_BOLD))
        self.l_margin = 10
        self.r_margin = 10
        self.t_margin = 0
        self.page_w = 210
        self.content_w = self.page_w - self.l_margin - self.r_margin

    def rgb(self, color):
        self.set_text_color(*color)

    def fill_rgb(self, color):
        self.set_fill_color(*color)

    def draw_rgb(self, color):
        self.set_draw_color(*color)

    def text_at(self, x, y, text, size=9, bold=False, color=C_TEXT, w=0, h=5, align="L"):
        self.set_xy(x, y)
        self.rgb(color)
        self.set_font("msyh", "B" if bold else "", size)
        if w:
            self.multi_cell(w, h, text, align=align, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            self.cell(0, h, text, align=align)

    def section_label(self, x, y, text, accent):
        self.fill_rgb(accent)
        self.rect(x, y, 2.5, 5, style="F")
        self.text_at(x + 4, y, text, size=8.5, bold=True, color=C_TEXT)

    def tag_row(self, x, y, tags, max_w=None):
        cx = x
        max_w = max_w or self.content_w
        for tag in tags:
            self.set_font("msyh", "", 7.5)
            tw = self.get_string_width(tag) + 5
            if cx + tw > x + max_w:
                cx = x
                y += 6
            self.fill_rgb(C_TAG_BG)
            self.draw_rgb((191, 219, 254))
            self.rect(cx, y, tw, 5.5, style="FD")
            self.rgb(C_TAG_TX)
            self.text_at(cx + 2.5, y + 0.6, tag, size=7.5, w=0)
            cx += tw + 2
        return y + 7

    def project_card(self, x, y, w, h, accent, data):
        self.fill_rgb(C_BG_CARD)
        self.draw_rgb((226, 232, 240))
        self.rect(x, y, w, h, style="FD")
        self.fill_rgb(accent)
        self.rect(x, y, 3, h, style="F")

        pad = 4
        cx = x + pad + 2
        cy = y + 4
        inner_w = w - pad * 2 - 2

        self.fill_rgb(accent)
        self.rect(cx, cy, 14, 6, style="F")
        self.rgb(C_WHITE)
        self.text_at(cx + 1, cy + 0.8, data["badge"], size=7.5, bold=True, color=C_WHITE)

        self.rgb(C_TEXT)
        self.text_at(cx + 16, cy + 0.5, data["title"], size=10, bold=True, w=inner_w - 16)

        cy += 8
        self.text_at(cx, cy, data["subtitle"], size=8, color=C_MUTED, w=inner_w)
        cy += 7

        self.rgb(accent)
        self.set_font("msyh", "", 7)
        self.set_xy(cx, cy)
        self.multi_cell(inner_w, 3.8, data["url"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        cy += 8

        for section_title, lines in data["sections"]:
            self.section_label(cx, cy, section_title, accent)
            cy += 6
            for line in lines:
                self.rgb(C_TEXT)
                self.set_font("msyh", "", 8)
                self.set_xy(cx + 1, cy)
                self.multi_cell(inner_w - 1, 4.2, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                cy = self.get_y() + 0.5
            cy += 1.5

        self.section_label(cx, cy, "技术栈", accent)
        cy += 6
        self.tag_row(cx, cy, data["tags"], max_w=inner_w)


def build_pdf() -> Path:
    pdf = PortfolioPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    pdf.fill_rgb(C_PRIMARY)
    pdf.rect(0, 0, 210, 32, style="F")
    pdf.rgb(C_WHITE)
    pdf.text_at(10, 8, "GitHub 项目作品集", size=18, bold=True, color=C_WHITE)
    pdf.text_at(10, 18, NAME, size=13, bold=True, color=(191, 219, 254))
    pdf.text_at(10, 25, "AI 应用 · 语音/NLP 评估 · 产品工具开发", size=8.5, color=(203, 213, 225))

    contact_y = 10
    for label, value in [("手机", PHONE), ("邮箱", EMAIL)]:
        pdf.fill_rgb((30, 64, 110))
        pdf.rect(128, contact_y, 72, 7, style="F")
        pdf.text_at(130, contact_y + 1.2, f"{label}  {value}", size=8, color=C_WHITE)
        contact_y += 9
    pdf.fill_rgb((30, 64, 110))
    pdf.rect(128, contact_y, 72, 7, style="F")
    pdf.text_at(130, contact_y + 1.2, "github.com/hannahcai961-skycatty", size=7.5, color=(147, 197, 253))

    y = 36
    pdf.fill_rgb(C_BG_LIGHT)
    pdf.rect(10, y, 190, 14, style="F")
    pdf.draw_rgb((203, 213, 225))
    pdf.rect(10, y, 190, 14, style="D")
    highlights = [
        ("500+ Cases", "车载语音批量评测"),
        ("Dify + LLM", "自动化评估流水线"),
        ("0–100 评分", "JD 智能匹配过滤"),
        ("MIT 开源", "双项目可商用"),
    ]
    hx = 14
    for num, desc in highlights:
        pdf.text_at(hx, y + 2.5, num, size=9, bold=True, color=C_ACCENT1)
        pdf.text_at(hx, y + 7.5, desc, size=7.5, color=C_MUTED)
        hx += 47

    card_y = 54
    card_h = 122
    col_w = 93
    gap = 4

    project1 = {
        "badge": "项目 01",
        "title": "Vehicle-Voice-Evaluator",
        "subtitle": "车载语音助手 · 批量评估与报告分析",
        "url": "github.com/hannahcai961-skycatty/-Vehicle-Voice-Evaluator",
        "sections": [
            ("项目背景", [
                "专为 In-Car Voice Assistant 设计，解决数百条多轮对话 case 无法人工逐条评测、"
                "报告分散难以汇总的问题，实现从测试到分析的全链路自动化。",
            ]),
            ("核心功能", [
                "· Dify Workflow 异步并发批量跑测（asyncio/aiohttp）",
                "· 数百篇 Markdown 报告自动分批合并与目录导航",
                "· 漏提槽位 Top 25、轮数衰减、意图混淆矩阵可视化",
                "· LLM 深度语义分析与错误模式聚类，输出终报告",
            ]),
            ("项目亮点", [
                "· 完整评估流程：准备数据 → Dify 跑测 → 合并报告 → 分析出图",
                "· 含 Dify 工作流配置、sample 数据集与 MIT 开源许可",
            ]),
        ],
        "tags": ["Python", "Dify", "LLM", "matplotlib", "aiohttp", "NLU评估"],
    }

    project2 = {
        "badge": "项目 02",
        "title": "Job Application Assistant",
        "subtitle": "AI 产品岗 · 本地 Web 求职助手",
        "url": "github.com/hannahcai961-skycatty/job-application-assistant",
        "sections": [
            ("项目背景", [
                "面向秋招 AI 产品岗（PM / 产品运营），解决 JD 与简历匹配判断耗时、"
                "定制招呼语/邮件效率低的问题；定位为「过滤器」而非海投工具。",
            ]),
            ("核心功能", [
                "· JD 结构化匹配分析（A–F 六块 + 0–100 分 + apply/skip 建议）",
                "· 经历素材库、多版本 Markdown 简历管理与微调建议",
                "· Boss 招呼语 / 邮件话术一键生成，人工审核后复制发送",
                "· 生成记录审计（本地 JSON + Markdown 报告）",
            ]),
            ("项目亮点", [
                "· Human-in-the-loop：只生成文案，不自动发送",
                "· 本地 Web 部署，数据不出本机，FastAPI + 原生 JS 轻量架构",
            ]),
        ],
        "tags": ["FastAPI", "JavaScript", "DeepSeek", "Web", "AI产品"],
    }

    pdf.project_card(10, card_y, col_w, card_h, C_ACCENT1, project1)
    pdf.project_card(10 + col_w + gap, card_y, col_w, card_h, C_ACCENT2, project2)

    bot_y = 180
    pdf.fill_rgb(C_PRIMARY)
    pdf.rect(10, bot_y, 190, 7, style="F")
    pdf.text_at(12, bot_y + 1.5, "技术能力总览 & 项目架构", size=9, bold=True, color=C_WHITE)

    bot_y += 10
    box_h = 58
    pdf.fill_rgb((239, 246, 255))
    pdf.rect(10, bot_y, 92, box_h, style="F")
    pdf.draw_rgb((191, 219, 254))
    pdf.rect(10, bot_y, 92, box_h, style="D")
    pdf.text_at(12, bot_y + 3, "Vehicle-Voice-Evaluator 评估流程", size=8.5, bold=True, color=C_ACCENT1)

    flow1 = [
        "① 准备测试数据（对话 query + 车辆 case）",
        "        ↓",
        "② 批量调用 Dify 工作流，并发跑测",
        "        ↓",
        "③ 生成每个 case 的 Markdown 报告",
        "        ↓",
        "④ 报告分批合并，自动生成目录",
        "        ↓",
        "⑤ 统计指标 + 绘制图表 + LLM 分析",
        "        ↓",
        "⑥ 输出最终评估报告与可视化图表",
    ]
    fy = bot_y + 9
    for line in flow1:
        is_arrow = line.strip() == "↓"
        pdf.text_at(12, fy, line, size=7.2, color=C_MUTED if is_arrow else C_TEXT)
        fy += 5.5

    pdf.fill_rgb((236, 253, 245))
    pdf.rect(108, bot_y, 92, box_h, style="F")
    pdf.draw_rgb((167, 243, 208))
    pdf.rect(108, bot_y, 92, box_h, style="D")
    pdf.text_at(110, bot_y + 3, "Job Application Assistant 工作流", size=8.5, bold=True, color=C_ACCENT2)

    flow2 = [
        "录入 JD + 选择简历版本",
        "        ↓  匹配度分析（A–F + 0–100）",
        "  <60 skip  |  60–74 consider  |  ≥75 apply",
        "        ↓  简历微调建议（可选）",
        "        ↓  Boss 招呼语 / 邮件正文",
        "人工复制发送 → 更新岗位投递状态",
    ]
    fy = bot_y + 9
    for line in flow2:
        pdf.text_at(110, fy, line, size=7.2, color=C_MUTED if "↓" in line else C_TEXT)
        fy += 5.2

    skill_y = 250
    pdf.fill_rgb(C_BG_LIGHT)
    pdf.rect(10, skill_y - 2, 190, 18, style="F")
    pdf.draw_rgb((226, 232, 240))
    pdf.rect(10, skill_y - 2, 190, 18, style="D")
    pdf.text_at(12, skill_y, "综合技能", size=8.5, bold=True, color=C_TEXT)
    skill_y += 5
    all_skills = [
        "Python", "FastAPI", "asyncio", "LLM应用", "Dify Workflow",
        "Prompt工程", "NLU评估", "数据可视化", "产品设计", "Git/GitHub",
    ]
    pdf.tag_row(12, skill_y, all_skills, max_w=186)

    pdf.fill_rgb(C_PRIMARY)
    pdf.rect(0, 272, 210, 25, style="F")
    pdf.text_at(
        10, 276,
        "以上项目均为本人独立开发并开源至 GitHub · 代码与 README 含完整运行说明 · MIT License 允许商用",
        size=7.5, color=(203, 213, 225),
    )
    pdf.text_at(10, 282, GITHUB, size=8, color=(147, 197, 253))
    pdf.text_at(10, 288, f"{NAME}  |  {PHONE}  |  {EMAIL}", size=7.5, color=(148, 163, 184))

    DESKTOP.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    return OUTPUT


if __name__ == "__main__":
    path = build_pdf()
    print(f"Saved: {path}")
