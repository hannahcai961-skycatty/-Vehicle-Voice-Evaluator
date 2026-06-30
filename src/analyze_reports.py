"""
车载语音助手评估报告 — 汇总分析脚本
qwen3-vl-flash 模型能力全面分析

使用方法：
  python analyze_reports.py

配置区在文件开头，填写路径和 API 信息即可。
"""

import json
import re
import ast
import os
import sys
import time
import requests
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ════════════════════════════════════════════════════════════
#  配置区 —— 可通过 .env 或环境变量覆盖
# ════════════════════════════════════════════════════════════
INPUT_DIR = Path(os.getenv("RESULTS_DIR", str(PROJECT_ROOT / "data" / "results")))
OUTPUT_DIR = Path(os.getenv("ANALYZE_DIR", str(PROJECT_ROOT / "data" / "analyze")))
ALL_CASE_JSON = Path(os.getenv(
    "ALL_CASE_JSON",
    str(PROJECT_ROOT / "data" / "input" / "car_dialogue_dataset_sample.json"),
))

API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3.5-plus")

CHINESE_FONT_PATH = os.getenv("CHINESE_FONT_PATH", "")
# ════════════════════════════════════════════════════════════

FIGURES_DIR   = OUTPUT_DIR / "figures"
PARSED_JSON   = OUTPUT_DIR / "parsed_data.json"
FINAL_REPORT  = OUTPUT_DIR / "final_report.md"
ALL_CASE_JSON_GLOB = "car_dialogue_dataset*.json"


# ─────────────────────────────────────────────
#  字体设置
# ─────────────────────────────────────────────
def setup_font():
    candidates = [
        CHINESE_FONT_PATH,
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for p in candidates:
        if p and Path(p).exists():
            prop = fm.FontProperties(fname=p)
            plt.rcParams["font.family"] = prop.get_name()
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"[字体] 使用：{p}")
            return prop
    # 回退：matplotlib 自带字体，中文可能显示为方框
    plt.rcParams["font.family"] = "DejaVu Sans"
    print("[字体] 未找到中文字体，标签将使用英文替代")
    return None

FONT_PROP = None   # 全局字体属性，setup_font() 后赋值


# ─────────────────────────────────────────────
#  第一阶段：解析 MD 文件
# ─────────────────────────────────────────────

def extract_float(text, pattern, default=None):
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return default


def extract_int(text, pattern, default=None):
    m = re.search(pattern, text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return default


def resolve_all_case_json() -> Path | None:
    """定位测试集 JSON（优先 ALL_CASE_JSON，否则在 data/input 下 glob）。"""
    if ALL_CASE_JSON.exists():
        return ALL_CASE_JSON
    input_dir = PROJECT_ROOT / "data" / "input"
    candidates = sorted(input_dir.glob(ALL_CASE_JSON_GLOB))
    return candidates[0] if candidates else None


def parse_missed_slots_table(section):
    """从槽位提取失败表格中解析漏提槽位"""
    missed = []
    # 匹配表格数据行
    pattern = re.compile(
        r'^\|\s*(\d+)\s*\|.*?\|\s*(\{.*?\})\s*\|\s*([\d.]+)\s*\|',
        re.MULTILINE
    )
    for m in pattern.finditer(section):
        turn_num = int(m.group(1))
        misses_str = m.group(2)
        f1 = float(m.group(3))
        try:
            misses_dict = json.loads(misses_str)
        except Exception:
            misses_dict = {}
        missed.append({
            "turn": turn_num,
            "misses": list(misses_dict.keys()),
            "f1": f1
        })
    return missed


def parse_intent_fail_table(text):
    """从意图理解失败表格中解析意图误判记录"""
    # 先截取“意图理解失败详情”章节，避免把“槽位提取失败详情”误解析进来
    start_m = re.search(r'###\s*意图理解失败详情\s*\n', text)
    if not start_m:
        return []

    tail = text[start_m.end():]
    end_m = re.search(r'\n###\s*槽位提取失败详情|\n##\s*对话记录|\Z', tail)
    section = tail[:end_m.start()] if end_m else tail

    fails = []
    pattern = re.compile(
        r'^\|\s*(\d+)\s*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]*)\|',
        re.MULTILINE
    )

    for m in pattern.finditer(section):
        turn_num_str = m.group(1).strip()
        if not turn_num_str.isdigit():
            continue
        fails.append({
            "turn": int(turn_num_str),
            "query": m.group(2).strip(),
            "ground": m.group(3).strip(),
            "predicted": m.group(4).strip(),
            "llm_answer": m.group(5).strip()
        })
    return fails


def parse_conversation(text):
    """解析对话记录段落，返回 turn 列表"""
    m = re.search(r'## 对话记录\s*\n+([\s\S]+?)(?:\n## |\Z)', text)
    if not m:
        return []
    raw = m.group(1).strip()
    try:
        turns = ast.literal_eval(raw)
        return [t for t in turns if isinstance(t, dict) and t]
    except Exception:
        return []


def parse_one_md(filepath: Path) -> dict | None:
    """解析单个 md 文件，返回结构化数据字典"""
    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  [读取失败] {filepath.name}: {e}")
        return None

    case_id = filepath.stem.split("_")[0]
    conv_id = filepath.stem.split("_")[1] if "_" in filepath.stem else ""

    result = {
        "case_id": case_id,
        "conv_id": conv_id,
        "filename": filepath.name,
        # 综合得分
        "final_score": extract_float(text, r'\*\*([\d.]+)\s*/\s*100\*\*'),
        "grade": None,
        "score_mode": None,
        # 客观层
        "objective_score": extract_float(text, r'客观层\s*\|\s*([\d.]+)\s*/\s*100'),
        "subjective_score": extract_float(text, r'主观层\s*\|\s*([\d.]+)\s*/\s*100'),
        "total_turns": extract_int(text, r'测试轮数\s*\|\s*(\d+)\s*轮'),
        "intent_accuracy": extract_float(text, r'意图准确率\s*\|\s*([\d.]+)'),
        "slot_f1_avg": extract_float(text, r'槽位\s*F1\s*均值\s*\|\s*([\d.]+)'),
        # 主观层文字
        "reasoning": "",
        "strengths": "",
        "weaknesses": "",
        # 原始失败模式归纳文字（每个 case 报告中 LLM 生成的段落）
        "failure_patterns_text": "",
        "improvement_text": "",
        # 失败案例
        "intent_fails": [],
        "slot_fails": [],
        # 对话记录
        "conversation": [],
        # 解析状态
        "parse_ok": True,
        "parse_errors": []
    }

    # 等级
    grade_m = re.search(r'等级[：:]\s*(\S+)', text)
    if grade_m:
        result["grade"] = grade_m.group(1)

    # 计分模式
    mode_m = re.search(r'计分模式[：:]\s*(\S+)', text)
    if mode_m:
        result["score_mode"] = mode_m.group(1)

    # 主观层文字
    for field, pattern in [
        ("reasoning", r'\*\*评分理由[：:]\*\*\s*(.+)'),
        ("strengths", r'\*\*模型优势[：:]\*\*\s*(.+)'),
        ("weaknesses", r'\*\*主要问题[：:]\*\*\s*(.+)'),
    ]:
        m = re.search(pattern, text)
        if m:
            result[field] = m.group(1).strip()

    # 提取原始报告中"失败模式归纳"段落
    pat_m = re.search(
        r'##\s*失败模式归纳\s*\n+([\s\S]+?)(?=\n##\s*改进建议|\n##\s*失败案例|\Z)',
        text
    )
    if pat_m:
        result["failure_patterns_text"] = pat_m.group(1).strip()

    # 提取"改进建议"段落
    imp_m = re.search(
        r'##\s*改进建议\s*\n+([\s\S]+?)(?=\n##\s*|\Z)',
        text
    )
    if imp_m:
        result["improvement_text"] = imp_m.group(1).strip()

    # 失败案例
    result["intent_fails"] = parse_intent_fail_table(text)
    result["slot_fails"]   = parse_missed_slots_table(text)

    # 对话记录
    result["conversation"] = parse_conversation(text)

    # 校验必要字段
    for field in ["final_score", "total_turns", "intent_accuracy", "slot_f1_avg"]:
        if result[field] is None:
            result["parse_errors"].append(f"缺失字段: {field}")
            result["parse_ok"] = False

    return result


def phase1_parse(md_files: list[Path]) -> list[dict]:
    print(f"\n{'='*50}")
    print(f"第一阶段：解析 {len(md_files)} 个 MD 文件")
    print(f"{'='*50}")
    records = []
    failed = []
    for i, fp in enumerate(md_files, 1):
        rec = parse_one_md(fp)
        if rec is None:
            failed.append(fp.name)
            continue
        if not rec["parse_ok"]:
            print(f"  [警告] {fp.name} 解析不完整: {rec['parse_errors']}")
        records.append(rec)
        if i % 50 == 0:
            print(f"  已处理 {i}/{len(md_files)}")

    print(f"\n解析完成：成功 {len(records)} 条，失败 {len(failed)} 条")
    if failed:
        print(f"  失败文件：{failed}")

    # 保存中间数据
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_JSON.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"中间数据已保存：{PARSED_JSON}")
    return records


# ─────────────────────────────────────────────
#  第二阶段：统计计算 + 可视化
# ─────────────────────────────────────────────

def safe_label(zh: str, en: str) -> str:
    """根据字体可用性返回中文或英文标签"""
    return zh if FONT_PROP else en


def fig01_score_distribution(records: list[dict]):
    scores = [r["final_score"] for r in records if r["final_score"] is not None]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores, bins=20, color="#4C72B0", edgecolor="white", linewidth=0.8)
    ax.axvline(np.mean(scores), color="#DD3B3B", linestyle="--", linewidth=1.5,
               label=f'{safe_label("均值", "Mean")} {np.mean(scores):.1f}')
    ax.axvline(np.median(scores), color="#F5A623", linestyle="--", linewidth=1.5,
               label=f'{safe_label("中位数", "Median")} {np.median(scores):.1f}')
    ax.set_xlabel(safe_label("综合得分", "Final Score"), fontproperties=FONT_PROP)
    ax.set_ylabel(safe_label("Case 数量", "Count"), fontproperties=FONT_PROP)
    ax.set_title(safe_label("综合得分分布", "Score Distribution"), fontproperties=FONT_PROP, fontsize=14)
    ax.legend(prop=FONT_PROP)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "01_score_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  图1 已保存：{out}")


def fig02_grade_pie(records: list[dict]):
    grade_counts = Counter(r["grade"] for r in records if r["grade"])
    order = ["优秀", "良好", "及格", "不及格"]
    en_order = ["Excellent", "Good", "Pass", "Fail"]
    labels_zh = [g for g in order if g in grade_counts]
    labels_en = [en_order[order.index(g)] for g in labels_zh]
    values = [grade_counts[g] for g in labels_zh]
    colors = ["#2ECC71", "#3498DB", "#F39C12", "#E74C3C"][:len(labels_zh)]
    labels = labels_zh if FONT_PROP else labels_en

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontproperties": FONT_PROP} if FONT_PROP else {}
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title(safe_label("等级分布", "Grade Distribution"), fontproperties=FONT_PROP, fontsize=14)
    fig.tight_layout()
    out = FIGURES_DIR / "02_grade_pie.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  图2 已保存：{out}")

def fig03_turns_vs_scores(records: list[dict]):
    """
    对话轮数分段比较：短(10-13轮) / 中(14-16轮) / 长(17-20轮)
    每档显示：case数量、综合得分均值±std、意图准确率均值、槽位F1均值
    同时在右侧附上精确轮数的散点图，供参考（样本量<5的轮数用空心点标注）
    """
    # ── 分段定义 ──
    segments = [
        ("短对话\n(10-13轮)", "Short\n(10-13)", range(10, 14)),
        ("中等对话\n(14-16轮)", "Mid\n(14-16)",  range(14, 17)),
        ("长对话\n(17-20轮)", "Long\n(17-20)",  range(17, 21)),
    ]
    MIN_SAMPLES = 5   # 精确轮数散点图中，样本量低于此值用虚线/空心点提示
 
    seg_labels   = []
    seg_n        = []
    seg_final    = []
    seg_final_std= []
    seg_intent   = []
    seg_slot     = []
 
    for zh, en, turn_range in segments:
        subset = [
            r for r in records
            if r.get("total_turns") in turn_range
            and r.get("final_score") is not None
        ]
        label = zh if FONT_PROP else en
        seg_labels.append(f"{label}\n(n={len(subset)})")
        seg_n.append(len(subset))
        if subset:
            finals  = [r["final_score"] for r in subset]
            intents = [r["intent_accuracy"] for r in subset if r.get("intent_accuracy") is not None]
            slots   = [r["slot_f1_avg"]    for r in subset if r.get("slot_f1_avg")    is not None]
            seg_final.append(np.mean(finals))
            seg_final_std.append(np.std(finals))
            seg_intent.append(np.mean(intents) * 100 if intents else 0)
            seg_slot.append(np.mean(slots)   * 100 if slots   else 0)
        else:
            seg_final.append(0); seg_final_std.append(0)
            seg_intent.append(0); seg_slot.append(0)
 
    # ── 精确轮数散点（右子图）──
    exact_buckets = defaultdict(lambda: {"final": [], "intent": [], "slot": []})
    for r in records:
        t = r.get("total_turns")
        if t is None or r.get("final_score") is None:
            continue
        exact_buckets[t]["final"].append(r["final_score"])
        if r.get("intent_accuracy") is not None:
            exact_buckets[t]["intent"].append(r["intent_accuracy"])
        if r.get("slot_f1_avg") is not None:
            exact_buckets[t]["slot"].append(r["slot_f1_avg"])
 
    turns_sorted = sorted(exact_buckets.keys())
    ex_final  = [np.mean(exact_buckets[t]["final"])  for t in turns_sorted]
    ex_intent = [np.mean(exact_buckets[t]["intent"]) * 100 for t in turns_sorted]
    ex_slot   = [np.mean(exact_buckets[t]["slot"])   * 100 for t in turns_sorted]
    ex_counts = [len(exact_buckets[t]["final"])            for t in turns_sorted]
 
    # ── 画图：左子图=分段柱状图，右子图=精确轮数折线 ──
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6),
                                             gridspec_kw={"width_ratios": [1, 1.4]})
 
    # 左：分段柱状图
    x = np.arange(len(seg_labels))
    w = 0.25
    b1 = ax_left.bar(x - w, seg_final,  w, label=safe_label("综合得分", "Final Score"),
                     color="#4C72B0", yerr=seg_final_std, capsize=4)
    b2 = ax_left.bar(x,     seg_intent, w, label=safe_label("意图准确率×100", "Intent Acc×100"),
                     color="#DD3B3B")
    b3 = ax_left.bar(x + w, seg_slot,   w, label=safe_label("槽位F1×100", "Slot F1×100"),
                     color="#2ECC71")
 
    # 数值标注
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax_left.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                             f"{h:.1f}", ha="center", va="bottom", fontsize=8)
 
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(seg_labels,
                             fontproperties=FONT_PROP if FONT_PROP else None, fontsize=9)
    ax_left.set_ylabel(safe_label("得分", "Score"), fontproperties=FONT_PROP)
    ax_left.set_title(safe_label("分段对话长度 vs 各维度均值（含±std）",
                                  "Segment: Turns vs Scores (±std)"),
                      fontproperties=FONT_PROP, fontsize=12)
    ax_left.set_ylim(0, 110)
    ax_left.legend(prop=FONT_PROP, fontsize=8)
    ax_left.grid(axis="y", alpha=0.3)
 
    # 右：精确轮数折线（小样本用空心点区分）
    for t, yf, yi, ys, cnt in zip(turns_sorted, ex_final, ex_intent, ex_slot, ex_counts):
        marker_style = "o" if cnt >= MIN_SAMPLES else "o"
        alpha = 1.0 if cnt >= MIN_SAMPLES else 0.4
        fill  = True if cnt >= MIN_SAMPLES else False
 
    # 分两组画：足够样本 vs 小样本
    def plot_line_with_reliability(ax, xs, ys_vals, color, label, counts, min_n):
        xs_ok  = [x for x, y, c in zip(xs, ys_vals, counts) if c >= min_n]
        ys_ok  = [y for x, y, c in zip(xs, ys_vals, counts) if c >= min_n]
        xs_low = [x for x, y, c in zip(xs, ys_vals, counts) if c < min_n]
        ys_low = [y for x, y, c in zip(xs, ys_vals, counts) if c < min_n]
        if xs_ok:
            ax.plot(xs_ok, ys_ok, "o-", color=color, linewidth=2, label=label)
        if xs_low:
            ax.plot(xs_low, ys_low, "o", color=color, alpha=0.35,
                    markerfacecolor="none", markeredgewidth=1.5, markersize=8)
            for x_, y_ in zip(xs_low, ys_low):
                ax.annotate(safe_label("样本少", "low n"),
                            (x_, y_), fontsize=6, color="gray",
                            xytext=(2, 4), textcoords="offset points")
 
    plot_line_with_reliability(ax_right, turns_sorted, ex_final,
                               "#4C72B0", safe_label("综合得分", "Final Score"), ex_counts, MIN_SAMPLES)
    plot_line_with_reliability(ax_right, turns_sorted, ex_intent,
                               "#DD3B3B", safe_label("意图准确率×100", "Intent Acc×100"), ex_counts, MIN_SAMPLES)
    plot_line_with_reliability(ax_right, turns_sorted, ex_slot,
                               "#2ECC71", safe_label("槽位F1×100", "Slot F1×100"), ex_counts, MIN_SAMPLES)
 
    # 次坐标轴：精确轮数的 case 数量
    ax_right2 = ax_right.twinx()
    ax_right2.bar(turns_sorted, ex_counts, alpha=0.12, color="gray",
                  label=safe_label("Case数", "N Cases"))
    ax_right2.set_ylabel(safe_label("Case 数量", "N Cases"), fontproperties=FONT_PROP, fontsize=9)
    ax_right2.axhline(MIN_SAMPLES, color="gray", linestyle=":", linewidth=1,
                      label=safe_label(f"最小样本线(n={MIN_SAMPLES})", f"min n={MIN_SAMPLES}"))
    ax_right2.legend(prop=FONT_PROP, fontsize=7, loc="upper right")
 
    ax_right.set_xlabel(safe_label("精确对话轮数", "Exact Turn Count"), fontproperties=FONT_PROP)
    ax_right.set_ylabel(safe_label("得分", "Score"), fontproperties=FONT_PROP)
    ax_right.set_title(safe_label("精确轮数折线（空心点=样本量不足，仅供参考）",
                                   "Exact Turns (hollow=low sample, reference only)"),
                       fontproperties=FONT_PROP, fontsize=11)
    ax_right.set_xticks(turns_sorted)
    ax_right.set_ylim(0, 110)
    ax_right.legend(prop=FONT_PROP, fontsize=8, loc="upper left")
    ax_right.grid(alpha=0.3)
 
    fig.suptitle(safe_label("对话轮数与模型表现关系分析", "Turn Length vs Model Performance"),
                 fontproperties=FONT_PROP, fontsize=14, y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "03_turns_vs_scores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图3 已保存：{out}")
 

def fig04_intra_case_decay(records: list[dict]):
    """
    单 case 内部各轮衰减分析：
    对每个 case，统计每轮是否意图命中（从 intent_fails 反推），
    然后按轮次编号取所有 case 均值，画衰减曲线。
    """
    # 先收集每个 case 每轮的意图命中情况
    # intent_fails 记录了失败的轮次；total_turns 是总轮数
    turn_hit = defaultdict(list)   # turn_num -> [0/1, ...]

    for r in records:
        total = r.get("total_turns")
        if not total:
            continue
        fail_turns = {f["turn"] for f in r.get("intent_fails", [])}
        for t in range(1, total + 1):
            turn_hit[t].append(0 if t in fail_turns else 1)

    max_turn = max(turn_hit.keys()) if turn_hit else 20
    x = list(range(1, max_turn + 1))
    y_mean = []
    y_count = []
    for t in x:
        vals = turn_hit.get(t, [])
        y_mean.append(np.mean(vals) if vals else None)
        y_count.append(len(vals))

    # 过滤掉 None
    x_plot = [t for t, v in zip(x, y_mean) if v is not None]
    y_plot = [v for v in y_mean if v is not None]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_plot, [v * 100 for v in y_plot], "o-", color="#9B59B6", linewidth=2.5)
    ax.fill_between(x_plot, [v * 100 for v in y_plot], alpha=0.15, color="#9B59B6")
    ax.set_xlabel(safe_label("轮次编号", "Turn Number"), fontproperties=FONT_PROP)
    ax.set_ylabel(safe_label("意图命中率 (%)", "Intent Hit Rate (%)"), fontproperties=FONT_PROP)
    ax.set_title(safe_label("单 Case 内部轮次衰减（跨所有 Case 均值）",
                             "Intra-Case Decay (avg across all cases)"),
                 fontproperties=FONT_PROP, fontsize=14)
    ax.set_ylim(0, 105)
    ax.set_xticks(x_plot)
    ax.grid(alpha=0.3)
    # 标注每个点的 case 数量
    for xi, yi, cnt in zip(x_plot, [v*100 for v in y_plot], [y_count[t-1] for t in x_plot]):
        ax.annotate(f"n={cnt}", (xi, yi), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7, color="gray")
    fig.tight_layout()
    out = FIGURES_DIR / "04_intra_case_decay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  图4 已保存：{out}")


def fig05_top_missed_slots(records: list[dict]):
    slot_counter = Counter()
    for r in records:
        for sf in r.get("slot_fails", []):
            for slot in sf.get("misses", []):
                slot_counter[slot.strip()] += 1

    top25 = slot_counter.most_common(25)
    if not top25:
        print("  [跳过] 没有槽位失败数据，跳过图5")
        return

    labels = [item[0] for item in top25]
    values = [item[1] for item in top25]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(labels[::-1], values[::-1], color="#E67E22", edgecolor="white")
    ax.set_xlabel(safe_label("漏提次数", "Miss Count"), fontproperties=FONT_PROP)
    ax.set_title(safe_label("高频漏提槽位 Top 25", "Top 25 Most Missed Slots"),
                 fontproperties=FONT_PROP, fontsize=14)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)
    if FONT_PROP:
        for label in ax.get_yticklabels():
            label.set_fontproperties(FONT_PROP)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "05_top_missed_slots.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  图5 已保存：{out}")


def fig06_intent_confusion(records: list[dict]):
    """意图混淆矩阵"""
    # 归一化意图标签（去掉空格，处理多意图用+连接的情况，只取第一个）
    intent_abbr = {
        "cabin_internal_perception": "cabin",
        "cabin_perception": "env",
        "vehicle_basic_info": "veh",
        "real_time_vehicle_status": "rtv",
        "location_and_navigation": "nav",
        "time_and_schedule": "time",
        "user_preference_and_habit": "pref",
        "communication": "comm",
        "entertainment_media": "media",
    }
    label_order = ["cabin", "env", "veh", "rtv", "nav", "time", "pref", "comm", "media"]
    valid_labels = set(label_order)

    def normalize(intent_str: str) -> str:
        if not intent_str:
            return None
        s = intent_str.strip()
        # 多意图场景只取第一个
        s = re.split(r'[+＋,，]', s)[0].strip()

        # 先长名 -> 缩写
        if s in intent_abbr:
            return intent_abbr[s]
        # 已经是缩写
        if s in valid_labels:
            return s
        # 其他都丢弃（如 0.5、{"traffic_event": null} 等污染值）
        return None

    pairs = []
    for r in records:
        for f in r.get("intent_fails", []):
            g = normalize(f.get("ground", ""))
            p = normalize(f.get("predicted", ""))
            if g and p:
                pairs.append((g, p))

    if not pairs:
        print("  [跳过] 没有意图失败数据，跳过图6")
        return
    all_labels = label_order
    label_idx = {l: i for i, l in enumerate(all_labels)}
    n = len(all_labels)
    matrix = np.zeros((n, n), dtype=int)
    for g, p in pairs:
        matrix[label_idx[g]][label_idx[p]] += 1
    '''
    # ── 新增：只保留出现频率最高的 Top N 标签，避免图像过大 ──
    MAX_LABELS = 20
    label_counter = Counter()
    for g, p in pairs:
        label_counter[g] += 1
        label_counter[p] += 1
    top_labels = {label for label, _ in label_counter.most_common(MAX_LABELS)}

     # 过滤只保留 top_labels 中的 pairs
    pairs = [(g, p) for g, p in pairs if g in top_labels and p in top_labels]
    if not pairs:
        print("  [跳过] 过滤后意图失败数据为空，跳过图6")
        return


    all_labels = sorted(set([p[0] for p in pairs] + [p[1] for p in pairs]))
    label_idx  = {l: i for i, l in enumerate(all_labels)}
    n = len(all_labels)
    matrix = np.zeros((n, n), dtype=int)
    for g, p in pairs:
        if g in label_idx and p in label_idx:
            matrix[label_idx[g]][label_idx[p]] += 1'''

    fig, ax = plt.subplots(figsize=(max(10, n), max(8, n - 2)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    tick_kw = {"fontproperties": FONT_PROP} if FONT_PROP else {}
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8, **tick_kw)
    ax.set_yticklabels(all_labels, fontsize=8, **tick_kw)
    ax.set_xlabel(safe_label("预测意图", "Predicted Intent"), fontproperties=FONT_PROP)
    ax.set_ylabel(safe_label("真实意图", "Ground Intent"), fontproperties=FONT_PROP)
    ax.set_title(safe_label("意图混淆矩阵（意图失败案例）",
                             "Intent Confusion Matrix (fail cases only)"),
                 fontproperties=FONT_PROP, fontsize=13)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] > 0:
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    out = FIGURES_DIR / "06_intent_confusion.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  图6 已保存：{out}")


def phase2_visualize(records: list[dict]):
    print(f"\n{'='*50}")
    print("第二阶段：统计计算 + 可视化")
    print(f"{'='*50}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    global FONT_PROP
    FONT_PROP = setup_font()

    fig01_score_distribution(records)
    fig02_grade_pie(records)
    fig03_turns_vs_scores(records)
    fig04_intra_case_decay(records)
    fig05_top_missed_slots(records)
    fig06_intent_confusion(records)
    print("可视化完成")


# ─────────────────────────────────────────────
#  第三阶段：API 语义归纳
# ─────────────────────────────────────────────

def build_quantitative_summary(records: list[dict]) -> str:
    """
    构建定量统计摘要：槽位漏提 Top25、意图误判路径 Top10、
    各轮次失败分布。作为 API prompt 的"数字锚点"。
    """
    lines = []

    # 槽位统计
    slot_counter = Counter()
    for r in records:
        for sf in r.get("slot_fails", []):
            for slot in sf.get("misses", []):
                slot_counter[slot.strip()] += 1

    lines.append("【定量：Top15 漏提槽位】")
    for slot, cnt in slot_counter.most_common(15):
        lines.append(f"  {slot}: {cnt} 次")

    # 意图误判路径
    intent_fail_counter = Counter()
    for r in records:
        for f in r.get("intent_fails", []):
            g = f.get("ground", "").strip()
            p = f.get("predicted", "").strip()
            if g and p and g != p:
                intent_fail_counter[f"{g} → {p}"] += 1

    lines.append("\n【定量：Top10 意图误判路径】")
    for path, cnt in intent_fail_counter.most_common(10):
        lines.append(f"  {path}: {cnt} 次")

    # 各轮次失败分布（靠后轮次是否更差）
    late_intent_fails  = 0   # turn >= 总轮数 * 0.6
    early_intent_fails = 0
    for r in records:
        total = r.get("total_turns") or 0
        threshold = max(1, int(total * 0.6))
        for f in r.get("intent_fails", []):
            if f["turn"] >= threshold:
                late_intent_fails += 1
            else:
                early_intent_fails += 1

    total_intent_fails = late_intent_fails + early_intent_fails
    lines.append(f"\n【定量：意图失败分布（后60%轮次 vs 前40%轮次）】")
    lines.append(f"  后60%轮次失败：{late_intent_fails} 次 "
                 f"({late_intent_fails/total_intent_fails*100:.1f}%)" if total_intent_fails else "  无数据")
    lines.append(f"  前40%轮次失败：{early_intent_fails} 次 "
                 f"({early_intent_fails/total_intent_fails*100:.1f}%)" if total_intent_fails else "")

    lines.append(f"\n【汇总数字】")
    lines.append(f"  总 case 数: {len(records)}")
    lines.append(f"  意图失败总轮次: {sum(len(r.get('intent_fails',[])) for r in records)}")
    lines.append(f"  槽位失败总轮次: {sum(len(r.get('slot_fails',[])) for r in records)}")

    return "\n".join(lines)


def build_qualitative_summary(records: list[dict], max_cases: int = 120) -> str:
    """
    从每个 case 的原始报告中提取 LLM 已归纳的"失败模式"文字，
    压缩后作为定性素材。同时附上几条典型原始 query 作为例证。
    """
    lines = []
    lines.append("【定性：各 Case 失败模式归纳原文（节选）】")
    lines.append("（以下每条来自一个独立测试 case，是单个 case 内部的局部归纳）\n")

    # 优先选有 failure_patterns_text 的 case，打乱顺序后取前 max_cases 条
    # 保留得分偏低的 case（更有代表性）
    candidates = [
        r for r in records
        if r.get("failure_patterns_text") and r.get("final_score") is not None
    ]
    candidates.sort(key=lambda r: r["final_score"])   # 低分优先
    selected = candidates[:max_cases]

    for r in selected:
        score_str = f"{r['final_score']:.1f}" if r.get("final_score") else "N/A"
        lines.append(
            f"--- [{r['case_id']} | 得分{score_str} | {r.get('total_turns','')}轮] ---"
        )
        # 失败模式原文（截断到 300 字避免单条过长）
        pat_text = r["failure_patterns_text"][:300]
        lines.append(pat_text)

        # 附上 1-2 条典型 query 作为"有图有真相"的例证
        intent_examples = r.get("intent_fails", [])[:1]
        slot_examples   = r.get("slot_fails", [])[:1]
        for f in intent_examples:
            lines.append(
                f"  [意图误判例] turn{f['turn']}: "
                f"ground={f.get('ground','')} predicted={f.get('predicted','')} "
                f"query={f.get('query','')[:50]}"
            )
        for sf in slot_examples:
            lines.append(
                f"  [槽位漏提例] turn{sf['turn']}: "
                f"漏槽={','.join(sf.get('misses',[])[:4])} f1={sf.get('f1','')}"
            )
        lines.append("")

    return "\n".join(lines)

PROMPT='''
# Role
你是一位资深的车载语音助手 NLU (自然语言理解) 模型评估专家，拥有 10 年 NLP 工程与数据标注经验。你擅长从 ASR (语音识别) 错误、意图识别偏差、槽位提取丢失、上下文管理失效等维度进行深度归因分析。

# Task
请对用户提供的【失败案例数据】进行深度分析。你需要不仅指出模型错在哪里，还要提出具体的优化建议。

# Analysis Framework (思考路径)
在输出前，请严格按照以下逻辑步骤进行分析：
1. **现象拆解**：分析 User 输入内容，对比模型实际输出与预期输出。
2. **根因归纳**：判断错误类型属于以下哪一种：
   - 意图分类错误（Intent Classification）
   - 槽位提取错误（Slot Filling）
   - 上下文/指代消解错误（Context/Coreference）
3. **改进建议**：针对根因，提出可执行的建议。

# Output Format (Markdown)
请严格按以下格式输出分析报告：

## 1. 案例概览
| 案例 ID | 用户输入 | 实际输出 | 预期输出 |
| :--- | :--- | :--- | :--- |
| [ID] | [文本] | [结果] | [结果] |

## 2. 根因分析
- **错误类型归类**：[如：ASR识别错误]
- **深度归因**：[简述为何发生此错误，例如：该指令包含生僻地名，模型缺乏相应词库]

## 3. 改进方案建议
- **策略建议**：[例如：加入该地名的同音/异读词作为训练语料]
- **优先级**：[高/中/低]

---
请接收我的数据，准备开始分析。'''
def call_api(prompt: str) -> str:
    """调用 openai_api_compatible 接口"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    PROMPT
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 4000
    }

    url = API_BASE_URL
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[API 调用失败] {e}"


def phase3_api_analysis(records: list[dict]) -> str:
    print(f"\n{'='*50}")
    print("第三阶段：API 语义归纳（定性 + 定量融合，1次调用）")
    print(f"{'='*50}")

    # 定量摘要（统计数字）
    quant = build_quantitative_summary(records)

    # 定性摘要（各 case 原始归纳文字 + 典型 query）
    qual = build_qualitative_summary(records, max_cases=120)

    print(f"  定量摘要字数：{len(quant)} 字")
    print(f"  定性摘要字数：{len(qual)} 字")

    # 整体指标
    valid = [r for r in records if r.get("final_score") is not None]
    avg_score  = np.mean([r["final_score"] for r in valid])
    avg_intent = np.mean([r["intent_accuracy"] for r in valid if r.get("intent_accuracy") is not None])
    avg_slot   = np.mean([r["slot_f1_avg"] for r in valid if r.get("slot_f1_avg") is not None])

    prompt = f"""
你收到了 qwen3-vl-flash 模型在车载语音助手场景下 {len(records)} 个多轮对话 case 的测试报告汇总。

【整体指标】
- 平均综合得分：{avg_score:.1f} / 100
- 平均意图准确率：{avg_intent:.1%}
- 平均槽位 F1：{avg_slot:.1%}
- 对话轮数范围：10-20 轮

{quant}

{qual}

---
请基于以上两类数据（定量统计 + 各 case 定性归纳原文）完成以下分析。
定量数据提供客观频率锚点，定性文字提供语义线索，两者互相印证。

## 一、失败模式归纳（5-8个）

将各 case 中反复出现的局部问题归纳为跨 case 的系统性失败模式。
每个模式格式：

### 模式N：[简洁名称]
- **触发条件**：什么情境/输入下会触发
- **发生频率**：结合定量数据估计（高/中/低，附数字依据）
- **典型原始案例**：从定性素材中摘取 1-2 个最有代表性的 case 例子（保留 case_id 和原始 query）
- **根本原因**：模型能力或数据层面的深层原因

## 二、上下文鲁棒性分析

结合定量的"前后轮次失败分布"数据和定性素材，分析：
- 靠后轮次（后60%）的失败是否显著多于前期？原因是什么？
- 哪类意图或槽位在长对话尾部最容易崩溃？
- 是否存在"前轮错误积累 → 后轮连锁失败"的现象？给出具体例子。

## 三、改进建议（针对 qwen3-vl-flash，5-8条）

针对归纳出的失败模式，每条格式：
- **对应问题**：
- **具体做法**：
- **预期效果**：

## 四、综合评价（300字以内）

模型核心优势 + 最需要优先解决的 2-3 个方向。
"""

    print(f"  Prompt 总字数：{len(prompt)} 字")
    print("  正在调用 API 进行语义归纳...")
    result = call_api(prompt)
    print(f"  API 返回字数：{len(result)} 字")
    return result


# ─────────────────────────────────────────────
#  生成最终报告
# ─────────────────────────────────────────────

def build_final_report(records: list[dict], api_analysis: str):
    valid = [r for r in records if r.get("final_score") is not None]
    scores = [r["final_score"] for r in valid]
    grade_counts = Counter(r["grade"] for r in valid if r["grade"])
    dataset_turn_counts = {t: 0 for t in range(10, 21)}
    completed_turn_counts = {t: 0 for t in range(10, 21)}

    # 测试集case数：强制从全量JSON读取（conversations[].turns）
    try:
        all_case_json = resolve_all_case_json()
        if all_case_json and all_case_json.exists():
            ds = json.loads(all_case_json.read_text(encoding="utf-8"))
            conversations = ds.get("conversations", []) if isinstance(ds, dict) else []
            for conv in conversations:
                if not isinstance(conv, dict):
                    continue
                turns = conv.get("turns", [])
                if isinstance(turns, list):
                    n_turns = len(turns)
                    if 10 <= n_turns <= 20:
                        dataset_turn_counts[n_turns] += 1
        else:
            print(f"  [警告] 未找到全量JSON，glob={ALL_CASE_JSON_GLOB}")
    except Exception as e:
        print(f"  [警告] 全量JSON读取失败：{e}")

    # 完成测试case数：来自已完成解析的records
    for r in valid:
        t = r.get("total_turns")
        if isinstance(t, int) and 10 <= t <= 20:
            completed_turn_counts[t] += 1

    turn_coverage_table = "\n".join(
        f"| {t} | {dataset_turn_counts[t]} | {completed_turn_counts[t]} |"
        for t in range(10, 21)
    )
    coverage_md = (
        "| 对话轮数 | 测试集case数 | 完成测试case数 |\n"
        "|------|------:|------:|\n"
        f"{turn_coverage_table}\n"
    )

    # 轮数分组统计
    turn_buckets = defaultdict(list)
    for r in valid:
        if r.get("total_turns"):
            turn_buckets[r["total_turns"]].append(r["final_score"])
    turn_summary = "\n".join(
        f"| {t} 轮 | {len(v)} | {np.mean(v):.1f} |"
        for t, v in sorted(turn_buckets.items())
    )

    # Top 10 漏提槽位
    slot_counter = Counter()
    for r in records:
        for sf in r.get("slot_fails", []):
            for slot in sf.get("misses", []):
                slot_counter[slot.strip()] += 1
    top25_slots = "\n".join(
        f"| {i+1} | {slot} | {cnt} |"
        for i, (slot, cnt) in enumerate(slot_counter.most_common(25))
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# qwen3-vl-flash 车载语音助手评估分析报告

> 生成时间：{now}  
> 测试 Case 总数：{len(records)}  
> 有效解析：{len(valid)} 条

---

## 一、整体得分概览

| 指标 | 数值 |
|------|------|
| 平均综合得分 | {np.mean(scores):.2f} / 100 |
| 得分中位数 | {np.median(scores):.2f} |
| 最高分 | {max(scores):.1f} |
| 最低分 | {min(scores):.1f} |
| 标准差 | {np.std(scores):.2f} |
| 平均意图准确率 | {np.mean([r["intent_accuracy"] for r in valid if r.get("intent_accuracy") is not None]):.1%} |
| 平均槽位 F1 | {np.mean([r["slot_f1_avg"] for r in valid if r.get("slot_f1_avg") is not None]):.1%} |

### 等级分布

| 等级 | 数量 | 占比 |
|------|------|------|
{"".join(f'| {g} | {grade_counts.get(g, 0)} | {grade_counts.get(g, 0)/len(valid)*100:.1f}% |{chr(10)}' for g in ["优秀","良好","及格","不及格"])}

---

## 二、可视化图表

![综合得分分布](figures/01_score_distribution.png)

![等级分布](figures/02_grade_pie.png)

![轮数 vs 各维度得分](figures/03_turns_vs_scores.png)

![单 Case 内部轮次衰减](figures/04_intra_case_decay.png)

![高频漏提槽位 Top 25](figures/05_top_missed_slots.png)

![意图混淆矩阵](figures/06_intent_confusion.png)

---

## 三、轮数影响分析

| 对话轮数 | Case 数 | 平均综合得分 |
|----------|---------|-------------|
{turn_summary}

---

## 四、高频漏提槽位 Top 25

| 排名 | 槽位名称 | 漏提次数 |
|------|----------|---------|
{top25_slots}

---

## 五、API 语义分析结果

{api_analysis}

---

*报告由自动化脚本生成，图表保存在 figures/ 目录下。*
"""

    report = report.replace("\n---\n\n## ", f"\n\n{coverage_md}---\n\n## ", 1)
    FINAL_REPORT.write_text(report, encoding="utf-8")
    print(f"\n最终报告已保存：{FINAL_REPORT}")


# ─────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────

def main():
    start = time.time()
    print("车载语音助手评估汇总分析")
    print(f"输入目录：{INPUT_DIR}")
    print(f"输出目录：{OUTPUT_DIR}")

    # 检查输入目录
    if not INPUT_DIR.exists():
        print(f"[错误] 输入目录不存在：{INPUT_DIR}")
        sys.exit(1)

    md_files = sorted(INPUT_DIR.glob("*.md"), key=lambda f: f.name)
    if not md_files:
        print(f"[错误] 未找到 md 文件，请检查路径：{INPUT_DIR}")
        sys.exit(1)

    print(f"找到 {len(md_files)} 个 md 文件")

    # 第一阶段：解析
    records = phase1_parse(md_files)

    # 第二阶段：可视化
    phase2_visualize(records)

    # 第三阶段：API 分析
    if API_KEY:
        api_result = phase3_api_analysis(records)
    else:
        api_result = "> [跳过] 未配置 API Key，请填写配置区后重新运行第三阶段。"
        print("  [跳过] API Key 未配置")

    # 生成最终报告
    build_final_report(records, api_result)

    print(f"\n全部完成，总耗时 {time.time()-start:.1f} 秒")
    print(f"报告位置：{FINAL_REPORT}")


if __name__ == "__main__":
    main()
