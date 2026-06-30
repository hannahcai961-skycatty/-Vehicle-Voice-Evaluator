"""
MD 文档批量汇总脚本
将指定目录下的 MD 文件按顺序每 50 个合并为一个汇总文档
"""

import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

INPUT_DIR = Path(os.getenv("RESULTS_DIR", str(PROJECT_ROOT / "data" / "results")))
OUTPUT_DIR = Path(os.getenv("SUMMARIES_DIR", str(PROJECT_ROOT / "data" / "summaries")))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))


def get_case_id(filename: str) -> str:
    """从文件名提取 case_id，取第一个下划线之前的部分"""
    return filename.split("_")[0]


def build_summaries():
    md_files = sorted(INPUT_DIR.glob("*.md"), key=lambda f: f.name)

    if not md_files:
        print(f"[!] 未在 {INPUT_DIR} 找到任何 .md 文件，请检查路径。")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    batches = [md_files[i : i + BATCH_SIZE] for i in range(0, len(md_files), BATCH_SIZE)]
    print(f"共找到 {len(md_files)} 个文件，将生成 {len(batches)} 个汇总文档")

    for idx, batch in enumerate(batches, start=1):
        first_case = get_case_id(batch[0].name)
        last_case = get_case_id(batch[-1].name)
        lines = []

        lines.append(f"# 车载语音助手评估报告 第{idx}批\n")
        lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
        lines.append(f"本批数量：{len(batch)} 篇  ")
        lines.append(f"编号范围：{first_case} ～ {last_case}\n")

        lines.append("## 目录\n")
        for f in batch:
            cid = get_case_id(f.name)
            lines.append(f"- [{cid}](#{cid.lower()})")
        lines.append("\n---\n")

        for f in batch:
            cid = get_case_id(f.name)
            content = f.read_text(encoding="utf-8")
            lines.append(f"## {cid}\n")
            lines.append(content.strip())
            lines.append("\n\n---\n")

        out_path = OUTPUT_DIR / f"batch_{idx:02d}.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  ✓ 第 {idx:02d} 批已保存：{out_path}（{len(batch)} 篇，{first_case} ～ {last_case}）")

    print(f"\n全部完成，汇总文档已保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    build_summaries()
