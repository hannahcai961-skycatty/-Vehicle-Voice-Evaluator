"""
Dify Workflow 批量评估脚本
"""

import asyncio
import aiohttp
import json
import os
import time
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_BASE = PROJECT_ROOT / "data"
RESULTS_DIR = OUTPUT_BASE / "results"
FAILED_FILE = OUTPUT_BASE / "failed.json"
SUMMARY_DIR = OUTPUT_BASE / "summaries"
LOG_DIR = PROJECT_ROOT / "logs"

API_BASE_URL = os.getenv("DIFY_API_URL", "https://your-dify-host/v1/workflows/run")
API_KEY = os.getenv("DIFY_API_KEY", "")
QUERY_FILE = Path(os.getenv("QUERY_FILE", str(INPUT_DIR / "car_dialogue_dataset_sample.json")))
CASES_FILE = Path(os.getenv("CASES_FILE", str(INPUT_DIR / "china_cases_sample.txt")))

CONCURRENCY = int(os.getenv("CONCURRENCY", "5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
RETRY_DELAYS = [5, 15, 30]
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "500"))

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "run.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def to_csv_escaped(json_str: str) -> str:
    return '"' + json_str.replace('"', '""') + '"'


async def read_sse(resp) -> str:
    async for line in resp.content:
        line = line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except Exception:
            continue
        if event.get("event") == "workflow_finished":
            status = event.get("data", {}).get("status", "")
            error = event.get("data", {}).get("error", "")
            if status == "failed":
                raise RuntimeError(f"Workflow 内部失败: {error}")
            return event.get("data", {}).get("outputs", {}).get("text", "")
    return ""


def load_data():
    log.info("正在加载数据文件...")
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        cases_list = json.loads(f.read().strip())
    log.info(f"cases 加载完成：{len(cases_list)} 条")
    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        query_raw = json.load(f)
    conversations = query_raw["conversations"]
    log.info(f"query 加载完成：{len(conversations)} 条")
    count = min(len(cases_list), len(conversations))
    pairs = []
    for i in range(count):
        case = cases_list[i]
        conv = conversations[i]
        pairs.append({
            "Case_ID": case.get("case_id", f"case{i+1:03d}"),
            "Conv_ID": conv.get("conv_id", f"C{i+1:03d}"),
            "test_cases": json.dumps(case, ensure_ascii=False),
            "query": json.dumps(conv, ensure_ascii=False),
        })
    log.info(f"数据匹配完成：共 {len(pairs)} 对")
    return pairs


async def call_workflow(session, pair, semaphore):
    case_id = pair["Case_ID"]
    conv_id = pair["Conv_ID"]
    payload = {
        "inputs": {
            "test_cases": pair["test_cases"],
            "query": to_csv_escaped(pair["query"]),
            "Case_ID": case_id,
            "Conv_ID": conv_id,
        },
        "response_mode": "streaming",
        "user": "eval-batch",
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            log.info(f"[→] {case_id}/{conv_id} 第{attempt+1}次尝试，发送请求...")
            try:
                timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                async with session.post(
                    API_BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
                    report_text = await read_sse(resp)
                    if not report_text:
                        raise RuntimeError("workflow_finished 未收到或 text 为空")
                    log.info(f"[←] {case_id}/{conv_id} 收到结果，长度 {len(report_text)} 字符")
                    return {"status": "success", "case_id": case_id, "conv_id": conv_id, "report": report_text}
            except Exception as e:
                wait = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else 30
                log.warning(
                    f"[!] {case_id}/{conv_id} 第{attempt+1}次失败: {e}，"
                    f"{'重试中...' if attempt < MAX_RETRIES - 1 else '放弃'}"
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(wait)

    return {"status": "failed", "case_id": case_id, "conv_id": conv_id, "report": ""}


async def run_batch(pairs):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done_ids = {f.stem for f in RESULTS_DIR.glob("*.md")}
    todo = [p for p in pairs if f"{p['Case_ID']}_{p['Conv_ID']}" not in done_ids]
    log.info(f"待处理：{len(todo)} 条（已完成：{len(done_ids)} 条，跳过）")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    failed_records = []
    reports = []
    for f in RESULTS_DIR.glob("*.md"):
        reports.append({"case_id": f.stem.split("_")[0], "report": f.read_text(encoding="utf-8")})
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 5)
    async with aiohttp.ClientSession(connector=connector, read_bufsize=10 * 1024 * 1024) as session:
        tasks = [call_workflow(session, pair, semaphore) for pair in todo]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result["status"] == "success":
                filename = RESULTS_DIR / f"{result['case_id']}_{result['conv_id']}.md"
                filename.write_text(result["report"], encoding="utf-8")
                reports.append(result)
            else:
                failed_records.append({"case_id": result["case_id"], "conv_id": result["conv_id"]})
    if failed_records:
        FAILED_FILE.write_text(json.dumps(failed_records, ensure_ascii=False, indent=2), encoding="utf-8")
        log.warning(f"失败 {len(failed_records)} 条，已记录到 {FAILED_FILE}")
    log.info(f"批量运行完成：成功 {len(reports)} 条，失败 {len(failed_records)} 条")
    return reports


def build_summaries(reports: list):
    """每50篇拼成一个 Markdown 文件，带目录"""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    batches = [reports[i : i + BATCH_SIZE] for i in range(0, len(reports), BATCH_SIZE)]
    log.info(f"开始生成批次文档，共 {len(batches)} 批...")

    for idx, batch in enumerate(batches):
        lines = []
        lines.append(f"# 车载语音助手评估报告 第{idx+1}批")
        lines.append(f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
        lines.append(f"本批数量：{len(batch)} 篇  ")
        lines.append(f"编号范围：{batch[0]['case_id']} ～ {batch[-1]['case_id']}\n")
        lines.append("## 目录\n")
        for r in batch:
            lines.append(f"- [{r['case_id']}](#{r['case_id'].lower()})")
        lines.append("\n---\n")
        for r in batch:
            lines.append(f"## {r['case_id']}\n")
            lines.append(r["report"])
            lines.append("\n\n---\n")

        output = SUMMARY_DIR / f"batch_{idx+1:02d}.md"
        output.write_text("\n".join(lines), encoding="utf-8")
        log.info(f"第 {idx+1} 批已保存：{output}（{len(batch)} 篇）")

    log.info(f"全部批次文档已生成，保存至 {SUMMARY_DIR}")


async def main():
    if not API_KEY:
        log.error("未配置 DIFY_API_KEY，请复制 .env.example 为 .env 并填写")
        return

    start = time.time()
    log.info("=" * 50)
    log.info("车载语音助手批量评估开始")
    log.info("=" * 50)
    pairs = load_data()
    reports = await run_batch(pairs)
    if reports:
        build_summaries(reports)
    else:
        log.error("没有成功的报告，跳过汇总")
    log.info(f"全部完成，耗时 {time.time() - start:.1f} 秒")


if __name__ == "__main__":
    asyncio.run(main())
