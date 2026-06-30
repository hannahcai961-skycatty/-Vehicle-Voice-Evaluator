# Vehicle-Voice-Evaluator

这是一个专为**车载语音助手（In-Car Voice Assistant）**设计的自动化批量评估、汇总与语义分析系统。项目结合了 **Dify Workflow** 自动化测试流程与大语言模型（如 Qwen 3.5）的深度分析能力，能够对数百个测试用例（Cases）进行批量并发测试，自动生成可视化图表及多维度的评估报告。

## 核心功能

1. **Dify 批量并发评估**：读取车载对话数据集，通过异步并发（`asyncio` / `aiohttp`）驱动 Dify 工作流批量运行测试，自动处理超时与重试。
2. **报告自动批处理合并**：支持将生成的数百篇 Markdown 单例报告，按指定数量（默认 50 篇）自动分批合并，并自动生成目录导航。
3. **多维度语义指标分析**：
   - 提取并分析多轮对话中的漏提槽位（Slots Missing Top 25）
   - 分析对话轮数（Turns）对模型最终得分的影响
   - 自动调用 LLM API 进行更深层次的语义对齐与错误聚类分析
4. **可视化图表生成**：自动绘制并导出混淆矩阵、得分分布等统计图表

## 项目目录结构

```text
Vehicle-Voice-Evaluator/
├── .env.example              # 环境变量模板（复制为 .env 后填写）
├── .gitignore
├── README.md
├── requirements.txt
├── config/
│   └── workflow_config.yml   # Dify 工作流导出文件
├── src/
│   ├── run_eval.py           # Dify 批量评估
│   ├── batch_summary.py      # 报告分批合并
│   ├── analyze_reports.py    # 汇总分析与图表
│   └── car_dialogue_generator.py  # 可选：LLM 生成测试数据集
├── data/
│   ├── input/                # 测试集（仓库内为 sample，全量请自行放置）
│   ├── results/              # 单例 md 报告（运行产物，已 gitignore）
│   ├── summaries/            # 分批合并报告
│   └── analyze/              # 最终报告与 figures/
└── logs/
```

## 快速开始

### 1. 环境准备

需要 Python 3.10+（脚本使用了 `Path | None` 等类型注解）。

```bash
pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
```

在 `.env` 中填写：

| 变量 | 用途 |
|------|------|
| `DIFY_API_URL` | Dify Workflow API 地址 |
| `DIFY_API_KEY` | Dify 应用 API Key |
| `LLM_API_BASE_URL` | 分析阶段 LLM 接口（OpenAI 兼容） |
| `LLM_API_KEY` | LLM API Key |
| `LLM_MODEL_NAME` | 分析用模型名，默认 `qwen3.5-plus` |

### 2. 准备测试数据

仓库默认包含 **5 条 sample** 数据，用于验证流程：

- `data/input/car_dialogue_dataset_sample.json` — 多轮对话 query
- `data/input/china_cases_sample.txt` — 对应 case 实体上下文

**使用完整 500 条数据集时**，将全量文件放入 `data/input/`（文件名示例）：

- `car_dialogue_dataset_500_高质量.json`
- `china_cases_500.txt`

然后在 `.env` 中取消注释并设置：

```env
QUERY_FILE=data/input/car_dialogue_dataset_500_高质量.json
CASES_FILE=data/input/china_cases_500.txt
ALL_CASE_JSON=data/input/car_dialogue_dataset_500_高质量.json
```

> 全量数据文件已在 `.gitignore` 中排除，不会提交到 GitHub。

### 3. 导入 Dify 工作流

1. 在 Dify 控制台导入 `config/workflow_config.yml`
2. 在工作流中配置 OpenAI Compatible 模型凭证
3. 将 `.env` 中的 `DIFY_API_URL` / `DIFY_API_KEY` 指向该应用

### 4. 运行评估流水线

在项目根目录执行：

```bash
python src/run_eval.py          # 批量调用 Dify，输出 data/results/*.md
python src/batch_summary.py       # 合并为 data/summaries/batch_*.md
python src/analyze_reports.py     # 生成 data/analyze/final_report.md 与图表
```

可选：生成新测试数据

```bash
python src/car_dialogue_generator.py
```

## 流水线示意

```text
data/input/（cases + conversations）
        ↓  run_eval.py
data/results/*.md
        ↓  batch_summary.py
data/summaries/batch_*.md
        ↓  analyze_reports.py
data/analyze/final_report.md + figures/
```

## 许可证

如需开源发布，请自行添加 LICENSE 文件。
