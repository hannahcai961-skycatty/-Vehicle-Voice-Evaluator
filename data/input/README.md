# 测试数据说明

本目录默认包含 **5 条 sample**，用于快速验证评估流水线：

| 文件 | 说明 |
|------|------|
| `car_dialogue_dataset_sample.json` | 多轮用户 query（5 个 conversation） |
| `china_cases_sample.txt` | 与 query 一一对应的 case 实体上下文 |

## 使用完整数据集

将全量文件复制到本目录，例如：

- `car_dialogue_dataset_500_高质量.json`
- `china_cases_500.txt`

并在项目根目录 `.env` 中设置 `QUERY_FILE`、`CASES_FILE`、`ALL_CASE_JSON` 指向上述文件。

全量文件已在 `.gitignore` 中排除，不会提交到 Git。
