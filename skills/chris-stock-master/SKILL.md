---
name: chris-stock-master
description: 深度分析全球上市公司财报（美/日/港/A股）。提供业务模式、竞争格局、历史对比、估值分析及投资建议。依赖 yfinance、akshare、matplotlib、polars、numpy、requests。
---

# Stock Master - 财报分析器

深度分析全球上市公司财报，支持美股、日股、港股、A股。

## 功能

| 功能 | 说明 |
|------|------|
| 财务分析 | YoY/QoQ/CAGR 增长率、利润率、ROE/ROA |
| 估值分析 | P/E 分位数、PEG、同行对比、DCF 参考 |
| 分析师预期 | 目标价、评级分布、EPS 修正趋势 |
| 可视化 | 收入趋势、利润率、股价走势图表 |
| 报告生成 | 结构化 Markdown 报告 |

## 支持市场

| 市场 | 示例 |
|------|------|
| 美股 | `AAPL`, `MSFT`, `GOOGL` |
| 日股 | `7203.T` (丰田) |
| A股 | `600519.SH` (茅台), `000858.SZ` (五粮液) |
| 港股 | `0700.HK` (腾讯) |

## 使用示例

```bash
# 分析苹果公司
分析 AAPL

# 分析贵州茅台
分析 600519.SH

# 分析腾讯控股
分析 0700.HK
```

## 执行步骤

### 1. 环境准备（首次运行）

```bash
python3 -m venv scripts/.venv
scripts/.venv/bin/pip install -r scripts/requirements.txt
```

### 2. 运行分析

```bash
scripts/.venv/bin/python scripts/run_report.py --symbol AAPL --years 1 --output output
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--symbol` | 股票代码（自动识别市场后缀） |
| `--years` | 获取年数（默认 1） |
| `--output` | 输出目录 |
| `--refresh` | 强制刷新缓存 |
| `--skip-valuation` | 跳过估值分析 |
| `--skip-analyst` | 跳过分析师预期 |
| `--skip-charts` | 跳过图表生成 |

## 输出文件

```
output/
├── {SYMBOL}_data.json       # 原始数据
├── {SYMBOL}_analysis.json   # 分析结果
├── {SYMBOL}_valuation.json  # 估值数据
├── {SYMBOL}_analyst.json    # 分析师预期
├── {SYMBOL}_report.md       # 完整报告
└── {SYMBOL}_charts/         # 图表目录
    └── *.png
```

## 数据源

- **美股/港股/日股**: yfinance
- **A股**: AkShare

详细分步操作见 [references/advanced.md](references/advanced.md)

---

本 Skill 仅提供分析参考，不构成投资建议。
