---
name: financial-report-analyzer
description: 深度分析全球上市公司财报（美/日/港/A股）。提供业务模式、竞争格局、历史对比、估值分析及投资建议。依赖 yfinance、akshare、pandas、matplotlib。
---

## 我能做什么

- 默认获取 1 年财报与行情数据，可按需调整
- 总结业务模式与竞争格局
- 生成关键财务指标对比 (YoY/QoQ/CAGR)
- 深度估值分析 (P/E 分位数、PEG、同行对比、DCF 参考)
- 分析师预期 (目标价、评级分布、EPS 修正趋势)
- 生成 PNG 图表并输出投资建议

## 何时使用

- 你需要快速分析某家上市公司财报
- 你希望对比历史财务指标与估值变化
- 你想把分析结果整理成结构化报告

## 输入示例

- "分析 AAPL"
- "分析 600519.SH 贵州茅台"
- "分析 0700.HK 腾讯"
- "分析 7203.T 丰田汽车"

## 执行步骤

### 环境准备 (首次运行)

```bash

python3 -m venv scripts/.venv
scripts/.venv/bin/pip install -r scripts/requirements.txt
```

### 运行分析

```bash
scripts/.venv/bin/python scripts/run_report.py --symbol AAPL --years 1 --output output
```

参数说明：

- 自动识别市场后缀（`.SH/.SZ/.HK/.T`）
- `--refresh` 强制刷新缓存
- `--skip-valuation/--skip-analyst/--skip-charts` 加速

详细分步操作见 [references/advanced.md](references/advanced.md)

## 数据源

- 美股/港股/日股: yfinance
- A股: AkShare

## 注意

本 Skill 仅提供分析参考，不构成投资建议。
