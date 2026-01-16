---
name: financial-report-analyzer
description: 分析上市公司财报，提供业务模式总结、竞争格局分析、历史财报对比、估值深度分析、分析师预期和投资建议，支持美股、日股、A股、港股。
license: MIT
compatibility: opencode, claude
metadata:
  category: finance
  markets: US, JP, CN, HK
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

### 0) 环境准备 (首次运行)

**自动检测项目根目录（推荐）：**
```bash
# 从 SKILL.md 所在目录自动检测
export SKILL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$SKILL_ROOT/.venv"
. "$SKILL_ROOT/.venv/bin/activate"
python -m pip install -r "$SKILL_ROOT/scripts/requirements.txt"
```

**或使用环境变量（适用于自定义路径）：**
```bash
# 如果已设置 SKILL_ROOT 环境变量，直接使用
python3 -m venv "$SKILL_ROOT/.venv"
. "$SKILL_ROOT/.venv/bin/activate"
python -m pip install -r "$SKILL_ROOT/scripts/requirements.txt"
```

> 已安装依赖可跳过本步。

### 1) 运行前准备 (每次执行)

**自动检测模式：**
```bash
export SKILL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SKILL_ROOT/.venv/bin/activate"
```

**或使用环境变量模式：**
```bash
. "$SKILL_ROOT/.venv/bin/activate"
```

### 2) 一键生成（推荐）
```bash
python "$SKILL_ROOT/scripts/run_report.py" \
  --symbol AAPL \
  --years 1 \
  --output "$SKILL_ROOT/output"
```

提示：
- 自动识别市场（如 `.SH/.SZ/.BJ/.HK/.T`），无需手动指定 `--market`
- 默认 24 小时内复用缓存数据，可用 `--refresh` 强制拉取最新数据
- 可用 `--skip-valuation/--skip-analyst/--skip-charts` 加速生成

### 3) 数据获取
```bash
YEARS=${YEARS:-1}
python "$SKILL_ROOT/scripts/fetch_data.py" \
  --symbol AAPL \
  --market US \
  --years "$YEARS" \
  --output "$SKILL_ROOT/output"
```

输出: `output/AAPL_data.json`

提示: 若用户指定年份（如“近 3 年”），设置 `YEARS=3`；未指定则使用默认 1 年。

### 4) 财务分析
```bash
python "$SKILL_ROOT/scripts/analyze.py" \
  --input "$SKILL_ROOT/output/AAPL_data.json" \
  --output "$SKILL_ROOT/output"
```

输出: `output/AAPL_analysis.json`

### 5) 估值与分析师预期
```bash
python "$SKILL_ROOT/scripts/valuation.py" \
  --input "$SKILL_ROOT/output/AAPL_data.json" \
  --analysis "$SKILL_ROOT/output/AAPL_analysis.json" \
  --output "$SKILL_ROOT/output"

python "$SKILL_ROOT/scripts/analyst.py" \
  --input "$SKILL_ROOT/output/AAPL_data.json" \
  --output "$SKILL_ROOT/output"
```

输出: `output/AAPL_valuation.json` / `output/AAPL_analyst.json`

### 6) 生成图表
```bash
python "$SKILL_ROOT/scripts/visualize.py" \
  --analysis "$SKILL_ROOT/output/AAPL_analysis.json" \
  --output "$SKILL_ROOT/output/AAPL_charts"
```

### 7) 生成报告
```bash
python "$SKILL_ROOT/scripts/report.py" \
  --analysis "$SKILL_ROOT/output/AAPL_analysis.json" \
  --valuation "$SKILL_ROOT/output/AAPL_valuation.json" \
  --analyst "$SKILL_ROOT/output/AAPL_analyst.json" \
  --output "$SKILL_ROOT/output"
```

输出: `output/AAPL_report.md`

## 输出报告结构
- 公司概况
- 业务模式分析
- 竞争格局
- 财务分析与历史对比
- 估值深度分析
- 分析师预期
- 图表与结论
- 投资建议 (基本面优先)

## 数据源
- 美股/港股/日股: yfinance
- A股: AkShare
- 可选付费 API: Alpha Vantage / FMP / Tushare Pro

## 注意
本 Skill 仅提供分析参考，不构成投资建议。
