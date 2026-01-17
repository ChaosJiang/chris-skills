## 详细步骤操作

### 1) 数据获取

```bash
scripts/.venv/bin/python scripts/fetch_data.py \
  --symbol AAPL \
  --market US \
  --years 1 \
  --output output
```

输出: `output/AAPL_data.json`

提示: 若用户指定年份（如“近 3 年”），设置 `YEARS=3`；未指定则使用默认 1 年。

### 2) 财务分析

```bash
scripts/.venv/bin/python scripts/analyze.py \
  --input output/AAPL_data.json \
  --output output
```

输出: `output/AAPL_analysis.json`

### 3) 估值与分析师预期

```bash
scripts/.venv/bin/python scripts/valuation.py \
  --input output/AAPL_data.json \
  --analysis output/AAPL_analysis.json \
  --output output

scripts/.venv/bin/python scripts/analyst.py \
  --input output/AAPL_data.json \
  --output output
```

输出: `output/AAPL_valuation.json` / `output/AAPL_analyst.json`

### 4) 生成图表

```bash
scripts/.venv/bin/python scripts/visualize.py \
  --analysis output/AAPL_analysis.json \
  --output output/AAPL_charts
```

### 5) 生成报告

```bash
scripts/.venv/bin/python scripts/report.py \
  --analysis output/AAPL_analysis.json \
  --valuation output/AAPL_valuation.json \
  --analyst output/AAPL_analyst.json \
  --output output
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
