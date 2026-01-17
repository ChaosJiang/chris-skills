# 高级用法

## 推荐方式

始终使用 `run_report.py` 作为主入口：

```bash
scripts/.venv/bin/python scripts/run_report.py --symbol AAPL --output output
```

输出结构：
```
output/AAPL_US/
├── report.md
├── data.json
├── analysis.json
├── valuation.json
├── analyst.json
└── charts/
    └── *.png
```

---

## 调试用途：分步执行

以下命令仅用于开发调试，输出路径与主入口不同。

### 1) 数据获取

```bash
scripts/.venv/bin/python scripts/fetch_data.py \
  --symbol AAPL \
  --market US \
  --years 1 \
  --output output
```

### 2) 财务分析

```bash
scripts/.venv/bin/python scripts/analyze.py \
  --input output/AAPL_data.json \
  --output output
```

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

---

## 报告结构

1. 公司概况
2. 业务模式分析
3. 竞争格局
4. 财务分析与历史对比
5. 估值深度分析
6. 分析师预期
7. 图表
8. 投资建议

## 数据源

| 市场 | 数据源 |
|------|--------|
| 美股/港股/日股 | yfinance |
| A股 | AkShare |

可选付费 API: Alpha Vantage / FMP / Tushare Pro
