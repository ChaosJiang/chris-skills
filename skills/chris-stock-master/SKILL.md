---
name: chris-stock-master
description: 分析上市公司财报（美/日/港/A股），生成估值分析和投资报告
---

# Stock Master

分析全球上市公司财报，生成结构化报告。

## 触发条件

当用户请求以下内容时使用此 Skill：
- "分析 AAPL" / "分析苹果公司"
- "看看腾讯的财报" / "分析 0700.HK"
- "茅台估值如何" / "分析 600519.SH"
- 任何关于股票财报、估值、财务分析的请求

## 支持市场

| 市场 | 代码格式 | 示例 |
|------|----------|------|
| 美股 | `SYMBOL` | AAPL, MSFT, GOOGL |
| 日股 | `CODE.T` | 7203.T (丰田) |
| A股 | `CODE.SH/SZ` | 600519.SH (茅台) |
| 港股 | `CODE.HK` | 0700.HK (腾讯) |

## 执行流程

### 1. 环境检查（首次）

```bash
cd "$SKILL_DIR"
if [ ! -d "scripts/.venv" ]; then
  python3 -m venv scripts/.venv
  scripts/.venv/bin/pip install -r scripts/requirements.txt
fi
```

### 2. 运行分析

```bash
cd "$SKILL_DIR"
scripts/.venv/bin/python scripts/run_report.py --symbol <SYMBOL> --output output
```

### 3. 读取并呈现结果

分析完成后，读取生成的报告文件并向用户呈现关键信息：

```bash
cat output/<SYMBOL>_report.md
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--symbol` | 股票代码 | 必填 |
| `--years` | 财报年数 | 1 |
| `--output` | 输出目录 | output |
| `--refresh` | 强制刷新缓存 | false |
| `--skip-charts` | 跳过图表生成 | false |

## 输出文件

```
output/
├── <SYMBOL>_report.md      # 主报告（呈现给用户）
├── <SYMBOL>_data.json      # 原始数据
├── <SYMBOL>_analysis.json  # 分析结果
├── <SYMBOL>_valuation.json # 估值数据
├── <SYMBOL>_analyst.json   # 分析师预期
└── <SYMBOL>_charts/        # 图表
```

## 呈现指南

向用户呈现结果时：
1. 先展示公司基本信息和最新股价
2. 重点呈现估值分析（P/E 分位数、与同行对比）
3. 展示关键财务指标趋势
4. 最后给出投资建议摘要
5. 如有图表，告知用户图表位置

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| A股数据获取失败 | 检查网络，akshare 需要访问国内数据源 |
| 缓存数据过旧 | 使用 `--refresh` 强制刷新 |
| 图表生成慢 | 使用 `--skip-charts` 跳过 |

## 注意事项

- 数据缓存 24 小时，使用 `--refresh` 强制更新
- 本 Skill 仅提供分析参考，不构成投资建议
- A股使用 AkShare，其他市场使用 yfinance
