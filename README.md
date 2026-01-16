# Financial Report Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI Status](https://github.com/your-username/financial-report-analyzer/workflows/ci/badge.svg)](https://github.com/your-username/financial-report-analyzer/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> 用于分析上市公司财报与股价数据，生成业务与竞争分析、估值指标、分析师预期、可视化图表与结构化报告。

支持美股、日股、A股、港股，并可在 OpenCode / Claude 中作为 Skill 使用。

## ✨ 功能特性

- 📊 **全面的财务分析**
  - 营收和利润趋势分析
  - YoY/QoQ/CAGR 增长率计算
  - 毛利率、营业利润率、净利率分析
  - ROE、ROA 等盈利能力指标
  - 负债率、流动比率等财务健康指标

- 💰 **深度估值分析**
  - P/E、P/S、P/B 比率计算
  - 5年历史分位数排名
  - PEG 成长性估值
  - EV/EBITDA 企业价值倍数
  - DCF 现金流折现参考

- 🎯 **分析师预期追踪**
  - 目标价和评级汇总
  - 评级分布统计
  - EPS 修正趋势

- 📈 **可视化图表**
  - 收入和利润趋势图
  - 利润率变化图
  - ROE/ROA 趋势图
  - 负债率变化图
  - 股价历史走势图

- 📝 **结构化报告**
  - 自动生成 Markdown 格式报告
  - 包含公司概况、业务模式、竞争格局
  - 完整的财务指标对比表
  - 投资建议总结

## 🌍 支持市场

| 市场 | 数据源 | 示例代码 |
|------|--------|----------|
| 🇺🇸 美股 | yfinance | `AAPL`, `MSFT`, `GOOGL` |
| 🇯🇵 日股 | yfinance | `7203.T` (丰田), `6758.T` (索尼) |
| 🇨🇳 A股 | akshare | `600519.SH` (贵州茅台), `000858.SZ` (五粮液) |
| 🇭🇰 港股 | yfinance | `0700.HK` (腾讯), `9988.HK` (阿里巴巴) |

## 📦 数据源

### 免费数据源（默认）
- **yfinance** - 美股、日股、港股数据
- **akshare** - A股数据

### 可选付费 API
- **Alpha Vantage** - [获取免费 API Key](https://www.alphavantage.co/support/#api-key)
- **Financial Modeling Prep** - [获取免费 API Key](https://site.financialmodelingprep.com/developer/docs)
- **Tushare Pro** - [获取 Token](https://tushare.pro/register)

## 🏗️ 项目结构

```
financial-report-analyzer/
├── SKILL.md              # Skill 定义文件（OpenCode/Claude）
├── README.md             # 项目说明
├── LICENSE               # MIT 许可证
├── CHANGELOG.md          # 版本变更记录
├── CONTRIBUTING.md       # 贡献指南
├── .env.example          # 环境变量示例
├── scripts/
│   ├── fetch_data.py     # 数据获取
│   ├── analyze.py        # 财务分析
│   ├── valuation.py      # 估值计算
│   ├── analyst.py        # 分析师预期
│   ├── visualize.py      # 图表生成
│   ├── report.py         # 报告生成
│   ├── config.py         # 配置管理
│   └── requirements.txt  # Python 依赖
├── .github/
│   └── workflows/
│       └── ci.yml        # CI/CD 流程
└── output/               # 输出目录（gitignore）
```

## 📥 安装

详细的安装说明请查看 [INSTALL.md](INSTALL.md)。

快速安装：

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/financial-report-analyzer.git
cd financial-report-analyzer

# 2. 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements.txt

# 3. 配置 OpenCode/Claude skill（可选）
mkdir -p ~/.opencode/skill
ln -s "$(pwd)" ~/.opencode/skill/financial-report-analyzer
```

详细安装指南：[INSTALL.md](INSTALL.md)

## 开发
- 主入口脚本都在 `scripts/` 中
- 每个步骤是独立脚本，可单独运行或串联

### Pre-commit
```bash
python -m pip install pre-commit
pre-commit install
```

示例流程：

```bash
python scripts/run_report.py --symbol AAPL --years 1 --output output
```

> `run_report.py` 会自动判断市场（如 `.SH/.SZ/.BJ/.HK/.T`），并默认在 24 小时内复用缓存数据；使用 `--refresh` 可强制更新。

完整逐步执行：

```bash
python scripts/fetch_data.py --symbol AAPL --market US --years 1 --output output
python scripts/analyze.py --input output/AAPL_data.json --output output
python scripts/valuation.py --input output/AAPL_data.json --analysis output/AAPL_analysis.json --output output
python scripts/analyst.py --input output/AAPL_data.json --output output
python scripts/visualize.py --analysis output/AAPL_analysis.json --output output/AAPL_charts
python scripts/report.py --analysis output/AAPL_analysis.json --valuation output/AAPL_valuation.json --analyst output/AAPL_analyst.json --output output
```

> 未指定 `--years` 时默认获取 1 年数据，可按需调整。

## 测试方式
端到端示例（AAPL）：

```bash
. .venv/bin/activate
python scripts/run_report.py --symbol AAPL --years 1 --output output
```

逐步执行：

```bash
python scripts/fetch_data.py --symbol AAPL --market US --years 1 --output output
python scripts/analyze.py --input output/AAPL_data.json --output output
python scripts/valuation.py --input output/AAPL_data.json --analysis output/AAPL_analysis.json --output output
python scripts/analyst.py --input output/AAPL_data.json --output output
python scripts/visualize.py --analysis output/AAPL_analysis.json --output output/AAPL_charts
python scripts/report.py --analysis output/AAPL_analysis.json --valuation output/AAPL_valuation.json --analyst output/AAPL_analyst.json --output output
```

输出文件位置：
- `output/AAPL_data.json`
- `output/AAPL_analysis.json`
- `output/AAPL_valuation.json`
- `output/AAPL_analyst.json`
- `output/AAPL_report.md`
- `output/AAPL_charts/*.png`

## 免责声明
本项目仅用于数据分析与研究参考，不构成任何投资建议。

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目开发。

### 贡献方式

- 🐛 报告 Bug
- 💡 提出新功能建议
- 📖 改进文档
- 🔧 提交代码

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 🔗 相关链接

- [更新日志](CHANGELOG.md)
- [贡献指南](CONTRIBUTING.md)
- [Issues](https://github.com/your-username/financial-report-analyzer/issues)
- [Discussions](https://github.com/your-username/financial-report-analyzer/discussions)

## 🙏 致谢

感谢以下开源项目：

- [yfinance](https://github.com/ranaroussi/yfinance) - 美股/日股/港股数据
- [akshare](https://github.com/akfamily/akshare) - A股数据
- [polars](https://pola.rs/) - 数据处理
- [matplotlib](https://matplotlib.org/) - 数据可视化

---

**Made with ❤️ for financial analysis**
