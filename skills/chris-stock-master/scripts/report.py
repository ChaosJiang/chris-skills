#!/usr/bin/env python3
"""Generate a markdown report from analysis outputs."""

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from series_utils import series_from_mapping, series_rows

logger = logging.getLogger(__name__)


def series_from_dict(data: dict[str, float]):
    return series_from_mapping(data)


def series_to_map(series) -> dict[Any, float]:
    return {dt: value for dt, value in series_rows(series)}


def format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def format_percent(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return str(value)


def format_currency(value: Any, currency: str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        if currency:
            return f"{value:,.2f} {currency}"
        return f"{value:,.2f}"
    return str(value)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def build_financial_table(analysis: dict[str, Any]) -> str:
    revenue = series_from_dict(analysis.get("financials", {}).get("revenue", {}))
    net_income = series_from_dict(analysis.get("financials", {}).get("net_income", {}))
    gross_margin = series_from_dict(analysis.get("ratios", {}).get("gross_margin", {}))
    net_margin = series_from_dict(analysis.get("ratios", {}).get("net_margin", {}))
    roe = series_from_dict(analysis.get("ratios_ttm", {}).get("roe", {}))
    roa = series_from_dict(analysis.get("ratios_ttm", {}).get("roa", {}))
    free_cash_flow = series_from_dict(
        analysis.get("financials", {}).get("free_cash_flow", {})
    )

    base_series = revenue if revenue.height > 0 else net_income
    if base_series.height == 0:
        return "数据不足，无法生成财务对比表。"

    base_rows = series_rows(base_series)
    dates = [row[0] for row in base_rows][-5:]
    headers = [date.strftime("%Y-%m-%d") for date in dates]

    rows = [
        ("Revenue", revenue),
        ("Net Income", net_income),
        ("Gross Margin", gross_margin),
        ("Net Margin", net_margin),
        ("ROE", roe),
        ("ROA", roa),
        ("Free Cash Flow", free_cash_flow),
    ]

    table = [
        "| 指标 | " + " | ".join(headers) + " |",
        "| --- | " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for label, series in rows:
        if series.height == 0:
            values = ["-"] * len(headers)
        else:
            series_map = series_to_map(series)
            values = []
            for date in dates:
                value = series_map.get(date)
                if label.endswith("Margin") or label in {"ROE", "ROA"}:
                    values.append(format_percent(value) if value is not None else "-")
                else:
                    values.append(format_number(value) if value is not None else "-")
        table.append("| " + label + " | " + " | ".join(values) + " |")

    return "\n".join(table)


def build_percentile_label(valuation: dict[str, Any]) -> str:
    window = valuation.get("window", {})
    start = window.get("start")
    end = window.get("end")
    days = window.get("valuation_days")

    if start and end and isinstance(days, int) and days > 0:
        return f"{days}天({start}~{end})分位"
    if start and end:
        return f"{start}~{end}分位"
    return "历史分位"


def build_valuation_table(valuation: dict[str, Any]) -> str:
    metrics = valuation.get("metrics", {})
    percentiles = valuation.get("percentiles", {})

    rows = [
        ("P/E", metrics.get("pe"), percentiles.get("pe")),
        ("P/S", metrics.get("ps"), percentiles.get("ps")),
        ("P/B", metrics.get("pb"), percentiles.get("pb")),
        ("EV/EBITDA", metrics.get("ev_to_ebitda"), percentiles.get("ev_to_ebitda")),
    ]

    percentile_label = build_percentile_label(valuation)
    table = [f"| 指标 | 当前值 | {percentile_label} |", "| --- | --- | --- |"]
    for label, value, pct in rows:
        table.append(
            "| "
            + label
            + " | "
            + format_number(value)
            + " | "
            + (f"{pct:.2f}%" if pct is not None else "-")
            + " |"
        )
    return "\n".join(table)


def build_currency_note(valuation: dict[str, Any]) -> str:
    currency = valuation.get("currency", {})
    market = currency.get("market")
    financial = currency.get("financial")
    fx_rate = currency.get("fx_rate")
    converted = currency.get("converted")

    if not market or not financial:
        return ""
    if market == financial:
        return f"- 估值币种: {market}"
    if fx_rate is None or not converted:
        return (
            f"- 估值币种: {market} (财报币种: {financial})\n"
            "- ⚠️ 未能获取汇率，历史估值分位与 DCF 可能不准确"
        )
    return f"- 估值币种: {market} (财报币种: {financial}, 汇率: {fx_rate:.4f})"


def build_analyst_section(analyst: dict[str, Any]) -> str:
    rating = analyst.get("rating", {})
    targets = analyst.get("price_targets", {})

    distribution = rating.get("recent_distribution", {})
    distribution_text = ", ".join(
        [f"{key}: {value}" for key, value in distribution.items()]
    )

    lines = [
        f"- 评级关键词: {rating.get('recommendation_key', '-')}",
        f"- 平均评级: {rating.get('recommendation_mean', '-')}",
        f"- 近 90 天评级分布: {distribution_text or '-'}",
        f"- 目标价区间: {format_number(targets.get('low'))} ~ {format_number(targets.get('high'))}",
        f"- 目标价均值: {format_number(targets.get('mean'))}",
    ]
    return "\n".join(lines)


def format_value_change(current: float | None, previous: float | None) -> str:
    if current is None or previous is None or previous == 0:
        return "-"
    change = (current / previous - 1) * 100
    return f"{change:.2f}%"


def latest_series_value(series_map: dict[str, Any]) -> float | None:
    if not series_map:
        return None
    for _, value in reversed(list(series_map.items())):
        try:
            if value is None:
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def summarize_growth(analysis: dict[str, Any]) -> list[str]:
    growth = analysis.get("growth", {})
    revenue_cagr = growth.get("revenue_cagr")
    net_income_cagr = growth.get("net_income_cagr")
    lines = []
    if isinstance(revenue_cagr, (int, float)):
        lines.append(f"- 收入 CAGR: {revenue_cagr * 100:.2f}%")
    if isinstance(net_income_cagr, (int, float)):
        lines.append(f"- 净利润 CAGR: {net_income_cagr * 100:.2f}%")
    return lines


def summarize_profitability(analysis: dict[str, Any]) -> list[str]:
    ratios = analysis.get("ratios", {})
    gross_margin = latest_series_value(ratios.get("gross_margin", {}))
    net_margin = latest_series_value(ratios.get("net_margin", {}))
    lines = []
    if gross_margin is not None:
        lines.append(f"- 最新毛利率: {gross_margin * 100:.2f}%")
    if net_margin is not None:
        lines.append(f"- 最新净利率: {net_margin * 100:.2f}%")
    return lines


def summarize_cashflow(analysis: dict[str, Any]) -> list[str]:
    financials_ttm = analysis.get("financials_ttm", {})
    fcf = latest_series_value(financials_ttm.get("free_cash_flow", {}))
    if fcf is None:
        return []
    return [f"- 最新自由现金流(TTM): {format_number(fcf)}"]


def summarize_balance_sheet(analysis: dict[str, Any]) -> list[str]:
    ratios = analysis.get("ratios", {})
    debt_ratio = latest_series_value(ratios.get("debt_to_equity", {}))
    lines = []
    if debt_ratio is not None:
        lines.append(f"- 负债权益比: {debt_ratio:.2f}")
    return lines


def build_business_model_section(analysis: dict[str, Any]) -> str:
    company = analysis.get("company", {})
    summary = clean_text(company.get("summary"))
    lines = []
    if summary:
        lines.append(f"- 业务概述: {summary}")
    if company.get("industry"):
        lines.append(f"- 行业定位: {company.get('industry')}")
    if company.get("sector"):
        lines.append(f"- 领域定位: {company.get('sector')}")

    metrics_lines = []
    metrics_lines.extend(summarize_growth(analysis))
    metrics_lines.extend(summarize_profitability(analysis))
    metrics_lines.extend(summarize_cashflow(analysis))

    if metrics_lines:
        lines.append("- 经营特征:")
        lines.extend(metrics_lines)

    if not lines:
        return "- 暂无业务模式解读，建议补充官方年报或公司介绍。"
    return "\n".join(lines)


def build_competitive_section(analysis: dict[str, Any]) -> str:
    company = analysis.get("company", {})
    lines = []
    if company.get("industry"):
        lines.append(f"- 行业类型: {company.get('industry')}")
    if company.get("sector"):
        lines.append(f"- 产业板块: {company.get('sector')}")

    lines.extend(summarize_profitability(analysis))
    lines.extend(summarize_balance_sheet(analysis))

    if not lines:
        return "- 暂无竞争格局解读，建议补充同行与行业数据。"
    return "\n".join(lines)


def build_investment_section(
    analysis: dict[str, Any], valuation: dict[str, Any], analyst: dict[str, Any]
) -> str:
    lines = []
    metrics = valuation.get("metrics", {})
    percentiles = valuation.get("percentiles", {})
    current_price = valuation.get("current", {}).get("price")
    dcf_value = valuation.get("dcf", {}).get("per_share")
    target_mean = analyst.get("price_targets", {}).get("mean")

    if current_price is not None and dcf_value is not None:
        diff = format_value_change(dcf_value, current_price)
        lines.append(f"- DCF 估值对比: {format_number(dcf_value)} (较现价 {diff})")

    if current_price is not None and target_mean is not None:
        diff = format_value_change(target_mean, current_price)
        lines.append(
            f"- 分析师目标价均值: {format_number(target_mean)} (较现价 {diff})"
        )

    if metrics:
        pe = metrics.get("pe")
        ps = metrics.get("ps")
        pb = metrics.get("pb")
        lines.append(
            "- 估值指标: "
            + ", ".join(
                [
                    f"P/E {format_number(pe)} ({format_number(percentiles.get('pe'))}%)",
                    f"P/S {format_number(ps)} ({format_number(percentiles.get('ps'))}%)",
                    f"P/B {format_number(pb)} ({format_number(percentiles.get('pb'))}%)",
                ]
            )
        )

    fundamentals = []
    fundamentals.extend(summarize_growth(analysis))
    fundamentals.extend(summarize_profitability(analysis))
    fundamentals.extend(summarize_balance_sheet(analysis))
    if fundamentals:
        lines.append("- 基本面提示:")
        lines.extend(fundamentals)

    if not lines:
        return "- 暂无投资建议输出，建议补充财务与估值数据后再生成。"
    lines.append("- 本建议仅供参考，请结合风险偏好与最新公告。")
    return "\n".join(lines)


def build_data_quality_section(analysis: dict[str, Any]) -> str:
    """Build data quality appendix section."""
    dq = analysis.get("data_quality", {})
    if not dq:
        return ""

    validation = dq.get("validation", {})
    field_matching = dq.get("field_matching", {})

    lines = ["## 附录：数据质量说明", ""]

    # Validation summary
    total_checks = validation.get("total_checks", 0)
    passed = validation.get("passed", 0)
    failed = validation.get("failed", 0)

    if total_checks > 0:
        lines.append("### 数据验证")
        lines.append(f"- 总验证检查: {total_checks}")
        lines.append(f"- 通过: {passed}")
        if failed > 0:
            lines.append(f"- **警告: {failed}**")

            # Show validation details
            results = validation.get("results", [])
            warnings = [r for r in results if not r.get("passed")]
            if warnings:
                lines.append("")
                lines.append("**验证警告详情:**")
                for warning in warnings[:5]:  # Show first 5
                    lines.append(f"- {warning.get('message', 'Unknown warning')}")
                if len(warnings) > 5:
                    lines.append(f"- ... 还有 {len(warnings) - 5} 个警告")
        lines.append("")

    # Field matching summary
    fuzzy_matches = field_matching.get("fuzzy_matches", 0)
    missing_fields = field_matching.get("missing_fields", 0)

    if fuzzy_matches > 0 or missing_fields > 0:
        lines.append("### 字段匹配")
        if fuzzy_matches > 0:
            lines.append(f"- 模糊匹配字段数: {fuzzy_matches}")
            lines.append("  * 某些财务字段使用了模糊匹配算法，可能存在匹配错误")

            # Show fuzzy match details
            fuzzy_details = field_matching.get("fuzzy_matches_detail", [])
            if fuzzy_details:
                lines.append("")
                lines.append("**模糊匹配详情:**")
                for match in fuzzy_details[:5]:  # Show first 5
                    field = match.get("field", "?")
                    matched = match.get("matched", "?")
                    confidence = match.get("confidence", 0)
                    lines.append(
                        f"- '{field}' → '{matched}' (置信度: {confidence:.2f})"
                    )
                if len(fuzzy_details) > 5:
                    lines.append(f"- ... 还有 {len(fuzzy_details) - 5} 个模糊匹配")

        if missing_fields > 0:
            lines.append(f"- 缺失字段数: {missing_fields}")
            lines.append("  * 某些预期的财务字段在数据源中未找到")

        lines.append("")

    # Data completeness note
    lines.append("### 数据完整性")
    lines.append("- 本报告基于公开数据源生成")
    lines.append("- 财务数据可能存在延迟或不完整")
    lines.append("- 建议结合官方财报进行验证")
    lines.append("")

    return "\n".join(lines)


def build_report(
    analysis: dict[str, Any], valuation: dict[str, Any], analyst: dict[str, Any]
) -> str:
    company = analysis.get("company", {})
    symbol = analysis.get("symbol")
    data_fetched_at = analysis.get("data_fetched_at")
    currency = company.get("currency")

    report_lines: list[str] = [
        f"# 财报分析报告 - {company.get('name') or symbol}",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
        f"数据更新时间: {data_fetched_at or '-'}",
        "",
        "## 一、公司概况",
        f"- 公司名称: {company.get('name') or '-'}",
        f"- 行业: {company.get('industry') or '-'}",
        f"- 领域: {company.get('sector') or '-'}",
        f"- 国家/地区: {company.get('country') or '-'}",
        f"- 官网: {company.get('website') or '-'}",
        f"- 当前股价: {format_currency(valuation.get('current', {}).get('price'), currency)}",
        f"- 总市值: {format_currency(valuation.get('current', {}).get('market_cap'), currency)}",
        "",
        "## 二、业务模式分析",
        build_business_model_section(analysis),
        "",
        "## 三、竞争格局",
        build_competitive_section(analysis),
        "",
        "## 四、财务分析",
        build_financial_table(analysis),
        "",
        "## 五、估值分析",
        build_valuation_table(valuation),
        build_currency_note(valuation),
        "",
        "## 六、分析师预期",
        build_analyst_section(analyst),
        "",
        "## 七、图表",
        "![收入与净利润](charts/revenue_net_income.png)",
        "![利润率趋势](charts/margin_trends.png)",
        "![ROE与ROA](charts/roe_roa.png)",
        "![负债权益比](charts/debt_to_equity.png)",
        "![股价历史](charts/price_history.png)",
        "",
        "## 八、投资建议",
        build_investment_section(analysis, valuation, analyst),
        "",
    ]

    # Add data quality section if available
    dq_section = build_data_quality_section(analysis)
    if dq_section:
        report_lines.extend(["", dq_section])

    return "\n".join(report_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report from analysis outputs"
    )
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--valuation", required=False)
    parser.add_argument("--analyst", required=False)
    parser.add_argument("--output", default="./output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.analysis, encoding="utf-8") as handle:
        analysis = json.load(handle)

    valuation = {}
    if args.valuation:
        with open(args.valuation, encoding="utf-8") as handle:
            valuation = json.load(handle)

    analyst = {}
    if args.analyst:
        with open(args.analyst, encoding="utf-8") as handle:
            analyst = json.load(handle)

    report = build_report(analysis, valuation, analyst)
    output_path = f"{args.output}/{analysis['symbol'].replace('.', '_')}_report.md"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    logger.info(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
