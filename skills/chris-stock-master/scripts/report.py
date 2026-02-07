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


def to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def format_compact_number(value: Any) -> str:
    numeric = to_number(value)
    if numeric is None:
        return "-"
    magnitude = abs(numeric)
    for threshold, suffix in (
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    ):
        if magnitude >= threshold:
            return f"{numeric / threshold:,.2f}{suffix}"
    return f"{numeric:,.2f}"


def format_compact_currency(value: Any, currency: str | None) -> str:
    formatted = format_compact_number(value)
    if formatted == "-":
        return "-"
    return f"{formatted} {currency}".strip() if currency else formatted


def emphasize(text: str) -> str:
    if not text or text == "-":
        return "-"
    return f"**{text}**"


def format_growth_phrase(value: Any) -> str:
    numeric = to_number(value)
    if numeric is None:
        return "-"
    direction = "同比增长" if numeric >= 0 else "同比下降"
    return f"{direction}{abs(numeric) * 100:.2f}%"


def latest_series_point(series_map: dict[str, Any]) -> tuple[str | None, float | None]:
    series = series_from_dict(series_map)
    rows = series_rows(series)
    if not rows:
        return None, None
    date, value = rows[-1]
    return date.strftime("%Y-%m-%d"), value


def build_milestone_note(
    series_map: dict[str, Any], latest_value: float | None, currency: str | None
) -> str | None:
    if latest_value is None:
        return None
    values = [to_number(value) for value in series_map.values()]
    values = [value for value in values if value is not None]
    if not values:
        return None
    if latest_value < max(values):
        return None
    thresholds = [1e10, 5e10, 1e11, 2e11, 3e11, 5e11, 1e12, 2e12]
    passed = [threshold for threshold in thresholds if latest_value >= threshold]
    if not passed:
        return None
    milestone = format_compact_currency(passed[-1], currency)
    return f"首次突破 {milestone}"


def build_financial_table(analysis: dict[str, Any]) -> str:
    revenue = series_from_dict(analysis.get("financials", {}).get("revenue", {}))
    net_income = series_from_dict(analysis.get("financials", {}).get("net_income", {}))
    gross_margin = series_from_dict(analysis.get("ratios", {}).get("gross_margin", {}))
    net_margin = series_from_dict(analysis.get("ratios", {}).get("net_margin", {}))

    # Try TTM ROE/ROA first, then annual ratios as fallback
    roe_series = analysis.get("ratios_ttm", {}).get("roe", {})
    if not roe_series or len(roe_series) == 0:
        roe_series = analysis.get("ratios", {}).get("roe", {})
    roa_series = analysis.get("ratios_ttm", {}).get("roa", {})
    if not roa_series or len(roa_series) == 0:
        roa_series = analysis.get("ratios", {}).get("roa", {})

    roe = series_from_dict(roe_series)
    roa = series_from_dict(roa_series)
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
                # If exact date not found for ROE/ROA, use latest available value
                if value is None and label in {"ROE", "ROA"} and series_map:
                    # Get the latest value from the series
                    sorted_dates = sorted(series_map.keys())
                    if sorted_dates:
                        value = series_map[sorted_dates[-1]]

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
        ("Forward P/E", metrics.get("forward_pe"), percentiles.get("forward_pe")),
        ("P/S", metrics.get("ps"), percentiles.get("ps")),
        ("P/B", metrics.get("pb"), percentiles.get("pb")),
        ("EV/EBITDA", metrics.get("ev_to_ebitda"), percentiles.get("ev_to_ebitda")),
        ("PEG", metrics.get("peg"), percentiles.get("peg")),
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


def build_chart_references() -> str:
    lines = [
        "### 图表",
        "",
        "![Revenue & Net Income](charts/revenue_net_income.png)",
        "![Margin Trends](charts/margin_trends.png)",
        "![ROE & ROA](charts/roe_roa.png)",
        "![Debt to Equity](charts/debt_to_equity.png)",
        "![Stock Price](charts/price_history.png)",
        "![PEG Ratio](charts/peg_ratio.png)",
    ]
    return "\n".join(lines)


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


def build_core_opinion(
    analysis: dict[str, Any], valuation: dict[str, Any], analyst: dict[str, Any]
) -> str:
    company = analysis.get("company", {})
    company_name = company.get("name") or analysis.get("symbol") or "该公司"

    growth = analysis.get("growth", {})
    revenue_yoy = growth.get("revenue_yoy_quarterly")
    if not isinstance(revenue_yoy, (int, float)):
        revenue_yoy = latest_series_value(growth.get("revenue_yoy", {}))

    net_income_yoy = growth.get("net_income_yoy_quarterly")
    if not isinstance(net_income_yoy, (int, float)):
        net_income_yoy = latest_series_value(growth.get("net_income_yoy", {}))

    trend_source = revenue_yoy if isinstance(revenue_yoy, (int, float)) else net_income_yoy
    trend_word = "稳定"
    if isinstance(trend_source, (int, float)):
        if trend_source >= 0.15:
            trend_word = "加速"
        elif trend_source <= 0.02:
            trend_word = "放缓"
        else:
            trend_word = "稳定"

    sentences: list[str] = []
    if isinstance(revenue_yoy, (int, float)):
        sentences.append(
            f"{company_name} 最新收入同比 {format_percent(revenue_yoy)}，增长趋势{trend_word}。"
        )
    elif isinstance(net_income_yoy, (int, float)):
        sentences.append(
            f"{company_name} 最新净利润同比 {format_percent(net_income_yoy)}，增长趋势{trend_word}。"
        )
    else:
        sentences.append(f"{company_name} 近期业务增长趋势{trend_word}，核心业务动能保持韧性。")

    pe_pct = valuation.get("percentiles", {}).get("pe")
    if isinstance(pe_pct, (int, float)):
        if pe_pct <= 30:
            valuation_status = "被低估"
        elif pe_pct >= 70:
            valuation_status = "偏高"
        else:
            valuation_status = "合理"
        sentences.append(
            f"估值方面，P/E 分位约 {pe_pct:.0f}% ，整体{valuation_status}。"
        )

    return "\n".join(sentences) if sentences else "暂无核心观点，建议补充增长与估值数据。"


def build_core_summary(
    analysis: dict[str, Any], valuation: dict[str, Any], analyst: dict[str, Any]
) -> str:
    return build_core_opinion(analysis, valuation, analyst)


def build_financial_highlights(
    analysis: dict[str, Any], valuation: dict[str, Any]
) -> str:
    company = analysis.get("company", {})
    currency = company.get("financial_currency") or company.get("currency")
    financials = analysis.get("financials", {})
    financials_q = analysis.get("financials_quarterly", {})
    growth = analysis.get("growth", {})

    revenue_annual = latest_series_value(financials.get("revenue", {}))
    net_income_annual = latest_series_value(financials.get("net_income", {}))
    revenue_quarter = latest_series_value(financials_q.get("revenue", {}))
    net_income_quarter = latest_series_value(financials_q.get("net_income", {}))

    revenue_yoy = latest_series_value(growth.get("revenue_yoy", {}))
    net_income_yoy = latest_series_value(growth.get("net_income_yoy", {}))
    revenue_yoy_quarter = growth.get("revenue_yoy_quarterly")
    net_income_yoy_quarter = growth.get("net_income_yoy_quarterly")

    lines: list[str] = []
    lines.append("最新财报关键指标")
    lines.append("")

    revenue_parts: list[str] = []
    if revenue_annual is not None:
        annual_text = f"最新年度收入 {emphasize(format_compact_currency(revenue_annual, currency))}"
        if isinstance(revenue_yoy, (int, float)):
            annual_text += f"，{format_growth_phrase(revenue_yoy)}"
        milestone_note = build_milestone_note(
            financials.get("revenue", {}), revenue_annual, currency
        )
        if milestone_note:
            annual_text += f"（{milestone_note}）"
        revenue_parts.append(annual_text)
    if revenue_quarter is not None:
        quarter_text = (
            f"最新季度收入 {emphasize(format_compact_currency(revenue_quarter, currency))}"
        )
        if isinstance(revenue_yoy_quarter, (int, float)):
            quarter_text += f"，{format_growth_phrase(revenue_yoy_quarter)}"
        revenue_parts.append(quarter_text)
    if revenue_parts:
        lines.append(f"- **收入规模突破**: " + "；".join(revenue_parts))

    profit_parts: list[str] = []
    if net_income_annual is not None:
        annual_text = (
            f"最新年度净利润 {emphasize(format_compact_currency(net_income_annual, currency))}"
        )
        if isinstance(net_income_yoy, (int, float)):
            annual_text += f"，{format_growth_phrase(net_income_yoy)}"
        profit_parts.append(annual_text)
    if net_income_quarter is not None:
        quarter_text = (
            f"最新季度净利润 {emphasize(format_compact_currency(net_income_quarter, currency))}"
        )
        if isinstance(net_income_yoy_quarter, (int, float)):
            quarter_text += f"，{format_growth_phrase(net_income_yoy_quarter)}"
        profit_parts.append(quarter_text)
    if profit_parts:
        lines.append(f"- **盈利能力提升**: " + "；".join(profit_parts))

    market_cap = valuation.get("current", {}).get("market_cap")
    price = valuation.get("current", {}).get("price")
    if market_cap is not None or price is not None:
        market_parts = []
        if market_cap is not None:
            market_parts.append(
                f"市值 {emphasize(format_compact_currency(market_cap, currency))}"
            )
        if price is not None:
            market_parts.append(
                f"股价 {emphasize(format_currency(price, currency))}"
            )
        if market_parts:
            lines.append(f"- **市值与股价**: " + "，".join(market_parts))

    if len(lines) <= 2:
        return "- 暂无可用的财务亮点数据，建议补充财报与估值信息。"
    return "\n".join(lines)


def build_product_research(analysis: dict[str, Any]) -> str:
    company = analysis.get("company", {})
    currency = company.get("financial_currency") or company.get("currency")
    segment = analysis.get("segment", {})
    segment_revenue = segment.get("revenue")

    if isinstance(segment_revenue, dict) and segment_revenue:
        items: list[tuple[str, float]] = []
        for name, value in segment_revenue.items():
            numeric = to_number(value)
            if numeric is None:
                continue
            items.append((str(name), numeric))
        if items:
            items.sort(key=lambda item: item[1], reverse=True)
            total = sum(abs(value) for _, value in items)
            use_ratio = (
                total > 0
                and all(abs(value) <= 1 for _, value in items)
                and total <= 1.2
            )
            use_percent = (
                total > 0
                and all(abs(value) <= 100 for _, value in items)
                and 80 <= total <= 120
            )
            lines = ["核心产品线表现", ""]
            for name, value in items:
                if use_ratio:
                    content = f"收入占比 {value * 100:.2f}%"
                elif use_percent:
                    content = f"收入占比 {value:.2f}%"
                else:
                    content = f"收入 {format_compact_currency(value, currency)}"
                lines.append(f"- **{name}**: {content}")
            return "\n".join(lines)

    summary = clean_text(company.get("summary"))
    if summary:
        sentences = re.split(r"[。.!?]", summary)
        sentences = [s.strip() for s in sentences if s.strip()]
        lines = ["核心产品线表现", ""]
        for sentence in sentences[:3]:
            lines.append(f"- {sentence}")
        return "\n".join(lines)

    return "- 暂无产品线拆分信息，建议补充业务分部披露。"


def build_management_guidance(analysis: dict[str, Any]) -> str:
    expectations = analysis.get("expectations", {})
    company = analysis.get("company", {})
    currency = company.get("financial_currency") or company.get("currency")

    lines: list[str] = []

    revenue_guidance = expectations.get("revenue_guidance")
    earnings_growth = expectations.get("earnings_growth")
    if isinstance(revenue_guidance, (int, float)) or isinstance(
        earnings_growth, (int, float)
    ):
        parts = []
        if isinstance(revenue_guidance, (int, float)):
            parts.append(f"收入指引 {emphasize(format_percent(revenue_guidance))}")
        if isinstance(earnings_growth, (int, float)):
            parts.append(f"盈利增长 {emphasize(format_percent(earnings_growth))}")
        lines.append(f"- **增长指引**: " + "，".join(parts))

    net_margin = latest_series_value(analysis.get("ratios", {}).get("net_margin", {}))
    if isinstance(net_margin, (int, float)):
        lines.append(
            f"- **效率优化**: 净利率约 {emphasize(format_percent(net_margin))}，强调运营效率与成本控制。"
        )

    dividend_rate = company.get("dividend_rate")
    dividend_yield = company.get("dividend_yield")
    payout_ratio = company.get("payout_ratio")
    if any(isinstance(val, (int, float)) for val in [dividend_rate, dividend_yield, payout_ratio]):
        details = []
        if isinstance(dividend_rate, (int, float)):
            details.append(f"每股分红 {emphasize(format_currency(dividend_rate, currency))}")
        if isinstance(dividend_yield, (int, float)):
            details.append(f"股息率 {emphasize(format_percent(dividend_yield))}")
        if isinstance(payout_ratio, (int, float)):
            details.append(f"派息率 {emphasize(format_percent(payout_ratio))}")
        lines.append(f"- **股东回报**: " + "，".join(details))

    if not lines:
        return "- 暂无管理层指引披露，建议关注公司财报或电话会。"
    return "\n".join(lines)


def build_geo_risk_note(analysis: dict[str, Any]) -> str | None:
    geo = analysis.get("segment", {}).get("geo")
    if not isinstance(geo, dict) or not geo:
        return None
    risk_regions = {"china", "taiwan", "hong kong", "singapore", "asia"}
    matched = []
    for region in geo.keys():
        normalized = str(region).strip().lower()
        if any(key in normalized for key in risk_regions):
            matched.append(region)
    if not matched:
        return None
    return (
        "- 区域风险提示: 收入涉及 "
        + "、".join(matched)
        + "，需关注监管与地缘政治影响。"
    )


def summarize_growth(analysis: dict[str, Any]) -> list[str]:
    growth = analysis.get("growth", {})
    revenue_cagr = growth.get("revenue_cagr")
    net_income_cagr = growth.get("net_income_cagr")
    revenue_yoy = growth.get("revenue_yoy_quarterly")
    net_income_yoy = growth.get("net_income_yoy_quarterly")
    lines = []
    if isinstance(revenue_cagr, (int, float)):
        lines.append(f"- 收入 CAGR: {revenue_cagr * 100:.2f}%")
    if isinstance(net_income_cagr, (int, float)):
        lines.append(f"- 净利润 CAGR: {net_income_cagr * 100:.2f}%")
    if isinstance(revenue_yoy, (int, float)):
        lines.append(f"- 收入 YoY(季度): {revenue_yoy * 100:.2f}%")
    if isinstance(net_income_yoy, (int, float)):
        lines.append(f"- 净利润 YoY(季度): {net_income_yoy * 100:.2f}%")
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
    if gross_margin is not None and net_margin is not None:
        gap = (gross_margin - net_margin) * 100
        lines.append(f"- 毛利/净利差: {gap:.2f}%")
    return lines


def summarize_cashflow(analysis: dict[str, Any]) -> list[str]:
    financials_ttm = analysis.get("financials_ttm", {})
    fcf = latest_series_value(financials_ttm.get("free_cash_flow", {}))
    if fcf is None:
        return []
    return [f"- 最新自由现金流(TTM): {format_number(fcf)}"]


def summarize_rnd(analysis: dict[str, Any]) -> list[str]:
    rnd_ratio = analysis.get("research_and_development", {}).get("ratio")
    if isinstance(rnd_ratio, (int, float)):
        return [f"- 研发投入比: {rnd_ratio * 100:.2f}%"]
    return []


def summarize_balance_sheet(analysis: dict[str, Any]) -> list[str]:
    ratios = analysis.get("ratios", {})
    debt_ratio = latest_series_value(ratios.get("debt_to_equity", {}))
    lines = []
    if debt_ratio is not None:
        lines.append(f"- 负债权益比: {debt_ratio:.2f}")
    return lines


def build_growth_table(growth_map: dict[str, Any], title: str) -> str:
    if not growth_map:
        return ""
    items = list(growth_map.items())
    if not items:
        return ""
    rows = sorted(items, key=lambda item: item[0])[-4:]
    table = [f"### {title}", "", "| 季度 | YoY |", "| --- | --- |"]
    for date_key, value in rows:
        table.append(f"| {date_key} | {format_percent(value)} |")
    return "\n".join(table)


def build_segment_table(analysis: dict[str, Any], data_key: str, title: str) -> str:
    """Build revenue segment breakdown table with graceful degradation."""
    segment = analysis.get("segment", {})
    segment_data = segment.get(data_key)
    if not isinstance(segment_data, dict) or not segment_data:
        # Graceful degradation: provide alternative information
        company = analysis.get("company", {})
        company_website = company.get("website")
        company_name = company.get("name", "该公司")

        fallback = [
            f"### {title}",
            "",
            f"*注：{company_name}未公开详细的业务板块收入数据。*",
            "",
        ]

        # If we have business summary, extract product/service info
        summary = company.get("summary", "")
        if summary and len(summary) > 100:
            # Extract first few sentences as business description
            sentences = summary.split(". ")[:3]
            if sentences:
                fallback.append("根据公司业务描述，主要业务领域包括：")
                fallback.append("")
                # Try to extract business lines from summary
                # This is a simple heuristic - could be improved
                for sentence in sentences[:2]:
                    if len(sentence) > 20:
                        fallback.append(f"- {sentence.strip()}")
                fallback.append("")

        if company_website:
            fallback.append(
                f"详细收入拆分请参考公司官方投资者关系页面：[{company_website}]({company_website})"
            )
        else:
            fallback.append("建议查阅公司官方财报获取详细收入拆分信息。")

        return "\n".join(fallback)

    table = [f"### {title}", "", "| 项目 | 收入占比 |", "| --- | --- |"]
    for name, value in segment_data.items():
        numeric = to_number(value)
        if numeric is None:
            continue
        if numeric > 1:
            pct = numeric / 100
        else:
            pct = numeric
        table.append(f"| {name} | {pct * 100:.2f}% |")
    return "\n".join(table)


def build_expectations_section(analysis: dict[str, Any]) -> str:
    expectations = analysis.get("expectations", {})
    if not expectations:
        return ""
    lines = ["## 九、前瞻与催化剂"]
    next_earnings = expectations.get("next_earnings_date")
    if next_earnings:
        lines.append(f"- 下一次财报日期: {next_earnings}")
    revenue_qoq = expectations.get("revenue_growth_qoq")
    revenue_yoy = expectations.get("revenue_growth_yoy")
    net_income_qoq = expectations.get("net_income_growth_qoq")
    net_income_yoy = expectations.get("net_income_growth_yoy")
    revenue_guidance = expectations.get("revenue_guidance")
    earnings_growth = expectations.get("earnings_growth")
    earnings_quarterly_growth = expectations.get("earnings_quarterly_growth")
    if any(isinstance(val, (int, float)) for val in [revenue_qoq, revenue_yoy]):
        lines.append(
            "- 收入增速: "
            + ", ".join(
                [
                    f"QoQ {format_percent(revenue_qoq)}"
                    if isinstance(revenue_qoq, (int, float))
                    else "QoQ -",
                    f"YoY {format_percent(revenue_yoy)}"
                    if isinstance(revenue_yoy, (int, float))
                    else "YoY -",
                ]
            )
        )
    if any(isinstance(val, (int, float)) for val in [net_income_qoq, net_income_yoy]):
        lines.append(
            "- 利润增速: "
            + ", ".join(
                [
                    f"QoQ {format_percent(net_income_qoq)}"
                    if isinstance(net_income_qoq, (int, float))
                    else "QoQ -",
                    f"YoY {format_percent(net_income_yoy)}"
                    if isinstance(net_income_yoy, (int, float))
                    else "YoY -",
                ]
            )
        )
    if any(
        isinstance(val, (int, float))
        for val in [revenue_guidance, earnings_growth, earnings_quarterly_growth]
    ):
        lines.append(
            "- 指引/预期: "
            + ", ".join(
                [
                    f"收入指引 {format_percent(revenue_guidance)}"
                    if isinstance(revenue_guidance, (int, float))
                    else "收入指引 -",
                    f"盈利增长 {format_percent(earnings_growth)}"
                    if isinstance(earnings_growth, (int, float))
                    else "盈利增长 -",
                    f"季度盈利增长 {format_percent(earnings_quarterly_growth)}"
                    if isinstance(earnings_quarterly_growth, (int, float))
                    else "季度盈利增长 -",
                ]
            )
        )
    business_notes = expectations.get("business_highlights")
    if isinstance(business_notes, list) and business_notes:
        lines.append("- 产品/业务进展: " + " / ".join(business_notes))
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def build_business_model_section(analysis: dict[str, Any]) -> str:
    company = analysis.get("company", {})
    summary = clean_text(company.get("summary"))
    lines = []
    if summary:
        lines.append(f"- 业务概述（原文）: {summary}")
        lines.append("- *注：以上为数据源原文，如需中文请自行翻译*")
    if company.get("industry"):
        lines.append(f"- 行业定位: {company.get('industry')}")
    if company.get("sector"):
        lines.append(f"- 领域定位: {company.get('sector')}")

    segment_business = build_segment_table(analysis, "revenue", "业务收入结构")
    # Always display segment section (will show graceful degradation if data not available)
    lines.append("")
    lines.append(segment_business)

    metrics_lines = []
    metrics_lines.extend(summarize_growth(analysis))
    metrics_lines.extend(summarize_profitability(analysis))
    metrics_lines.extend(summarize_cashflow(analysis))
    metrics_lines.extend(summarize_rnd(analysis))

    if metrics_lines:
        lines.append("- 经营特征:")
        lines.extend(metrics_lines)

    if not lines:
        return "- 暂无业务模式解读，建议补充官方年报或公司介绍。"
    return "\n".join(lines)


def build_competitive_insights(
    analysis: dict[str, Any], peers: list[dict[str, Any]]
) -> str:
    """Build competitive analysis insights explaining margins and competitive position."""
    if not peers:
        return ""

    lines = ["### 竞争力分析", ""]

    # Get company metrics
    company_name = analysis.get("company", {}).get("name", "本公司")
    ratios = analysis.get("ratios", {})
    latest_gross_margin = None
    latest_net_margin = None
    if ratios.get("gross_margin"):
        margins = list(ratios["gross_margin"].values())
        latest_gross_margin = margins[-1] if margins else None
    if ratios.get("net_margin"):
        margins = list(ratios["net_margin"].values())
        latest_net_margin = margins[-1] if margins else None

    # Calculate peer averages
    peer_gross_margins = [
        p.get("gross_margin")
        for p in peers
        if p.get("gross_margin") is not None and p.get("name") != company_name
    ]
    peer_net_margins = [
        p.get("net_margin")
        for p in peers
        if p.get("net_margin") is not None and p.get("name") != company_name
    ]

    # Margin comparison
    if latest_gross_margin is not None and peer_gross_margins:
        avg_peer_gross = sum(peer_gross_margins) / len(peer_gross_margins)
        diff = (latest_gross_margin - avg_peer_gross) * 100
        comparison = "高于" if diff > 0 else "低于"
        lines.append(
            f"- 毛利率 {latest_gross_margin:.2%} {comparison}同行平均 {avg_peer_gross:.2%} ({abs(diff):.1f}pp)"
        )

    # Explain margin gap if both margins available
    if latest_gross_margin is not None and latest_net_margin is not None:
        margin_gap = (latest_gross_margin - latest_net_margin) * 100
        if margin_gap > 30:  # Significant gap
            lines.append(
                f"- 净利率 {latest_net_margin:.2%} 低于毛利率 {margin_gap:.1f}pp，主要原因可能包括："
            )

            # Check R&D intensity
            r_and_d = analysis.get("research_and_development", {}).get("ratio")
            if r_and_d and r_and_d > 0.15:  # High R&D >15%
                lines.append(
                    f"  * 高额研发投入 {r_and_d:.2%}（维持技术竞争力的必要成本）"
                )

            # Check debt level
            debt_to_equity = None
            if ratios.get("debt_to_equity"):
                debt_values = list(ratios["debt_to_equity"].values())
                debt_to_equity = debt_values[-1] if debt_values else None

            if debt_to_equity is not None and debt_to_equity > 0.5:
                lines.append(
                    f"  * 较高财务杠杆（负债权益比 {debt_to_equity:.2f}）产生的利息支出"
                )
            elif debt_to_equity is not None and debt_to_equity < 0.3:
                lines.append(
                    f"  * 低财务杠杆（负债权益比 {debt_to_equity:.2f}）表明财务结构稳健"
                )

            # Industry-specific factors
            industry = analysis.get("company", {}).get("industry", "")
            if "Semiconductor" in industry or "半导体" in industry:
                lines.append("  * 半导体行业特有的高额资本支出和折旧摊销")

    if len(lines) <= 2:  # Only header was added
        return ""

    return "\n".join(lines)


def build_peer_table(peers: list[dict[str, Any]], company_name: str = None) -> str:
    """Build enhanced peer comparison table with more metrics."""
    if not peers:
        return ""
    table = [
        "### 同行对标",
        "",
        "| 公司 | 市值 | 毛利率 | 净利率 | 负债权益比 | P/E |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for peer in peers:
        name = peer.get("name", "-")
        # Highlight the target company with **bold**
        if company_name and name == company_name:
            name = f"**{name}**"

        market_cap = peer.get("market_cap")
        if market_cap and market_cap > 1e9:
            # Format in billions
            market_cap_str = f"${market_cap / 1e9:.1f}B"
        elif market_cap and market_cap > 1e6:
            # Format in millions
            market_cap_str = f"${market_cap / 1e6:.1f}M"
        else:
            market_cap_str = format_number(market_cap) if market_cap else "-"

        table.append(
            "| "
            + name
            + " | "
            + market_cap_str
            + " | "
            + format_percent(peer.get("gross_margin"))
            + " | "
            + format_percent(peer.get("net_margin"))
            + " | "
            + format_number(peer.get("debt_to_equity"))
            + " | "
            + format_number(peer.get("pe"))
            + " |"
        )
    return "\n".join(table)


def build_competitive_section(analysis: dict[str, Any]) -> str:
    company = analysis.get("company", {})
    company_name = company.get("name") or "本公司"
    industry = company.get("industry") or company.get("sector") or "所在行业"

    peers = analysis.get("peers", [])
    peers_list = [peer for peer in peers if isinstance(peer, dict)]

    lines = ["### 主要竞争对手分析"]
    competitor_lines = []
    if peers_list:
        peers_sorted = sorted(
            peers_list, key=lambda peer: peer.get("market_cap") or 0, reverse=True
        )
        for peer in peers_sorted[:6]:
            name = peer.get("name") or "-"
            if name == company_name:
                continue
            parts = []
            market_cap = peer.get("market_cap")
            if isinstance(market_cap, (int, float)):
                parts.append(f"市值 {format_compact_currency(market_cap, company.get('currency'))}")
            gross_margin = peer.get("gross_margin")
            if isinstance(gross_margin, (int, float)):
                parts.append(f"毛利率 {format_percent(gross_margin)}")
            net_margin = peer.get("net_margin")
            if isinstance(net_margin, (int, float)):
                parts.append(f"净利率 {format_percent(net_margin)}")
            if not parts:
                parts.append(f"在{industry}形成直接竞争")
            competitor_lines.append(f"- **{name}**: " + "，".join(parts))

    if not competitor_lines:
        competitor_lines.append(
            f"- **行业竞争**: {industry} 竞争者众多，主要围绕产品差异化与成本效率展开。"
        )

    lines.extend(competitor_lines)
    lines.append("")
    lines.append("### 竞争地位与策略")

    ratios = analysis.get("ratios", {})
    latest_gross_margin = latest_series_value(ratios.get("gross_margin", {}))
    latest_net_margin = latest_series_value(ratios.get("net_margin", {}))

    peer_gross_margins = [
        peer.get("gross_margin")
        for peer in peers_list
        if isinstance(peer.get("gross_margin"), (int, float))
        and peer.get("name") != company_name
    ]

    advantage_parts = []
    if isinstance(latest_gross_margin, (int, float)) and peer_gross_margins:
        avg_peer_gross = sum(peer_gross_margins) / len(peer_gross_margins)
        diff = latest_gross_margin - avg_peer_gross
        if diff >= 0.05:
            advantage_parts.append("毛利率领先同行，具备产品溢价或规模优势")
        elif diff <= -0.05:
            advantage_parts.append("毛利率弱于同行，盈利质量承压")
        else:
            advantage_parts.append("毛利率与同行接近，竞争格局较为均衡")
    elif isinstance(latest_gross_margin, (int, float)):
        advantage_parts.append(f"毛利率约 {format_percent(latest_gross_margin)}，盈利质量保持稳定")

    if isinstance(latest_net_margin, (int, float)):
        advantage_parts.append(f"净利率约 {format_percent(latest_net_margin)}")

    if not advantage_parts:
        advantage_parts.append(f"在{industry}内具备一定规模与品牌优势")

    lines.append(f"- **优势**: " + "；".join(advantage_parts))

    strategy_parts = []
    r_and_d_ratio = analysis.get("research_and_development", {}).get("ratio")
    if isinstance(r_and_d_ratio, (int, float)):
        if r_and_d_ratio >= 0.1:
            strategy_parts.append(
                f"维持高研发投入（{format_percent(r_and_d_ratio)}）强化技术壁垒"
            )
        elif r_and_d_ratio >= 0.05:
            strategy_parts.append(
                f"持续研发投入（{format_percent(r_and_d_ratio)}）推动产品迭代"
            )

    debt_to_equity = latest_series_value(ratios.get("debt_to_equity", {}))
    if isinstance(debt_to_equity, (int, float)):
        if debt_to_equity <= 0.3:
            strategy_parts.append("财务结构稳健，具备扩张与投资空间")
        elif debt_to_equity >= 0.8:
            strategy_parts.append("关注杠杆水平，强调资本效率与现金流管理")

    if not strategy_parts:
        strategy_parts.append("通过产品结构优化与成本控制提升竞争力")

    lines.append(f"- **策略**: " + "；".join(strategy_parts))

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
        forward_pe = metrics.get("forward_pe")
        ps = metrics.get("ps")
        pb = metrics.get("pb")
        peg = metrics.get("peg")
        lines.append(
            "- 估值指标: "
            + ", ".join(
                [
                    f"P/E {format_number(pe)} ({format_number(percentiles.get('pe'))}%)",
                    f"Forward P/E {format_number(forward_pe)}",
                    f"PEG {format_number(peg)}",
                    f"P/S {format_number(ps)} ({format_number(percentiles.get('ps'))}%)",
                    f"P/B {format_number(pb)} ({format_number(percentiles.get('pb'))}%)",
                ]
            )
        )

        # Add valuation interpretation
        valuation_insights = []
        if peg is not None and pe is not None:
            if peg < 1:
                valuation_insights.append(
                    f"PEG {format_number(peg)} < 1.0 表明当前估值相对增长预期偏低，可能被低估"
                )
            elif peg < 2:
                valuation_insights.append(
                    f"PEG {format_number(peg)} < 2.0 表明增长预期可支撑当前估值水平"
                )
            else:
                valuation_insights.append(
                    f"PEG {format_number(peg)} > 2.0 表明估值相对增长预期偏高，需谨慎"
                )

        if forward_pe is not None and pe is not None and forward_pe > 0:
            implied_growth = (pe / forward_pe - 1) * 100
            if implied_growth > 0:
                valuation_insights.append(
                    f"Forward P/E {format_number(forward_pe)} 隐含市场预期明年盈利增长 {format_number(implied_growth)}%"
                )

        if valuation_insights:
            lines.append("- 估值合理性分析:")
            for insight in valuation_insights:
                lines.append(f"  * {insight}")

    fundamentals = []
    fundamentals.extend(summarize_growth(analysis))
    fundamentals.extend(summarize_profitability(analysis))
    fundamentals.extend(summarize_balance_sheet(analysis))
    fundamentals.extend(summarize_rnd(analysis))
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
                    lines.append(f"- {warning.get('message', '未知警告')}")
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

    report_lines: list[str] = [
        f"# 财报分析报告 - {company.get('name') or symbol}",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
        f"数据更新时间: {data_fetched_at or '-'}",
        "",
        "## 1. 核心观点",
        build_core_opinion(analysis, valuation, analyst),
        "",
        "## 2. 财务亮点 (Financial Highlight)",
        build_financial_highlights(analysis, valuation),
        "",
        "## 3. 产品研究 (Product Research)",
        build_product_research(analysis),
        "",
        "## 4. 竞争格局 (Competitive Landscape)",
        build_competitive_section(analysis),
        "",
        "## 5. 管理层指引 (Management Guidance)",
        build_management_guidance(analysis),
        "",
        "## 6. 估值分析",
        build_valuation_table(valuation),
        build_currency_note(valuation),
        "",
        build_chart_references(),
        "",
        "## 7. 投资建议",
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
