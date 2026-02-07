#!/usr/bin/env python3
"""Generate a markdown report from analysis outputs."""

import argparse
import json
import logging
import re
from datetime import datetime
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


def normalize_ratio_value(value: Any, aggressive_small_percent: bool = False) -> float | None:
    """Normalize ratio-like values that may come as 0-1 or 0-100."""
    numeric = to_number(value)
    if numeric is None:
        return None
    if 1 < abs(numeric) <= 100:
        return numeric / 100
    if aggressive_small_percent and 0.2 < abs(numeric) <= 1:
        return numeric / 100
    return numeric


def format_ratio_percent(value: Any, aggressive_small_percent: bool = False) -> str:
    normalized = normalize_ratio_value(
        value, aggressive_small_percent=aggressive_small_percent
    )
    return format_percent(normalized) if normalized is not None else "-"


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


def format_analysis_date(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return "-"
    normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return text
    return f"{dt.year}年{dt.month}月{dt.day}日"


def parse_date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def quarter_tag(date_key: str) -> str:
    parsed = parse_date(date_key)
    if parsed is None:
        return date_key
    quarter = (parsed.month - 1) // 3 + 1
    return f"{parsed.year}Q{quarter}"


def latest_series_items(
    series_map: dict[str, Any], limit: int = 3
) -> list[tuple[str, float]]:
    if not isinstance(series_map, dict):
        return []
    items: list[tuple[str, float]] = []
    for date_key, raw_value in series_map.items():
        numeric = to_number(raw_value)
        if numeric is None:
            continue
        items.append((str(date_key), numeric))
    items.sort(key=lambda item: item[0])
    if limit <= 0:
        return items
    return items[-limit:]


def format_growth_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:+.2f}%"


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

    filtered_rows = [row for row in rows if row[1] is not None or row[2] is not None]
    if not filtered_rows:
        return "暂无可用估值指标（估值快照不足，建议补充更长历史价格与财务数据）。"

    percentile_label = build_percentile_label(valuation)
    table = [f"| 指标 | 当前值 | {percentile_label} |", "| --- | --- | --- |"]
    for label, value, pct in filtered_rows:
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


def build_chart_references(analysis: dict[str, Any]) -> str:
    chart_paths = analysis.get("charts")
    if not isinstance(chart_paths, list) or not chart_paths:
        return ""

    lines = ["### 图表", ""]
    for path in chart_paths:
        if not isinstance(path, str) or not path.strip():
            continue
        label = path.split("/")[-1].replace("_", " ").replace(".png", "")
        lines.append(f"![{label}]({path})")
    return "\n".join(lines) if len(lines) > 2 else ""


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


def trend_word_from_growth(growth_value: float | None) -> str:
    if growth_value is None:
        return "稳定"
    if growth_value >= 0.15:
        return "加速"
    if growth_value >= 0.03:
        return "稳定"
    if growth_value >= 0:
        return "放缓"
    return "承压"


def valuation_status_from_percentile(percentile: float | None) -> str | None:
    if percentile is None:
        return None
    if percentile <= 30:
        return "被低估"
    if percentile >= 70:
        return "偏高"
    return "合理"


def normalize_segment_revenue(segment_revenue: Any) -> dict[str, float]:
    """Normalize segment revenue to a simple name->value mapping."""
    if not isinstance(segment_revenue, dict) or not segment_revenue:
        return {}

    direct_values: dict[str, float] = {}
    for name, raw_value in segment_revenue.items():
        numeric = to_number(raw_value)
        if numeric is not None:
            direct_values[str(name)] = numeric
    if direct_values:
        return direct_values

    snapshots: list[tuple[str, dict[str, float]]] = []
    for period, payload in segment_revenue.items():
        if not isinstance(payload, dict):
            continue
        period_values: dict[str, float] = {}
        for name, raw_value in payload.items():
            numeric = to_number(raw_value)
            if numeric is not None:
                period_values[str(name)] = numeric
        if period_values:
            snapshots.append((str(period), period_values))

    if not snapshots:
        return {}
    snapshots.sort(key=lambda item: item[0])
    return snapshots[-1][1]


def split_summary_points(summary: str, company_name: str | None = None) -> list[str]:
    if not summary:
        return []

    text = clean_text(summary)
    if not text:
        return []

    # Keep common company abbreviations from being treated as sentence boundaries.
    text = re.sub(r"\b(Inc|Ltd|Corp|Co|LLC|plc)\.", r"\1", text)
    text = text.replace("e.g.", "eg").replace("i.e.", "ie")

    parts = re.split(r"[；;。！？!?]|(?<=\w)\.\s+(?=[A-Z])", text)
    if len(parts) <= 1:
        parts = re.split(r",\s+(?:and|while|with|including)\s+", text)

    points: list[str] = []
    for part in parts:
        candidate = clean_text(part).rstrip(".")
        if not candidate or len(candidate) < 18:
            continue
        if company_name and candidate.lower() == company_name.strip().lower():
            continue
        points.append(candidate)

    deduped: list[str] = []
    seen: set[str] = set()
    for point in points:
        key = point.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
        if len(deduped) >= 3:
            break
    return deduped


def extract_segment_names_from_summary(summary: str) -> list[str]:
    text = clean_text(summary)
    if not text:
        return []

    segments: list[str] = []
    through_match = re.search(r"operates through (.+?) segments?", text, re.IGNORECASE)
    if through_match:
        raw = through_match.group(1)
        pieces = re.split(r",| and ", raw)
        for piece in pieces:
            normalized = clean_text(piece)
            normalized = re.sub(r"\b(the|its)\b", "", normalized, flags=re.IGNORECASE)
            normalized = re.sub(
                r"\bsegment\b", "", normalized, flags=re.IGNORECASE
            ).strip(" ,.;")
            if normalized:
                segments.append(normalized)

    if not segments:
        for match in re.findall(r"([A-Z][A-Za-z0-9&/\- ]+?) segment", text):
            normalized = clean_text(match).strip(" ,.;")
            if normalized:
                segments.append(normalized)

    deduped: list[str] = []
    seen: set[str] = set()
    for name in segments:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped[:5]


def describe_segment_focus(segment_name: str) -> str:
    lowered = segment_name.lower()
    if "cloud" in lowered or "infrastructure" in lowered:
        return "通常对应企业服务与算力相关业务，受数字化与AI需求变化影响较大。"
    if "service" in lowered or "platform" in lowered:
        return "通常覆盖核心用户入口，并承担主要商业化场景。"
    if "ads" in lowered or "advert" in lowered:
        return "主要关注广告投放效率与客户预算份额。"
    if "device" in lowered or "hardware" in lowered:
        return "以软硬件协同为主，影响生态黏性与终端触达。"
    if "bet" in lowered or "venture" in lowered:
        return "通常承担前沿方向探索，短期对利润贡献有限。"
    return "是公司业务组合中的重要板块，对增长质量与盈利结构有影响。"


def infer_product_lines_from_summary(summary: str) -> list[str]:
    segment_names = extract_segment_names_from_summary(summary)
    if not segment_names:
        return []
    lines: list[str] = []
    for name in segment_names:
        lines.append(f"- **{name}**: {describe_segment_focus(name)}")
    return lines


def extract_focus_tags(summary: str) -> list[str]:
    text = clean_text(summary).lower()
    if not text:
        return []

    tag_rules = [
        (("search", "ads", "advertising"), "广告与流量分发"),
        (("cloud", "infrastructure"), "云与基础设施能力"),
        (("ai", "machine learning", "gemini"), "AI 产品化"),
        (("device", "hardware", "android"), "终端与生态协同"),
        (("subscription", "music", "premium", "tv"), "订阅与平台服务"),
        (("enterprise", "workspace", "security"), "企业服务"),
    ]

    tags: list[str] = []
    for keywords, tag in tag_rules:
        if any(keyword in text for keyword in keywords):
            tags.append(tag)

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped[:3]


def build_core_opinion(
    analysis: dict[str, Any], valuation: dict[str, Any], analyst: dict[str, Any]
) -> str:
    company = analysis.get("company", {})
    company_name = company.get("name") or analysis.get("symbol") or "该公司"

    growth = analysis.get("growth", {})
    expectations = analysis.get("expectations", {})
    revenue_yoy = growth.get("revenue_yoy_quarterly")
    if not isinstance(revenue_yoy, (int, float)):
        revenue_yoy = latest_series_value(growth.get("revenue_yoy", {}))
    revenue_qoq = expectations.get("revenue_growth_qoq")
    net_income_yoy = growth.get("net_income_yoy_quarterly")
    if not isinstance(net_income_yoy, (int, float)):
        net_income_yoy = latest_series_value(growth.get("net_income_yoy", {}))
    net_income_qoq = expectations.get("net_income_growth_qoq")
    earnings_growth = expectations.get("earnings_growth")

    trend_source = None
    for candidate in [revenue_yoy, net_income_yoy, revenue_qoq, net_income_qoq]:
        if isinstance(candidate, (int, float)):
            trend_source = float(candidate)
            break
    trend_word = trend_word_from_growth(trend_source)

    sentences: list[str] = []
    if isinstance(revenue_yoy, (int, float)):
        sentences.append(
            f"{company_name} 最新收入同比 {format_percent(revenue_yoy)}，增长趋势{trend_word}。"
        )
    elif isinstance(revenue_qoq, (int, float)):
        sentences.append(
            f"{company_name} 最近季度收入环比 {format_percent(revenue_qoq)}，短期经营变化{trend_word}。"
        )
    elif isinstance(net_income_yoy, (int, float)):
        sentences.append(
            f"{company_name} 最新净利润同比 {format_percent(net_income_yoy)}，增长趋势{trend_word}。"
        )
    elif isinstance(net_income_qoq, (int, float)):
        sentences.append(
            f"{company_name} 最近季度净利润环比 {format_percent(net_income_qoq)}，盈利变化{trend_word}。"
        )
    else:
        sentences.append(f"{company_name} 当前可得公开数据对增长判断有限，短期趋势暂按{trend_word}处理。")

    percentiles = valuation.get("percentiles", {})
    metrics = valuation.get("metrics", {})
    pe_pct = percentiles.get("pe")
    pb_pct = percentiles.get("pb")
    pe_status = valuation_status_from_percentile(pe_pct if isinstance(pe_pct, (int, float)) else None)
    pb_status = valuation_status_from_percentile(pb_pct if isinstance(pb_pct, (int, float)) else None)
    if isinstance(pe_pct, (int, float)) and pe_status:
        sentences.append(f"估值方面，P/E 分位约 {pe_pct:.0f}%，整体{pe_status}。")
    elif isinstance(pb_pct, (int, float)) and pb_status:
        sentences.append(f"估值方面，P/B 分位约 {pb_pct:.0f}%，资产端估值处于{pb_status}分位。")
    elif isinstance(metrics.get("forward_pe"), (int, float)):
        sentences.append(
            f"估值方面，Forward P/E 约 {format_number(metrics.get('forward_pe'))} 倍，需结合后续盈利兑现节奏观察。"
        )

    free_cash_flow = latest_series_value(
        analysis.get("financials_ttm", {}).get("free_cash_flow", {})
    )
    debt_to_equity = latest_series_value(
        analysis.get("ratios", {}).get("debt_to_equity", {})
    )
    if free_cash_flow is not None:
        if free_cash_flow > 0 and (debt_to_equity is None or debt_to_equity <= 0.6):
            sentences.append("现金流与资产负债表目前保持稳健，短期财务韧性相对可控。")
        elif debt_to_equity is not None and debt_to_equity > 1:
            sentences.append("杠杆水平偏高，后续需关注盈利兑现与现金流安全边际。")
    elif isinstance(earnings_growth, (int, float)) and earnings_growth > 0.2:
        sentences.append("盈利预期仍为正增长，但需持续验证利润兑现质量。")

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
    eps_ttm = latest_series_value(analysis.get("per_share_ttm", {}).get("eps", {}))
    latest_cash = latest_series_value(analysis.get("balance_quarterly", {}).get("cash", {}))
    latest_fcf = latest_series_value(analysis.get("financials_ttm", {}).get("free_cash_flow", {}))

    revenue_yoy = latest_series_value(growth.get("revenue_yoy", {}))
    net_income_yoy = latest_series_value(growth.get("net_income_yoy", {}))
    revenue_yoy_quarter = growth.get("revenue_yoy_quarterly")
    net_income_yoy_quarter = growth.get("net_income_yoy_quarterly")

    revenue_quarter_items = latest_series_items(financials_q.get("revenue", {}), limit=3)
    net_income_quarter_items = latest_series_items(
        financials_q.get("net_income", {}), limit=3
    )

    latest_quarter_tag = "-"
    if revenue_quarter_items:
        latest_quarter_tag = quarter_tag(revenue_quarter_items[-1][0])
    elif net_income_quarter_items:
        latest_quarter_tag = quarter_tag(net_income_quarter_items[-1][0])

    lines: list[str] = [f"最新财报关键指标（截至{latest_quarter_tag}）", ""]

    revenue_label = "收入表现"
    if isinstance(revenue_yoy_quarter, (int, float)):
        if revenue_yoy_quarter >= 0.08:
            revenue_label = "收入增长"
        elif revenue_yoy_quarter < 0:
            revenue_label = "收入承压"
    elif isinstance(analysis.get("expectations", {}).get("revenue_growth_qoq"), (int, float)):
        if analysis.get("expectations", {}).get("revenue_growth_qoq") >= 0.05:
            revenue_label = "收入增长"

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
        elif isinstance(analysis.get("expectations", {}).get("revenue_growth_qoq"), (int, float)):
            quarter_text += (
                "，环比"
                + format_growth_rate(analysis.get("expectations", {}).get("revenue_growth_qoq"))
            )
        revenue_parts.append(quarter_text)
    if revenue_annual is None and len(revenue_quarter_items) >= 2:
        first_q_date, first_q_value = revenue_quarter_items[0]
        last_q_date, last_q_value = revenue_quarter_items[-1]
        trend = (
            (last_q_value / first_q_value - 1)
            if first_q_value not in (0, None)
            else None
        )
        revenue_parts.append(
            "近"
            + str(len(revenue_quarter_items))
            + "个季度收入由 "
            + f"{quarter_tag(first_q_date)} {format_compact_currency(first_q_value, currency)} "
            + "提升至 "
            + f"{quarter_tag(last_q_date)} {emphasize(format_compact_currency(last_q_value, currency))}"
            + (
                f"（累计{format_growth_rate(trend)}）"
                if isinstance(trend, float)
                else ""
            )
        )
    if not revenue_parts and revenue_quarter_items:
        revenue_parts.append(
            f"最新季度收入 {emphasize(format_compact_currency(revenue_quarter_items[-1][1], currency))}"
        )
    if revenue_parts:
        lines.append(f"- **{revenue_label}**: " + "；".join(revenue_parts))

    profit_label = "盈利表现"
    if isinstance(net_income_yoy_quarter, (int, float)):
        if net_income_yoy_quarter >= 0.08:
            profit_label = "盈利改善"
        elif net_income_yoy_quarter < 0:
            profit_label = "盈利承压"
    elif isinstance(analysis.get("expectations", {}).get("net_income_growth_qoq"), (int, float)):
        if analysis.get("expectations", {}).get("net_income_growth_qoq") >= 0.1:
            profit_label = "盈利改善"

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
        elif isinstance(analysis.get("expectations", {}).get("net_income_growth_qoq"), (int, float)):
            quarter_text += (
                "，环比"
                + format_growth_rate(
                    analysis.get("expectations", {}).get("net_income_growth_qoq")
                )
            )
        profit_parts.append(quarter_text)
    if net_income_annual is None and len(net_income_quarter_items) >= 2:
        first_q_date, first_q_value = net_income_quarter_items[0]
        last_q_date, last_q_value = net_income_quarter_items[-1]
        trend = (
            (last_q_value / first_q_value - 1)
            if first_q_value not in (0, None)
            else None
        )
        profit_parts.append(
            "近"
            + str(len(net_income_quarter_items))
            + "个季度净利润由 "
            + f"{quarter_tag(first_q_date)} {format_compact_currency(first_q_value, currency)} "
            + "变化至 "
            + f"{quarter_tag(last_q_date)} {emphasize(format_compact_currency(last_q_value, currency))}"
            + (
                f"（累计{format_growth_rate(trend)}）"
                if isinstance(trend, float)
                else ""
            )
        )
    if not profit_parts and net_income_quarter_items:
        profit_parts.append(
            f"最新季度净利润 {emphasize(format_compact_currency(net_income_quarter_items[-1][1], currency))}"
        )
    if profit_parts:
        lines.append(f"- **{profit_label}**: " + "；".join(profit_parts))

    cash_parts: list[str] = []
    if latest_cash is not None:
        cash_parts.append(f"现金储备约 {emphasize(format_compact_currency(latest_cash, currency))}")
    if latest_fcf is not None:
        cash_parts.append(
            f"自由现金流(TTM)约 {emphasize(format_compact_currency(latest_fcf, currency))}"
        )
    if eps_ttm is not None:
        cash_parts.append(f"EPS(TTM)约 {emphasize(format_number(eps_ttm))}")
    if cash_parts:
        lines.append(f"- **现金与资本效率**: " + "；".join(cash_parts))

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
    segment_revenue = normalize_segment_revenue(segment.get("revenue"))
    summary_text = clean_text(company.get("summary"))
    inferred_lines = infer_product_lines_from_summary(summary_text)

    lines = ["核心产品线表现", ""]
    if segment_revenue:
        items = sorted(segment_revenue.items(), key=lambda item: item[1], reverse=True)
        total = sum(abs(value) for _, value in items)
        use_ratio = total > 0 and all(abs(value) <= 1 for _, value in items) and total <= 1.2
        use_percent = (
            total > 0
            and all(abs(value) <= 100 for _, value in items)
            and 80 <= total <= 120
        )
        for name, value in items[:6]:
            if use_ratio:
                content = f"收入占比 {value * 100:.2f}%"
            elif use_percent:
                content = f"收入占比 {value:.2f}%"
            else:
                content = f"收入 {format_compact_currency(value, currency)}"
            lines.append(f"- **{name}**: {content}")
    elif inferred_lines:
        lines.extend(inferred_lines[:4])
    else:
        summary_points = split_summary_points(
            summary_text, company.get("name")
        )
        if summary_points:
            for point in summary_points:
                lines.append(f"- {point}")
        else:
            lines.append("- 暂无产品线拆分信息，建议补充业务分部披露。")

    r_and_d_ratio = analysis.get("research_and_development", {}).get("ratio")
    expectations = analysis.get("expectations", {})
    if isinstance(r_and_d_ratio, (int, float)) or expectations:
        lines.extend(["", "研发与新产品", ""])
        if isinstance(r_and_d_ratio, (int, float)):
            lines.append(
                f"- 当前研发投入强度约 {format_percent(r_and_d_ratio)}，建议结合后续收入与利润变化评估投入产出。"
            )

        revenue_growth_qoq = expectations.get("revenue_growth_qoq")
        earnings_growth = expectations.get("earnings_growth")
        if isinstance(revenue_growth_qoq, (int, float)) or isinstance(earnings_growth, (int, float)):
            growth_parts = []
            if isinstance(revenue_growth_qoq, (int, float)):
                growth_parts.append(f"季度收入环比 {format_percent(revenue_growth_qoq)}")
            if isinstance(earnings_growth, (int, float)):
                growth_parts.append(f"盈利预期增速 {format_percent(earnings_growth)}")
            lines.append("- 近期经营变化: " + "，".join(growth_parts))

    return "\n".join(lines)


def build_management_guidance(analysis: dict[str, Any]) -> str:
    expectations = analysis.get("expectations", {})
    company = analysis.get("company", {})
    currency = company.get("financial_currency") or company.get("currency")
    focus_tags = extract_focus_tags(company.get("summary"))

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
        guidance_text = "，".join(parts)
        if focus_tags:
            guidance_text += f"（重点观察：{'、'.join(focus_tags)}）"
        lines.append(f"- **增长指引**: {guidance_text}")

    net_margin = latest_series_value(analysis.get("ratios", {}).get("net_margin", {}))
    if isinstance(net_margin, (int, float)):
        lines.append(
            f"- **效率优化**: 净利率约 {emphasize(format_percent(net_margin))}，强调运营效率与成本控制。"
        )
    else:
        revenue_qoq = expectations.get("revenue_growth_qoq")
        earnings_qoq = expectations.get("net_income_growth_qoq")
        if isinstance(revenue_qoq, (int, float)) or isinstance(earnings_qoq, (int, float)):
            perf_parts = []
            if isinstance(revenue_qoq, (int, float)):
                perf_parts.append(f"收入环比 {format_percent(revenue_qoq)}")
            if isinstance(earnings_qoq, (int, float)):
                perf_parts.append(f"净利润环比 {format_percent(earnings_qoq)}")
            lines.append(
                "- **效率优化**: 公开预期数据显示，近期经营变化为"
                + "，".join(perf_parts)
                + "。"
            )

    next_earnings = expectations.get("next_earnings_date")
    if next_earnings:
        lines.append(f"- **后续观察点**: 下一次财报窗口为 {next_earnings}。")

    dividend_rate = company.get("dividend_rate")
    dividend_yield = company.get("dividend_yield")
    payout_ratio = company.get("payout_ratio")
    if any(
        isinstance(val, (int, float))
        for val in [dividend_rate, dividend_yield, payout_ratio]
    ):
        details = []
        if isinstance(dividend_rate, (int, float)):
            details.append(f"每股分红 {emphasize(format_currency(dividend_rate, currency))}")
        if isinstance(dividend_yield, (int, float)):
            details.append(
                f"股息率 {emphasize(format_ratio_percent(dividend_yield, aggressive_small_percent=True))}"
            )
        if isinstance(payout_ratio, (int, float)):
            details.append(f"派息率 {emphasize(format_ratio_percent(payout_ratio))}")
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
    expectations = analysis.get("expectations", {})
    revenue_cagr = growth.get("revenue_cagr")
    net_income_cagr = growth.get("net_income_cagr")
    revenue_yoy = growth.get("revenue_yoy_quarterly")
    net_income_yoy = growth.get("net_income_yoy_quarterly")
    revenue_qoq = expectations.get("revenue_growth_qoq")
    net_income_qoq = expectations.get("net_income_growth_qoq")
    lines = []
    if isinstance(revenue_cagr, (int, float)):
        lines.append(f"- 收入 CAGR: {revenue_cagr * 100:.2f}%")
    if isinstance(net_income_cagr, (int, float)):
        lines.append(f"- 净利润 CAGR: {net_income_cagr * 100:.2f}%")
    if isinstance(revenue_yoy, (int, float)):
        lines.append(f"- 收入 YoY(季度): {revenue_yoy * 100:.2f}%")
    elif isinstance(revenue_qoq, (int, float)):
        lines.append(f"- 收入 QoQ(季度): {revenue_qoq * 100:.2f}%")
    if isinstance(net_income_yoy, (int, float)):
        lines.append(f"- 净利润 YoY(季度): {net_income_yoy * 100:.2f}%")
    elif isinstance(net_income_qoq, (int, float)):
        lines.append(f"- 净利润 QoQ(季度): {net_income_qoq * 100:.2f}%")
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


def infer_default_competitors(
    industry: str | None, sector: str | None, focus_tags: list[str] | None = None
) -> list[str]:
    industry_text = (industry or "").lower()
    sector_text = (sector or "").lower()
    focus_text = "、".join(focus_tags or [])

    if "internet content" in industry_text or "interactive media" in industry_text:
        return [
            "- **流量入口竞争**: 行业内平台普遍围绕用户时长与分发效率持续竞争。",
            "- **预算分配竞争**: 广告与商业化预算在不同内容生态间动态迁移。",
            (
                "- **技术能力竞争**: 云与AI能力正成为中长期差异化来源。"
                if not focus_text
                else f"- **技术能力竞争**: 公司当前聚焦于{focus_text}，需持续验证差异化优势。"
            ),
        ]

    if (
        "software" in industry_text
        or "technology" in sector_text
        or "cloud" in industry_text
    ):
        return [
            "- **头部平台竞争**: 主要体现在产品迭代速度、客户黏性与定价策略。",
            "- **垂直场景竞争**: 行业化厂商通过细分场景切入，分流通用需求。",
            "- **新进入者竞争**: 开源生态与低成本方案持续压缩行业利润空间。",
        ]

    return [
        "- **规模竞争**: 行业龙头依靠品牌、渠道与资金能力维持市场份额。",
        "- **效率竞争**: 效率型玩家通常通过成本与执行节奏争取盈利空间。",
    ]


def build_competitive_section(analysis: dict[str, Any]) -> str:
    company = analysis.get("company", {})
    company_name = company.get("name") or "本公司"
    industry = company.get("industry") or company.get("sector") or "所在行业"
    sector = company.get("sector")
    focus_tags = extract_focus_tags(company.get("summary"))

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
        inferred = infer_default_competitors(company.get("industry"), sector, focus_tags)
        if inferred:
            competitor_lines.extend(inferred)
        else:
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
        if focus_tags:
            advantage_parts.append(
                f"在{industry}中围绕“{'、'.join(focus_tags)}”形成相对明确的业务定位"
            )
        else:
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
        if focus_tags:
            strategy_parts.append(
                f"后续可重点跟踪“{'、'.join(focus_tags)}”相关投入与商业化效率变化"
            )
        else:
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
        metric_parts: list[str] = []
        if pe is not None:
            pe_pct = percentiles.get("pe")
            if isinstance(pe_pct, (int, float)):
                metric_parts.append(f"P/E {format_number(pe)} ({pe_pct:.2f}%)")
            else:
                metric_parts.append(f"P/E {format_number(pe)}")
        if forward_pe is not None:
            metric_parts.append(f"Forward P/E {format_number(forward_pe)}")
        if peg is not None:
            metric_parts.append(f"PEG {format_number(peg)}")
        if ps is not None:
            ps_pct = percentiles.get("ps")
            if isinstance(ps_pct, (int, float)):
                metric_parts.append(f"P/S {format_number(ps)} ({ps_pct:.2f}%)")
            else:
                metric_parts.append(f"P/S {format_number(ps)}")
        if pb is not None:
            pb_pct = percentiles.get("pb")
            if isinstance(pb_pct, (int, float)):
                metric_parts.append(f"P/B {format_number(pb)} ({pb_pct:.2f}%)")
            else:
                metric_parts.append(f"P/B {format_number(pb)}")
        if metric_parts:
            lines.append("- 估值指标: " + ", ".join(metric_parts))

        # Add valuation interpretation
        valuation_insights = []
        if peg is not None:
            if peg < 1:
                valuation_insights.append(
                    f"PEG {format_number(peg)} < 1.0，通常意味着估值与增长预期的匹配度相对较高"
                )
            elif peg < 2:
                valuation_insights.append(
                    f"PEG {format_number(peg)} < 2.0，估值与增长预期整体处于可解释区间"
                )
            else:
                valuation_insights.append(
                    f"PEG {format_number(peg)} > 2.0，估值对增长兑现的要求相对更高"
                )

        if forward_pe is not None and pe is not None and forward_pe > 0:
            implied_growth = (pe / forward_pe - 1) * 100
            if implied_growth > 0:
                valuation_insights.append(
                    f"若以当前 P/E 与 Forward P/E 对比，市场隐含的下一年盈利增速约为 {format_number(implied_growth)}%"
                )
        elif forward_pe is not None and pe is None:
            valuation_insights.append(
                f"Forward P/E {format_number(forward_pe)} 可作为当前主要估值锚，需关注后续盈利兑现。"
            )

        if valuation_insights:
            lines.append("")
            lines.append("估值合理性分析:")
            for insight in valuation_insights:
                lines.append(f"- {insight}")

    fundamentals = []
    fundamentals.extend(summarize_growth(analysis))
    fundamentals.extend(summarize_profitability(analysis))
    fundamentals.extend(summarize_balance_sheet(analysis))
    fundamentals.extend(summarize_rnd(analysis))
    if fundamentals:
        lines.append("")
        lines.append("基本面提示:")
        lines.extend(fundamentals)

    pe_pct = percentiles.get("pe")
    revenue_yoy_quarterly = analysis.get("growth", {}).get("revenue_yoy_quarterly")
    sentiment_score = 0
    if isinstance(pe_pct, (int, float)):
        if pe_pct <= 30:
            sentiment_score += 1
        elif pe_pct >= 80:
            sentiment_score -= 1
    if isinstance(revenue_yoy_quarterly, (int, float)):
        if revenue_yoy_quarterly >= 0.12:
            sentiment_score += 1
        elif revenue_yoy_quarterly < 0:
            sentiment_score -= 1
    if isinstance(current_price, (int, float)) and isinstance(target_mean, (int, float)):
        if target_mean >= current_price * 1.1:
            sentiment_score += 1
        elif target_mean <= current_price * 0.9:
            sentiment_score -= 1
    if isinstance(current_price, (int, float)) and isinstance(dcf_value, (int, float)):
        if dcf_value >= current_price * 1.1:
            sentiment_score += 1
        elif dcf_value <= current_price * 0.9:
            sentiment_score -= 1

    if sentiment_score >= 2:
        conclusion = "综合信号偏积极，但仍需结合后续财报与估值波动动态评估。"
    elif sentiment_score <= -2:
        conclusion = "综合信号偏谨慎，建议优先关注盈利兑现与估值回归风险。"
    else:
        conclusion = "综合信号中性，建议持续跟踪后续财报与管理层指引变化。"
    lines.append(f"- 结论: {conclusion}")

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


def build_report_title(analysis: dict[str, Any], valuation: dict[str, Any]) -> str:
    company = analysis.get("company", {})
    company_name = company.get("name") or analysis.get("symbol") or "该公司"
    growth = analysis.get("growth", {})
    expectations = analysis.get("expectations", {})
    revenue_growth = growth.get("revenue_yoy_quarterly")
    if not isinstance(revenue_growth, (int, float)):
        revenue_growth = expectations.get("revenue_growth_qoq")
    if not isinstance(revenue_growth, (int, float)):
        revenue_growth = expectations.get("revenue_guidance")

    if isinstance(revenue_growth, (int, float)):
        if revenue_growth >= 0.15:
            growth_tag = "增长加速"
        elif revenue_growth >= 0.05:
            growth_tag = "稳健增长"
        elif revenue_growth >= 0:
            growth_tag = "增长放缓"
        else:
            growth_tag = "增长承压"
    else:
        growth_tag = "经营趋势跟踪"

    percentiles = valuation.get("percentiles", {})
    metrics = valuation.get("metrics", {})
    pe_pct = percentiles.get("pe")
    pb_pct = percentiles.get("pb")
    if isinstance(pe_pct, (int, float)):
        if pe_pct >= 80:
            valuation_tag = "估值偏高"
        elif pe_pct <= 30:
            valuation_tag = "估值具吸引力"
        else:
            valuation_tag = "估值处中枢"
    elif isinstance(pb_pct, (int, float)):
        if pb_pct >= 80:
            valuation_tag = "高估值区间"
        elif pb_pct <= 30:
            valuation_tag = "低估值区间"
        else:
            valuation_tag = "估值分位中性"
    elif isinstance(metrics.get("forward_pe"), (int, float)):
        valuation_tag = "估值锚点偏前瞻"
    else:
        valuation_tag = "估值待验证"

    return f"{company_name}财报深度分析：{growth_tag}，{valuation_tag}"


def build_report(
    analysis: dict[str, Any], valuation: dict[str, Any], analyst: dict[str, Any]
) -> str:
    data_fetched_at = analysis.get("data_fetched_at")
    title = build_report_title(analysis, valuation)
    opening = build_core_opinion(analysis, valuation, analyst)
    analysis_date = format_analysis_date(data_fetched_at or analysis.get("generated_at"))
    chart_section = build_chart_references(analysis)

    report_lines: list[str] = [
        f"# {title}",
        "",
        opening,
        "",
        f"分析时点: {analysis_date}",
        "",
        "## 1. 财务亮点 (Financial Highlight)",
        build_financial_highlights(analysis, valuation),
        "",
        "## 2. 产品研究 (Product Research)",
        build_product_research(analysis),
        "",
        "## 3. 竞争格局 (Competitive Landscape)",
        build_competitive_section(analysis),
        "",
        "## 4. 管理层指引 (Management Guidance)",
        build_management_guidance(analysis),
        "",
        "## 5. 估值分析",
        build_valuation_table(valuation),
        build_currency_note(valuation),
        "",
        "## 6. 投资建议",
        build_investment_section(analysis, valuation, analyst),
        "",
    ]

    if chart_section:
        try:
            investment_index = report_lines.index("## 6. 投资建议")
        except ValueError:
            investment_index = len(report_lines)
        report_lines[investment_index:investment_index] = ["", chart_section, ""]

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
