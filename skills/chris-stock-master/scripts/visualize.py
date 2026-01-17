#!/usr/bin/env python3
"""Generate PNG charts from analysis output."""

import argparse
import json
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
from series_utils import series_from_mapping, series_rows

logger = logging.getLogger(__name__)


def series_from_dict(data: dict[str, float]):
    return series_from_mapping(data)


def plot_series(
    series_list: dict[str, object], title: str, ylabel: str, output_path: str
) -> None:
    if not series_list:
        return
    plt.figure(figsize=(9, 4.5))
    plotted = False
    for label, series in series_list.items():
        rows = series_rows(series)
        if not rows:
            continue
        dates = [row[0] for row in rows]
        values = [row[1] for row in rows]
        plt.plot(dates, values, marker="o", label=label)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def expand_growth_series(
    growth_map: dict[str, Any], dates: list, default: float | None = None
) -> list[float | None]:
    if not growth_map or not dates:
        return [default for _ in dates]
    growth_rows = series_rows(series_from_dict(growth_map))
    if not growth_rows:
        return [default for _ in dates]
    growth_rows.sort(key=lambda row: row[0])
    result: list[float | None] = []
    idx = 0
    current_growth: float | None = default
    for date in dates:
        while idx < len(growth_rows) and growth_rows[idx][0] <= date:
            current_growth = growth_rows[idx][1]
            idx += 1
        result.append(current_growth)
    return result


def build_peg_series(pe_series, growth_map: dict[str, Any]) -> list[tuple[Any, float]]:
    rows = series_rows(pe_series)
    if not rows:
        return []
    dates = [row[0] for row in rows]
    growth_values = expand_growth_series(growth_map, dates)
    peg_rows: list[tuple[Any, float]] = []
    for (date, pe_value), growth in zip(rows, growth_values, strict=False):
        if pe_value is None or growth is None:
            continue
        if growth == 0:
            continue
        peg_rows.append((date, pe_value / (growth * 100)))
    return peg_rows


def generate_charts(
    analysis: dict[str, Any], output_dir: str, valuation: dict[str, Any] | None = None
) -> None:
    ensure_dir(output_dir)

    revenue = series_from_dict(analysis.get("financials", {}).get("revenue", {}))
    net_income = series_from_dict(analysis.get("financials", {}).get("net_income", {}))
    gross_margin = series_from_dict(analysis.get("ratios", {}).get("gross_margin", {}))
    net_margin = series_from_dict(analysis.get("ratios", {}).get("net_margin", {}))
    roe = series_from_dict(analysis.get("ratios", {}).get("roe", {}))
    roa = series_from_dict(analysis.get("ratios", {}).get("roa", {}))
    debt_to_equity = series_from_dict(
        analysis.get("ratios", {}).get("debt_to_equity", {})
    )
    price = series_from_dict(analysis.get("price", {}).get("history", {}))

    plot_series(
        {"Revenue": revenue, "Net Income": net_income},
        "Revenue & Net Income",
        "Amount",
        os.path.join(output_dir, "revenue_net_income.png"),
    )

    plot_series(
        {"Gross Margin": gross_margin, "Net Margin": net_margin},
        "Margin Trends",
        "Ratio",
        os.path.join(output_dir, "margin_trends.png"),
    )

    plot_series(
        {"ROE": roe, "ROA": roa},
        "ROE & ROA",
        "Ratio",
        os.path.join(output_dir, "roe_roa.png"),
    )

    plot_series(
        {"Debt/Equity": debt_to_equity},
        "Debt to Equity",
        "Ratio",
        os.path.join(output_dir, "debt_to_equity.png"),
    )

    plot_series(
        {"Price": price},
        "Stock Price",
        "Price",
        os.path.join(output_dir, "price_history.png"),
    )

    if valuation:
        pe_history = valuation.get("history", {}).get("pe", {})
        pe_series = series_from_dict(pe_history)
        growth_map = analysis.get("growth", {}).get("net_income_yoy", {})
        if not growth_map:
            growth_map = analysis.get("growth", {}).get("revenue_yoy", {})
        peg_rows = build_peg_series(pe_series, growth_map)
        if peg_rows:
            peg_dates = [row[0] for row in peg_rows]
            peg_values = [row[1] for row in peg_rows]
            plt.figure(figsize=(9, 4.5))
            plt.plot(peg_dates, peg_values, marker="o", label="PEG")
            plt.title("PEG (P/E vs Growth YoY)")
            plt.ylabel("PEG")
            plt.xlabel("Date")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "peg_ratio.png"), dpi=150)
            plt.close()

    logger.info(f"Saved charts to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate charts from analysis JSON")
    parser.add_argument("--analysis", required=True, help="Path to analysis JSON")
    parser.add_argument("--output", required=True, help="Output directory for charts")
    args = parser.parse_args()

    with open(args.analysis, encoding="utf-8") as handle:
        analysis = json.load(handle)

    generate_charts(analysis, args.output)


if __name__ == "__main__":
    main()
