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


def generate_charts(analysis: dict[str, Any], output_dir: str) -> None:
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
