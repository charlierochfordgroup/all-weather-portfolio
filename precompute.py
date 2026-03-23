"""Pre-compute optimised portfolio weights for default settings.

Run this script locally whenever the data or default constraints change.
It saves weights to precomputed_weights.pkl which ships with the repo
so the Streamlit app loads instantly without running the optimizer.

Usage:  python precompute.py
"""

import pickle
import sys
import io
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

from data import ASSETS, load_data
from stats import calc_stats, compute_asset_starts
from optimizer import run_optimization
from regime import load_regime_data, classify_regimes, optimize_per_regime
from dd_momentum import (
    compute_dd_adjustments, compute_dd_adjustments_scheduled,
    build_dd_momentum_schedule, load_optimal_bump_schedule,
)

# ── Default settings (must match app.py defaults exactly) ──
RISK_FREE_RATE = 0.04

DEFAULT_MIN = {a: 0.5 for a in ASSETS}
DEFAULT_MAX = {
    "Cash": 20.0, "Nasdaq": 30.0, "S&P 500": 30.0, "Russell 2000": 30.0,
    "ASX200": 30.0, "Emerging Markets": 30.0,
    "Corporate Bonds": 40.0, "Long-Term Treasuries": 40.0, "Short-Term Treasuries": 40.0,
    "Real Estate": 20.0, "Commodities": 10.0, "Gold": 20.0,
    "Bitcoin": 15.0, "Infrastructure": 20.0,
    "Japan Equities": 30.0, "UK Equities": 30.0, "EU Equities": 30.0,
}
DEFAULT_GROUP_MAX = {
    "US Equities": 35.0, "Intl Equities": 30.0, "Bonds": 40.0,
    "Real Assets": 30.0, "Alternatives": 20.0,
}

BASE_TARGETS = [
    "Max Sharpe Ratio", "Min Volatility", "Max Calmar Ratio",
    "Minimize Max Drawdown",
    "Inverse Volatility", "Equal Risk Contribution", "Hierarchical Risk Parity",
]
DD_TARGETS = ["Max Sharpe (DD \u2264 X%)", "Max Calmar (DD \u2264 X%)"]
DD_LEVELS = [5, 10, 15, 20, 25, 30]


def main():
    print("Loading data...")
    data_path = str(Path(__file__).resolve().parent / "data_template.xlsx")
    all_returns = load_data(data_path, use_processing=True)

    available = [a for a in ASSETS if a in all_returns.columns]
    returns_full = all_returns[available]

    asset_starts = compute_asset_starts(returns_full)
    overlap_start = max(asset_starts.values())
    opt_returns = returns_full.loc[overlap_start:]

    min_w = np.array([DEFAULT_MIN.get(a, 0.5) / 100.0 for a in ASSETS])
    max_w = np.array([DEFAULT_MAX.get(a, 30.0) / 100.0 for a in ASSETS])
    group_max = {g: v / 100.0 for g, v in DEFAULT_GROUP_MAX.items()}
    rf = RISK_FREE_RATE

    # Full backtest returns for DD constraint evaluation
    bt_returns = returns_full
    bt_asset_starts = asset_starts

    print(f"Overlap period: {overlap_start.date()} to {returns_full.index[-1].date()}")
    print(f"Optimising {len(BASE_TARGETS)} base + {len(DD_TARGETS)}x{len(DD_LEVELS)} DD-constrained portfolios...")

    # Base strategies
    base_w = {}
    for i, tgt in enumerate(BASE_TARGETS, 1):
        print(f"  [{i}/{len(BASE_TARGETS)}] {tgt}...")
        w = run_optimization(opt_returns, tgt, min_w, max_w, group_max, rf, rebalance="daily")
        base_w[tgt] = w

    # DD-constrained strategies (warm-started from previous level)
    dd_w = {}
    prev_w = {}
    total_dd = len(DD_LEVELS) * len(DD_TARGETS)
    count = 0
    for dd_pct in sorted(DD_LEVELS):
        dd_val = dd_pct / 100.0
        dd_w[dd_pct] = {}
        for tgt in DD_TARGETS:
            count += 1
            print(f"  [{count}/{total_dd}] {tgt.replace('X%', f'{dd_pct}%')}...")
            w = run_optimization(
                opt_returns, tgt, min_w, max_w, group_max, rf,
                rebalance="daily", dd_constraint=dd_val,
                current_weights=prev_w.get(tgt),
                dd_returns=bt_returns, dd_asset_starts=bt_asset_starts,
            )
            dd_w[dd_pct][tgt] = w
            prev_w[tgt] = w

    # ── Dynamic strategies ──

    # DD Momentum: compute annual adjustments using Max Sharpe as base
    print("\nComputing DD Momentum adjustments...")
    ms_weights = base_w["Max Sharpe Ratio"]
    years = sorted(set(returns_full.index.year))
    checkpoints = []
    for y in years:
        yr_dates = returns_full.index[returns_full.index.year == y]
        if len(yr_dates) > 0:
            checkpoints.append(yr_dates[0])
    # Use optimised per-rank bump schedule if available, otherwise parametric
    optimal_sched = load_optimal_bump_schedule()
    if optimal_sched is not None:
        print("  Using optimised per-rank bump schedule")
        dd_adj = compute_dd_adjustments_scheduled(returns_full, checkpoints, optimal_sched)
    else:
        print("  Using default parametric bump schedule (bump_max=50%)")
        dd_adj = compute_dd_adjustments(returns_full, checkpoints)
    dd_momentum_schedule = build_dd_momentum_schedule(ms_weights, dd_adj)
    print(f"  {len(dd_adj)} annual checkpoints computed")

    # Regime: optimise per regime if macro data available
    macro_path = Path(__file__).resolve().parent / "Inflation and IR.xlsx"
    macro_data = load_regime_data(macro_path)
    regime_weights = None
    regime_series = None
    if macro_data is not None:
        print("\nComputing regime portfolios...")
        regime_series = classify_regimes(macro_data)
        regime_weights = optimize_per_regime(
            opt_returns, regime_series, "Max Sharpe Ratio",
            min_w, max_w, group_max, rf, rebalance="daily",
        )
        for label_id, w in regime_weights.items():
            print(f"  Regime {label_id}: done")
    else:
        print("\nNo macro data found — skipping regime pre-computation")

    # Save weights only (stats are recomputed at runtime for any backtest period)
    output = {
        "base_weights": base_w,
        "dd_weights": dd_w,
        "dd_momentum_adjustments": dd_adj,
        "dd_momentum_base": ms_weights,
        "regime_weights": regime_weights,
        "regime_series": regime_series,
        "overlap_start": overlap_start,
        "data_end": returns_full.index[-1],
        "default_min": DEFAULT_MIN,
        "default_max": DEFAULT_MAX,
        "default_group_max": DEFAULT_GROUP_MAX,
        "risk_free_rate": RISK_FREE_RATE,
    }

    out_path = Path(__file__).resolve().parent / "precomputed_weights.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved to {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
    print("Done!")


if __name__ == "__main__":
    main()
