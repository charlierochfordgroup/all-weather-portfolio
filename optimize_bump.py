"""Grid-search optimal bump schedule for DD P-Value Momentum strategy.

Optimises the full bump shape (per-rank allocation adjustments) to
maximise CAGR. Uses a parameterised shape with 4 degrees of freedom:
  - top_bump:    bump for rank 0 (deepest DD asset)
  - top_decay:   exponential decay rate for top 10 bumps
  - bottom_cut:  reduction for worst-ranked asset
  - bottom_decay: exponential decay rate for bottom reductions

This avoids overfitting vs optimising all 17 ranks independently,
while allowing non-linear bump shapes (e.g., rank 1 gets +60%,
rank 2 gets +20%, rather than forced linear interpolation).

Usage:  python optimize_bump.py
"""

import numpy as np
import pandas as pd
import pickle
import itertools
from pathlib import Path

from data import ASSETS, load_data
from stats import calc_stats, compute_asset_starts
from optimizer import run_optimization
from dd_momentum import (
    _cumulative_prices, detect_drawdown_episodes, compute_pvalue,
    build_dd_momentum_schedule,
)

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


def compute_adjustments_custom(
    returns: pd.DataFrame,
    checkpoint_dates: list,
    bump_schedule: np.ndarray,
    relative_threshold: float = 0.25,
) -> dict:
    """Like compute_dd_adjustments but with a fully custom per-rank bump schedule.

    bump_schedule: array of length n_assets, where bump_schedule[rank] gives the
    adjustment factor for that rank (rank 0 = lowest p-value = deepest DD).
    Positive = bump up, negative = reduce.
    """
    n_assets = returns.shape[1]
    adjustments = {}

    for cp_date in checkpoint_dates:
        r_up_to = returns[returns.index <= cp_date]
        if len(r_up_to) < 20:
            adjustments[cp_date] = np.zeros(n_assets)
            continue

        pvalues = np.ones(n_assets)
        for i, col in enumerate(returns.columns):
            asset_rets = r_up_to[col]
            nz = asset_rets[asset_rets != 0]
            if len(nz) < 20:
                continue
            prices = _cumulative_prices(nz)
            episodes = detect_drawdown_episodes(prices, relative_threshold)
            peak_val = prices.cummax().iloc[-1]
            current_dd = prices.iloc[-1] / peak_val - 1.0
            pvalues[i] = compute_pvalue(current_dd, episodes)

        ranks = np.argsort(np.argsort(pvalues))
        adj = np.zeros(n_assets)
        for i in range(n_assets):
            adj[i] = bump_schedule[ranks[i]]

        adjustments[cp_date] = adj

    return adjustments


def make_bump_schedule(top_bump, top_decay, bottom_cut, bottom_decay, n_assets=17):
    """Build a per-rank bump schedule from 4 shape parameters.

    Top 10 ranks (0-9): bumps decay exponentially from top_bump.
      bump[k] = top_bump * exp(-top_decay * k)

    Bottom ranks (10+): reductions grow exponentially.
      cut[j] = -bottom_cut * exp(bottom_decay * j / (n_remaining - 1))
      where j goes from 0 to n_remaining-1

    This allows convex, concave, and linear shapes depending on decay params.
    """
    schedule = np.zeros(n_assets)

    # Top 10: exponential decay of bumps
    for k in range(min(10, n_assets)):
        schedule[k] = top_bump * np.exp(-top_decay * k)

    # Bottom: exponential growth of reductions
    n_bottom = n_assets - 10
    if n_bottom > 0:
        for j in range(n_bottom):
            t = j / max(n_bottom - 1, 1)  # 0 to 1
            schedule[10 + j] = -0.05 - (bottom_cut - 0.05) * np.exp(bottom_decay * t) / np.exp(bottom_decay)

    return schedule


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

    # Compute base weights (Max Sharpe)
    print("Computing base weights (Max Sharpe)...")
    base_weights = run_optimization(
        opt_returns, "Max Sharpe Ratio", min_w, max_w, group_max, RISK_FREE_RATE,
        rebalance="daily",
    )

    # Annual checkpoints
    years = sorted(set(returns_full.index.year))
    checkpoints = []
    for y in years:
        yr_dates = returns_full.index[returns_full.index.year == y]
        if len(yr_dates) > 0:
            checkpoints.append(yr_dates[0])

    # Grid search over shape parameters
    top_bumps = [0.3, 0.5, 0.7, 1.0, 1.3]
    top_decays = [0.0, 0.1, 0.3, 0.5, 0.8]  # 0 = flat, higher = steeper decay
    bottom_cuts = [0.1, 0.2, 0.3, 0.5]
    bottom_decays = [0.0, 0.5, 1.0, 1.5]  # 0 = flat reduction

    n_combos = len(top_bumps) * len(top_decays) * len(bottom_cuts) * len(bottom_decays)
    print(f"\nGrid searching {n_combos} parameter combinations...")
    print(f"{'top_bump':>10} {'top_decay':>10} {'bot_cut':>10} {'bot_decay':>10} "
          f"{'CAGR':>10} {'Sharpe':>10} {'MaxDD':>10}")
    print("-" * 80)

    results = []
    best_cagr = -np.inf
    best_params = None

    for tb, td, bc, bd in itertools.product(top_bumps, top_decays, bottom_cuts, bottom_decays):
        schedule = make_bump_schedule(tb, td, bc, bd, n_assets=len(ASSETS))
        adj = compute_adjustments_custom(returns_full, checkpoints, schedule)
        w_schedule = build_dd_momentum_schedule(base_weights, adj)

        stats = calc_stats(
            returns_full, base_weights, RISK_FREE_RATE,
            rebalance="annual", asset_starts=asset_starts,
            weights_schedule=w_schedule,
        )

        results.append({
            "top_bump": tb, "top_decay": td,
            "bottom_cut": bc, "bottom_decay": bd,
            "cagr": stats.cagr, "sharpe": stats.sharpe,
            "max_drawdown": stats.max_drawdown, "calmar": stats.calmar,
            "volatility": stats.volatility,
        })

        if stats.cagr > best_cagr:
            best_cagr = stats.cagr
            best_params = (tb, td, bc, bd)
            print(f"{tb:>9.1f} {td:>9.1f} {bc:>9.1f} {bd:>9.1f} "
                  f"{stats.cagr:>9.2%} {stats.sharpe:>9.2f} {stats.max_drawdown:>9.2%} *NEW BEST*")

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["cagr"].idxmax()]

    print(f"\n{'='*80}")
    print(f"OPTIMAL PARAMETERS:")
    print(f"  top_bump:    {best_row['top_bump']:.1f}")
    print(f"  top_decay:   {best_row['top_decay']:.1f}")
    print(f"  bottom_cut:  {best_row['bottom_cut']:.1f}")
    print(f"  bottom_decay:{best_row['bottom_decay']:.1f}")
    print(f"\n  CAGR:     {best_row['cagr']:.2%}")
    print(f"  Vol:      {best_row['volatility']:.2%}")
    print(f"  Sharpe:   {best_row['sharpe']:.2f}")
    print(f"  Max DD:   {best_row['max_drawdown']:.2%}")
    print(f"  Calmar:   {best_row['calmar']:.2f}")

    # Build the optimal schedule
    optimal_schedule = make_bump_schedule(
        best_row["top_bump"], best_row["top_decay"],
        best_row["bottom_cut"], best_row["bottom_decay"],
        n_assets=len(ASSETS),
    )

    print(f"\nOptimal per-rank bump schedule:")
    for r in range(len(ASSETS)):
        label = "BUMP" if optimal_schedule[r] > 0 else "CUT"
        print(f"  Rank {r:2d}: {optimal_schedule[r]:+.1%} ({label})")

    # Also compute stats for base (no DD momentum) for comparison
    base_stats = calc_stats(returns_full, base_weights, RISK_FREE_RATE,
                            rebalance="annual", asset_starts=asset_starts)
    print(f"\nBase (no DD momentum, static Max Sharpe):")
    print(f"  CAGR:     {base_stats.cagr:.2%}")
    print(f"  Sharpe:   {base_stats.sharpe:.2f}")
    print(f"  Max DD:   {base_stats.max_drawdown:.2%}")

    # Save optimal schedule and parameters
    out_path = Path(__file__).resolve().parent / "optimal_bump_schedule.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "optimal_schedule": optimal_schedule,
            "params": {
                "top_bump": float(best_row["top_bump"]),
                "top_decay": float(best_row["top_decay"]),
                "bottom_cut": float(best_row["bottom_cut"]),
                "bottom_decay": float(best_row["bottom_decay"]),
            },
            "grid_results": results_df.to_dict("records"),
            "n_assets": len(ASSETS),
        }, f)
    print(f"\nSaved optimal schedule to {out_path.name}")


if __name__ == "__main__":
    main()
