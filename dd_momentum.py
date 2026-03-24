"""Drawdown P-Value Momentum strategy.

Annually assesses each asset's current drawdown relative to its historical
drawdown distribution. Assets with unusually deep drawdowns (low p-value)
get allocation bumps; assets near highs get reductions.

Supports two modes:
1. Parametric (legacy): bump_max controls a linear interpolation.
2. Optimised schedule: per-rank bump factors loaded from optimal_bump_schedule.pkl,
   found via grid search over shape parameters (see optimize_bump.py).
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path


def _cumulative_prices(log_returns: pd.Series) -> pd.Series:
    """Convert log returns to cumulative price series starting at 1.0."""
    return np.exp(log_returns.cumsum())


def detect_drawdown_episodes(
    prices: pd.Series,
    relative_threshold: float = 0.25,
) -> list[dict]:
    """Find peak-to-trough drawdown episodes for a single asset.

    Only keeps episodes where depth >= relative_threshold * max historical DD.

    Returns list of {"start": date, "trough": date, "depth": float}.
    depth is negative (e.g. -0.30 for 30% drawdown).
    """
    if len(prices) < 2:
        return []

    vals = prices.values
    dates = prices.index
    peak = np.maximum.accumulate(vals)
    dd = vals / peak - 1.0

    # Find max DD to set threshold
    max_dd = np.min(dd)
    if max_dd >= 0:
        return []
    abs_threshold = abs(max_dd) * relative_threshold

    # Vectorised episode detection: find contiguous stretches where dd < 0
    in_dd_mask = dd < 0
    padded = np.concatenate(([False], in_dd_mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    ep_starts = np.where(changes == 1)[0]
    ep_ends = np.where(changes == -1)[0]

    episodes = []
    for s, e in zip(ep_starts, ep_ends):
        trough_idx = s + np.argmin(dd[s:e])
        trough_val = dd[trough_idx]
        if abs(trough_val) >= abs_threshold:
            episodes.append({
                "start": dates[s],
                "trough": dates[trough_idx],
                "depth": trough_val,
            })

    return episodes


def compute_pvalue(current_dd: float, episodes: list[dict]) -> float:
    """Compute p-value: fraction of episodes with worse (deeper) drawdown.

    Low p-value = current drawdown is historically rare = buying opportunity.
    Returns 1.0 if no episodes or no current drawdown.
    """
    if not episodes or current_dd >= 0:
        return 1.0
    worse_count = sum(1 for ep in episodes if ep["depth"] <= current_dd)
    return worse_count / len(episodes)


def compute_dd_adjustments(
    returns: pd.DataFrame,
    checkpoint_dates: list[pd.Timestamp],
    relative_threshold: float = 0.25,
    bump_max: float = 0.50,
) -> dict[pd.Timestamp, np.ndarray]:
    """Compute per-asset adjustment factors at each annual checkpoint.

    At each checkpoint:
    - Compute p-value per asset (using only data up to that date)
    - Rank assets by p-value (lowest first)
    - Bump factors scale from +bump_max (rank 0) down to +5% (rank 9),
      then reductions from -5% to -(bump_max * 0.6) for the rest.

    bump_max: maximum positive bump for the most deeply drawn-down asset
              (default 0.50 = +50%). Configurable via the UI.

    Returns dict mapping checkpoint dates to adjustment arrays (n_assets,).
    """
    n_assets = returns.shape[1]
    adjustments = {}
    # Scale the per-rank step and reduction proportionally to bump_max
    bump_step = (bump_max - 0.05) / max(9, 1)  # step between ranks in top 10
    reduction_max = bump_max * 0.6  # max reduction for worst-ranked asset

    for cp_date in checkpoint_dates:
        # Use only data up to checkpoint (no lookahead)
        r_up_to = returns[returns.index <= cp_date]
        if len(r_up_to) < 20:
            adjustments[cp_date] = np.zeros(n_assets)
            continue

        pvalues = np.ones(n_assets)
        current_dds = np.zeros(n_assets)

        for i, col in enumerate(returns.columns):
            asset_rets = r_up_to[col]
            # Only use non-zero returns
            nz = asset_rets[asset_rets != 0]
            if len(nz) < 20:
                continue

            prices = _cumulative_prices(nz)
            episodes = detect_drawdown_episodes(prices, relative_threshold)

            # Current drawdown
            peak_val = prices.cummax().iloc[-1]
            current_dd = prices.iloc[-1] / peak_val - 1.0
            current_dds[i] = current_dd

            pvalues[i] = compute_pvalue(current_dd, episodes)

        # Rank by p-value (lowest first = deepest relative drawdown)
        ranks = np.argsort(np.argsort(pvalues))  # rank 0 = lowest p-value

        adj = np.zeros(n_assets)
        for i in range(n_assets):
            rank = ranks[i]
            if rank < 10:
                # Top 10 get bumps scaling from bump_max down to ~+5%
                adj[i] = bump_max - rank * bump_step
            else:
                # Remaining get reductions scaling from -5% to -reduction_max
                remaining = n_assets - 10
                if remaining > 0:
                    pos_in_tail = rank - 10  # 0 to remaining-1
                    adj[i] = -0.05 - (reduction_max - 0.05) * (pos_in_tail / max(remaining - 1, 1))
                else:
                    adj[i] = -0.05

        adjustments[cp_date] = adj

    return adjustments


def load_optimal_bump_schedule() -> np.ndarray | None:
    """Load the pre-computed optimal per-rank bump schedule.

    Returns array of length n_assets where schedule[rank] gives the
    adjustment factor for that rank, or None if not found.
    """
    path = Path(__file__).resolve().parent / "optimal_bump_schedule.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["optimal_schedule"]
    except Exception:
        return None


def compute_dd_adjustments_scheduled(
    returns: pd.DataFrame,
    checkpoint_dates: list[pd.Timestamp],
    bump_schedule: np.ndarray,
    relative_threshold: float = 0.25,
) -> dict[pd.Timestamp, np.ndarray]:
    """Compute per-asset adjustments using a custom per-rank bump schedule.

    bump_schedule: array where bump_schedule[rank] gives the adjustment
    for that p-value rank (rank 0 = lowest p-value = deepest DD).
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
            adj[i] = bump_schedule[min(ranks[i], len(bump_schedule) - 1)]

        adjustments[cp_date] = adj

    return adjustments


def build_dd_momentum_schedule(
    base_weights: np.ndarray,
    adjustments: dict[pd.Timestamp, np.ndarray],
    dd_constraint: float | None = None,
    returns: pd.DataFrame | None = None,
    risk_free_rate: float = 0.04,
    rebalance: str = "annual",
    asset_starts: dict | None = None,
) -> dict[pd.Timestamp, np.ndarray]:
    """Build time-varying weight schedule from base weights and adjustments.

    adjusted = base_weights * (1 + adjustment_factor), clipped and normalised.

    If dd_constraint is set and returns are provided, enforces the max drawdown
    constraint by scaling back adjustment factors via binary search when the
    resulting portfolio exceeds the limit.
    """
    from stats import calc_stats

    schedule = {}
    for date, adj in adjustments.items():
        adjusted = base_weights * (1.0 + adj)
        adjusted = np.maximum(adjusted, 0.0)  # no negative weights
        total = adjusted.sum()
        if total > 1e-12:
            adjusted /= total
        else:
            adjusted = base_weights.copy()
        schedule[date] = adjusted

        # Enforce DD constraint: if the schedule so far exceeds the limit,
        # scale back this checkpoint's adjustments via binary search.
        if dd_constraint is not None and returns is not None:
            s = calc_stats(
                returns, base_weights, risk_free_rate,
                rebalance=rebalance, asset_starts=asset_starts,
                weights_schedule=schedule,
            )
            if abs(s.max_drawdown) > dd_constraint:
                lo, hi = 0.0, 1.0
                for _ in range(10):
                    mid = (lo + hi) / 2.0
                    scaled = base_weights * (1.0 + adj * mid)
                    scaled = np.maximum(scaled, 0.0)
                    t = scaled.sum()
                    scaled = scaled / t if t > 1e-12 else base_weights.copy()
                    schedule[date] = scaled
                    s2 = calc_stats(
                        returns, base_weights, risk_free_rate,
                        rebalance=rebalance, asset_starts=asset_starts,
                        weights_schedule=schedule,
                    )
                    if abs(s2.max_drawdown) > dd_constraint:
                        hi = mid
                    else:
                        lo = mid
                # Use the conservative (lower) scaling factor
                final_scaled = base_weights * (1.0 + adj * lo)
                final_scaled = np.maximum(final_scaled, 0.0)
                t = final_scaled.sum()
                schedule[date] = final_scaled / t if t > 1e-12 else base_weights.copy()

    return schedule


def dd_analytics(
    returns: pd.DataFrame,
    as_of_date: pd.Timestamp,
    relative_threshold: float = 0.25,
    bump_max: float = 0.50,
) -> dict:
    """Compute analytics for the Dynamic Strategies UI tab.

    Returns per-asset data: p-values, current drawdowns, episode counts, bump factors.
    """
    n_assets = returns.shape[1]
    r_up_to = returns[returns.index <= as_of_date]

    result = {
        "assets": list(returns.columns),
        "pvalues": np.ones(n_assets),
        "current_dd": np.zeros(n_assets),
        "episode_counts": np.zeros(n_assets, dtype=int),
        "bump_factors": np.zeros(n_assets),
    }

    if len(r_up_to) < 20:
        return result

    for i, col in enumerate(returns.columns):
        asset_rets = r_up_to[col]
        nz = asset_rets[asset_rets != 0]
        if len(nz) < 20:
            continue

        prices = _cumulative_prices(nz)
        episodes = detect_drawdown_episodes(prices, relative_threshold)

        peak_val = prices.cummax().iloc[-1]
        current_dd = prices.iloc[-1] / peak_val - 1.0

        result["current_dd"][i] = current_dd
        result["episode_counts"][i] = len(episodes)
        result["pvalues"][i] = compute_pvalue(current_dd, episodes)

    # Rank and assign bumps (configurable via bump_max)
    bump_step = (bump_max - 0.05) / max(9, 1)
    reduction_max = bump_max * 0.6
    ranks = np.argsort(np.argsort(result["pvalues"]))
    for i in range(n_assets):
        rank = ranks[i]
        if rank < 10:
            result["bump_factors"][i] = bump_max - rank * bump_step
        else:
            remaining = n_assets - 10
            if remaining > 0:
                pos_in_tail = rank - 10
                result["bump_factors"][i] = -0.05 - (reduction_max - 0.05) * (pos_in_tail / max(remaining - 1, 1))
            else:
                result["bump_factors"][i] = -0.05

    return result
