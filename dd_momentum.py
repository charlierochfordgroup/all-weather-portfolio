"""Drawdown P-Value Momentum strategy.

Annually assesses each asset's current drawdown relative to its historical
drawdown distribution. Assets with unusually deep drawdowns (low p-value)
get allocation bumps; assets near highs get reductions.
"""

import numpy as np
import pandas as pd


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

    # Identify episodes: contiguous stretches where dd < 0
    episodes = []
    in_dd = False
    ep_start = None
    ep_trough_idx = None
    ep_trough_val = 0.0

    for i in range(len(dd)):
        if dd[i] < 0:
            if not in_dd:
                in_dd = True
                ep_start = i
                ep_trough_idx = i
                ep_trough_val = dd[i]
            elif dd[i] < ep_trough_val:
                ep_trough_idx = i
                ep_trough_val = dd[i]
        else:
            if in_dd:
                # Episode ended — record if significant enough
                if abs(ep_trough_val) >= abs_threshold:
                    episodes.append({
                        "start": dates[ep_start],
                        "trough": dates[ep_trough_idx],
                        "depth": ep_trough_val,
                    })
                in_dd = False

    # Handle ongoing drawdown at end of series
    if in_dd and abs(ep_trough_val) >= abs_threshold:
        episodes.append({
            "start": dates[ep_start],
            "trough": dates[ep_trough_idx],
            "depth": ep_trough_val,
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


def build_dd_momentum_schedule(
    base_weights: np.ndarray,
    adjustments: dict[pd.Timestamp, np.ndarray],
) -> dict[pd.Timestamp, np.ndarray]:
    """Build time-varying weight schedule from base weights and adjustments.

    adjusted = base_weights * (1 + adjustment_factor), clipped and normalized.
    """
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
