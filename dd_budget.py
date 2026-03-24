"""Drawdown Budget Allocation – dynamic risk scaling based on DD headroom.

When the portfolio is at its peak, full exposure is maintained.  As drawdown
deepens, exposure is scaled linearly toward cash so that the maximum tolerable
drawdown (the "budget") is never fully consumed in a single move.

    scale = max(0, (budget - |current_dd|) / budget)

Weights that sum to less than 1.0 implicitly hold the remainder in cash
(0 % return), which ``stats._periodic_rebal_returns`` already supports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stats import calc_stats, _make_period_labels


def build_dd_budget_schedule(
    base_weights: np.ndarray,
    returns: pd.DataFrame,
    budget: float,
    risk_free_rate: float = 0.04,
    rebalance: str = "annual",
    asset_starts: dict | None = None,
) -> dict[pd.Timestamp, np.ndarray]:
    """Build a time-varying weight schedule that scales exposure by DD headroom.

    Parameters
    ----------
    base_weights : array of target weights (sum = 1.0).
    returns : daily log-return DataFrame (columns = asset names).
    budget : maximum tolerable drawdown as a positive fraction (e.g. 0.20).
    rebalance : period granularity for checking drawdown and updating weights.
    asset_starts : optional per-asset start dates for pro-rata redistribution.

    Returns
    -------
    dict mapping rebalance-boundary timestamps to (possibly sub-1.0) weight arrays.
    """
    if budget <= 0:
        return {}

    periods = _make_period_labels(returns.index, rebalance)
    boundaries = np.where(np.diff(periods) != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries])

    schedule: dict[pd.Timestamp, np.ndarray] = {}
    peak = 1.0
    equity = 1.0

    # Convert log returns to simple for equity tracking
    simple_rets = np.exp(returns.values) - 1.0

    for seg_idx, (seg_start, seg_end) in enumerate(
        zip(boundaries, np.append(boundaries[1:], len(returns)))
    ):
        date = returns.index[seg_start]

        # Compute scale factor based on current drawdown vs budget
        current_dd = (equity / peak) - 1.0  # negative when in drawdown
        scale = max(0.0, (budget - abs(current_dd)) / budget)

        schedule[date] = base_weights * scale

        # Simulate equity through this segment using scaled weights
        cur_w = base_weights * scale
        for t in range(seg_start, seg_end):
            sr = simple_rets[t]
            day_r = cur_w @ sr
            equity *= (1.0 + day_r)
            if equity > peak:
                peak = equity

            # Drift weights (preserve cash allocation)
            w_sum = cur_w.sum()
            if w_sum > 1e-12:
                grown = cur_w * (1.0 + sr)
                grown_sum = grown.sum()
                if grown_sum > 1e-12:
                    cur_w = grown * (w_sum / grown_sum)

    return schedule


def dd_budget_analytics(
    returns: pd.DataFrame,
    base_weights: np.ndarray,
    budget: float,
    schedule: dict[pd.Timestamp, np.ndarray],
) -> dict:
    """Compute analytics for the Dynamic Strategies UI tab.

    Returns dict with 'dates', 'scale_factors', 'drawdowns' arrays.
    """
    if not schedule:
        return {"dates": [], "scale_factors": [], "drawdowns": []}

    sorted_dates = sorted(schedule.keys())
    scale_factors = []
    drawdowns = []

    bw_sum = base_weights.sum()
    for dt in sorted_dates:
        w = schedule[dt]
        w_sum = w.sum()
        sf = w_sum / bw_sum if bw_sum > 1e-12 else 0.0
        scale_factors.append(sf)

    # Compute the equity curve and drawdowns at each schedule date
    simple_rets = np.exp(returns.values) - 1.0
    peak = 1.0
    equity = 1.0
    sched_idx = 0
    cur_w = schedule[sorted_dates[0]] if sorted_dates else base_weights

    dd_at_dates = []
    for t in range(len(returns)):
        date = returns.index[t]
        # Check if we've hit a new schedule date
        if sched_idx < len(sorted_dates) and date >= sorted_dates[sched_idx]:
            dd_at_dates.append((equity / peak) - 1.0)
            cur_w = schedule[sorted_dates[sched_idx]]
            sched_idx += 1

        sr = simple_rets[t]
        day_r = cur_w @ sr
        equity *= (1.0 + day_r)
        if equity > peak:
            peak = equity

        w_sum = cur_w.sum()
        if w_sum > 1e-12:
            grown = cur_w * (1.0 + sr)
            grown_sum = grown.sum()
            if grown_sum > 1e-12:
                cur_w = grown * (w_sum / grown_sum)

    # Pad if needed
    while len(dd_at_dates) < len(sorted_dates):
        dd_at_dates.append((equity / peak) - 1.0)

    return {
        "dates": sorted_dates,
        "scale_factors": scale_factors,
        "drawdowns": dd_at_dates,
    }
