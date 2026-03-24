"""Ensemble Meta-Strategy – allocate across existing strategies by trailing performance.

Treats each static strategy as a sub-portfolio and blends them using
inverse-volatility weighting based on trailing risk-adjusted returns.
This diversifies across *methodologies*, not just assets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stats import calc_stats, _make_period_labels


def build_ensemble_schedule(
    strategy_weights: dict[str, np.ndarray],
    returns: pd.DataFrame,
    risk_free_rate: float = 0.04,
    rebalance: str = "quarterly",
    lookback: int = 252,
    asset_starts: dict | None = None,
) -> dict[pd.Timestamp, np.ndarray]:
    """Build a time-varying weight schedule blending static strategies.

    Parameters
    ----------
    strategy_weights : mapping of strategy name -> weight array (sum = 1.0 each).
    returns : daily log-return DataFrame.
    risk_free_rate : annualised risk-free rate.
    rebalance : period granularity for re-evaluating strategy allocations.
    lookback : trailing window in trading days for performance evaluation.
    asset_starts : optional per-asset start dates.

    Returns
    -------
    dict mapping timestamps to blended asset-weight arrays.
    """
    if not strategy_weights:
        return {}

    names = list(strategy_weights.keys())
    weights_arr = [strategy_weights[n] for n in names]
    n_strategies = len(names)

    periods = _make_period_labels(returns.index, rebalance)
    boundaries = np.where(np.diff(periods) != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries])

    schedule: dict[pd.Timestamp, np.ndarray] = {}

    for b_idx in boundaries:
        date = returns.index[b_idx]

        # Trailing window for performance evaluation
        start_idx = max(0, b_idx - lookback)
        trailing = returns.iloc[start_idx:b_idx]

        if len(trailing) < 30:
            # Not enough data — equal-weight blend
            alloc = np.ones(n_strategies) / n_strategies
        else:
            # Compute trailing volatility for each strategy
            vols = np.zeros(n_strategies)
            for i, w in enumerate(weights_arr):
                s = calc_stats(trailing, w, risk_free_rate, rebalance="daily")
                vols[i] = max(s.volatility, 0.01)

            # Inverse-vol allocation across strategies
            inv_vol = 1.0 / vols
            alloc = inv_vol / inv_vol.sum()

        # Blend: weighted average of strategy weight vectors
        blended = np.zeros(len(weights_arr[0]))
        for i, w in enumerate(weights_arr):
            blended += alloc[i] * w

        # Normalise to sum to 1
        total = blended.sum()
        if total > 1e-12:
            blended /= total

        schedule[date] = blended

    return schedule


def ensemble_analytics(
    strategy_weights: dict[str, np.ndarray],
    returns: pd.DataFrame,
    risk_free_rate: float,
    lookback: int = 252,
) -> dict:
    """Compute current strategy allocations for the UI.

    Returns dict with 'strategy_names', 'allocations', 'trailing_vols',
    'trailing_sharpes'.
    """
    names = list(strategy_weights.keys())
    n = len(names)

    # Use last {lookback} days for evaluation
    trailing = returns.iloc[-lookback:] if len(returns) >= lookback else returns

    vols = np.zeros(n)
    sharpes = np.zeros(n)
    for i, name in enumerate(names):
        w = strategy_weights[name]
        s = calc_stats(trailing, w, risk_free_rate, rebalance="daily")
        vols[i] = max(s.volatility, 0.01)
        sharpes[i] = s.sharpe

    inv_vol = 1.0 / vols
    alloc = inv_vol / inv_vol.sum()

    return {
        "strategy_names": names,
        "allocations": alloc * 100.0,  # as percentage
        "trailing_vols": vols * 100.0,
        "trailing_sharpes": sharpes,
    }
