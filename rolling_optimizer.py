"""Rolling-window portfolio re-optimisation for time-varying strategies.

Provides `build_rolling_optimization_schedule()` which re-optimises a given
target (e.g. Max Sharpe, Leverage-Optimal) at each annual boundary using a
trailing window of returns. Produces a weights_schedule dict that can be
passed directly to `calc_stats(..., weights_schedule=schedule)`.
"""

import numpy as np
import pandas as pd

from data import ASSETS
from stats import _make_period_labels
from optimizer import run_optimization


def build_rolling_optimization_schedule(
    returns: pd.DataFrame,
    target: str,
    min_w: np.ndarray,
    max_w: np.ndarray,
    group_max: dict[str, float],
    risk_free_rate: float = 0.04,
    window_years: int = 5,
    rebalance: str = "annual",
    leverage: float = 1.0,
    financing_rate: float = 0.065,
) -> dict[pd.Timestamp, np.ndarray]:
    """Re-optimise `target` at each annual boundary using a trailing window.

    At each rebalance date, uses the most recent `window_years` of data
    to find optimal weights. Before enough history exists (< 2 years),
    the checkpoint is skipped.

    Parameters
    ----------
    returns : daily log returns DataFrame (full backtest period)
    target : optimisation target name (e.g. "Max Sharpe Ratio")
    window_years : trailing lookback in years (default 5)
    rebalance : period label for boundary detection (default "annual")
    leverage / financing_rate : passed through for Leverage-Optimal target

    Returns
    -------
    dict mapping rebalance dates to weight arrays
    """
    periods = _make_period_labels(returns.index, rebalance)
    boundaries = np.where(np.diff(periods) != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries])

    schedule = {}
    min_days = 504       # ~2 years minimum data requirement
    window_days = window_years * 252

    for b_idx in boundaries:
        date = returns.index[b_idx]
        # Trailing window up to this date (no lookahead)
        start_idx = max(0, b_idx - window_days)
        if b_idx - start_idx < min_days:
            continue  # not enough history yet

        window_rets = returns.iloc[start_idx:b_idx]

        w = run_optimization(
            window_rets, target, min_w, max_w, group_max,
            risk_free_rate, rebalance="daily",
            leverage=leverage, financing_rate=financing_rate,
        )
        schedule[date] = w

    return schedule


def rolling_optimization_analytics(
    schedule: dict[pd.Timestamp, np.ndarray],
) -> dict:
    """Compute analytics for a rolling optimization schedule.

    Returns dict with:
      - dates: list of rebalance dates
      - weight_history: list of weight arrays over time
      - turnover: list of L1 turnover at each rebalance
    """
    dates = sorted(schedule.keys())
    weight_history = [schedule[d] for d in dates]

    turnover = [0.0]  # no turnover at first date
    for i in range(1, len(weight_history)):
        turnover.append(float(np.sum(np.abs(weight_history[i] - weight_history[i - 1]))))

    return {
        "dates": dates,
        "weight_history": weight_history,
        "turnover": turnover,
    }
