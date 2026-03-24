"""Yield Curve Signal Overlay – tilt defensive when rates are rising fast.

Uses the 12-month rolling change in the Fed Funds rate as a macro signal.
When rates are rising above a threshold (default 200 bp/yr), the portfolio
tilts toward defensive assets (short-term treasuries, gold, cash).
When stable or falling, runs the full base allocation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import ASSETS, GROUP_MAP
from stats import _make_period_labels


# Defensive tilt: boost these asset groups
_DEFENSIVE_GROUPS = {"Bonds", "Real Assets", "Alternatives"}
# Specifically favour these assets in defensive mode
_DEFENSIVE_BOOST = {"Short-Term Treasuries": 2.0, "Gold": 1.5, "Cash": 1.5}
# Scale down equity groups
_RISK_GROUPS = {"US Equities", "Intl Equities"}


def _compute_ff_changes(macro_data: pd.DataFrame) -> pd.Series:
    """Compute 12-month rolling change in Fed Funds rate (percentage points)."""
    if macro_data is None or "FedFunds" not in macro_data.columns:
        return pd.Series(dtype=float)
    ff = macro_data["FedFunds"].dropna()
    return ff - ff.shift(12)


def _defensive_weights(base_weights: np.ndarray) -> np.ndarray:
    """Compute defensive tilt weights from base allocation.

    Boosts bonds, gold, cash; reduces equities.
    """
    tilt = base_weights.copy()

    for i, asset in enumerate(ASSETS):
        group = GROUP_MAP[asset]
        if asset in _DEFENSIVE_BOOST:
            tilt[i] *= _DEFENSIVE_BOOST[asset]
        elif group in _RISK_GROUPS:
            tilt[i] *= 0.5  # halve equity exposure

    # Normalise
    total = tilt.sum()
    if total > 1e-12:
        tilt /= total
    return tilt


def build_yield_signal_schedule(
    base_weights: np.ndarray,
    macro_data: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance: str = "monthly",
    threshold_bp: float = 200.0,
) -> dict[pd.Timestamp, np.ndarray]:
    """Build a time-varying weight schedule based on Fed Funds rate changes.

    Parameters
    ----------
    base_weights : target weights when signal is neutral.
    macro_data : DataFrame with 'FedFunds' column (monthly).
    returns : daily log-return DataFrame (for rebalance boundary dates).
    rebalance : rebalance granularity.
    threshold_bp : rising-rate threshold in basis points per year.

    Returns
    -------
    dict mapping timestamps to weight arrays.
    """
    ff_changes = _compute_ff_changes(macro_data)
    if ff_changes.empty:
        return {}

    threshold = threshold_bp / 100.0  # convert bp to percentage points

    defensive = _defensive_weights(base_weights)

    periods = _make_period_labels(returns.index, rebalance)
    boundaries = np.where(np.diff(periods) != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries])

    # Pre-sort ff_changes for fast lookup
    ff_dates = ff_changes.index
    ff_vals = ff_changes.values
    ff_ts = np.array([d.value for d in ff_dates], dtype=np.int64)

    schedule: dict[pd.Timestamp, np.ndarray] = {}

    for b_idx in boundaries:
        date = returns.index[b_idx]

        # Find most recent FF change value
        idx = np.searchsorted(ff_ts, date.value, side="right") - 1
        if idx < 0:
            # No macro data yet — use base weights
            schedule[date] = base_weights.copy()
            continue

        ff_change = ff_vals[idx]

        if np.isnan(ff_change):
            schedule[date] = base_weights.copy()
        elif ff_change > threshold:
            # Rates rising fast — go defensive
            schedule[date] = defensive.copy()
        else:
            # Rates stable or falling — full risk
            schedule[date] = base_weights.copy()

    return schedule


def yield_signal_analytics(macro_data: pd.DataFrame, threshold_bp: float = 200.0) -> dict:
    """Compute analytics for the Dynamic Strategies UI tab.

    Returns dict with 'dates', 'ff_changes', 'signals' (True = defensive).
    """
    ff_changes = _compute_ff_changes(macro_data)
    if ff_changes.empty:
        return {"dates": [], "ff_changes": [], "signals": []}

    threshold = threshold_bp / 100.0
    dates = ff_changes.dropna().index.tolist()
    values = ff_changes.dropna().values.tolist()
    signals = [v > threshold for v in values]

    return {
        "dates": dates,
        "ff_changes": values,
        "signals": signals,
    }
