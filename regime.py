"""Regime-Based Dynamic Allocation strategy.

Classifies periods by inflation/interest rate regime, optimises separately
per regime, and switches allocations at rebalance dates.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from stats import _make_period_labels


def load_regime_data(path: str | Path) -> pd.DataFrame | None:
    """Load inflation and interest rate data from Excel.

    Expected format (Sheet1):
    - Col A: Date (monthly), Col B: CPI YoY (%)
    - Col C: Date (daily), Col D: Fed Funds Rate (%)

    Returns monthly DataFrame with columns ["CPI", "FedFunds"] or None.
    """
    path = Path(path)
    if not path.exists():
        return None

    try:
        raw = pd.read_excel(path, sheet_name="Sheet1", header=0)
    except Exception:
        return None

    if raw.shape[1] < 4:
        return None

    # Extract CPI series (monthly)
    cpi_dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    cpi_vals = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    cpi_mask = cpi_dates.notna() & cpi_vals.notna()
    cpi = pd.Series(cpi_vals[cpi_mask].values, index=cpi_dates[cpi_mask].values, name="CPI")

    # Extract Fed Funds series (daily)
    ff_dates = pd.to_datetime(raw.iloc[:, 2], errors="coerce")
    ff_vals = pd.to_numeric(raw.iloc[:, 3], errors="coerce")
    ff_mask = ff_dates.notna() & ff_vals.notna()
    ff = pd.Series(ff_vals[ff_mask].values, index=ff_dates[ff_mask].values, name="FedFunds")

    if len(cpi) == 0 or len(ff) == 0:
        return None

    # Resample Fed Funds to monthly average
    ff_monthly = ff.resample("ME").mean().dropna()

    # Merge on month
    cpi.index = cpi.index.to_period("M")
    ff_monthly.index = ff_monthly.index.to_period("M")

    merged = pd.DataFrame({"CPI": cpi, "FedFunds": ff_monthly}).dropna()
    merged.index = merged.index.to_timestamp()

    if len(merged) == 0:
        return None

    return merged


REGIME_LABELS = {
    1: "High Inflation + High IR",
    2: "High Inflation + Low IR",
    3: "Low Inflation + High IR",
    4: "Low Inflation + Low IR",
}


def classify_regimes(macro_data: pd.DataFrame) -> pd.Series:
    """Classify each month into one of 4 regimes using expanding-window medians.

    At each month t, the median is computed from all data up to and including t,
    avoiding look-ahead bias (no future data is used to classify past months).

    Requires a minimum of 24 months of history before classification begins;
    earlier months are forward-filled from the first valid classification.

    1: High Inflation + High IR
    2: High Inflation + Low IR
    3: Low Inflation + High IR
    4: Low Inflation + Low IR
    """
    MIN_HISTORY = 24  # need at least 2 years for stable median

    # Vectorised expanding-window medians (C-implemented, much faster than Python loop)
    cpi_expanding_med = macro_data["CPI"].expanding(min_periods=MIN_HISTORY).median()
    ff_expanding_med = macro_data["FedFunds"].expanding(min_periods=MIN_HISTORY).median()

    high_inf = macro_data["CPI"] > cpi_expanding_med
    high_ir = macro_data["FedFunds"] > ff_expanding_med

    regime = pd.Series(4, index=macro_data.index, dtype=int)  # default: low/low
    regime[high_inf & high_ir] = 1
    regime[high_inf & ~high_ir] = 2
    regime[~high_inf & high_ir] = 3
    # regime 4 is the default (low/low) – already set

    # Forward-fill the first valid classification to cover the initial warm-up period
    if len(macro_data) >= MIN_HISTORY:
        first_valid = regime.iloc[MIN_HISTORY - 1]
        regime.iloc[: MIN_HISTORY - 1] = first_valid

    return regime


def optimize_per_regime(
    returns: pd.DataFrame,
    regime_series: pd.Series,
    target: str,
    min_w: np.ndarray,
    max_w: np.ndarray,
    group_max: dict[str, float],
    risk_free_rate: float = 0.04,
    rebalance: str = "daily",
    dd_constraint: float | None = None,
    dd_returns: pd.DataFrame | None = None,
    dd_asset_starts: dict | None = None,
) -> dict[int, np.ndarray]:
    """Optimise independently on each regime's date subset.

    Returns {regime_label: optimal_weights}.
    """
    from optimizer import run_optimization

    regime_weights = {}

    # Align regime series to returns index (forward-fill monthly to daily)
    regime_daily = regime_series.reindex(returns.index, method="ffill")
    regime_daily = regime_daily.bfill()

    # Use DD-constrained target name if dd_constraint is set
    opt_target = target
    if dd_constraint is not None and target == "Max Sharpe Ratio":
        opt_target = "Max Sharpe (DD \u2264 X%)"

    for label in sorted(regime_series.unique()):
        mask = regime_daily == label
        regime_returns = returns[mask]

        if len(regime_returns) < 63:
            # Not enough data for meaningful 17-asset optimisation — use equal weight
            n = returns.shape[1]
            w = np.ones(n) / n
        else:
            w = run_optimization(
                regime_returns, opt_target, min_w, max_w, group_max,
                risk_free_rate, rebalance=rebalance,
                dd_constraint=dd_constraint,
                dd_returns=dd_returns, dd_asset_starts=dd_asset_starts,
            )

        regime_weights[label] = w

    return regime_weights


def build_regime_schedule(
    returns_index: pd.DatetimeIndex,
    regime_series: pd.Series,
    regime_weights: dict[int, np.ndarray],
    rebalance: str = "annual",
) -> dict[pd.Timestamp, np.ndarray]:
    """Build weights_schedule by checking regime at each rebalance boundary.

    At each rebalance date, looks up the current regime and maps to that
    regime's optimal weights.
    """
    # Align regime to daily
    regime_daily = regime_series.reindex(returns_index, method="ffill")
    regime_daily = regime_daily.bfill()

    # Find rebalance boundaries
    periods = _make_period_labels(returns_index, rebalance)
    boundaries = np.where(np.diff(periods) != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries])

    schedule = {}
    for b_idx in boundaries:
        date = returns_index[b_idx]
        regime = regime_daily.iloc[b_idx]
        if regime in regime_weights:
            schedule[date] = regime_weights[regime].copy()

    return schedule


def regime_analytics(
    regime_series: pd.Series,
    regime_weights: dict[int, np.ndarray],
    asset_names: list[str],
) -> dict:
    """Compute analytics for the Dynamic Strategies UI tab."""
    # Regime duration stats
    regime_changes = regime_series != regime_series.shift(1)
    segments = []
    current_start = regime_series.index[0]
    current_regime = regime_series.iloc[0]

    for i in range(1, len(regime_series)):
        if regime_changes.iloc[i]:
            segments.append({
                "regime": current_regime,
                "start": current_start,
                "end": regime_series.index[i - 1],
                "months": i - regime_series.index.get_loc(current_start) if hasattr(regime_series.index, 'get_loc') else 0,
            })
            current_start = regime_series.index[i]
            current_regime = regime_series.iloc[i]

    # Final segment
    segments.append({
        "regime": current_regime,
        "start": current_start,
        "end": regime_series.index[-1],
    })

    # Calculate months per segment
    for seg in segments:
        delta = seg["end"] - seg["start"]
        seg["months"] = max(1, int(delta.days / 30.44))

    # Regime counts and avg duration
    from collections import Counter
    regime_counts = Counter(s["regime"] for s in segments)
    regime_avg_months = {}
    for label in regime_counts:
        label_segs = [s for s in segments if s["regime"] == label]
        regime_avg_months[label] = np.mean([s["months"] for s in label_segs])

    # Weights per regime
    weights_table = {}
    for label, w in regime_weights.items():
        weights_table[REGIME_LABELS.get(label, f"Regime {label}")] = {
            asset_names[i]: w[i] for i in range(len(w))
        }

    return {
        "segments": segments,
        "current_regime": current_regime,
        "regime_counts": dict(regime_counts),
        "regime_avg_months": regime_avg_months,
        "weights_table": weights_table,
        "n_transitions": len(segments) - 1,
        "regime_labels": REGIME_LABELS,
    }
