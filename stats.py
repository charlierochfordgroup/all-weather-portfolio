"""Portfolio statistics calculations."""

import numpy as np
import pandas as pd
from dataclasses import dataclass

TD = 252  # trading days per year


def compute_asset_starts(returns: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Find the first date with non-zero return for each asset."""
    starts = {}
    for col in returns.columns:
        nz = returns[col][returns[col] != 0]
        if len(nz) > 0:
            starts[col] = nz.index[0]
    return starts


def _effective_weights(
    weights: np.ndarray,
    index: pd.DatetimeIndex,
    columns: list[str],
    asset_starts: dict[str, pd.Timestamp],
) -> np.ndarray:
    """Compute per-day effective weights with pro-rata redistribution.

    On days where an asset is not yet available, its target weight is
    redistributed proportionally to the available assets.

    Returns (n_days, n_assets) array.
    """
    n_assets = len(weights)
    avail = np.ones((len(index), n_assets), dtype=bool)
    for i, col in enumerate(columns):
        if col in asset_starts:
            avail[:, i] = index >= asset_starts[col]

    eff = np.where(avail, weights[np.newaxis, :], 0.0)
    row_sums = eff.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    eff /= row_sums
    return eff


def _daily_port_log(returns_vals: np.ndarray, weights: np.ndarray, eff_w) -> np.ndarray:
    """Compute daily portfolio log returns, using effective weights if provided.

    NOTE: This is the geometric approximation (Σ w_i * log(1+r_i)) and is
    retained for legacy compatibility. Use _daily_port_simple for all new code.
    """
    if eff_w is not None:
        return np.sum(returns_vals * eff_w, axis=1)
    return returns_vals @ weights


def _daily_port_simple(returns_vals: np.ndarray, weights: np.ndarray, eff_w) -> np.ndarray:
    """Compute daily portfolio simple returns (arithmetic), using effective weights if provided.

    Converts log returns to simple first, then takes the weighted arithmetic
    sum — the correct formula for a daily-rebalanced portfolio:
        r_portfolio = Σ w_i * (exp(log_r_i) - 1)
    """
    simple_vals = np.exp(returns_vals) - 1.0
    if eff_w is not None:
        return np.sum(simple_vals * eff_w, axis=1)
    return simple_vals @ weights


@dataclass
class PortfolioStats:
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    best_year: float = 0.0
    worst_year: float = 0.0
    pct_positive: float = 0.0
    longest_dd: int = 0
    current_drawdown: float = 0.0


# Map rebalance labels to pandas period frequencies
_REBAL_FREQ = {
    "monthly": "M",
    "quarterly": "Q",
    "semi-annual": "6M",
    "annual": "Y",
}


def _make_period_labels(index: pd.DatetimeIndex, rebalance: str) -> np.ndarray:
    """Create integer period labels for rebalancing boundaries."""
    if rebalance == "semi-annual":
        # pandas to_period('6M') doesn't group correctly; use year*10 + half
        return (index.year * 10 + (index.month > 6).astype(int)).values
    freq = _REBAL_FREQ.get(rebalance, "M")
    return index.to_period(freq).astype("int64")


def _resolve_schedule_weights(
    date: pd.Timestamp,
    weights_schedule: dict[pd.Timestamp, np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    """Look up the most recent weight vector from a schedule for a given date."""
    best_date = None
    for d in weights_schedule:
        if d <= date:
            if best_date is None or d > best_date:
                best_date = d
    if best_date is not None:
        return weights_schedule[best_date].copy()
    return fallback.copy()


def _periodic_rebal_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
    rebalance: str = "monthly",
    eff_weights: np.ndarray | None = None,
    weights_schedule: dict[pd.Timestamp, np.ndarray] | None = None,
) -> np.ndarray:
    """Compute daily simple portfolio returns with periodic rebalancing.

    At the start of each period, weights are reset to target.
    Within each period, weights drift with realised simple returns.

    If eff_weights is provided (n_days x n_assets), use per-day target weights
    at rebalance boundaries (for pro-rata redistribution of missing assets).

    If weights_schedule is provided (dict mapping dates to weight arrays),
    the target weights change over time. At each rebalance boundary, the
    most recent schedule entry is used instead of the static weights.

    rebalance: "monthly", "quarterly", "semi-annual", or "annual".
    """
    simple_rets = np.exp(returns.values) - 1.0  # log -> simple
    periods = _make_period_labels(returns.index, rebalance)
    n_days = len(returns)

    # Find rebalance boundary indices (where period label changes)
    boundaries = np.where(np.diff(periods) != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries])

    port_simple = np.empty(n_days)
    has_eff = eff_weights is not None
    has_schedule = weights_schedule is not None

    # Pre-sort schedule dates and convert to int64 for fast searchsorted lookup
    if has_schedule:
        sorted_sched_dates = sorted(weights_schedule.keys())
        sorted_sched_weights = [weights_schedule[d] for d in sorted_sched_dates]
        _sched_ts = np.array([d.value for d in sorted_sched_dates], dtype=np.int64)

    for seg_start, seg_end in zip(
        boundaries, np.append(boundaries[1:], n_days)
    ):
        # Determine target weights at this boundary
        if has_schedule:
            boundary_date = returns.index[seg_start]
            # Fast binary search for most recent schedule entry
            target_w = weights.copy()
            _idx = np.searchsorted(_sched_ts, boundary_date.value, side='right') - 1
            if _idx >= 0:
                target_w = sorted_sched_weights[_idx].copy()
        else:
            target_w = weights

        # Reset to target weights at period boundary
        if has_eff and not has_schedule:
            cur_w = eff_weights[seg_start].copy()
        else:
            cur_w = target_w.copy()

        # Track whether this schedule uses sub-1.0 weights (cash allocation)
        _target_sum = target_w.sum()
        _has_cash = _target_sum < 1.0 - 1e-8

        for t in range(seg_start, seg_end):
            sr = simple_rets[t]
            day_r = cur_w @ sr
            port_simple[t] = day_r

            # Drift weights
            grown = cur_w * (1.0 + sr)
            total = grown.sum()
            if total > 1e-12:
                if _has_cash:
                    # Preserve cash allocation: normalise risky weights
                    # back to their original total (target_sum), keeping
                    # the remainder as implicit cash.
                    cur_w = grown * (_target_sum / total)
                else:
                    cur_w = grown / total
            else:
                if has_eff and not has_schedule:
                    cur_w = eff_weights[t].copy()
                else:
                    cur_w = target_w.copy()

    return port_simple


def calc_stats(
    returns: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = 0.04,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    rebalance: str = "daily",
    asset_starts: dict[str, pd.Timestamp] | None = None,
    weights_schedule: dict[pd.Timestamp, np.ndarray] | None = None,
) -> PortfolioStats:
    """Calculate portfolio statistics from log returns and weights.

    rebalance: "daily" for continuous rebalancing (original behaviour),
               "monthly" for monthly rebalancing with weight drift.
    asset_starts: when provided, weights for not-yet-available assets are
                  redistributed pro-rata to available assets on each day.
    weights_schedule: when provided, target weights change over time. Forces
                      periodic rebalancing (falls back to "monthly" if "daily").
    """
    if start_date is not None:
        returns = returns[returns.index >= start_date]
    if end_date is not None:
        returns = returns[returns.index <= end_date]

    if returns.empty:
        return PortfolioStats()

    n_days = len(returns)
    if n_days == 0:
        return PortfolioStats()

    # weights_schedule forces periodic rebalancing
    if weights_schedule is not None and rebalance == "daily":
        rebalance = "monthly"

    # Compute effective weights (pro-rata redistribution for missing assets)
    eff_w = None
    if asset_starts is not None and weights_schedule is None:
        eff_w = _effective_weights(weights, returns.index, list(returns.columns), asset_starts)

    if rebalance != "daily":
        port_simple = _periodic_rebal_returns(
            returns, weights, rebalance, eff_w,
            weights_schedule=weights_schedule,
        )
        idx = np.cumprod(1.0 + port_simple)
        # Volatility from simple returns for consistency with Sharpe numerator
        var_r = np.var(port_simple, ddof=1) if n_days > 1 else 0.0
        # Arithmetic mean of simple returns for Sharpe numerator
        arith_mean_daily = np.mean(port_simple) if n_days > 0 else 0.0
    else:
        # Arithmetic simple returns: convert log→simple then weighted sum.
        # Correct formula for a daily-rebalanced portfolio.
        port_simple = _daily_port_simple(returns.values, weights, eff_w)
        idx = np.cumprod(1.0 + port_simple)
        var_r = np.var(port_simple, ddof=1) if n_days > 1 else 0.0
        arith_mean_daily = np.mean(port_simple) if n_days > 0 else 0.0

    vol = np.sqrt(var_r * TD)

    # Arithmetic mean annualised (for Sharpe — same basis as vol)
    arith_annual = arith_mean_daily * TD

    # CAGR using calendar days (for display, Calmar, etc.)
    total_calendar_days = (returns.index[-1] - returns.index[0]).days
    if total_calendar_days > 0 and idx[-1] > 0:
        cagr = idx[-1] ** (365.0 / total_calendar_days) - 1.0
    else:
        cagr = 0.0

    # Sharpe: arithmetic excess return / volatility (both from simple returns)
    sharpe = (arith_annual - risk_free_rate) / vol if vol > 1e-4 else 0.0

    # Max drawdown and longest drawdown (skip all-zero days)
    nonzero_mask = np.any(returns.values != 0, axis=1)

    # Both daily and periodic paths now produce port_simple (arithmetic simple returns)
    port_nz_simple = port_simple[nonzero_mask]
    idx_nz = np.cumprod(1.0 + port_nz_simple)

    peak = np.maximum.accumulate(idx_nz)
    drawdowns = idx_nz / peak - 1.0
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

    # Vectorised longest drawdown: find max consecutive run where idx < peak
    in_dd = idx_nz < peak
    if np.any(in_dd):
        changes = np.diff(np.concatenate(([False], in_dd, [False])).astype(np.int8))
        run_starts = np.where(changes == 1)[0]
        run_ends = np.where(changes == -1)[0]
        longest_dd = int(np.max(run_ends - run_starts))
    else:
        longest_dd = 0

    # Current drawdown: latest equity value relative to peak
    current_dd = drawdowns[-1] if len(drawdowns) > 0 else 0.0

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-4 else 0.0

    # Yearly returns and pct positive — both paths use simple returns
    dates_nz = returns.index[nonzero_mask]
    yearly_df = pd.DataFrame({"ret": port_nz_simple}, index=dates_nz)
    annual = yearly_df.groupby(yearly_df.index.year)["ret"].apply(
        lambda x: np.prod(1.0 + x) - 1.0
    )

    if len(annual) > 0:
        best_year = annual.max()
        worst_year = annual.min()
    else:
        best_year = 0.0
        worst_year = 0.0

    pct_pos = np.mean(port_nz_simple > 0) if len(port_nz_simple) > 0 else 0.0

    return PortfolioStats(
        cagr=cagr,
        volatility=vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        best_year=best_year,
        worst_year=worst_year,
        pct_positive=pct_pos,
        longest_dd=longest_dd,
        current_drawdown=current_dd,
    )


def compute_equity_curve(
    returns: pd.DataFrame,
    weights: np.ndarray,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    rebalance: str = "daily",
    asset_starts: dict[str, pd.Timestamp] | None = None,
    weights_schedule: dict[pd.Timestamp, np.ndarray] | None = None,
) -> pd.Series:
    """Compute cumulative equity curve (non-zero days only for clean chart)."""
    if start_date is not None:
        returns = returns[returns.index >= start_date]
    if end_date is not None:
        returns = returns[returns.index <= end_date]
    nonzero_mask = np.any(returns.values != 0, axis=1)
    r = returns[nonzero_mask]

    if weights_schedule is not None and rebalance == "daily":
        rebalance = "monthly"

    eff_w = None
    if asset_starts is not None and weights_schedule is None:
        eff_w = _effective_weights(weights, r.index, list(r.columns), asset_starts)

    if rebalance != "daily":
        port_simple = _periodic_rebal_returns(
            r, weights, rebalance, eff_w,
            weights_schedule=weights_schedule,
        )
    else:
        port_simple = _daily_port_simple(r.values, weights, eff_w)

    cum = np.cumprod(1.0 + port_simple)
    return pd.Series(cum, index=r.index, name="Equity")


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute drawdown series from equity curve."""
    peak = equity.cummax()
    return (equity / peak - 1.0).rename("Drawdown")


def compute_annual_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    rebalance: str = "daily",
    asset_starts: dict[str, pd.Timestamp] | None = None,
    weights_schedule: dict[pd.Timestamp, np.ndarray] | None = None,
) -> pd.Series:
    """Compute annual returns."""
    if start_date is not None:
        returns = returns[returns.index >= start_date]
    if end_date is not None:
        returns = returns[returns.index <= end_date]
    nonzero_mask = np.any(returns.values != 0, axis=1)
    r = returns[nonzero_mask]

    if weights_schedule is not None and rebalance == "daily":
        rebalance = "monthly"

    eff_w = None
    if asset_starts is not None and weights_schedule is None:
        eff_w = _effective_weights(weights, r.index, list(r.columns), asset_starts)

    if rebalance != "daily":
        port_simple = _periodic_rebal_returns(
            r, weights, rebalance, eff_w,
            weights_schedule=weights_schedule,
        )
    else:
        port_simple = _daily_port_simple(r.values, weights, eff_w)

    sr = pd.Series(port_simple, index=r.index)
    annual = sr.groupby(sr.index.year).apply(lambda x: np.prod(1.0 + x) - 1.0)
    return annual.rename("Annual Return")
