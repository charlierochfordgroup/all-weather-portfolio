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
    """Compute daily portfolio log returns, using effective weights if provided."""
    if eff_w is not None:
        return np.sum(returns_vals * eff_w, axis=1)
    return returns_vals @ weights


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


def _periodic_rebal_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
    rebalance: str = "monthly",
    eff_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute daily simple portfolio returns with periodic rebalancing.

    At the start of each period, weights are reset to target.
    Within each period, weights drift with realised simple returns.

    If eff_weights is provided (n_days x n_assets), use per-day target weights
    at rebalance boundaries (for pro-rata redistribution of missing assets).

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

    for seg_start, seg_end in zip(
        boundaries, np.append(boundaries[1:], n_days)
    ):
        # Reset to target weights at period boundary
        cur_w = eff_weights[seg_start].copy() if has_eff else weights.copy()

        for t in range(seg_start, seg_end):
            sr = simple_rets[t]
            day_r = cur_w @ sr
            port_simple[t] = day_r

            # Drift weights
            grown = cur_w * (1.0 + sr)
            total = grown.sum()
            if total > 1e-12:
                cur_w = grown / total
            else:
                cur_w = eff_weights[t].copy() if has_eff else weights.copy()

    return port_simple


def calc_stats(
    returns: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = 0.04,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    rebalance: str = "daily",
    asset_starts: dict[str, pd.Timestamp] | None = None,
) -> PortfolioStats:
    """Calculate portfolio statistics from log returns and weights.

    rebalance: "daily" for continuous rebalancing (original behaviour),
               "monthly" for monthly rebalancing with weight drift.
    asset_starts: when provided, weights for not-yet-available assets are
                  redistributed pro-rata to available assets on each day.
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

    # Compute effective weights (pro-rata redistribution for missing assets)
    eff_w = None
    if asset_starts is not None:
        eff_w = _effective_weights(weights, returns.index, list(returns.columns), asset_starts)

    if rebalance != "daily":
        port_simple = _periodic_rebal_returns(returns, weights, rebalance, eff_w)
        idx = np.cumprod(1.0 + port_simple)
        var_r = np.var(port_simple, ddof=1) if n_days > 1 else 0.0
    else:
        port_log = _daily_port_log(returns.values, weights, eff_w)
        idx = np.exp(np.cumsum(port_log))
        var_r = np.var(port_log, ddof=1) if n_days > 1 else 0.0

    vol = np.sqrt(var_r * TD)

    # CAGR using calendar days
    total_calendar_days = (returns.index[-1] - returns.index[0]).days
    if total_calendar_days > 0 and idx[-1] > 0:
        cagr = idx[-1] ** (365.0 / total_calendar_days) - 1.0
    else:
        cagr = 0.0

    # Sharpe
    sharpe = (cagr - risk_free_rate) / vol if vol > 1e-4 else 0.0

    # Max drawdown and longest drawdown (skip all-zero days)
    nonzero_mask = np.any(returns.values != 0, axis=1)

    if rebalance != "daily":
        port_nz_simple = port_simple[nonzero_mask]
        idx_nz = np.cumprod(1.0 + port_nz_simple)
    else:
        port_nz_log = _daily_port_log(returns.values, weights, eff_w)[nonzero_mask]
        idx_nz = np.exp(np.cumsum(port_nz_log))

    peak = np.maximum.accumulate(idx_nz)
    drawdowns = idx_nz / peak - 1.0
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

    dd_days = 0
    longest_dd = 0
    for i in range(len(idx_nz)):
        if idx_nz[i] < peak[i]:
            dd_days += 1
            longest_dd = max(longest_dd, dd_days)
        else:
            dd_days = 0

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-4 else 0.0

    # Yearly returns (non-zero days grouped by calendar year)
    dates_nz = returns.index[nonzero_mask]
    if rebalance != "daily":
        yearly_df = pd.DataFrame({"ret": port_nz_simple}, index=dates_nz)
        annual = yearly_df.groupby(yearly_df.index.year)["ret"].apply(
            lambda x: np.prod(1.0 + x) - 1.0
        )
    else:
        port_nz_log = _daily_port_log(returns.values, weights, eff_w)[nonzero_mask]
        yearly_df = pd.DataFrame({"ret": port_nz_log}, index=dates_nz)
        annual = yearly_df.groupby(yearly_df.index.year)["ret"].sum()
        annual = np.exp(annual) - 1.0

    if len(annual) > 0:
        best_year = annual.max()
        worst_year = annual.min()
    else:
        best_year = 0.0
        worst_year = 0.0

    # Pct positive days (non-zero days only)
    if rebalance != "daily":
        pct_pos = np.mean(port_nz_simple > 0) if len(port_nz_simple) > 0 else 0.0
    else:
        port_nz_log = _daily_port_log(returns.values, weights, eff_w)[nonzero_mask]
        pct_pos = np.mean(port_nz_log > 0) if len(port_nz_log) > 0 else 0.0

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
    )


def compute_equity_curve(
    returns: pd.DataFrame,
    weights: np.ndarray,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    rebalance: str = "daily",
    asset_starts: dict[str, pd.Timestamp] | None = None,
) -> pd.Series:
    """Compute cumulative equity curve (non-zero days only for clean chart)."""
    if start_date is not None:
        returns = returns[returns.index >= start_date]
    if end_date is not None:
        returns = returns[returns.index <= end_date]
    nonzero_mask = np.any(returns.values != 0, axis=1)
    r = returns[nonzero_mask]

    eff_w = None
    if asset_starts is not None:
        eff_w = _effective_weights(weights, r.index, list(r.columns), asset_starts)

    if rebalance != "daily":
        port_simple = _periodic_rebal_returns(r, weights, rebalance, eff_w)
        cum = np.cumprod(1.0 + port_simple)
    else:
        port_log = _daily_port_log(r.values, weights, eff_w)
        cum = np.exp(np.cumsum(port_log))

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
) -> pd.Series:
    """Compute annual returns."""
    if start_date is not None:
        returns = returns[returns.index >= start_date]
    if end_date is not None:
        returns = returns[returns.index <= end_date]
    nonzero_mask = np.any(returns.values != 0, axis=1)
    r = returns[nonzero_mask]

    eff_w = None
    if asset_starts is not None:
        eff_w = _effective_weights(weights, r.index, list(r.columns), asset_starts)

    if rebalance != "daily":
        port_simple = _periodic_rebal_returns(r, weights, rebalance, eff_w)
        sr = pd.Series(port_simple, index=r.index)
        annual = sr.groupby(sr.index.year).apply(lambda x: np.prod(1.0 + x) - 1.0)
    else:
        port_log = pd.Series(_daily_port_log(r.values, weights, eff_w), index=r.index)
        annual_log = port_log.groupby(port_log.index.year).sum()
        annual = np.exp(annual_log) - 1.0

    return annual.rename("Annual Return")
