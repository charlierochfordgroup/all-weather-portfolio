"""Streamlit dashboard for All Weather Portfolio Analyser."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import hashlib
from pathlib import Path

from data import ASSETS, GROUP_MAP, GROUP_NAMES, load_data
from stats import (
    calc_stats, compute_equity_curve, compute_drawdown_series,
    compute_annual_returns, compute_asset_starts,
)
from optimizer import run_optimization, _group_indices
from cfd import analyze_cfd
from regime import load_regime_data, classify_regimes, optimize_per_regime, build_regime_schedule, regime_analytics, REGIME_LABELS
from dd_momentum import compute_dd_adjustments, build_dd_momentum_schedule, dd_analytics

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
PRECOMPUTED_FILE = Path(__file__).resolve().parent / "precomputed_weights.pkl"

st.set_page_config(page_title="All Weather Portfolio", layout="wide")
st.title("All Weather Portfolio Analyser")

# ──────────────────────────────────────────────
# SIDEBAR: Global settings
# ──────────────────────────────────────────────
st.sidebar.header("Data Source")

default_path = str(Path(__file__).resolve().parent / "data_template.xlsx")
data_path = st.sidebar.text_input("Excel file path", value=default_path)
use_processing = st.sidebar.checkbox("Use pre-computed returns (Processing sheet)", value=True)

# Load data
@st.cache_data(show_spinner="Loading data...")
def cached_load(path, use_proc):
    return load_data(path, use_proc)

try:
    all_returns = cached_load(data_path, use_processing)
    data_loaded = True
except Exception as e:
    st.error(f"Failed to load data: {e}")
    data_loaded = False
    all_returns = pd.DataFrame()

if not data_loaded:
    st.stop()

# Filter to available assets (Bitcoin exclusion applied after sidebar is set up)
available = [a for a in ASSETS if a in all_returns.columns]
returns_full = all_returns[available]

# Placeholder — Bitcoin exclusion applied after sidebar checkbox

# Compute asset start dates (first non-zero return per asset)
asset_starts = compute_asset_starts(returns_full)

if not asset_starts:
    st.error("No valid return data found.")
    st.stop()

# Overlap start = latest first-data date (when ALL assets have data, e.g. ~2010)
overlap_start_date = max(asset_starts.values()).date()
data_start_earliest = min(asset_starts.values()).date()
data_end = returns_full.index[-1].date()

st.sidebar.header("Analysis Settings")

# Inception mode: if user clicked inception, override the default start date
_start_default = overlap_start_date
if st.session_state.get("_use_inception", False):
    _start_default = data_start_earliest

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Backtest Start", value=_start_default,
                             min_value=data_start_earliest, max_value=data_end,
                             format="DD/MM/YYYY")
end_date = col2.date_input("End Date", value=data_end,
                           min_value=data_start_earliest, max_value=data_end,
                           format="DD/MM/YYYY")
if st.sidebar.button("Inception (Earliest Data)"):
    st.session_state["_use_inception"] = True
    st.rerun()

exclude_bitcoin = st.sidebar.checkbox("Exclude Bitcoin", value=False)

# Determine if we're in extended backtest mode (start before all assets exist)
extended_backtest = start_date < overlap_start_date
if extended_backtest:
    st.sidebar.caption(
        f"Extended backtest: assets without data before their start date "
        f"have weights redistributed pro-rata. "
        f"Optimiser uses overlap period from {overlap_start_date}."
    )

risk_free_pct = st.sidebar.number_input("Risk-Free Rate (%)", value=4.0, min_value=0.0, max_value=20.0, step=0.5, format="%.1f")
risk_free_rate = risk_free_pct / 100.0

_REBAL_OPTIONS = {
    "Daily (Continuous)": "daily",
    "Monthly": "monthly",
    "Quarterly": "quarterly",
    "Semi-Annual": "semi-annual",
    "Annual": "annual",
}
rebalance_freq = st.sidebar.selectbox("Rebalancing Frequency", list(_REBAL_OPTIONS.keys()))
rebalance = _REBAL_OPTIONS[rebalance_freq]

_DD_LEVELS = [5, 10, 15, 20, 25, 30]
dd_constraint_pct = st.sidebar.selectbox(
    "Max Drawdown Constraint (%)",
    _DD_LEVELS,
    index=_DD_LEVELS.index(20),
    key="dd_constraint_select",
)
dd_constraint_val = dd_constraint_pct / 100.0

# DD Momentum bump strength
st.sidebar.header("Dynamic Strategy Settings")
dd_bump_max_pct = st.sidebar.slider(
    "DD Momentum Max Bump (%)",
    min_value=10, max_value=100, value=50, step=5,
    help="Maximum allocation bump for the most deeply drawn-down asset.",
    key="dd_bump_max",
)
dd_bump_max = dd_bump_max_pct / 100.0

# Base strategies (always computed once)
_BASE_TARGETS = [
    "Max Sharpe Ratio", "Min Volatility", "Max Calmar Ratio",
    "Minimize Max Drawdown",
    "Inverse Volatility", "Equal Risk Contribution", "Hierarchical Risk Parity",
]

# DD-constrained targets (computed for each DD level)
_DD_TARGETS = ["Max Sharpe (DD \u2264 X%)", "Max Calmar (DD \u2264 X%)"]

# Slice returns to date range
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)
overlap_ts = pd.Timestamp(overlap_start_date)

# Full backtest returns (may start before all assets exist)
returns = returns_full[(returns_full.index >= start_ts) & (returns_full.index <= end_ts)]

# Optimisation returns: use overlap period where all assets have data
if extended_backtest:
    opt_returns = returns_full[(returns_full.index >= overlap_ts) & (returns_full.index <= end_ts)]
    bt_asset_starts = asset_starts  # pass to stats for pro-rata redistribution
else:
    opt_returns = returns
    bt_asset_starts = None  # all assets available, no redistribution needed

n = len(ASSETS)

# Default constraints (used to initialise session state)
_DEFAULT_MIN = {a: 0.5 for a in ASSETS}
_DEFAULT_MAX = {
    "Cash": 20.0, "Nasdaq": 30.0, "S&P 500": 30.0, "Russell 2000": 30.0,
    "ASX200": 30.0, "Emerging Markets": 30.0,
    "Corporate Bonds": 40.0, "Long-Term Treasuries": 40.0, "Short-Term Treasuries": 40.0,
    "Real Estate": 20.0, "Commodities": 10.0, "Gold": 20.0,
    "Bitcoin": 15.0, "Infrastructure": 20.0,
    "Japan Equities": 30.0, "UK Equities": 30.0, "EU Equities": 30.0,
}
_DEFAULT_GROUP_MAX = {
    "US Equities": 35.0, "Intl Equities": 30.0, "Bonds": 40.0,
    "Real Assets": 30.0, "Alternatives": 20.0,
}

# Initialise constraint session state
if "asset_min" not in st.session_state:
    st.session_state.asset_min = dict(_DEFAULT_MIN)
if "asset_max" not in st.session_state:
    st.session_state.asset_max = dict(_DEFAULT_MAX)
if "group_max" not in st.session_state:
    st.session_state.group_max = dict(_DEFAULT_GROUP_MAX)

# Build constraint arrays from session state (values stored as percentages)
min_w_active = np.array([st.session_state.asset_min.get(a, 1.0) / 100.0 for a in ASSETS])
max_w_active = np.array([st.session_state.asset_max.get(a, 30.0) / 100.0 for a in ASSETS])
group_max_active = {g: v / 100.0 for g, v in st.session_state.group_max.items()}

# Exclude Bitcoin: force min=0, max=0 so optimizer gives it zero weight
if exclude_bitcoin:
    btc_idx = ASSETS.index("Bitcoin")
    min_w_active[btc_idx] = 0.0
    max_w_active[btc_idx] = 0.0

# ──────────────────────────────────────────────
# FIXED BENCHMARK PORTFOLIOS
# ──────────────────────────────────────────────
# Ray Dalio All Weather: 30% Stocks, 40% LT Bonds, 15% IT Bonds, 7.5% Gold, 7.5% Commodities
_DALIO_WEIGHTS = np.zeros(n)
_DALIO_MAP = {
    "S&P 500": 0.30,
    "Long-Term Treasuries": 0.40,
    "Short-Term Treasuries": 0.15,
    "Gold": 0.075,
    "Commodities": 0.075,
}
for _asset, _w in _DALIO_MAP.items():
    _DALIO_WEIGHTS[ASSETS.index(_asset)] = _w

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "portfolios" not in st.session_state:
    st.session_state.portfolios = {}

# ──────────────────────────────────────────────
# AUTO-COMPUTE DEFAULT PORTFOLIOS (disk-cached)
# ──────────────────────────────────────────────
min_w_default = min_w_active
max_w_default = max_w_active
default_group_max = group_max_active

def _cache_key(sd, ed, rf, rb):
    # Include constraint fingerprint so changes to min/max/group trigger recomputation
    constraints_str = (f"{sorted(st.session_state.asset_min.items())}_"
                       f"{sorted(st.session_state.asset_max.items())}_"
                       f"{sorted(st.session_state.group_max.items())}_"
                       f"excl_btc={exclude_bitcoin}")
    raw = f"{sd}_{ed}_{rf}_{rb}_{constraints_str}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _compute_all(opt_data, bt_data, rf, rb, a_starts):
    """Compute base strategies + DD-constrained variants for all DD levels.

    opt_data:  returns for the overlap period (used by the optimiser).
    bt_data:   returns for the full backtest period (used for stats).
    a_starts:  asset start dates for pro-rata redistribution (None if not extended).

    Returns (base_results, dd_results) where:
      base_results: {target: {"weights": w, "stats": s}}
      dd_results:   {dd_pct: {target: {"weights": w, "stats": s}}}
    """
    base_results = {}
    for tgt in _BASE_TARGETS:
        # Minimize Max Drawdown needs to evaluate DD on full backtest period
        if tgt == "Minimize Max Drawdown" and a_starts is not None:
            w = run_optimization(
                opt_data, tgt, min_w_default, max_w_default, default_group_max,
                rf, rebalance="daily",
                dd_returns=bt_data, dd_asset_starts=a_starts,
            )
        else:
            w = run_optimization(
                opt_data, tgt, min_w_default, max_w_default, default_group_max,
                rf, rebalance="daily",
            )
        s = calc_stats(bt_data, w, rf, rebalance=rb, asset_starts=a_starts)
        base_results[tgt] = {"weights": w, "stats": s}

    dd_results = {}
    # Process DD levels from tightest to loosest, warm-starting from the
    # previous level's result so the optimizer converges much faster.
    prev_w = {}  # {target: weights from previous level}
    for dd_pct in sorted(_DD_LEVELS):
        dd_val = dd_pct / 100.0
        dd_results[dd_pct] = {}
        for tgt in _DD_TARGETS:
            w = run_optimization(
                opt_data, tgt, min_w_default, max_w_default, default_group_max,
                rf, rebalance="daily", dd_constraint=dd_val,
                current_weights=prev_w.get(tgt),
                dd_returns=bt_data, dd_asset_starts=a_starts,
            )
            s = calc_stats(bt_data, w, rf, rebalance=rb, asset_starts=a_starts)
            dd_results[dd_pct][tgt] = {"weights": w, "stats": s}
            prev_w[tgt] = w

    return base_results, dd_results


def _assemble_portfolios(base_results, dd_results, dd_pct):
    """Combine base strategies with the selected DD level's constrained variants."""
    portfolios = dict(base_results)
    dd_data = dd_results.get(dd_pct, {})
    for tgt in _DD_TARGETS:
        if tgt in dd_data:
            display_name = tgt.replace("X%", f"{dd_pct}%")
            portfolios[display_name] = dd_data[tgt]
    return portfolios


# Cache keys — include both backtest start and overlap start so extended backtests
# get their own cache entries. No DD param since all levels are pre-computed together.
cache_key = _cache_key(f"{start_date}_{overlap_start_date}", end_date, risk_free_rate, rebalance)
cache_file = CACHE_DIR / f"all_portfolios_{cache_key}.pkl"

# Opt weights only depend on the overlap period (optimiser input)
opt_cache_key = _cache_key(overlap_start_date if extended_backtest else start_date,
                           end_date, risk_free_rate, "daily")
opt_cache_file = CACHE_DIR / f"all_opt_weights_{opt_cache_key}.pkl"


def _constraints_are_default():
    """Check if the user's current constraints match the shipped defaults."""
    return (st.session_state.asset_min == _DEFAULT_MIN and
            st.session_state.asset_max == _DEFAULT_MAX and
            st.session_state.group_max == _DEFAULT_GROUP_MAX and
            abs(risk_free_rate - 0.04) < 1e-9 and
            not exclude_bitcoin)


def _stats_from_weights(base_w, dd_w, bt_data, rf, rb, a_starts):
    """Compute stats for pre-loaded weights (fast — no optimisation)."""
    base_results = {}
    for tgt, w in base_w.items():
        s = calc_stats(bt_data, w, rf, rebalance=rb, asset_starts=a_starts)
        base_results[tgt] = {"weights": w, "stats": s}
    dd_results = {}
    for dd_pct, targets in dd_w.items():
        dd_results[dd_pct] = {}
        for tgt, w in targets.items():
            s = calc_stats(bt_data, w, rf, rebalance=rb, asset_starts=a_starts)
            dd_results[dd_pct][tgt] = {"weights": w, "stats": s}
    return base_results, dd_results


def _load_or_compute(opt_data, bt_data, rf, rb,
                     _cache_file, _opt_cache_file, a_starts):
    """Load cached portfolios or compute them.

    Returns (base_results, dd_results).
    Priority: 1) disk cache  2) pre-computed weights  3) cached opt weights  4) full optimisation.
    """
    # Check full cache (exact match including rebalance)
    if _cache_file.exists():
        try:
            with open(_cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    # Try pre-computed weights (shipped with repo) if constraints are default
    if _constraints_are_default() and PRECOMPUTED_FILE.exists():
        try:
            with open(PRECOMPUTED_FILE, "rb") as f:
                pre = pickle.load(f)
            base_results, dd_results = _stats_from_weights(
                pre["base_weights"], pre["dd_weights"], bt_data, rf, rb, a_starts)
            with open(_cache_file, "wb") as f:
                pickle.dump((base_results, dd_results), f)
            return base_results, dd_results
        except Exception:
            pass

    # Check if we have cached optimised weights (rebalance-independent)
    opt_weights = None
    if _opt_cache_file.exists():
        try:
            with open(_opt_cache_file, "rb") as f:
                opt_weights = pickle.load(f)
        except Exception:
            pass

    if opt_weights is not None:
        base_w, dd_w = opt_weights
        base_results, dd_results = _stats_from_weights(
            base_w, dd_w, bt_data, rf, rb, a_starts)
    else:
        base_results, dd_results = _compute_all(opt_data, bt_data, rf, rb, a_starts)
        # Cache weights separately
        base_w = {t: d["weights"] for t, d in base_results.items()}
        dd_w = {
            dd_pct: {t: d["weights"] for t, d in targets.items()}
            for dd_pct, targets in dd_results.items()
        }
        with open(_opt_cache_file, "wb") as f:
            pickle.dump((base_w, dd_w), f)

    # Cache full results
    with open(_cache_file, "wb") as f:
        pickle.dump((base_results, dd_results), f)
    return base_results, dd_results


if st.session_state.get("defaults_cache_key") != cache_key:
    with st.sidebar:
        with st.spinner("Computing portfolios..."):
            base_results, dd_results = _load_or_compute(
                opt_returns, returns, risk_free_rate, rebalance,
                cache_file, opt_cache_file, bt_asset_starts,
            )
            st.session_state.base_results = base_results
            st.session_state.dd_results = dd_results
            st.session_state.defaults_cache_key = cache_key

# Assemble displayed portfolios based on selected DD level (instant)
if "base_results" in st.session_state:
    st.session_state.portfolios = _assemble_portfolios(
        st.session_state.base_results,
        st.session_state.dd_results,
        dd_constraint_pct,
    )

# Add Dalio All Weather as a fixed benchmark (no optimisation needed)
_dalio_stats = calc_stats(returns, _DALIO_WEIGHTS, risk_free_rate,
                          rebalance=rebalance, asset_starts=bt_asset_starts)
st.session_state.portfolios["Dalio All Weather"] = {
    "weights": _DALIO_WEIGHTS,
    "stats": _dalio_stats,
}

# ──────────────────────────────────────────────
# DYNAMIC STRATEGIES
# ──────────────────────────────────────────────
_MACRO_FILE = Path(__file__).resolve().parent / "Inflation and IR.xlsx"

# --- DD Momentum Strategy ---
# Always available (uses only return data)
# Uses DD-constrained Max Sharpe as base if available, otherwise unconstrained
if "base_results" in st.session_state and "Max Sharpe Ratio" in st.session_state.base_results:
    _dd_base_key = f"Max Sharpe (DD \u2264 X%)"
    if ("dd_results" in st.session_state and dd_constraint_pct in st.session_state.dd_results
            and _dd_base_key in st.session_state.dd_results[dd_constraint_pct]):
        _ms_weights = st.session_state.dd_results[dd_constraint_pct][_dd_base_key]["weights"]
    else:
        _ms_weights = st.session_state.base_results["Max Sharpe Ratio"]["weights"]

    # Annual checkpoint dates within backtest range
    _years = sorted(set(returns.index.year))
    _checkpoints = []
    for y in _years:
        yr_dates = returns.index[returns.index.year == y]
        if len(yr_dates) > 0:
            _checkpoints.append(yr_dates[0])  # first trading day of each year

    _dd_adj = compute_dd_adjustments(returns, _checkpoints, bump_max=dd_bump_max)
    _dd_schedule = build_dd_momentum_schedule(_ms_weights, _dd_adj)

    _rebal_for_dynamic = rebalance if rebalance != "daily" else "annual"
    _dd_stats = calc_stats(
        returns, _ms_weights, risk_free_rate,
        rebalance=_rebal_for_dynamic, asset_starts=bt_asset_starts,
        weights_schedule=_dd_schedule,
    )
    # Representative weights = latest in schedule
    _dd_repr_w = _dd_schedule[max(_dd_schedule.keys())] if _dd_schedule else _ms_weights
    st.session_state.portfolios["DD P-Value Momentum (time-varying)"] = {
        "weights": _dd_repr_w,
        "stats": _dd_stats,
        "weights_schedule": _dd_schedule,
        "strategy_type": "dd_momentum",
    }
    st.session_state._dd_schedule = _dd_schedule
    st.session_state._dd_checkpoints = _checkpoints

# --- Regime Strategy ---
_macro_data = load_regime_data(_MACRO_FILE)
st.session_state._macro_available = _macro_data is not None

if _macro_data is not None and "base_results" in st.session_state:
    _regime_series = classify_regimes(_macro_data)
    st.session_state._regime_series = _regime_series

    # Only optimise regimes if we haven't cached them for this config
    _regime_cache_key = _cache_key(f"regime_{overlap_start_date}_{dd_constraint_pct}", end_date, risk_free_rate, "daily")
    if st.session_state.get("_regime_cache_key") != _regime_cache_key:
        with st.sidebar:
            with st.spinner("Computing regime portfolios..."):
                _regime_weights = optimize_per_regime(
                    opt_returns, _regime_series, "Max Sharpe Ratio",
                    min_w_default, max_w_default, default_group_max,
                    risk_free_rate, rebalance="daily",
                )
                st.session_state._regime_weights = _regime_weights
                st.session_state._regime_cache_key = _regime_cache_key
    else:
        _regime_weights = st.session_state._regime_weights

    _rebal_for_regime = rebalance if rebalance != "daily" else "annual"
    _regime_schedule = build_regime_schedule(
        returns.index, _regime_series, _regime_weights, _rebal_for_regime,
    )
    st.session_state._regime_schedule = _regime_schedule

    # Average weights across regimes as representative
    _regime_repr_w = np.mean(list(_regime_weights.values()), axis=0)
    _regime_stats = calc_stats(
        returns, _regime_repr_w, risk_free_rate,
        rebalance=_rebal_for_regime, asset_starts=bt_asset_starts,
        weights_schedule=_regime_schedule,
    )
    st.session_state.portfolios["Regime-Based (time-varying)"] = {
        "weights": _regime_repr_w,
        "stats": _regime_stats,
        "weights_schedule": _regime_schedule,
        "strategy_type": "regime",
    }

portfolio_names = list(st.session_state.portfolios.keys())

# ──────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────

_CHART_COLORS = [
    "#2962FF", "#FF6D00", "#00C853", "#AA00FF", "#FF1744",
    "#00BCD4", "#FFD600", "#E91E63", "#4CAF50", "#795548",
    "#F57C00",
]


def _stats_row_numeric(s):
    """Build a dict of numeric stats for sorting (percentages pre-multiplied by 100)."""
    return {
        "CAGR": s.cagr * 100,
        "Volatility": s.volatility * 100,
        "Sharpe": s.sharpe,
        "Max Drawdown": s.max_drawdown * 100,
        "Calmar": s.calmar,
        "Best Year": s.best_year * 100,
        "Worst Year": s.worst_year * 100,
        "% Pos Days": s.pct_positive * 100,
        "Longest DD": s.longest_dd,
    }

_STATS_COL_CONFIG = {
    "CAGR": st.column_config.NumberColumn("CAGR", format="%.2f%%"),
    "Volatility": st.column_config.NumberColumn("Volatility", format="%.2f%%"),
    "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
    "Max Drawdown": st.column_config.NumberColumn("Max Drawdown", format="%.2f%%"),
    "Calmar": st.column_config.NumberColumn("Calmar", format="%.2f"),
    "Best Year": st.column_config.NumberColumn("Best Year", format="%.2f%%"),
    "Worst Year": st.column_config.NumberColumn("Worst Year", format="%.2f%%"),
    "% Pos Days": st.column_config.NumberColumn("% Pos Days", format="%.2f%%"),
    "Longest DD": st.column_config.NumberColumn("Longest DD", format="%d days"),
}


def _risk_analytics(weights, label="Portfolio"):
    """Render risk analytics block for given weights."""
    returns_clean = returns[ASSETS].clip(lower=-0.10, upper=0.10)
    cov = returns_clean.cov().values * 252

    # Override Cash covariance with realistic money-market volatility
    _CASH_IDX = ASSETS.index("Cash")
    _cash_ann_vol = 0.002
    cov[_CASH_IDX, :] = 0.0
    cov[:, _CASH_IDX] = 0.0
    cov[_CASH_IDX, _CASH_IDX] = _cash_ann_vol ** 2

    sigma_w = cov @ weights
    port_vol = np.sqrt(weights @ cov @ weights)
    if port_vol > 1e-6:
        vol_contrib = weights * sigma_w / port_vol
        vol_pct = vol_contrib / vol_contrib.sum() * 100
    else:
        vol_contrib = np.zeros(n)
        vol_pct = np.zeros(n)

    # Clip negative risk contributions to 0 (hedging assets)
    vol_pct = np.maximum(vol_pct, 0.0)

    rc_left, rc_right = st.columns(2)

    with rc_left:
        st.subheader("Volatility Contribution")
        vol_df = pd.DataFrame({
            "Asset": ASSETS,
            "Weight": weights * 100,
            "Ann. Vol": [np.sqrt(cov[i, i]) * 100 for i in range(n)],
            "Vol Contrib": vol_contrib * 100,
            "% of Total": vol_pct,
        })
        st.dataframe(
            vol_df,
            column_config={
                "Asset": st.column_config.TextColumn("Asset"),
                "Weight": st.column_config.NumberColumn("Weight", format="%.2f%%"),
                "Ann. Vol": st.column_config.NumberColumn("Ann. Vol", format="%.2f%%"),
                "Vol Contrib": st.column_config.NumberColumn("Vol Contrib", format="%.2f%%"),
                "% of Total": st.column_config.NumberColumn("% of Total", format="%.1f%%"),
            },
            width="stretch",
            hide_index=True,
        )
        st.caption("Hedging assets (negative marginal contribution) are shown as 0%.")

    with rc_right:
        st.subheader("Risk Contribution Chart")
        sort_idx = np.argsort(vol_pct)[::-1]
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Bar(
            x=[ASSETS[i] for i in sort_idx],
            y=[vol_pct[i] for i in sort_idx],
            marker_color="#2962FF",
        ))
        fig_rc.update_layout(
            yaxis_title="% of Portfolio Risk",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig_rc, width="stretch")

    # Correlation heatmap
    st.subheader("Correlation Matrix")
    corr = returns_clean.corr()
    short_names = ["Cash", "NDX", "SPX", "R2K", "ASX", "EM", "Corp", "LTT", "STT", "RE", "Cmdty", "Gold", "BTC", "Infra", "Jpn", "UK", "EU"]

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=short_names,
        y=short_names,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 9},
    ))
    fig_corr.update_layout(
        height=500,
        template="plotly_white",
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig_corr, width="stretch")


def _make_pie_data(weights, threshold=0.03):
    """Group small allocations into 'Other' for cleaner pie charts."""
    items = [(ASSETS[i], weights[i]) for i in range(n) if weights[i] > 0.001]
    items.sort(key=lambda x: x[1], reverse=True)
    labels, vals, other = [], [], 0.0
    for lbl, v in items:
        if v >= threshold:
            labels.append(lbl)
            vals.append(v)
        else:
            other += v
    if other > 0.001:
        labels.append("Other")
        vals.append(other)
    return labels, vals


# ──────────────────────────────────────────────
# MAIN: TABS
# ──────────────────────────────────────────────
tab_compare, tab_dynamic, tab_cfd, tab_settings = st.tabs(["Compare", "Dynamic Strategies", "CFD Analysis", "Settings"])

# ======================================================================
# TAB 1: COMPARE
# ======================================================================
with tab_compare:
    if not portfolio_names:
        st.info("Computing default portfolios...")
    else:
        # Strategy Comparison (all portfolios)
        st.header("Strategy Comparison")
        st.caption(f"Backtest period: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')} "
                   f"({(end_date - start_date).days // 365} years). "
                   f"CAGR annualised using calendar days; volatility using {252} trading days/year.")
        comp_rows = []
        for pname, pdata in st.session_state.portfolios.items():
            comp_rows.append({"Strategy": pname, **_stats_row_numeric(pdata["stats"])})
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(
            comp_df,
            column_config={"Strategy": st.column_config.TextColumn("Strategy"), **_STATS_COL_CONFIG},
            width="stretch",
            hide_index=True,
        )

        with st.expander("Allocation Comparison (weights by strategy)"):
            alloc_rows = []
            for pname, pdata in st.session_state.portfolios.items():
                row = {"Strategy": pname}
                for i, asset in enumerate(ASSETS):
                    row[asset] = pdata["weights"][i] * 100
                alloc_rows.append(row)
            alloc_df = pd.DataFrame(alloc_rows)
            alloc_config = {"Strategy": st.column_config.TextColumn("Strategy")}
            for asset in ASSETS:
                alloc_config[asset] = st.column_config.NumberColumn(asset, format="%.2f%%")
            st.dataframe(alloc_df, column_config=alloc_config, width="stretch", hide_index=True)

        # ── Charts ──
        st.header("Charts")

        # Equity curve (log scale) — multiselect
        # Preserve user selections when DD constraint changes portfolio names.
        # When DD changes, the DD-constrained names change (e.g. "DD ≤ 20%" -> "DD ≤ 25%").
        # We keep the base strategy selections and swap the DD-labelled ones.
        _prev_dd = st.session_state.get("_eq_dd_pct")
        if _prev_dd is not None and _prev_dd != dd_constraint_pct and "eq_curve_select" in st.session_state:
            old_suffix = f"{_prev_dd}%)"
            new_suffix = f"{dd_constraint_pct}%)"
            updated = []
            for p in st.session_state.eq_curve_select:
                if old_suffix in p:
                    new_name = p.replace(old_suffix, new_suffix)
                    if new_name in portfolio_names:
                        updated.append(new_name)
                elif p in portfolio_names:
                    updated.append(p)
            st.session_state.eq_curve_select = updated
        st.session_state._eq_dd_pct = dd_constraint_pct

        eq_selected = st.multiselect(
            "Select portfolios",
            portfolio_names,
            default=portfolio_names,
            key="eq_curve_select",
        )

        # Pre-compute equity curves once — reused for equity chart, ticks, and drawdown
        _eq_cache = {}
        for pname in set(eq_selected) | set(st.session_state.get("dd_select", [])):
            if pname in st.session_state.portfolios:
                pdata = st.session_state.portfolios[pname]
                pw = pdata["weights"]
                ws = pdata.get("weights_schedule")
                _eq_cache[pname] = compute_equity_curve(
                    returns, pw, start_ts, end_ts, rebalance=rebalance,
                    asset_starts=bt_asset_starts, weights_schedule=ws,
                )

        fig_eq = go.Figure()
        for idx, pname in enumerate(eq_selected):
            eq = _eq_cache[pname]
            color = _CHART_COLORS[idx % len(_CHART_COLORS)]
            fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name=pname, line=dict(color=color)))
        # Compute clean log-scale ticks for the equity curve
        _eq_all = np.concatenate([_eq_cache[p].values for p in eq_selected])
        _eq_ymin = max(_eq_all.min() * 0.8, 0.5)
        _eq_ymax = _eq_all.max() * 1.2
        _eq_ticks = []
        # Generate ticks: 0.5, 1, 2, 5, 10, 20, 50, 100, ...
        for exp in range(-1, 5):
            base = 10 ** exp
            for mult in [1, 2, 5]:
                val = base * mult
                if _eq_ymin <= val <= _eq_ymax:
                    _eq_ticks.append(val)
        if not _eq_ticks:
            _eq_ticks = [1]

        def _fmt_growth(v):
            if v >= 1000:
                return f"${v:,.0f}"
            if v >= 1:
                return f"${v:.0f}" if v == int(v) else f"${v:.1f}"
            return f"${v:.2f}"

        fig_eq.update_layout(
            title="Cumulative Equity Curve (Log Scale)",
            yaxis_title="Growth of $1 (Log Scale)",
            yaxis_type="log",
            xaxis_title="Date",
            hovermode="x unified",
            template="plotly_white",
            height=450,
            yaxis=dict(
                tickmode="array",
                tickvals=_eq_ticks,
                ticktext=[_fmt_growth(v) for v in _eq_ticks],
                range=[np.log10(_eq_ymin), np.log10(_eq_ymax)],
                gridcolor="rgba(128,128,128,0.15)",
            ),
        )
        st.plotly_chart(fig_eq, width="stretch")

        # Drawdown — multiselect (preserve selections on DD change)
        if _prev_dd is not None and _prev_dd != dd_constraint_pct and "dd_select" in st.session_state:
            old_suffix = f"{_prev_dd}%)"
            new_suffix = f"{dd_constraint_pct}%)"
            updated = []
            for p in st.session_state.dd_select:
                if old_suffix in p:
                    new_name = p.replace(old_suffix, new_suffix)
                    if new_name in portfolio_names:
                        updated.append(new_name)
                elif p in portfolio_names:
                    updated.append(p)
            st.session_state.dd_select = updated

        dd_selected = st.multiselect(
            "Select portfolios",
            portfolio_names,
            default=portfolio_names,
            key="dd_select",
        )

        fig_dd = go.Figure()
        for idx, pname in enumerate(dd_selected):
            if pname not in _eq_cache:
                pdata = st.session_state.portfolios[pname]
                pw = pdata["weights"]
                ws = pdata.get("weights_schedule")
                _eq_cache[pname] = compute_equity_curve(
                    returns, pw, start_ts, end_ts, rebalance=rebalance,
                    asset_starts=bt_asset_starts, weights_schedule=ws,
                )
            dd = compute_drawdown_series(_eq_cache[pname])
            color = _CHART_COLORS[idx % len(_CHART_COLORS)]
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name=pname, fill="tozeroy",
                                       line=dict(color=color), fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)"))
        fig_dd.update_layout(
            title="Drawdown",
            yaxis_title="Drawdown",
            yaxis_tickformat=".2%",
            xaxis_title="Date",
            hovermode="x unified",
            template="plotly_white",
            height=400,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            ),
            margin=dict(b=120),
        )
        st.plotly_chart(fig_dd, width="stretch")

        # Annual returns — multiselect (preserve selections on DD change)
        if _prev_dd is not None and _prev_dd != dd_constraint_pct and "ann_select" in st.session_state:
            old_suffix = f"{_prev_dd}%)"
            new_suffix = f"{dd_constraint_pct}%)"
            updated = []
            for p in st.session_state.ann_select:
                if old_suffix in p:
                    new_name = p.replace(old_suffix, new_suffix)
                    if new_name in portfolio_names:
                        updated.append(new_name)
                elif p in portfolio_names:
                    updated.append(p)
            st.session_state.ann_select = updated

        ann_selected = st.multiselect(
            "Select portfolios",
            portfolio_names,
            default=portfolio_names[:3],
            key="ann_select",
        )

        fig_ann = go.Figure()
        for idx, pname in enumerate(ann_selected):
            pdata = st.session_state.portfolios[pname]
            pw = pdata["weights"]
            ws = pdata.get("weights_schedule")
            ann = compute_annual_returns(returns, pw, start_ts, end_ts, rebalance=rebalance,
                                         asset_starts=bt_asset_starts, weights_schedule=ws)
            color = _CHART_COLORS[idx % len(_CHART_COLORS)]
            fig_ann.add_trace(go.Bar(x=ann.index.astype(str), y=ann.values, name=pname, marker_color=color))
        fig_ann.update_layout(
            title="Annual Returns",
            yaxis_title="Return",
            yaxis_tickformat=".2%",
            barmode="group",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig_ann, width="stretch")

        # Pie charts — side-by-side comparison
        pie_col1, pie_col2 = st.columns(2)
        with pie_col1:
            pie_left = st.selectbox(
                "Portfolio A",
                portfolio_names,
                index=0,
                key="pie_select_left",
            )
        with pie_col2:
            pie_right = st.selectbox(
                "Portfolio B",
                portfolio_names,
                index=min(1, len(portfolio_names) - 1),
                key="pie_select_right",
            )

        pie_chart_col1, pie_chart_col2 = st.columns(2)
        for col, pname, chart_key in [
            (pie_chart_col1, pie_left, "pie_chart_left"),
            (pie_chart_col2, pie_right, "pie_chart_right"),
        ]:
            with col:
                pie_w = st.session_state.portfolios[pname]["weights"]
                labels, vals = _make_pie_data(pie_w)
                if labels:
                    fig_pie = go.Figure()
                    fig_pie.add_trace(go.Pie(
                        labels=labels, values=vals,
                        textinfo="label+percent", textposition="auto",
                        hole=0.35, sort=False,
                    ))
                    fig_pie.update_layout(
                        title=pname, height=400, template="plotly_white",
                        showlegend=False, margin=dict(t=40, b=20, l=20, r=20),
                    )
                    st.plotly_chart(fig_pie, width="stretch", key=chart_key)

        # Risk Analytics — single select
        risk_selected = st.selectbox(
            "Select portfolio for risk analytics",
            portfolio_names,
            index=0,
            key="risk_select",
        )
        st.header(f"Risk Analytics \u2014 {risk_selected}")
        _risk_analytics(st.session_state.portfolios[risk_selected]["weights"], risk_selected)

# ======================================================================
# TAB 2: DYNAMIC STRATEGIES
# ======================================================================
with tab_dynamic:
    st.header("Dynamic Strategies")
    st.caption("Strategies that adjust allocations over time based on market conditions.")

    # ── Section A: Regime-Based Allocation ──
    st.subheader("Regime-Based Allocation")
    if not st.session_state.get("_macro_available", False):
        st.info("Regime data not available. Place 'Inflation and IR.xlsx' in the app directory with CPI and Fed Funds Rate data.")
    elif "_regime_series" in st.session_state and "_regime_weights" in st.session_state:
        _rs = st.session_state._regime_series
        _rw = st.session_state._regime_weights
        _analytics = regime_analytics(_rs, _rw, ASSETS)

        # Current regime
        st.metric("Current Regime", REGIME_LABELS.get(_analytics["current_regime"], "Unknown"))

        # Regime timeline chart — stacked bar with one bar per segment
        fig_regime = go.Figure()
        _regime_colors = {1: "#FF1744", 2: "#FF6D00", 3: "#2962FF", 4: "#00C853"}
        # Build a daily regime series for the timeline
        _regime_daily = _rs.reindex(pd.date_range(_rs.index[0], _rs.index[-1], freq="MS")).ffill()
        for label_id, label_name in REGIME_LABELS.items():
            mask = _regime_daily == label_id
            y_vals = mask.astype(int)
            fig_regime.add_trace(go.Bar(
                x=_regime_daily.index,
                y=y_vals,
                name=label_name,
                marker_color=_regime_colors.get(label_id, "#999"),
                width=30 * 86400000,  # ~30 days in ms
            ))
        fig_regime.update_layout(
            title="Regime Timeline",
            xaxis_title="Date",
            yaxis=dict(visible=False, range=[0, 1.1]),
            barmode="stack",
            height=220,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
            margin=dict(b=80, t=40),
            bargap=0,
        )
        st.plotly_chart(fig_regime, width="stretch")

        # Regime stats
        stat_cols = st.columns(4)
        for i, (label_id, label_name) in enumerate(REGIME_LABELS.items()):
            with stat_cols[i]:
                count = _analytics["regime_counts"].get(label_id, 0)
                avg_m = _analytics["regime_avg_months"].get(label_id, 0)
                st.metric(label_name, f"{count} periods", f"Avg {avg_m:.0f} months",
                          delta_color="off")

        # Weights per regime table (transposed: assets as rows, regimes as columns)
        st.subheader("Optimal Weights per Regime")
        regime_w_data = {"Asset": ASSETS}
        for label_id in sorted(_rw.keys()):
            col_name = REGIME_LABELS.get(label_id, f"Regime {label_id}")
            regime_w_data[col_name] = [_rw[label_id][i] * 100 for i in range(len(ASSETS))]
        regime_w_df = pd.DataFrame(regime_w_data)
        regime_w_config = {"Asset": st.column_config.TextColumn("Asset")}
        for label_name in REGIME_LABELS.values():
            regime_w_config[label_name] = st.column_config.NumberColumn(label_name, format="%.2f%%")
        st.dataframe(regime_w_df, column_config=regime_w_config, width="stretch", hide_index=True)

        # Equity curve comparison: Regime vs Max Sharpe
        if "Regime-Based (time-varying)" in st.session_state.portfolios and "Max Sharpe Ratio" in st.session_state.portfolios:
            st.subheader("Regime Strategy vs Max Sharpe (Static)")
            _r_sched = st.session_state.get("_regime_schedule")
            _r_repr = st.session_state.portfolios["Regime-Based (time-varying)"]["weights"]
            _ms_w = st.session_state.portfolios["Max Sharpe Ratio"]["weights"]
            _rebal_dyn = rebalance if rebalance != "daily" else "annual"

            eq_regime = compute_equity_curve(
                returns, _r_repr, start_ts, end_ts, rebalance=_rebal_dyn,
                asset_starts=bt_asset_starts, weights_schedule=_r_sched,
            )
            eq_sharpe = compute_equity_curve(
                returns, _ms_w, start_ts, end_ts, rebalance=rebalance,
                asset_starts=bt_asset_starts,
            )

            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(x=eq_regime.index, y=eq_regime.values, name="Regime-Based (time-varying)", line=dict(color="#FF6D00")))
            fig_rv.add_trace(go.Scatter(x=eq_sharpe.index, y=eq_sharpe.values, name="Max Sharpe (Static)", line=dict(color="#2962FF")))
            fig_rv.update_layout(
                yaxis_type="log", yaxis_title="Growth of $1 (Log)",
                xaxis_title="Date", hovermode="x unified",
                template="plotly_white", height=400,
            )
            st.plotly_chart(fig_rv, width="stretch")

    st.markdown("---")

    # ── Section B: DD P-Value Momentum ──
    st.subheader("Drawdown P-Value Momentum")
    st.caption("Boosts allocation to assets with historically rare drawdowns (buying opportunities) "
               "and reduces allocation to assets near all-time highs.")

    if "_dd_checkpoints" in st.session_state and "base_results" in st.session_state:
        _dd_an = dd_analytics(returns, end_ts, bump_max=dd_bump_max)

        # Current p-values and bump factors table
        _dd_rows = []
        sorted_idx = np.argsort(_dd_an["pvalues"])
        for i in sorted_idx:
            _dd_rows.append({
                "Asset": _dd_an["assets"][i],
                "Current DD": _dd_an["current_dd"][i] * 100,
                "P-Value": _dd_an["pvalues"][i],
                "Episodes": int(_dd_an["episode_counts"][i]),
                "Bump Factor": _dd_an["bump_factors"][i] * 100,
            })
        _dd_df = pd.DataFrame(_dd_rows)
        st.dataframe(
            _dd_df,
            column_config={
                "Asset": st.column_config.TextColumn("Asset"),
                "Current DD": st.column_config.NumberColumn("Current DD", format="%.2f%%"),
                "P-Value": st.column_config.NumberColumn("P-Value", format="%.3f"),
                "Episodes": st.column_config.NumberColumn("Episodes", format="%d"),
                "Bump Factor": st.column_config.NumberColumn("Bump Factor", format="%+.0f%%"),
            },
            width="stretch",
            hide_index=True,
        )

        # Allocation at a given date
        if "_dd_schedule" in st.session_state and st.session_state._dd_schedule:
            st.subheader("Allocation at Date")
            _dd_sched_dates = sorted(st.session_state._dd_schedule.keys())
            _dd_date_select = st.select_slider(
                "Select rebalance date",
                options=_dd_sched_dates,
                value=_dd_sched_dates[-1],
                format_func=lambda d: d.strftime("%d/%m/%Y"),
                key="dd_alloc_date",
            )
            _dd_w_at_date = st.session_state._dd_schedule[_dd_date_select]
            _ms_base = st.session_state.base_results["Max Sharpe Ratio"]["weights"]
            _alloc_rows = []
            for i, asset in enumerate(ASSETS):
                _alloc_rows.append({
                    "Asset": asset,
                    "Base (Max Sharpe)": _ms_base[i] * 100,
                    "DD Momentum": _dd_w_at_date[i] * 100,
                    "Change": (_dd_w_at_date[i] - _ms_base[i]) * 100,
                })
            _alloc_df = pd.DataFrame(_alloc_rows)
            st.dataframe(
                _alloc_df,
                column_config={
                    "Asset": st.column_config.TextColumn("Asset"),
                    "Base (Max Sharpe)": st.column_config.NumberColumn("Base (Max Sharpe)", format="%.2f%%"),
                    "DD Momentum": st.column_config.NumberColumn("DD Momentum", format="%.2f%%"),
                    "Change": st.column_config.NumberColumn("Change", format="%+.2f%%"),
                },
                width="stretch",
                hide_index=True,
            )

        # Equity curve comparison: DD Momentum vs Max Sharpe
        if "DD P-Value Momentum (time-varying)" in st.session_state.portfolios and "Max Sharpe Ratio" in st.session_state.portfolios:
            st.subheader("DD Momentum vs Max Sharpe (Static)")
            _dd_sched = st.session_state.get("_dd_schedule")
            _dd_repr = st.session_state.portfolios["DD P-Value Momentum (time-varying)"]["weights"]
            _ms_w2 = st.session_state.portfolios["Max Sharpe Ratio"]["weights"]
            _rebal_dyn2 = rebalance if rebalance != "daily" else "annual"

            eq_ddm = compute_equity_curve(
                returns, _dd_repr, start_ts, end_ts, rebalance=_rebal_dyn2,
                asset_starts=bt_asset_starts, weights_schedule=_dd_sched,
            )
            eq_sharpe2 = compute_equity_curve(
                returns, _ms_w2, start_ts, end_ts, rebalance=rebalance,
                asset_starts=bt_asset_starts,
            )

            fig_ddv = go.Figure()
            fig_ddv.add_trace(go.Scatter(x=eq_ddm.index, y=eq_ddm.values, name="DD P-Value Momentum (time-varying)", line=dict(color="#00C853")))
            fig_ddv.add_trace(go.Scatter(x=eq_sharpe2.index, y=eq_sharpe2.values, name="Max Sharpe (Static)", line=dict(color="#2962FF")))
            fig_ddv.update_layout(
                yaxis_type="log", yaxis_title="Growth of $1 (Log)",
                xaxis_title="Date", hovermode="x unified",
                template="plotly_white", height=400,
            )
            st.plotly_chart(fig_ddv, width="stretch")
    else:
        st.info("DD Momentum data not available — ensure base portfolios are computed.")

# ======================================================================
# TAB 3: CFD ANALYSIS
# ======================================================================
with tab_cfd:
    if not portfolio_names:
        st.info("Computing default portfolios...")
    else:
        cfd_pick_cols = st.columns([2, 1, 1, 1, 1])
        with cfd_pick_cols[0]:
            cfd_port_name = st.selectbox("Analyse Portfolio", portfolio_names, key="cfd_port_pick")
        with cfd_pick_cols[1]:
            cfd_capital = st.number_input("Total Capital ($)", value=10000, min_value=1000, step=1000, key="cfd_capital")
        with cfd_pick_cols[2]:
            cfd_leverage = st.number_input("Leverage", value=5.0, min_value=1.0, max_value=20.0, step=0.5, key="cfd_leverage")
        with cfd_pick_cols[3]:
            cfd_fin_pct = st.number_input("Financing Rate (%)", value=6.5, min_value=0.0, max_value=20.0, step=0.5, format="%.1f", key="cfd_financing")
            cfd_financing = cfd_fin_pct / 100.0
        with cfd_pick_cols[4]:
            cfd_margin_pct_input = st.number_input("Margin Req. (%)", value=20.0, min_value=1.0, max_value=100.0, step=1.0, format="%.0f", key="cfd_margin_pct")
            cfd_margin_pct = cfd_margin_pct_input / 100.0

        cfd_port = st.session_state.portfolios[cfd_port_name]
        cfd_weights = cfd_port["weights"]
        cfd_stats = cfd_port["stats"]

        cfd_result = analyze_cfd(
            cfd_weights, cfd_stats, cfd_capital, cfd_leverage,
            cfd_financing, cfd_margin_pct, risk_free_rate,
        )

        # Capital allocation banner
        st.markdown("---")
        alloc_cols = st.columns(4)
        with alloc_cols[0]:
            st.metric("Total Capital", f"${cfd_result.total_capital:,.0f}")
        with alloc_cols[1]:
            st.metric("Deploy to Assets", f"${cfd_result.deployed_capital:,.0f}")
        with alloc_cols[2]:
            st.metric("Cash Reserve (Liquidity Buffer)", f"${cfd_result.cash_reserve:,.0f}")
        with alloc_cols[3]:
            reserve_pct = cfd_result.cash_reserve / cfd_result.total_capital * 100 if cfd_result.total_capital > 0 else 0
            st.metric("Reserve %", f"{reserve_pct:.1f}%")
        st.caption("Cash reserve is sized to survive the worst historical drawdown at this leverage without a margin call.")

        st.markdown("---")

        # CFD metrics as a clean table
        cfd_metrics_data = [
            ("Notional Exposure", f"${cfd_result.notional_exposure:,.0f}",
             "Total value of leveraged positions"),
            ("Margin Required", f"${cfd_result.margin_required:,.0f}",
             "Minimum collateral required by broker"),
            ("Free Margin", f"${cfd_result.free_margin:,.0f}",
             "Deployed capital beyond margin requirement"),
            ("Margin Utilisation", f"{cfd_result.margin_utilisation:.1%}",
             "Margin as % of deployed capital"),
            ("Max Drawdown ($)", f"${cfd_result.max_drawdown_dollars:,.0f}",
             "Worst-case loss in dollar terms at this leverage"),
            ("Leveraged CAGR (Gross)", f"{cfd_result.gross_cagr:.2%}",
             "Leveraged CAGR accounting for vol drag, before financing"),
            ("Financing Drag", f"-{cfd_result.financing_drag:.2%}",
             "Annual cost of borrowed notional"),
            ("Net CAGR (on Deployed)", f"{cfd_result.net_cagr:.2%}",
             "CAGR after financing, on deployed capital"),
            ("Effective CAGR (on Total Capital)", f"{cfd_result.effective_cagr:.2%}",
             "Net return as % of total capital incl. reserve"),
            ("Leveraged Volatility", f"{cfd_result.leveraged_volatility:.2%}",
             "Portfolio volatility scaled by leverage"),
            ("Net Sharpe Ratio", f"{cfd_result.net_sharpe:.2f}",
             "Risk-adjusted return after financing costs"),
        ]

        cfd_metrics_df = pd.DataFrame(cfd_metrics_data, columns=["Metric", "Value", "Description"])
        st.dataframe(
            cfd_metrics_df,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="small"),
                "Description": st.column_config.TextColumn("Description", width="large"),
            },
            width="stretch",
            hide_index=True,
        )

        # Per-asset allocation table
        st.subheader("Per-Asset CFD Allocation")
        cfd_alloc_rows = []
        for asset in ASSETS:
            notional_asset = cfd_result.capital_per_asset.get(asset, 0.0)
            w_pct = cfd_weights[ASSETS.index(asset)]
            if notional_asset > 0.5:
                cfd_alloc_rows.append({
                    "Asset": asset,
                    "Weight": w_pct * 100,
                    "Notional": round(notional_asset),
                    "Margin": round(notional_asset * cfd_margin_pct),
                })
        cfd_alloc_df = pd.DataFrame(cfd_alloc_rows)
        st.dataframe(
            cfd_alloc_df,
            column_config={
                "Asset": st.column_config.TextColumn("Asset"),
                "Weight": st.column_config.NumberColumn("Weight", format="%.2f%%"),
                "Notional": st.column_config.NumberColumn("Notional ($)", format="$%.0f"),
                "Margin": st.column_config.NumberColumn("Margin ($)", format="$%.0f"),
            },
            width="stretch",
            hide_index=True,
        )

        # ── Monte Carlo: Leveraged vs Unleveraged forward projection ──
        st.markdown("---")
        st.header("10-Year Monte Carlo Projection")
        st.caption("Forward simulation using historical CAGR and volatility with geometric Brownian motion (500 paths).")

        n_paths = 2000
        n_years = 10
        n_days_mc = 252 * n_years
        years_axis = np.linspace(0, n_years, n_days_mc + 1)

        # Allow user to resample MC paths
        mc_seed = st.session_state.get("mc_seed", 42)
        if st.button("Resample Monte Carlo", key="mc_resample"):
            mc_seed = int(np.random.default_rng().integers(0, 2**31))
            st.session_state.mc_seed = mc_seed
            st.rerun()

        rng = np.random.RandomState(mc_seed)
        z = rng.randn(n_paths, n_days_mc)

        def _mc_fan(start_val, cagr, vol):
            """Run GBM Monte Carlo and return percentile bands."""
            # CAGR is the geometric mean = mean of log returns (annualised).
            # So daily_mu IS the correct mean for daily log returns — no
            # additional -σ²/2 adjustment (that would double-count vol drag).
            daily_mu = np.log(1.0 + cagr) / 252.0
            daily_sigma = vol / np.sqrt(252.0)
            daily_log_ret = daily_mu + daily_sigma * z
            cum_log = np.cumsum(daily_log_ret, axis=1)
            # Prepend 0 for day 0
            cum_log = np.hstack([np.zeros((n_paths, 1)), cum_log])
            paths = start_val * np.exp(cum_log)
            p5 = np.percentile(paths, 5, axis=0)
            p25 = np.percentile(paths, 25, axis=0)
            p50 = np.percentile(paths, 50, axis=0)
            p75 = np.percentile(paths, 75, axis=0)
            p95 = np.percentile(paths, 95, axis=0)
            return p5, p25, p50, p75, p95

        # Unleveraged: full capital, base stats
        ul_p5, ul_p25, ul_p50, ul_p75, ul_p95 = _mc_fan(
            cfd_capital, cfd_stats.cagr, cfd_stats.volatility,
        )

        # Leveraged: deployed capital, leveraged stats net of financing
        lev_cagr = cfd_result.net_cagr
        lev_vol = cfd_result.leveraged_volatility
        lv_p5, lv_p25, lv_p50, lv_p75, lv_p95 = _mc_fan(
            cfd_result.deployed_capital, lev_cagr, lev_vol,
        )

        fig_mc = go.Figure()

        # Unleveraged bands
        fig_mc.add_trace(go.Scatter(
            x=years_axis, y=ul_p95, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig_mc.add_trace(go.Scatter(
            x=years_axis, y=ul_p5, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(41,98,255,0.10)",
            showlegend=False, hoverinfo="skip",
        ))
        fig_mc.add_trace(go.Scatter(
            x=years_axis, y=ul_p75, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig_mc.add_trace(go.Scatter(
            x=years_axis, y=ul_p25, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(41,98,255,0.20)",
            showlegend=False, hoverinfo="skip",
        ))
        fig_mc.add_trace(go.Scatter(
            x=years_axis, y=ul_p50, mode="lines",
            line=dict(color="#2962FF", width=2),
            name=f"Unleveraged Median (${cfd_capital:,.0f})",
            hovertemplate="Year %{x:.2f}: $%{y:,.0f}<extra></extra>",
        ))

        # Leveraged bands
        fig_mc.add_trace(go.Scatter(
            x=years_axis, y=lv_p95, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig_mc.add_trace(go.Scatter(
            x=years_axis, y=lv_p5, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(255,109,0,0.10)",
            showlegend=False, hoverinfo="skip",
        ))
        fig_mc.add_trace(go.Scatter(
            x=years_axis, y=lv_p75, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig_mc.add_trace(go.Scatter(
            x=years_axis, y=lv_p25, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(255,109,0,0.20)",
            showlegend=False, hoverinfo="skip",
        ))
        fig_mc.add_trace(go.Scatter(
            x=years_axis, y=lv_p50, mode="lines",
            line=dict(color="#FF6D00", width=2),
            name=f"Leveraged {cfd_leverage:.0f}x Median (${cfd_result.deployed_capital:,.0f})",
            hovertemplate="Year %{x:.2f}: $%{y:,.0f}<extra></extra>",
        ))

        # Compute clean tick values for log scale based on data range
        _mc_all_vals = np.concatenate([ul_p5, ul_p95, lv_p5, lv_p95])
        _mc_ymin = max(_mc_all_vals.min() * 0.8, 1)
        _mc_ymax = _mc_all_vals.max() * 1.2

        # Generate log-spaced ticks: $1, $2, $5, $10, ... up to $100M
        _nice_ticks = []
        for exp in range(0, 9):  # 1 to 100M
            base = 10 ** exp
            for mult in [1, 2, 5]:
                val = base * mult
                if _mc_ymin <= val <= _mc_ymax:
                    _nice_ticks.append(val)

        def _fmt_dollar(v):
            if v >= 1_000_000:
                return f"${v / 1_000_000:.0f}M"
            elif v >= 1_000:
                return f"${v / 1_000:.0f}k"
            return f"${v:.0f}"

        fig_mc.update_layout(
            yaxis_title="Portfolio Value (Log Scale)",
            yaxis_type="log",
            xaxis_title="Years",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            yaxis=dict(
                tickmode="array",
                tickvals=_nice_ticks,
                ticktext=[_fmt_dollar(v) for v in _nice_ticks],
                range=[np.log10(_mc_ymin), np.log10(_mc_ymax)],
                gridcolor="rgba(128,128,128,0.15)",
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_mc, width="stretch")
        st.caption("Shaded bands: P5\u2013P95 (light) and P25\u2013P75 (dark). Solid line: median (P50).")

        # ── 10-Year Median Projection Table (all portfolios) ──
        st.markdown("---")
        st.header("10-Year Median Projection")
        st.caption(f"Median portfolio value after 10 years per ${cfd_capital:,.0f} capital, computed from Monte Carlo simulation ({n_paths} paths). "
                   f"Leveraged column uses {cfd_leverage:.0f}x leverage with {cfd_fin_pct:.1f}% financing.")

        def _mc_median_10y(start_val, cagr, vol):
            """Compute the actual MC median at year 10 (not the deterministic formula)."""
            daily_mu = np.log(1.0 + cagr) / 252.0
            daily_sigma = vol / np.sqrt(252.0)
            daily_log_ret = daily_mu + daily_sigma * z  # reuse the shared z matrix
            cum_log = np.sum(daily_log_ret, axis=1)  # sum over all days for final value
            final_vals = start_val * np.exp(cum_log)
            return round(np.median(final_vals))

        proj_rows = []
        for pname, pdata in st.session_state.portfolios.items():
            p_stats = pdata["stats"]
            p_weights = pdata["weights"]

            # Unleveraged median from MC simulation
            ul_median_10y = _mc_median_10y(cfd_capital, p_stats.cagr, p_stats.volatility)

            # Leveraged: compute CFD metrics for this portfolio
            p_cfd = analyze_cfd(
                p_weights, p_stats, cfd_capital, cfd_leverage,
                cfd_financing, cfd_margin_pct, risk_free_rate,
            )
            lev_median_10y = _mc_median_10y(p_cfd.deployed_capital, p_cfd.net_cagr, p_cfd.leveraged_volatility)

            proj_rows.append({
                "Strategy": pname,
                "Unleveraged Median": ul_median_10y,
                f"Leveraged {cfd_leverage:.0f}x Median": lev_median_10y,
            })

        lev_col_name = f"Leveraged {cfd_leverage:.0f}x Median"
        proj_df = pd.DataFrame(proj_rows)
        # Ensure native Python int for Streamlit formatting
        proj_df["Unleveraged Median"] = proj_df["Unleveraged Median"].astype(int)
        proj_df[lev_col_name] = proj_df[lev_col_name].astype(int)
        st.dataframe(
            proj_df,
            column_config={
                "Strategy": st.column_config.TextColumn("Strategy"),
                "Unleveraged Median": st.column_config.NumberColumn("Unleveraged Median", format="$%.0f"),
                lev_col_name: st.column_config.NumberColumn(lev_col_name, format="$%.0f"),
            },
            width="stretch",
            hide_index=True,
        )

# ──────────────────────────────────────────────
# SETTINGS TAB
# ──────────────────────────────────────────────
with tab_settings:
    st.header("Allocation Constraints")
    st.caption("Adjust minimum and maximum weights for individual assets and asset classes. "
               "Changes will trigger a full recomputation of all optimised portfolios.")

    # ── Per-asset constraints ──
    st.subheader("Per-Asset Constraints (%)")

    asset_rows = []
    for a in ASSETS:
        asset_rows.append({
            "Asset": a,
            "Group": GROUP_MAP[a],
            "Min (%)": st.session_state.asset_min.get(a, 1.0),
            "Max (%)": st.session_state.asset_max.get(a, 30.0),
        })
    asset_df = pd.DataFrame(asset_rows)

    edited_assets = st.data_editor(
        asset_df,
        column_config={
            "Asset": st.column_config.TextColumn("Asset", disabled=True),
            "Group": st.column_config.TextColumn("Group", disabled=True),
            "Min (%)": st.column_config.NumberColumn("Min (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
            "Max (%)": st.column_config.NumberColumn("Max (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"),
        },
        hide_index=True,
        width="stretch",
        key="asset_constraints_editor",
    )

    # Sync edits back to session state
    _asset_min_new = {row["Asset"]: row["Min (%)"] for _, row in edited_assets.iterrows()}
    _asset_max_new = {row["Asset"]: row["Max (%)"] for _, row in edited_assets.iterrows()}
    if _asset_min_new != st.session_state.asset_min or _asset_max_new != st.session_state.asset_max:
        st.session_state.asset_min = _asset_min_new
        st.session_state.asset_max = _asset_max_new
        st.rerun()

    # ── Asset class constraints ──
    st.subheader("Asset Class Constraints (%)")

    group_rows = []
    for g in GROUP_NAMES:
        group_rows.append({
            "Asset Class": g,
            "Max (%)": st.session_state.group_max.get(g, 30.0),
        })
    group_df = pd.DataFrame(group_rows)

    edited_groups = st.data_editor(
        group_df,
        column_config={
            "Asset Class": st.column_config.TextColumn("Asset Class", disabled=True),
            "Max (%)": st.column_config.NumberColumn("Max (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.0f"),
        },
        hide_index=True,
        width="stretch",
        key="group_constraints_editor",
    )

    _group_max_new = {row["Asset Class"]: row["Max (%)"] for _, row in edited_groups.iterrows()}
    if _group_max_new != st.session_state.group_max:
        st.session_state.group_max = _group_max_new
        st.rerun()

    # Reset button
    if st.button("Reset to Defaults"):
        st.session_state.asset_min = dict(_DEFAULT_MIN)
        st.session_state.asset_max = dict(_DEFAULT_MAX)
        st.session_state.group_max = dict(_DEFAULT_GROUP_MAX)
        st.rerun()
