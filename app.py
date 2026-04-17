"""Streamlit dashboard for All Weather Portfolio Analyser."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import hashlib
import subprocess
from pathlib import Path

_APP_MAJOR_MINOR = "1.4"

def _get_app_version():
    try:
        count = subprocess.check_output(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return f"{_APP_MAJOR_MINOR}.{count}"
    except Exception:
        return f"{_APP_MAJOR_MINOR}.0"

APP_VERSION = _get_app_version()

from data import ASSETS, GROUP_MAP, GROUP_NAMES, load_data
from stats import (
    calc_stats, compute_equity_curve, compute_drawdown_series,
    compute_annual_returns, compute_asset_starts,
)
from optimizer import run_optimization
from cfd import analyze_cfd, ASSET_MARGIN_RATES
from regime import load_regime_data, classify_regimes, optimize_per_regime, build_regime_schedule, regime_analytics, REGIME_LABELS
from dd_momentum import (
    compute_dd_adjustments, compute_dd_adjustments_scheduled,
    build_dd_momentum_schedule, dd_analytics, load_optimal_bump_schedule,
)

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
PRECOMPUTED_FILE = Path(__file__).resolve().parent / "precomputed_weights.pkl"

st.set_page_config(page_title="All Weather Portfolio", layout="wide")
st.title("All Weather Portfolio Analyser")

# ──────────────────────────────────────────────
# SIDEBAR: Global settings
# ──────────────────────────────────────────────
# Excel file path lives in Settings tab; initialised here from session state
if "data_path" not in st.session_state:
    st.session_state.data_path = str(Path(__file__).resolve().parent / "data_template.xlsx")
data_path = st.session_state.data_path
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

# Default start: date when 50% of assets have data (1983-08-31)
_start_default_50pct = pd.Timestamp("1983-08-31").date()
_start_default = _start_default_50pct
if st.session_state.get("_use_inception", False):
    _start_default = data_start_earliest

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Backtest Start", value=_start_default,
                             min_value=data_start_earliest, max_value=data_end,
                             format="DD/MM/YYYY")
end_date = col2.date_input("End Date", value=data_end,
                           min_value=data_start_earliest, max_value=data_end,
                           format="DD/MM/YYYY")
# Inception — small text-link style right under backtest dates
if st.sidebar.button("Inception", type="tertiary", use_container_width=False):
    st.session_state["_use_inception"] = True
    st.rerun()

# Determine if we're in extended backtest mode (start before all assets exist)
extended_backtest = start_date < overlap_start_date
if extended_backtest:
    st.sidebar.caption(
        f"Extended backtest: assets without data before their start date "
        f"have weights redistributed pro-rata. "
        f"Optimiser uses overlap period from {overlap_start_date}."
    )

risk_free_pct = st.sidebar.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=20.0, step=0.5, format="%.1f",
                                        help="Used for Sharpe ratio calculation and CFD cash reserve growth rate.")
risk_free_rate = risk_free_pct / 100.0

_REBAL_OPTIONS = {
    "Daily (Continuous)": "daily",
    "Monthly": "monthly",
    "Quarterly": "quarterly",
    "Semi-Annual": "semi-annual",
    "Annual": "annual",
}
rebalance_freq = st.sidebar.selectbox("Rebalancing Frequency", list(_REBAL_OPTIONS.keys()),
                                      index=list(_REBAL_OPTIONS.keys()).index("Monthly"))
rebalance = _REBAL_OPTIONS[rebalance_freq]

_DD_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
dd_constraint_pct = st.sidebar.selectbox(
    "Max Drawdown Constraint (%)",
    _DD_LEVELS,
    index=_DD_LEVELS.index(50),
    key="dd_constraint_select",
)
dd_constraint_val = dd_constraint_pct / 100.0
if rebalance != "monthly":
    st.sidebar.caption(
        f"⚠️ DD constraints optimised for monthly rebalancing. "
        f"At {rebalance_freq} rebalancing, actual drawdowns may differ slightly."
    )
_TV_DD_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
tv_dd_constraint_pct = st.sidebar.selectbox(
    "Max DD \u2013 Time-Varying (%)",
    _TV_DD_LEVELS,
    index=_TV_DD_LEVELS.index(50),
    key="tv_dd_constraint_select",
    help="Separate drawdown cap for Regime-Based and DD P-Value Momentum strategies. "
         "Enforced as a hard constraint on the full backtest equity curve.",
)
tv_dd_constraint_val = tv_dd_constraint_pct / 100.0
exclude_bitcoin = st.sidebar.checkbox("Exclude Bitcoin", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Leverage Settings**")
cfd_leverage_opt = st.sidebar.number_input(
    "Target Leverage (x)", min_value=1.0, max_value=10.0, value=5.0, step=0.5,
    key="cfd_leverage_opt",
    help="Target leverage for the Leverage-Optimal strategy. "
         "Higher leverage penalises volatility quadratically via vol drag.",
)
cfd_financing_opt = st.sidebar.number_input(
    "Financing Rate (% p.a.)", min_value=0.0, max_value=15.0, value=6.5, step=0.5,
    key="cfd_financing_opt",
    help="Annual CFD financing rate for leverage-aware strategies.",
) / 100.0

# Base strategies (always computed once)
_BASE_TARGETS = [
    "Max Sharpe Ratio",
    "Leverage-Optimal",
    "Max Sharpe (Unconstrained)",
    "Leverage-Optimal (Unconstrained)",
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
    "US REITs": 20.0, "Industrial Metals": 15.0, "Gold": 20.0,
    "Bitcoin": 15.0, "Infrastructure": 20.0,
    "Japan Equities": 30.0, "UK Equities": 30.0, "EU Equities": 30.0,
    "US TIPS": 30.0, "High Yield": 20.0, "EM Debt": 20.0,
    "JPY": 20.0, "CHF": 20.0, "CNY": 10.0,
    "China Equities": 20.0, "Copper": 15.0, "Soft Commodities": 15.0,
}
_DEFAULT_GROUP_MAX = {
    "US Equities": 35.0, "Intl Equities": 30.0, "Bonds": 50.0,
    "Real Assets": 30.0, "Alternatives": 20.0, "Currencies": 25.0,
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
# Ray Dalio All Weather: 30% Stocks, 40% LT Bonds, 15% IT Bonds, 7.5% Gold, 7.5% Industrial Metals
_DALIO_WEIGHTS = np.zeros(n)
_DALIO_MAP = {
    "S&P 500": 0.30,
    "Long-Term Treasuries": 0.40,
    "Short-Term Treasuries": 0.15,
    "Gold": 0.075,
    "Industrial Metals": 0.075,
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


def _compute_all(opt_data, bt_data, rf, rb, a_starts,
                  leverage=1.0, financing_rate=0.065):
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
                leverage=leverage, financing_rate=financing_rate,
            )
        else:
            w = run_optimization(
                opt_data, tgt, min_w_default, max_w_default, default_group_max,
                rf, rebalance="daily",
                leverage=leverage, financing_rate=financing_rate,
            )
        s = calc_stats(bt_data, w, rf, rebalance=rb, asset_starts=a_starts)
        base_results[tgt] = {"weights": w, "stats": s}

    dd_results = {}
    # Process DD levels from tightest to loosest, warm-starting from the
    # previous level's result so the optimizer converges much faster.
    # Use monthly rebalancing for DD evaluation so the constraint is tuned
    # to the same rebalancing regime as the default app setting.
    prev_w = {}  # {target: weights from previous level}
    for dd_pct in sorted(_DD_LEVELS):
        dd_val = dd_pct / 100.0
        dd_results[dd_pct] = {}
        for tgt in _DD_TARGETS:
            w = run_optimization(
                opt_data, tgt, min_w_default, max_w_default, default_group_max,
                rf, rebalance="monthly", dd_constraint=dd_val,
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
            abs(risk_free_rate - 0.05) < 1e-9)


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


def _enforce_schedule_dd(
    schedule: dict,
    returns: pd.DataFrame,
    base_weights,
    dd_limit: float,
    risk_free_rate: float,
    rebalance: str,
    asset_starts: dict | None = None,
) -> dict:
    """Hard-enforce a max-drawdown cap on a time-varying weight schedule.

    If the full-backtest max drawdown exceeds *dd_limit*, scale down every
    checkpoint's weights toward cash (zeros) via binary search until the
    constraint is satisfied.  The unallocated portion (1 – sum(weights))
    implicitly earns 0 % – i.e. cash.
    """
    if not schedule:
        return schedule

    s = calc_stats(
        returns, base_weights, risk_free_rate,
        rebalance=rebalance, asset_starts=asset_starts,
        weights_schedule=schedule,
    )
    if abs(s.max_drawdown) <= dd_limit:
        return schedule  # already within limit

    # Binary-search a scalar scale factor: 0 = 100 % cash, 1 = original weights
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        blended = {dt: w * mid for dt, w in schedule.items()}
        s2 = calc_stats(
            returns, base_weights * mid, risk_free_rate,
            rebalance=rebalance, asset_starts=asset_starts,
            weights_schedule=blended,
        )
        if abs(s2.max_drawdown) > dd_limit:
            hi = mid
        else:
            lo = mid

    # Use conservative (lower) factor
    final = {dt: w * lo for dt, w in schedule.items()}
    return final


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
            # Use BTC-specific variant if available
            btc_key = "excl_btc" if exclude_bitcoin else "incl_btc"
            if "btc_variants" in pre and btc_key in pre["btc_variants"]:
                variant = pre["btc_variants"][btc_key]
            else:
                variant = pre  # backward compat: old format has no btc_variants
            base_results, dd_results = _stats_from_weights(
                variant["base_weights"], variant["dd_weights"], bt_data, rf, rb, a_starts)
            # Compute any targets not in the precomputed file
            missing = [t for t in _BASE_TARGETS if t not in base_results]
            if missing:
                for tgt in missing:
                    w = run_optimization(
                        opt_data, tgt, min_w_default, max_w_default, default_group_max,
                        rf, rebalance="daily",
                        leverage=cfd_leverage_opt, financing_rate=cfd_financing_opt,
                    )
                    s = calc_stats(bt_data, w, rf, rebalance=rb, asset_starts=a_starts)
                    base_results[tgt] = {"weights": w, "stats": s}
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
        # Compute any targets not in the cached weights
        missing = [t for t in _BASE_TARGETS if t not in base_results]
        if missing:
            for tgt in missing:
                w = run_optimization(
                    opt_data, tgt, min_w_default, max_w_default, default_group_max,
                    rf, rebalance="daily",
                    leverage=cfd_leverage_opt, financing_rate=cfd_financing_opt,
                )
                s = calc_stats(bt_data, w, rf, rebalance=rb, asset_starts=a_starts)
                base_results[tgt] = {"weights": w, "stats": s}
    else:
        base_results, dd_results = _compute_all(
            opt_data, bt_data, rf, rb, a_starts,
            leverage=cfd_leverage_opt, financing_rate=cfd_financing_opt,
        )
        # Cache weights separately
        base_w = {t: d["weights"] for t, d in base_results.items()}
        dd_w = {
            dd_pct: {t: d["weights"] for t, d in targets.items()}
            for dd_pct, targets in dd_results.items()
        }
        with open(_opt_cache_file, "wb") as f:
            pickle.dump((base_w, dd_w), f)

    # Cache full results — convert stats to dicts to avoid pickle issues
    # with Streamlit's module reloading (PortfolioStats class identity changes)
    try:
        _cacheable_base = {
            t: {"weights": d["weights"], "stats": vars(d["stats"]) if hasattr(d["stats"], '__dict__') else d["stats"]}
            for t, d in base_results.items()
        }
        _cacheable_dd = {
            dd_pct: {
                t: {"weights": d["weights"], "stats": vars(d["stats"]) if hasattr(d["stats"], '__dict__') else d["stats"]}
                for t, d in targets.items()
            }
            for dd_pct, targets in dd_results.items()
        }
        with open(_cache_file, "wb") as f:
            pickle.dump((_cacheable_base, _cacheable_dd), f)
    except Exception:
        pass  # Non-fatal: cache miss next time, but don't block the app
    return base_results, dd_results


def _exclude_btc_fast(base_results, dd_results, bt_data, rf, rb, a_starts):
    """Fast path: zero out Bitcoin and redistribute weights pro-rata, then recompute stats.

    This avoids a full reoptimization when toggling Exclude Bitcoin.
    """
    btc_idx = ASSETS.index("Bitcoin")

    def _zero_btc(w):
        w_new = w.copy()
        btc_w = w_new[btc_idx]
        w_new[btc_idx] = 0.0
        remaining = w_new.sum()
        if remaining > 1e-12:
            w_new *= (1.0 / remaining)  # redistribute pro-rata
        return w_new

    new_base = {}
    for tgt, data in base_results.items():
        w = _zero_btc(data["weights"])
        s = calc_stats(bt_data, w, rf, rebalance=rb, asset_starts=a_starts)
        new_base[tgt] = {"weights": w, "stats": s}

    new_dd = {}
    for dd_pct, targets in dd_results.items():
        new_dd[dd_pct] = {}
        for tgt, data in targets.items():
            w = _zero_btc(data["weights"])
            s = calc_stats(bt_data, w, rf, rebalance=rb, asset_starts=a_starts)
            new_dd[dd_pct][tgt] = {"weights": w, "stats": s}

    return new_base, new_dd


if st.session_state.get("defaults_cache_key") != cache_key:
    # Fast path: if only exclude_bitcoin changed and we have cached results,
    # zero out BTC and redistribute rather than full reoptimization.
    # Works both directions: toggling BTC on OR off.
    _prev_key = st.session_state.get("defaults_cache_key", "")
    _have_cached = "base_results" in st.session_state and "dd_results" in st.session_state
    _prev_btc = st.session_state.get("_prev_exclude_btc", False)
    _btc_toggled = _have_cached and (exclude_bitcoin != _prev_btc) and _prev_key

    if _btc_toggled and not cache_file.exists():
        if exclude_bitcoin:
            # Toggling ON: zero out BTC from current cached weights
            base_results, dd_results = _exclude_btc_fast(
                st.session_state.base_results, st.session_state.dd_results,
                returns, risk_free_rate, rebalance, bt_asset_starts,
            )
        else:
            # Toggling OFF: reload from the non-BTC-excluded cache/precomputed weights
            # Build the non-BTC cache key to find pre-existing results
            # Try loading from disk cache or precomputed weights (fast path)
            base_results, dd_results = _load_or_compute(
                opt_returns, returns, risk_free_rate, rebalance,
                cache_file, opt_cache_file, bt_asset_starts,
            )
        st.session_state.base_results = base_results
        st.session_state.dd_results = dd_results
        st.session_state.defaults_cache_key = cache_key
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((base_results, dd_results), f)
        except Exception:
            pass  # PortfolioStats can't be pickled; non-fatal
    else:
        with st.spinner("⏳ Computing portfolios — this may take a couple of minutes on first run..."):
            base_results, dd_results = _load_or_compute(
                opt_returns, returns, risk_free_rate, rebalance,
                cache_file, opt_cache_file, bt_asset_starts,
            )
            st.session_state.base_results = base_results
            st.session_state.dd_results = dd_results
            st.session_state.defaults_cache_key = cache_key

    st.session_state._prev_exclude_btc = exclude_bitcoin

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

    # Cache DD momentum adjustments to avoid recomputing on every page load
    _dd_adj_cache_key = f"dd_adj_{cache_key}"
    if st.session_state.get("_dd_adj_key") == _dd_adj_cache_key:
        _dd_adj = st.session_state._dd_adj_cached
    else:
        _dd_adj = None
        # Try precomputed adjustments first (instant load on default settings)
        if _constraints_are_default() and PRECOMPUTED_FILE.exists():
            try:
                with open(PRECOMPUTED_FILE, "rb") as f:
                    _pre_dd = pickle.load(f)
                _btc_key = "excl_btc" if exclude_bitcoin else "incl_btc"
                if "btc_variants" in _pre_dd and _btc_key in _pre_dd["btc_variants"]:
                    _dd_adj = _pre_dd["btc_variants"][_btc_key].get("dd_momentum_adjustments")
                else:
                    _dd_adj = _pre_dd.get("dd_momentum_adjustments")
            except Exception:
                pass
        if _dd_adj is None:
            _optimal_bump_sched = load_optimal_bump_schedule()
            if _optimal_bump_sched is not None:
                _dd_adj = compute_dd_adjustments_scheduled(returns, _checkpoints, _optimal_bump_sched)
            else:
                _dd_adj = compute_dd_adjustments(returns, _checkpoints)
        st.session_state._dd_adj_cached = _dd_adj
        st.session_state._dd_adj_key = _dd_adj_cache_key

    _rebal_for_dynamic = rebalance if rebalance != "daily" else "annual"
    _dd_schedule = build_dd_momentum_schedule(
        _ms_weights, _dd_adj,
        dd_constraint=tv_dd_constraint_val,
        returns=returns,
        risk_free_rate=risk_free_rate,
        rebalance=_rebal_for_dynamic,
        asset_starts=bt_asset_starts,
    )
    # Hard-enforce the time-varying DD cap on the full backtest
    _dd_schedule = _enforce_schedule_dd(
        _dd_schedule, returns, _ms_weights, tv_dd_constraint_val,
        risk_free_rate, _rebal_for_dynamic, bt_asset_starts,
    )

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
    _regime_base_key = _cache_key(f"regime_{overlap_start_date}_{tv_dd_constraint_pct}", end_date, risk_free_rate, "daily")
    _regime_cache_key = f"{_regime_base_key}_btc={exclude_bitcoin}"

    if st.session_state.get("_regime_cache_key") != _regime_cache_key:
        # Fast path: if only BTC toggled and we have cached regime weights,
        # just zero out BTC and redistribute rather than full reoptimization
        _prev_regime_key = st.session_state.get("_regime_cache_key", "")
        _have_regime_cached = "_regime_weights" in st.session_state
        _regime_btc_fast = (_have_regime_cached
                            and _regime_base_key in _prev_regime_key
                            and f"btc={not exclude_bitcoin}" in _prev_regime_key)

        if _regime_btc_fast:
            _cached_rw = st.session_state._regime_weights
            if exclude_bitcoin:
                btc_idx = ASSETS.index("Bitcoin")
                _regime_weights = {}
                for label, w in _cached_rw.items():
                    w_new = w.copy()
                    w_new[btc_idx] = 0.0
                    remaining = w_new.sum()
                    if remaining > 1e-12:
                        w_new /= remaining
                    _regime_weights[label] = w_new
            else:
                # Toggling OFF: need to reoptimize from scratch
                with st.sidebar:
                    with st.spinner("Computing regime portfolios..."):
                        _regime_weights = optimize_per_regime(
                            opt_returns, _regime_series, "Max Sharpe Ratio",
                            min_w_default, max_w_default, default_group_max,
                            risk_free_rate, rebalance="monthly",
                            dd_constraint=tv_dd_constraint_val,
                            dd_returns=returns, dd_asset_starts=bt_asset_starts,
                        )
        else:
            # Try precomputed regime weights first (instant load)
            _regime_weights = None
            if _constraints_are_default() and PRECOMPUTED_FILE.exists():
                try:
                    with open(PRECOMPUTED_FILE, "rb") as f:
                        _pre = pickle.load(f)
                    _btc_key = "excl_btc" if exclude_bitcoin else "incl_btc"
                    if "btc_variants" in _pre and _btc_key in _pre["btc_variants"]:
                        _pre_variant = _pre["btc_variants"][_btc_key]
                    else:
                        _pre_variant = _pre
                    _pre_rw_by_dd = _pre_variant.get("regime_weights_by_dd", {})
                    if tv_dd_constraint_pct in _pre_rw_by_dd:
                        _regime_weights = _pre_rw_by_dd[tv_dd_constraint_pct]
                    elif _pre_variant.get("regime_weights") is not None:
                        _regime_weights = _pre_variant["regime_weights"]
                except Exception:
                    pass

            if _regime_weights is None:
                with st.sidebar:
                    with st.spinner("Computing regime portfolios..."):
                        _regime_weights = optimize_per_regime(
                            opt_returns, _regime_series, "Max Sharpe Ratio",
                            min_w_default, max_w_default, default_group_max,
                            risk_free_rate, rebalance="monthly",
                            dd_constraint=tv_dd_constraint_val,
                            dd_returns=returns, dd_asset_starts=bt_asset_starts,
                        )
        st.session_state._regime_weights = _regime_weights
        st.session_state._regime_cache_key = _regime_cache_key
    else:
        _regime_weights = st.session_state._regime_weights

    _rebal_for_regime = rebalance if rebalance != "daily" else "annual"
    _regime_schedule = build_regime_schedule(
        returns.index, _regime_series, _regime_weights, _rebal_for_regime,
    )
    # Hard-enforce the time-varying DD cap on the full backtest
    _regime_repr_w_pre = np.mean(list(_regime_weights.values()), axis=0)
    _regime_schedule = _enforce_schedule_dd(
        _regime_schedule, returns, _regime_repr_w_pre, tv_dd_constraint_val,
        risk_free_rate, _rebal_for_regime, bt_asset_starts,
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

# --- Rolling Re-Optimisation (Time-Varying) ---
if "base_results" in st.session_state:
    from rolling_optimizer import build_rolling_optimization_schedule

    _ms_base_w = st.session_state.base_results.get(
        "Max Sharpe Ratio", list(st.session_state.base_results.values())[0]
    )["weights"]

    with st.sidebar:
        with st.spinner("Computing Max Sharpe (time-varying)..."):
            _ms_tv_schedule = build_rolling_optimization_schedule(
                returns, "Max Sharpe Ratio",
                min_w_default, max_w_default, default_group_max, risk_free_rate,
                window_years=5, leverage=cfd_leverage_opt, financing_rate=cfd_financing_opt,
            )
    if _ms_tv_schedule:
        _ms_tv_repr_w = _ms_tv_schedule[max(_ms_tv_schedule.keys())]
        _rebal_for_tv = rebalance if rebalance != "daily" else "annual"
        _ms_tv_stats = calc_stats(
            returns, _ms_base_w, risk_free_rate,
            rebalance=_rebal_for_tv, asset_starts=bt_asset_starts,
            weights_schedule=_ms_tv_schedule,
        )
        st.session_state.portfolios["Max Sharpe (time-varying)"] = {
            "weights": _ms_tv_repr_w,
            "stats": _ms_tv_stats,
            "weights_schedule": _ms_tv_schedule,
            "strategy_type": "rolling_sharpe",
        }

    with st.sidebar:
        with st.spinner("Computing Leverage-Optimal (time-varying)..."):
            _lo_tv_schedule = build_rolling_optimization_schedule(
                returns, "Leverage-Optimal",
                min_w_default, max_w_default, default_group_max, risk_free_rate,
                window_years=5, leverage=cfd_leverage_opt, financing_rate=cfd_financing_opt,
            )
    if _lo_tv_schedule:
        _lo_tv_repr_w = _lo_tv_schedule[max(_lo_tv_schedule.keys())]
        _lo_tv_stats = calc_stats(
            returns, _ms_base_w, risk_free_rate,
            rebalance=_rebal_for_tv, asset_starts=bt_asset_starts,
            weights_schedule=_lo_tv_schedule,
        )
        st.session_state.portfolios["Leverage-Optimal (time-varying)"] = {
            "weights": _lo_tv_repr_w,
            "stats": _lo_tv_stats,
            "weights_schedule": _lo_tv_schedule,
            "strategy_type": "rolling_leverage",
        }


# --- Yield Curve Signal Overlay ---
if _macro_data is not None and "base_results" in st.session_state:
    from yield_signal import build_yield_signal_schedule, yield_signal_analytics

    _yc_base_w = _ms_base_w  # Max Sharpe base
    _rebal_for_yc = rebalance if rebalance != "daily" else "monthly"
    _yc_schedule = build_yield_signal_schedule(
        _yc_base_w, _macro_data, returns,
        rebalance=_rebal_for_yc,
    )
    _yc_schedule = _enforce_schedule_dd(
        _yc_schedule, returns, _yc_base_w, tv_dd_constraint_val,
        risk_free_rate, _rebal_for_yc, bt_asset_starts,
    )
    _yc_repr_w = _yc_schedule[max(_yc_schedule.keys())] if _yc_schedule else _yc_base_w
    _yc_stats = calc_stats(
        returns, _yc_base_w, risk_free_rate,
        rebalance=_rebal_for_yc, asset_starts=bt_asset_starts,
        weights_schedule=_yc_schedule,
    )
    st.session_state.portfolios["Yield Curve Signal (time-varying)"] = {
        "weights": _yc_repr_w,
        "stats": _yc_stats,
        "weights_schedule": _yc_schedule,
        "strategy_type": "yield_signal",
    }
    st.session_state._yc_schedule = _yc_schedule

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
        "Current DD": s.current_drawdown * 100,
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
    "Current DD": st.column_config.NumberColumn("Current DD", format="%.1f%%"),
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
tab_compare, tab_dynamic, tab_cfd, tab_guide, tab_settings = st.tabs(["Compare", "Dynamic Strategies", "CFD Analysis", "Methodology", "Settings"])

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

        with st.expander("ℹ️ Strategy descriptions"):
            _STRATEGY_DESCRIPTIONS = {
                "Max Sharpe Ratio": "Maximises return per unit of risk (Sharpe ratio). Constrained by asset/group limits.",
                "Leverage-Optimal": "Finds the unleveraged allocation that compounds best when leveraged. Penalises vol drag quadratically — tends to favour low-vol assets.",
                "Max Sharpe (Unconstrained)": "Max Sharpe without a drawdown cap — allows higher concentration in outperforming assets.",
                "Leverage-Optimal (Unconstrained)": "Leverage-Optimal without a drawdown cap.",
                f"Max Sharpe (DD \u2264 {dd_constraint_pct}%)": f"Max Sharpe with a hard {dd_constraint_pct}% max drawdown ceiling enforced during optimisation (monthly rebalancing).",
                f"Max Calmar (DD \u2264 {dd_constraint_pct}%)": f"Max Calmar with a hard {dd_constraint_pct}% max drawdown ceiling enforced during optimisation (monthly rebalancing).",
                "Regime-Based (time-varying)": "Switches weights based on the current macro regime (high/low inflation × high/low interest rates). A separate portfolio is optimised per regime.",
                "DD P-Value Momentum (time-varying)": "Annually boosts assets with historically rare drawdowns (buying opportunities) and trims assets near all-time highs. Includes confidence scaling and trend filter.",
                "Yield Curve Signal (time-varying)": "Tilts defensive when the Fed Funds rate rises >200bp/yr (rate tightening signal). Runs full allocation when rates are stable or falling.",
                "Max Sharpe (time-varying)": "Re-optimises Max Sharpe annually using a trailing 5-year window. Adapts to changing market conditions without lookahead.",
                "Leverage-Optimal (time-varying)": "Re-optimises Leverage-Optimal annually using a trailing 5-year window. Adapts post-leverage Sharpe to evolving vol/correlation regimes.",
                "Dalio All Weather": "Ray Dalio's classic: 30% S&P 500, 40% LT Treasuries, 15% ST Treasuries, 7.5% Gold, 7.5% Commodities. An unoptimised reference benchmark.",
            }
            for sname, sdesc in _STRATEGY_DESCRIPTIONS.items():
                if any(sname in p or p == sname for p in st.session_state.portfolios):
                    st.markdown(f"**{sname}** — {sdesc}")

        st.subheader("Allocation Weights by Strategy")
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

        # Sort selected portfolios by final equity value (highest first) for legend ordering
        _eq_sorted = sorted(eq_selected, key=lambda p: _eq_cache[p].iloc[-1], reverse=True)

        fig_eq = go.Figure()
        for idx, pname in enumerate(_eq_sorted):
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
            margin=dict(l=70, r=20, t=50, b=50),
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

        # Default to top 2 portfolios by CAGR
        _sorted_by_cagr = sorted(portfolio_names,
                                  key=lambda p: st.session_state.portfolios[p]["stats"].cagr,
                                  reverse=True)
        _ann_default = _sorted_by_cagr[:2]

        ann_selected = st.multiselect(
            "Select portfolios",
            portfolio_names,
            default=_ann_default,
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
            fig_ann.add_trace(go.Scatter(
                x=ann.index.astype(str), y=ann.values, name=pname,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6),
            ))
        fig_ann.update_layout(
            title="Annual Returns",
            yaxis_title="Return",
            yaxis_tickformat=".2%",
            hovermode="x unified",
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
    st.caption(f"Strategies that adjust allocations over time based on market conditions. "
               f"Max drawdown constraint ({tv_dd_constraint_pct}%) applied via sidebar (hard-enforced on full backtest).")

    # ── Summary metric cards ──
    _dyn_summary = [
        ("Regime-Based (time-varying)", "Regime-Based"),
        ("DD P-Value Momentum (time-varying)", "DD Momentum"),
        ("Yield Curve Signal (time-varying)", "Yield Curve"),
        ("Max Sharpe (time-varying)", "Max Sharpe TV"),
        ("Leverage-Optimal (time-varying)", "Lev-Opt TV"),
    ]
    _dyn_summary_cols = st.columns(5)
    for _col, (_skey, _slabel) in zip(_dyn_summary_cols, _dyn_summary):
        with _col:
            if _skey in st.session_state.portfolios:
                _s = st.session_state.portfolios[_skey]["stats"]
                st.metric(
                    _slabel,
                    f"{_s.cagr * 100:.1f}% CAGR",
                    f"Sharpe {_s.sharpe:.2f}  ·  Max DD {_s.max_drawdown * 100:.0f}%",
                )

    st.divider()

    # ── Section A: Regime-Based Allocation ──
    st.subheader("Regime-Based Allocation")
    if not st.session_state.get("_macro_available", False):
        st.info("Regime data not available. Place 'Inflation and IR.xlsx' in the app directory with CPI and Fed Funds Rate data.")
    elif "_regime_series" in st.session_state and "_regime_weights" in st.session_state:
        _rs = st.session_state._regime_series
        _rw = st.session_state._regime_weights
        _analytics = regime_analytics(_rs, _rw, ASSETS)

        # Current regime
        _cur_regime_label = REGIME_LABELS.get(_analytics['current_regime'], 'Unknown')
        _cur_regime_color = {1: "#FF1744", 2: "#FF6D00", 3: "#2962FF", 4: "#00C853"}.get(
            _analytics['current_regime'], "#888")
        st.markdown(
            f"**Current Regime:** "
            f'<span style="color:{_cur_regime_color}; font-weight:600">{_cur_regime_label}</span>',
            unsafe_allow_html=True,
        )
        fig_regime = go.Figure()
        _regime_colors = {1: "#FF1744", 2: "#FF6D00", 3: "#2962FF", 4: "#00C853"}
        _regime_daily = _rs.reindex(pd.date_range(_rs.index[0], _rs.index[-1], freq="MS")).ffill()

        # Build contiguous segments for each regime
        dates = _regime_daily.index
        regimes_arr = _regime_daily.values
        for label_id, label_name in REGIME_LABELS.items():
            color = _regime_colors.get(label_id, "#999")
            # Find contiguous segments for this regime
            segments = []
            seg_start = None
            for j in range(len(dates)):
                if regimes_arr[j] == label_id:
                    if seg_start is None:
                        seg_start = j
                else:
                    if seg_start is not None:
                        segments.append((seg_start, j - 1))
                        seg_start = None
            if seg_start is not None:
                segments.append((seg_start, len(dates) - 1))

            # Add one shape per segment, but only one legend entry
            for k, (s, e) in enumerate(segments):
                fig_regime.add_trace(go.Scatter(
                    x=[dates[s], dates[e], dates[e], dates[s], dates[s]],
                    y=[0, 0, 1, 1, 0],
                    fill="toself",
                    fillcolor=color,
                    line=dict(width=0),
                    mode="lines",
                    name=label_name,
                    showlegend=(k == 0),
                    hoverinfo="name+x",
                ))

        fig_regime.update_layout(
            xaxis_title="Date",
            yaxis=dict(visible=False, range=[0, 1]),
            height=200,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5,
                font=dict(size=10),
                itemsizing="constant",
                traceorder="normal",
                itemwidth=30,
            ),
            margin=dict(b=40, t=50, l=40, r=40),
        )
        st.plotly_chart(fig_regime, width="stretch")

        # Regime stats — colour-coded labels
        _regime_text_colors = {1: "#FF1744", 2: "#FF6D00", 3: "#2962FF", 4: "#00C853"}
        stat_cols = st.columns(4)
        for i, (label_id, label_name) in enumerate(REGIME_LABELS.items()):
            with stat_cols[i]:
                _rc = _regime_text_colors.get(label_id, "#888")
                count = _analytics["regime_counts"].get(label_id, 0)
                avg_m = _analytics["regime_avg_months"].get(label_id, 0)
                st.markdown(
                    f'<span style="color:{_rc}; font-weight:600; font-size:0.95em">{label_name}</span><br>'
                    f'<span style="color:#888; font-size:0.85em">{count} periods · avg {avg_m:.0f} months</span>',
                    unsafe_allow_html=True,
                )

        # Weights per regime table (transposed: assets as rows, regimes as columns)
        st.caption("Optimal Weights per Regime")
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
            _rv_all = np.concatenate([eq_regime.values, eq_sharpe.values])
            _rv_ymin = max(_rv_all.min() * 0.8, 0.5)
            _rv_ymax = _rv_all.max() * 1.2
            _rv_ticks = []
            for exp in range(-1, 5):
                base = 10 ** exp
                for mult in [1, 2, 5]:
                    val = base * mult
                    if _rv_ymin <= val <= _rv_ymax:
                        _rv_ticks.append(val)
            if not _rv_ticks:
                _rv_ticks = [1]
            fig_rv.update_layout(
                yaxis_type="log", yaxis_title="Growth of $1 (Log)",
                xaxis_title="Date", hovermode="x unified",
                template="plotly_white", height=400,
                yaxis=dict(
                    tickmode="array",
                    tickvals=_rv_ticks,
                    ticktext=[_fmt_growth(v) for v in _rv_ticks],
                    range=[np.log10(max(_rv_ymin, 0.1)), np.log10(max(_rv_ymax, 1.1))],
                    gridcolor="rgba(128,128,128,0.15)",
                ),
            )
            st.plotly_chart(fig_rv, width="stretch")

    st.divider()

    # ── Section B: DD P-Value Momentum ──
    st.subheader("Drawdown P-Value Momentum")
    st.caption("Boosts allocation to assets with historically rare drawdowns (buying opportunities) "
               "and reduces allocation to assets near all-time highs. "
               "Confidence scaling dampens bumps for assets with short history; "
               "trend filter halves positive bumps for assets in sustained downtrends.")

    if "_dd_checkpoints" in st.session_state and "base_results" in st.session_state:
        _dd_an = dd_analytics(returns, end_ts)

        # Current p-values and bump factors table
        _dd_rows = []
        sorted_idx = np.argsort(_dd_an["pvalues"])
        for i in sorted_idx:
            _dd_rows.append({
                "Asset": _dd_an["assets"][i],
                "Current DD": _dd_an["current_dd"][i] * 100,
                "P-Value": _dd_an["pvalues"][i],
                "Episodes": int(_dd_an["episode_counts"][i]),
                "Confidence": _dd_an["confidence"][i] * 100,
                "Trending Down": bool(_dd_an["in_downtrend"][i]),
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
                "Confidence": st.column_config.NumberColumn("Confidence", format="%.0f%%",
                    help="How much to trust the p-value signal. Scales with number of historical episodes (100% = 15+ episodes)."),
                "Trending Down": st.column_config.CheckboxColumn("Trending Down",
                    help="Asset is below its 252-day SMA. Positive bumps are halved to avoid catching falling knives."),
                "Bump Factor": st.column_config.NumberColumn("Bump Factor", format="%+.0f%%",
                    help="Net adjustment applied to base allocation (after confidence scaling and trend filter)."),
            },
            width="stretch",
            hide_index=True,
        )

        # Allocation at a given date
        if "_dd_schedule" in st.session_state and st.session_state._dd_schedule:
            st.caption("Allocation at Date")
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
            _ddv_all = np.concatenate([eq_ddm.values, eq_sharpe2.values])
            _ddv_ymin = max(_ddv_all.min() * 0.8, 0.5)
            _ddv_ymax = _ddv_all.max() * 1.2
            _ddv_ticks = []
            for exp in range(-1, 5):
                base = 10 ** exp
                for mult in [1, 2, 5]:
                    val = base * mult
                    if _ddv_ymin <= val <= _ddv_ymax:
                        _ddv_ticks.append(val)
            if not _ddv_ticks:
                _ddv_ticks = [1]
            fig_ddv.update_layout(
                yaxis_type="log", yaxis_title="Growth of $1 (Log)",
                xaxis_title="Date", hovermode="x unified",
                template="plotly_white", height=400,
                yaxis=dict(
                    tickmode="array",
                    tickvals=_ddv_ticks,
                    ticktext=[_fmt_growth(v) for v in _ddv_ticks],
                    range=[np.log10(max(_ddv_ymin, 0.1)), np.log10(max(_ddv_ymax, 1.1))],
                    gridcolor="rgba(128,128,128,0.15)",
                ),
            )
            st.plotly_chart(fig_ddv, width="stretch")
    else:
        st.info("DD Momentum data not available — ensure base portfolios are computed.")

    # ── Section C: Yield Curve Signal Overlay ──
    st.divider()
    st.subheader("Yield Curve Signal Overlay")
    st.caption("Tilts defensive when the Fed Funds rate is rising >200 bp/yr. "
               "Uses 12-month rolling change in Fed Funds as signal.")

    if "_yc_schedule" in st.session_state and st.session_state._yc_schedule:
        _ya = yield_signal_analytics(_macro_data)
        if _ya["dates"]:
            fig_yc = go.Figure()
            colors = ["rgba(255,23,68,0.3)" if s else "rgba(0,200,83,0.3)" for s in _ya["signals"]]
            fig_yc.add_trace(go.Bar(
                x=_ya["dates"], y=_ya["ff_changes"],
                marker_color=colors, name="FF 12m Change (pp)",
            ))
            fig_yc.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="200bp threshold")
            fig_yc.update_layout(
                yaxis_title="Fed Funds 12m Change (pp)", xaxis_title="Date",
                template="plotly_white", height=250,
            )
            st.plotly_chart(fig_yc, width="stretch")

            # Equity curve comparison
            if "Yield Curve Signal (time-varying)" in st.session_state.portfolios and "Max Sharpe Ratio" in st.session_state.portfolios:
                st.caption("Yield Curve Signal vs Max Sharpe (Static)")
                eq_yc = compute_equity_curve(returns, _yc_base_w,
                                              rebalance=_rebal_for_yc, asset_starts=bt_asset_starts,
                                              weights_schedule=st.session_state._yc_schedule)
                _ms_w_y = st.session_state.portfolios["Max Sharpe Ratio"]["weights"]
                eq_sharpe_y = compute_equity_curve(returns, _ms_w_y,
                                                    rebalance=rebalance, asset_starts=bt_asset_starts)
                fig_ycv = go.Figure()
                fig_ycv.add_trace(go.Scatter(x=eq_yc.index, y=eq_yc.values, name="Yield Curve Signal (time-varying)", line=dict(color="#26A69A")))
                fig_ycv.add_trace(go.Scatter(x=eq_sharpe_y.index, y=eq_sharpe_y.values, name="Max Sharpe (Static)", line=dict(color="#2962FF")))
                fig_ycv.update_layout(yaxis_type="log", yaxis_title="Growth of $1 (Log)", xaxis_title="Date",
                                       template="plotly_white", height=350, hovermode="x unified")
                st.plotly_chart(fig_ycv, width="stretch")
    elif _macro_data is None:
        st.info("Yield Curve Signal requires macro data (Inflation and IR.xlsx).")
    else:
        st.info("Yield Curve Signal data not available.")

    # ── Section D: Rolling Re-Optimisation Strategies ──
    st.divider()
    st.subheader("Rolling Re-Optimisation")
    st.caption("Re-optimises the objective function annually using a trailing 5-year window. "
               "No lookahead bias — each year's allocation only uses data available at the time.")

    for _tv_key, _tv_label, _static_key, _tv_color in [
        ("Max Sharpe (time-varying)", "Max Sharpe (Time-Varying)", "Max Sharpe Ratio", "#9C27B0"),
        ("Leverage-Optimal (time-varying)", "Leverage-Optimal (Time-Varying)", "Leverage-Optimal", "#E91E63"),
    ]:
        if _tv_key in st.session_state.portfolios:
            _tv_data = st.session_state.portfolios[_tv_key]
            _tv_s = _tv_data["stats"]
            st.markdown(f"**{_tv_label}** — CAGR {_tv_s.cagr*100:.1f}%, "
                        f"Sharpe {_tv_s.sharpe:.2f}, Max DD {_tv_s.max_drawdown*100:.0f}%")

            # Equity curve comparison vs static counterpart
            if _static_key in st.session_state.portfolios:
                _static_w = st.session_state.portfolios[_static_key]["weights"]
                _tv_sched = _tv_data.get("weights_schedule", {})
                if _tv_sched:
                    _tv_base_w = list(_tv_sched.values())[0]
                    _rebal_tv = rebalance if rebalance != "daily" else "annual"
                    eq_tv = compute_equity_curve(returns, _tv_base_w,
                                                 rebalance=_rebal_tv, asset_starts=bt_asset_starts,
                                                 weights_schedule=_tv_sched)
                    eq_static = compute_equity_curve(returns, _static_w,
                                                      rebalance=rebalance, asset_starts=bt_asset_starts)
                    fig_tv = go.Figure()
                    fig_tv.add_trace(go.Scatter(x=eq_tv.index, y=eq_tv.values,
                                                name=_tv_label, line=dict(color=_tv_color)))
                    fig_tv.add_trace(go.Scatter(x=eq_static.index, y=eq_static.values,
                                                name=f"{_static_key} (Static)", line=dict(color="#2962FF")))
                    fig_tv.update_layout(yaxis_type="log", yaxis_title="Growth of $1 (Log)", xaxis_title="Date",
                                         template="plotly_white", height=350, hovermode="x unified",
                                         margin=dict(l=70, r=20, t=30, b=50))
                    st.plotly_chart(fig_tv, use_container_width=True)

    if "Max Sharpe (time-varying)" not in st.session_state.portfolios:
        st.info("Rolling re-optimisation not available — ensure base portfolios are computed.")

# ======================================================================
# TAB 3: CFD ANALYSIS
# ======================================================================
with tab_cfd:
    if not portfolio_names:
        st.info("Computing default portfolios...")
    else:
        cfd_pick_cols = st.columns([2, 1, 1, 1])
        with cfd_pick_cols[0]:
            cfd_port_name = st.selectbox("Analyse Portfolio", portfolio_names, key="cfd_port_pick")
        with cfd_pick_cols[1]:
            cfd_capital = st.number_input("Total Capital ($)", value=10000, min_value=1000, step=1000, key="cfd_capital")
        with cfd_pick_cols[2]:
            cfd_leverage = st.number_input("Leverage", value=5.0, min_value=1.0, max_value=20.0, step=0.5, key="cfd_leverage")
        with cfd_pick_cols[3]:
            cfd_fin_pct = st.number_input("Financing Rate (%)", value=6.5, min_value=0.0, max_value=20.0, step=0.5, format="%.1f", key="cfd_financing")
            cfd_financing = cfd_fin_pct / 100.0

        cfd_port = st.session_state.portfolios[cfd_port_name]
        cfd_weights = cfd_port["weights"]
        cfd_stats = cfd_port["stats"]

        cfd_result = analyze_cfd(
            cfd_weights, cfd_stats, cfd_capital, cfd_leverage,
            cfd_financing, risk_free_rate=risk_free_rate,
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
             "Leveraged CAGR after vol drag and dividend tax drag, before financing"),
            ("Dividend / Coupon Drag", f"-{cfd_result.dividend_drag:.2%}",
             "Tax drag on dividend adjustments (est. yield × 18% × leverage) — already reflected in Gross CAGR above"),
            ("Financing Drag", f"-{cfd_result.financing_drag:.2%}",
             "Annual holding cost on full notional (rate × leverage)"),
            ("Net CAGR (on Deployed)", f"{cfd_result.net_cagr:.2%}",
             "CAGR after financing and dividend drag, on deployed capital"),
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
                    "Margin %": ASSET_MARGIN_RATES.get(asset, 0.20) * 100,
                    "Margin ($)": round(cfd_result.margin_per_asset.get(asset, 0.0)),
                })
        cfd_alloc_df = pd.DataFrame(cfd_alloc_rows)
        st.dataframe(
            cfd_alloc_df,
            column_config={
                "Asset": st.column_config.TextColumn("Asset"),
                "Weight": st.column_config.NumberColumn("Weight", format="%.2f%%"),
                "Notional": st.column_config.NumberColumn("Notional ($)", format="$%.0f"),
                "Margin %": st.column_config.NumberColumn("Margin %", format="%.0f%%"),
                "Margin ($)": st.column_config.NumberColumn("Margin ($)", format="$%.0f"),
            },
            width="stretch",
            hide_index=True,
        )

        # ── CMC Markets Financing Rate Reference ──
        st.markdown("---")
        with st.expander("CMC Markets Financing Rate Reference", expanded=False):
            st.markdown(
                "CFD financing is charged **daily** on the full notional value of open positions held overnight. "
                "The rate comprises a benchmark interbank rate plus a CMC spread (typically 2.5% p.a.)."
            )
            _fin_rates = [
                ("Cash", "N/A", "N/A", "N/A", "Not a CFD position"),
                ("Nasdaq", "Index CFD", "Interbank + 2.5% (~7.0%)", "Interbank \u2013 2.5%", "Fed Funds benchmark"),
                ("S&P 500", "Index CFD", "Interbank + 2.5% (~7.0%)", "Interbank \u2013 2.5%", "Fed Funds benchmark"),
                ("Russell 2000", "Index CFD", "Interbank + 2.5% (~7.0%)", "Interbank \u2013 2.5%", "Fed Funds benchmark"),
                ("ASX200", "Index CFD", "Interbank + 2.5% (~6.6%)", "Interbank \u2013 2.5%", "RBA cash rate benchmark"),
                ("Emerging Markets", "Index CFD", "Interbank + 2.5% (~7.0%)", "Interbank \u2013 2.5%", "USD benchmark (typical)"),
                ("Corporate Bonds", "Index CFD", "Interbank + 2.5%", "Interbank \u2013 2.5%", "Varies by underlying"),
                ("Long-Term Treasuries", "Index CFD", "Interbank + 2.5%", "Interbank \u2013 2.5%", "Varies by underlying"),
                ("Short-Term Treasuries", "Index CFD", "Interbank + 2.5%", "Interbank \u2013 2.5%", "Varies by underlying"),
                ("US REITs", "Share CFD (US)", "Interbank + 2.5% (~7.5%)", "Interbank \u2013 2.5%", "USD benchmark"),
                ("Industrial Metals", "Commodity CFD", "Built into spread", "Built into spread", "Forward-based pricing"),
                ("Gold", "Commodity CFD", "SOFR + spread (~6.8%)", "SOFR \u2013 spread", "Precious metals pricing"),
                ("Bitcoin", "Crypto CFD", "Interbank + 2.5%", "Interbank \u2013 2.5%", "Higher spreads typical"),
                ("Infrastructure", "Index CFD", "Interbank + 2.5%", "Interbank \u2013 2.5%", "Varies by underlying"),
                ("Japan Equities", "Index CFD", "Interbank + 2.5% (~3.0%)", "Interbank \u2013 2.5%", "BoJ rate benchmark"),
                ("UK Equities", "Index CFD", "Interbank + 2.5% (~7.0%)", "Interbank \u2013 2.5%", "BoE rate benchmark"),
                ("EU Equities", "Index CFD", "Interbank + 2.5% (~5.2%)", "Interbank \u2013 2.5%", "ECB refi rate benchmark"),
                ("US TIPS", "Index CFD", "Interbank + 2.5%", "Interbank \u2013 2.5%", "USD benchmark"),
                ("High Yield", "Index CFD", "Interbank + 2.5%", "Interbank \u2013 2.5%", "USD benchmark"),
                ("EM Debt", "Index CFD", "Interbank + 2.5%", "Interbank \u2013 2.5%", "USD benchmark"),
                ("JPY", "FX CFD", "Tom-next + 1.0%", "Tom-next \u2013 1.0%", "JPY/USD rate differential"),
                ("CHF", "FX CFD", "Tom-next + 1.0%", "Tom-next \u2013 1.0%", "CHF/USD rate differential"),
                ("CNY", "FX CFD", "Tom-next + 1.0%", "Tom-next \u2013 1.0%", "CNH/USD rate differential"),
                ("China Equities", "Index CFD", "Interbank + 2.5%", "Interbank \u2013 2.5%", "CNH benchmark"),
                ("Copper", "Commodity CFD", "Built into spread", "Built into spread", "Forward-based pricing"),
                ("Soft Commodities", "Commodity CFD", "Built into spread", "Built into spread", "Forward-based pricing"),
            ]
            _fin_df = pd.DataFrame(_fin_rates, columns=["Asset", "CMC Product Type", "Long Rate", "Short Rate", "Notes"])
            st.dataframe(_fin_df, hide_index=True, width="stretch")

            st.markdown(
                "**Additional notes:**\n"
                "- Charged/credited daily at 5pm New York time. Weekend = 3\u00d7 daily charge (Wed or Thu depending on product).\n"
                "- Short share positions may attract additional borrowing fees for hard-to-borrow stocks.\n"
                "- AU share CFD commission: 0.10% per side (min $10 AUD). US share CFDs: US$0.02/share (min $10 USD).\n"
                "- If interbank rate \u2264 2.5%, short positions may incur a charge rather than receiving a credit.\n"
                "- Rates are indicative as at March 2026 \u2013 verify on CMC\u2019s platform."
            )

            st.subheader("Daily Financing Cost Calculator")
            st.caption(
                "Quick what-if calculator. Pre-filled with this portfolio's notional "
                f"(${cfd_result.notional_exposure:,.0f}) and the financing rate above "
                f"({cfd_financing*100:.1f}%) — adjust to model any position."
            )
            _calc_cols = st.columns(3)
            with _calc_cols[0]:
                _calc_notional = st.number_input(
                    "Notional Position Value ($)",
                    value=int(round(cfd_result.notional_exposure)),
                    min_value=0, step=1000,
                    key="fin_calc_notional",
                )
            with _calc_cols[1]:
                _calc_rate = st.number_input(
                    "Annual Financing Rate (%)",
                    value=float(cfd_financing * 100),
                    min_value=0.0, max_value=20.0,
                    step=0.1, format="%.1f", key="fin_calc_rate",
                )
            with _calc_cols[2]:
                _calc_daily = _calc_notional * (_calc_rate / 100.0) / 365.0
                _calc_monthly = _calc_daily * 21  # 21 trading days
                st.metric("Daily Cost", f"${_calc_daily:.2f} AUD")
                st.metric("Monthly Cost (21 trading days)", f"${_calc_monthly:.2f} AUD")

        # ── Monte Carlo: Leveraged vs Unleveraged forward projection ──
        st.markdown("---")
        st.header("10-Year Monte Carlo Projection")
        st.caption("Forward simulation using historical CAGR and volatility with geometric Brownian motion (500 paths).")

        n_paths = 500
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

        # Leveraged: simulate deployed capital growth, then add back cash reserve.
        # The reserve sits in cash earning the risk-free rate.
        lev_cagr = cfd_result.net_cagr
        lev_vol = cfd_result.leveraged_volatility

        def _mc_fan_leveraged(deployed, reserve, cagr, vol, rf_rate):
            """MC fan for leveraged portfolio: deployed grows via GBM, reserve sits as cash (0% return)."""
            daily_mu = np.log(1.0 + cagr) / 252.0
            daily_sigma = vol / np.sqrt(252.0)
            daily_log_ret = daily_mu + daily_sigma * z
            cum_log = np.cumsum(daily_log_ret, axis=1)
            cum_log = np.hstack([np.zeros((n_paths, 1)), cum_log])
            deployed_paths = deployed * np.exp(cum_log)
            # Reserve earns 0% — CMC Markets does not pay interest on idle cash
            total_paths = deployed_paths + reserve
            p5 = np.percentile(total_paths, 5, axis=0)
            p25 = np.percentile(total_paths, 25, axis=0)
            p50 = np.percentile(total_paths, 50, axis=0)
            p75 = np.percentile(total_paths, 75, axis=0)
            p95 = np.percentile(total_paths, 95, axis=0)
            return p5, p25, p50, p75, p95

        lv_p5, lv_p25, lv_p50, lv_p75, lv_p95 = _mc_fan_leveraged(
            cfd_result.deployed_capital, cfd_result.cash_reserve,
            lev_cagr, lev_vol, risk_free_rate,
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
        st.caption("Shaded bands: P5–P95 (light) and P25–P75 (dark). Solid line: median (P50). "
                   "Leveraged returns account for volatility drag (L(L-1)σ²/2) and financing costs on full notional. "
                   "Cash reserve earns 0% (CMC does not pay interest on idle cash).")

        # ── 10-Year Median Projection Table (all portfolios) ──
        st.markdown("---")
        st.header("10-Year Median Projection")
        st.caption(f"Median portfolio value after 10 years per ${cfd_capital:,.0f} capital, computed from Monte Carlo simulation ({n_paths} paths). "
                   f"Leveraged column uses {cfd_leverage:.0f}x leverage with {cfd_fin_pct:.1f}% financing.")

        def _mc_median_10y(start_val, cagr, vol, reserve=0.0, rf_rate=0.0):
            """Compute the actual MC median at year 10 (not the deterministic formula)."""
            daily_mu = np.log(1.0 + cagr) / 252.0
            daily_sigma = vol / np.sqrt(252.0)
            daily_log_ret = daily_mu + daily_sigma * z  # reuse the shared z matrix
            cum_log = np.sum(daily_log_ret, axis=1)  # sum over all days for final value
            deployed_final = start_val * np.exp(cum_log)
            # Reserve earns 0% — CMC does not pay interest on idle cash
            final_vals = deployed_final + reserve
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
                cfd_financing, risk_free_rate=risk_free_rate,
            )
            lev_median_10y = _mc_median_10y(
                p_cfd.deployed_capital, p_cfd.net_cagr, p_cfd.leveraged_volatility,
                reserve=p_cfd.cash_reserve, rf_rate=risk_free_rate,
            )

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
# METHODOLOGY TAB
# ──────────────────────────────────────────────
with tab_guide:
    st.header("Methodology")
    st.caption("How each portfolio strategy works, what the key metrics mean, and how the analysis tools are built.")

    # ── Optimisation Strategies ──
    with st.expander("Optimisation Strategies", expanded=True):
        st.markdown(
            "**Max Sharpe Ratio** maximises return per unit of risk. "
            "**Min Volatility** targets the lowest portfolio standard deviation. "
            "**Minimize Max Drawdown** directly targets the smallest peak-to-trough loss."
        )
        st.markdown(
            "**Inverse Volatility** weights assets inversely to their vol — simple and transparent. "
            "**Equal Risk Contribution** equalises each asset's risk contribution, accounting for correlations. "
            "**Hierarchical Risk Parity** clusters correlated assets and allocates top-down for robustness."
        )

    # ── Leverage-Aware Strategies ──
    with st.expander("Leverage-Aware Strategies"):
        st.markdown(
            "**Leverage-Optimal** maximises post-leverage, post-financing Sharpe ratio directly. "
            "The objective accounts for volatility drag (L(L−1)σ²/2) and financing costs, "
            "finding the unleveraged allocation that compounds best when leveraged. "
            "Tends to favour low-vol assets because vol drag scales quadratically with leverage."
        )
        st.markdown(
            "**Adaptive Lookback Blend** runs Max Sharpe optimisation over four trailing windows "
            "(63, 126, 252, 504 days) and equal-weight blends the results. "
            "Smooths allocations across time horizons, reducing sensitivity to lookback choice."
        )

    # ── Drawdown-Constrained ──
    with st.expander("Drawdown-Constrained Strategies"):
        st.markdown(
            f"**Max Sharpe (DD ≤ X%)** and **Max Calmar (DD ≤ X%)** add a drawdown penalty "
            f"to their unconstrained counterparts. Currently set to **{dd_constraint_pct}%** via the sidebar."
        )

    # ── Dynamic Strategies ──
    with st.expander("Dynamic Strategies"):
        st.markdown(
            "**Regime-Based Allocation** classifies each month into one of four macro regimes "
            "(high/low inflation crossed with high/low interest rates) using expanding-window medians. "
            "A separate portfolio is optimised per regime; weights switch at rebalance dates."
        )
        st.markdown(
            "**DD P-Value Momentum** ranks each asset's current drawdown against its own history. "
            "Assets with historically rare drawdowns (low p-value) receive allocation bumps. "
            "The bump schedule shape is optimised via grid search to maximise CAGR."
        )
        st.markdown(
            "**Max Sharpe (Time-Varying)** and **Leverage-Optimal (Time-Varying)** re-optimise "
            "their respective objectives annually using a trailing 5-year window of returns. "
            "This adapts allocations to changing market conditions without lookahead bias."
        )
        st.markdown(
            "**Ensemble Meta-Strategy** treats static strategies as sub-portfolios and "
            "allocates across them using trailing 12-month inverse-volatility weighting. "
            "Diversifies across *methodologies*, producing smoother performance by "
            "always partially allocated to whichever approach is currently working."
        )
        st.markdown(
            "**Yield Curve Signal Overlay** monitors the 12-month change in the Fed Funds rate. "
            "When rates rise above 200bp/yr (indicating tightening), tilts toward defensive assets "
            "(short-term treasuries, gold, cash). When stable or falling, runs the full base allocation."
        )

    # ── Benchmark ──
    with st.expander("Benchmark"):
        st.markdown(
            "**Dalio All Weather** — Ray Dalio's classic: 30% S&P 500, 40% LT Treasuries, "
            "15% ST Treasuries, 7.5% Gold, 7.5% Commodities."
        )

    # ── CFD & Monte Carlo ──
    with st.expander("CFD Analysis & Monte Carlo"):
        st.markdown(
            "The CFD tab models leveraged positions with margin requirements and financing costs. "
            "A cash reserve is sized to survive the worst historical drawdown at the given leverage. "
            "Financing is charged on the full notional value (rate × leverage), matching CMC Markets."
        )
        st.markdown(
            "The Monte Carlo projection runs 500 GBM paths calibrated to each portfolio's CAGR and "
            "volatility, showing P5–P95 outcomes over 10 years. Leveraged returns account for volatility "
            "drag and financing costs. The cash reserve earns 0% (CMC does not pay interest on idle cash)."
        )

    # ── Key Metrics ──
    with st.expander("Key Metrics"):
        _met_data = [
            ("CAGR", "Compound annual growth rate, calendar-day annualised"),
            ("Sharpe Ratio", f"(Arithmetic return − {risk_free_pct:.1f}% RFR) / volatility"),
            ("Calmar Ratio", "CAGR / |max drawdown|"),
            ("Max Drawdown", "Largest peak-to-trough decline over the backtest"),
            ("Volatility", "Annualised standard deviation of daily returns (252 trading days)"),
            ("Risk-Free Rate", f"Currently {risk_free_pct:.1f}%. Used for Sharpe ratio and as the CFD benchmark"),
            ("Rebalancing", "How often weights reset to targets. Weights drift between rebalances"),
        ]
        _met_df = pd.DataFrame(_met_data, columns=["Metric", "Definition"])
        st.dataframe(_met_df, hide_index=True, width="stretch")

    # ── FAQ ──
    with st.expander("Frequently Asked Questions"):
        st.markdown("""
**Why does Max Sharpe sometimes show a lower Sharpe ratio than Regime-Based or Leverage-Optimal in the comparison table?**

The optimiser maximises Sharpe on the *overlap period* — the window where all 26 assets have data (approximately 2010 onward).
But the strategy comparison table reports statistics over the *full backtest* (default start 1983).
On the 1983–2010 period, Max Sharpe weights weren't designed to be optimal, so the realised Sharpe can underperform strategies that adapt year-by-year.

Additionally, time-varying strategies (Regime-Based, DD Momentum) are evaluated on the same data they were fitted on — a form of in-sample evaluation.
Their superior backtest numbers partly reflect this look-ahead.
Without a held-out out-of-sample period, direct Sharpe comparisons between static and time-varying strategies should be treated with caution.

---

**Why is volatility so similar across most strategies?**

All strategies share the same 26-asset universe with a 0.5% minimum allocation per asset.
This structural floor creates a consistent diversification baseline — it's hard to get very different risk profiles within those constraints.
The optimiser operates in a tight feasible space, which tends to produce portfolios with similar annualised volatility.
You can widen the spread by lowering minimum weights in the Settings tab.

---

**Why are best-year returns very high?**

If Bitcoin is included (checkbox in sidebar), its extreme annual returns dominate the "best year" figure even at a 5–15% allocation.
For example, a 15% BTC allocation in a year where BTC returns +200% adds +30 percentage points to portfolio returns that year.
Tick "Exclude Bitcoin" in the sidebar to see more typical figures.

---

""")

# ──────────────────────────────────────────────
# SETTINGS TAB
# ──────────────────────────────────────────────
with tab_settings:
    # ── Data Source ──
    st.subheader("Data Source")
    _settings_path = st.text_input(
        "Excel file path",
        value=st.session_state.data_path,
        key="settings_data_path",
        help="Path to data_template.xlsx. Changes take effect on the next reload.",
    )
    if _settings_path != st.session_state.data_path:
        st.session_state.data_path = _settings_path
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
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

    st.markdown("---")
    st.subheader("Cache")
    st.caption("Clear the optimisation cache to force a full recompute. Use this after updating the data file or precomputed weights.")
    if st.button("Clear Optimisation Cache"):
        import shutil as _shutil
        if CACHE_DIR.exists():
            _shutil.rmtree(str(CACHE_DIR))
            CACHE_DIR.mkdir(exist_ok=True)
        st.cache_data.clear()
        for _k in ["base_results", "dd_results", "defaults_cache_key"]:
            st.session_state.pop(_k, None)
        st.success("Cache cleared — portfolios will recompute on next load.")
        st.rerun()

# ──────────────────────────────────────────────
# Version (bottom of page)
# ──────────────────────────────────────────────
st.caption(f"v{APP_VERSION}")
