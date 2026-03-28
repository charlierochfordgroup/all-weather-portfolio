"""Pre-compute optimised portfolio weights for default settings.

Run this script locally whenever the data or default constraints change.
It saves weights to precomputed_weights.pkl which ships with the repo
so the Streamlit app loads instantly without running the optimizer.

Usage:  python precompute.py

Optimisations applied:
  1. Incremental checkpointing — progress is saved after each strategy so a
     crash or interruption can be resumed from where it left off.
  2. Warm starts — DD-constrained sweep passes the previous level's solution
     as the starting point, reducing optimizer iterations.
  3. Parallel base strategies — the 6 base strategies are computed in parallel
     using a multiprocessing Pool.
  4. Hash-based skip — if the input data and parameters haven't changed since
     the last run, a completed variant is skipped entirely.
"""

import hashlib
import pickle
import sys
import io
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

from data import ASSETS, load_data
from stats import calc_stats, compute_asset_starts
from optimizer import run_optimization
from regime import load_regime_data, classify_regimes, optimize_per_regime
from dd_momentum import (
    compute_dd_adjustments, compute_dd_adjustments_scheduled,
    build_dd_momentum_schedule, load_optimal_bump_schedule,
)

# ── Default settings (must match app.py defaults exactly) ──
RISK_FREE_RATE = 0.05
LEVERAGE = 5.0
FINANCING_RATE = 0.065

DEFAULT_MIN = {a: 0.5 for a in ASSETS}
DEFAULT_MAX = {
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
DEFAULT_GROUP_MAX = {
    "US Equities": 35.0, "Intl Equities": 30.0, "Bonds": 50.0,
    "Real Assets": 30.0, "Alternatives": 20.0, "Currencies": 25.0,
}

BASE_TARGETS = [
    "Max Sharpe Ratio", "Max Calmar Ratio",
    "Leverage-Optimal",
    "Carry-Adjusted Risk Parity",
    "Max Sharpe (Unconstrained)",
    "Leverage-Optimal (Unconstrained)",
]
DD_TARGETS = ["Max Sharpe (DD \u2264 X%)", "Max Calmar (DD \u2264 X%)"]
DD_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = _DIR / "precomputed_weights.pkl"
CHECKPOINT_FILE = _DIR / "_precompute_checkpoint.pkl"


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _params_hash(returns_df):
    """Hash input data + key parameters. If this matches the checkpoint, skip."""
    h = hashlib.md5()
    h.update(pd.util.hash_pandas_object(returns_df, index=True).values.tobytes())
    for val in [RISK_FREE_RATE, LEVERAGE, FINANCING_RATE]:
        h.update(str(val).encode())
    h.update(str(sorted(BASE_TARGETS + DD_TARGETS)).encode())
    h.update(str(sorted(DD_LEVELS)).encode())
    return h.hexdigest()


def _load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                ckpt = pickle.load(f)
            print(f"  Loaded checkpoint ({CHECKPOINT_FILE.name})")
            return ckpt
        except Exception as e:
            print(f"  Warning: checkpoint unreadable ({e}), starting fresh")
    return {}


def _save_checkpoint(ckpt):
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump(ckpt, f)


def _variant_complete(ckpt, btc_label):
    """Return True if this variant has all required sections in the checkpoint."""
    v = ckpt.get(btc_label, {})
    required = {"base_weights", "dd_weights", "dd_momentum_adjustments",
                "dd_momentum_base", "regime_weights", "regime_weights_by_dd", "regime_series"}
    return required.issubset(v.keys())


# ── Workers for parallel strategies ──────────────────────────────────────────

def _run_base_strategy(args):
    """Top-level worker function (required for multiprocessing on Windows)."""
    tgt, returns_arr, returns_cols, returns_idx, min_w, max_w, group_max, rf = args
    returns = pd.DataFrame(returns_arr, columns=returns_cols, index=returns_idx)
    w = run_optimization(
        returns, tgt, min_w, max_w, group_max, rf,
        rebalance="daily", leverage=LEVERAGE, financing_rate=FINANCING_RATE,
    )
    return tgt, w


def _run_regime_dd(args):
    """Worker for one regime DD level — runs independently."""
    (dd_pct,
     returns_arr, returns_cols, returns_idx,
     bt_arr, bt_cols, bt_idx,
     regime_series_arr, regime_series_idx,
     min_w, max_w, group_max, rf, asset_starts_dict) = args
    returns = pd.DataFrame(returns_arr, columns=returns_cols, index=returns_idx)
    bt_returns = pd.DataFrame(bt_arr, columns=bt_cols, index=bt_idx)
    regime_series = pd.Series(regime_series_arr, index=pd.DatetimeIndex(regime_series_idx))
    asset_starts = {k: pd.Timestamp(v) for k, v in asset_starts_dict.items()}
    dd_val = dd_pct / 100.0
    rw = optimize_per_regime(
        returns, regime_series, "Max Sharpe Ratio",
        min_w, max_w, group_max, rf, rebalance="daily",
        dd_constraint=dd_val,
        dd_returns=bt_returns, dd_asset_starts=asset_starts,
    )
    return dd_pct, rw


def _run_dd_strategy(args):
    """Worker for one (dd_pct, target) pair — runs independently, no warm start."""
    (tgt, dd_pct,
     returns_arr, returns_cols, returns_idx,
     bt_arr, bt_cols, bt_idx,
     min_w, max_w, group_max, rf, asset_starts_dict) = args
    returns = pd.DataFrame(returns_arr, columns=returns_cols, index=returns_idx)
    bt_returns = pd.DataFrame(bt_arr, columns=bt_cols, index=bt_idx)
    asset_starts = {k: pd.Timestamp(v) for k, v in asset_starts_dict.items()}
    dd_val = dd_pct / 100.0
    w = run_optimization(
        returns, tgt, min_w, max_w, group_max, rf,
        rebalance="daily", dd_constraint=dd_val,
        dd_returns=bt_returns, dd_asset_starts=asset_starts,
    )
    return dd_pct, tgt, w


# ── Per-variant computation ───────────────────────────────────────────────────

def compute_variant(btc_label, exclude_btc, returns_full, asset_starts, ckpt, data_hash):
    """Compute all weights for one BTC variant, checkpointing after each section."""

    # Skip if already complete and data unchanged
    if ckpt.get("data_hash") == data_hash and _variant_complete(ckpt, btc_label):
        print(f"  [{btc_label}] Unchanged — skipping (hash match)")
        return ckpt[btc_label]

    v = ckpt.setdefault(btc_label, {})

    overlap_start = max(asset_starts.values())
    opt_returns = returns_full.loc[overlap_start:]

    group_max = {g: val / 100.0 for g, val in DEFAULT_GROUP_MAX.items()}
    btc_idx = ASSETS.index("Bitcoin")

    min_w = np.array([DEFAULT_MIN.get(a, 0.5) / 100.0 for a in ASSETS])
    max_w = np.array([DEFAULT_MAX.get(a, 30.0) / 100.0 for a in ASSETS])
    if exclude_btc:
        min_w[btc_idx] = 0.0
        max_w[btc_idx] = 0.0

    # ── Section 1: Base strategies (parallel) ────────────────────────────────
    if "base_weights" not in v:
        n_base = len(BASE_TARGETS)
        print(f"  [{btc_label}] Optimising {n_base} base strategies in parallel...")

        # Serialise returns as numpy for pickling efficiency
        arr = opt_returns.values
        cols = list(opt_returns.columns)
        idx = opt_returns.index

        args_list = [
            (tgt, arr, cols, idx, min_w, max_w, group_max, RISK_FREE_RATE)
            for tgt in BASE_TARGETS
        ]

        base_w = {}
        with Pool(processes=min(n_base, 4)) as pool:
            for i, (tgt, w) in enumerate(pool.imap_unordered(_run_base_strategy, args_list), 1):
                base_w[tgt] = w
                print(f"    ✓ [{i}/{n_base}] {tgt}")

        v["base_weights"] = base_w
        _save_checkpoint(ckpt)
        print(f"  [{btc_label}] Base strategies saved to checkpoint.")
    else:
        base_w = v["base_weights"]
        print(f"  [{btc_label}] Base strategies already in checkpoint — skipping.")

    # ── Section 2: DD-constrained strategies (parallel, per-result checkpoint) ─
    dd_w = v.setdefault("dd_weights", {})
    remaining_pairs = [
        (tgt, dd_pct)
        for dd_pct in sorted(DD_LEVELS)
        for tgt in DD_TARGETS
        if dd_pct not in dd_w or tgt not in dd_w.get(dd_pct, {})
    ]
    if remaining_pairs:
        total_dd = len(DD_LEVELS) * len(DD_TARGETS)
        done = total_dd - len(remaining_pairs)
        print(f"  [{btc_label}] Optimising {len(remaining_pairs)} DD-constrained portfolios in parallel...")
        if done:
            print(f"    Resuming from checkpoint ({done}/{total_dd} already done)...")

        arr = opt_returns.values
        cols = list(opt_returns.columns)
        idx = opt_returns.index
        bt_arr = returns_full.values
        bt_cols = list(returns_full.columns)
        bt_idx = returns_full.index
        asset_starts_dict = {k: str(v) for k, v in asset_starts.items()}

        args_list = [
            (tgt, dd_pct, arr, cols, idx, bt_arr, bt_cols, bt_idx,
             min_w, max_w, group_max, RISK_FREE_RATE, asset_starts_dict)
            for tgt, dd_pct in remaining_pairs
        ]

        n_workers = min(len(remaining_pairs), 6)
        count = done
        with Pool(processes=n_workers) as pool:
            for dd_pct, tgt, w in pool.imap_unordered(_run_dd_strategy, args_list):
                dd_w.setdefault(dd_pct, {})[tgt] = w
                count += 1
                print(f"    ✓ [{count}/{total_dd}] {tgt.replace('X%', f'{dd_pct}%')}")
                _save_checkpoint(ckpt)

        print(f"  [{btc_label}] DD-constrained weights complete.")
    else:
        print(f"  [{btc_label}] DD-constrained weights already in checkpoint — skipping.")

    # ── Section 3: DD Momentum ───────────────────────────────────────────────
    if "dd_momentum_adjustments" not in v:
        print(f"  [{btc_label}] Computing DD Momentum adjustments...")
        ms_weights = base_w["Max Sharpe Ratio"]
        years = sorted(set(returns_full.index.year))
        checkpoints = []
        for y in years:
            yr_dates = returns_full.index[returns_full.index.year == y]
            if len(yr_dates) > 0:
                checkpoints.append(yr_dates[0])
        optimal_sched = load_optimal_bump_schedule()
        if optimal_sched is not None:
            print("    Using optimised per-rank bump schedule")
            dd_adj = compute_dd_adjustments_scheduled(returns_full, checkpoints, optimal_sched)
        else:
            print("    Using default parametric bump schedule (bump_max=50%)")
            dd_adj = compute_dd_adjustments(returns_full, checkpoints)
        print(f"    {len(dd_adj)} annual checkpoints computed")

        v["dd_momentum_adjustments"] = dd_adj
        v["dd_momentum_base"] = ms_weights
        _save_checkpoint(ckpt)
        print(f"  [{btc_label}] DD Momentum saved to checkpoint.")
    else:
        print(f"  [{btc_label}] DD Momentum already in checkpoint — skipping.")

    # ── Section 4: Regime portfolios (incremental checkpoint per DD level) ───
    macro_path = _DIR / "Inflation and IR.xlsx"
    macro_data = load_regime_data(macro_path)
    if macro_data is None:
        if "regime_weights" not in v:
            print(f"  [{btc_label}] No macro data found — skipping regime pre-computation")
            v["regime_weights"] = None
            v["regime_weights_by_dd"] = {}
            v["regime_series"] = None
            _save_checkpoint(ckpt)
    else:
        # Step 4a: unconstrained regime weights (one-off, saved immediately)
        if "regime_weights" not in v:
            print(f"  [{btc_label}] Computing regime portfolios (unconstrained)...")
            regime_series = classify_regimes(macro_data)
            regime_weights = optimize_per_regime(
                opt_returns, regime_series, "Max Sharpe Ratio",
                min_w, max_w, group_max, RISK_FREE_RATE, rebalance="daily",
            )
            for label_id in regime_weights:
                print(f"    Regime {label_id}: done")
            v["regime_weights"] = regime_weights
            v["regime_series"] = regime_series
            _save_checkpoint(ckpt)
        else:
            print(f"  [{btc_label}] Regime unconstrained weights already in checkpoint — skipping.")

        # Step 4b: DD-constrained regime weights (parallel, per-result checkpoint)
        regime_weights_by_dd = v.setdefault("regime_weights_by_dd", {})
        remaining_regime_levels = [l for l in DD_LEVELS if l not in regime_weights_by_dd]
        if remaining_regime_levels:
            regime_series = v["regime_series"]
            done_r = len(DD_LEVELS) - len(remaining_regime_levels)
            print(f"  [{btc_label}] Computing {len(remaining_regime_levels)} regime DD levels in parallel "
                  f"({done_r}/{len(DD_LEVELS)} already done)...")

            arr = opt_returns.values
            cols = list(opt_returns.columns)
            idx = opt_returns.index
            bt_arr = returns_full.values
            bt_cols = list(returns_full.columns)
            bt_idx = returns_full.index
            asset_starts_dict = {k: str(v) for k, v in asset_starts.items()}

            args_list = [
                (dd_pct, arr, cols, idx, bt_arr, bt_cols, bt_idx,
                 regime_series.values, regime_series.index,
                 min_w, max_w, group_max, RISK_FREE_RATE, asset_starts_dict)
                for dd_pct in remaining_regime_levels
            ]

            n_workers = min(len(remaining_regime_levels), 6)
            count = done_r
            with Pool(processes=n_workers) as pool:
                for dd_pct, rw in pool.imap_unordered(_run_regime_dd, args_list):
                    regime_weights_by_dd[dd_pct] = rw
                    count += 1
                    print(f"    ✓ [{count}/{len(DD_LEVELS)}] Regime DD≤{dd_pct}%")
                    _save_checkpoint(ckpt)

            print(f"  [{btc_label}] Regime DD-constrained weights complete.")
        else:
            print(f"  [{btc_label}] Regime DD-constrained weights already in checkpoint — skipping.")

    return v


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data_path = str(_DIR / "data_template.xlsx")
    all_returns = load_data(data_path, use_processing=True)

    available = [a for a in ASSETS if a in all_returns.columns]
    returns_full = all_returns[available]
    asset_starts = compute_asset_starts(returns_full)
    overlap_start = max(asset_starts.values())

    print(f"Overlap period: {overlap_start.date()} to {returns_full.index[-1].date()}")

    data_hash = _params_hash(returns_full)
    print(f"Data hash: {data_hash[:12]}...")

    ckpt = _load_checkpoint()
    if ckpt.get("data_hash") != data_hash:
        if ckpt:
            print("  Data or parameters changed — clearing stale checkpoint.")
        ckpt = {"data_hash": data_hash}  # wipe old variant data so nothing is skipped

    all_variants = {}
    for btc_label, exclude_btc in [("excl_btc", True), ("incl_btc", False)]:
        print(f"\n{'='*60}")
        print(f"  Variant: {btc_label}")
        print(f"{'='*60}")
        all_variants[btc_label] = compute_variant(
            btc_label, exclude_btc, returns_full, asset_starts, ckpt, data_hash,
        )

    # ── Assemble and save final output ───────────────────────────────────────
    output = {
        **all_variants["excl_btc"],
        "btc_variants": all_variants,
        "overlap_start": overlap_start,
        "data_end": returns_full.index[-1],
        "default_min": DEFAULT_MIN,
        "default_max": DEFAULT_MAX,
        "default_group_max": DEFAULT_GROUP_MAX,
        "risk_free_rate": RISK_FREE_RATE,
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved to {OUTPUT_FILE.name} ({OUTPUT_FILE.stat().st_size / 1024:.1f} KB)")

    # Clean up checkpoint on successful completion
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint file removed.")

    print("Done!")


if __name__ == "__main__":
    main()
