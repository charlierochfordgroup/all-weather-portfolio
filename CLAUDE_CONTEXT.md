# All Weather Portfolio Analyser — Claude Context

> Written for a future Claude instance opening this project for the first time.
> Last updated: 2026-03-23. 66 tests passing.

---

## Project Overview

This is a **multi-asset portfolio optimisation and backtesting tool** built with Streamlit. It takes historical price/return data for 17 assets across 5 asset classes, runs multiple optimisation strategies, and presents interactive comparisons via a browser dashboard.

The core question it answers: *"Given these assets, what allocation maximises risk-adjusted returns — and how does that change under leverage, drawdown constraints, or regime-switching?"*

### What It Does

1. **Optimises portfolios** across 9+ strategies (Max Sharpe, Min Volatility, Max Calmar, ERC, HRP, Inverse Volatility, Minimize Max Drawdown, plus DD-constrained variants)
2. **Backtests with extended history** — assets that didn't exist early (e.g., Bitcoin from ~2014) have their target weights redistributed pro-rata to available assets
3. **Dynamic strategies** — Regime-Based allocation (inflation/interest-rate regimes) and DD P-Value Momentum (buys historically rare drawdowns)
4. **CFD leverage analysis** — models leveraged returns with vol drag, financing costs, margin requirements, and cash reserves
5. **Monte Carlo projections** — 2000-path GBM forward simulation with percentile bands
6. **Includes Ray Dalio's All Weather** as a fixed benchmark (30% S&P, 40% LT Bonds, 15% ST Bonds, 7.5% Gold, 7.5% Commodities)

### Key Dependencies

```
numpy, pandas, scipy          — computation
streamlit                     — web dashboard
plotly                        — interactive charts
openpyxl                      — Excel I/O
```

No Bloomberg connection. Data comes from an Excel file (`data_template.xlsx`) with pre-computed log returns on the "Processing" sheet and raw prices on the "Data" sheet.

---

## Architecture

### File Map

```
app.py              — Streamlit UI (4 tabs: Compare, Dynamic Strategies, CFD Analysis, Settings)
stats.py            — PortfolioStats dataclass + calc_stats(), equity curves, drawdowns, annual returns
optimizer.py        — run_optimization() dispatcher → SLSQP, ERC, HRP, inverse-vol, DD minimisation
data.py             — ASSETS list, GROUP_MAP, Excel loading, log return computation, cash sanitization
cfd.py              — analyze_cfd() — leveraged CAGR with vol drag, margin/reserve sizing
regime.py           — load_regime_data(), classify_regimes(), optimize_per_regime(), build_regime_schedule()
dd_momentum.py      — detect_drawdown_episodes(), compute_pvalue(), compute_dd_adjustments()
precompute.py       — Offline script: runs all optimisations, saves to precomputed_weights.pkl
test_suite.py       — 66 pytest tests across all modules
data_template.xlsx  — 17-asset price/return data (Processing + Data sheets)
Inflation and IR.xlsx — Monthly CPI YoY + Daily Fed Funds Rate (for regime classification)
```

### Data Flow

```
data_template.xlsx
    ↓ load_data() [data.py]
pd.DataFrame of daily log returns (17 columns, ~16k rows from ~1960–2026)
    ↓
┌─────────────────────────────────┐
│  Overlap period (all assets     │ → optimizer.py → optimal weights
│  have data, ~2010–2026)         │
└─────────────────────────────────┘
    ↓ weights
┌─────────────────────────────────┐
│  Full backtest period           │ → stats.py → PortfolioStats, equity curves
│  (user-selected start–end)      │    (with pro-rata redistribution for missing assets)
└─────────────────────────────────┘
    ↓ stats + weights
app.py renders tables, charts, CFD analysis, Monte Carlo
```

### Key Design Patterns

**Separation of optimisation and evaluation windows.** The optimizer always runs on the *overlap period* (where all 17 assets have data, ~2010–2026). Stats and equity curves are computed on the *full backtest period* (potentially back to 1960), with `_effective_weights()` handling pro-rata redistribution for assets that don't exist yet.

**Pre-computation for instant loading.** `precompute.py` runs all optimisations offline and saves weights to `precomputed_weights.pkl`. The Streamlit app checks for this file first; if constraints match defaults, it loads weights instantly and only computes stats (fast). If the user changes constraints, it falls back to full optimisation with disk caching (`.cache/` directory).

**Time-varying weights via `weights_schedule`.** Dynamic strategies (regime, DD momentum) produce a `dict[pd.Timestamp, np.ndarray]` mapping rebalance dates to weight vectors. `_periodic_rebal_returns()` in `stats.py` accepts this schedule and switches weights at rebalance boundaries while allowing drift within each period.

**Soft drawdown constraint.** DD-constrained strategies (Max Sharpe DD≤X%, Max Calmar DD≤X%) add a penalty term `100 * max(|DD| - limit, 0)` to the objective rather than using a hard constraint. This lets SLSQP find solutions even when the exact constraint is barely feasible. The DD penalty is evaluated on the *full backtest period* (via `dd_returns` / `dd_asset_starts` parameters), not just the optimisation window.

**Two-phase drawdown minimisation.** `_optimize_drawdown()` in `optimizer.py` first minimises portfolio volatility (smooth, fast) as a proxy, then refines the top candidates on the true max-drawdown objective (non-smooth, slow).

### Stats Calculation Details (`stats.py`)

- **CAGR:** Annualised using calendar days: `idx[-1] ^ (365 / total_calendar_days) - 1`
- **Volatility:** `sqrt(var(simple_returns, ddof=1) * 252)` — computed from simple returns (not log) for consistency with Sharpe
- **Sharpe:** `(arithmetic_mean_annual - Rf) / volatility` — both numerator and denominator on simple return basis
- **Max Drawdown:** Peak-to-trough on cumulative index, skipping all-zero days
- **Calmar:** `CAGR / |max_drawdown|`
- **Rebalancing:** Daily = continuous (log return summation). Monthly/quarterly/etc = weight drift with simple returns, reset at period boundaries

### Regime Strategy (`regime.py`)

Uses expanding-window medians (not full-history) to avoid look-ahead bias. At each month `t`, `median(CPI[:t])` and `median(FedFunds[:t])` determine the threshold. Requires 24 months of warm-up data before classification begins. Earlier months are forward-filled from the first valid classification.

### DD Momentum Strategy (`dd_momentum.py`)

At each annual checkpoint, computes each asset's current drawdown relative to its historical drawdown episodes. The p-value (fraction of past episodes that were worse) determines the bump factor. The `bump_max` parameter (configurable via sidebar, default 50%) controls the maximum allocation increase for the most deeply drawn-down asset.

### CFD Leverage (`cfd.py`)

Leveraged CAGR uses the continuous-time formula: `(1+CAGR)^L * exp(-L*(L-1)*σ²/2) - 1`. The `exp` term captures quadratic vol drag — at 5x leverage with 15% vol, this reduces CAGR from a naive 5×8%=40% to ~17%. Cash reserves are sized so that `free_margin + reserve ≥ max_DD × leverage × deployed`.

---

## Current State

### Working

- All 9 base optimisation strategies produce valid, sum-to-1 weights
- Extended backtests with staggered asset availability (pro-rata redistribution)
- DD-constrained optimisation with warm-starting across DD levels
- Regime classification with expanding-window medians (no look-ahead)
- DD P-Value Momentum with configurable bump factors
- CFD leverage analysis with vol drag, financing, margin, and reserve calculations
- 10-year Monte Carlo projection (2000 paths, resample button)
- Disk caching and precomputation pipeline
- 66 tests passing across all modules
- Streamlit Cloud deployment ready (`.streamlit/config.toml`, `requirements.txt`, `.gitignore`)

### Known Limitations / Not Bugs But Worth Knowing

- **Cash return sanitization** clips daily returns to ±0.5% when raw vol exceeds 1%. This is a data-quality heuristic, not a financial model. A `UserWarning` is emitted when capping is applied.
- **ERC convergence** uses a fixed multiplicative learning rate (`lr=0.0005`). For highly correlated or ill-conditioned covariance matrices, it may not converge in 2000 iterations. A `UserWarning` is emitted if this happens.
- **`clip_normalize` feasibility** — if user-set constraints are contradictory (e.g., all max weights sum to <100%), a `UserWarning` is emitted and results are approximate.
- **Monte Carlo** uses a seeded RNG for reproducibility. The "Resample" button generates a new random seed. With 2000 paths, percentile estimates are stable but not perfectly smooth.
- **`weights_schedule` and `asset_starts` are mutually exclusive** in `calc_stats()` — line 215: `if asset_starts is not None and weights_schedule is None`. Dynamic strategies that use `weights_schedule` don't also get pro-rata redistribution. This is intentional (the schedule already encodes the desired weights) but worth noting.

### Edge Cases Verified by Tests

- Single-day returns: doesn't crash, returns zero stats
- All-zero returns: CAGR=0, vol=0, DD=0
- 100% allocation to one asset: matches standalone asset stats exactly
- Infeasible constraints: warns, returns finite (approximate) weights
- 1x leverage CFD: matches unleveraged stats exactly
- 10x leverage with 30% vol: correctly produces negative gross CAGR
- Annual returns compound to match total return within 1%

---

## Design Decisions

### Why log returns internally, simple returns for Sharpe?

Log returns are additive across time (convenient for cumulation: `exp(cumsum(log_returns))`) and naturally handle compounding. But the Sharpe ratio requires arithmetic (simple) returns in both numerator and denominator to be on the same basis. After an audit, the code was updated to convert log→simple for Sharpe/vol calculation: `port_simple = exp(port_log) - 1`.

### Why separate optimisation and backtest windows?

The optimizer needs all assets to have data simultaneously (otherwise covariance matrices are undefined for missing assets). But the user wants to see how the portfolio would have performed over a longer history. Solution: optimise on the overlap period (~2010–2026), then evaluate/backtest on the full range with pro-rata weight redistribution.

### Why soft DD constraint (penalty) instead of hard constraint?

Max drawdown is a non-smooth function — SLSQP can't compute gradients for it reliably as a hard constraint. The penalty approach (`100 * max(|DD| - limit, 0)`) added to the objective is smooth enough for SLSQP to handle and produces solutions that respect the constraint in practice.

### Why "Inverse Volatility" instead of "Risk Parity"?

True risk parity (equal risk contribution) accounts for correlations between assets. Simple `w = 1/vol` only accounts for individual volatilities. The code has both: `inverse_volatility()` (the simple one) and `equal_risk_contribution()` (the proper ERC). The label was corrected from "Risk Parity" to "Inverse Volatility" during an audit to avoid confusion.

### Why expanding-window median for regime classification?

The original implementation used full-history median, which is look-ahead bias: future data influenced past regime labels. The fix uses `median(data[:t])` at each month `t`, with a 24-month warm-up period. This means real-time implementation would have produced the same classifications.

---

## Relationship to Excel-Based All Weather Portfolio Analyser

This Python project is a **precursor and replacement** for an Excel-based version of the same tool. The Excel version uses:

- A **Processing sheet** for pre-computed log returns (same concept as `load_returns_from_processing()` in `data.py`)
- **VBA-stored Portfolio B snapshots** for comparing allocations
- **Named ranges** for date inputs and constraint parameters
- A **SolverHelper isolation pattern** that runs Excel Solver in a controlled way
- A **multi-method optimiser** supporting mean-variance, ERC, HRP, and Risk Parity

### Known bugs in the Excel version

- `ReadConstraints` has **wrong column references** — reads from incorrect columns when loading constraint parameters
- **Max Sharpe and Max Calmar Solver objectives produce identical results** due to a bug in the objective cell references (both point to the same cell)

The Python version fixes both of these issues and adds capabilities the Excel version lacks (dynamic strategies, CFD analysis, Monte Carlo, extended backtests).

---

## How to Pick Up This Project

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (should see 66 passed)
python -m pytest test_suite.py -v

# Pre-compute weights (takes ~10 minutes, only needed if data or defaults change)
python precompute.py

# Launch the app
streamlit run app.py
```

### Required Data Files

- `data_template.xlsx` — must be in the project root. Contains price data on "Data" sheet and pre-computed log returns on "Processing" sheet.
- `Inflation and IR.xlsx` — optional but needed for regime strategy. Monthly CPI in columns A-B, daily Fed Funds in columns C-D.

### Streamlit Cloud Deployment

The app is configured for Streamlit Community Cloud hosting. The repo is at `charlierochfordgroup/all-weather-portfolio`. Key deployment files: `requirements.txt`, `.streamlit/config.toml`, `.gitignore`. Data files (`data_template.xlsx`, `Inflation and IR.xlsx`, `precomputed_weights.pkl`) must be committed to the repo.

### Suggested Starting Points for Development

1. **To add a new optimisation strategy:** Add the function in `optimizer.py`, register it in `run_optimization()`, add it to `_BASE_TARGETS` in both `app.py` and `precompute.py`, run `precompute.py`.

2. **To add a new asset:** Add to `ASSETS` list and `GROUP_MAP` in `data.py`, add its price data to `data_template.xlsx`, update constraint defaults in `app.py` and `precompute.py`.

3. **To modify the dynamic strategies:** `regime.py` and `dd_momentum.py` are self-contained. The `weights_schedule` mechanism in `stats.py` is generic — any strategy that produces `dict[Timestamp, ndarray]` can plug in.

4. **To change the Streamlit UI:** All UI code is in `app.py`. The four tabs are independent sections. Charts use Plotly. Session state manages constraints and cached results.

5. **To debug calculation issues:** Run `python -m pytest test_suite.py -v -k "test_name"` for targeted tests. The `TestAuditMetrics` class contains manual cross-reference calculations that verify CAGR, vol, Sharpe, and max DD against hand-computed values.

### Key Invariants to Preserve

- Weights always sum to 1.0 (enforced by `clip_normalize()`)
- `calc_stats()` must produce identical results whether called with `rebalance="daily"` or via the periodic rebalancing path for a single-day period
- 1x leverage in `analyze_cfd()` must exactly reproduce unleveraged stats
- `_effective_weights()` must preserve relative proportions when redistributing
- Expanding-window regime classification must produce identical results for month `t` regardless of how much future data exists
