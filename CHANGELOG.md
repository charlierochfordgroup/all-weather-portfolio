# Change Log

## 2026-03-24 – Performance, DD constraint propagation, current drawdown, CMC financing reference

### 1. Performance Optimisation

- **Vectorised `detect_drawdown_episodes()`** (`dd_momentum.py`) – replaced Python for-loop with NumPy `np.diff` on boolean mask for episode boundary detection and `np.argmin` for trough finding. Eliminates per-element iteration over ~4,000 daily values per asset per checkpoint.

- **Vectorised `classify_regimes()`** (`regime.py`) – replaced Python for-loop with `pd.Series.expanding().median()` (C-implemented) and vectorised boolean conditions. The original loop called `np.median()` on growing slices at each timestep; the new version is a single-pass expanding window.

- **Vectorised `longest_dd` computation** (`stats.py`) – replaced Python for-loop tracking consecutive drawdown days with `np.diff` on a boolean mask to find run lengths in one pass.

- **Optimised schedule lookup in `_periodic_rebal_returns()`** (`stats.py`) – replaced reverse linear scan over schedule dates with `np.searchsorted` on pre-converted int64 timestamps. O(log n) instead of O(n) per rebalance boundary.

- **Cached DD Momentum adjustments** (`app.py`) – on default constraints, loads pre-computed adjustments from `precomputed_weights.pkl` instead of recomputing on every page load. On non-default constraints, caches in `st.session_state` keyed on configuration hash. Avoids the most expensive per-load computation.

- **Added `current_drawdown` field to `PortfolioStats`** (`stats.py`) – computed during `calc_stats()` from the existing drawdown series, avoiding a separate equity curve recomputation for the comparison table.

### 2. Max Drawdown Constraint – Propagated to DD Momentum

- **`build_dd_momentum_schedule()`** (`dd_momentum.py`) now accepts `dd_constraint`, `returns`, `risk_free_rate`, `rebalance`, and `asset_starts` parameters. After computing adjusted weights at each annual checkpoint, if the resulting portfolio's max drawdown exceeds the constraint, the adjustment vector is scaled back via binary search (10 iterations) until the constraint is satisfied.

- **Regime portfolio** already respected the DD constraint (confirmed – no changes needed).

- **Caller updated** (`app.py`) to pass `dd_constraint_val` and related parameters to `build_dd_momentum_schedule()`.

### 3. Current Drawdown for Each Allocation

- Added **"Current DD"** column to the strategy comparison table in the Compare tab (`app.py`). Shows the percentage decline from peak cumulative value to latest value for each portfolio allocation.

- Formatted as percentage with one decimal place.

- **Colour-coded** via `pandas.Styler.applymap()`:
  - Green – at peak (0%)
  - Amber – between 0% and –10%
  - Red – worse than –10%

### 4. CMC Markets Financing Rate Reference

- Added **"CMC Markets Financing Rate Reference"** expander in the CFD Analysis tab (`app.py`), positioned between the per-asset allocation table and the Monte Carlo projection.

- **Rate table** maps each of the 17 portfolio assets to their CMC product type, long rate, short rate, and notes (interbank benchmarks).

- **Additional notes** cover daily charging times, weekend triple charges, borrowing fees for shorts, commission rates (AU and US), and rate disclaimers.

- **Daily financing cost calculator** with inputs for notional position value and annual rate (default 7.0%), outputting daily and monthly costs in AUD.
