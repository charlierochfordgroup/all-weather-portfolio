# All Weather Python

Multi-asset portfolio optimiser and Streamlit dashboard. Run with `streamlit run app.py`.
Precompute weights offline with `python precompute.py`.

## Architecture

| Module | Purpose |
|---|---|
| `data.py` | Load Excel prices, align dates, compute log returns. Exports `ASSETS` (26) and `GROUP_MAP`. |
| `stats.py` | Portfolio metrics: CAGR, Sharpe, vol, max DD, Calmar. Core: `calc_stats()`, `calc_stats_cached()`. |
| `optimizer.py` | 8 optimisation strategies + constraint enforcement (`clip_normalize`). |
| `cfd.py` | CFD leverage analysis: financing cost, dividend drag, margin, reserve sizing. |
| `precompute.py` | Offline parallel optimisation pipeline with incremental checkpointing. |
| `regime.py` | Inflation/rate regime classification and per-regime allocation. |
| `dd_momentum.py` | Drawdown p-value momentum: bump assets with unusually deep drawdowns. |
| `dd_budget.py` | Dynamic risk scaling: reduce exposure as drawdown consumes budget. |
| `yield_signal.py` | Fed Funds rate-rise overlay: tilt defensive when rates rising fast. |
| `ensemble.py` | Blend strategies by trailing inverse-vol weighting. |
| `optimize_bump.py` | Grid-search optimal DD momentum bump shape (offline calibration). |
| `app.py` | Streamlit dashboard (~2400 lines). |
| `test_suite.py` | ~88 pytest tests. Run with `pytest test_suite.py -v`. |

## Critical math conventions

**Log vs simple returns.** All stored returns are log returns. Portfolio math converts per-asset:
```
simple_rets = np.exp(log_rets) - 1.0
portfolio_return = simple_rets @ weights  # NOT log_rets @ weights
```
`calc_stats` does this conversion internally. Tests must match this methodology.

**Periodic rebalancing.** Weights reset to target at period boundaries (month/quarter/year).
Within a period, weights drift with realised returns. The vectorised implementation is in
`_periodic_rebal_returns_vectorized()`. Weights summing to <1.0 means the remainder is cash
(0% return) -- used by `dd_budget.py`.

**Leveraged CAGR** (`cfd.py`). Dividend drag is subtracted *before* leveraging:
```
adjusted_cagr = unleveraged_cagr - portfolio_dividend_drag(weights)
gross_cagr = (1 + adjusted_cagr)^L * exp(-L*(L-1)*vol^2/2) - 1
```
The vol-drag term `exp(-L*(L-1)*vol^2/2)` penalises leverage on volatile portfolios.
Tests comparing `gross_cagr` must pre-subtract `portfolio_dividend_drag(w)` from `stats.cagr`.

**Carry-adjusted risk parity** (`optimizer.py`). Softmax tilt on ERC weights:
```
carry_score = exp(clip(net_carry * carry_sensitivity, -5, 5))
```
`carry_sensitivity` (default 5.0) controls concentration. Saturates for spreads >1%.

## Asset universe (26 assets, 6 groups)

- **US Equities**: Nasdaq, S&P 500, Russell 2000
- **Intl Equities**: ASX200, Emerging Markets, China Equities, Japan Equities, UK Equities, EU Equities
- **Bonds**: Corporate Bonds, Long-Term Treasuries, Short-Term Treasuries, US TIPS, High Yield, EM Debt
- **Real Assets**: US REITs, Industrial Metals, Gold, Copper, Soft Commodities, Infrastructure
- **Alternatives**: Cash, Bitcoin
- **Currencies**: JPY, CHF, CNY

When writing tests, `len(ASSETS) == 26`. Feasibility check: `n_assets * max_w` must exceed 1.0
for the constraint set to be feasible (e.g. 26 assets at 3% max = 78% < 100% = infeasible).

## Default constraints (precompute.py)

- Min weight: 0.5% per asset
- Max weight: 10-40% per asset (varies, see `DEFAULT_MAX`)
- Group caps: US Eq 35%, Intl Eq 30%, Bonds 50%, Real Assets 30%, Alternatives 20%, Currencies 25%
- Leverage: 5x, Financing rate: 6.5%, Risk-free rate: 5%
- Dividend tax rate: 18% (`CFD_DIVIDEND_TAX_RATE`)

## Testing

```
pytest test_suite.py -v
```

Key test helpers in `test_suite.py`:
- `_make_returns(n_days, seed)` -- synthetic 26-asset daily log returns
- `_equal_weights()` -- uniform 1/26 allocation
- `_default_bounds()` -- [0%, 100%] per asset
- `_default_group_max()` -- 100% per group (unconstrained)

## Known gotchas

- **Windows multiprocessing**: `Pool(maxtasksperchild=1)` prevents worker accumulation hangs.
- **Locked Excel files**: `_safe_open()` in `data.py` copies to temp file when Excel is open.
  Temp files cleaned up via `atexit` handler. The `_temp_files` list tracks them.
- **DD momentum confidence**: Assets with <15 observed drawdown episodes get scaled-down bumps
  (`sqrt(n_episodes/15)`). Bitcoin, CNY naturally get weaker signals.
- **Regime classification**: Uses expanding-window medians (no lookahead bias). CPI threshold 3%,
  Fed Funds 12-month change threshold +/-1%.
