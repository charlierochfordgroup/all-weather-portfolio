"""Portfolio optimization strategies."""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from data import ASSETS, GROUP_MAP, GROUP_NAMES
from stats import calc_stats, calc_stats_cached, StatsCache, TD
from cfd import portfolio_dividend_drag


def _group_indices() -> dict[str, list[int]]:
    """Map group name -> list of asset indices."""
    groups: dict[str, list[int]] = {g: [] for g in GROUP_NAMES}
    for i, a in enumerate(ASSETS):
        groups[GROUP_MAP[a]].append(i)
    return groups


def clip_normalize(
    w: np.ndarray,
    min_w: np.ndarray,
    max_w: np.ndarray,
    group_max: dict[str, float],
    max_iter: int = 20,
) -> np.ndarray:
    """Project weights onto the feasible set (box + group + sum-to-1 constraints)."""
    # Feasibility check: sum of max weights must be >= 1 and sum of min weights <= 1
    if max_w.sum() < 1.0 - 1e-6:
        warnings.warn(
            f"Infeasible constraints: sum of max weights ({max_w.sum():.2f}) < 1.0. "
            f"Cannot find a valid allocation. Results will be approximate.",
            stacklevel=2,
        )
    if min_w.sum() > 1.0 + 1e-6:
        warnings.warn(
            f"Infeasible constraints: sum of min weights ({min_w.sum():.2f}) > 1.0. "
            f"Cannot find a valid allocation. Results will be approximate.",
            stacklevel=2,
        )

    w = w.copy()
    n = len(w)
    groups = _group_indices()

    converged = False
    for _ in range(max_iter):
        w = np.clip(w, min_w, max_w)

        # Enforce group caps
        for gname, indices in groups.items():
            gs = w[indices].sum()
            cap = group_max.get(gname, 1.0)
            if gs > cap and gs > 0:
                w[indices] *= cap / gs

        total = w.sum()
        if abs(total - 1.0) < 1e-4:
            if np.all(w >= min_w - 5e-4):
                converged = True
                break

        if total > 1.0 + 1e-4:
            excess = total - 1.0
            flex = w - min_w
            flex = np.maximum(flex, 0)
            fs = flex.sum()
            if fs > 1e-5:
                ratio = min(excess / fs, 1.0)
                w -= flex * ratio
            else:
                w /= total
        elif total < 1.0 - 1e-4:
            deficit = 1.0 - total
            room = max_w - w
            room = np.maximum(room, 0)
            rs = room.sum()
            if rs > 1e-5:
                ratio = min(deficit / rs, 1.0)
                w += room * ratio
            else:
                w /= total

    if not converged:
        warnings.warn(
            f"clip_normalize did not converge after {max_iter} iterations "
            f"(sum={w.sum():.6f}). Constraints may be only approximately satisfied.",
            stacklevel=2,
        )

    return w


def _objective(
    w: np.ndarray,
    returns: pd.DataFrame,
    target: str,
    risk_free_rate: float,
    rebalance: str = "daily",
    dd_constraint: float | None = None,
    dd_returns: pd.DataFrame | None = None,
    dd_asset_starts: dict | None = None,
    leverage: float = 1.0,
    financing_rate: float = 0.065,
) -> float:
    """Objective function for optimization (negated for minimization).

    When dd_constraint is set, a heavy penalty is added if drawdown exceeds the limit.
    dd_returns / dd_asset_starts: if provided, evaluate the DD penalty on this
    (potentially longer) backtest period instead of the optimisation returns.
    leverage / financing_rate: used by the Leverage-Optimal target.
    """
    s = calc_stats(returns, w, risk_free_rate, rebalance=rebalance)

    # DD penalty: large cost if drawdown exceeds the constraint
    dd_penalty = 0.0
    if dd_constraint is not None:
        # Evaluate DD on the backtest period if provided, else on opt period
        if dd_returns is not None:
            s_dd = calc_stats(dd_returns, w, risk_free_rate, rebalance=rebalance,
                              asset_starts=dd_asset_starts)
        else:
            s_dd = s
        dd_excess = abs(s_dd.max_drawdown) - dd_constraint
        if dd_excess > 0:
            dd_penalty = 100.0 * dd_excess  # heavy penalty

    return _score(s, target, risk_free_rate, dd_penalty, leverage, financing_rate,
                  dd_returns, dd_asset_starts, rebalance, w)


def _objective_cached(
    w: np.ndarray,
    opt_cache: StatsCache,
    dd_cache: StatsCache | None,
    target: str,
    risk_free_rate: float,
    dd_constraint: float | None = None,
    leverage: float = 1.0,
    financing_rate: float = 0.065,
) -> float:
    """Fast objective using pre-computed StatsCache (avoids redundant work)."""
    s = calc_stats_cached(opt_cache, w, risk_free_rate)

    dd_penalty = 0.0
    if dd_constraint is not None:
        s_dd = calc_stats_cached(dd_cache, w, risk_free_rate) if dd_cache is not None else s
        dd_excess = abs(s_dd.max_drawdown) - dd_constraint
        if dd_excess > 0:
            dd_penalty = 100.0 * dd_excess

    return _score(s, target, risk_free_rate, dd_penalty, leverage, financing_rate,
                  w=w)


def _score(s, target, risk_free_rate, dd_penalty,
           leverage=1.0, financing_rate=0.065,
           dd_returns=None, dd_asset_starts=None, rebalance="daily", w=None):
    """Compute objective score from PortfolioStats — shared by both objective paths."""
    if target in ("Max Sharpe Ratio", "Max Sharpe (DD ≤ X%)"):
        return -s.sharpe + dd_penalty
    elif target == "Min Volatility":
        return s.volatility + dd_penalty
    elif target in ("Max Calmar Ratio", "Max Calmar (DD ≤ X%)"):
        return -s.calmar + dd_penalty
    elif target == "Minimize Max Drawdown":
        if dd_returns is not None and w is not None:
            s_dd = calc_stats(dd_returns, w, risk_free_rate, rebalance=rebalance,
                              asset_starts=dd_asset_starts)
            return abs(s_dd.max_drawdown) + s_dd.longest_dd * 5e-5
        return abs(s.max_drawdown) + s.longest_dd * 5e-5
    elif target == "Leverage-Optimal":
        L = leverage
        vol = s.volatility
        # Subtract dividend tax drag before leveraging (consistent with cfd.py).
        # CFD dividend adjustments are taxed as income; the total-return index
        # assumes gross reinvestment, so we apply the shortfall pre-leverage.
        div_drag = portfolio_dividend_drag(w) if w is not None else 0.0
        adjusted_cagr = s.cagr - div_drag
        # CMC Markets charges financing on FULL notional (not just borrowed portion).
        # financing_drag = financing_rate × L (per unit of deployed capital).
        post_lev_cagr = ((1.0 + adjusted_cagr) ** L
                         * np.exp(-0.5 * L * (L - 1) * vol**2)
                         - 1.0
                         - financing_rate * L)
        post_lev_vol = vol * L
        if post_lev_vol > 1e-4:
            post_lev_sharpe = (post_lev_cagr - risk_free_rate) / post_lev_vol
        else:
            post_lev_sharpe = 0.0
        return -post_lev_sharpe + dd_penalty
    return -s.sharpe + dd_penalty


def optimize_portfolio(
    returns: pd.DataFrame,
    target: str,
    min_w: np.ndarray,
    max_w: np.ndarray,
    group_max: dict[str, float],
    risk_free_rate: float = 0.04,
    current_weights: np.ndarray | None = None,
    rebalance: str = "daily",
    dd_constraint: float | None = None,
    dd_returns: pd.DataFrame | None = None,
    dd_asset_starts: dict | None = None,
    leverage: float = 1.0,
    financing_rate: float = 0.065,
) -> np.ndarray:
    """Run constrained optimization for Sharpe/Volatility/Calmar/Drawdown/Leverage targets."""
    n = len(ASSETS)
    groups = _group_indices()

    # Build constraints for scipy
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    for gname, indices in groups.items():
        cap = group_max.get(gname, 1.0)
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=indices, c=cap: c - w[idx].sum(),
        })

    bounds = list(zip(min_w, max_w))

    # ── Build pre-computed caches for fast objective evaluation ──
    opt_cache = StatsCache.build(returns, rebalance, asset_starts=None)
    dd_cache = None
    if dd_constraint is not None and dd_returns is not None:
        dd_cache = StatsCache.build(dd_returns, rebalance, asset_starts=dd_asset_starts)
    elif dd_constraint is not None:
        dd_cache = opt_cache  # evaluate DD on the same period

    def obj_cached(w):
        return _objective_cached(w, opt_cache, dd_cache, target, risk_free_rate,
                                 dd_constraint, leverage, financing_rate)

    # Legacy uncached objective (for _optimize_drawdown which uses differential_evolution)
    def obj(w):
        return _objective(w, returns, target, risk_free_rate, rebalance,
                          dd_constraint, dd_returns, dd_asset_starts,
                          leverage, financing_rate)

    # For drawdown minimisation, use penalty-based objective with
    # differential_evolution (gradient-free global optimizer) since
    # max drawdown is non-smooth and SLSQP struggles with it.
    if target == "Minimize Max Drawdown":
        return _optimize_drawdown(
            returns, min_w, max_w, group_max, risk_free_rate,
            current_weights, rebalance, constraints, bounds, obj,
        )

    # ── Two-phase warm-start for periodic rebalancing with DD constraints ──
    # Phase 1: solve with daily rebalancing (very fast, ~0.1ms/eval).
    # Phase 2: polish with the target rebalancing from that starting point.
    # Risk mitigation: if the polished result is worse than a full search
    # from standard starts, we keep the better result.
    daily_warm_start = None
    if rebalance != "daily" and dd_constraint is not None and current_weights is None:
        # Phase 1: fast daily solve
        daily_opt_cache = StatsCache.build(returns, "daily", asset_starts=None)
        daily_dd_cache = None
        if dd_returns is not None:
            daily_dd_cache = StatsCache.build(dd_returns, "daily", asset_starts=dd_asset_starts)
        else:
            daily_dd_cache = daily_opt_cache

        def obj_daily(w):
            return _objective_cached(w, daily_opt_cache, daily_dd_cache, target,
                                     risk_free_rate, dd_constraint, leverage, financing_rate)

        daily_starts = _build_starts(returns, min_w, max_w, group_max, None, target)
        daily_best_w, daily_best_val = None, np.inf
        for w0 in daily_starts:
            w0 = clip_normalize(w0, min_w, max_w, group_max)
            try:
                result = minimize(
                    obj_daily, w0, method="SLSQP", bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 300, "ftol": 1e-8},
                )
                if result.fun < daily_best_val:
                    daily_best_val = result.fun
                    daily_best_w = result.x
            except Exception:
                continue
        if daily_best_w is not None:
            daily_warm_start = clip_normalize(daily_best_w, min_w, max_w, group_max)

    # ── Standard SLSQP multi-start approach ──
    best_w = None
    best_val = np.inf

    starts = _build_starts(returns, min_w, max_w, group_max, current_weights, target)

    # When warm-starting from a previous solution, use fewer iterations
    # and only try the warm start + 1 other to speed up DD-constrained sweeps.
    if current_weights is not None and dd_constraint is not None:
        starts = starts[:2]  # warm start + inverse-vol only
        max_iter = 150
    elif daily_warm_start is not None:
        # Two-phase: use daily solution as primary start, keep inv-vol as backup
        starts = [daily_warm_start] + starts[:2]
        max_iter = 200  # fewer iterations needed — already near optimal
    else:
        max_iter = 300

    for w0 in starts:
        w0 = clip_normalize(w0, min_w, max_w, group_max)
        try:
            result = minimize(
                obj_cached, w0, method="SLSQP", bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iter, "ftol": 1e-8},
            )
            if result.fun < best_val:
                best_val = result.fun
                best_w = result.x
        except Exception:
            continue

    if best_w is None:
        best_w = np.full(n, 1.0 / n)

    return clip_normalize(best_w, min_w, max_w, group_max)


def _build_starts(returns, min_w, max_w, group_max, current_weights, target):
    """Build a list of starting points for optimisation."""
    n = len(ASSETS)
    starts = []

    if current_weights is not None:
        starts.append(current_weights)

    # Inverse-vol start
    vols = returns.std().reindex(ASSETS).fillna(returns.std().mean()).values
    vols = np.maximum(vols, 1e-6)
    iv = 1.0 / vols
    iv /= iv.sum()
    starts.append(iv)

    # Equal weight start
    starts.append(np.full(n, 1.0 / n))

    # For drawdown-focused targets, add low-vol biased starts
    if target in ("Minimize Max Drawdown", "Max Sharpe (DD \u2264 X%)", "Max Calmar (DD \u2264 X%)",
                  "Leverage-Optimal"):
        low_vol = iv.copy()
        low_vol **= 2
        low_vol /= low_vol.sum()
        starts.append(low_vol)

        # Bond-heavy start
        bond_w = np.full(n, 0.01)
        for i, a in enumerate(ASSETS):
            if GROUP_MAP[a] == "Bonds":
                bond_w[i] = 0.30
        bond_w /= bond_w.sum()
        starts.append(bond_w)

    return starts


def _optimize_drawdown(
    returns, min_w, max_w, group_max, risk_free_rate,
    current_weights, rebalance, constraints, bounds, obj,
):
    """Specialised optimiser for Minimize Max Drawdown.

    Phase 1: Minimise portfolio volatility as a fast, smooth proxy for drawdown
             (lower vol portfolios tend to have lower max drawdown).
    Phase 2: Also run SLSQP directly on the DD objective from multiple starts.
    Phase 3: Evaluate the true max-drawdown on ALL candidates and pick the best.
    Phase 4: Refine with Nelder-Mead (gradient-free, more robust for non-smooth DD).
    """
    n = len(ASSETS)
    cov = returns[ASSETS].cov().values * TD  # annualised covariance

    # Phase 1: Fast covariance-based search for low-vol portfolios
    def vol_obj(w):
        return np.sqrt(w @ cov @ w)

    vol_starts = _build_starts(returns, min_w, max_w, group_max, current_weights, "Minimize Max Drawdown")
    vol_candidates = []

    for w0 in vol_starts:
        w0 = clip_normalize(w0, min_w, max_w, group_max)
        try:
            result = minimize(
                vol_obj, w0, method="SLSQP", bounds=bounds,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-10},
            )
            w_cand = clip_normalize(result.x, min_w, max_w, group_max)
            vol_candidates.append(w_cand)
        except Exception:
            continue

    if not vol_candidates:
        vol_candidates = [clip_normalize(np.full(n, 1.0 / n), min_w, max_w, group_max)]

    # Also add random perturbations of top vol candidates for diversity
    vol_candidates.sort(key=vol_obj)
    rng = np.random.default_rng(42)
    for base_w in vol_candidates[:2]:
        for _ in range(3):
            perturbed = base_w + rng.normal(0, 0.01, n)
            perturbed = clip_normalize(np.maximum(perturbed, 0), min_w, max_w, group_max)
            vol_candidates.append(perturbed)

    # Phase 2: Also try direct SLSQP on the DD objective from multiple starts.
    # DD is non-smooth but SLSQP can still find local minima near good starts.
    dd_starts = vol_candidates[:3] + vol_starts[:2]
    for w0 in dd_starts:
        w0 = clip_normalize(w0, min_w, max_w, group_max)
        try:
            result = minimize(
                obj, w0, method="SLSQP", bounds=bounds,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-9},
            )
            w_cand = clip_normalize(result.x, min_w, max_w, group_max)
            vol_candidates.append(w_cand)
        except Exception:
            continue

    # Add defensive starts: max-bond and max-cash-bond blends
    # These are the most likely to have very low drawdowns.
    n_bonds = sum(1 for a in ASSETS if GROUP_MAP[a] == "Bonds")
    for bond_frac in [0.4, 0.6, 0.8]:
        w_bond = np.full(n, (1.0 - bond_frac) / n)
        for i, a in enumerate(ASSETS):
            if GROUP_MAP[a] == "Bonds":
                w_bond[i] = bond_frac / max(n_bonds, 1)
        w_bond = clip_normalize(w_bond, min_w, max_w, group_max)
        vol_candidates.append(w_bond)

    # Cash-heavy start (cash + short-term treasuries)
    w_cash = np.full(n, min_w.mean())
    for i, a in enumerate(ASSETS):
        if a in ("Cash", "Short-Term Treasuries"):
            w_cash[i] = 0.30
        elif a == "Gold":
            w_cash[i] = 0.10
    w_cash = clip_normalize(w_cash, min_w, max_w, group_max)
    vol_candidates.append(w_cash)

    # Min-vol result from Phase 1 with DD-focused SLSQP refinement
    # Use more perturbations of the best vol candidate
    if vol_candidates:
        best_vol = min(vol_candidates, key=vol_obj)
        for scale in [0.005, 0.01, 0.02, 0.03]:
            for _ in range(3):
                perturbed = best_vol + rng.normal(0, scale, n)
                perturbed = clip_normalize(np.maximum(perturbed, 0), min_w, max_w, group_max)
                vol_candidates.append(perturbed)

    # Phase 3: Evaluate true max-DD on ALL candidates and pick the best
    best_w = None
    best_val = np.inf

    for w_cand in vol_candidates:
        try:
            val = obj(w_cand)
            if val < best_val:
                best_val = val
                best_w = w_cand
        except Exception:
            continue

    # Phase 4: Refine the best candidate with Nelder-Mead (gradient-free)
    # More iterations and a better penalty structure for convergence.
    if best_w is not None:
        def constrained_obj(w):
            w_proj = clip_normalize(w, min_w, max_w, group_max)
            # Penalise deviation from feasible set (encourages staying feasible)
            deviation = np.sum((w - w_proj) ** 2)
            return obj(w_proj) + 50.0 * deviation

        try:
            result = minimize(
                constrained_obj, best_w, method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-10},
            )
            w_nm = clip_normalize(result.x, min_w, max_w, group_max)
            val_nm = obj(w_nm)
            if val_nm < best_val:
                best_w = w_nm
                best_val = val_nm
        except Exception:
            pass

        # Try a second Nelder-Mead pass with a perturbed start
        try:
            w_perturbed = best_w + rng.normal(0, 0.005, n)
            w_perturbed = clip_normalize(np.maximum(w_perturbed, 0), min_w, max_w, group_max)
            result2 = minimize(
                constrained_obj, w_perturbed, method="Nelder-Mead",
                options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-9},
            )
            w_nm2 = clip_normalize(result2.x, min_w, max_w, group_max)
            val_nm2 = obj(w_nm2)
            if val_nm2 < best_val:
                best_w = w_nm2
                best_val = val_nm2
        except Exception:
            pass

    if best_w is None:
        best_w = vol_candidates[0]

    return clip_normalize(best_w, min_w, max_w, group_max)


def inverse_volatility(
    returns: pd.DataFrame,
    min_w: np.ndarray,
    max_w: np.ndarray,
    group_max: dict[str, float],
) -> np.ndarray:
    """Inverse-volatility weighting (simplified risk parity)."""
    vols = returns.std().reindex(ASSETS).fillna(returns.std().mean()).values
    vols = np.maximum(vols, 1e-6)
    w = 1.0 / vols
    w /= w.sum()
    return clip_normalize(w, min_w, max_w, group_max)


def equal_risk_contribution(
    returns: pd.DataFrame,
    min_w: np.ndarray,
    max_w: np.ndarray,
    group_max: dict[str, float],
    max_iter: int = 2000,
    lr: float = 0.0005,
) -> np.ndarray:
    """Equal Risk Contribution (ERC) portfolio."""
    n = len(ASSETS)
    cov = returns[ASSETS].cov().values

    # Start from inverse-vol
    vols = np.sqrt(np.diag(cov))
    vols = np.maximum(vols, 1e-6)
    w = 1.0 / vols
    w /= w.sum()

    converged = False
    for i in range(max_iter):
        sigma_w = cov @ w
        risk_contrib = w * sigma_w
        port_var = risk_contrib.sum()
        target_rc = port_var / n

        max_dev = np.max(np.abs(risk_contrib - target_rc))
        if max_dev < 1e-7:
            converged = True
            break

        for j in range(n):
            if risk_contrib[j] > 0:
                w[j] *= (target_rc / risk_contrib[j]) ** lr
            w[j] = max(w[j], 1e-4)

        w /= w.sum()

    if not converged:
        warnings.warn(
            f"ERC did not converge after {max_iter} iterations "
            f"(max deviation: {max_dev:.2e}). Results may be approximate.",
            stacklevel=2,
        )

    return clip_normalize(w, min_w, max_w, group_max)


def hierarchical_risk_parity(
    returns: pd.DataFrame,
    min_w: np.ndarray,
    max_w: np.ndarray,
    group_max: dict[str, float],
) -> np.ndarray:
    """Hierarchical Risk Parity (HRP) - Marcos Lopez de Prado."""
    n = len(ASSETS)
    corr = returns[ASSETS].corr().values
    cov = returns[ASSETS].cov().values
    vols = np.sqrt(np.diag(cov))

    # Distance matrix from correlation
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist)
    link = linkage(condensed, method="ward")
    order = leaves_list(link).astype(int)

    # Recursive bisection
    w = np.ones(n)

    def _subport_var(indices):
        if len(indices) == 1:
            return cov[indices[0], indices[0]]
        sub_vols = vols[indices]
        sub_vols = np.maximum(sub_vols, 1e-6)
        inv_v = 1.0 / sub_vols
        inv_v /= inv_v.sum()
        sub_cov = cov[np.ix_(indices, indices)]
        return inv_v @ sub_cov @ inv_v

    def _bisect(order_slice):
        if len(order_slice) <= 1:
            return
        mid = len(order_slice) // 2
        left = order_slice[:mid]
        right = order_slice[mid:]

        var_left = _subport_var(left)
        var_right = _subport_var(right)

        total_var = var_left + var_right
        alpha = 1.0 - var_left / total_var if total_var > 0 else 0.5

        for i in left:
            w[i] *= alpha
        for i in right:
            w[i] *= (1.0 - alpha)

        _bisect(left)
        _bisect(right)

    _bisect(list(order))
    w /= w.sum()

    return clip_normalize(w, min_w, max_w, group_max)


def carry_adjusted_risk_parity(
    returns: pd.DataFrame,
    min_w: np.ndarray,
    max_w: np.ndarray,
    group_max: dict[str, float],
    financing_rate: float = 0.065,
    leverage: float = 5.0,
    carry_sensitivity: float = 5.0,
) -> np.ndarray:
    """Equal Risk Contribution adjusted for CFD financing costs.

    Assets with positive net carry (CAGR > financing cost) receive higher
    weights; those with negative carry are underweighted.

    carry_sensitivity controls how aggressively carry differences tilt weights.
    Higher values concentrate more into high-carry assets; lower values keep
    the allocation closer to base ERC. The exponent is clamped to [-5, 5]
    regardless, so values above ~5.0 saturate for carry spreads > 1%.
    """
    # Base ERC weights
    erc_w = equal_risk_contribution(returns, min_w, max_w, group_max)

    # Per-asset annualised CAGR from daily log returns
    n_days = len(returns)
    total_calendar = (returns.index[-1] - returns.index[0]).days
    asset_cagrs = np.zeros(len(ASSETS))
    for i, a in enumerate(ASSETS):
        cum_log = returns[a].sum()
        if total_calendar > 0:
            asset_cagrs[i] = np.exp(cum_log * 365.0 / total_calendar) - 1.0

    # Net carry = asset return minus financing cost per unit leverage
    net_carry = asset_cagrs - financing_rate

    # Softmax-style carry score: exp(net_carry * carry_sensitivity).
    # Clamp the exponent to [-5, 5] (≈ [0.007, 148] range) before applying
    # the floor, to prevent overflow from extreme-carry assets (e.g. Bitcoin
    # in a bull run) dominating the pre-normalisation weights.
    carry_score = np.exp(np.clip(net_carry * carry_sensitivity, -5.0, 5.0))
    carry_score = np.maximum(carry_score, 0.1)  # floor to avoid zeroing out

    adjusted = erc_w * carry_score
    adjusted = np.maximum(adjusted, 0.0)
    total = adjusted.sum()
    if total > 1e-12:
        adjusted /= total

    return clip_normalize(adjusted, min_w, max_w, group_max)


def adaptive_lookback_blend(
    returns: pd.DataFrame,
    min_w: np.ndarray,
    max_w: np.ndarray,
    group_max: dict[str, float],
    risk_free_rate: float = 0.04,
    windows: list[int] | None = None,
) -> np.ndarray:
    """Blend Max Sharpe weights optimised across multiple trailing windows.

    Produces smoother allocations that adapt across time horizons, reducing
    sensitivity to the choice of lookback period.
    """
    if windows is None:
        windows = [63, 126, 252, 504]

    n = len(ASSETS)
    all_weights = []

    for w_len in windows:
        subset = returns.iloc[-w_len:] if len(returns) >= w_len else returns
        if len(subset) < 30:
            # Too few observations — use equal weight as fallback
            all_weights.append(np.full(n, 1.0 / n))
            continue
        w = optimize_portfolio(subset, "Max Sharpe Ratio", min_w, max_w,
                               group_max, risk_free_rate)
        all_weights.append(w)

    # Equal-weight blend of all window results
    blended = np.mean(all_weights, axis=0)
    return clip_normalize(blended, min_w, max_w, group_max)


def run_optimization(
    returns: pd.DataFrame,
    target: str,
    min_w: np.ndarray,
    max_w: np.ndarray,
    group_max: dict[str, float],
    risk_free_rate: float = 0.04,
    current_weights: np.ndarray | None = None,
    rebalance: str = "daily",
    dd_constraint: float | None = None,
    dd_returns: pd.DataFrame | None = None,
    dd_asset_starts: dict | None = None,
    leverage: float = 1.0,
    financing_rate: float = 0.065,
) -> np.ndarray:
    """Dispatch to the appropriate optimization strategy."""
    # Unconstrained variants: strip group caps, use the underlying objective
    if target == "Max Sharpe (Unconstrained)":
        return optimize_portfolio(
            returns, "Max Sharpe Ratio", min_w, max_w, {},
            risk_free_rate, current_weights, rebalance, dd_constraint,
            dd_returns, dd_asset_starts, leverage, financing_rate,
        )
    elif target == "Leverage-Optimal (Unconstrained)":
        return optimize_portfolio(
            returns, "Leverage-Optimal", min_w, max_w, {},
            risk_free_rate, current_weights, rebalance, dd_constraint,
            dd_returns, dd_asset_starts, leverage, financing_rate,
        )
    elif target == "Inverse Volatility":
        return inverse_volatility(returns, min_w, max_w, group_max)
    elif target == "Equal Risk Contribution":
        return equal_risk_contribution(returns, min_w, max_w, group_max)
    elif target == "Hierarchical Risk Parity":
        return hierarchical_risk_parity(returns, min_w, max_w, group_max)
    elif target == "Carry-Adjusted Risk Parity":
        return carry_adjusted_risk_parity(
            returns, min_w, max_w, group_max, financing_rate, leverage,
        )
    elif target == "Adaptive Lookback Blend":
        return adaptive_lookback_blend(
            returns, min_w, max_w, group_max, risk_free_rate,
        )
    else:
        return optimize_portfolio(
            returns, target, min_w, max_w, group_max,
            risk_free_rate, current_weights, rebalance, dd_constraint,
            dd_returns, dd_asset_starts, leverage, financing_rate,
        )
