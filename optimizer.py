"""Portfolio optimization strategies."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from data import ASSETS, GROUP_MAP, GROUP_NAMES
from stats import calc_stats, TD


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
    w = w.copy()
    n = len(w)
    groups = _group_indices()

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
) -> float:
    """Objective function for optimization (negated for minimization).

    When dd_constraint is set, a heavy penalty is added if drawdown exceeds the limit.
    dd_returns / dd_asset_starts: if provided, evaluate the DD penalty on this
    (potentially longer) backtest period instead of the optimisation returns.
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

    if target in ("Max Sharpe Ratio", "Max Sharpe (DD \u2264 X%)"):
        return -s.sharpe + dd_penalty
    elif target == "Min Volatility":
        return s.volatility + dd_penalty
    elif target in ("Max Calmar Ratio", "Max Calmar (DD \u2264 X%)"):
        return -s.calmar + dd_penalty
    elif target == "Minimize Max Drawdown":
        return abs(s.max_drawdown) + s.longest_dd * 5e-5
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
) -> np.ndarray:
    """Run constrained optimization for Sharpe/Volatility/Calmar/Drawdown targets."""
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

    def obj(w):
        return _objective(w, returns, target, risk_free_rate, rebalance,
                          dd_constraint, dd_returns, dd_asset_starts)

    # For drawdown minimisation, use penalty-based objective with
    # differential_evolution (gradient-free global optimizer) since
    # max drawdown is non-smooth and SLSQP struggles with it.
    if target == "Minimize Max Drawdown":
        return _optimize_drawdown(
            returns, min_w, max_w, group_max, risk_free_rate,
            current_weights, rebalance, constraints, bounds, obj,
        )

    # ── Standard SLSQP multi-start approach for smooth objectives ──
    best_w = None
    best_val = np.inf

    starts = _build_starts(returns, min_w, max_w, group_max, current_weights, target)

    # When warm-starting from a previous solution, use fewer iterations
    # and only try the warm start + 1 other to speed up DD-constrained sweeps.
    if current_weights is not None and dd_constraint is not None:
        starts = starts[:2]  # warm start + inverse-vol only
        max_iter = 150
    else:
        max_iter = 300

    for w0 in starts:
        w0 = clip_normalize(w0, min_w, max_w, group_max)
        try:
            result = minimize(
                obj, w0, method="SLSQP", bounds=bounds,
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
    if target in ("Minimize Max Drawdown", "Max Sharpe (DD \u2264 X%)", "Max Calmar (DD \u2264 X%)"):
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
    Phase 2: Pick the best vol-minimised candidate and refine with SLSQP on the
             true max-drawdown objective (limited iterations for speed).
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
        return clip_normalize(np.full(n, 1.0 / n), min_w, max_w, group_max)

    # Phase 2: Refine the top 3 vol candidates on true max-drawdown objective
    # Sort by vol and take the best few — keeps Phase 2 fast
    vol_candidates.sort(key=vol_obj)
    phase2_starts = vol_candidates[:3]

    best_w = None
    best_val = np.inf

    for w0 in phase2_starts:
        try:
            result = minimize(
                obj, w0, method="SLSQP", bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-8},
            )
            if result.fun < best_val:
                best_val = result.fun
                best_w = result.x
        except Exception:
            continue

    if best_w is None:
        best_w = vol_candidates[0]

    return clip_normalize(best_w, min_w, max_w, group_max)


def risk_parity(
    returns: pd.DataFrame,
    min_w: np.ndarray,
    max_w: np.ndarray,
    group_max: dict[str, float],
) -> np.ndarray:
    """Inverse-volatility (risk parity) weighting."""
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

    for i in range(max_iter):
        sigma_w = cov @ w
        risk_contrib = w * sigma_w
        port_var = risk_contrib.sum()
        target_rc = port_var / n

        max_dev = np.max(np.abs(risk_contrib - target_rc))
        if max_dev < 1e-7:
            break

        for j in range(n):
            if risk_contrib[j] > 0:
                w[j] *= (target_rc / risk_contrib[j]) ** lr
            w[j] = max(w[j], 1e-4)

        w /= w.sum()

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
) -> np.ndarray:
    """Dispatch to the appropriate optimization strategy."""
    if target == "Risk Parity":
        return risk_parity(returns, min_w, max_w, group_max)
    elif target == "Equal Risk Contribution":
        return equal_risk_contribution(returns, min_w, max_w, group_max)
    elif target == "Hierarchical Risk Parity":
        return hierarchical_risk_parity(returns, min_w, max_w, group_max)
    else:
        return optimize_portfolio(
            returns, target, min_w, max_w, group_max,
            risk_free_rate, current_weights, rebalance, dd_constraint,
            dd_returns, dd_asset_starts,
        )
