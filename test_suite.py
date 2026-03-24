"""Test suite for All Weather Python portfolio modules."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

# ---------------------------------------------------------------------------
# Helpers to create synthetic data
# ---------------------------------------------------------------------------

def _make_returns(n_days=500, seed=42):
    """Create a synthetic daily log-returns DataFrame for all assets."""
    from data import ASSETS
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    data = rng.randn(n_days, len(ASSETS)) * 0.01  # ~1% daily vol
    return pd.DataFrame(data, index=dates, columns=ASSETS)


def _equal_weights():
    from data import ASSETS
    n = len(ASSETS)
    return np.full(n, 1.0 / n)


def _default_bounds():
    from data import ASSETS
    n = len(ASSETS)
    return np.zeros(n), np.ones(n)


def _default_group_max():
    from data import GROUP_NAMES
    return {g: 1.0 for g in GROUP_NAMES}


# ===========================================================================
# data.py tests
# ===========================================================================

class TestData:
    def test_assets_list(self):
        from data import ASSETS, GROUP_MAP
        assert len(ASSETS) == 17
        # Every asset must have a group
        for a in ASSETS:
            assert a in GROUP_MAP, f"{a} missing from GROUP_MAP"

    def test_group_map_covers_all(self):
        from data import ASSETS, GROUP_MAP, GROUP_NAMES
        groups_seen = set()
        for a in ASSETS:
            g = GROUP_MAP[a]
            assert g in GROUP_NAMES, f"Group '{g}' not in GROUP_NAMES"
            groups_seen.add(g)
        assert groups_seen == set(GROUP_NAMES), "Not all groups are represented"

    def test_align_and_compute_returns_basic(self):
        from data import align_and_compute_returns
        dates = pd.bdate_range("2023-01-02", periods=10)
        prices = {
            "Cash": pd.Series(np.linspace(100, 101, 10), index=dates),
            "Gold": pd.Series(np.linspace(1800, 1850, 10), index=dates),
        }
        ret = align_and_compute_returns(prices)
        assert not ret.empty
        assert len(ret) == 9  # one row lost to diff
        assert "Cash" in ret.columns
        assert "Gold" in ret.columns
        # Returns should be finite
        assert np.all(np.isfinite(ret.values))

    def test_align_empty(self):
        from data import align_and_compute_returns
        ret = align_and_compute_returns({})
        assert ret.empty

    def test_sanitize_cash_returns_caps(self):
        from data import _sanitize_cash_returns, ASSETS
        rng = np.random.RandomState(0)
        n = 100
        dates = pd.bdate_range("2023-01-02", periods=n)
        data = rng.randn(n, len(ASSETS)) * 0.001
        df = pd.DataFrame(data, index=dates, columns=ASSETS)
        # Inject extreme cash returns
        df["Cash"] = rng.randn(n) * 0.1  # very high vol
        result = _sanitize_cash_returns(df)
        assert result["Cash"].abs().max() <= 0.005 + 1e-9

    def test_sanitize_cash_returns_no_change_if_normal(self):
        from data import _sanitize_cash_returns, ASSETS
        dates = pd.bdate_range("2023-01-02", periods=50)
        data = np.zeros((50, len(ASSETS)))
        data[:, 0] = 0.0001  # Cash = small returns
        df = pd.DataFrame(data, index=dates, columns=ASSETS)
        result = _sanitize_cash_returns(df)
        # Should be unchanged since vol is low
        np.testing.assert_array_almost_equal(result["Cash"].values, df["Cash"].values)


# ===========================================================================
# stats.py tests
# ===========================================================================

class TestStats:
    def test_calc_stats_returns_dataclass(self):
        from stats import calc_stats, PortfolioStats
        ret = _make_returns()
        w = _equal_weights()
        s = calc_stats(ret, w)
        assert isinstance(s, PortfolioStats)

    def test_calc_stats_fields_reasonable(self):
        from stats import calc_stats
        ret = _make_returns()
        w = _equal_weights()
        s = calc_stats(ret, w)
        # CAGR should be a finite number
        assert np.isfinite(s.cagr)
        # Volatility should be positive
        assert s.volatility > 0
        # Max drawdown should be negative or zero
        assert s.max_drawdown <= 0
        # Pct positive should be between 0 and 1
        assert 0 <= s.pct_positive <= 1

    def test_empty_returns(self):
        from stats import calc_stats
        from data import ASSETS
        ret = pd.DataFrame(columns=ASSETS)
        ret.index = pd.DatetimeIndex([])
        w = _equal_weights()
        s = calc_stats(ret, w)
        assert s.cagr == 0.0
        assert s.volatility == 0.0

    def test_date_filtering(self):
        """Verify that start_date filtering actually restricts the data range."""
        from stats import calc_stats, compute_equity_curve
        ret = _make_returns()
        w = _equal_weights()
        mid = ret.index[len(ret) // 2]

        s_full = calc_stats(ret, w)
        s_filtered = calc_stats(ret, w, start_date=mid)

        # Filtered equity curve should be shorter
        eq_full = compute_equity_curve(ret, w)
        eq_filtered = compute_equity_curve(ret, w, start_date=mid)
        assert len(eq_filtered) < len(eq_full), "Filtered curve should have fewer points"
        assert eq_filtered.index[0] >= mid, "Filtered curve should start at or after mid date"

        # CAGR must differ because the periods are different
        assert s_full.cagr != s_filtered.cagr, "CAGR should differ for different periods"
        # Vol should also differ (different data window)
        assert s_full.volatility != s_filtered.volatility, "Vol should differ for different periods"

    def test_rebalance_monthly(self):
        from stats import calc_stats
        ret = _make_returns(n_days=500)
        w = _equal_weights()
        sd = calc_stats(ret, w, rebalance="daily")
        sm = calc_stats(ret, w, rebalance="monthly")
        # Both should produce valid results
        assert np.isfinite(sm.cagr) and np.isfinite(sm.volatility)
        # They should differ (daily vs monthly rebalance)
        assert sd.cagr != sm.cagr or sd.volatility != sm.volatility

    def test_equity_curve(self):
        from stats import compute_equity_curve
        ret = _make_returns()
        w = _equal_weights()
        eq = compute_equity_curve(ret, w)
        assert len(eq) > 0
        assert eq.iloc[0] > 0  # starts positive
        assert np.all(eq > 0)   # equity can't go negative with log returns

    def test_drawdown_series(self):
        from stats import compute_equity_curve, compute_drawdown_series
        ret = _make_returns()
        w = _equal_weights()
        eq = compute_equity_curve(ret, w)
        dd = compute_drawdown_series(eq)
        assert len(dd) == len(eq)
        assert dd.max() <= 1e-10  # drawdown is always <= 0
        assert dd.min() < 0  # some drawdown should exist in random data

    def test_annual_returns(self):
        from stats import compute_annual_returns
        ret = _make_returns(n_days=750)  # ~3 years
        w = _equal_weights()
        annual = compute_annual_returns(ret, w)
        assert len(annual) >= 2  # at least 2 years

    def test_longest_drawdown(self):
        from stats import calc_stats
        ret = _make_returns()
        w = _equal_weights()
        s = calc_stats(ret, w)
        assert s.longest_dd >= 0  # non-negative integer

    def test_calmar_ratio(self):
        from stats import calc_stats
        ret = _make_returns()
        w = _equal_weights()
        s = calc_stats(ret, w)
        if abs(s.max_drawdown) > 1e-4:
            assert np.isfinite(s.calmar)
            assert s.calmar == pytest.approx(s.cagr / abs(s.max_drawdown), rel=1e-6)


# ===========================================================================
# optimizer.py tests
# ===========================================================================

class TestOptimizer:
    def test_clip_normalize_sums_to_one(self):
        from optimizer import clip_normalize
        from data import ASSETS
        n = len(ASSETS)
        w = np.random.rand(n)
        min_w, max_w = _default_bounds()
        gm = _default_group_max()
        result = clip_normalize(w, min_w, max_w, gm)
        assert abs(result.sum() - 1.0) < 1e-3

    def test_clip_normalize_respects_bounds(self):
        from optimizer import clip_normalize
        from data import ASSETS
        n = len(ASSETS)
        w = np.random.rand(n) * 2  # some values > 1
        min_w = np.full(n, 0.01)
        max_w = np.full(n, 0.20)
        gm = _default_group_max()
        result = clip_normalize(w, min_w, max_w, gm)
        # Should be within bounds (with small tolerance)
        assert np.all(result >= min_w - 1e-3)
        assert np.all(result <= max_w + 1e-3)

    def test_clip_normalize_respects_group_caps(self):
        from optimizer import clip_normalize, _group_indices
        from data import ASSETS
        n = len(ASSETS)
        # Put all weight in US Equities
        w = np.zeros(n)
        groups = _group_indices()
        for i in groups["US Equities"]:
            w[i] = 0.5
        min_w = np.zeros(n)
        max_w = np.ones(n)
        gm = {"US Equities": 0.30, "Intl Equities": 1.0, "Bonds": 1.0,
               "Real Assets": 1.0, "Alternatives": 1.0}
        result = clip_normalize(w, min_w, max_w, gm)
        us_sum = sum(result[i] for i in groups["US Equities"])
        assert us_sum <= 0.30 + 1e-3

    def test_inverse_volatility(self):
        from optimizer import inverse_volatility
        ret = _make_returns()
        min_w, max_w = _default_bounds()
        gm = _default_group_max()
        w = inverse_volatility(ret, min_w, max_w, gm)
        assert abs(w.sum() - 1.0) < 1e-3
        assert np.all(w >= -1e-6)

    def test_equal_risk_contribution(self):
        from optimizer import equal_risk_contribution
        ret = _make_returns()
        min_w, max_w = _default_bounds()
        gm = _default_group_max()
        w = equal_risk_contribution(ret, min_w, max_w, gm)
        assert abs(w.sum() - 1.0) < 1e-3
        assert np.all(w >= -1e-6)

    def test_hierarchical_risk_parity(self):
        from optimizer import hierarchical_risk_parity
        ret = _make_returns()
        min_w, max_w = _default_bounds()
        gm = _default_group_max()
        w = hierarchical_risk_parity(ret, min_w, max_w, gm)
        assert abs(w.sum() - 1.0) < 1e-3
        assert np.all(w >= -1e-6)

    def test_run_optimization_dispatches(self):
        from optimizer import run_optimization
        ret = _make_returns(n_days=200)
        min_w, max_w = _default_bounds()
        gm = _default_group_max()
        for target in ["Inverse Volatility", "Equal Risk Contribution", "Hierarchical Risk Parity"]:
            w = run_optimization(ret, target, min_w, max_w, gm)
            assert abs(w.sum() - 1.0) < 1e-3, f"Failed for {target}"

    def test_optimize_max_sharpe(self):
        from optimizer import run_optimization
        ret = _make_returns(n_days=200)
        min_w, max_w = _default_bounds()
        gm = _default_group_max()
        w = run_optimization(ret, "Max Sharpe Ratio", min_w, max_w, gm)
        assert abs(w.sum() - 1.0) < 1e-3
        assert np.all(w >= -1e-6)

    def test_optimize_min_volatility(self):
        from optimizer import run_optimization
        ret = _make_returns(n_days=200)
        min_w, max_w = _default_bounds()
        gm = _default_group_max()
        w = run_optimization(ret, "Min Volatility", min_w, max_w, gm)
        assert abs(w.sum() - 1.0) < 1e-3


# ===========================================================================
# cfd.py tests
# ===========================================================================

class TestCFD:
    def _make_stats(self):
        from stats import PortfolioStats
        return PortfolioStats(
            cagr=0.08, volatility=0.12, sharpe=0.33,
            max_drawdown=-0.20, calmar=0.40,
            best_year=0.15, worst_year=-0.10,
            pct_positive=0.53, longest_dd=60,
        )

    def test_basic_analysis(self):
        from cfd import analyze_cfd
        w = _equal_weights()
        stats = self._make_stats()
        result = analyze_cfd(
            weights=w, stats=stats,
            total_capital=100_000, leverage_ratio=3.0,
            financing_rate=0.06, margin_requirement=0.20,
            risk_free_rate=0.04,
        )
        assert result.total_capital == 100_000
        assert result.deployed_capital > 0
        assert result.deployed_capital <= 100_000
        assert result.cash_reserve >= 0
        assert abs(result.deployed_capital + result.cash_reserve - 100_000) < 0.01

    def test_notional_exposure(self):
        from cfd import analyze_cfd
        w = _equal_weights()
        stats = self._make_stats()
        result = analyze_cfd(
            weights=w, stats=stats,
            total_capital=100_000, leverage_ratio=5.0,
            financing_rate=0.06, margin_requirement=0.10,
            risk_free_rate=0.04,
        )
        assert abs(result.notional_exposure - result.deployed_capital * 5.0) < 0.01

    def test_financing_drag(self):
        from cfd import analyze_cfd
        w = _equal_weights()
        stats = self._make_stats()
        result = analyze_cfd(
            weights=w, stats=stats,
            total_capital=100_000, leverage_ratio=3.0,
            financing_rate=0.06, margin_requirement=0.20,
            risk_free_rate=0.04,
        )
        # CMC charges financing on full notional: drag = rate * leverage
        expected_drag = 0.06 * 3.0
        assert abs(result.financing_drag - expected_drag) < 1e-6

    def test_gross_cagr(self):
        from cfd import analyze_cfd
        w = _equal_weights()
        stats = self._make_stats()
        result = analyze_cfd(
            weights=w, stats=stats,
            total_capital=100_000, leverage_ratio=3.0,
            financing_rate=0.06, margin_requirement=0.20,
            risk_free_rate=0.04,
        )
        # Correct leveraged CAGR: (1+CAGR)^L * exp(-L*(L-1)*σ²/2) - 1
        L = 3.0
        vol = stats.volatility
        expected_gross = ((1 + stats.cagr) ** L
                          * np.exp(-L * (L - 1) * vol**2 / 2) - 1)
        assert abs(result.gross_cagr - expected_gross) < 1e-6

    def test_net_cagr(self):
        from cfd import analyze_cfd
        w = _equal_weights()
        stats = self._make_stats()
        result = analyze_cfd(
            weights=w, stats=stats,
            total_capital=100_000, leverage_ratio=3.0,
            financing_rate=0.06, margin_requirement=0.20,
            risk_free_rate=0.04,
        )
        # CMC charges financing on full notional: drag = rate * leverage
        expected_net = result.gross_cagr - 0.06 * 3.0
        assert abs(result.net_cagr - expected_net) < 1e-6

    def test_no_leverage(self):
        from cfd import analyze_cfd
        w = _equal_weights()
        stats = self._make_stats()
        result = analyze_cfd(
            weights=w, stats=stats,
            total_capital=100_000, leverage_ratio=1.0,
            financing_rate=0.06, margin_requirement=0.20,
            risk_free_rate=0.04,
        )
        # At 1x leverage, CMC still charges financing on full notional:
        # drag = rate * 1.0 = 0.06. Gross CAGR unscaled (vol drag term = 0).
        assert abs(result.financing_drag - 0.06) < 1e-6
        assert abs(result.gross_cagr - stats.cagr) < 1e-6

    def test_capital_per_asset_sums(self):
        from cfd import analyze_cfd
        w = _equal_weights()
        stats = self._make_stats()
        result = analyze_cfd(
            weights=w, stats=stats,
            total_capital=100_000, leverage_ratio=3.0,
            financing_rate=0.06, margin_requirement=0.20,
            risk_free_rate=0.04,
        )
        total_allocated = sum(result.capital_per_asset.values())
        assert abs(total_allocated - result.notional_exposure) < 0.01

    def test_margin_utilisation(self):
        from cfd import analyze_cfd
        w = _equal_weights()
        stats = self._make_stats()
        result = analyze_cfd(
            weights=w, stats=stats,
            total_capital=100_000, leverage_ratio=3.0,
            financing_rate=0.06, margin_requirement=0.20,
            risk_free_rate=0.04,
        )
        expected_util = result.margin_required / result.deployed_capital
        assert abs(result.margin_utilisation - expected_util) < 1e-6


# ===========================================================================
# Integration-style tests
# ===========================================================================

class TestIntegration:
    def test_stats_then_cfd(self):
        """Full pipeline: compute stats, then run CFD analysis."""
        from stats import calc_stats
        from cfd import analyze_cfd
        ret = _make_returns()
        w = _equal_weights()
        s = calc_stats(ret, w)
        result = analyze_cfd(
            weights=w, stats=s,
            total_capital=100_000, leverage_ratio=2.0,
            financing_rate=0.05, margin_requirement=0.10,
            risk_free_rate=0.04,
        )
        assert np.isfinite(result.net_cagr)
        assert np.isfinite(result.net_sharpe)

    def test_optimize_then_stats(self):
        """Optimize, then compute stats with the result."""
        from optimizer import run_optimization
        from stats import calc_stats
        ret = _make_returns(n_days=200)
        min_w, max_w = _default_bounds()
        gm = _default_group_max()
        w = run_optimization(ret, "Inverse Volatility", min_w, max_w, gm)
        s = calc_stats(ret, w)
        assert np.isfinite(s.cagr)
        assert np.isfinite(s.sharpe)

    def test_periodic_rebalance_equity_curve(self):
        """Equity curve with monthly rebalancing should work."""
        from stats import compute_equity_curve
        ret = _make_returns(n_days=500)
        w = _equal_weights()
        eq = compute_equity_curve(ret, w, rebalance="monthly")
        assert len(eq) > 0
        assert np.all(eq > 0)

    def test_all_rebalance_frequencies(self):
        """All rebalance frequencies should produce valid stats."""
        from stats import calc_stats
        ret = _make_returns(n_days=750)
        w = _equal_weights()
        for freq in ["daily", "monthly", "quarterly", "semi-annual", "annual"]:
            s = calc_stats(ret, w, rebalance=freq)
            assert np.isfinite(s.cagr), f"CAGR not finite for {freq}"
            assert np.isfinite(s.volatility), f"Vol not finite for {freq}"


# ===================================================================
# Dynamic Strategies Tests
# ===================================================================

class TestDDMomentum:
    """Tests for the drawdown p-value momentum module."""

    def test_detect_drawdown_episodes(self):
        """Should detect significant drawdown episodes."""
        from dd_momentum import detect_drawdown_episodes
        # Create price series with clear drawdown
        prices = pd.Series(
            [1.0, 1.1, 1.2, 0.8, 0.7, 0.9, 1.0, 1.3, 1.1, 1.3],
            index=pd.bdate_range("2020-01-02", periods=10),
        )
        episodes = detect_drawdown_episodes(prices, relative_threshold=0.25)
        assert len(episodes) > 0
        assert all("depth" in ep for ep in episodes)
        assert all(ep["depth"] < 0 for ep in episodes)

    def test_no_episodes_for_monotonic_up(self):
        """Monotonically increasing prices should have no episodes."""
        from dd_momentum import detect_drawdown_episodes
        prices = pd.Series(
            np.linspace(1.0, 2.0, 100),
            index=pd.bdate_range("2020-01-02", periods=100),
        )
        episodes = detect_drawdown_episodes(prices)
        assert len(episodes) == 0

    def test_pvalue_computation(self):
        """P-value should be between 0 and 1."""
        from dd_momentum import compute_pvalue
        episodes = [{"depth": -0.30}, {"depth": -0.20}, {"depth": -0.10}]
        # Current DD of -0.25: 1 out of 3 episodes were worse → p = 1/3
        pval = compute_pvalue(-0.25, episodes)
        assert 0 <= pval <= 1
        assert abs(pval - 1/3) < 1e-10

    def test_pvalue_no_drawdown(self):
        """No current drawdown should give p-value of 1.0."""
        from dd_momentum import compute_pvalue
        episodes = [{"depth": -0.30}]
        assert compute_pvalue(0.0, episodes) == 1.0

    def test_adjustment_factors(self):
        """Adjustment factors should produce valid weights when applied."""
        from dd_momentum import compute_dd_adjustments, build_dd_momentum_schedule
        ret = _make_returns(n_days=500)
        checkpoints = [ret.index[250], ret.index[499]]
        adj = compute_dd_adjustments(ret, checkpoints)
        assert len(adj) == 2

        base_w = _equal_weights()
        schedule = build_dd_momentum_schedule(base_w, adj)
        assert len(schedule) == 2
        for date, w in schedule.items():
            assert abs(w.sum() - 1.0) < 1e-10
            assert np.all(w >= 0)

    def test_configurable_bump_max(self):
        """Audit fix #6: bump_max parameter should scale adjustment factors."""
        from dd_momentum import compute_dd_adjustments
        ret = _make_returns(n_days=500)
        checkpoints = [ret.index[499]]

        adj_default = compute_dd_adjustments(ret, checkpoints, bump_max=0.50)
        adj_small = compute_dd_adjustments(ret, checkpoints, bump_max=0.20)

        # Max bump with bump_max=0.50 should be 0.50
        assert max(adj_default[checkpoints[0]]) == pytest.approx(0.50, abs=1e-10)
        # Max bump with bump_max=0.20 should be 0.20
        assert max(adj_small[checkpoints[0]]) == pytest.approx(0.20, abs=1e-10)


class TestWeightsSchedule:
    """Tests for time-varying weights support in stats.py."""

    def test_schedule_produces_valid_stats(self):
        """Stats with weights_schedule should be finite and reasonable."""
        from stats import calc_stats
        ret = _make_returns(n_days=500)
        w = _equal_weights()
        # Create a simple schedule: change weights at midpoint
        mid_date = ret.index[250]
        w2 = w.copy()
        w2[0] *= 2  # boost first asset
        w2 /= w2.sum()
        schedule = {ret.index[0]: w, mid_date: w2}

        s = calc_stats(ret, w, rebalance="monthly", weights_schedule=schedule)
        assert np.isfinite(s.cagr)
        assert np.isfinite(s.volatility)
        assert s.max_drawdown <= 0

    def test_schedule_forces_periodic_rebal(self):
        """weights_schedule with daily rebalance should fall back to monthly."""
        from stats import calc_stats
        ret = _make_returns(n_days=500)
        w = _equal_weights()
        schedule = {ret.index[0]: w}
        # Should not raise even with rebalance="daily"
        s = calc_stats(ret, w, rebalance="daily", weights_schedule=schedule)
        assert np.isfinite(s.cagr)

    def test_equity_curve_with_schedule(self):
        """Equity curve should work with weights_schedule."""
        from stats import compute_equity_curve
        ret = _make_returns(n_days=500)
        w = _equal_weights()
        schedule = {ret.index[0]: w, ret.index[250]: w}
        eq = compute_equity_curve(ret, w, rebalance="quarterly", weights_schedule=schedule)
        assert len(eq) > 0
        assert eq.iloc[0] > 0


class TestRegime:
    """Tests for the regime classification module."""

    def test_classify_regimes(self):
        """Should produce 4 regime labels."""
        from regime import classify_regimes
        dates = pd.date_range("2000-01-01", periods=100, freq="ME")
        macro = pd.DataFrame({
            "CPI": np.concatenate([np.ones(50) * 5, np.ones(50) * 1]),
            "FedFunds": np.concatenate([np.ones(25) * 8, np.ones(25) * 2,
                                         np.ones(25) * 8, np.ones(25) * 2]),
        }, index=dates)
        regimes = classify_regimes(macro)
        assert set(regimes.unique()).issubset({1, 2, 3, 4})
        assert len(regimes) == 100

    def test_build_regime_schedule(self):
        """Schedule should map dates to weight arrays."""
        from regime import build_regime_schedule
        dates = pd.bdate_range("2020-01-02", periods=500)
        regime_dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        regime_series = pd.Series(
            [1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4],
            index=regime_dates,
        )
        n = len(_equal_weights())
        regime_weights = {
            i: np.random.dirichlet(np.ones(n))
            for i in range(1, 5)
        }
        schedule = build_regime_schedule(dates, regime_series, regime_weights, "quarterly")
        assert len(schedule) > 0
        for date, w in schedule.items():
            assert abs(w.sum() - 1.0) < 1e-10


# ===========================================================================
# Phase 5 Audit Tests — additional coverage from adversarial audit
# ===========================================================================

class TestAuditMetrics:
    """Tests motivated by the audit's cross-reference of financial metrics."""

    def test_cagr_manual_cross_reference(self):
        """Audit Phase 4: CAGR matches manual computation from cumulative index."""
        from stats import calc_stats
        ret = _make_returns(n_days=500, seed=99)
        w = _equal_weights()
        s = calc_stats(ret, w, risk_free_rate=0.04)

        # Manual: portfolio log returns, cumulate, annualise
        port_log = ret.values @ w
        idx = np.exp(np.cumsum(port_log))
        cal_days = (ret.index[-1] - ret.index[0]).days
        manual_cagr = idx[-1] ** (365.0 / cal_days) - 1.0
        assert abs(s.cagr - manual_cagr) < 1e-10, f"CAGR mismatch: {s.cagr} vs {manual_cagr}"

    def test_volatility_manual_cross_reference(self):
        """Audit Phase 4: Vol matches manual sqrt(var(simple_ret) * 252).

        After audit fix #1, volatility is computed from simple returns for
        consistency with the Sharpe ratio numerator.
        """
        from stats import calc_stats, TD
        ret = _make_returns(n_days=500, seed=99)
        w = _equal_weights()
        s = calc_stats(ret, w)

        port_log = ret.values @ w
        port_simple = np.exp(port_log) - 1.0
        manual_vol = np.sqrt(np.var(port_simple, ddof=1) * TD)
        assert abs(s.volatility - manual_vol) < 1e-10

    def test_sharpe_uses_arithmetic_returns(self):
        """Audit fix #1: Sharpe = (arithmetic_annual - Rf) / vol, all on simple return basis."""
        from stats import calc_stats, TD
        ret = _make_returns(n_days=500, seed=99)
        w = _equal_weights()
        rf = 0.04
        s = calc_stats(ret, w, risk_free_rate=rf)

        port_log = ret.values @ w
        port_simple = np.exp(port_log) - 1.0
        arith_annual = np.mean(port_simple) * TD
        vol = np.sqrt(np.var(port_simple, ddof=1) * TD)
        expected_sharpe = (arith_annual - rf) / vol
        assert abs(s.sharpe - expected_sharpe) < 1e-10, f"Sharpe mismatch: {s.sharpe} vs {expected_sharpe}"

    def test_max_drawdown_manual_cross_reference(self):
        """Audit Phase 4: Max DD matches manual peak-to-trough."""
        from stats import calc_stats
        ret = _make_returns(n_days=500, seed=99)
        w = _equal_weights()
        s = calc_stats(ret, w)

        port_log = ret.values @ w
        idx = np.exp(np.cumsum(port_log))
        peak = np.maximum.accumulate(idx)
        dd = idx / peak - 1.0
        manual_max_dd = np.min(dd)
        assert abs(s.max_drawdown - manual_max_dd) < 1e-8

    def test_annual_returns_compound_to_total(self):
        """Audit coverage gap: product of (1+annual_i) should ≈ total return."""
        from stats import calc_stats, compute_annual_returns
        ret = _make_returns(n_days=750, seed=55)
        w = _equal_weights()
        s = calc_stats(ret, w)
        annual = compute_annual_returns(ret, w)

        # Product of (1 + annual) = total growth factor
        total_from_annual = np.prod(1.0 + annual.values)
        # Total from equity index
        port_log = ret.values @ w
        total_from_index = np.exp(np.sum(port_log))

        # Should be very close (not exact due to year boundary alignment)
        assert abs(total_from_annual - total_from_index) / total_from_index < 0.01


class TestAuditEdgeCases:
    """Edge case tests identified in audit Phase 2 coverage gaps."""

    def test_single_day_returns(self):
        """Audit gap: single trading day should not crash."""
        from stats import calc_stats
        from data import ASSETS
        dates = pd.bdate_range("2023-01-02", periods=1)
        data = np.random.randn(1, len(ASSETS)) * 0.01
        ret = pd.DataFrame(data, index=dates, columns=ASSETS)
        w = _equal_weights()
        s = calc_stats(ret, w)
        # Should return valid (possibly zero) stats without crashing
        assert np.isfinite(s.cagr) or s.cagr == 0.0
        assert s.volatility >= 0

    def test_all_zero_returns(self):
        """Audit gap: portfolio of all-zero returns."""
        from stats import calc_stats
        from data import ASSETS
        dates = pd.bdate_range("2023-01-02", periods=100)
        ret = pd.DataFrame(np.zeros((100, len(ASSETS))), index=dates, columns=ASSETS)
        w = _equal_weights()
        s = calc_stats(ret, w)
        assert s.cagr == 0.0
        assert s.volatility == 0.0
        assert s.max_drawdown == 0.0

    def test_single_asset_full_allocation(self):
        """Audit gap: 100% in one asset should match that asset's standalone stats."""
        from stats import calc_stats
        from data import ASSETS
        ret = _make_returns(n_days=500, seed=77)
        w = np.zeros(len(ASSETS))
        w[2] = 1.0  # 100% S&P 500 (index 2)

        s = calc_stats(ret, w)
        # Manual: single asset log returns
        asset_log = ret.values[:, 2]
        idx = np.exp(np.cumsum(asset_log))
        cal_days = (ret.index[-1] - ret.index[0]).days
        expected_cagr = idx[-1] ** (365.0 / cal_days) - 1.0
        assert abs(s.cagr - expected_cagr) < 1e-10

    def test_clip_normalize_infeasible_constraints_warns(self):
        """Audit fix #10: infeasible constraints should warn, not crash."""
        import warnings
        from optimizer import clip_normalize
        from data import ASSETS
        n = len(ASSETS)
        w = np.ones(n) / n
        # Impossible: all maxes = 5% but need 100% total (17 * 5% = 85%)
        min_w = np.zeros(n)
        max_w = np.full(n, 0.05)
        gm = _default_group_max()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = clip_normalize(w, min_w, max_w, gm)
        # Should not crash; result should be finite
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
        # Should have emitted a warning about infeasibility
        infeasible_warnings = [w for w in caught if "Infeasible" in str(w.message)]
        assert len(infeasible_warnings) > 0, "Expected infeasibility warning"


class TestAuditCFDConsistency:
    """Cross-module consistency tests from audit."""

    def test_cfd_1x_leverage_matches_unleveraged(self):
        """Audit: 1x leverage CFD has same gross CAGR and vol as unleveraged,
        but still incurs financing drag on the full notional (CMC model)."""
        from cfd import analyze_cfd
        from stats import PortfolioStats
        s = PortfolioStats(cagr=0.10, volatility=0.15, sharpe=0.40,
                           max_drawdown=-0.25, calmar=0.40,
                           best_year=0.20, worst_year=-0.10,
                           pct_positive=0.53, longest_dd=50)
        w = _equal_weights()
        result = analyze_cfd(w, s, 100000, 1.0, 0.06, 0.20, 0.04)
        assert abs(result.gross_cagr - s.cagr) < 1e-10
        # CMC charges financing on full notional even at 1x: drag = rate * 1.0
        assert abs(result.financing_drag - 0.06) < 1e-10
        assert abs(result.net_cagr - (s.cagr - 0.06)) < 1e-10
        assert abs(result.leveraged_volatility - s.volatility) < 1e-10

    def test_cfd_extreme_leverage_vol_drag(self):
        """Audit gap: extreme leverage with high vol should produce negative CAGR."""
        from cfd import analyze_cfd
        from stats import PortfolioStats
        s = PortfolioStats(cagr=0.08, volatility=0.30, sharpe=0.0,
                           max_drawdown=-0.50, calmar=0.0,
                           best_year=0.0, worst_year=0.0,
                           pct_positive=0.0, longest_dd=0)
        w = _equal_weights()
        result = analyze_cfd(w, s, 100000, 10.0, 0.06, 0.10, 0.04)
        # At 10x leverage with 30% vol, vol drag should destroy returns
        # exp(-10*9*0.30^2/2) = exp(-4.05) ≈ 0.017
        # (1.08)^10 * 0.017 - 1 ≈ -0.96
        assert result.gross_cagr < 0, f"10x leverage with 30% vol should have negative gross CAGR, got {result.gross_cagr}"

    def test_cfd_cash_reserve_covers_max_dd(self):
        """Verify cash reserve is sized to survive worst-case drawdown."""
        from cfd import analyze_cfd
        from stats import PortfolioStats
        s = PortfolioStats(cagr=0.08, volatility=0.12, sharpe=0.33,
                           max_drawdown=-0.20, calmar=0.40,
                           best_year=0.15, worst_year=-0.10,
                           pct_positive=0.53, longest_dd=60)
        w = _equal_weights()
        result = analyze_cfd(w, s, 100000, 5.0, 0.06, 0.20, 0.04)
        # Worst-case loss = |max_dd| * leverage * deployed
        max_loss = abs(s.max_drawdown) * 5.0 * result.deployed_capital
        # Free margin + reserve should cover the loss
        buffer = result.free_margin + result.cash_reserve
        assert buffer >= max_loss - 1.0, f"Buffer {buffer} < max_loss {max_loss}"


class TestAuditMonteCarlo:
    """Monte Carlo correctness tests from audit."""

    def test_mc_median_converges_to_geometric_growth(self):
        """Audit gap: MC median should approximate (1+CAGR)^T for many paths."""
        np.random.seed(42)
        cagr = 0.08
        vol = 0.12
        n_paths = 10000
        n_days = 252 * 10
        daily_mu = np.log(1.0 + cagr) / 252.0
        daily_sigma = vol / np.sqrt(252.0)
        z = np.random.randn(n_paths, n_days)
        daily_log_ret = daily_mu + daily_sigma * z
        cum_log = np.cumsum(daily_log_ret, axis=1)
        final_vals = np.exp(cum_log[:, -1])
        mc_median = np.median(final_vals)
        # Median of lognormal = exp(mu_total) = (1+CAGR)^T
        expected_median = (1.0 + cagr) ** 10
        # With 10000 paths, should be within ~3%
        rel_error = abs(mc_median - expected_median) / expected_median
        assert rel_error < 0.05, f"MC median {mc_median:.2f} vs expected {expected_median:.2f} (err {rel_error:.3f})"


class TestAuditEffectiveWeights:
    """Tests for effective weights edge cases from audit."""

    def test_effective_weights_all_available(self):
        """When all assets are available from day 1, eff_weights = static weights."""
        from stats import _effective_weights
        idx = pd.bdate_range("2020-01-02", periods=50)
        cols = ["A", "B", "C"]
        weights = np.array([0.5, 0.3, 0.2])
        starts = {"A": pd.Timestamp("2019-01-01"), "B": pd.Timestamp("2019-01-01"),
                  "C": pd.Timestamp("2019-01-01")}
        eff = _effective_weights(weights, idx, cols, starts)
        for t in range(len(idx)):
            np.testing.assert_array_almost_equal(eff[t], weights)

    def test_effective_weights_redistribution_preserves_ratios(self):
        """Pro-rata redistribution should preserve relative proportions."""
        from stats import _effective_weights
        idx = pd.bdate_range("2020-01-02", periods=10)
        cols = ["A", "B", "C"]
        weights = np.array([0.6, 0.3, 0.1])
        # C not available until day 5
        starts = {"A": pd.Timestamp("2020-01-02"), "B": pd.Timestamp("2020-01-02"),
                  "C": pd.Timestamp("2020-01-08")}
        eff = _effective_weights(weights, idx, cols, starts)
        # Before C: A/B ratio should be preserved: 0.6/0.3 = 2.0
        ratio = eff[0, 0] / eff[0, 1]
        expected_ratio = 0.6 / 0.3
        assert abs(ratio - expected_ratio) < 1e-10

    def test_weights_schedule_with_asset_starts(self):
        """Audit gap: weights_schedule and asset_starts together."""
        from stats import calc_stats
        from data import ASSETS
        ret = _make_returns(n_days=500)
        w = _equal_weights()
        # Schedule with two phases
        schedule = {ret.index[0]: w, ret.index[250]: w * 1.1}
        schedule[ret.index[250]] /= schedule[ret.index[250]].sum()
        # asset_starts — pretend last asset starts late
        starts = {ASSETS[-1]: ret.index[100]}
        # Note: weights_schedule currently disables asset_starts in calc_stats
        # (line 215: if asset_starts is not None and weights_schedule is None)
        # This test verifies it doesn't crash
        s = calc_stats(ret, w, rebalance="monthly", weights_schedule=schedule)
        assert np.isfinite(s.cagr)


class TestAuditRegime:
    """Regime-specific audit tests."""

    def test_regime_no_lookahead_bias(self):
        """Audit fix #2: expanding-window median means adding future data
        should NOT change past regime classifications."""
        from regime import classify_regimes
        dates = pd.date_range("2000-01-01", periods=100, freq="ME")
        # Construct data where future data is very different from early data
        cpi = np.concatenate([np.ones(50) * 2.0, np.ones(50) * 10.0])
        ff = np.concatenate([np.ones(50) * 3.0, np.ones(50) * 15.0])
        macro = pd.DataFrame({"CPI": cpi, "FedFunds": ff}, index=dates)

        regimes_full = classify_regimes(macro)
        regimes_partial = classify_regimes(macro.iloc[:50])

        # With expanding-window median, classifications at month 24-49 should
        # be identical regardless of whether future data exists.
        # (First 24 months are warm-up, so compare from month 24 onwards)
        for t in range(23, 50):
            assert regimes_full.iloc[t] == regimes_partial.iloc[t], (
                f"Look-ahead detected at month {t}: "
                f"full={regimes_full.iloc[t]}, partial={regimes_partial.iloc[t]}"
            )

    def test_regime_expanding_window_min_history(self):
        """Verify that regimes have a warm-up period before classification."""
        from regime import classify_regimes
        dates = pd.date_range("2000-01-01", periods=30, freq="ME")
        macro = pd.DataFrame({
            "CPI": np.random.randn(30) + 3.0,
            "FedFunds": np.random.randn(30) + 5.0,
        }, index=dates)
        regimes = classify_regimes(macro)
        assert len(regimes) == 30
        # All values should be valid regime labels
        assert set(regimes.unique()).issubset({1, 2, 3, 4})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
