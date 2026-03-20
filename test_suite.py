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
        from stats import calc_stats
        ret = _make_returns()
        w = _equal_weights()
        mid = ret.index[len(ret) // 2]
        s1 = calc_stats(ret, w, start_date=mid)
        s2 = calc_stats(ret, w)
        # s1 covers fewer days -> generally different stats
        assert s1.cagr != s2.cagr or s1.volatility != s2.volatility

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

    def test_risk_parity(self):
        from optimizer import risk_parity
        ret = _make_returns()
        min_w, max_w = _default_bounds()
        gm = _default_group_max()
        w = risk_parity(ret, min_w, max_w, gm)
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
        for target in ["Risk Parity", "Equal Risk Contribution", "Hierarchical Risk Parity"]:
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
        # Financing drag = rate * (leverage - 1)
        expected_drag = 0.06 * 2.0
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
        expected_net = result.gross_cagr - 0.06 * 2.0
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
        # With 1x leverage, financing drag should be 0 and CAGR unchanged
        # (1+CAGR)^1 * exp(0) - 1 = CAGR
        assert abs(result.financing_drag) < 1e-6
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
        w = run_optimization(ret, "Risk Parity", min_w, max_w, gm)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
