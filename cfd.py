"""CFD leverage analysis for portfolio positions."""

import numpy as np
from dataclasses import dataclass

from data import ASSETS
from stats import PortfolioStats

# Estimated annual dividend / coupon yield per asset (as a fraction).
# Used to compute the tax drag from CFD dividend adjustments, which are
# paid as cash (taxed as ordinary income) rather than reinvested gross.
# Yields are approximate long-run averages — the haircut captures the
# incremental cost vs. a gross total-return index.
ASSET_DIVIDEND_YIELDS = {
    "Cash":                 0.000,
    "Nasdaq":               0.008,
    "S&P 500":              0.015,
    "Russell 2000":         0.015,
    "ASX200":               0.035,
    "Emerging Markets":     0.025,
    "Corporate Bonds":      0.045,
    "Long-Term Treasuries": 0.040,
    "Short-Term Treasuries":0.045,
    "US REITs":             0.040,
    "Industrial Metals":    0.000,
    "Gold":                 0.000,
    "Bitcoin":              0.000,
    "Infrastructure":       0.035,
    "Japan Equities":       0.020,
    "UK Equities":          0.035,
    "EU Equities":          0.025,
    "US TIPS":              0.025,
    "High Yield":           0.060,
    "EM Debt":              0.050,
    "JPY":                  0.000,
    "CHF":                  0.000,
    "CNY":                  0.000,
    "China Equities":       0.020,
    "Copper":               0.000,
    "Soft Commodities":     0.000,
}

CFD_DIVIDEND_TAX_RATE = 0.18  # marginal tax rate applied to dividend adjustments


def portfolio_dividend_drag(weights: np.ndarray, tax_rate: float = CFD_DIVIDEND_TAX_RATE) -> float:
    """Return the annual CAGR drag from taxed dividend adjustments (unleveraged).

    drag = sum_i( weight_i * yield_i ) * tax_rate
    """
    total_yield = sum(
        weights[i] * ASSET_DIVIDEND_YIELDS.get(asset, 0.0)
        for i, asset in enumerate(ASSETS)
    )
    return total_yield * tax_rate


@dataclass
class CFDAnalysis:
    """Results of CFD leverage analysis."""
    # Capital allocation
    total_capital: float              # total capital available (user input)
    cash_reserve: float               # cash held back as liquidation buffer
    deployed_capital: float           # capital actually deployed (total - reserve)
    notional_exposure: float          # total leveraged position size
    capital_per_asset: dict[str, float]  # notional $ per asset
    margin_required: float            # total margin needed
    free_margin: float                # deployed capital minus margin

    # Risk metrics
    max_drawdown_dollars: float       # absolute $ drawdown at max historical DD
    margin_utilisation: float         # margin / deployed capital as %

    # Return metrics (after financing)
    gross_cagr: float                 # leveraged CAGR before financing or dividend drag
    dividend_drag: float              # annual drag from taxed dividend adjustments (leveraged)
    financing_drag: float             # annual financing cost
    net_cagr: float                   # CAGR after all drags on deployed capital
    leveraged_volatility: float       # vol scaled by leverage
    net_sharpe: float                 # Sharpe after all costs

    # Capital-adjusted return (CAGR relative to total capital including reserve)
    effective_cagr: float             # net return as % of total capital


def analyze_cfd(
    weights: np.ndarray,
    stats: PortfolioStats,
    total_capital: float,
    leverage_ratio: float,
    financing_rate: float,
    margin_requirement: float,
    risk_free_rate: float,
) -> CFDAnalysis:
    """Analyse a portfolio under CFD leverage.

    Parameters
    ----------
    weights : array of portfolio weights (sum to 1)
    stats : unleveraged portfolio statistics
    total_capital : total capital available (some will be held as cash reserve)
    leverage_ratio : e.g. 5.0 for 5x leverage
    financing_rate : annual interest charged on borrowed notional (e.g. 0.06)
    margin_requirement : fraction of notional required as margin (e.g. 0.20)
    risk_free_rate : for Sharpe calculation
    """
    # Figure out how much capital to deploy vs hold as cash reserve.
    # The cash reserve covers the worst-case leveraged drawdown so that
    # a margin call is avoided.
    #
    # Let D = deployed, R = reserve, T = total = D + R
    # Notional = D * leverage
    # Margin = Notional * margin_req = D * leverage * margin_req
    # Free margin = D - Margin = D * (1 - leverage * margin_req)
    # Max DD$ = abs(max_dd) * leverage * D
    # Reserve needed = Max DD$ - Free margin
    #   = D * (abs(max_dd)*leverage - 1 + leverage*margin_req)
    #   = D * k   where k = abs(max_dd)*leverage - 1 + leverage*margin_req
    # Since D + R = T and R = D*k:
    #   D(1 + k) = T  =>  D = T / (1 + k)

    abs_dd = abs(stats.max_drawdown)
    k = abs_dd * leverage_ratio - 1.0 + leverage_ratio * margin_requirement
    if k > 0:
        deployed_capital = total_capital / (1.0 + k)
        cash_reserve = total_capital - deployed_capital
    else:
        deployed_capital = total_capital
        cash_reserve = 0.0

    notional = deployed_capital * leverage_ratio
    margin_req = notional * margin_requirement
    free_margin = max(deployed_capital - margin_req, 0.0)

    # Per-asset notional
    capital_per_asset = {}
    for i, asset in enumerate(ASSETS):
        capital_per_asset[asset] = weights[i] * notional

    # Dividend drag (unleveraged): reduce base CAGR before leveraging.
    # Dividend adjustments are taxed as ordinary income; the total-return
    # index assumes reinvestment gross of tax, so we apply the shortfall.
    div_drag_ul = portfolio_dividend_drag(weights)
    adjusted_cagr = stats.cagr - div_drag_ul

    # Leveraged returns — account for volatility drag of leverage.
    # Daily-rebalanced leverage scales daily returns by L, so:
    #   CAGR_L = (1+CAGR)^L * exp(-L*(L-1)*σ²/2) - 1
    # The exp term captures the excess vol drag from leverage (scales as L²).
    vol = stats.volatility
    gross_cagr = ((1.0 + adjusted_cagr) ** leverage_ratio
                  * np.exp(-leverage_ratio * (leverage_ratio - 1.0) * vol**2 / 2.0)
                  - 1.0)

    # Exact leveraged dividend drag: difference between gross CAGR with and
    # without the dividend haircut. This is more accurate than the linear
    # approximation (div_drag_ul * L) at high leverage or high CAGR.
    gross_cagr_no_drag = ((1.0 + stats.cagr) ** leverage_ratio
                          * np.exp(-leverage_ratio * (leverage_ratio - 1.0) * vol**2 / 2.0)
                          - 1.0)
    div_drag_leveraged = gross_cagr_no_drag - gross_cagr

    # CMC Markets charges financing on the FULL notional (not just borrowed portion).
    # Annual drag = financing_rate × leverage (since notional = deployed × leverage).
    # Expressed as drag on deployed capital: financing_rate × leverage.
    financing_drag = financing_rate * leverage_ratio
    net_cagr = gross_cagr - financing_drag

    leveraged_vol = stats.volatility * leverage_ratio
    net_sharpe = (net_cagr - risk_free_rate) / leveraged_vol if leveraged_vol > 1e-4 else 0.0

    # Drawdown in dollars on deployed capital
    max_dd_dollars = abs_dd * leverage_ratio * deployed_capital

    margin_util = margin_req / deployed_capital if deployed_capital > 0 else 0.0

    # Effective CAGR: net dollar return relative to total capital.
    # Cash reserve is assumed to earn the risk-free rate (e.g. money market).
    if total_capital > 0:
        effective_cagr = (net_cagr * deployed_capital
                          + risk_free_rate * cash_reserve) / total_capital
    else:
        effective_cagr = net_cagr

    return CFDAnalysis(
        total_capital=total_capital,
        cash_reserve=cash_reserve,
        deployed_capital=deployed_capital,
        notional_exposure=notional,
        capital_per_asset=capital_per_asset,
        margin_required=margin_req,
        free_margin=free_margin,
        max_drawdown_dollars=max_dd_dollars,
        margin_utilisation=margin_util,
        gross_cagr=gross_cagr,
        dividend_drag=div_drag_leveraged,
        financing_drag=financing_drag,
        net_cagr=net_cagr,
        leveraged_volatility=leveraged_vol,
        net_sharpe=net_sharpe,
        effective_cagr=effective_cagr,
    )
