# Future Enhancements

## Time-Varying Strategy Variants
- [ ] Max Sharpe (Time-Varying) — re-optimises annually using a rolling window, same framework as regime-based and DD p-value
- [ ] Leverage-Optimal (Time-Varying) — re-optimises the post-leverage Sharpe objective annually, adapting to changing vol/correlation regimes

## Performance / UX
- [ ] Create a simplified "fast mode" version that only runs 3-4 core portfolios for quicker iteration
- [ ] When building simplified portfolio, remove any asset with ≤0.5% allocation to create a super simplified version

---

# Done

## Asset Universe
- [x] Add new assets: TIPS, JPY, High-Yield, EM Debt, CHF, China Equities, CNY, Copper, Industrial Metals, Soft Commodities
- [x] Replace Commodities (BCOM) with Industrial Metals (BCOMINTR Index)
- [x] Replace Real Estate (ENXG) with US REITs (FNERTR Index)
- [x] Update LT Treasuries to I00094US Index (30Y), ST Treasuries to LUATTRUU Index (2Y)
- [x] Update Infrastructure to SPGTINTR Index
- [x] Check which assets are available as CFDs on CMC Markets
- [x] Apply ~18% dividend/coupon haircut to total return indices for CFD tax drag (CFD tab only)
- [x] Find optimal backtest start date — set to 1983-08-31 (when 50% of assets available)
