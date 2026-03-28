"""Data loading, alignment, and log return computation."""

import warnings
import numpy as np
import pandas as pd
import shutil
import tempfile
from pathlib import Path


ASSETS = [
    "Cash", "Nasdaq", "S&P 500", "Russell 2000", "ASX200",
    "Emerging Markets", "Corporate Bonds", "Long-Term Treasuries",
    "Short-Term Treasuries", "US REITs", "Industrial Metals", "Gold",
    "Bitcoin", "Infrastructure", "Japan Equities", "UK Equities", "EU Equities",
    "US TIPS", "High Yield", "EM Debt",
    "JPY", "CHF", "CNY",
    "China Equities", "Copper", "Soft Commodities",
]

GROUP_MAP = {
    "Cash": "Alternatives",
    "Nasdaq": "US Equities",
    "S&P 500": "US Equities",
    "Russell 2000": "US Equities",
    "ASX200": "Intl Equities",
    "Emerging Markets": "Intl Equities",
    "China Equities": "Intl Equities",
    "Corporate Bonds": "Bonds",
    "Long-Term Treasuries": "Bonds",
    "Short-Term Treasuries": "Bonds",
    "US TIPS": "Bonds",
    "High Yield": "Bonds",
    "EM Debt": "Bonds",
    "US REITs": "Real Assets",
    "Industrial Metals": "Real Assets",
    "Gold": "Real Assets",
    "Copper": "Real Assets",
    "Soft Commodities": "Real Assets",
    "Infrastructure": "Real Assets",
    "Bitcoin": "Alternatives",
    "Japan Equities": "Intl Equities",
    "UK Equities": "Intl Equities",
    "EU Equities": "Intl Equities",
    "JPY": "Currencies",
    "CHF": "Currencies",
    "CNY": "Currencies",
}

GROUP_NAMES = ["US Equities", "Intl Equities", "Bonds", "Real Assets", "Alternatives", "Currencies"]


def load_prices_from_excel(path: str) -> dict[str, pd.Series]:
    """Load raw price data from the Data sheet.

    Expected format: pairs of (Date, Value) columns per asset.
    Row 1: asset names, Row 2: tickers, Row 3: 'Date'/'Value' headers, Row 4+: data.
    """
    df = pd.read_excel(path, sheet_name="Data", header=None)
    prices = {}
    col = 0
    while col < df.shape[1]:
        name = df.iloc[0, col]
        if pd.isna(name) or str(name).strip() == "":
            col += 1
            continue
        name = str(name).strip()
        if name not in ASSETS:
            col += 1
            continue
        date_col = col
        val_col = col + 1
        if val_col >= df.shape[1]:
            break
        dates = pd.to_datetime(df.iloc[3:, date_col], errors="coerce")
        vals = pd.to_numeric(df.iloc[3:, val_col], errors="coerce")
        mask = dates.notna() & vals.notna()
        s = pd.Series(vals[mask].values, index=dates[mask].values, name=name)
        s = s[~s.index.duplicated(keep="first")].sort_index()
        prices[name] = s
        col += 2
    return prices


def load_returns_from_processing(path: str) -> pd.DataFrame:
    """Load pre-computed log returns from the Processing sheet."""
    df = pd.read_excel(path, sheet_name="Processing", header=0)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    cols = [c for c in ASSETS if c in df.columns]
    returns = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return _sanitize_cash_returns(returns)


def _sanitize_cash_returns(returns: pd.DataFrame, max_daily_abs: float = 0.005) -> pd.DataFrame:
    """Cap Cash daily log returns to realistic money-market levels.

    Money market instruments typically have annualized vol of ~0.5-2%.
    Daily returns beyond +-0.5% are almost certainly data artifacts.
    """
    if "Cash" in returns.columns:
        raw_cash_std = returns["Cash"].std()
        if raw_cash_std > 0.01:  # unrealistically high daily vol
            n_capped = ((returns["Cash"].abs() > max_daily_abs)).sum()
            warnings.warn(
                f"Cash returns sanitized: daily vol ({raw_cash_std:.4f}) exceeded threshold. "
                f"{n_capped} values capped to +/-{max_daily_abs:.3f}.",
                stacklevel=2,
            )
            returns = returns.copy()
            returns["Cash"] = returns["Cash"].clip(lower=-max_daily_abs, upper=max_daily_abs)
    return returns


def align_and_compute_returns(prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Align prices to a common daily grid and compute log returns."""
    if not prices:
        return pd.DataFrame()
    all_dates = pd.DatetimeIndex(sorted(set().union(*(s.index for s in prices.values()))))
    aligned = pd.DataFrame(index=all_dates)
    for name in ASSETS:
        if name in prices:
            aligned[name] = prices[name].reindex(all_dates).ffill()
        else:
            aligned[name] = np.nan
    log_returns = np.log(aligned / aligned.shift(1))
    log_returns = log_returns.iloc[1:]
    log_returns = log_returns.fillna(0.0)
    return _sanitize_cash_returns(log_returns)


def _safe_open(path: str) -> str:
    """If the file is locked (e.g. open in Excel), try a sibling copy or temp copy."""
    try:
        with open(path, "rb") as f:
            f.read(1)  # test read
        return path
    except PermissionError:
        # Try a sibling copy with _copy suffix
        p = Path(path)
        copy_path = p.parent / f"{p.stem}_copy{p.suffix}"
        if copy_path.exists():
            return str(copy_path)
        # Try temp copy (may also fail if source is locked)
        try:
            tmp = Path(tempfile.gettempdir()) / f"_awp_{p.name}"
            shutil.copy2(path, tmp)
            return str(tmp)
        except PermissionError:
            raise PermissionError(
                f"Cannot read '{p.name}' — it appears to be open in Excel. "
                f"Either close Excel or place a copy named '{copy_path.name}' in the same folder."
            )


def load_data(path: str, use_processing: bool = True) -> pd.DataFrame:
    """Load returns data from Excel file.

    If use_processing=True and a Processing sheet exists, reads pre-computed returns.
    Otherwise reads from Data sheet and computes returns.
    """
    safe_path = _safe_open(path)
    if use_processing:
        try:
            return load_returns_from_processing(safe_path)
        except Exception:
            pass
    prices = load_prices_from_excel(safe_path)
    return align_and_compute_returns(prices)
