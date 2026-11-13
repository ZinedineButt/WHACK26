"""Typed variable registry. The AI may only pick variables from here (or a
syntactically valid ticker that the market-data provider actually serves).

Timing convention (documented in the API docs): a daily observation dated t is
treated as a market close that is known at the close of t. FRED publication
delays are NOT modelled for market-based series, and monthly macro series are
excluded from predictive/causal tests (no point-in-time vintages implemented).
"""
import re
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")

# transform ids
LOG_RETURN = "log_return"       # 100*diff(log price)                         (price)
DIFF = "diff"                   # first difference in native units (pp)       (rate / spread)
LOG_DIFF = "log_diff"           # 100*diff(log level)                         (index e.g. VIX)
LEVEL = "level"                 # raw level (stationary-looking series only)  (index / spread)
FWD_LOGVOL_5 = "fwd_logvol_5"   # log annualised realised vol of returns t+1..t+5 (outcome only)
FWD_LOGVOL_10 = "fwd_logvol_10"
TRAILING_LOGVOL_5 = "trailing_logvol_5"   # control only
FORWARD_OUTCOMES = {FWD_LOGVOL_5: 5, FWD_LOGVOL_10: 10}

KIND_TRANSFORMS = {
    "price": (LOG_RETURN, FWD_LOGVOL_5, FWD_LOGVOL_10),
    "rate": (DIFF,),
    "spread": (DIFF, LEVEL),
    "index": (LOG_DIFF, LEVEL),
    "macro_monthly": (),
}
DEFAULT_TRANSFORM = {"price": LOG_RETURN, "rate": DIFF, "spread": DIFF, "index": LOG_DIFF}


@dataclass(frozen=True)
class Variable:
    id: str
    source: str                 # fred | equity | fixture
    label: str
    units: str
    frequency: str
    kind: str
    category: str
    description: str
    supported: bool = True
    unsupported_reason: Optional[str] = None
    release_lag: str = "known at the close of the observation date"

    @property
    def transforms(self) -> Tuple[str, ...]:
        return KIND_TRANSFORMS[self.kind] if self.supported else ()

    @property
    def default_transform(self):
        return DEFAULT_TRANSFORM.get(self.kind)

    def public(self):
        d = asdict(self)
        d["transforms"] = list(self.transforms)
        d["default_transform"] = self.default_transform
        return d


def _fred(id_, label, units, kind, category, desc, **kw):
    return Variable(id_, "fred", label, units, "daily", kind, category, desc, **kw)


FRED_VARIABLES = [
    _fred("DGS10", "10-year Treasury yield", "percent", "rate", "rates", "US 10-year constant-maturity Treasury yield (H.15)."),
    _fred("DGS2", "2-year Treasury yield", "percent", "rate", "rates", "US 2-year constant-maturity Treasury yield (H.15)."),
    _fred("DFF", "Effective federal funds rate", "percent", "rate", "rates", "Daily effective fed funds rate."),
    _fred("T10Y2Y", "10y-2y Treasury spread", "percentage points", "spread", "rates", "Yield-curve slope."),
    _fred("BAMLH0A0HYM2", "US high-yield credit spread", "percent", "spread", "credit", "ICE BofA US High Yield option-adjusted spread."),
    _fred("VIXCLS", "CBOE VIX", "index points", "index", "volatility", "Implied volatility of S&P 500 options."),
    _fred("DCOILWTICO", "WTI crude oil price", "USD/barrel", "price", "commodities", "West Texas Intermediate spot price."),
    _fred("DCOILBRENTEU", "Brent crude oil price", "USD/barrel", "price", "commodities", "Europe Brent spot price."),
    _fred("DTWEXBGS", "Broad US dollar index", "index", "price", "fx", "Trade-weighted broad dollar index (daily)."),
    _fred("NASDAQ100", "NASDAQ-100 index", "index points", "price", "equity_index", "NASDAQ-100 price index (FRED)."),
    _fred("VXNCLS", "CBOE NASDAQ-100 volatility (VXN)", "index points", "index", "volatility", "Implied volatility of NASDAQ-100 options."),
    _fred("OVXCLS", "CBOE crude oil volatility (OVX)", "index points", "index", "volatility", "Implied volatility of oil ETF options."),
    _fred("GVZCLS", "CBOE gold volatility (GVZ)", "index points", "index", "volatility", "Implied volatility of gold ETF options."),
    _fred("DHHNGSP", "Henry Hub natural gas price", "USD/MMBtu", "price", "commodities", "US natural gas spot price."),
    _fred("DEXUSEU", "USD per euro", "USD", "price", "fx", "EUR/USD exchange rate (USD per 1 EUR)."),
    _fred("DEXJPUS", "Japanese yen per USD", "JPY", "price", "fx", "USD/JPY exchange rate (JPY per 1 USD)."),
    _fred("DTWEXEMEGS", "Emerging-markets dollar index", "index", "price", "fx", "Trade-weighted dollar vs emerging-market economies."),
    _fred("SP500", "S&P 500 index", "index points", "price", "equity_index", "S&P 500 price index (FRED, price return, no dividends)."),
    _fred("NASDAQCOM", "NASDAQ Composite", "index points", "price", "equity_index", "NASDAQ Composite price index."),
    _fred("DJIA", "Dow Jones Industrial Average", "index points", "price", "equity_index", "DJIA price index."),
    _fred("CPIAUCSL", "CPI (all urban consumers)", "index", "macro_monthly", "inflation",
          "Monthly CPI.", supported=False,
          unsupported_reason="Monthly series needs release-date-aware (ALFRED vintage) data, which is not implemented; excluded from predictive/causal tests."),
    _fred("UNRATE", "Unemployment rate", "percent", "macro_monthly", "labour",
          "Monthly unemployment rate.", supported=False,
          unsupported_reason="Monthly series needs release-date-aware (ALFRED vintage) data, which is not implemented; excluded from predictive/causal tests."),
]

# Seed list shown in the catalogue; ANY valid ticker served by the provider is accepted.
SUGGESTED_TICKERS = {
    "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ (Nasdaq-100) ETF", "IWM": "iShares Russell 2000 ETF",
    "DIA": "SPDR Dow Jones ETF", "XLK": "Technology Select Sector ETF", "XLE": "Energy Select Sector ETF",
    "XLF": "Financial Select Sector ETF", "JETS": "US Global Jets (airline) ETF", "TLT": "iShares 20+ Year Treasury ETF",
    "GLD": "SPDR Gold Shares", "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta", "TSLA": "Tesla", "AMD": "AMD", "XOM": "Exxon Mobil", "JPM": "JPMorgan",
    "DAL": "Delta Air Lines", "UAL": "United Airlines",
    "SMH": "VanEck Semiconductor ETF", "SOXX": "iShares Semiconductor ETF", "BOTZ": "Global X Robotics & AI ETF", "AIQ": "Global X AI & Technology ETF",
    "AVGO": "Broadcom", "TSM": "TSMC", "ASML": "ASML", "ORCL": "Oracle", "PLTR": "Palantir", "SMCI": "Super Micro Computer",
}

FIXTURE_VARIABLES = [
    Variable("SYN_OIL", "fixture", "Synthetic oil price", "USD", "daily", "price", "synthetic", "SYNTHETIC test series."),
    Variable("SYN_AIR", "fixture", "Synthetic airline index", "index", "daily", "price", "synthetic",
             "SYNTHETIC: planted lag-1 effect from SYN_OIL returns."),
    Variable("SYN_AIR2", "fixture", "Synthetic airline index 2", "index", "daily", "price", "synthetic",
             "SYNTHETIC peer: same planted lag-1 effect from SYN_OIL as SYN_AIR."),
    Variable("SYN_MKT", "fixture", "Synthetic market index", "index", "daily", "price", "synthetic", "SYNTHETIC market factor."),
    Variable("SYN_RATE", "fixture", "Synthetic 10y yield", "percent", "daily", "rate", "synthetic", "SYNTHETIC; unrelated to SYN_TECH."),
    Variable("SYN_TECH", "fixture", "Synthetic tech index", "index", "daily", "price", "synthetic", "SYNTHETIC; loads on SYN_MKT only."),
    Variable("SYN_CONF_X", "fixture", "Synthetic bank index", "index", "daily", "price", "synthetic", "SYNTHETIC; loads on market at t."),
    Variable("SYN_CONF_Y", "fixture", "Synthetic insurer index", "index", "daily", "price", "synthetic",
             "SYNTHETIC; loads on market at t-1 (confounded with SYN_CONF_X)."),
]


class Registry:
    """Availability depends on which providers are configured right now."""

    def __init__(self, fred_available=True, equity_available=False, equity_reason=None):
        self.fred_available, self.equity_available = fred_available, equity_available
        self.equity_reason = equity_reason or ("No stock/ETF data provider is configured, so tickers cannot be fetched "
                                               "(set TIINGO_API_KEY; Alpaca keys also work)")
        self._by_id = {v.id: v for v in FRED_VARIABLES + FIXTURE_VARIABLES}

    def resolve(self, vid):
        """Return (Variable|None, reason|None). Never raises."""
        if not isinstance(vid, str):
            return None, "variable id must be a string"
        vid = vid.strip().upper()
        v = self._by_id.get(vid)
        if v is not None:
            return v, self.unavailable_reason(v)
        if TICKER_RE.match(vid):
            if not self.equity_available:
                return None, self.equity_reason
            name = SUGGESTED_TICKERS.get(vid, vid)
            return Variable(vid, "equity", name, "USD", "daily", "price", "equity",
                            f"{name} split/dividend-adjusted daily close from Alpaca."), None
        return None, f"unknown variable '{vid}': not in the catalogue and not a valid ticker"

    def unavailable_reason(self, v):
        if not v.supported:
            return v.unsupported_reason
        if v.source == "fred" and not self.fred_available:
            return "FRED unavailable"
        if v.source == "equity" and not self.equity_available:
            return self.equity_reason
        return None

    def live_catalog(self):
        """Variables Gemini may choose from in live modes (no fixtures)."""
        out = [v for v in FRED_VARIABLES if v.supported and self.unavailable_reason(v) is None]
        if self.equity_available:
            out += [self.resolve(t)[0] for t in SUGGESTED_TICKERS]
        return out

    def fixture_catalog(self):
        return list(FIXTURE_VARIABLES)

    def public_catalog(self):
        rows = []
        for v in FRED_VARIABLES:
            r = v.public(); r["available"] = self.unavailable_reason(v) is None
            r["unavailable_reason"] = self.unavailable_reason(v); rows.append(r)
        for t in SUGGESTED_TICKERS:
            v, why = self.resolve(t)
            if v is None:
                v = Variable(t, "equity", SUGGESTED_TICKERS[t], "USD", "daily", "price", "equity", "")
                why = self.unavailable_reason(v)
            r = v.public(); r["available"] = why is None; r["unavailable_reason"] = why; rows.append(r)
        return rows
