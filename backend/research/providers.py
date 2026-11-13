"""Data providers behind a small adapter interface, with retry/backoff,
timeouts and an on-disk (sqlite) response cache. Another provider can be
plugged in by implementing `fetch(variable, start, end)`."""
import datetime as dt
import io
import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd
import requests

CACHE_TTL_S = int(os.environ.get("RESEARCH_CACHE_TTL", 6 * 3600))
HTTP_TIMEOUT = 20
RETRIES = 3


class ProviderError(Exception):
    pass


def _get(url, **kw):
    """GET with exponential backoff on network errors / 429 / 5xx."""
    last = None
    for i in range(RETRIES):
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                last = ProviderError(f"HTTP {r.status_code} from {url.split('?')[0]}")
            else:
                return r
        except requests.RequestException as e:
            last = ProviderError(f"network error: {type(e).__name__}")
        time.sleep(0.5 * 2 ** i)
    raise last


class FredProvider:
    name = "fred"

    def __init__(self, api_key=None):
        self.api_key = api_key if api_key is not None else os.environ.get("FRED_API_KEY", "")

    @property
    def mode(self):
        return "fred_api" if self.api_key else "fred_public_csv"

    def fetch(self, variable, start, end):
        if self.api_key:
            r = _get("https://api.stlouisfed.org/fred/series/observations",
                     params={"series_id": variable.id, "api_key": self.api_key, "file_type": "json",
                             "observation_start": start.isoformat(), "observation_end": end.isoformat()})
            if r.status_code != 200:
                raise ProviderError(f"FRED API error {r.status_code} for {variable.id}")
            obs = r.json().get("observations", [])
            s = pd.Series({pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce") for o in obs}, dtype=float)
        else:
            r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                     params={"id": variable.id, "cosd": start.isoformat(), "coed": end.isoformat()})
            if r.status_code != 200:
                raise ProviderError(f"FRED public CSV error {r.status_code} for {variable.id}")
            df = pd.read_csv(io.StringIO(r.text))
            if df.shape[1] < 2:
                raise ProviderError(f"FRED returned no data for {variable.id}")
            s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                          index=pd.to_datetime(df.iloc[:, 0]), dtype=float)
        s = s.dropna().sort_index()
        if s.empty:
            raise ProviderError(f"no observations for {variable.id} in {start}..{end}")
        return s, {"provider": self.mode, "series_id": variable.id,
                   "note": "latest-revised values; point-in-time vintages not used (valid for market-based daily series)"}


class AlpacaProvider:
    """Historical daily bars only (no streaming, no trading)."""
    name = "alpaca"
    URL = "https://data.alpaca.markets/v2/stocks/bars"

    def __init__(self, key=None, secret=None, feed=None):
        # explicit arguments (even "") are authoritative; None falls back to the environment
        self.key = key if key is not None else os.environ.get("ALPACA_API_KEY", "")
        self.secret = secret if secret is not None else (
            os.environ.get("ALPACA_API_SECRET") or os.environ.get("ALPACA_SECRET_KEY", ""))
        self.feed = (feed or os.environ.get("ALPACA_DATA_FEED") or "iex").lower()

    @property
    def configured(self):
        return bool(self.key and self.secret)

    _verified = None      # (timestamp, ok, message)
    VERIFY_TTL = 300

    def check(self):
        """Cheap authenticated probe, cached. Returns (usable, message).
        Keys that are set but rejected (revoked/rotated) are NOT usable, so the
        catalogue and the AI never offer tickers that cannot be fetched."""
        if not self.configured:
            return False, "Alpaca credentials not configured (set ALPACA_API_KEY / ALPACA_API_SECRET)"
        now = time.time()
        if self._verified and now - self._verified[0] < self.VERIFY_TTL:
            return self._verified[1], self._verified[2]
        try:
            r = requests.get(self.URL, timeout=10, headers={"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret},
                             params={"symbols": "SPY", "timeframe": "1Day", "limit": 1, "feed": self.feed,
                                     "start": (dt.date.today() - dt.timedelta(days=10)).isoformat()})
            if r.status_code in (401, 403):
                res = (False, f"Alpaca rejected the credentials (HTTP {r.status_code}); the keys are invalid, revoked or rotated")
            else:
                res = (True, "ok")
        except requests.RequestException:
            res = (True, "ok")           # network trouble is not proof the keys are bad; fetches will report it
        self._verified = (now, *res)
        return res

    @property
    def usable(self):
        return self.check()[0]

    def fetch(self, variable, start, end):
        if not self.configured:
            raise ProviderError("Alpaca credentials not configured")
        headers = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}
        rows, token = [], None
        for _ in range(50):  # pagination guard
            params = {"symbols": variable.id, "timeframe": "1Day", "start": start.isoformat(),
                      "end": end.isoformat(), "adjustment": "all", "feed": self.feed, "limit": 10000}
            if token:
                params["page_token"] = token
            r = _get(self.URL, params=params, headers=headers)
            if r.status_code in (401, 403):
                raise ProviderError("Alpaca rejected the credentials (HTTP %d)" % r.status_code)
            if r.status_code == 422 or r.status_code == 400:
                raise ProviderError(f"Alpaca does not recognise symbol {variable.id}")
            if r.status_code != 200:
                raise ProviderError(f"Alpaca error HTTP {r.status_code}")
            body = r.json()
            rows += body.get("bars", {}).get(variable.id, [])
            token = body.get("next_page_token")
            if not token:
                break
        if not rows:
            raise ProviderError(f"Alpaca returned no bars for {variable.id}")
        s = pd.Series({pd.Timestamp(b["t"]).tz_convert(None).normalize(): float(b["c"]) for b in rows}).sort_index()
        return s, {"provider": "alpaca", "series_id": variable.id, "feed": self.feed.upper(),
                   "adjustment": "all (splits + dividends)", "note": "daily close of adjusted bars"}


class TiingoProvider:
    """Tiingo end-of-day prices (free key, licensed API). Uses adjClose, which is adjusted for splits AND dividends."""
    name = "tiingo"
    BASE = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, key=None):
        self.key = key if key is not None else os.environ.get("TIINGO_API_KEY", "")

    @property
    def configured(self):
        return bool(self.key)

    _verified = None
    VERIFY_TTL = 300

    def check(self):
        if not self.configured:
            return False, "TIINGO_API_KEY is not set"
        now = time.time()
        if self._verified and now - self._verified[0] < self.VERIFY_TTL:
            return self._verified[1], self._verified[2]
        try:
            r = requests.get(f"{self.BASE}/SPY/prices", timeout=10, headers={"Authorization": f"Token {self.key}"},
                             params={"startDate": (dt.date.today() - dt.timedelta(days=10)).isoformat(), "format": "json"})
            res = (False, f"Tiingo rejected the key (HTTP {r.status_code})") if r.status_code in (401, 403) else (True, "ok")
        except requests.RequestException:
            res = (True, "ok")
        self._verified = (now, *res)
        return res

    def fetch(self, variable, start, end):
        if not self.configured:
            raise ProviderError("Tiingo key not configured")
        r = _get(f"{self.BASE}/{variable.id}/prices", headers={"Authorization": f"Token {self.key}"},
                 params={"startDate": start.isoformat(), "endDate": end.isoformat(), "format": "json", "resampleFreq": "daily"})
        if r.status_code in (401, 403):
            raise ProviderError(f"Tiingo rejected the key (HTTP {r.status_code})")
        if r.status_code == 404:
            raise ProviderError(f"Tiingo does not recognise symbol {variable.id}")
        if r.status_code != 200:
            raise ProviderError(f"Tiingo error HTTP {r.status_code}")
        rows = r.json()
        if not isinstance(rows, list) or not rows:
            raise ProviderError(f"Tiingo returned no prices for {variable.id}")
        s = pd.Series({pd.Timestamp(b["date"]).tz_localize(None).normalize(): float(b["adjClose"]) for b in rows
                       if b.get("adjClose") is not None}).sort_index()
        if s.empty:
            raise ProviderError(f"Tiingo returned no adjusted closes for {variable.id}")
        return s, {"provider": "tiingo", "series_id": variable.id, "adjustment": "splits + dividends (adjClose)",
                   "note": "end-of-day adjusted close"}


class FixtureProvider:
    """Deterministic SYNTHETIC series with known generating processes."""
    name = "fixture"
    N = 1000
    SEED = 7

    @classmethod
    def _frame(cls):
        rng = np.random.default_rng(cls.SEED)
        n = cls.N
        t = lambda: rng.standard_t(6, n) / np.sqrt(6 / 4)
        mkt, oil_r, rate_d = t(), t(), t()
        air = 0.5 * np.r_[0, oil_r[:-1]] + 0.5 * mkt + t()                  # planted lag-1 oil -> airline
        tech = 0.9 * mkt + t()                                                # no rate link
        cx = 0.8 * mkt + t()                                                  # confounded pair:
        cy = 0.8 * np.r_[0, mkt[:-1]] + t()                                   # y_t loads on market_{t-1}
        air2 = 0.5 * np.r_[0, oil_r[:-1]] + 0.5 * mkt + t()                 # second airline, same planted effect (drawn LAST)
        px = lambda r, base: base * np.exp(np.cumsum(r) / 100)
        idx = pd.bdate_range("2021-01-04", periods=n)
        return pd.DataFrame({
            "SYN_MKT": px(mkt, 100), "SYN_OIL": px(oil_r, 70), "SYN_AIR": px(air, 50), "SYN_TECH": px(tech, 80),
            "SYN_CONF_X": px(cx, 60), "SYN_CONF_Y": px(cy, 60), "SYN_AIR2": px(air2, 40),
            "SYN_RATE": 3 + np.cumsum(rate_d) * 0.02,
        }, index=idx)

    def fetch(self, variable, start, end):
        df = self._frame()
        if variable.id not in df:
            raise ProviderError(f"unknown fixture series {variable.id}")
        return df[variable.id], {
            "provider": "fixture", "series_id": variable.id, "synthetic": True,
            "note": f"SYNTHETIC data (seed {self.SEED}); fixed 2021-2024 calendar, lookback is ignored"}


class DataService:
    """Resolves a Variable to its provider, caches responses, records provenance."""

    def __init__(self, cache_path=None, fred=None, alpaca=None, fixture=None, ttl=CACHE_TTL_S, tiingo=None):
        self.fred = fred or FredProvider()
        self.alpaca = alpaca or AlpacaProvider()
        self.tiingo = tiingo or TiingoProvider()
        self.fixture = fixture or FixtureProvider()
        self.cache_path, self.ttl = cache_path, ttl
        if cache_path:
            with sqlite3.connect(cache_path) as c:
                c.execute("CREATE TABLE IF NOT EXISTS provider_cache (key TEXT PRIMARY KEY, fetched_at REAL, payload TEXT)")

    def equity_status(self):
        """(usable, message, provider). Stock/ETF data comes from the first working provider:
        Tiingo (own key) -> Alpaca (optional). The research engine does not depend on Alpaca."""
        reasons = []
        for prov in (self.tiingo, self.alpaca):
            ok, why = prov.check()
            if ok:
                return True, "ok", prov
            reasons.append(why)
        if not self.tiingo.configured and not self.alpaca.configured:
            return False, ("No stock/ETF data provider is configured. Set TIINGO_API_KEY in backend/.env.local "
                           "(free key at tiingo.com; Alpaca keys also work)"), None
        return False, "; ".join(r for r in reasons if "not configured" not in r and "not set" not in r), None

    def _provider(self, v):
        if v.source == "equity":
            ok, why, prov = self.equity_status()
            if not ok:
                raise ProviderError(why)
            return prov
        return {"fred": self.fred, "fixture": self.fixture}[v.source]

    def _cache_get(self, key):
        if not self.cache_path:
            return None
        with sqlite3.connect(self.cache_path) as c:
            row = c.execute("SELECT fetched_at, payload FROM provider_cache WHERE key=?", (key,)).fetchone()
        if row and time.time() - row[0] < self.ttl:
            d = json.loads(row[1])
            s = pd.Series(d["values"], index=pd.to_datetime(d["dates"]), dtype=float)
            return s, d["prov"], row[0]
        return None

    def _cache_put(self, key, s, prov, ts):
        if not self.cache_path:
            return
        d = {"dates": [x.date().isoformat() for x in s.index], "values": [float(v) for v in s.values], "prov": prov}
        with sqlite3.connect(self.cache_path) as c:
            c.execute("INSERT OR REPLACE INTO provider_cache VALUES (?,?,?)", (key, ts, json.dumps(d)))

    def get(self, variable, lookback_years, today=None):
        """Returns (level Series, provenance). Raises ProviderError."""
        today = today or dt.date.today()
        end = today
        start = today - dt.timedelta(days=int(365.25 * lookback_years))
        prov_obj = self._provider(variable)
        key = f"{variable.source}:{variable.id}:{getattr(prov_obj, 'name', '')}:{start}:{end}:{getattr(prov_obj, 'feed', '')}"
        hit = self._cache_get(key) if variable.source != "fixture" else None
        if hit:
            s, prov, ts = hit
            cached = True
        else:
            s, prov = prov_obj.fetch(variable, start, end)
            ts = time.time()
            if variable.source != "fixture":
                self._cache_put(key, s, prov, ts)
            cached = False
        prov = dict(prov, variable_id=variable.id, label=variable.label, units=variable.units,
                    retrieved_at=dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat(), from_cache=cached,
                    observations=int(len(s)), first_date=s.index[0].date().isoformat(),
                    last_date=s.index[-1].date().isoformat(), requested_start=start.isoformat(),
                    requested_end=end.isoformat())
        return s, prov
