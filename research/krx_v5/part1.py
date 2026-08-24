from __future__ import annotations

import concurrent.futures as cf
import io
import json
import math
import os
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from huggingface_hub import hf_hub_download
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

ROOT = Path("/tmp/krx_v5")
OUT = ROOT / "outputs"
CACHE = ROOT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

PAGE_URL = "https://www.ai-stock.co.kr/krx-daily.html"
CSV_BASE = "https://www.ai-stock.co.kr/data/krx_ohlcv/merged/"
HF_MAP_REPO = "aikstockdata/korea-equity-daily-2026-08"
HF_MAP_FILE = "daily_prices.csv"
START_DATE = pd.Timestamp("2019-01-01")
END_DATE = pd.Timestamp("2026-08-21")
INITIAL_CAPITAL = 2_000_000.0
BUY_COST = 0.0005
SELL_COST = 0.0025
MAX_POSITIONS = 2
RISK_PER_TRADE = 0.0125
MAX_POSITION_WEIGHT = 0.55
MAX_GROSS_EXPOSURE = 0.95
MONTHLY_LOSS_CUT = -0.12

INDEX_SYMBOLS = {"KOSPI": "^KS11", "KOSDAQ": "^KQ11"}

EXCLUDE_NAME_PATTERNS = [
    r"스팩", r"리츠", r"ETF", r"ETN", r"인버스", r"레버리지", r"선물", r"채권",
    r"KODEX", r"TIGER", r"KOSEF", r"KBSTAR", r"ARIRANG", r"ACE ", r"SOL ",
    r"HANARO", r"TIMEFOLIO", r"FOCUS", r"히어로즈", r"마이티", r"TREX", r"파워 ",
]

PATTERN_RULES: dict[str, dict[str, float | int]] = {
    "gap_event": {"stop": 0.045, "target": 0.095, "hold": 3},
    "breakout": {"stop": 0.040, "target": 0.085, "hold": 3},
    "pullback": {"stop": 0.035, "target": 0.075, "hold": 3},
    "transition_reclaim": {"stop": 0.032, "target": 0.065, "hold": 2},
    "panic_rs": {"stop": 0.030, "target": 0.060, "hold": 1},
    "panic_reversal": {"stop": 0.035, "target": 0.070, "hold": 1},
}

MODEL_NUMERIC_FEATURES = [
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "ret60", "ret120",
    "rel1", "rel5", "rel20", "rel60", "gap", "intraday", "range_pct", "close_loc",
    "volume_ratio", "turnover_ratio", "adv20", "marcap", "vol10", "vol20",
    "drawdown20", "drawdown60", "dist_ma20_atr", "atr_pct", "beta20",
    "rank_rel5", "rank_rel20", "rank_rel60", "rank_ret20", "rank_volume_ratio",
    "rank_adv20", "rank_low_vol20", "breadth20", "bench_ret5", "bench_ret20",
    "bench_vol20", "bench_dist_ma20", "market_gap", "market_intraday",
]


@dataclass(frozen=True)
class ThresholdConfig:
    edge_quantile: float
    min_probability: float
    top_n: int
    panic_probability_add: float

    @property
    def config_id(self) -> str:
        return (
            f"q{self.edge_quantile:.2f}-p{self.min_probability:.2f}"
            f"-n{self.top_n}-pp{self.panic_probability_add:.2f}"
        )


def http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 KRX-research-audit/1.0"})
    return s


def discover_urls() -> dict[str, str]:
    s = http_session()
    text = s.get(PAGE_URL, timeout=180).text
    paths = sorted(set(re.findall(r"data/krx_ohlcv/merged/fdr_KRX_p1d_(\d{6})\.csv", text)))
    if not paths:
        paths = sorted(set(re.findall(r"fdr_KRX_p1d_(\d{6})\.csv", text)))
    if len(paths) < 1000:
        raise RuntimeError(f"Only {len(paths)} KRX files discovered")
    return {code: urljoin(CSV_BASE, f"fdr_KRX_p1d_{code}.csv") for code in paths}


def current_mapping() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    try:
        import FinanceDataReader as fdr

        listing = fdr.StockListing("KRX")
        cols = {str(c).lower(): c for c in listing.columns}
        code_col = cols.get("code") or cols.get("symbol")
        name_col = cols.get("name")
        market_col = cols.get("market")
        dept_col = cols.get("dept")
        if code_col and name_col and market_col:
            d = listing[[code_col, name_col, market_col] + ([dept_col] if dept_col else [])].copy()
            d.columns = ["code", "name", "market"] + (["dept"] if dept_col else [])
            frames.append(d)
    except Exception as exc:
        errors.append(f"FinanceDataReader: {exc!r}")

    try:
        local = hf_hub_download(repo_id=HF_MAP_REPO, filename=HF_MAP_FILE, repo_type="dataset")
        h = pd.read_csv(local, dtype={"code": str}, usecols=["code", "name_ko", "market"])
        h = h.rename(columns={"name_ko": "name"})
        frames.append(h)
    except Exception as exc:
        errors.append(f"HF mapping: {exc!r}")

    if not frames:
        raise RuntimeError("No market mapping: " + " | ".join(errors))

    d = pd.concat(frames, ignore_index=True, sort=False)
    d["code"] = d["code"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    d["name"] = d["name"].astype(str).str.strip()
    d["market"] = d["market"].astype(str).str.upper().str.strip()
    d["market"] = d["market"].replace({"STK": "KOSPI", "KSQ": "KOSDAQ", "KOSDAQ GLOBAL": "KOSDAQ"})
    d = d[d["market"].isin(["KOSPI", "KOSDAQ"])].copy()
    d = d.dropna(subset=["code", "name"]).drop_duplicates("code", keep="first")

    exclusion = pd.Series(False, index=d.index)
    for pat in EXCLUDE_NAME_PATTERNS:
        exclusion |= d["name"].str.contains(pat, case=False, regex=True, na=False)
    exclusion |= d["name"].str.contains(r"(?:^|\s)\S*우(?:B|C)?$|[23]우B$", regex=True, na=False)
    if "dept" in d.columns:
        exclusion |= d["dept"].astype(str).str.contains("관리|투자주의", regex=True, na=False)
    d = d[~exclusion].copy()
    return d[["code", "name", "market"]]


def _download_one(rec: tuple[str, str, str, str], retries: int = 3) -> tuple[pd.DataFrame | None, str | None]:
    code, name, market, url = rec
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=180, headers={"User-Agent": "Mozilla/5.0 KRX-research-audit/1.0"})
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
            required = {"date", "open", "high", "low", "close", "volume", "marcap", "shares"}
            if not required.issubset(df.columns):
                raise ValueError(f"missing columns {required - set(df.columns)}")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[df["date"].between(START_DATE, END_DATE)].copy()
            if df.empty:
                return None, f"{code}: no rows in date range"
            for c in ["open", "high", "low", "close", "volume", "marcap", "shares"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["code"] = code
            df["name"] = name
            df["market"] = market
            return df[["date", "code", "name", "market", "open", "high", "low", "close", "volume", "marcap", "shares"]], None
        except Exception as exc:
            last = exc
            time.sleep(0.5 * (attempt + 1))
    return None, f"{code}: {last!r}"


def download_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    urls = discover_urls()
    mapping = current_mapping()
    mapping = mapping[mapping["code"].isin(urls)].copy()
    records = [(r.code, r.name, r.market, urls[r.code]) for r in mapping.itertuples(index=False)]
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=32) as ex:
        futures = [ex.submit(_download_one, rec) for rec in records]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            frame, error = fut.result()
            if frame is not None:
                frames.append(frame)
            if error:
                errors.append(error)
            if i % 100 == 0:
                print(f"downloaded {i}/{len(futures)} files, valid={len(frames)}, errors={len(errors)}")
    if len(frames) < 800:
        raise RuntimeError(f"Only {len(frames)} valid stock files")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["date", "open", "high", "low", "close", "volume", "marcap", "shares"])
    panel = panel[(panel[["open", "high", "low", "close"]] > 0).all(axis=1)]
    panel = panel[(panel["high"] >= panel[["open", "close", "low"]].max(axis=1)) & (panel["low"] <= panel[["open", "close", "high"]].min(axis=1))]
    panel = panel.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"]).reset_index(drop=True)
    quality = {
        "discovered_files": len(urls),
        "mapped_common_stocks": len(mapping),
        "downloaded_valid_files": len(frames),
        "errors_count": len(errors),
        "errors_sample": errors[:100],
        "rows_raw_filtered": len(panel),
        "min_date": str(panel["date"].min().date()),
        "max_date": str(panel["date"].max().date()),
        "stocks": int(panel["code"].nunique()),
        "market_counts": panel[["code", "market"]].drop_duplicates()["market"].value_counts().to_dict(),
    }
    return panel, quality


def adjust_splits(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out: list[pd.DataFrame] = []
    events: list[dict[str, Any]] = []
    for code, g in panel.groupby("code", sort=False):
        g = g.sort_values("date").copy()
        price_ratio = g["close"] / g["close"].shift(1)
        share_ratio = g["shares"] / g["shares"].shift(1)
        marcap_ratio = g["marcap"] / g["marcap"].shift(1)
        product = price_ratio * share_ratio
        split = (
            ((share_ratio >= 1.25) | (share_ratio <= 0.80))
            & product.between(0.80, 1.25)
            & marcap_ratio.between(0.55, 1.80)
            & share_ratio.notna()
        )
        event_ratio = pd.Series(np.where(split, share_ratio, 1.0), index=g.index, dtype=float)
        factor = event_ratio.shift(-1, fill_value=1.0).iloc[::-1].cumprod().iloc[::-1]
        factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for c in ["open", "high", "low", "close"]:
            g[f"adj_{c}"] = g[c] / factor
        g["adj_shares"] = g["shares"] * factor
        g["split_event"] = split.astype(int)
        g["split_factor"] = factor
        g["corp_event_recent"] = split.astype(float).rolling(6, min_periods=1).max().fillna(0.0).astype(int)
        if split.any():
            for r in g.loc[split, ["date", "code", "name", "market"]].itertuples(index=False):
                events.append({"date": str(r.date.date()), "code": r.code, "name": r.name, "market": r.market})
        out.append(g)
    p = pd.concat(out, ignore_index=True).sort_values(["market", "code", "date"]).reset_index(drop=True)
    p["adj_ret1_check"] = p.groupby("code")["adj_close"].pct_change()
    p["bad_adjustment"] = (p["adj_ret1_check"].abs() > 0.45).astype(int)
    quality = {
        "split_events": len(events),
        "split_event_sample": events[:100],
        "bad_adjustment_rows": int(p["bad_adjustment"].sum()),
    }
    return p, quality
