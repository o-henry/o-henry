def download_benchmarks(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for market, symbol in INDEX_SYMBOLS.items():
        raw = yf.download(
            symbol,
            start=(min_date - pd.Timedelta(days=180)).strftime("%Y-%m-%d"),
            end=(max_date + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=120,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty or "Close" not in raw.columns:
            raise RuntimeError(f"Benchmark unavailable: {market}")
        d = raw[["Open", "High", "Low", "Close", "Volume"]].reset_index()
        d.columns = ["date", "bench_open", "bench_high", "bench_low", "bench_close", "bench_volume"]
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
        d["market"] = market
        frames.append(d)
    return pd.concat(frames, ignore_index=True).sort_values(["market", "date"]).reset_index(drop=True)


def _rank_pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def build_features(panel: pd.DataFrame, benchmarks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = panel.copy().sort_values(["market", "code", "date"]).reset_index(drop=True)
    g = p.groupby("code", group_keys=False)
    prev_close = g["adj_close"].shift(1)
    p["ret1"] = g["adj_close"].pct_change()
    for n in [2, 3, 5, 10, 20, 60, 120]:
        p[f"ret{n}"] = g["adj_close"].pct_change(n)
    p["gap"] = p["adj_open"] / prev_close - 1.0
    p["intraday"] = p["adj_close"] / p["adj_open"] - 1.0
    p["range_pct"] = (p["adj_high"] - p["adj_low"]) / prev_close
    p["close_loc"] = (p["adj_close"] - p["adj_low"]) / (p["adj_high"] - p["adj_low"]).replace(0.0, np.nan)
    for n in [10, 20, 60]:
        p[f"ma{n}"] = g["adj_close"].transform(lambda s, n=n: s.rolling(n, min_periods=n).mean())
    p["high20_prev"] = g["adj_high"].transform(lambda s: s.rolling(20, min_periods=20).max().shift(1))
    p["high60_prev"] = g["adj_high"].transform(lambda s: s.rolling(60, min_periods=60).max().shift(1))
    p["low20_prev"] = g["adj_low"].transform(lambda s: s.rolling(20, min_periods=20).min().shift(1))
    p["prev_high"] = g["adj_high"].shift(1)
    p["prev_low"] = g["adj_low"].shift(1)
    p["prev_close"] = prev_close
    tr = pd.concat(
        [
            p["adj_high"] - p["adj_low"],
            (p["adj_high"] - prev_close).abs(),
            (p["adj_low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    p["tr"] = tr
    p["atr14"] = p.groupby("code")["tr"].transform(lambda s: s.rolling(14, min_periods=14).mean())
    p["atr_pct"] = p["atr14"] / p["adj_close"]
    p["vol10"] = g["ret1"].transform(lambda s: s.rolling(10, min_periods=10).std())
    p["vol20"] = g["ret1"].transform(lambda s: s.rolling(20, min_periods=20).std())
    p["avgvol20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    p["volume_ratio"] = p["volume"] / p["avgvol20"].replace(0.0, np.nan)
    p["turnover"] = p["adj_close"] * p["volume"]
    p["adv20"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    p["adv60"] = g["turnover"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    p["turnover_ratio"] = p["turnover"] / p["adv20"].replace(0.0, np.nan)
    p["drawdown20"] = p["adj_close"] / p["high20_prev"] - 1.0
    p["drawdown60"] = p["adj_close"] / p["high60_prev"] - 1.0
    p["dist_ma20_atr"] = (p["adj_close"] - p["ma20"]) / p["atr14"].replace(0.0, np.nan)
    p["history_n"] = g.cumcount() + 1
    p["max_abs_ret20"] = g["ret1"].transform(lambda s: s.abs().rolling(20, min_periods=20).max())

    b = benchmarks.copy().sort_values(["market", "date"]).reset_index(drop=True)
    bg = b.groupby("market", group_keys=False)
    b["bench_ret1"] = bg["bench_close"].pct_change()
    for n in [5, 20, 60]:
        b[f"bench_ret{n}"] = bg["bench_close"].pct_change(n)
    b["bench_ma20"] = bg["bench_close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    b["bench_ma60"] = bg["bench_close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    b["bench_vol20"] = bg["bench_ret1"].transform(lambda s: s.rolling(20, min_periods=20).std())
    b["bench_slope20"] = bg["bench_ma20"].pct_change(5)
    b["bench_dist_ma20"] = b["bench_close"] / b["bench_ma20"] - 1.0
    b["market_gap"] = b["bench_open"] / bg["bench_close"].shift(1) - 1.0
    b["market_intraday"] = b["bench_close"] / b["bench_open"] - 1.0

    p = p.merge(
        b[["market", "date", "bench_ret1", "bench_ret5", "bench_ret20", "bench_ret60", "bench_vol20", "bench_close", "bench_ma20", "bench_ma60", "bench_slope20", "bench_dist_ma20", "market_gap", "market_intraday"]],
        on=["market", "date"], how="left",
    )
    for n in [1, 5, 20, 60]:
        p[f"rel{n}"] = p[f"ret{n}"] - p[f"bench_ret{n}"]

    beta_parts: list[pd.Series] = []
    for _, x in p.groupby("code", sort=False):
        cov = x["ret1"].rolling(20, min_periods=20).cov(x["bench_ret1"])
        var = x["bench_ret1"].rolling(20, min_periods=20).var()
        beta_parts.append((cov / var.replace(0.0, np.nan)).set_axis(x.index))
    p["beta20"] = pd.concat(beta_parts).sort_index()

    rank_cols = ["rel5", "rel20", "rel60", "ret20", "volume_ratio", "adv20", "vol20"]
    for c in rank_cols:
        p[f"rank_{c}"] = p.groupby(["market", "date"])[c].transform(_rank_pct)
    p["rank_low_vol20"] = 1.0 - p["rank_vol20"]

    breadth_eligible = (
        (p["history_n"] >= 120)
        & (p["adj_close"] >= 1500)
        & (p["marcap"] >= 80_000_000_000)
        & (p["adv20"] >= 500_000_000)
        & (p["max_abs_ret20"] <= 0.40)
        & (p["corp_event_recent"] == 0)
    )
    breadth = (
        p[breadth_eligible]
        .assign(above20=lambda x: (x["adj_close"] > x["ma20"]).astype(float))
        .groupby(["market", "date"], as_index=False)["above20"]
        .mean()
        .rename(columns={"above20": "breadth20"})
    )
    p = p.merge(breadth, on=["market", "date"], how="left")
    b = b.merge(breadth, on=["market", "date"], how="left")

    def regime_row(r: pd.Series) -> str:
        if pd.isna(r["bench_ma60"]) or pd.isna(r["breadth20"]):
            return "unknown"
        panic = r["bench_ret5"] <= -0.065 or (r["bench_vol20"] >= 0.025 and r["bench_ret20"] < -0.04)
        bull = (
            r["bench_close"] > r["bench_ma20"] > r["bench_ma60"]
            and r["bench_ret20"] > 0.025
            and r["breadth20"] >= 0.54
        )
        bear = (
            r["bench_close"] < r["bench_ma60"]
            and r["bench_ret20"] < -0.025
            and r["breadth20"] <= 0.40
        )
        transition = (
            abs(r["bench_dist_ma20"]) <= 0.025
            or abs(r["bench_slope20"]) <= 0.008
            or 0.40 < r["breadth20"] < 0.54
        )
        if panic:
            return "panic"
        if bull:
            return "bull"
        if bear:
            return "bear"
        if transition:
            return "transition"
        return "neutral"

    b["regime"] = b.apply(regime_row, axis=1)
    p = p.merge(b[["market", "date", "regime"]], on=["market", "date"], how="left")

    for k in [1, 2, 3]:
        for c in ["adj_open", "adj_high", "adj_low", "adj_close"]:
            p[f"f{k}_{c}"] = p.groupby("code")[c].shift(-k)
        p[f"f{k}_date"] = p.groupby("code")["date"].shift(-k)
    return p, b


def generate_candidates(p: pd.DataFrame) -> pd.DataFrame:
    min_adv = np.where(p["market"].eq("KOSPI"), 2_000_000_000.0, 1_000_000_000.0)
    base = (
        (p["date"] >= pd.Timestamp("2020-01-01"))
        & (p["history_n"] >= 120)
        & p["adj_close"].between(1500, 700000)
        & (p["marcap"] >= 100_000_000_000)
        & (p["adv20"] >= min_adv)
        & p["vol20"].between(0.008, 0.13)
        & (p["max_abs_ret20"] <= 0.40)
        & (p["corp_event_recent"] == 0)
        & (p["bad_adjustment"] == 0)
        & p["f1_adj_open"].notna()
        & (p["ret1"].abs() < 0.295)
    )
    d = p[base].copy()
    if d.empty:
        raise RuntimeError("No eligible candidate rows")

    masks: dict[str, pd.Series] = {}
    masks["gap_event"] = (
        d["regime"].isin(["bull", "transition", "neutral"])
        & d["gap"].between(0.02, 0.14)
        & d["ret1"].between(0.03, 0.22)
        & (d["intraday"] > 0.005)
        & (d["close_loc"] >= 0.72)
        & (d["volume_ratio"] >= 1.8)
        & (d["dist_ma20_atr"] <= 3.5)
        & (d["range_pct"] <= d["atr_pct"] * 3.2)
    )
    masks["breakout"] = (
        d["regime"].isin(["bull", "transition", "neutral"])
        & (d["adj_close"] >= d["high20_prev"] * 1.003)
        & d["ret1"].between(0.02, 0.16)
        & (d["volume_ratio"] >= 1.45)
        & (d["close_loc"] >= 0.75)
        & (d["gap"] <= 0.08)
        & (d["ret20"] > -0.03)
        & (d["dist_ma20_atr"] <= 3.2)
    )
    masks["pullback"] = (
        d["regime"].eq("bull")
        & (d["ret20"] >= 0.06)
        & (d["rank_rel20"] >= 0.82)
        & d["ret3"].between(-0.10, -0.005)
        & (d["intraday"] > 0.002)
        & (d["close_loc"] >= 0.62)
        & (d["adj_close"] >= d["ma20"] * 0.97)
        & (d["volume_ratio"] >= 0.75)
        & (d["dist_ma20_atr"] <= 2.2)
    )
    masks["transition_reclaim"] = (
        d["regime"].isin(["transition", "neutral"])
        & (d["ret60"] >= 0.08)
        & (d["rank_rel20"] >= 0.85)
        & d["ret5"].between(-0.13, 0.01)
        & (d["intraday"] >= 0.012)
        & (d["adj_close"] >= d["prev_high"] * 0.997)
        & (d["close_loc"] >= 0.70)
        & (d["volume_ratio"] >= 0.95)
        & (d["dist_ma20_atr"] <= 2.0)
    )
    masks["panic_rs"] = (
        d["regime"].isin(["bear", "panic"])
        & (d["rank_rel20"] >= 0.94)
        & (d["rel1"] >= 0.035)
        & (d["ret1"] >= -0.01)
        & (d["close_loc"] >= 0.72)
        & (d["volume_ratio"] >= 1.25)
        & ((d["ret20"] > 0.0) | (d["adj_close"] > d["ma20"]))
        & (d["gap"] > -0.08)
    )
    masks["panic_reversal"] = (
        d["regime"].eq("panic")
        & (d["ret5"] <= -0.18)
        & (d["gap"] <= -0.025)
        & (d["intraday"] >= 0.06)
        & (d["close_loc"] >= 0.87)
        & (d["volume_ratio"] >= 2.5)
        & (d["rank_adv20"] >= 0.82)
    )

    pieces: list[pd.DataFrame] = []
    for pattern, mask in masks.items():
        x = d[mask].copy()
        if x.empty:
            continue
        x["pattern"] = pattern
        rule = PATTERN_RULES[pattern]
        x["stop_pct"] = float(rule["stop"])
        x["target_pct"] = float(rule["target"])
        x["hold_days"] = int(rule["hold"])
        pieces.append(x)
    if not pieces:
        raise RuntimeError("No pattern candidates")
    c = pd.concat(pieces, ignore_index=True)
    c = c.drop_duplicates(["market", "code", "date", "pattern"]).reset_index(drop=True)
    return label_candidates(c)


def _simulate_candidate_row(r: pd.Series, cost_mult: float = 1.0) -> dict[str, Any]:
    entry = float(r["f1_adj_open"])
    stop = entry * (1.0 - float(r["stop_pct"]))
    target = entry * (1.0 + float(r["target_pct"]))
    hold = int(r["hold_days"])
    exit_price = np.nan
    exit_date = pd.NaT
    reason = "missing"
    mae = 0.0
    mfe = 0.0
    for k in range(1, hold + 1):
        o = r.get(f"f{k}_adj_open")
        h = r.get(f"f{k}_adj_high")
        l = r.get(f"f{k}_adj_low")
        cl = r.get(f"f{k}_adj_close")
        dt = r.get(f"f{k}_date")
        if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(cl) or pd.isna(dt):
            break
        o = float(o); h = float(h); l = float(l); cl = float(cl)
        mae = min(mae, l / entry - 1.0)
        mfe = max(mfe, h / entry - 1.0)
        if o <= stop:
            exit_price, exit_date, reason = o, pd.Timestamp(dt), "gap_stop"
            break
        if o >= target:
            exit_price, exit_date, reason = o, pd.Timestamp(dt), "gap_target"
            break
        stop_hit = l <= stop
        target_hit = h >= target
        if stop_hit:
            exit_price, exit_date, reason = stop, pd.Timestamp(dt), "stop"
            break
        if target_hit:
            exit_price, exit_date, reason = target, pd.Timestamp(dt), "target"
            break
        if k == hold:
            exit_price, exit_date, reason = cl, pd.Timestamp(dt), "time"
    if pd.isna(exit_price) or pd.isna(exit_date) or entry <= 0:
        return {"label_valid": False}
    net = (exit_price * (1.0 - SELL_COST * cost_mult)) / (entry * (1.0 + BUY_COST * cost_mult)) - 1.0
    return {
        "label_valid": True,
        "entry_date": pd.Timestamp(r["f1_date"]),
        "entry_price": entry,
        "exit_date": exit_date,
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "net_return": float(net),
        "mae": float(mae),
        "mfe": float(mfe),
    }


def label_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    records = [_simulate_candidate_row(r) for _, r in candidates.iterrows()]
    lab = pd.DataFrame(records, index=candidates.index)
    out = pd.concat([candidates, lab], axis=1)
    out = out[out["label_valid"].fillna(False)].copy()
    out["target_positive"] = (out["net_return"] > 0.0).astype(int)
    out["target_return_clip"] = out["net_return"].clip(-0.15, 0.20)
    return out.reset_index(drop=True)
