from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_v1 as core

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v3"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 2_000_000.0
BASE_BUY_COST = 0.0005
SELL_COST_2025 = 0.0020
SELL_COST_2026 = 0.0025

FULL_START = pd.Timestamp("2026-01-01")
CALIBRATION_END = pd.Timestamp("2026-04-30")
FINAL_START = pd.Timestamp("2026-05-01")
REQUESTED_END = pd.Timestamp("2026-08-21")

KOSPI_CORE = "069500"


@dataclass
class Position:
    code: str
    name: str
    role: str
    regime: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    invested: float
    hold_days: int
    stop_pct: float
    target_pct: float
    age: int = 0


def extend_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    info: dict[str, Any] = {
        "attempted": True,
        "requested_end": str(REQUESTED_END.date()),
        "original_max": str(prices["date"].max().date()),
        "appended_rows": 0,
        "actual_max": str(prices["date"].max().date()),
        "errors": [],
    }
    if prices["date"].max() >= REQUESTED_END:
        return prices, info

    name_map = (
        prices.sort_values("date")
        .drop_duplicates(["market", "code"], keep="last")
        .set_index(["market", "code"])["name_ko"]
        .to_dict()
    )

    try:
        from pykrx import stock
    except Exception as exc:
        info["errors"].append(f"pykrx import failed: {exc!r}")
        return prices, info

    records: list[dict[str, Any]] = []
    for date in pd.bdate_range(prices["date"].max() + pd.Timedelta(days=1), REQUESTED_END):
        ds = date.strftime("%Y%m%d")
        for market in ["KOSPI", "KOSDAQ"]:
            try:
                frame = stock.get_market_ohlcv_by_ticker(ds, market=market)
                if frame is None or frame.empty:
                    continue
                frame = frame.reset_index()
                ticker_col = frame.columns[0]
                close_col = "종가" if "종가" in frame.columns else None
                volume_col = "거래량" if "거래량" in frame.columns else None
                if close_col is None or volume_col is None:
                    info["errors"].append(f"missing columns {ds} {market}: {list(frame.columns)}")
                    continue
                for row in frame[[ticker_col, close_col, volume_col]].itertuples(index=False):
                    code = str(row[0]).zfill(6)
                    close = float(row[1])
                    volume = float(row[2])
                    if not np.isfinite(close) or close <= 0 or volume < 0:
                        continue
                    name = name_map.get((market, code))
                    if name is None:
                        try:
                            name = stock.get_market_ticker_name(code)
                        except Exception:
                            name = code
                    records.append({"code": code, "name_ko": name or code, "market": market, "date": pd.Timestamp(date), "close": close, "volume": volume})
            except Exception as exc:
                info["errors"].append(f"{ds} {market}: {exc!r}")

    if records:
        ext = pd.DataFrame(records)
        prices = pd.concat([prices, ext], ignore_index=True)
        prices = prices.drop_duplicates(["market", "code", "date"], keep="last").sort_values(["market", "code", "date"]).reset_index(drop=True)
        info["appended_rows"] = int(len(ext))
        info["actual_max"] = str(prices["date"].max().date())
    return prices, info


def eligibility(day: pd.DataFrame, market: str) -> pd.DataFrame:
    d = day.copy()
    min_adv = 3_000_000_000 if market == "KOSPI" else 1_500_000_000
    mask = (
        (d["history_n"] >= 120)
        & (d["close"] >= 2000)
        & (d["adv20"] >= min_adv)
        & (d["max_abs_ret20"] <= 0.35)
        & d["vol20"].between(0.008, 0.12)
        & d["ma10"].notna()
        & d["ma20"].notna()
        & d["ma60"].notna()
        & d["rank_rel20"].notna()
    )
    return d[mask].copy()


def select_candidates(day: pd.DataFrame, market: str, regime: str) -> pd.DataFrame:
    d = eligibility(day, market)
    if d.empty:
        return d

    if regime == "bull":
        if market == "KOSPI":
            cond = (
                (d["close"] > d["ma20"])
                & (d["ma20"] >= d["ma60"] * 0.995)
                & (d["ret20"] > 0.02)
                & (d["rel20"] > 0.0)
                & (d["ret3"] > -0.06)
                & (d["dist_ma20_sigma"] <= 2.5)
            )
            d["score"] = 0.30*d["rank_rel20"] + 0.20*d["rank_rel60"] + 0.18*d["rank_ret20"] + 0.10*d["rank_ret60"] + 0.08*d["rank_volume_ratio"] + 0.08*d["rank_adv20"] + 0.06*d["rank_low_vol20"]
        else:
            cond = (
                (d["close"] > d["ma20"])
                & (d["ma20"] >= d["ma60"] * 0.99)
                & (d["ret20"] > 0.03)
                & (d["rel20"] > 0.0)
                & d["ret3"].between(-0.07, 0.04)
                & d["drawdown20"].between(-0.12, -0.005)
                & (d["dist_ma20_sigma"] <= 1.7)
            )
            d["score"] = 0.28*d["rank_rel20"] + 0.20*d["rank_rel60"] + 0.16*d["rank_ret20"] + 0.12*d["rank_ret60"] + 0.10*d["rank_volume_ratio"] + 0.08*d["rank_adv20"] + 0.06*d["rank_low_vol20"]
        n = 2
    elif regime in {"transition", "neutral"}:
        min_rank = 0.72 if market == "KOSPI" else 0.78
        cond = (
            (d["ret60"] > 0.0)
            & (d["rel20"] > 0.0)
            & (d["rank_rel20"] >= min_rank)
            & d["ret3"].between(-0.10, -0.005)
            & (d["ret1"] > 0.0)
            & (d["close"] >= d["ma10"] * 0.98)
            & (d["volume_ratio"] >= 0.90)
            & (d["dist_ma20_sigma"] <= 2.0)
        )
        d["score"] = 0.30*d["rank_rel20"] + 0.18*d["rank_rel60"] + 0.16*d["rank_ret60"] + 0.14*d["rank_volume_ratio"] + 0.12*d["rank_low_vol20"] + 0.10*(1.0 - (d["ret3"].abs()/0.10).clip(0, 1))
        n = 2
    elif regime == "bear":
        cond = (
            (d["ret20"] > 0.02)
            & (d["rel20"] > 0.06)
            & (d["rank_rel20"] >= 0.92)
            & (d["close"] > d["ma20"])
            & (d["ret5"] > -0.03)
            & ((d["beta20"] <= 0.85) | (d["volume_ratio"] >= 1.6))
            & (d["dist_ma20_sigma"] <= 1.8)
        )
        d["score"] = 0.36*d["rank_rel20"] + 0.18*d["rank_rel5"] + 0.14*d["rank_ret20"] + 0.12*d["rank_volume_ratio"] + 0.10*d["rank_low_vol20"] + 0.10*(1.0 - d["beta20"].clip(0,2)/2.0)
        n = 1
    else:
        reversal = (d["ret5"] <= -0.12) & (d["ret1"] >= 0.04) & (d["volume_ratio"] >= 1.7) & (d["rel1"] > 0.025) & (d["rank_adv20"] >= 0.75)
        absolute_rs = (d["ret20"] > 0.03) & (d["rel20"] > 0.08) & (d["rank_rel20"] >= 0.95) & (d["close"] > d["ma20"]) & (d["dist_ma20_sigma"] <= 1.5)
        cond = reversal | absolute_rs
        d["score"] = 0.36*d["rank_rel20"] + 0.18*d["rank_rel5"] + 0.14*d["rank_volume_ratio"] + 0.12*d["rank_adv20"] + 0.10*d["rank_low_vol20"] + 0.10*(1.0 - d["beta20"].clip(0,2)/2.0)
        d.loc[reversal, "score"] += 0.10
        n = 1

    out = d[cond & d["score"].notna()].copy()
    return out.sort_values(["score", "adv20"], ascending=[False, False]).head(n)


def regime_params(market: str, regime: str) -> dict[str, Any]:
    if market == "KOSPI":
        table = {
            "bull": dict(hold=2, exposure=1.00, stop=-0.065, target=0.13, core=0.40),
            "transition": dict(hold=1, exposure=0.70, stop=-0.050, target=0.09, core=0.00),
            "neutral": dict(hold=1, exposure=0.45, stop=-0.045, target=0.08, core=0.00),
            "bear": dict(hold=1, exposure=0.20, stop=-0.040, target=0.07, core=0.00),
            "panic": dict(hold=1, exposure=0.15, stop=-0.040, target=0.07, core=0.00),
            "unknown": dict(hold=1, exposure=0.00, stop=-0.04, target=0.07, core=0.00),
        }
    else:
        table = {
            "bull": dict(hold=2, exposure=1.00, stop=-0.070, target=0.14, core=0.00),
            "transition": dict(hold=1, exposure=0.65, stop=-0.055, target=0.10, core=0.00),
            "neutral": dict(hold=1, exposure=0.35, stop=-0.050, target=0.09, core=0.00),
            "bear": dict(hold=1, exposure=0.20, stop=-0.045, target=0.08, core=0.00),
            "panic": dict(hold=1, exposure=0.20, stop=-0.040, target=0.075, core=0.00),
            "unknown": dict(hold=1, exposure=0.00, stop=-0.04, target=0.07, core=0.00),
        }
    return table.get(regime, table["unknown"])


def sell_cost(date: pd.Timestamp, stress: float) -> float:
    return (SELL_COST_2026 if date >= pd.Timestamp("2026-01-01") else SELL_COST_2025) * stress


def simulate(market: str, features: pd.DataFrame, benchmark: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, *, stress: float = 1.0, exposure_scale: float = 1.0) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    data = features[(features["market"] == market) & features["date"].between(start, end)].copy()
    if data.empty:
        raise RuntimeError(f"No data for {market} {start} {end}")
    dates = sorted(pd.to_datetime(data["date"].unique()))
    by_date = {pd.Timestamp(d): x.copy() for d, x in data.groupby("date")}
    close_map = {(pd.Timestamp(r.date), str(r.code)): float(r.close) for r in data[["date", "code", "close"]].itertuples(index=False)}
    b = benchmark[(benchmark["market"] == market) & benchmark["date"].between(start, end)].copy()
    b = b.set_index("date").reindex(dates).ffill().reset_index()
    b["bench_daily_return"] = b["bench_close"].pct_change().fillna(0.0)

    cash = INITIAL_CAPITAL
    positions: list[Position] = []
    pending: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    prev_equity = INITIAL_CAPITAL

    for i, date in enumerate(dates):
        date = pd.Timestamp(date)
        day = by_date[date]
        for order in pending.pop(date, []):
            px = close_map.get((date, order["code"]))
            if px is None or px <= 0:
                continue
            max_affordable = cash/(1.0 + BASE_BUY_COST*stress)
            desired = min(order["notional"], max_affordable)
            shares = int(math.floor(desired/px))
            if shares < 1:
                continue
            invested = shares*px
            buy_cost = invested*BASE_BUY_COST*stress
            if invested + buy_cost > cash:
                continue
            cash -= invested + buy_cost
            positions.append(Position(order["code"], order["name"], order["role"], order["regime"], order["signal_date"], date, px, shares, invested, order["hold"], order["stop"], order["target"]))

        remaining: list[Position] = []
        for pos in positions:
            px = close_map.get((date, pos.code))
            if px is None:
                remaining.append(pos)
                continue
            if date > pos.entry_date:
                pos.age += 1
            gross_ret = px/pos.entry_price - 1.0
            reason = None
            if gross_ret <= pos.stop_pct:
                reason = "close_stop"
            elif gross_ret >= pos.target_pct:
                reason = "close_target"
            elif pos.age >= pos.hold_days:
                reason = "time_exit"
            if reason is None:
                remaining.append(pos)
            else:
                gross = pos.shares*px
                proceeds = gross - gross*sell_cost(date, stress)
                cash += proceeds
                bc = pos.invested*BASE_BUY_COST*stress
                pnl = proceeds - pos.invested - bc
                trades.append({"market":market,"code":pos.code,"name":pos.name,"role":pos.role,"regime":pos.regime,"signal_date":str(pos.signal_date.date()),"entry_date":str(pos.entry_date.date()),"exit_date":str(date.date()),"entry_price":pos.entry_price,"exit_price":px,"shares":pos.shares,"hold_days":pos.age,"exit_reason":reason,"gross_return":gross_ret,"net_pnl":pnl,"net_return_on_position":pnl/pos.invested if pos.invested else 0.0})
        positions = remaining

        marked = cash
        for pos in positions:
            marked += pos.shares*close_map.get((date,pos.code), pos.entry_price)
        daily_return = marked/prev_equity - 1.0 if prev_equity > 0 else 0.0
        prev_equity = marked
        regime = str(day["regime"].dropna().iloc[0]) if day["regime"].notna().any() else "unknown"
        bench_ret = float(b.loc[b["date"] == date, "bench_daily_return"].iloc[0]) if (b["date"] == date).any() else 0.0
        equity_rows.append({"date":date,"market":market,"equity":marked,"daily_return":daily_return,"benchmark_daily_return":bench_ret,"regime":regime,"active_positions":len(positions)})

        if i + 1 >= len(dates):
            continue
        next_date = pd.Timestamp(dates[i+1])
        params = regime_params(market, regime)
        total_exposure = min(1.0, params["exposure"]*exposure_scale)
        if total_exposure <= 0:
            continue
        orders: list[dict[str, Any]] = []
        core_weight = min(params["core"], total_exposure)
        leader_weight = max(0.0, total_exposure-core_weight)
        if core_weight > 0 and (date, KOSPI_CORE) in close_map:
            orders.append({"code":KOSPI_CORE,"name":"KODEX 200","role":"bull_core","regime":regime,"signal_date":date,"hold":params["hold"],"stop":params["stop"],"target":params["target"],"notional":marked*core_weight/max(1,params["hold"])})
        candidates = select_candidates(day, market, regime)
        if leader_weight > 0 and not candidates.empty:
            each = marked*leader_weight/max(1,params["hold"])/len(candidates)
            for r in candidates.itertuples(index=False):
                orders.append({"code":str(r.code),"name":str(r.name_ko),"role":"leader","regime":regime,"signal_date":date,"hold":params["hold"],"stop":params["stop"],"target":params["target"],"notional":each})
        pending[next_date] = orders

    last = pd.Timestamp(dates[-1])
    for pos in positions:
        px = close_map.get((last,pos.code), pos.entry_price)
        gross = pos.shares*px
        proceeds = gross - gross*sell_cost(last, stress)
        cash += proceeds
        bc = pos.invested*BASE_BUY_COST*stress
        pnl = proceeds - pos.invested - bc
        trades.append({"market":market,"code":pos.code,"name":pos.name,"role":pos.role,"regime":pos.regime,"signal_date":str(pos.signal_date.date()),"entry_date":str(pos.entry_date.date()),"exit_date":str(last.date()),"entry_price":pos.entry_price,"exit_price":px,"shares":pos.shares,"hold_days":pos.age,"exit_reason":"period_end","gross_return":px/pos.entry_price-1.0,"net_pnl":pnl,"net_return_on_position":pnl/pos.invested if pos.invested else 0.0})

    eq = pd.DataFrame(equity_rows)
    if not eq.empty:
        eq.loc[eq.index[-1],"equity"] = cash
        eq["daily_return"] = eq["equity"].pct_change().fillna(0.0)
    tr = pd.DataFrame(trades)
    metrics = core.compute_metrics(eq, tr, b, start, end)
    metrics.update({"market":market,"start":str(start.date()),"end":str(end.date()),"stress":stress,"exposure_scale":exposure_scale})
    return metrics, tr, eq


def acceptance(metrics: dict[str, Any], stress: dict[str, Any], *, final: bool) -> dict[str, bool]:
    r = metrics["regimes"]
    transition = r["transition"]["strategy_return"] + r["neutral"]["strategy_return"]
    defensive = r["bear"]["strategy_return"] + r["panic"]["strategy_return"]
    return {
        "return_positive": metrics["return"] > 0,
        "monthly_geom": metrics["monthly_geom"] >= (0.05 if final else 0.035),
        "profit_factor": metrics["profit_factor"] >= 1.30,
        "max_drawdown": metrics["max_drawdown"] >= -0.18,
        "positive_month_ratio": metrics["positive_month_ratio"] >= 0.60,
        "cost_stress": stress["return"] > 0,
        "alpha_positive": metrics["alpha_sum"] > 0,
        "beta_below_0_9": metrics["beta"] <= 0.90,
        "bull_positive": r["bull"]["strategy_return"] > 0,
        "transition_nonnegative": transition >= 0,
        "defensive_nonnegative": defensive >= 0,
    }


def main() -> None:
    prices = core.load_prices()
    prices, extension = extend_prices(prices)
    actual_end = min(REQUESTED_END, prices["date"].max())
    benchmarks = core.load_benchmarks(prices["date"].min(), actual_end, prices)
    features, bench_features = core.build_features(prices, benchmarks)
    output: dict[str, Any] = {"version":"korea-regime-v3-fixed","rules_frozen_before_final":True,"data":{"rows":int(len(prices)),"min_date":str(prices["date"].min().date()),"max_date":str(prices["date"].max().date()),"extension":extension,"limitations":["HF base dataset has about 250 sessions","current-covered universe implies survivorship bias","close/volume only; next-close execution","corporate-action adjustment not independently audited","no complete point-in-time news archive"]},"periods":{"full_requested":[str(FULL_START.date()),str(actual_end.date())],"design_informed":[str(FULL_START.date()),str(CALIBRATION_END.date())],"final_oos":[str(FINAL_START.date()),str(actual_end.date())]},"markets":{}}
    for market in ["KOSPI","KOSDAQ"]:
        full, full_tr, full_eq = simulate(market, features, bench_features, FULL_START, actual_end)
        calibration, _, _ = simulate(market, features, bench_features, FULL_START, CALIBRATION_END)
        final, final_tr, final_eq = simulate(market, features, bench_features, FINAL_START, actual_end)
        final_stress, _, _ = simulate(market, features, bench_features, FINAL_START, actual_end, stress=1.5)
        low_exp, _, _ = simulate(market, features, bench_features, FINAL_START, actual_end, exposure_scale=0.85)
        high_exp, _, _ = simulate(market, features, bench_features, FINAL_START, actual_end, exposure_scale=1.15)
        gates = acceptance(final, final_stress, final=True)
        output["markets"][market] = {"full_jan_aug":full,"design_informed_jan_apr":calibration,"final_oos_may_aug":final,"final_cost_stress_1_5x":final_stress,"final_exposure_sensitivity":{"0.85x":low_exp,"1.15x":high_exp},"final_gates":gates,"accepted":all(gates.values())}
        full_tr.to_csv(OUT/f"{market.lower()}_full_trades.csv", index=False)
        full_eq.to_csv(OUT/f"{market.lower()}_full_equity.csv", index=False)
        final_tr.to_csv(OUT/f"{market.lower()}_final_trades.csv", index=False)
        final_eq.to_csv(OUT/f"{market.lower()}_final_equity.csv", index=False)
    output["accepted_both"] = all(output["markets"][m]["accepted"] for m in ["KOSPI","KOSDAQ"])
    (OUT/"final_result.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print("===KOREA_REGIME_V3_RESULT_BEGIN===")
    print(json.dumps(output,ensure_ascii=False,indent=2))
    print("===KOREA_REGIME_V3_RESULT_END===")


if __name__ == "__main__":
    main()
