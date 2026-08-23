def create_pine_template(path: Path) -> None:
    pine = r'''//@version=6
strategy("KRX v5 선택종목 실행층 — 검증 후보", overlay=true, initial_capital=2000000,
     currency=currency.KRW, pyramiding=0, commission_type=strategy.commission.percent,
     commission_value=0.05, slippage=1, process_orders_on_close=false,
     calc_on_every_tick=false, calc_on_order_fills=false, use_bar_magnifier=false)

// 이 Pine은 전 종목 스캐너가 아니다.
// 외부 시점고정 랭킹/모델이 선택한 종목의 실행과 시각화만 담당한다.
externalApproved = input.bool(false, "외부 종목선정 승인")
profile = input.string("pullback", "실행 프로필", options=["gap_event","breakout","pullback","transition_reclaim","panic_rs","panic_reversal"])
maxHold = input.int(3, "최대 보유 거래일", minval=1, maxval=3)

float stopPct = profile == "gap_event" ? 0.045 : profile == "breakout" ? 0.040 : profile == "pullback" ? 0.035 : profile == "transition_reclaim" ? 0.032 : profile == "panic_rs" ? 0.030 : 0.035
float targetPct = profile == "gap_event" ? 0.095 : profile == "breakout" ? 0.085 : profile == "pullback" ? 0.075 : profile == "transition_reclaim" ? 0.065 : profile == "panic_rs" ? 0.060 : 0.070

float ema20 = ta.ema(close,20)
float ema60 = ta.ema(close,60)
float atr14 = ta.atr(14)
float resistance = ta.highest(high,20)[1]
float support = ta.lowest(low,20)[1]
float volumeRatio = volume / ta.sma(volume,20)[1]
float closeLoc = high > low ? (close-low)/(high-low) : 0.5
float gap = open / close[1] - 1.0
float intraday = close/open - 1.0

bool rawSetup =
     profile == "gap_event" ? gap >= 0.02 and gap <= 0.14 and closeLoc >= 0.72 and volumeRatio >= 1.8 and intraday > 0.005 :
     profile == "breakout" ? close >= resistance*1.003 and closeLoc >= 0.75 and volumeRatio >= 1.45 :
     profile == "pullback" ? close > ema60 and low <= ema20*1.02 and close > open and closeLoc >= 0.62 :
     profile == "transition_reclaim" ? close > high[1]*0.997 and close > open and closeLoc >= 0.70 and volumeRatio >= 0.95 :
     profile == "panic_rs" ? close > open and closeLoc >= 0.72 and volumeRatio >= 1.25 :
     close > open and intraday >= 0.06 and closeLoc >= 0.87 and volumeRatio >= 2.5

bool signal = externalApproved and strategy.position_size == 0 and rawSetup
if signal
    strategy.entry("L", strategy.long)

var float stopLine = na
var float targetLine = na
var int entryBar = na
bool newLong = strategy.position_size > 0 and strategy.position_size[1] <= 0
if newLong
    stopLine := strategy.position_avg_price * (1-stopPct)
    targetLine := strategy.position_avg_price * (1+targetPct)
    entryBar := bar_index
if strategy.position_size > 0
    strategy.exit("X", "L", stop=stopLine, limit=targetLine)
    if bar_index-entryBar+1 >= maxHold
        strategy.close("L", comment="시간청산", immediately=true)
bool closedNow = strategy.position_size == 0 and strategy.position_size[1] > 0
if closedNow
    stopLine := na
    targetLine := na
    entryBar := na

plot(support, "20일 지지", color=color.rgb(35,125,80), linewidth=2)
plot(resistance, "20일 저항", color=color.rgb(205,120,25), linewidth=2)
plot(strategy.position_size>0 ? strategy.position_avg_price : na, "매수 평단", color=color.rgb(65,90,190), linewidth=2, style=plot.style_linebr)
plot(strategy.position_size>0 ? stopLine : na, "손절", color=color.rgb(190,45,45), linewidth=2, style=plot.style_linebr)
plot(strategy.position_size>0 ? targetLine : na, "목표", color=color.rgb(20,115,100), linewidth=2, style=plot.style_linebr)
plotshape(signal, title="매수 준비", text="매수 준비", style=shape.labelup, location=location.belowbar, color=color.rgb(245,205,75), textcolor=color.black, size=size.tiny)
plotshape(newLong, title="매수", text="매수", style=shape.labelup, location=location.belowbar, color=color.rgb(100,205,120), textcolor=color.black, size=size.tiny)
plotshape(closedNow, title="매도", text="매도", style=shape.labeldown, location=location.abovebar, color=color.rgb(235,115,100), textcolor=color.black, size=size.tiny)
'''
    path.write_text(pine, encoding="utf-8")


def gate_2025(m: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    return {
        "return_20pct": m["return"] >= 0.20,
        "monthly_geom_1_5pct": m["monthly_geom"] >= 0.015,
        "profit_factor_1_25": m["profit_factor"] >= 1.25,
        "max_drawdown_18pct": m["max_drawdown"] >= -0.18,
        "positive_months_58pct": m["positive_month_ratio"] >= 0.58,
        "trades_30": m["trades"] >= 30,
        "cost_stress_positive": stress["return"] > 0,
    }


def gate_2026(m: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    regime = m.get("regime_stats", {})
    defensive_pnl = sum(float(regime.get(x, {}).get("pnl", 0.0)) for x in ["bear", "panic"])
    transition_pnl = float(regime.get("transition", {}).get("pnl", 0.0)) + float(regime.get("neutral", {}).get("pnl", 0.0))
    return {
        "return_30pct": m["return"] >= 0.30,
        "monthly_geom_3_5pct": m["monthly_geom"] >= 0.035,
        "profit_factor_1_30": m["profit_factor"] >= 1.30,
        "max_drawdown_20pct": m["max_drawdown"] >= -0.20,
        "positive_months_60pct": m["positive_month_ratio"] >= 0.60,
        "trades_25": m["trades"] >= 25,
        "transition_nonnegative": transition_pnl >= 0,
        "defensive_nonnegative": defensive_pnl >= 0,
        "cost_stress_positive": stress["return"] > 0,
    }


def main() -> None:
    print("STEP 1: download long OHLCV panel")
    panel_raw, download_quality = download_panel()
    panel, adjustment_quality = adjust_splits(panel_raw)
    print(json.dumps({"download": download_quality, "adjustment": adjustment_quality}, ensure_ascii=False, indent=2))

    print("STEP 2: features and candidates")
    benchmarks = download_benchmarks(panel["date"].min(), panel["date"].max())
    features, bench = build_features(panel, benchmarks)
    candidates = generate_candidates(features)
    print("candidate rows", len(candidates), "period", candidates["date"].min(), candidates["date"].max())
    print(candidates.groupby(["pattern", "regime"]).size().to_string())

    train_2024 = candidates[candidates["date"] <= pd.Timestamp("2023-12-31")]
    cal_2024 = candidates[candidates["date"].between(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))]
    reg24, cls24, columns24, fit24 = fit_models(train_2024)
    scored24 = score_candidates(cal_2024, reg24, cls24, columns24)

    calibration_rows: list[dict[str, Any]] = []
    best: tuple[ThresholdConfig, float, dict[str, Any]] | None = None
    best_score = -999.0
    for cfg, edge in threshold_grid(scored24):
        m, _, _ = run_portfolio(scored24, features, bench, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"), cfg, edge)
        s = calibration_score(m)
        calibration_rows.append({"config_id": cfg.config_id, "edge_threshold": edge, "score": s, **{k: m[k] for k in ["return", "profit_factor", "max_drawdown", "trades", "monthly_geom", "positive_month_ratio"]}})
        if s > best_score:
            best_score = s
            best = (cfg, edge, m)
    if best is None:
        raise RuntimeError("No threshold calibration result")
    selected_cfg, selected_edge_2024, selected_cal_metrics = best
    selected_quantile = selected_cfg.edge_quantile
    print("selected threshold config", selected_cfg, "calibration", json.dumps(selected_cal_metrics, ensure_ascii=False))

    train_2025 = candidates[candidates["date"] <= pd.Timestamp("2024-12-31")]
    test_2025 = candidates[candidates["date"].between(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))]
    reg25, cls25, columns25, fit25 = fit_models(train_2025)
    scored25 = score_candidates(test_2025, reg25, cls25, columns25)
    edge25 = float(scored25["edge_score"].quantile(selected_quantile))
    m25, tr25, eq25 = run_portfolio(scored25, features, bench, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), selected_cfg, edge25)
    s25, _, _ = run_portfolio(scored25, features, bench, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), selected_cfg, edge25, cost_mult=1.5)
    gates25 = gate_2025(m25, s25)

    train_2026 = candidates[candidates["date"] <= pd.Timestamp("2025-12-31")]
    test_2026 = candidates[candidates["date"].between(pd.Timestamp("2026-01-01"), END_DATE)]
    reg26, cls26, columns26, fit26 = fit_models(train_2026)
    scored26 = score_candidates(test_2026, reg26, cls26, columns26)
    edge26 = float(scored26["edge_score"].quantile(selected_quantile))
    m26, tr26, eq26 = run_portfolio(scored26, features, bench, pd.Timestamp("2026-01-01"), END_DATE, selected_cfg, edge26)
    s26, _, _ = run_portfolio(scored26, features, bench, pd.Timestamp("2026-01-01"), END_DATE, selected_cfg, edge26, cost_mult=1.5)
    gates26 = gate_2026(m26, s26)

    result = {
        "version": "krx-v5-long-ohlcv-ml-event-swing",
        "status": "ACCEPTED" if all(gates25.values()) and all(gates26.values()) else "REJECTED",
        "accepted": bool(all(gates25.values()) and all(gates26.values())),
        "integrity": {
            "model_hyperparameters_fixed": True,
            "threshold_selected_from": "2024 only",
            "2025_used_for_threshold_selection": False,
            "2026_used_for_threshold_selection": False,
            "signal_to_fill": "close t -> next open t+1",
            "same_bar_ambiguity": "stop first",
            "integer_shares": True,
            "long_only": True,
            "leverage": False,
            "max_holding_sessions": 3,
        },
        "data_quality": {"download": download_quality, "adjustment": adjustment_quality},
        "candidate_rows": int(len(candidates)),
        "candidate_counts": {f"{a}|{b}": int(v) for (a, b), v in candidates.groupby(["pattern", "regime"]).size().items()},
        "selected_threshold": asdict(selected_cfg),
        "selected_quantile": selected_quantile,
        "calibration_2024": selected_cal_metrics,
        "model_fit": {"for_2024": fit24, "for_2025": fit25, "for_2026": fit26},
        "validation_2025": {"metrics": m25, "cost_stress_1_5x": s25, "gates": gates25, "passed": all(gates25.values())},
        "stress_2026_01_08_21": {"metrics": m26, "cost_stress_1_5x": s26, "gates": gates26, "passed": all(gates26.values())},
        "limitations": [
            "Current listing mapping can retain survivorship bias despite date-presence filtering.",
            "Corporate actions are algorithmically split-adjusted from shares/market-cap changes, not exchange-certified adjustment factors.",
            "No complete point-in-time news archive; gap/volume event reaction is used as a catalyst proxy.",
            "2026 is a reused stress window in this research conversation, not pristine unseen OOS.",
        ],
    }

    (OUT / "final_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(calibration_rows).sort_values("score", ascending=False).to_csv(OUT / "calibration_2024.csv", index=False)
    tr25.to_csv(OUT / "trades_2025.csv", index=False)
    eq25.to_csv(OUT / "equity_2025.csv", index=False)
    tr26.to_csv(OUT / "trades_2026.csv", index=False)
    eq26.to_csv(OUT / "equity_2026.csv", index=False)
    candidates[["date", "entry_date", "exit_date", "market", "code", "name", "regime", "pattern", "net_return", "mae", "mfe"]].to_parquet(OUT / "candidate_audit.parquet", index=False)
    create_pine_template(OUT / "KRX_v5_selected_stock_execution.txt")

    print("===KRX_V5_RESULT_BEGIN===")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print("===KRX_V5_RESULT_END===")


if __name__ == "__main__":
    main()
