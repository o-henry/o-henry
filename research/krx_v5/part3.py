def make_design(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    num = df[MODEL_NUMERIC_FEATURES].copy()
    cat = pd.get_dummies(df[["market", "regime", "pattern"]].astype(str), prefix=["market", "regime", "pattern"], dtype=float)
    x = pd.concat([num, cat], axis=1).replace([np.inf, -np.inf], np.nan)
    if columns is None:
        columns = list(x.columns)
    x = x.reindex(columns=columns, fill_value=0.0)
    return x, columns


def fit_models(train: pd.DataFrame) -> tuple[Any, Any, list[str], dict[str, Any]]:
    use = train.dropna(subset=["target_return_clip", "target_positive"]).copy()
    if len(use) > 350_000:
        use["year"] = use["date"].dt.year
        parts = []
        for _, x in use.groupby(["year", "pattern"], group_keys=False):
            n = min(len(x), max(500, int(350_000 * len(x) / len(use))))
            parts.append(x.sample(n=min(n, len(x)), random_state=42))
        use = pd.concat(parts, ignore_index=True).drop(columns=["year"], errors="ignore")
    x, columns = make_design(use)
    y_reg = use["target_return_clip"].to_numpy()
    y_cls = use["target_positive"].to_numpy()
    reg = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=180, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=6.0, random_state=42,
    )
    cls = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=160, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=6.0, random_state=43,
    )
    reg.fit(x, y_reg)
    cls.fit(x, y_cls)
    info = {
        "rows": int(len(use)),
        "positive_rate": float(y_cls.mean()),
        "mean_target": float(y_reg.mean()),
        "start": str(use["date"].min().date()),
        "end": str(use["date"].max().date()),
    }
    return reg, cls, columns, info


def score_candidates(df: pd.DataFrame, reg: Any, cls: Any, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    x, _ = make_design(out, columns)
    out["pred_return"] = reg.predict(x)
    out["pred_probability"] = cls.predict_proba(x)[:, 1]
    out["edge_score"] = out["pred_return"] * (0.55 + out["pred_probability"])
    return out


def month_returns(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    x = equity.copy()
    x["month"] = x["date"].dt.to_period("M")
    out: dict[str, float] = {}
    for month, g in x.groupby("month"):
        out[str(month)] = float((1.0 + g["daily_return"].fillna(0.0)).prod() - 1.0)
    return out


def benchmark_return(bench: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    out: dict[str, float] = {}
    for market in ["KOSPI", "KOSDAQ"]:
        x = bench[(bench["market"] == market) & bench["date"].between(start, end)].sort_values("date")
        out[market] = float(x["bench_close"].iloc[-1] / x["bench_close"].iloc[0] - 1.0) if len(x) >= 2 else 0.0
    return out


def run_portfolio(
    scored: pd.DataFrame,
    panel: pd.DataFrame,
    bench: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: ThresholdConfig,
    edge_threshold: float,
    *,
    cost_mult: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    signals = scored[scored["date"].between(start, end)].copy()
    signals = signals[(signals["edge_score"] >= edge_threshold) & (signals["pred_probability"] >= cfg.min_probability)].copy()
    signals["required_probability"] = cfg.min_probability + np.where(signals["regime"].isin(["bear", "panic"]), cfg.panic_probability_add, 0.0)
    signals = signals[signals["pred_probability"] >= signals["required_probability"]]
    signals = signals.sort_values(["date", "edge_score", "adv20"], ascending=[True, False, False])
    signals = signals.drop_duplicates(["date", "code"], keep="first")
    by_signal_date = {pd.Timestamp(d): x.head(cfg.top_n * 3).copy() for d, x in signals.groupby("date")}

    prices = panel[panel["date"].between(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10))].copy()
    close_map = {(pd.Timestamp(r.date), str(r.code)): float(r.adj_close) for r in prices[["date", "code", "adj_close"]].itertuples(index=False)}
    dates = sorted(pd.Timestamp(x) for x in prices[prices["date"].between(start, end)]["date"].unique())
    if not dates:
        raise RuntimeError("No portfolio dates")

    entry_orders: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for signal_date, day in by_signal_date.items():
        for r in day.itertuples(index=False):
            entry_orders.setdefault(pd.Timestamp(r.entry_date), []).append(r._asdict())

    cash = INITIAL_CAPITAL
    positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    pattern_loss_streak: dict[str, int] = {}
    pattern_cooldown_until: dict[str, pd.Timestamp] = {}
    month_anchor_equity = INITIAL_CAPITAL
    current_month: str | None = None
    month_blocked = False
    prev_equity = INITIAL_CAPITAL

    for date in dates:
        month = str(date.to_period("M"))
        if current_month != month:
            current_month = month
            month_anchor_equity = prev_equity
            month_blocked = False

        existing_codes = {p["code"] for p in positions}
        orders = sorted(entry_orders.get(date, []), key=lambda x: (x["edge_score"], x["adv20"]), reverse=True)
        if month_blocked:
            orders = []
        slots = max(0, MAX_POSITIONS - len(positions))
        for order in orders:
            if slots <= 0:
                break
            code = str(order["code"])
            pattern = str(order["pattern"])
            if code in existing_codes:
                continue
            if pattern_cooldown_until.get(pattern, pd.Timestamp.min) >= date:
                continue
            entry = float(order["entry_price"])
            if entry <= 0:
                continue
            stop_pct = float(order["stop_pct"])
            regime_scale = 1.0 if order["regime"] in {"bull", "transition", "neutral"} else 0.55
            risk_budget = prev_equity * RISK_PER_TRADE * regime_scale
            desired = min(prev_equity * MAX_POSITION_WEIGHT * regime_scale, risk_budget / max(stop_pct, 0.01))
            open_notional = sum(p["shares"] * p["entry_price"] for p in positions)
            desired = min(desired, max(0.0, prev_equity * MAX_GROSS_EXPOSURE - open_notional))
            affordable = cash / (1.0 + BUY_COST * cost_mult)
            desired = min(desired, affordable)
            shares = math.floor(desired / entry)
            if shares < 1:
                continue
            notional = shares * entry
            buy_fee = notional * BUY_COST * cost_mult
            if notional + buy_fee > cash:
                continue
            cash -= notional + buy_fee
            pos = dict(order)
            pos.update({"shares": shares, "notional": notional, "buy_fee": buy_fee})
            positions.append(pos)
            existing_codes.add(code)
            slots -= 1

        remaining: list[dict[str, Any]] = []
        for pos in positions:
            if pd.Timestamp(pos["exit_date"]) != date:
                remaining.append(pos)
                continue
            exit_price = float(pos["exit_price"])
            gross = pos["shares"] * exit_price
            sell_fee = gross * SELL_COST * cost_mult
            cash += gross - sell_fee
            pnl = gross - sell_fee - pos["notional"] - pos["buy_fee"]
            net_ret = pnl / pos["notional"] if pos["notional"] else 0.0
            trade = {
                "market": pos["market"], "code": pos["code"], "name": pos["name"],
                "pattern": pos["pattern"], "regime": pos["regime"],
                "signal_date": str(pd.Timestamp(pos["date"]).date()),
                "entry_date": str(pd.Timestamp(pos["entry_date"]).date()),
                "exit_date": str(date.date()), "entry_price": pos["entry_price"],
                "exit_price": exit_price, "shares": pos["shares"], "notional": pos["notional"],
                "pnl": pnl, "net_return_on_position": net_ret, "exit_reason": pos["exit_reason"],
                "pred_return": pos["pred_return"], "pred_probability": pos["pred_probability"],
                "edge_score": pos["edge_score"], "mae": pos["mae"], "mfe": pos["mfe"],
                "cost_mult": cost_mult,
            }
            trades.append(trade)
            pattern = str(pos["pattern"])
            if pnl < 0:
                pattern_loss_streak[pattern] = pattern_loss_streak.get(pattern, 0) + 1
                if pattern_loss_streak[pattern] >= 2:
                    pattern_cooldown_until[pattern] = date + pd.Timedelta(days=7)
                    pattern_loss_streak[pattern] = 0
            else:
                pattern_loss_streak[pattern] = 0
        positions = remaining

        marked = 0.0
        for pos in positions:
            px = close_map.get((date, str(pos["code"])), float(pos["entry_price"]))
            marked += pos["shares"] * px
        equity = cash + marked
        daily_ret = equity / prev_equity - 1.0 if prev_equity else 0.0
        if month_anchor_equity > 0 and equity / month_anchor_equity - 1.0 <= MONTHLY_LOSS_CUT:
            month_blocked = True
        regimes = [str(p["regime"]) for p in positions]
        equity_rows.append({
            "date": date, "equity": equity, "cash": cash, "open_positions": len(positions),
            "daily_return": daily_ret, "month_blocked": month_blocked,
            "active_regimes": ",".join(sorted(set(regimes))),
        })
        prev_equity = equity

    last_date = dates[-1]
    for pos in positions:
        px = close_map.get((last_date, str(pos["code"])), float(pos["entry_price"]))
        gross = pos["shares"] * px
        sell_fee = gross * SELL_COST * cost_mult
        cash += gross - sell_fee
        pnl = gross - sell_fee - pos["notional"] - pos["buy_fee"]
        trades.append({
            "market": pos["market"], "code": pos["code"], "name": pos["name"], "pattern": pos["pattern"],
            "regime": pos["regime"], "signal_date": str(pd.Timestamp(pos["date"]).date()),
            "entry_date": str(pd.Timestamp(pos["entry_date"]).date()), "exit_date": str(last_date.date()),
            "entry_price": pos["entry_price"], "exit_price": px, "shares": pos["shares"],
            "notional": pos["notional"], "pnl": pnl, "net_return_on_position": pnl / pos["notional"],
            "exit_reason": "period_end", "pred_return": pos["pred_return"],
            "pred_probability": pos["pred_probability"], "edge_score": pos["edge_score"],
            "mae": pos["mae"], "mfe": pos["mfe"], "cost_mult": cost_mult,
        })
    if positions:
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["open_positions"] = 0

    tr = pd.DataFrame(trades)
    eq = pd.DataFrame(equity_rows)
    if eq.empty:
        raise RuntimeError("No equity rows")
    eq["daily_return"] = eq["equity"].pct_change()
    eq.loc[eq.index[0], "daily_return"] = eq.loc[eq.index[0], "equity"] / INITIAL_CAPITAL - 1.0
    eq["daily_return"] = eq["daily_return"].fillna(0.0)
    eq["peak"] = eq["equity"].cummax()
    eq["drawdown"] = eq["equity"] / eq["peak"] - 1.0
    ret = float(eq["equity"].iloc[-1] / INITIAL_CAPITAL - 1.0)
    gross_profit = float(tr.loc[tr["pnl"] > 0, "pnl"].sum()) if not tr.empty else 0.0
    gross_loss = float(-tr.loc[tr["pnl"] < 0, "pnl"].sum()) if not tr.empty else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    monthly = month_returns(eq)
    monthly_values = list(monthly.values())
    geom = float(np.prod([1 + x for x in monthly_values]) ** (1 / len(monthly_values)) - 1) if monthly_values else 0.0
    metrics = {
        "start": str(start.date()), "end": str(end.date()), "return": ret,
        "final_equity": float(eq["equity"].iloc[-1]), "net_profit": float(eq["equity"].iloc[-1] - INITIAL_CAPITAL),
        "profit_factor": float(pf), "max_drawdown": float(eq["drawdown"].min()),
        "trades": int(len(tr)), "wins": int((tr["pnl"] > 0).sum()) if not tr.empty else 0,
        "win_rate": float((tr["pnl"] > 0).mean()) if not tr.empty else 0.0,
        "monthly_geom": geom, "monthly_median": float(np.median(monthly_values)) if monthly_values else 0.0,
        "positive_month_ratio": float(np.mean(np.array(monthly_values) > 0)) if monthly_values else 0.0,
        "monthly_returns": monthly, "benchmark_returns": benchmark_return(bench, start, end),
        "threshold_config": asdict(cfg), "edge_threshold": float(edge_threshold), "cost_mult": cost_mult,
    }
    if not tr.empty:
        metrics["pattern_stats"] = {
            str(k): {
                "trades": int(len(x)), "pnl": float(x["pnl"].sum()),
                "pf": float(x.loc[x["pnl"] > 0, "pnl"].sum() / max(1e-9, -x.loc[x["pnl"] < 0, "pnl"].sum())),
                "win_rate": float((x["pnl"] > 0).mean()),
            }
            for k, x in tr.groupby("pattern")
        }
        metrics["regime_stats"] = {
            str(k): {
                "trades": int(len(x)), "pnl": float(x["pnl"].sum()),
                "pf": float(x.loc[x["pnl"] > 0, "pnl"].sum() / max(1e-9, -x.loc[x["pnl"] < 0, "pnl"].sum())),
                "win_rate": float((x["pnl"] > 0).mean()),
            }
            for k, x in tr.groupby("regime")
        }
        metrics["market_stats"] = {
            str(k): {"trades": int(len(x)), "pnl": float(x["pnl"].sum()), "win_rate": float((x["pnl"] > 0).mean())}
            for k, x in tr.groupby("market")
        }
    return metrics, tr, eq


def threshold_grid(scored_2024: pd.DataFrame) -> list[tuple[ThresholdConfig, float]]:
    valid_edges = scored_2024["edge_score"].replace([np.inf, -np.inf], np.nan).dropna()
    out: list[tuple[ThresholdConfig, float]] = []
    for q in [0.70, 0.78, 0.84, 0.89, 0.93, 0.96]:
        edge = float(valid_edges.quantile(q))
        for p in [0.50, 0.55, 0.60]:
            for n in [1, 2]:
                out.append((ThresholdConfig(q, p, n, 0.05), edge))
    return out


def calibration_score(m: dict[str, Any]) -> float:
    if m["trades"] < 18 or m["profit_factor"] < 1.0 or m["max_drawdown"] < -0.25:
        return -999.0
    return (
        2.0 * m["return"]
        + 1.2 * m["monthly_geom"]
        + 0.12 * min(m["profit_factor"], 3.0)
        + 0.08 * m["positive_month_ratio"]
        + 0.8 * min(0.0, m["max_drawdown"] + 0.12)
    )
