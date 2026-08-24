def _v6_trade_stats(x: pd.DataFrame) -> dict[str, Any]:
    if x.empty:
        return {'trades':0,'pnl':0.0,'pf':0.0,'win_rate':0.0,'mean_return':0.0}
    return {
        'trades': int(len(x)),
        'pnl': float(x['pnl'].sum()),
        'pf': float(_v6_pf(x['pnl'])),
        'win_rate': float((x['pnl'] > 0).mean()),
        'mean_return': float(x['net_return_on_position'].mean()),
    }


def run_v6_portfolio(
    stock_signals: pd.DataFrame,
    index_signals: pd.DataFrame,
    features: pd.DataFrame,
    etfs: pd.DataFrame,
    bench: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: V6Config,
    *,
    cost_mult: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(start); end = pd.Timestamp(end)
    s = stock_signals[
        stock_signals['entry_date'].between(start, end)
        & stock_signals['exit_date'].between(start, end)
    ].copy()
    if not s.empty:
        s['asset_type'] = 'stock'
        s = s.sort_values(['date', 'signal_score', 'adv20'], ascending=[True, False, False])
        s = s.groupby('date', group_keys=False).head(max(3, cfg.top_n * 4))
    ix = index_signals[
        index_signals['entry_date'].between(start, end)
        & index_signals['exit_date'].between(start, end)
    ].copy() if not index_signals.empty else pd.DataFrame()

    stock_orders: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for _, r in s.iterrows():
        stock_orders.setdefault(pd.Timestamp(r['entry_date']), []).append(r.to_dict())
    index_orders: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    if not ix.empty:
        for _, r in ix.iterrows():
            index_orders.setdefault(pd.Timestamp(r['entry_date']), []).append(r.to_dict())

    stock_close = {
        (pd.Timestamp(r.date), str(r.code)): float(r.adj_close)
        for r in features[features['date'].between(start, end)][['date','code','adj_close']].itertuples(index=False)
    }
    etf_close = {
        (pd.Timestamp(r.date), str(r.code)): float(r.close)
        for r in etfs[etfs['date'].between(start, end)][['date','code','close']].itertuples(index=False)
    }
    dates = sorted(pd.Timestamp(x) for x in bench[bench['date'].between(start, end)]['date'].unique())
    if not dates:
        raise RuntimeError('V6: no simulation dates')

    cash = INITIAL_CAPITAL
    positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    pattern_loss_streak: dict[str, int] = {}
    pattern_cooldown: dict[str, pd.Timestamp] = {}
    market_loss_streak: dict[str, int] = {}
    market_cooldown: dict[str, pd.Timestamp] = {}
    current_month: str | None = None
    month_anchor = INITIAL_CAPITAL
    month_blocked = False
    prev_equity = INITIAL_CAPITAL

    def mark_position(pos: dict[str, Any], date: pd.Timestamp) -> float:
        if pos['asset_type'] == 'index':
            return pos['shares'] * etf_close.get((date, str(pos['code'])), pos['entry_price'])
        return pos['shares'] * stock_close.get((date, str(pos['code'])), pos['entry_price'])

    for date in dates:
        month = str(date.to_period('M'))
        if month != current_month:
            current_month = month
            month_anchor = prev_equity
            month_blocked = False

        marked_before = sum(mark_position(p, date) for p in positions)
        equity_before = cash + marked_before
        existing_codes = {str(p['code']) for p in positions}
        gross_before = sum(mark_position(p, date) for p in positions)

        # Entries are processed before same-day exits. This prevents reusing capital that
        # becomes available only after an intraday stop/target or the closing auction.
        has_index = any(p['asset_type'] == 'index' for p in positions)
        if not month_blocked and not has_index:
            orders = sorted(index_orders.get(date, []), key=lambda z: float(z.get('signal_score', 0.0)), reverse=True)
            for order in orders:
                regime = str(order['regime'])
                weight = cfg.index_weight if regime == 'bull' else cfg.transition_index_weight
                if weight <= 0:
                    continue
                entry = float(order['entry_price'])
                desired = min(equity_before * weight, max(0.0, equity_before * V6_MAX_GROSS - gross_before))
                affordable = cash / (1.0 + V6_ETF_BUY_COST * cost_mult)
                desired = min(desired, affordable)
                shares = int(math.floor(desired / entry))
                if shares < 1:
                    continue
                notional = shares * entry
                fee = notional * V6_ETF_BUY_COST * cost_mult
                cash -= notional + fee
                pos = dict(order)
                pos.update({'shares':shares,'notional':notional,'buy_fee':fee})
                positions.append(pos)
                existing_codes.add(str(order['code']))
                gross_before += notional
                break

        if not month_blocked:
            open_stock_count = sum(p['asset_type'] == 'stock' for p in positions)
            slots = max(0, V6_MAX_STOCK_POSITIONS - open_stock_count)
            orders = sorted(
                stock_orders.get(date, []),
                key=lambda z: (float(z.get('signal_score',0.0)), float(z.get('adv20',0.0))),
                reverse=True,
            )
            entered_today = 0
            for order in orders:
                if slots <= 0 or entered_today >= cfg.top_n:
                    break
                code = str(order['code']); pattern = str(order['pattern']); market = str(order['market'])
                if code in existing_codes:
                    continue
                if pattern_cooldown.get(pattern, pd.Timestamp.min) >= date:
                    continue
                if market_cooldown.get(market, pd.Timestamp.min) >= date:
                    continue
                entry = float(order['entry_price']); stop_pct = float(order['stop_pct'])
                if entry <= 0 or stop_pct <= 0:
                    continue
                regime_scale = 1.0 if order['regime'] == 'bull' else (0.86 if order['regime'] in {'transition','neutral'} else 0.58)
                risk_budget = equity_before * cfg.risk_per_trade * regime_scale
                desired = min(equity_before * V6_MAX_STOCK_WEIGHT * regime_scale, risk_budget / stop_pct)
                desired = min(desired, max(0.0, equity_before * V6_MAX_GROSS - gross_before))
                affordable = cash / (1.0 + BUY_COST * cost_mult)
                desired = min(desired, affordable)
                shares = int(math.floor(desired / entry))
                if shares < 1:
                    continue
                notional = shares * entry
                fee = notional * BUY_COST * cost_mult
                if notional + fee > cash:
                    continue
                cash -= notional + fee
                pos = dict(order)
                pos.update({'shares':shares,'notional':notional,'buy_fee':fee})
                positions.append(pos)
                existing_codes.add(code)
                gross_before += notional
                slots -= 1
                entered_today += 1

        remaining: list[dict[str, Any]] = []
        for pos in positions:
            if pd.Timestamp(pos['exit_date']) != date:
                remaining.append(pos)
                continue
            exit_price = float(pos['exit_price'])
            gross = pos['shares'] * exit_price
            sell_rate = V6_ETF_SELL_COST if pos['asset_type'] == 'index' else SELL_COST
            sell_fee = gross * sell_rate * cost_mult
            cash += gross - sell_fee
            pnl = gross - sell_fee - pos['notional'] - pos['buy_fee']
            net_ret = pnl / pos['notional'] if pos['notional'] else 0.0
            trades.append({
                'asset_type':pos['asset_type'],'market':pos['market'],'code':pos['code'],'name':pos['name'],
                'pattern':pos['pattern'],'regime':pos['regime'],'signal_date':str(pd.Timestamp(pos['date']).date()),
                'entry_date':str(pd.Timestamp(pos['entry_date']).date()),'exit_date':str(date.date()),
                'entry_price':pos['entry_price'],'exit_price':exit_price,'shares':pos['shares'],
                'notional':pos['notional'],'pnl':pnl,'net_return_on_position':net_ret,
                'exit_reason':pos['exit_reason'],'signal_score':pos.get('signal_score',0.0),
                'quality_rank':pos.get('quality_rank',np.nan),'robust_edge':pos.get('robust_edge',np.nan),
                'historical_pf':pos.get('pf',np.nan),'historical_n':pos.get('n',np.nan),
                'mae':pos.get('mae',np.nan),'mfe':pos.get('mfe',np.nan),'cost_mult':cost_mult,
            })
            if pos['asset_type'] == 'stock':
                pattern = str(pos['pattern']); market = str(pos['market'])
                if pnl < 0:
                    pattern_loss_streak[pattern] = pattern_loss_streak.get(pattern,0) + 1
                    market_loss_streak[market] = market_loss_streak.get(market,0) + 1
                    if pattern_loss_streak[pattern] >= 2:
                        pattern_cooldown[pattern] = date + pd.Timedelta(days=7)
                        pattern_loss_streak[pattern] = 0
                    if market_loss_streak[market] >= 3:
                        market_cooldown[market] = date + pd.Timedelta(days=5)
                        market_loss_streak[market] = 0
                else:
                    pattern_loss_streak[pattern] = 0
                    market_loss_streak[market] = 0
        positions = remaining

        marked = sum(mark_position(p, date) for p in positions)
        equity = cash + marked
        daily_ret = equity / prev_equity - 1.0 if prev_equity > 0 else 0.0
        if month_anchor > 0 and equity / month_anchor - 1.0 <= V6_MONTHLY_LOSS_CUT:
            month_blocked = True
        equity_rows.append({
            'date':date,'equity':equity,'cash':cash,'open_positions':len(positions),
            'daily_return':daily_ret,'month_blocked':month_blocked,
            'stock_positions':sum(p['asset_type']=='stock' for p in positions),
            'index_positions':sum(p['asset_type']=='index' for p in positions),
        })
        prev_equity = equity

    last_date = dates[-1]
    for pos in positions:
        px = mark_position(pos, last_date) / pos['shares']
        gross = pos['shares'] * px
        sell_rate = V6_ETF_SELL_COST if pos['asset_type'] == 'index' else SELL_COST
        sell_fee = gross * sell_rate * cost_mult
        cash += gross - sell_fee
        pnl = gross - sell_fee - pos['notional'] - pos['buy_fee']
        trades.append({
            'asset_type':pos['asset_type'],'market':pos['market'],'code':pos['code'],'name':pos['name'],
            'pattern':pos['pattern'],'regime':pos['regime'],'signal_date':str(pd.Timestamp(pos['date']).date()),
            'entry_date':str(pd.Timestamp(pos['entry_date']).date()),'exit_date':str(last_date.date()),
            'entry_price':pos['entry_price'],'exit_price':px,'shares':pos['shares'],'notional':pos['notional'],
            'pnl':pnl,'net_return_on_position':pnl/pos['notional'],'exit_reason':'period_end',
            'signal_score':pos.get('signal_score',0.0),'quality_rank':pos.get('quality_rank',np.nan),
            'robust_edge':pos.get('robust_edge',np.nan),'historical_pf':pos.get('pf',np.nan),
            'historical_n':pos.get('n',np.nan),'mae':pos.get('mae',np.nan),'mfe':pos.get('mfe',np.nan),
            'cost_mult':cost_mult,
        })
    if positions:
        equity_rows[-1]['equity'] = cash
        equity_rows[-1]['cash'] = cash
        equity_rows[-1]['open_positions'] = 0
        equity_rows[-1]['stock_positions'] = 0
        equity_rows[-1]['index_positions'] = 0

    tr = pd.DataFrame(trades)
    eq = pd.DataFrame(equity_rows)
    eq['daily_return'] = eq['equity'].pct_change().fillna(eq['equity']/INITIAL_CAPITAL-1.0)
    eq['peak'] = eq['equity'].cummax()
    eq['drawdown'] = eq['equity']/eq['peak']-1.0
    ret = float(eq['equity'].iloc[-1]/INITIAL_CAPITAL-1.0)
    pf = float(_v6_pf(tr['pnl'])) if not tr.empty else 0.0
    eq['_month'] = eq['date'].dt.to_period('M')
    monthly_s = eq.groupby('_month')['daily_return'].apply(lambda z: float((1.0+z).prod()-1.0))
    monthly = {str(k):float(v) for k,v in monthly_s.items()}
    geom = float((1.0+monthly_s).prod()**(1.0/len(monthly_s))-1.0) if len(monthly_s) else 0.0
    bench_ret = benchmark_return(bench,start,end)
    best_bench = max(bench_ret.values()) if bench_ret else 0.0
    opportunity_adjusted = ret - 0.35*max(0.0,best_bench-ret)
    metrics: dict[str,Any] = {
        'start':str(start.date()),'end':str(end.date()),'return':ret,
        'final_equity':float(eq['equity'].iloc[-1]),'net_profit':float(eq['equity'].iloc[-1]-INITIAL_CAPITAL),
        'profit_factor':pf,'max_drawdown':float(eq['drawdown'].min()),'trades':int(len(tr)),
        'wins':int((tr['pnl']>0).sum()) if not tr.empty else 0,
        'win_rate':float((tr['pnl']>0).mean()) if not tr.empty else 0.0,
        'monthly_geom':geom,'monthly_median':float(monthly_s.median()) if len(monthly_s) else 0.0,
        'positive_month_ratio':float((monthly_s>0).mean()) if len(monthly_s) else 0.0,
        'monthly_returns':monthly,'benchmark_returns':bench_ret,
        'opportunity_adjusted':opportunity_adjusted,'cost_mult':cost_mult,
        'config':_v6_asdict(cfg),
    }
    if not tr.empty:
        metrics['asset_stats']={str(k):_v6_trade_stats(x) for k,x in tr.groupby('asset_type')}
        metrics['pattern_stats']={str(k):_v6_trade_stats(x) for k,x in tr.groupby('pattern')}
        metrics['regime_stats']={str(k):_v6_trade_stats(x) for k,x in tr.groupby('regime')}
        metrics['market_stats']={str(k):_v6_trade_stats(x) for k,x in tr.groupby('market')}
    return metrics,tr,eq.drop(columns=['_month'])


def v6_config_grid() -> list[V6Config]:
    signal_gates = [
        (18,1.00,0.0000,0.85,0.55,0.50,False),
        (24,1.05,0.0000,0.90,0.60,0.50,False),
        (30,1.08,0.0005,0.95,0.65,0.50,False),
        (38,1.12,0.0008,1.00,0.68,0.60,False),
        (48,1.18,0.0012,1.05,0.72,0.67,False),
        (24,1.08,0.0005,0.95,0.65,0.50,True),
        (36,1.15,0.0010,1.00,0.70,0.60,True),
    ]
    portfolios = [
        (1,0.0125,0.00,0.00),
        (1,0.0175,0.00,0.00),
        (1,0.0225,0.00,0.00),
        (2,0.0125,0.00,0.00),
        (2,0.0175,0.00,0.00),
        (1,0.0150,0.35,0.00),
        (1,0.0175,0.50,0.00),
        (1,0.0200,0.55,0.15),
        (2,0.0125,0.35,0.00),
        (2,0.0150,0.45,0.15),
    ]
    out=[]
    for n,pf,edge,rpf,q,yr,panic in signal_gates:
        for top,risk,iw,tw in portfolios:
            out.append(V6Config(n,pf,edge,rpf,q,yr,top,risk,iw,tw,panic))
    return out


def score_v6_development(year_metrics: list[dict[str,Any]]) -> float:
    returns=np.array([m['return'] for m in year_metrics],dtype=float)
    pfs=np.array([m['profit_factor'] for m in year_metrics],dtype=float)
    mdds=np.array([m['max_drawdown'] for m in year_metrics],dtype=float)
    trades=sum(int(m['trades']) for m in year_metrics)
    positive=float((returns>0).mean())
    compound=float(np.prod(1.0+returns)-1.0)
    worst=float(returns.min())
    median=float(np.median(returns))
    opportunity=float(np.mean([m['opportunity_adjusted'] for m in year_metrics]))
    penalty=0.0
    if trades<30: penalty+=(30-trades)*0.015
    if positive<2/3: penalty+=(2/3-positive)*0.8
    if worst<-0.16: penalty+=abs(worst+0.16)*2.5
    if np.nanmedian(pfs)<1.0: penalty+=(1.0-np.nanmedian(pfs))*0.5
    return 1.8*compound+0.8*median+0.25*positive+0.15*np.nanmedian(np.minimum(pfs,3.0))+0.35*opportunity+0.8*np.minimum(0.0,mdds+0.12).sum()-penalty
