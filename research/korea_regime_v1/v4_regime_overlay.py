from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path('/tmp/korea_regime_v4')
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location('korea_core_v4', ROOT / 'run_v1.py')
if spec is None or spec.loader is None:
    raise RuntimeError('unable to import core')
core = importlib.util.module_from_spec(spec)
sys.modules['korea_core_v4'] = core
spec.loader.exec_module(core)

# Rolling public dataset. It is regenerated after each Korean trading day.
core.DATASET_REPO = 'aikstockdata/korea-equity-daily'

DEV_START = pd.Timestamp('2025-10-20')
DEV_END = pd.Timestamp('2025-12-31')
VALID_START = pd.Timestamp('2026-01-01')
VALID_END = pd.Timestamp('2026-04-30')
FINAL_START = pd.Timestamp('2026-05-01')
FINAL_END = pd.Timestamp('2026-08-21')
INITIAL_CAPITAL = 2_000_000.0
BUY_COST = 0.0005
STOCK_SELL_COST = 0.0025
INDEX_SELL_COST = 0.0005


@dataclass(frozen=True)
class Config:
    bull_hold: int
    transition_hold: int
    n_select: int
    bull_stock_weight: float
    bull_index_weight: float
    transition_stock_weight: float
    transition_index_weight: float
    rebound_stock_weight: float
    overheat_sigma: float
    bull_style: str
    transition_style: str

    @property
    def config_id(self) -> str:
        return (
            f'v4-bh{self.bull_hold}-th{self.transition_hold}-n{self.n_select}'
            f'-bs{self.bull_stock_weight:.2f}-bi{self.bull_index_weight:.2f}'
            f'-ts{self.transition_stock_weight:.2f}-ti{self.transition_index_weight:.2f}'
            f'-rb{self.rebound_stock_weight:.2f}-oh{self.overheat_sigma:.1f}'
            f'-b{self.bull_style[0]}-t{self.transition_style[0]}'
        )


@dataclass
class StockPosition:
    code: str
    name: str
    regime: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    invested: float
    buy_cost: float
    hold_days: int
    age: int = 0


@dataclass
class IndexPosition:
    regime: str
    entry_date: pd.Timestamp
    entry_price: float
    units: float
    invested: float
    buy_cost: float
    hold_days: int
    age: int = 0


def enrich_regime(features: pd.DataFrame, bench: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = bench.sort_values(['market', 'date']).copy()
    g = b.groupby('market', group_keys=False)
    b['bench_ma5'] = g['bench_close'].transform(lambda s: s.rolling(5, min_periods=5).mean())
    b['breadth_delta5'] = g['breadth20'].diff(5)
    b['bench_ret3'] = g['bench_close'].pct_change(3)
    b['bench_ret10'] = g['bench_close'].pct_change(10)
    cols = ['market', 'date', 'bench_ma5', 'breadth_delta5', 'bench_ret3', 'bench_ret10']
    p = features.merge(b[cols], on=['market', 'date'], how='left')
    return p, b


def candidate_scores(day: pd.DataFrame, style: str, overheat_sigma: float) -> pd.DataFrame:
    if day.empty:
        return day
    market = str(day['market'].iloc[0])
    min_adv = 3_000_000_000 if market == 'KOSPI' else 1_500_000_000
    d = day.copy()
    eligible = (
        (d['history_n'] >= 60)
        & (d['close'] >= 2000)
        & (d['adv20'] >= min_adv)
        & (d['max_abs_ret20'] <= 0.32)
        & d['vol20'].between(0.007, 0.10)
        & d['ma20'].notna()
        & d['ma60'].notna()
        & d['beta20'].notna()
        & (d['dist_ma20_sigma'] <= overheat_sigma)
        & (d['ret1'].abs() <= 0.20)
    )
    d = d[eligible].copy()
    if d.empty:
        return d

    if style == 'bull_continuous':
        cond = (
            (d['close'] > d['ma20'])
            & (d['ma20'] >= d['ma60'] * 0.985)
            & (d['ret20'] > 0.025)
            & (d['rel20'] > 0.015)
            & (d['rank_rel20'] >= 0.70)
            & (d['ret5'] > -0.025)
            & (d['ret1'] > -0.06)
            & (d['volume_ratio'] >= 0.55)
        )
        d['score'] = (
            0.30 * d['rank_rel20']
            + 0.18 * d['rank_rel60']
            + 0.15 * d['rank_ret20']
            + 0.12 * d['rank_ret60']
            + 0.10 * d['rank_volume_ratio']
            + 0.08 * d['rank_adv20']
            + 0.07 * d['rank_low_vol20']
        )
    elif style == 'bull_pullback':
        cond = (
            (d['close'] > d['ma20'] * 0.985)
            & (d['ma20'] >= d['ma60'] * 0.985)
            & (d['ret20'] > 0.02)
            & (d['rel20'] > 0.01)
            & d['ret3'].between(-0.09, 0.015)
            & d['drawdown20'].between(-0.15, -0.003)
            & (d['ret1'] > -0.045)
            & (d['volume_ratio'] >= 0.50)
        )
        d['score'] = (
            0.28 * d['rank_rel20']
            + 0.18 * d['rank_rel60']
            + 0.14 * d['rank_ret60']
            + 0.12 * d['rank_volume_ratio']
            + 0.10 * d['rank_adv20']
            + 0.10 * d['rank_low_vol20']
            + 0.08 * (1.0 - (d['ret3'].abs() / 0.09).clip(0, 1))
        )
    elif style == 'transition_defensive':
        cond = (
            (d['ret60'] > 0.0)
            & (d['rel20'] > 0.025)
            & (d['rank_rel20'] >= 0.82)
            & (d['beta20'] <= 0.90)
            & (d['close'] >= d['ma20'] * 0.985)
            & (d['ret5'] > -0.055)
            & (d['ret1'] > -0.025)
            & (d['volume_ratio'] >= 0.65)
        )
        d['score'] = (
            0.34 * d['rank_rel20']
            + 0.18 * d['rank_rel60']
            + 0.13 * d['rank_ret60']
            + 0.12 * d['rank_low_vol20']
            + 0.11 * d['rank_volume_ratio']
            + 0.07 * d['rank_adv20']
            + 0.05 * (1.0 - d['beta20'].clip(0, 2) / 2.0)
        )
    elif style == 'transition_rebound':
        cond = (
            (d['ret60'] > 0.0)
            & (d['rel20'] > 0.02)
            & (d['rank_rel20'] >= 0.78)
            & d['ret5'].between(-0.12, -0.008)
            & (d['ret1'] >= 0.012)
            & (d['close'] >= d['ma10'] * 0.975)
            & (d['volume_ratio'] >= 0.85)
            & (d['beta20'] <= 1.20)
        )
        d['score'] = (
            0.31 * d['rank_rel20']
            + 0.16 * d['rank_rel60']
            + 0.14 * d['rank_volume_ratio']
            + 0.12 * d['rank_adv20']
            + 0.11 * d['rank_low_vol20']
            + 0.10 * (1.0 - (d['ret5'].abs() / 0.12).clip(0, 1))
            + 0.06 * (1.0 - d['beta20'].clip(0, 2) / 2.0)
        )
    elif style == 'panic_rebound':
        cond = (
            (d['ret20'] > 0.015)
            & (d['rel20'] > 0.08)
            & (d['rank_rel20'] >= 0.95)
            & (d['close'] > d['ma20'])
            & (d['ret1'] >= 0.018)
            & (d['ret5'] > -0.04)
            & (d['volume_ratio'] >= 1.10)
            & (d['beta20'] <= 0.85)
        )
        d['score'] = (
            0.38 * d['rank_rel20']
            + 0.18 * d['rank_rel5']
            + 0.14 * d['rank_volume_ratio']
            + 0.10 * d['rank_adv20']
            + 0.10 * d['rank_low_vol20']
            + 0.10 * (1.0 - d['beta20'].clip(0, 2) / 2.0)
        )
    else:
        return d.iloc[0:0]

    return d[cond & d['score'].notna()].sort_values(
        ['score', 'adv20'], ascending=[False, False]
    )


def get_weights(day: pd.DataFrame, cfg: Config) -> tuple[float, float, int, str]:
    regime = str(day['regime'].dropna().iloc[0]) if day['regime'].notna().any() else 'unknown'
    row = day.iloc[0]
    vol_scale = 0.65 if float(row.get('bench_vol20', 0.0) or 0.0) >= 0.03 else 1.0

    if regime == 'bull' and row['bench_close'] > row['bench_ma20'] and row['bench_ret5'] > 0:
        return (
            cfg.bull_stock_weight * vol_scale,
            cfg.bull_index_weight * vol_scale,
            cfg.bull_hold,
            cfg.bull_style,
        )

    if regime in {'transition', 'neutral'}:
        improving = (
            row['bench_ret1'] > 0
            and row['bench_ret3'] > -0.02
            and (pd.isna(row['breadth_delta5']) or row['breadth_delta5'] >= -0.04)
        )
        if improving:
            return (
                cfg.transition_stock_weight * vol_scale,
                cfg.transition_index_weight * vol_scale,
                cfg.transition_hold,
                cfg.transition_style,
            )
        return (0.0, 0.0, 1, cfg.transition_style)

    if regime in {'bear', 'panic'}:
        confirmed_rebound = (
            row['bench_ret1'] >= 0.018
            and row['bench_ret3'] >= 0.015
            and row['bench_close'] > row['bench_ma5']
            and row['breadth20'] >= 0.42
            and (pd.isna(row['breadth_delta5']) or row['breadth_delta5'] >= 0.06)
        )
        if confirmed_rebound:
            return (cfg.rebound_stock_weight * 0.65, 0.0, 1, 'panic_rebound')
        return (0.0, 0.0, 1, 'panic_rebound')

    return (0.0, 0.0, 1, cfg.transition_style)


def simulate(
    market: str,
    cfg: Config,
    features: pd.DataFrame,
    bench: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    stress: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    data = features[(features['market'] == market) & features['date'].between(start, end)].copy()
    if data.empty:
        raise RuntimeError(f'no data for {market} {start} {end}')
    dates = [pd.Timestamp(x) for x in sorted(data['date'].unique())]
    by_date = {pd.Timestamp(d): x.copy() for d, x in data.groupby('date')}
    close_map = {
        (pd.Timestamp(r.date), str(r.code)): float(r.close)
        for r in data[['date', 'code', 'close']].itertuples(index=False)
    }
    b = bench[(bench['market'] == market) & bench['date'].between(start, end)].copy()
    b = b.set_index('date').reindex(dates).ffill().reset_index()
    bench_close = {pd.Timestamp(r.date): float(r.bench_close) for r in b.itertuples(index=False)}
    b['benchmark_daily_return'] = b['bench_close'].pct_change().fillna(0.0)
    bench_ret_map = {pd.Timestamp(r.date): float(r.benchmark_daily_return) for r in b.itertuples(index=False)}

    cash = INITIAL_CAPITAL
    stocks: list[StockPosition] = []
    indexes: list[IndexPosition] = []
    pending: dict[pd.Timestamp, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    eq_rows: list[dict[str, Any]] = []
    prev_equity = INITIAL_CAPITAL

    for i, date in enumerate(dates):
        day = by_date[date]
        order = pending.pop(date, None)
        if order:
            equity_before = cash + sum(
                p.shares * close_map.get((date, p.code), p.entry_price) for p in stocks
            ) + sum(
                p.units * bench_close.get(date, p.entry_price) for p in indexes
            )
            stock_total = equity_before * order['stock_weight'] / max(1, order['hold_days'])
            candidates = order['candidates']
            if stock_total > 0 and not candidates.empty:
                each = stock_total / len(candidates)
                for r in candidates.itertuples(index=False):
                    px = close_map.get((date, str(r.code)))
                    if px is None or px <= 0:
                        continue
                    max_cash = cash / (1.0 + BUY_COST * stress)
                    notional = min(each, max_cash)
                    shares = int(math.floor(notional / px))
                    if shares < 1:
                        continue
                    invested = shares * px
                    buy_cost = invested * BUY_COST * stress
                    if invested + buy_cost > cash:
                        continue
                    cash -= invested + buy_cost
                    stocks.append(StockPosition(
                        code=str(r.code), name=str(r.name_ko), regime=order['regime'],
                        entry_date=date, entry_price=px, shares=shares,
                        invested=invested, buy_cost=buy_cost, hold_days=order['hold_days'],
                    ))

            index_total = equity_before * order['index_weight'] / max(1, order['hold_days'])
            if index_total > 0:
                px = bench_close.get(date)
                if px and px > 0:
                    max_cash = cash / (1.0 + BUY_COST * stress)
                    invested = min(index_total, max_cash)
                    if invested >= 100_000:
                        buy_cost = invested * BUY_COST * stress
                        cash -= invested + buy_cost
                        indexes.append(IndexPosition(
                            regime=order['regime'], entry_date=date, entry_price=px,
                            units=invested / px, invested=invested, buy_cost=buy_cost,
                            hold_days=order['hold_days'],
                        ))

        kept_stocks: list[StockPosition] = []
        for p in stocks:
            px = close_map.get((date, p.code))
            if px is None:
                kept_stocks.append(p)
                continue
            if date > p.entry_date:
                p.age += 1
            if p.age >= p.hold_days:
                gross = p.shares * px
                sell_cost = gross * STOCK_SELL_COST * stress
                pnl = gross - sell_cost - p.invested - p.buy_cost
                cash += gross - sell_cost
                trades.append({
                    'market': market, 'asset_type': 'stock', 'config_id': cfg.config_id,
                    'code': p.code, 'name': p.name, 'regime': p.regime,
                    'entry_date': str(p.entry_date.date()), 'exit_date': str(date.date()),
                    'entry_price': p.entry_price, 'exit_price': px, 'hold_days': p.age,
                    'net_pnl': pnl, 'net_return_on_position': pnl / p.invested,
                })
            else:
                kept_stocks.append(p)
        stocks = kept_stocks

        kept_indexes: list[IndexPosition] = []
        for p in indexes:
            px = bench_close.get(date)
            if px is None:
                kept_indexes.append(p)
                continue
            if date > p.entry_date:
                p.age += 1
            if p.age >= p.hold_days:
                gross = p.units * px
                sell_cost = gross * INDEX_SELL_COST * stress
                pnl = gross - sell_cost - p.invested - p.buy_cost
                cash += gross - sell_cost
                trades.append({
                    'market': market, 'asset_type': 'index', 'config_id': cfg.config_id,
                    'code': f'{market}_INDEX', 'name': f'{market} index sleeve',
                    'regime': p.regime, 'entry_date': str(p.entry_date.date()),
                    'exit_date': str(date.date()), 'entry_price': p.entry_price,
                    'exit_price': px, 'hold_days': p.age, 'net_pnl': pnl,
                    'net_return_on_position': pnl / p.invested,
                })
            else:
                kept_indexes.append(p)
        indexes = kept_indexes

        mark = sum(p.shares * close_map.get((date, p.code), p.entry_price) for p in stocks)
        mark += sum(p.units * bench_close.get(date, p.entry_price) for p in indexes)
        equity = cash + mark
        daily_ret = equity / prev_equity - 1.0 if prev_equity > 0 else 0.0
        prev_equity = equity
        regime = str(day['regime'].dropna().iloc[0]) if day['regime'].notna().any() else 'unknown'
        eq_rows.append({
            'date': date, 'market': market, 'equity': equity,
            'daily_return': daily_ret, 'benchmark_daily_return': bench_ret_map.get(date, 0.0),
            'regime': regime, 'stock_positions': len(stocks), 'index_positions': len(indexes),
        })

        if i + 1 < len(dates):
            next_date = dates[i + 1]
            stock_weight, index_weight, hold_days, style = get_weights(day, cfg)
            if stock_weight > 0 or index_weight > 0:
                candidates = candidate_scores(day, style, cfg.overheat_sigma).head(cfg.n_select)
                if stock_weight > 0 and candidates.empty:
                    stock_weight = 0.0
                pending[next_date] = {
                    'stock_weight': stock_weight,
                    'index_weight': index_weight,
                    'hold_days': hold_days,
                    'regime': regime,
                    'candidates': candidates,
                }

    last_date = dates[-1]
    for p in stocks:
        px = close_map.get((last_date, p.code), p.entry_price)
        gross = p.shares * px
        sell_cost = gross * STOCK_SELL_COST * stress
        pnl = gross - sell_cost - p.invested - p.buy_cost
        cash += gross - sell_cost
        trades.append({
            'market': market, 'asset_type': 'stock', 'config_id': cfg.config_id,
            'code': p.code, 'name': p.name, 'regime': p.regime,
            'entry_date': str(p.entry_date.date()), 'exit_date': str(last_date.date()),
            'entry_price': p.entry_price, 'exit_price': px, 'hold_days': p.age,
            'net_pnl': pnl, 'net_return_on_position': pnl / p.invested,
        })
    for p in indexes:
        px = bench_close.get(last_date, p.entry_price)
        gross = p.units * px
        sell_cost = gross * INDEX_SELL_COST * stress
        pnl = gross - sell_cost - p.invested - p.buy_cost
        cash += gross - sell_cost
        trades.append({
            'market': market, 'asset_type': 'index', 'config_id': cfg.config_id,
            'code': f'{market}_INDEX', 'name': f'{market} index sleeve',
            'regime': p.regime, 'entry_date': str(p.entry_date.date()),
            'exit_date': str(last_date.date()), 'entry_price': p.entry_price,
            'exit_price': px, 'hold_days': p.age, 'net_pnl': pnl,
            'net_return_on_position': pnl / p.invested,
        })

    eq = pd.DataFrame(eq_rows)
    if not eq.empty:
        eq.loc[eq.index[-1], 'equity'] = cash
        eq['daily_return'] = eq['equity'].pct_change().fillna(0.0)
    tr = pd.DataFrame(trades)
    metrics = compute_metrics(eq, tr, b, start, end)
    metrics['config_id'] = cfg.config_id
    return metrics, tr, eq


def compute_metrics(eq: pd.DataFrame, trades: pd.DataFrame, bench: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    if eq.empty:
        return {'return': 0.0, 'profit_factor': 0.0, 'max_drawdown': 0.0, 'trades': 0}
    total = float(eq['equity'].iloc[-1] / INITIAL_CAPITAL - 1.0)
    dd = eq['equity'] / eq['equity'].cummax() - 1.0
    pnl = trades['net_pnl'].astype(float) if not trades.empty else pd.Series(dtype=float)
    gp = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
    gl = float(-pnl[pnl < 0].sum()) if len(pnl) else 0.0
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    eq2 = eq.copy()
    eq2['month'] = eq2['date'].dt.to_period('M')
    monthly = eq2.groupby('month')['daily_return'].apply(lambda x: float((1+x).prod()-1))
    monthly_geom = float((1+monthly).prod() ** (1/len(monthly)) - 1) if len(monthly) else 0.0
    bret = float(bench['bench_close'].iloc[-1] / bench['bench_close'].iloc[0] - 1.0) if len(bench) >= 2 else 0.0
    daily = eq['daily_return'].astype(float)
    bd = eq['benchmark_daily_return'].astype(float)
    beta = float(daily.cov(bd) / bd.var()) if bd.var() > 0 else 0.0
    alpha = float((daily - beta * bd).sum())
    regimes: dict[str, Any] = {}
    for label in ['bull', 'transition', 'neutral', 'bear', 'panic']:
        se = eq[eq['regime'] == label]
        st = trades[trades['regime'] == label] if not trades.empty else pd.DataFrame()
        sp = st['net_pnl'].astype(float) if not st.empty else pd.Series(dtype=float)
        rgp = float(sp[sp > 0].sum()) if len(sp) else 0.0
        rgl = float(-sp[sp < 0].sum()) if len(sp) else 0.0
        regimes[label] = {
            'days': int(len(se)),
            'strategy_return': float((1+se['daily_return']).prod()-1) if len(se) else 0.0,
            'benchmark_return': float((1+se['benchmark_daily_return']).prod()-1) if len(se) else 0.0,
            'trades': int(len(st)),
            'trade_pnl': float(sp.sum()) if len(sp) else 0.0,
            'pf': rgp / rgl if rgl > 0 else (999.0 if rgp > 0 else 0.0),
        }
    return {
        'return': total, 'final_equity': float(eq['equity'].iloc[-1]),
        'net_profit': float(eq['equity'].iloc[-1] - INITIAL_CAPITAL),
        'profit_factor': float(pf), 'max_drawdown': float(dd.min()),
        'trades': int(len(trades)), 'wins': int((pnl > 0).sum()) if len(pnl) else 0,
        'win_rate': float((pnl > 0).mean()) if len(pnl) else 0.0,
        'monthly_geom': monthly_geom,
        'monthly_median': float(monthly.median()) if len(monthly) else 0.0,
        'positive_month_ratio': float((monthly > 0).mean()) if len(monthly) else 0.0,
        'monthly_returns': {str(k): float(v) for k,v in monthly.items()},
        'benchmark_return': bret,
        'opportunity_adjusted': total - 0.5 * max(0.0, bret-total),
        'beta': beta, 'alpha_sum': alpha, 'regimes': regimes,
    }


def configs() -> list[Config]:
    out: list[Config] = []
    bull_pairs = [(0.35, 0.60), (0.50, 0.45), (0.65, 0.30), (0.80, 0.15)]
    transition_pairs = [(0.0, 0.0), (0.25, 0.20), (0.40, 0.10)]
    for bh, n, bp, tp, rb, oh, bs, ts in itertools.product(
        [2, 3], [1, 2], bull_pairs, transition_pairs,
        [0.0, 0.15], [1.5, 2.5],
        ['bull_continuous', 'bull_pullback'],
        ['transition_defensive', 'transition_rebound'],
    ):
        out.append(Config(
            bull_hold=bh, transition_hold=1, n_select=n,
            bull_stock_weight=bp[0], bull_index_weight=bp[1],
            transition_stock_weight=tp[0], transition_index_weight=tp[1],
            rebound_stock_weight=rb, overheat_sigma=oh,
            bull_style=bs, transition_style=ts,
        ))
    return out


def score(m: dict[str, Any]) -> float:
    r = m['regimes']
    defensive = r['bear']['strategy_return'] + r['panic']['strategy_return']
    transition = r['transition']['strategy_return'] + r['neutral']['strategy_return']
    return (
        0.40*m['return'] + 0.90*m['monthly_geom'] + 0.35*m['alpha_sum']
        + 0.10*m['positive_month_ratio'] + 0.03*min(m['profit_factor'], 3.0)
        + 0.15*max(0.0, transition) + 0.15*max(0.0, defensive)
        + 0.7*min(0.0, m['max_drawdown'] + 0.12)
    )


def validation_gate(m: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    r = m['regimes']
    trans = r['transition']['strategy_return'] + r['neutral']['strategy_return']
    defensive = r['bear']['strategy_return'] + r['panic']['strategy_return']
    return {
        'return_10pct': m['return'] >= 0.10,
        'monthly_geom_3pct': m['monthly_geom'] >= 0.03,
        'pf_1_20': m['profit_factor'] >= 1.20,
        'mdd_18pct': m['max_drawdown'] >= -0.18,
        'positive_months_half': m['positive_month_ratio'] >= 0.50,
        'cost_stress_positive': stress['return'] > 0,
        'alpha_positive': m['alpha_sum'] > 0,
        'transition_not_bad': trans >= -0.03,
        'defensive_not_bad': defensive >= -0.03,
        'trades_10': m['trades'] >= 10,
    }


def final_gate(m: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    r = m['regimes']
    trans = r['transition']['strategy_return'] + r['neutral']['strategy_return']
    defensive = r['bear']['strategy_return'] + r['panic']['strategy_return']
    return {
        'return_nonnegative': m['return'] >= 0.0,
        'pf_1_0_or_few_trades': m['trades'] < 5 or m['profit_factor'] >= 1.0,
        'mdd_12pct': m['max_drawdown'] >= -0.12,
        'cost_stress_nonnegative': stress['return'] >= -0.01,
        'transition_nonnegative': trans >= -0.02,
        'defensive_nonnegative': defensive >= -0.02,
    }


def main() -> None:
    prices = core.load_prices()
    benchmarks = core.load_benchmarks(prices['date'].min(), prices['date'].max(), prices)
    features, bench = core.build_features(prices, benchmarks)
    features, bench = enrich_regime(features, bench)
    all_cfg = configs()
    print('rows', len(prices), 'range', prices['date'].min(), prices['date'].max(), 'configs', len(all_cfg))

    result: dict[str, Any] = {
        'version': 'korea-regime-v4-overlay-cash-defense',
        'data_range': [str(prices['date'].min().date()), str(prices['date'].max().date())],
        'periods': {
            'development': [str(DEV_START.date()), str(DEV_END.date())],
            'validation': [str(VALID_START.date()), str(VALID_END.date())],
            'final': [str(FINAL_START.date()), str(min(FINAL_END, prices['date'].max()).date())],
        },
        'markets': {},
    }

    for market in ['KOSPI', 'KOSDAQ']:
        dev_rows = []
        for i, cfg in enumerate(all_cfg):
            m, _, _ = simulate(market, cfg, features, bench, DEV_START, DEV_END)
            dev_rows.append((score(m), cfg, m))
            if (i+1) % 64 == 0:
                print(market, 'dev', i+1, '/', len(all_cfg))
        dev_rows.sort(key=lambda x: x[0], reverse=True)
        top = dev_rows[:32]

        val_rows = []
        for _, cfg, dm in top:
            vm, _, _ = simulate(market, cfg, features, bench, VALID_START, VALID_END)
            vs, _, _ = simulate(market, cfg, features, bench, VALID_START, VALID_END, stress=1.5)
            gates = validation_gate(vm, vs)
            val_rows.append({
                'cfg': cfg, 'development': dm, 'validation': vm, 'stress': vs,
                'gates': gates, 'passed': all(gates.values()), 'score': score(vm),
            })
        passing = [x for x in val_rows if x['passed']]
        pool = passing if passing else val_rows
        best = max(pool, key=lambda x: x['score'])
        cfg = best['cfg']

        final_end = min(FINAL_END, prices['date'].max())
        fm, ftr, feq = simulate(market, cfg, features, bench, FINAL_START, final_end)
        fs, _, _ = simulate(market, cfg, features, bench, FINAL_START, final_end, stress=1.5)
        fg = final_gate(fm, fs)

        full_m, full_tr, full_eq = simulate(market, cfg, features, bench, VALID_START, final_end)
        full_s, _, _ = simulate(market, cfg, features, bench, VALID_START, final_end, stress=1.5)

        result['markets'][market] = {
            'selected_config': asdict(cfg),
            'selected_config_id': cfg.config_id,
            'development_top_score': float(top[0][0]),
            'validation': {k:v for k,v in best.items() if k != 'cfg'},
            'final': {'metrics': fm, 'stress': fs, 'gates': fg, 'passed': all(fg.values())},
            'full_2026': {'metrics': full_m, 'stress': full_s},
            'accepted': bool(best['passed'] and all(fg.values()) and full_m['return'] > 0),
        }
        ftr.to_csv(OUT / f'{market.lower()}_final_trades.csv', index=False)
        feq.to_csv(OUT / f'{market.lower()}_final_equity.csv', index=False)
        full_tr.to_csv(OUT / f'{market.lower()}_full_trades.csv', index=False)
        full_eq.to_csv(OUT / f'{market.lower()}_full_equity.csv', index=False)

    result['accepted_both'] = all(result['markets'][m]['accepted'] for m in ['KOSPI', 'KOSDAQ'])
    (OUT / 'v4_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('===KOREA_V4_RESULT_BEGIN===')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('===KOREA_V4_RESULT_END===')


if __name__ == '__main__':
    main()
