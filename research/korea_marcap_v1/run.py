from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path('/tmp/korea_marcap_v1')
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 2_000_000.0
BUY_FEE = 0.0005
SELL_FEE_TAX = 0.0025
SLIPPAGE = 0.0005
BASE_COST = BUY_FEE + SELL_FEE_TAX + 2 * SLIPPAGE
YEARS = range(2019, 2027)
FEATURE_START = pd.Timestamp('2020-01-01')
MODEL_TRAIN_END = pd.Timestamp('2023-12-31')
MODEL_SELECT_START = pd.Timestamp('2024-01-01')
MODEL_SELECT_END = pd.Timestamp('2024-12-31')
STRATEGY_SELECT_START = pd.Timestamp('2025-01-01')
STRATEGY_SELECT_END = pd.Timestamp('2025-12-31')
FINAL_START = pd.Timestamp('2026-01-01')
FINAL_END = pd.Timestamp('2026-08-21')

FEATURES = [
    'ret1','ret2','ret3','ret5','ret10','ret20','ret60',
    'gap1','intraday','range1','close_loc','vol5','vol20',
    'dist_ma5','dist_ma10','dist_ma20','dist_ma60','dd20','dd60',
    'volume_ratio5','volume_ratio20','amount_ratio5','amount_ratio20',
    'turnover','log_marcap','stocks_chg5','beta20','beta60',
    'rel5','rel20','rel60','mkt_ret1','mkt_ret5','mkt_ret20',
    'mkt_vol20','mkt_dd20','breadth20','breadth60','event_score',
    'ret5_rank','ret20_rank','ret60_rank','rel5_rank','rel20_rank',
    'rel60_rank','turnover_rank','amount_rank','size_rank','beta20_rank',
    'event_rank','dd20_rank','close_loc_rank'
]

BAD_NAME_RE = r'(스팩|SPAC|리츠|REIT|ETF|ETN|인버스|레버리지|선물|우$|우B$|우C$|우선주)'


def download_year(year: int) -> Path:
    path = DATA / f'marcap-{year}.parquet'
    if path.exists() and path.stat().st_size > 100_000:
        return path
    url = f'https://raw.githubusercontent.com/FinanceData/marcap/master/data/marcap-{year}.parquet'
    print('download', year, flush=True)
    urllib.request.urlretrieve(url, path)
    return path


def load_data() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        path = download_year(year)
        df = pd.read_parquet(path)
        if 'Date' not in df.columns:
            df = df.reset_index()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    rename = {'ChagesRatio': 'ChangesRatio'}
    df = df.rename(columns=rename)
    needed = ['Date','Code','Name','Open','High','Low','Close','Volume','Amount','Marcap','Stocks','Market']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f'missing columns: {missing}; got={list(df.columns)}')
    df = df[needed].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Code'] = df['Code'].astype(str).str.zfill(6)
    for c in ['Open','High','Low','Close','Volume','Amount','Marcap','Stocks']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[df['Market'].isin(['KOSPI','KOSDAQ'])]
    df = df[df['Date'].between(pd.Timestamp('2019-01-01'), FINAL_END)]
    df = df.dropna(subset=['Date','Code','Open','High','Low','Close','Volume','Amount','Marcap'])
    df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]
    df = df[~df['Name'].astype(str).str.contains(BAD_NAME_RE, regex=True, na=False)]
    df = df.sort_values(['Code','Date']).drop_duplicates(['Code','Date'], keep='last').reset_index(drop=True)
    return df


def roll(s: pd.Series, codes: pd.Series, n: int, kind: str = 'mean') -> pd.Series:
    r = s.groupby(codes, sort=False).rolling(n, min_periods=n)
    if kind == 'mean':
        out = r.mean()
    elif kind == 'std':
        out = r.std(ddof=0)
    elif kind == 'max':
        out = r.max()
    elif kind == 'min':
        out = r.min()
    elif kind == 'median':
        out = r.median()
    else:
        raise ValueError(kind)
    return out.reset_index(level=0, drop=True).sort_index()


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy().sort_values(['Code','Date']).reset_index(drop=True)
    codes = df['Code']
    g = df.groupby('Code', sort=False)
    prev_close = g['Close'].shift(1)
    df['ret1'] = df['Close'] / prev_close - 1.0
    for n in [2,3,5,10,20,60]:
        df[f'ret{n}'] = df['Close'] / g['Close'].shift(n) - 1.0
    df['gap1'] = df['Open'] / prev_close - 1.0
    df['intraday'] = df['Close'] / df['Open'] - 1.0
    df['range1'] = (df['High'] - df['Low']) / prev_close
    df['close_loc'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)).clip(0, 1)
    df['vol5'] = roll(df['ret1'], codes, 5, 'std')
    df['vol20'] = roll(df['ret1'], codes, 20, 'std')
    for n in [5,10,20,60]:
        ma = roll(df['Close'], codes, n, 'mean')
        df[f'ma{n}'] = ma
        df[f'dist_ma{n}'] = df['Close'] / ma - 1.0
    high20 = roll(df['Close'], codes, 20, 'max')
    high60 = roll(df['Close'], codes, 60, 'max')
    df['dd20'] = df['Close'] / high20 - 1.0
    df['dd60'] = df['Close'] / high60 - 1.0
    for n in [5,20]:
        vmean = roll(df['Volume'], codes, n, 'mean')
        amean = roll(df['Amount'], codes, n, 'mean')
        df[f'volume_ratio{n}'] = df['Volume'] / vmean.replace(0, np.nan)
        df[f'amount_ratio{n}'] = df['Amount'] / amean.replace(0, np.nan)
    df['amount20'] = roll(df['Amount'], codes, 20, 'median')
    df['turnover'] = df['Amount'] / df['Marcap'].replace(0, np.nan)
    df['log_marcap'] = np.log1p(df['Marcap'])
    df['stocks_chg5'] = df['Stocks'] / g['Stocks'].shift(5) - 1.0

    # Previous-day capitalization weights prevent same-day price lookahead.
    df['prev_marcap'] = g['Marcap'].shift(1)
    valid_ret = df['ret1'].between(-0.35, 0.35)
    weighted = df.loc[valid_ret, ['Date','Market','ret1','prev_marcap']].dropna()
    weighted['wr'] = weighted['ret1'] * weighted['prev_marcap']
    mkt = weighted.groupby(['Date','Market'], as_index=False).agg(wr=('wr','sum'), w=('prev_marcap','sum'))
    mkt['mkt_ret1'] = mkt['wr'] / mkt['w']
    mkt = mkt.sort_values(['Market','Date']).reset_index(drop=True)
    mg = mkt.groupby('Market', sort=False)
    mkt['mkt_index'] = mg['mkt_ret1'].transform(lambda s: (1+s.fillna(0)).cumprod())
    mkt['mkt_ret5'] = mkt['mkt_index'] / mg['mkt_index'].shift(5) - 1.0
    mkt['mkt_ret20'] = mkt['mkt_index'] / mg['mkt_index'].shift(20) - 1.0
    mkt['mkt_vol20'] = mg['mkt_ret1'].rolling(20, min_periods=20).std(ddof=0).reset_index(level=0, drop=True).sort_index()
    mkt['mkt_ma20'] = mg['mkt_index'].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True).sort_index()
    mkt['mkt_ma60'] = mg['mkt_index'].rolling(60, min_periods=60).mean().reset_index(level=0, drop=True).sort_index()
    mkt['mkt_high20'] = mg['mkt_index'].rolling(20, min_periods=20).max().reset_index(level=0, drop=True).sort_index()
    mkt['mkt_dd20'] = mkt['mkt_index'] / mkt['mkt_high20'] - 1.0

    breadth = df.groupby(['Date','Market']).agg(
        breadth20=('dist_ma20', lambda s: float((s > 0).mean())),
        breadth60=('dist_ma60', lambda s: float((s > 0).mean())),
    ).reset_index()
    mkt = mkt.merge(breadth, on=['Date','Market'], how='left')

    def regime_row(r: pd.Series) -> str:
        if pd.isna(r['mkt_ma60']):
            return 'neutral'
        if (r['mkt_ret5'] <= -0.07) or (r['mkt_dd20'] <= -0.10 and r['breadth20'] < 0.30):
            return 'panic'
        if r['mkt_index'] < r['mkt_ma60'] and r['breadth60'] < 0.42:
            return 'bear'
        if r['mkt_index'] > r['mkt_ma20'] > r['mkt_ma60'] and r['breadth20'] >= 0.55:
            return 'bull'
        if (r['mkt_index'] < r['mkt_ma20']) or (r['breadth20'] < 0.48) or (r['mkt_vol20'] > 0.025):
            return 'transition'
        return 'neutral'

    mkt['regime'] = mkt.apply(regime_row, axis=1)
    df = df.merge(mkt[['Date','Market','mkt_ret1','mkt_ret5','mkt_ret20','mkt_vol20','mkt_dd20','breadth20','breadth60','regime']], on=['Date','Market'], how='left')
    df['rel5'] = df['ret5'] - df['mkt_ret5']
    df['rel20'] = df['ret20'] - df['mkt_ret20']
    df['rel60'] = df['ret60'] - (df['mkt_index_proxy'] if 'mkt_index_proxy' in df else 0.0)
    # rel60 from market 60-day compounded returns.
    mkt60 = mkt[['Date','Market','mkt_index']].copy()
    mkt60['mkt_ret60'] = mkt60['mkt_index'] / mkt60.groupby('Market')['mkt_index'].shift(60) - 1.0
    df = df.merge(mkt60[['Date','Market','mkt_ret60']], on=['Date','Market'], how='left')
    df['rel60'] = df['ret60'] - df['mkt_ret60']

    # Rolling beta using covariance identities.
    for n in [20,60]:
        prod = df['ret1'] * df['mkt_ret1']
        mean_prod = roll(prod, codes, n, 'mean')
        mean_s = roll(df['ret1'], codes, n, 'mean')
        mean_m = roll(df['mkt_ret1'], codes, n, 'mean')
        var_m = roll(df['mkt_ret1'] ** 2, codes, n, 'mean') - mean_m ** 2
        df[f'beta{n}'] = (mean_prod - mean_s * mean_m) / var_m.replace(0, np.nan)

    df['event_score'] = (
        0.30 * df['gap1'].clip(-0.15, 0.15) / 0.15
        + 0.25 * (df['volume_ratio20'].clip(0, 4) - 1) / 3
        + 0.25 * (df['amount_ratio20'].clip(0, 4) - 1) / 3
        + 0.20 * (df['close_loc'] - 0.5) * 2
    )

    rank_cols = ['ret5','ret20','ret60','rel5','rel20','rel60','turnover','Amount','Marcap','beta20','event_score','dd20','close_loc']
    names = ['ret5_rank','ret20_rank','ret60_rank','rel5_rank','rel20_rank','rel60_rank','turnover_rank','amount_rank','size_rank','beta20_rank','event_rank','dd20_rank','close_loc_rank']
    for col, name in zip(rank_cols, names):
        df[name] = df.groupby(['Date','Market'])[col].rank(pct=True, method='average')

    # Future labels: signal at close t, enter at open t+1, exit at close t+h.
    g = df.groupby('Code', sort=False)
    df['entry_open'] = g['Open'].shift(-1)
    df['entry_date'] = g['Date'].shift(-1)
    for h in [1,2,3]:
        df[f'exit_close_{h}'] = g['Close'].shift(-h)
        df[f'exit_date_{h}'] = g['Date'].shift(-h)
        df[f'fwd{h}'] = df[f'exit_close_{h}'] / df['entry_open'] - 1.0
        df[f'fwd{h}_rank'] = df.groupby(['Date','Market'])[f'fwd{h}'].rank(pct=True, method='average')
    df['atr20'] = roll((df['High'] - df['Low']).abs(), codes, 20, 'mean')

    # Eligibility is point-in-time and conservative.
    df['eligible'] = (
        (df['Close'] >= 1000)
        & (df['amount20'] >= 1_000_000_000)
        & (df['Marcap'] >= 50_000_000_000)
        & (df['Volume'] > 0)
        & (df['Amount'] > 0)
        & (df['stocks_chg5'].abs().fillna(0) < 0.20)
        & (df['ret1'].abs().fillna(0) <= 0.31)
        & df['atr20'].notna()
    )
    df = df[df['Date'] >= FEATURE_START].reset_index(drop=True)
    return df, mkt


def fit_model(train: pd.DataFrame, params: dict[str, Any]) -> lgb.LGBMRegressor:
    X = train[FEATURES].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    y = train['fwd3_rank'].astype(float)
    w = np.sqrt(train['amount_rank'].clip(0.05, 1.0))
    model = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        num_leaves=params['num_leaves'],
        min_child_samples=params['min_child_samples'],
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=20260823,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X, y, sample_weight=w)
    return model


def predict(model: lgb.LGBMRegressor, frame: pd.DataFrame) -> np.ndarray:
    X = frame[FEATURES].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return model.predict(X)


def model_quality(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    q = frame.dropna(subset=[pred_col,'fwd3']).copy()
    if q.empty:
        return {'ic': -9, 'top_return': -9, 'spread': -9}
    daily_ic = q.groupby(['Date','Market']).apply(lambda x: x[pred_col].corr(x['fwd3_rank']), include_groups=False).dropna()
    q['pred_rank'] = q.groupby(['Date','Market'])[pred_col].rank(pct=True)
    top = q[q['pred_rank'] >= 0.95]['fwd3'].mean()
    bottom = q[q['pred_rank'] <= 0.05]['fwd3'].mean()
    return {'ic': float(daily_ic.mean()), 'top_return': float(top), 'spread': float(top-bottom)}


@dataclasses.dataclass(frozen=True)
class Config:
    hold_days: int
    top_n: int
    min_score: float
    stop_atr: float
    target_r: float
    bull_exp: float
    neutral_exp: float
    transition_exp: float
    bear_exp: float
    panic_exp: float
    pullback_weight: float

    @property
    def id(self) -> str:
        return (
            f'h{self.hold_days}-n{self.top_n}-s{self.min_score:.2f}-st{self.stop_atr:.1f}'
            f'-t{self.target_r:.1f}-be{self.bull_exp:.2f}-ne{self.neutral_exp:.2f}'
            f'-tr{self.transition_exp:.2f}-ba{self.bear_exp:.2f}-pa{self.panic_exp:.2f}'
            f'-pw{self.pullback_weight:.2f}'
        )


def configs() -> list[Config]:
    out = []
    for hold, topn, score, stop, target, tx, ba, pa, pw in itertools.product(
        [2,3], [1,2], [0.78,0.84,0.90], [1.5,2.0,2.5], [2.5,4.0],
        [0.45,0.65], [0.15,0.30], [0.10,0.25], [0.25,0.45]
    ):
        out.append(Config(hold, topn, score, stop, target, 1.0, 0.80, tx, ba, pa, pw))
    return out


def regime_exposure(cfg: Config, regime: str) -> float:
    return {
        'bull': cfg.bull_exp,
        'neutral': cfg.neutral_exp,
        'transition': cfg.transition_exp,
        'bear': cfg.bear_exp,
        'panic': cfg.panic_exp,
    }.get(regime, cfg.neutral_exp)


def build_score(frame: pd.DataFrame, pred_col: str, cfg: Config) -> pd.Series:
    p = frame.groupby(['Date','Market'])[pred_col].rank(pct=True)
    trend = 0.50*p + 0.20*frame['rel20_rank'] + 0.15*frame['ret60_rank'] + 0.10*frame['event_rank'] + 0.05*frame['close_loc_rank']
    pullback = 0.50*p + 0.20*frame['rel60_rank'] + 0.15*(1-frame['ret3'].groupby([frame['Date'],frame['Market']]).rank(pct=True)) + 0.10*frame['close_loc_rank'] + 0.05*frame['event_rank']
    defensive = 0.55*p + 0.20*frame['rel20_rank'] + 0.15*(1-frame['beta20_rank']) + 0.10*frame['ret5_rank']
    panic = 0.50*p + 0.20*(1-frame['ret3'].groupby([frame['Date'],frame['Market']]).rank(pct=True)) + 0.15*frame['close_loc_rank'] + 0.15*frame['event_rank']
    r = frame['regime'].astype(str)
    score = trend.copy()
    score = np.where(r.eq('transition'), (1-cfg.pullback_weight)*trend + cfg.pullback_weight*pullback, score)
    score = np.where(r.eq('bear'), defensive, score)
    score = np.where(r.eq('panic'), panic, score)
    return pd.Series(score, index=frame.index)


def regime_filter(row: pd.Series) -> bool:
    r = str(row['regime'])
    if r == 'bull':
        return bool(row['dist_ma60'] > -0.03 and row['rel20_rank'] >= 0.55 and row['dd20'] >= -0.18)
    if r == 'transition':
        return bool(row['ret60'] > 0 and row['rel20_rank'] >= 0.70 and -0.18 <= row['dd20'] <= -0.01 and row['close_loc'] >= 0.45)
    if r == 'neutral':
        return bool(row['rel20_rank'] >= 0.65 and row['dist_ma20'] > -0.05)
    if r == 'bear':
        return bool(row['ret20'] > 0 and row['rel20_rank'] >= 0.90 and row['beta20'] <= 0.9 and row['dist_ma20'] > -0.03)
    if r == 'panic':
        reversal = row['ret3'] <= -0.08 and row['close_loc'] >= 0.75 and row['volume_ratio20'] >= 1.4
        leader = row['ret5'] > 0 and row['rel5_rank'] >= 0.95 and row['close_loc'] >= 0.60
        return bool(reversal or leader)
    return False


def simulate(frame: pd.DataFrame, cfg: Config, pred_col: str, start: pd.Timestamp, end: pd.Timestamp, cost_mult: float = 1.0) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    x = frame[frame['Date'].between(start, end)].copy()
    x = x[x['eligible'] & x[pred_col].notna() & x['entry_open'].notna()].copy()
    if x.empty:
        return empty_metrics(start, end), pd.DataFrame(), pd.DataFrame()
    x['score'] = build_score(x, pred_col, cfg)
    x = x[x.apply(regime_filter, axis=1)]
    x = x[x['score'] >= cfg.min_score]
    x = x.sort_values(['Date','score','Amount'], ascending=[True,False,False])
    dates = sorted(x['Date'].unique())
    all_dates = sorted(frame[frame['Date'].between(start, end)]['Date'].unique())
    date_to_i = {pd.Timestamp(d): i for i,d in enumerate(all_dates)}
    rows_by_date = {pd.Timestamp(d): g for d,g in x.groupby('Date')}
    equity = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    open_positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    last_close_map: dict[str,float] = {}

    raw = frame[frame['Date'].between(start, end + pd.Timedelta(days=10))].sort_values(['Code','Date'])
    by_code = {c: g.set_index('Date').sort_index() for c,g in raw.groupby('Code')}

    for current in all_dates:
        current = pd.Timestamp(current)
        # Mark-to-market and process exits using the current daily OHLC.
        still_open = []
        for pos in open_positions:
            hist = by_code.get(pos['Code'])
            if hist is None or current not in hist.index:
                still_open.append(pos)
                continue
            bar = hist.loc[current]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            last_close_map[pos['Code']] = float(bar['Close'])
            exit_price = None
            reason = None
            if current >= pos['entry_date']:
                if float(bar['Low']) <= pos['stop']:
                    exit_price = min(float(bar['Open']), pos['stop'])
                    reason = 'stop'
                elif float(bar['High']) >= pos['target']:
                    exit_price = pos['target']
                    reason = 'target'
                elif current >= pos['planned_exit_date']:
                    exit_price = float(bar['Close'])
                    reason = 'time'
            if exit_price is None:
                still_open.append(pos)
            else:
                proceeds = pos['qty'] * exit_price * (1 - (SELL_FEE_TAX + SLIPPAGE) * cost_mult)
                cash += proceeds
                pnl = proceeds - pos['cash_out']
                trades.append({**pos, 'exit_date': str(current.date()), 'exit_price': exit_price, 'reason': reason, 'pnl': pnl, 'return': pnl / pos['cash_out']})
        open_positions = still_open

        marked = 0.0
        for pos in open_positions:
            px = last_close_map.get(pos['Code'], pos['entry_price'])
            marked += pos['qty'] * px
        equity = cash + marked

        # Signal was known at prior close; entry occurs at today's open.
        signal_date_candidates = [d for d in dates if pd.Timestamp(d) < current and date_to_i.get(current, 0) - date_to_i.get(pd.Timestamp(d), -999) == 1]
        if signal_date_candidates:
            signal_date = pd.Timestamp(signal_date_candidates[-1])
            cand = rows_by_date.get(signal_date)
            if cand is not None and not cand.empty:
                regime = str(cand.iloc[0]['regime'])
                exposure = regime_exposure(cfg, regime)
                cohort_budget = equity * exposure / cfg.hold_days
                held_codes = {p['Code'] for p in open_positions}
                selected = cand[~cand['Code'].isin(held_codes)].head(cfg.top_n)
                if not selected.empty and cohort_budget > 10_000:
                    per_name = min(cohort_budget / len(selected), cash / len(selected))
                    for _, row in selected.iterrows():
                        hist = by_code.get(row['Code'])
                        if hist is None or current not in hist.index:
                            continue
                        bar = hist.loc[current]
                        if isinstance(bar, pd.DataFrame):
                            bar = bar.iloc[-1]
                        entry_price = float(bar['Open']) * (1 + (BUY_FEE + SLIPPAGE) * cost_mult)
                        qty = int(per_name // entry_price)
                        if qty < 1:
                            continue
                        cash_out = qty * entry_price
                        if cash_out > cash:
                            continue
                        atr = float(row['atr20'])
                        stop = entry_price - cfg.stop_atr * atr
                        risk = max(entry_price - stop, entry_price * 0.01)
                        target = entry_price + cfg.target_r * risk
                        idx = date_to_i[current]
                        exit_idx = min(idx + cfg.hold_days - 1, len(all_dates)-1)
                        planned_exit = pd.Timestamp(all_dates[exit_idx])
                        cash -= cash_out
                        open_positions.append({
                            'signal_date': str(signal_date.date()), 'entry_date': current,
                            'planned_exit_date': planned_exit, 'Code': row['Code'], 'Name': row['Name'],
                            'Market': row['Market'], 'regime': regime, 'score': float(row['score']),
                            'entry_price': entry_price, 'qty': qty, 'cash_out': cash_out,
                            'stop': stop, 'target': target, 'config_id': cfg.id,
                        })
                        last_close_map[row['Code']] = float(bar['Close'])

        marked = 0.0
        for pos in open_positions:
            px = last_close_map.get(pos['Code'], pos['entry_price'])
            marked += pos['qty'] * px
        equity = cash + marked
        equity_rows.append({'date': str(current.date()), 'equity': equity, 'cash': cash, 'open_positions': len(open_positions)})

    # Liquidate remaining at last available close.
    if all_dates:
        current = pd.Timestamp(all_dates[-1])
        for pos in open_positions:
            hist = by_code.get(pos['Code'])
            if hist is None or current not in hist.index:
                continue
            bar = hist.loc[current]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            exit_price = float(bar['Close'])
            proceeds = pos['qty'] * exit_price * (1 - (SELL_FEE_TAX + SLIPPAGE) * cost_mult)
            cash += proceeds
            pnl = proceeds - pos['cash_out']
            trades.append({**pos, 'exit_date': str(current.date()), 'exit_price': exit_price, 'reason': 'end', 'pnl': pnl, 'return': pnl / pos['cash_out']})
        equity = cash
        equity_rows.append({'date': str(current.date()), 'equity': equity, 'cash': cash, 'open_positions': 0})

    tr = pd.DataFrame(trades)
    eq = pd.DataFrame(equity_rows).drop_duplicates('date', keep='last')
    metrics = calc_metrics(tr, eq, start, end)
    return metrics, tr, eq


def empty_metrics(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    return {'start': str(start.date()), 'end': str(end.date()), 'return': 0.0, 'final_equity': INITIAL_CAPITAL, 'profit_factor': 0.0, 'max_drawdown': 0.0, 'trades': 0, 'wins': 0, 'win_rate': 0.0, 'monthly_geom': 0.0, 'monthly_median': 0.0, 'positive_month_ratio': 0.0, 'monthly_returns': {}, 'market_pnl': {}}


def calc_metrics(tr: pd.DataFrame, eq: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    if eq.empty:
        return empty_metrics(start, end)
    eq = eq.copy()
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.sort_values('date')
    ret = float(eq['equity'].iloc[-1] / INITIAL_CAPITAL - 1.0)
    dd = eq['equity'] / eq['equity'].cummax() - 1.0
    eq['daily_return'] = eq['equity'].pct_change().fillna(0)
    eq['month'] = eq['date'].dt.to_period('M')
    monthly = eq.groupby('month')['daily_return'].apply(lambda s: float((1+s).prod()-1))
    gross_profit = float(tr.loc[tr['pnl'] > 0, 'pnl'].sum()) if not tr.empty else 0.0
    gross_loss = float(-tr.loc[tr['pnl'] < 0, 'pnl'].sum()) if not tr.empty else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    market_pnl = tr.groupby('Market')['pnl'].sum().to_dict() if not tr.empty else {}
    return {
        'start': str(start.date()), 'end': str(end.date()), 'return': ret,
        'final_equity': float(eq['equity'].iloc[-1]), 'profit_factor': float(pf),
        'max_drawdown': float(dd.min()), 'trades': int(len(tr)),
        'wins': int((tr['pnl'] > 0).sum()) if not tr.empty else 0,
        'win_rate': float((tr['pnl'] > 0).mean()) if not tr.empty else 0.0,
        'monthly_geom': float((1+monthly).prod() ** (1/max(1,len(monthly))) - 1),
        'monthly_median': float(monthly.median()) if len(monthly) else 0.0,
        'positive_month_ratio': float((monthly > 0).mean()) if len(monthly) else 0.0,
        'monthly_returns': {str(k): float(v) for k,v in monthly.items()},
        'market_pnl': {str(k): float(v) for k,v in market_pnl.items()},
    }


def config_score(m: dict[str, Any], stress: dict[str, Any]) -> float:
    market_ok = len([v for v in m.get('market_pnl', {}).values() if v > 0])
    return (
        2.2*m['monthly_geom'] + 0.8*m['return'] + 0.10*min(m['profit_factor'], 3.0)
        + 0.15*m['positive_month_ratio'] + 0.20*min(0.0, m['max_drawdown'] + 0.12)
        + 0.10*market_ok + 0.40*min(0.0, stress['return'])
    )


def strategy_gate(m: dict[str, Any], stress: dict[str, Any]) -> dict[str,bool]:
    mp = m.get('market_pnl', {})
    return {
        'return_15pct': m['return'] >= 0.15,
        'monthly_geom_1pct': m['monthly_geom'] >= 0.01,
        'pf_1_20': m['profit_factor'] >= 1.20,
        'mdd_20pct': m['max_drawdown'] >= -0.20,
        'positive_months_55pct': m['positive_month_ratio'] >= 0.55,
        'trades_30': m['trades'] >= 30,
        'stress_positive': stress['return'] > 0,
        'both_markets_nonnegative': mp.get('KOSPI', -1) >= 0 and mp.get('KOSDAQ', -1) >= 0,
    }


def final_gate(m: dict[str, Any], stress: dict[str, Any]) -> dict[str,bool]:
    mp = m.get('market_pnl', {})
    return {
        'return_positive': m['return'] > 0,
        'monthly_geom_5pct': m['monthly_geom'] >= 0.05,
        'pf_1_40': m['profit_factor'] >= 1.40,
        'mdd_15pct': m['max_drawdown'] >= -0.15,
        'positive_months_75pct': m['positive_month_ratio'] >= 0.75,
        'trades_40': m['trades'] >= 40,
        'stress_positive': stress['return'] > 0,
        'both_markets_positive': mp.get('KOSPI', -1) > 0 and mp.get('KOSDAQ', -1) > 0,
    }


def main() -> None:
    raw = load_data()
    print('raw', len(raw), raw['Date'].min(), raw['Date'].max(), raw['Code'].nunique(), flush=True)
    frame, market = build_features(raw)
    print('features', len(frame), frame['Date'].min(), frame['Date'].max(), flush=True)
    frame.to_parquet(OUT / 'feature_sample.parquet', index=False)

    model_params = [
        {'n_estimators':300,'learning_rate':0.035,'num_leaves':15,'min_child_samples':180,'reg_alpha':0.1,'reg_lambda':1.0},
        {'n_estimators':450,'learning_rate':0.025,'num_leaves':31,'min_child_samples':250,'reg_alpha':0.2,'reg_lambda':1.5},
        {'n_estimators':250,'learning_rate':0.045,'num_leaves':12,'min_child_samples':120,'reg_alpha':0.0,'reg_lambda':2.0},
    ]
    model_selection: dict[str,Any] = {}
    selected_params: dict[str,dict[str,Any]] = {}
    for market_name in ['KOSPI','KOSDAQ']:
        train = frame[(frame['Market']==market_name) & frame['Date'].between(FEATURE_START, MODEL_TRAIN_END) & frame['eligible'] & frame['fwd3_rank'].notna()].copy()
        valid = frame[(frame['Market']==market_name) & frame['Date'].between(MODEL_SELECT_START, MODEL_SELECT_END) & frame['eligible'] & frame['fwd3_rank'].notna()].copy()
        if len(train) > 650_000:
            train = train.sample(650_000, random_state=20260823)
        rows = []
        best = None
        for i,p in enumerate(model_params):
            model = fit_model(train,p)
            valid[f'pred_tmp_{i}'] = predict(model, valid)
            q = model_quality(valid, f'pred_tmp_{i}')
            row = {'id':i,'params':p,**q}
            row['score'] = q['ic'] + 3*q['spread'] + q['top_return']
            rows.append(row)
            if best is None or row['score'] > best['score']:
                best = row
        assert best is not None
        selected_params[market_name] = best['params']
        model_selection[market_name] = {'candidates':rows,'selected':best}
        print('model', market_name, best, flush=True)

    # Fit through 2024 and generate 2025 validation predictions.
    frame['pred_2025'] = np.nan
    for market_name in ['KOSPI','KOSDAQ']:
        train = frame[(frame['Market']==market_name) & frame['Date'].between(FEATURE_START, MODEL_SELECT_END) & frame['eligible'] & frame['fwd3_rank'].notna()].copy()
        if len(train) > 850_000:
            train = train.sample(850_000, random_state=20260823)
        model = fit_model(train, selected_params[market_name])
        mask = (frame['Market']==market_name) & frame['Date'].between(STRATEGY_SELECT_START, STRATEGY_SELECT_END)
        frame.loc[mask,'pred_2025'] = predict(model, frame.loc[mask])

    config_rows = []
    best_cfg = None
    best_pack = None
    all_cfg = configs()
    print('configs', len(all_cfg), flush=True)
    for i,cfg in enumerate(all_cfg):
        m,tr,eq = simulate(frame,cfg,'pred_2025',STRATEGY_SELECT_START,STRATEGY_SELECT_END,1.0)
        st,_,_ = simulate(frame,cfg,'pred_2025',STRATEGY_SELECT_START,STRATEGY_SELECT_END,1.5)
        gates = strategy_gate(m,st)
        row = {'config_id':cfg.id,'config':dataclasses.asdict(cfg),'metrics':m,'stress':st,'gates':gates,'passed':all(gates.values()),'score':config_score(m,st)}
        config_rows.append(row)
        if row['passed'] and (best_cfg is None or row['score'] > best_pack['score']):
            best_cfg = cfg
            best_pack = row
        if (i+1) % 100 == 0:
            print(i+1,'/',len(all_cfg), flush=True)

    result: dict[str,Any] = {
        'version':'korea-marcap-v1',
        'data':{'rows':len(raw),'min_date':str(raw['Date'].min().date()),'max_date':str(raw['Date'].max().date()),'symbols':int(raw['Code'].nunique()),'source':'FinanceData/marcap'},
        'periods':{'model_train':[str(FEATURE_START.date()),str(MODEL_TRAIN_END.date())],'model_select':[str(MODEL_SELECT_START.date()),str(MODEL_SELECT_END.date())],'strategy_select':[str(STRATEGY_SELECT_START.date()),str(STRATEGY_SELECT_END.date())],'final':[str(FINAL_START.date()),str(FINAL_END.date())]},
        'model_selection':model_selection,
        'config_trials':len(config_rows),
        'passing_configs':sum(1 for r in config_rows if r['passed']),
        'strategy_gate_passed':best_cfg is not None,
        'final_opened':best_cfg is not None,
    }
    pd.DataFrame([{k:v for k,v in r.items() if k not in {'metrics','stress','gates','config'}} | {f'm_{k}':v for k,v in r['metrics'].items() if not isinstance(v,dict)} | {f's_{k}':v for k,v in r['stress'].items() if not isinstance(v,dict)} for r in config_rows]).to_csv(OUT/'config_summary.csv',index=False)

    if best_cfg is None:
        result['accepted'] = False
        result['reason'] = 'No 2025 strategy configuration passed the preregistered gate; 2026 remained unopened.'
    else:
        result['selected_config'] = best_pack
        frame['pred_2026'] = np.nan
        for market_name in ['KOSPI','KOSDAQ']:
            train = frame[(frame['Market']==market_name) & frame['Date'].between(FEATURE_START, STRATEGY_SELECT_END) & frame['eligible'] & frame['fwd3_rank'].notna()].copy()
            if len(train) > 1_050_000:
                train = train.sample(1_050_000, random_state=20260823)
            model = fit_model(train, selected_params[market_name])
            mask = (frame['Market']==market_name) & frame['Date'].between(FINAL_START, FINAL_END)
            frame.loc[mask,'pred_2026'] = predict(model, frame.loc[mask])
        fm,tr,eq = simulate(frame,best_cfg,'pred_2026',FINAL_START,FINAL_END,1.0)
        fs,_,_ = simulate(frame,best_cfg,'pred_2026',FINAL_START,FINAL_END,1.5)
        gates = final_gate(fm,fs)
        result['final'] = {'metrics':fm,'stress':fs,'gates':gates,'accepted':all(gates.values())}
        result['accepted'] = all(gates.values())
        tr.to_csv(OUT/'final_trades.csv',index=False)
        eq.to_csv(OUT/'final_equity.csv',index=False)
        frame[frame['Date'].between(FINAL_START,FINAL_END)][['Date','Code','Name','Market','regime','pred_2026','eligible']].to_csv(OUT/'final_predictions.csv',index=False)

    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('===KOREA_MARCAP_V1_RESULT_BEGIN===')
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('===KOREA_MARCAP_V1_RESULT_END===')


if __name__ == '__main__':
    main()
