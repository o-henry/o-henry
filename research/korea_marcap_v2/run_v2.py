from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V1_PATH = HERE.parent / 'korea_marcap_v1' / 'run.py'
SPEC = importlib.util.spec_from_file_location('korea_marcap_v1_core_for_v2', V1_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError('unable to import v1 core')
core = importlib.util.module_from_spec(SPEC)
sys.modules['korea_marcap_v1_core_for_v2'] = core
SPEC.loader.exec_module(core)

ROOT = Path('/tmp/korea_marcap_v2')
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 2_000_000.0
TRAIN_END = pd.Timestamp('2023-12-31')
CAL_START = pd.Timestamp('2024-01-01')
CAL_END = pd.Timestamp('2024-12-31')
VALID_START = pd.Timestamp('2025-01-01')
VALID_END = pd.Timestamp('2025-12-31')
FINAL_START = pd.Timestamp('2026-01-01')
FINAL_END = pd.Timestamp('2026-08-20')

PARAMS = {
    'KOSPI': {
        'n_estimators': 450,
        'learning_rate': 0.025,
        'num_leaves': 31,
        'min_child_samples': 250,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
    },
    'KOSDAQ': {
        'n_estimators': 250,
        'learning_rate': 0.045,
        'num_leaves': 12,
        'min_child_samples': 120,
        'reg_alpha': 0.0,
        'reg_lambda': 2.0,
    },
}

# Keep a liquid, point-in-time universe before expensive feature construction.
_ORIG_LOAD = core.load_data


def load_liquid_data() -> pd.DataFrame:
    d = _ORIG_LOAD()
    d['mcap_rank_day'] = d.groupby(['Date', 'Market'])['Marcap'].rank(method='first', ascending=False)
    d['amount_rank_day'] = d.groupby(['Date', 'Market'])['Amount'].rank(method='first', ascending=False)
    d = d[(d['mcap_rank_day'] <= 500) | (d['amount_rank_day'] <= 250)].copy()
    return d.drop(columns=['mcap_rank_day', 'amount_rank_day'])


core.load_data = load_liquid_data

MODEL_FEATURES = core.FEATURES + ['regime_code']
REGIME_CODE = {'panic': 0, 'bear': 1, 'transition': 2, 'neutral': 3, 'bull': 4}


def add_regime_code(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame['regime_code'] = frame['regime'].map(REGIME_CODE).fillna(2).astype(float)
    return frame


def fit_rank_model(train: pd.DataFrame, target: str, market: str) -> lgb.LGBMRegressor:
    p = PARAMS[market]
    x = train[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train[target].astype(float)
    w = np.sqrt(train['amount_rank'].clip(0.05, 1.0))
    model = lgb.LGBMRegressor(
        objective='regression_l1',
        random_state=20260823,
        n_jobs=-1,
        verbosity=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        **p,
    )
    model.fit(x, y, sample_weight=w)
    return model


def predict(model: lgb.LGBMRegressor, frame: pd.DataFrame) -> np.ndarray:
    x = frame[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return model.predict(x)


def add_prediction_ranks(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    frame = frame.copy()
    for h in (1, 2):
        pcol = f'pred_h{h}_{suffix}'
        frame[f'pred_rank_h{h}_{suffix}'] = frame.groupby(['Date', 'Market'])[pcol].rank(pct=True, method='average')
    return frame


def calibration_table(frame: pd.DataFrame, suffix: str) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for market in ('KOSPI', 'KOSDAQ'):
        tables[market] = {}
        q0 = frame[(frame['Market'] == market) & frame['eligible']].copy()
        for h in (1, 2):
            rank_col = f'pred_rank_h{h}_{suffix}'
            target_col = f'fwd{h}'
            q = q0.dropna(subset=[rank_col, target_col]).copy()
            q['bin'] = np.ceil(q[rank_col].clip(0.001, 1.0) * 10).astype(int).clip(1, 10)
            q['net_target'] = q[target_col].clip(-0.25, 0.25) - core.BASE_COST
            global_mean = float(q['net_target'].mean()) if len(q) else -core.BASE_COST
            by_bin = q.groupby('bin')['net_target'].agg(['mean', 'count']).reset_index()
            bin_map: dict[int, float] = {}
            for _, row in by_bin.iterrows():
                n = float(row['count'])
                bin_map[int(row['bin'])] = float((n * row['mean'] + 120.0 * global_mean) / (n + 120.0))
            by_regime = q.groupby(['regime', 'bin'])['net_target'].agg(['mean', 'count']).reset_index()
            regime_map: dict[str, dict[int, float]] = {}
            for _, row in by_regime.iterrows():
                regime = str(row['regime'])
                b = int(row['bin'])
                prior = bin_map.get(b, global_mean)
                n = float(row['count'])
                regime_map.setdefault(regime, {})[b] = float((n * row['mean'] + 80.0 * prior) / (n + 80.0))
            tables[market][h] = {
                'global_mean': global_mean,
                'bin_map': {str(k): v for k, v in bin_map.items()},
                'regime_map': {r: {str(k): v for k, v in m.items()} for r, m in regime_map.items()},
            }
    return tables


def apply_calibration(frame: pd.DataFrame, suffix: str, tables: dict[str, Any]) -> pd.DataFrame:
    frame = frame.copy()
    for h in (1, 2):
        rank_col = f'pred_rank_h{h}_{suffix}'
        out = np.full(len(frame), np.nan, dtype=float)
        for market in ('KOSPI', 'KOSDAQ'):
            mask = frame['Market'].eq(market) & frame[rank_col].notna()
            if not mask.any():
                continue
            t = tables[market][h]
            ranks = frame.loc[mask, rank_col].clip(0.001, 1.0)
            bins = np.ceil(ranks * 10).astype(int).clip(1, 10)
            regimes = frame.loc[mask, 'regime'].astype(str)
            vals: list[float] = []
            for regime, b in zip(regimes, bins):
                rm = t['regime_map'].get(regime, {})
                vals.append(float(rm.get(str(int(b)), t['bin_map'].get(str(int(b)), t['global_mean']))))
            out[np.flatnonzero(mask.to_numpy())] = vals
        frame[f'exp_h{h}_{suffix}'] = out
    e1 = frame[f'exp_h1_{suffix}'].fillna(-9.0)
    e2 = frame[f'exp_h2_{suffix}'].fillna(-9.0)
    frame[f'best_h_{suffix}'] = np.where(e2 > e1, 2, 1)
    frame[f'expected_net_{suffix}'] = np.maximum(e1, e2)
    frame[f'best_rank_{suffix}'] = np.where(
        frame[f'best_h_{suffix}'].eq(2),
        frame[f'pred_rank_h2_{suffix}'],
        frame[f'pred_rank_h1_{suffix}'],
    )
    return frame


def candidate_score(frame: pd.DataFrame, suffix: str) -> pd.Series:
    expected = frame[f'expected_net_{suffix}'].fillna(-9.0)
    model_rank = frame[f'best_rank_{suffix}'].fillna(0.0)
    r = frame['regime'].astype(str)
    trend = 0.35 * frame['rel20_rank'] + 0.25 * frame['ret60_rank'] + 0.20 * frame['event_rank'] + 0.20 * frame['close_loc_rank']
    pullback = 0.35 * frame['rel60_rank'] + 0.25 * (1.0 - frame.groupby(['Date', 'Market'])['ret3'].rank(pct=True)) + 0.20 * frame['close_loc_rank'] + 0.20 * frame['event_rank']
    defensive = 0.40 * frame['rel20_rank'] + 0.25 * (1.0 - frame['beta20_rank']) + 0.20 * frame['ret5_rank'] + 0.15 * frame['event_rank']
    tech = trend.copy()
    tech = np.where(r.eq('transition'), pullback, tech)
    tech = np.where(r.eq('bear'), defensive, tech)
    tech = np.where(r.eq('panic'), defensive, tech)
    return expected + 0.006 * model_rank + 0.004 * pd.Series(tech, index=frame.index)


def base_candidate_mask(frame: pd.DataFrame) -> pd.Series:
    r = frame['regime'].astype(str)
    bull = (
        frame['dist_ma60'].gt(-0.04)
        & frame['rel20_rank'].ge(0.60)
        & frame['dd20'].ge(-0.20)
        & frame['event_rank'].ge(0.35)
    )
    neutral = (
        frame['rel20_rank'].ge(0.72)
        & frame['dist_ma20'].gt(-0.04)
        & frame['close_loc'].ge(0.45)
    )
    transition = (
        frame['ret60'].gt(0)
        & frame['rel20_rank'].ge(0.82)
        & frame['ret3'].between(-0.12, 0.02)
        & frame['dd20'].between(-0.20, -0.005)
        & frame['close_loc'].ge(0.55)
        & frame['event_rank'].ge(0.50)
    )
    bear = (
        frame['ret20'].gt(0)
        & frame['rel20_rank'].ge(0.96)
        & frame['beta20'].le(0.80)
        & frame['dist_ma20'].gt(-0.025)
        & frame['event_rank'].ge(0.60)
    )
    panic = (
        frame['ret5'].gt(0)
        & frame['rel5_rank'].ge(0.985)
        & frame['event_rank'].ge(0.85)
        & frame['close_loc'].ge(0.70)
        & frame['beta20'].le(1.0)
    )
    return (
        (r.eq('bull') & bull)
        | (r.eq('neutral') & neutral)
        | (r.eq('transition') & transition)
        | (r.eq('bear') & bear)
        | (r.eq('panic') & panic)
    )


@dataclasses.dataclass(frozen=True)
class Config:
    min_rank: float
    min_expected: float
    stop_atr: float
    target_r: float
    bull_exp: float
    neutral_exp: float
    transition_exp: float
    bear_exp: float
    panic_exp: float
    monthly_stop: float
    loss_pause: int

    @property
    def id(self) -> str:
        return (
            f'r{self.min_rank:.2f}-e{self.min_expected:.3f}-st{self.stop_atr:.2f}'
            f'-t{self.target_r:.1f}-bu{self.bull_exp:.2f}-ne{self.neutral_exp:.2f}'
            f'-tr{self.transition_exp:.2f}-ba{self.bear_exp:.2f}-pa{self.panic_exp:.2f}'
            f'-ms{abs(self.monthly_stop):.2f}-lp{self.loss_pause}'
        )


def configs() -> list[Config]:
    out: list[Config] = []
    for rank, expected, stop, target, transition, panic in itertools.product(
        [0.88, 0.92, 0.95],
        [0.003, 0.006, 0.009],
        [1.25, 1.75],
        [2.5, 4.0],
        [0.30, 0.50],
        [0.00, 0.10],
    ):
        out.append(Config(rank, expected, stop, target, 0.75, 0.50, transition, 0.20, panic, -0.08, 3))
    return out


def regime_exposure(cfg: Config, regime: str) -> float:
    return {
        'bull': cfg.bull_exp,
        'neutral': cfg.neutral_exp,
        'transition': cfg.transition_exp,
        'bear': cfg.bear_exp,
        'panic': cfg.panic_exp,
    }.get(regime, cfg.neutral_exp)


def regime_risk(regime: str) -> float:
    return {
        'bull': 0.015,
        'neutral': 0.011,
        'transition': 0.008,
        'bear': 0.005,
        'panic': 0.003,
    }.get(regime, 0.008)


def prepare_candidates(frame: pd.DataFrame, suffix: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    x = frame[frame['Date'].between(start, end)].copy()
    x = x[x['eligible'] & x[f'expected_net_{suffix}'].notna() & x[f'best_rank_{suffix}'].notna()]
    x = x[base_candidate_mask(x)].copy()
    x['candidate_score'] = candidate_score(x, suffix)
    return x.sort_values(['Date', 'candidate_score', 'Amount'], ascending=[True, False, False])


def empty_metrics(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    return {
        'start': str(start.date()), 'end': str(end.date()), 'return': 0.0,
        'final_equity': INITIAL_CAPITAL, 'profit_factor': 0.0, 'max_drawdown': 0.0,
        'trades': 0, 'wins': 0, 'win_rate': 0.0, 'monthly_geom': 0.0,
        'monthly_median': 0.0, 'positive_month_ratio': 0.0, 'monthly_returns': {},
        'market_pnl': {}, 'market_trades': {}, 'regime_pnl': {}, 'regime_trades': {},
        'largest_winner_share': 0.0,
    }


def calc_metrics(trades: pd.DataFrame, equity: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    if equity.empty:
        return empty_metrics(start, end)
    eq = equity.copy()
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.sort_values('date').drop_duplicates('date', keep='last')
    eq['daily_return'] = eq['equity'].pct_change().fillna(0.0)
    eq['month'] = eq['date'].dt.to_period('M')
    monthly = eq.groupby('month')['daily_return'].apply(lambda s: float((1 + s).prod() - 1))
    ret = float(eq['equity'].iloc[-1] / INITIAL_CAPITAL - 1.0)
    dd = eq['equity'] / eq['equity'].cummax() - 1.0
    if trades.empty:
        return empty_metrics(start, end) | {
            'return': ret,
            'final_equity': float(eq['equity'].iloc[-1]),
            'max_drawdown': float(dd.min()),
            'monthly_geom': float((1 + monthly).prod() ** (1 / max(1, len(monthly))) - 1),
            'monthly_median': float(monthly.median()) if len(monthly) else 0.0,
            'positive_month_ratio': float((monthly > 0).mean()) if len(monthly) else 0.0,
            'monthly_returns': {str(k): float(v) for k, v in monthly.items()},
        }
    gp = float(trades.loc[trades['pnl'] > 0, 'pnl'].sum())
    gl = float(-trades.loc[trades['pnl'] < 0, 'pnl'].sum())
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)
    winners = trades.loc[trades['pnl'] > 0, 'pnl']
    largest_share = float(winners.max() / winners.sum()) if len(winners) and winners.sum() > 0 else 0.0
    return {
        'start': str(start.date()), 'end': str(end.date()), 'return': ret,
        'final_equity': float(eq['equity'].iloc[-1]), 'profit_factor': float(pf),
        'max_drawdown': float(dd.min()), 'trades': int(len(trades)),
        'wins': int((trades['pnl'] > 0).sum()), 'win_rate': float((trades['pnl'] > 0).mean()),
        'monthly_geom': float((1 + monthly).prod() ** (1 / max(1, len(monthly))) - 1),
        'monthly_median': float(monthly.median()) if len(monthly) else 0.0,
        'positive_month_ratio': float((monthly > 0).mean()) if len(monthly) else 0.0,
        'monthly_returns': {str(k): float(v) for k, v in monthly.items()},
        'market_pnl': {str(k): float(v) for k, v in trades.groupby('Market')['pnl'].sum().to_dict().items()},
        'market_trades': {str(k): int(v) for k, v in trades.groupby('Market').size().to_dict().items()},
        'regime_pnl': {str(k): float(v) for k, v in trades.groupby('regime')['pnl'].sum().to_dict().items()},
        'regime_trades': {str(k): int(v) for k, v in trades.groupby('regime').size().to_dict().items()},
        'largest_winner_share': largest_share,
    }


def simulate(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    cfg: Config,
    suffix: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_mult: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    raw = frame[frame['Date'].between(start, end + pd.Timedelta(days=10))].sort_values(['Code', 'Date'])
    all_dates = [pd.Timestamp(d) for d in sorted(raw[raw['Date'].between(start, end)]['Date'].unique())]
    if not all_dates:
        return empty_metrics(start, end), pd.DataFrame(), pd.DataFrame()
    date_to_i = {d: i for i, d in enumerate(all_dates)}
    by_code = {c: g.set_index('Date').sort_index() for c, g in raw.groupby('Code')}
    by_signal = {pd.Timestamp(d): g for d, g in candidates.groupby('Date')}

    cash = INITIAL_CAPITAL
    equity_value = INITIAL_CAPITAL
    positions: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    last_close: dict[str, float] = {}
    market_loss_streak = {'KOSPI': 0, 'KOSDAQ': 0}
    market_block_until = {'KOSPI': -1, 'KOSDAQ': -1}
    month_key = None
    month_start_equity = INITIAL_CAPITAL
    month_blocked = False
    week_peak = INITIAL_CAPITAL

    def close_trade(pos: dict[str, Any], current: pd.Timestamp, exit_price: float, reason: str) -> None:
        nonlocal cash
        proceeds = pos['qty'] * exit_price * (1 - (core.SELL_FEE_TAX + core.SLIPPAGE) * cost_mult)
        cash += proceeds
        pnl = proceeds - pos['cash_out']
        market = pos['Market']
        if pnl < 0:
            market_loss_streak[market] += 1
            if market_loss_streak[market] >= 2:
                market_block_until[market] = date_to_i.get(current, 0) + cfg.loss_pause
        else:
            market_loss_streak[market] = 0
        trade_rows.append({
            **{k: v for k, v in pos.items() if k != 'best_high'},
            'exit_date': str(current.date()), 'exit_price': float(exit_price),
            'reason': reason, 'pnl': float(pnl), 'return': float(pnl / pos['cash_out']),
        })

    for current in all_dates:
        idx = date_to_i[current]
        current_month = current.to_period('M')
        if month_key != current_month:
            month_key = current_month
            marked0 = sum(p['qty'] * last_close.get(p['Code'], p['entry_price']) for p in positions)
            month_start_equity = cash + marked0
            month_blocked = False
        if current.weekday() == 0:
            week_peak = equity_value

        still_open: list[dict[str, Any]] = []
        for pos in positions:
            hist = by_code.get(pos['Code'])
            if hist is None or current not in hist.index:
                still_open.append(pos)
                continue
            bar = hist.loc[current]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            open_px = float(bar['Open'])
            high_px = float(bar['High'])
            low_px = float(bar['Low'])
            close_px = float(bar['Close'])
            last_close[pos['Code']] = close_px
            managed_stop = pos['stop']
            if pos['best_high'] >= pos['entry_price'] + pos['risk']:
                managed_stop = max(managed_stop, pos['entry_price'] * (1 + core.BASE_COST * cost_mult))
            exit_price = None
            reason = None
            if low_px <= managed_stop:
                exit_price = min(open_px, managed_stop)
                reason = 'stop'
            elif high_px >= pos['target']:
                exit_price = pos['target']
                reason = 'target'
            elif current >= pos['planned_exit_date']:
                exit_price = close_px
                reason = 'time'
            if exit_price is None:
                pos['best_high'] = max(pos['best_high'], high_px)
                still_open.append(pos)
            else:
                close_trade(pos, current, exit_price, reason)
        positions = still_open

        marked = sum(p['qty'] * last_close.get(p['Code'], p['entry_price']) for p in positions)
        equity_value = cash + marked
        week_peak = max(week_peak, equity_value)
        if month_start_equity > 0 and equity_value / month_start_equity - 1.0 <= cfg.monthly_stop:
            month_blocked = True
        week_throttle = 0.5 if week_peak > 0 and equity_value / week_peak - 1.0 <= -0.04 else 1.0

        signal_date = all_dates[idx - 1] if idx > 0 else None
        if signal_date is not None and not month_blocked and len(positions) < 2:
            cand = by_signal.get(signal_date)
            if cand is not None and not cand.empty:
                cand = cand[
                    cand[f'best_rank_{suffix}'].ge(cfg.min_rank)
                    & cand[f'expected_net_{suffix}'].ge(cfg.min_expected)
                ].copy()
                if not cand.empty:
                    held = {p['Code'] for p in positions}
                    cand = cand[~cand['Code'].isin(held)]
                    cand = cand.sort_values(['candidate_score', f'expected_net_{suffix}', 'Amount'], ascending=False)
                    selected = None
                    for _, row in cand.iterrows():
                        market = str(row['Market'])
                        regime = str(row['regime'])
                        premium = 0.003 if regime == 'bear' else 0.006 if regime == 'panic' else 0.0
                        if float(row[f'expected_net_{suffix}']) < cfg.min_expected + premium:
                            continue
                        if idx <= market_block_until.get(market, -1):
                            continue
                        if regime == 'panic' and cfg.panic_exp <= 0:
                            continue
                        selected = row
                        break
                    if selected is not None:
                        row = selected
                        hist = by_code.get(row['Code'])
                        if hist is not None and current in hist.index:
                            bar = hist.loc[current]
                            if isinstance(bar, pd.DataFrame):
                                bar = bar.iloc[-1]
                            regime = str(row['regime'])
                            exposure = regime_exposure(cfg, regime) * week_throttle
                            entry_price = float(bar['Open']) * (1 + (core.BUY_FEE + core.SLIPPAGE) * cost_mult)
                            atr = max(float(row['atr20']), entry_price * 0.005)
                            risk = max(cfg.stop_atr * atr, entry_price * 0.012)
                            qty_cap = int((equity_value * exposure) // entry_price)
                            qty_risk = int((equity_value * regime_risk(regime)) // risk)
                            qty_cash = int(cash // entry_price)
                            qty = max(0, min(qty_cap, qty_risk, qty_cash))
                            if qty >= 1:
                                horizon = int(row[f'best_h_{suffix}'])
                                cash_out = qty * entry_price
                                stop = entry_price - risk
                                target = entry_price + cfg.target_r * risk
                                if horizon == 1:
                                    open_px = float(bar['Open'])
                                    high_px = float(bar['High'])
                                    low_px = float(bar['Low'])
                                    close_px = float(bar['Close'])
                                    cash -= cash_out
                                    pos = {
                                        'signal_date': str(signal_date.date()), 'entry_date': str(current.date()),
                                        'planned_exit_date': str(current.date()), 'Code': row['Code'], 'Name': row['Name'],
                                        'Market': row['Market'], 'regime': regime, 'candidate_score': float(row['candidate_score']),
                                        'expected_net': float(row[f'expected_net_{suffix}']), 'model_rank': float(row[f'best_rank_{suffix}']),
                                        'horizon': horizon, 'entry_price': entry_price, 'qty': qty, 'cash_out': cash_out,
                                        'stop': stop, 'target': target, 'risk': risk, 'best_high': high_px,
                                        'config_id': cfg.id,
                                    }
                                    if low_px <= stop:
                                        close_trade(pos, current, min(open_px, stop), 'same_day_stop')
                                    elif high_px >= target:
                                        close_trade(pos, current, target, 'same_day_target')
                                    else:
                                        close_trade(pos, current, close_px, 'same_day_time')
                                else:
                                    exit_idx = min(idx + horizon - 1, len(all_dates) - 1)
                                    planned_exit = all_dates[exit_idx]
                                    cash -= cash_out
                                    positions.append({
                                        'signal_date': str(signal_date.date()), 'entry_date': current,
                                        'planned_exit_date': planned_exit, 'Code': row['Code'], 'Name': row['Name'],
                                        'Market': row['Market'], 'regime': regime, 'candidate_score': float(row['candidate_score']),
                                        'expected_net': float(row[f'expected_net_{suffix}']), 'model_rank': float(row[f'best_rank_{suffix}']),
                                        'horizon': horizon, 'entry_price': entry_price, 'qty': qty, 'cash_out': cash_out,
                                        'stop': stop, 'target': target, 'risk': risk, 'best_high': float(bar['High']),
                                        'config_id': cfg.id,
                                    })
                                    last_close[row['Code']] = float(bar['Close'])

        marked = sum(p['qty'] * last_close.get(p['Code'], p['entry_price']) for p in positions)
        equity_value = cash + marked
        equity_rows.append({
            'date': str(current.date()), 'equity': float(equity_value), 'cash': float(cash),
            'open_positions': len(positions), 'month_blocked': month_blocked,
        })

    if positions:
        current = all_dates[-1]
        for pos in positions:
            hist = by_code.get(pos['Code'])
            if hist is None or current not in hist.index:
                continue
            bar = hist.loc[current]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            close_trade(pos, current, float(bar['Close']), 'end')
        positions = []
        equity_value = cash
        equity_rows.append({'date': str(current.date()), 'equity': float(equity_value), 'cash': float(cash), 'open_positions': 0, 'month_blocked': month_blocked})

    tr = pd.DataFrame(trade_rows)
    eq = pd.DataFrame(equity_rows)
    return calc_metrics(tr, eq, start, end), tr, eq


def strategy_score(metrics: dict[str, Any], stress: dict[str, Any]) -> float:
    positive_markets = sum(1 for v in metrics.get('market_pnl', {}).values() if v >= 0)
    return (
        2.0 * metrics['monthly_geom'] + 0.8 * metrics['return']
        + 0.12 * min(metrics['profit_factor'], 3.0)
        + 0.15 * metrics['positive_month_ratio']
        + 0.08 * positive_markets
        + 0.30 * min(0.0, metrics['max_drawdown'] + 0.12)
        + 0.40 * min(0.0, stress['return'])
        - 0.15 * max(0.0, metrics['largest_winner_share'] - 0.30)
    )


def strategy_gate(metrics: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    mp = metrics.get('market_pnl', {})
    mt = metrics.get('market_trades', {})
    return {
        'return_10pct': metrics['return'] >= 0.10,
        'monthly_geom_0_8pct': metrics['monthly_geom'] >= 0.008,
        'profit_factor_1_20': metrics['profit_factor'] >= 1.20,
        'max_drawdown_15pct': metrics['max_drawdown'] >= -0.15,
        'positive_months_58pct': metrics['positive_month_ratio'] >= 0.58,
        'trades_30': metrics['trades'] >= 30,
        'cost_stress_positive': stress['return'] > 0,
        'kospi_nonnegative': mp.get('KOSPI', -1) >= 0,
        'kosdaq_nonnegative': mp.get('KOSDAQ', -1) >= 0,
        'kospi_trades_5': mt.get('KOSPI', 0) >= 5,
        'kosdaq_trades_5': mt.get('KOSDAQ', 0) >= 5,
        'largest_winner_below_35pct': metrics['largest_winner_share'] <= 0.35,
    }


def final_gate(metrics: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    mp = metrics.get('market_pnl', {})
    mt = metrics.get('market_trades', {})
    return {
        'return_positive': metrics['return'] > 0,
        'monthly_geom_5pct': metrics['monthly_geom'] >= 0.05,
        'profit_factor_1_40': metrics['profit_factor'] >= 1.40,
        'max_drawdown_15pct': metrics['max_drawdown'] >= -0.15,
        'positive_months_75pct': metrics['positive_month_ratio'] >= 0.75,
        'trades_35': metrics['trades'] >= 35,
        'cost_stress_positive': stress['return'] > 0,
        'kospi_positive': mp.get('KOSPI', -1) > 0,
        'kosdaq_positive': mp.get('KOSDAQ', -1) > 0,
        'kospi_trades_5': mt.get('KOSPI', 0) >= 5,
        'kosdaq_trades_5': mt.get('KOSDAQ', 0) >= 5,
        'largest_winner_below_30pct': metrics['largest_winner_share'] <= 0.30,
    }


def train_predict_period(
    frame: pd.DataFrame,
    train_end: pd.Timestamp,
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
    suffix: str,
) -> pd.DataFrame:
    frame = frame.copy()
    for market in ('KOSPI', 'KOSDAQ'):
        train = frame[
            frame['Market'].eq(market)
            & frame['Date'].between(pd.Timestamp('2020-01-01'), train_end)
            & frame['eligible']
        ].copy()
        if len(train) > 700_000:
            train = train.sample(700_000, random_state=20260823)
        mask = frame['Market'].eq(market) & frame['Date'].between(pred_start, pred_end)
        for h in (1, 2):
            target = f'fwd{h}_rank'
            t = train[train[target].notna()].copy()
            model = fit_rank_model(t, target, market)
            frame.loc[mask, f'pred_h{h}_{suffix}'] = predict(model, frame.loc[mask])
    return add_prediction_ranks(frame, suffix)


def main() -> None:
    raw = core.load_data()
    print('raw', len(raw), raw['Date'].min(), raw['Date'].max(), raw['Code'].nunique(), flush=True)
    frame, market = core.build_features(raw)
    frame = add_regime_code(frame)
    print('features', len(frame), frame['Date'].min(), frame['Date'].max(), flush=True)

    # 2024 is calibration-only, using models trained through 2023.
    frame = train_predict_period(frame, TRAIN_END, CAL_START, CAL_END, 'cal')
    cal_tables = calibration_table(frame[frame['Date'].between(CAL_START, CAL_END)], 'cal')

    # 2025 strategy selection uses models trained through 2024 and calibration frozen from 2024.
    frame = train_predict_period(frame, CAL_END, VALID_START, VALID_END, 'valid')
    frame = apply_calibration(frame, 'valid', cal_tables)
    candidates_2025 = prepare_candidates(frame, 'valid', VALID_START, VALID_END)
    print('candidate rows 2025', len(candidates_2025), flush=True)

    config_rows: list[dict[str, Any]] = []
    best_pass: tuple[Config, dict[str, Any]] | None = None
    best_any: tuple[Config, dict[str, Any]] | None = None
    all_configs = configs()
    print('configs', len(all_configs), flush=True)
    for i, cfg in enumerate(all_configs):
        metrics, trades, equity = simulate(frame, candidates_2025, cfg, 'valid', VALID_START, VALID_END, 1.0)
        stress, _, _ = simulate(frame, candidates_2025, cfg, 'valid', VALID_START, VALID_END, 1.5)
        gates = strategy_gate(metrics, stress)
        pack = {
            'config_id': cfg.id, 'config': dataclasses.asdict(cfg), 'metrics': metrics,
            'stress': stress, 'gates': gates, 'passed': all(gates.values()),
            'score': strategy_score(metrics, stress),
        }
        config_rows.append(pack)
        if best_any is None or pack['score'] > best_any[1]['score']:
            best_any = (cfg, pack)
        if pack['passed'] and (best_pass is None or pack['score'] > best_pass[1]['score']):
            best_pass = (cfg, pack)
        if (i + 1) % 24 == 0:
            print(i + 1, '/', len(all_configs), flush=True)

    flat_rows = []
    for p in config_rows:
        row = {'config_id': p['config_id'], 'passed': p['passed'], 'score': p['score']}
        row.update({f'c_{k}': v for k, v in p['config'].items()})
        row.update({f'm_{k}': v for k, v in p['metrics'].items() if not isinstance(v, dict)})
        row.update({f's_{k}': v for k, v in p['stress'].items() if not isinstance(v, dict)})
        flat_rows.append(row)
    pd.DataFrame(flat_rows).sort_values('score', ascending=False).to_csv(OUT / 'config_summary.csv', index=False)

    result: dict[str, Any] = {
        'version': 'korea-marcap-v2-dual-horizon-calibrated',
        'data': {
            'rows': int(len(raw)), 'symbols': int(raw['Code'].nunique()),
            'min_date': str(raw['Date'].min().date()), 'max_date': str(raw['Date'].max().date()),
            'source': 'FinanceData/marcap',
        },
        'periods': {
            'model_train': ['2020-01-01', str(TRAIN_END.date())],
            'calibration': [str(CAL_START.date()), str(CAL_END.date())],
            'strategy_select': [str(VALID_START.date()), str(VALID_END.date())],
            'final': [str(FINAL_START.date()), str(FINAL_END.date())],
        },
        'config_trials': len(config_rows),
        'passing_configs': sum(1 for p in config_rows if p['passed']),
        'best_2025': best_any[1] if best_any else None,
        'strategy_gate_passed': best_pass is not None,
        'final_opened': best_pass is not None,
    }

    if best_pass is None:
        result['accepted'] = False
        result['reason'] = 'No v2 configuration passed the frozen 2025 gate; 2026 remained unopened.'
    else:
        cfg, selected_pack = best_pass
        result['selected_2025'] = selected_pack

        # 2025 OOS predictions and outcomes form the only calibration for the untouched 2026 interval.
        cal_2025 = calibration_table(frame[frame['Date'].between(VALID_START, VALID_END)], 'valid')
        frame = train_predict_period(frame, VALID_END, FINAL_START, FINAL_END, 'final')
        frame = apply_calibration(frame, 'final', cal_2025)
        candidates_2026 = prepare_candidates(frame, 'final', FINAL_START, FINAL_END)
        metrics, trades, equity = simulate(frame, candidates_2026, cfg, 'final', FINAL_START, FINAL_END, 1.0)
        stress, _, _ = simulate(frame, candidates_2026, cfg, 'final', FINAL_START, FINAL_END, 1.5)
        gates = final_gate(metrics, stress)
        accepted = all(gates.values())
        result['final'] = {'metrics': metrics, 'stress': stress, 'gates': gates, 'accepted': accepted}
        result['accepted'] = accepted
        trades.to_csv(OUT / 'final_trades.csv', index=False)
        equity.to_csv(OUT / 'final_equity.csv', index=False)
        candidates_2026[[
            'Date', 'Code', 'Name', 'Market', 'regime', 'candidate_score',
            'expected_net_final', 'best_rank_final', 'best_h_final', 'Amount', 'Marcap',
        ]].to_csv(OUT / 'final_candidates.csv', index=False)

    (OUT / 'calibration_2024.json').write_text(json.dumps(cal_tables, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('===KOREA_MARCAP_V2_RESULT_BEGIN===')
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print('===KOREA_MARCAP_V2_RESULT_END===')


if __name__ == '__main__':
    main()
