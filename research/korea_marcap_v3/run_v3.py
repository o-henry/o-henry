from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V2_PATH = HERE.parent / 'korea_marcap_v2' / 'run_v2.py'
SPEC = importlib.util.spec_from_file_location('korea_marcap_v2_core_for_v3', V2_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError('unable to import v2 core')
v2 = importlib.util.module_from_spec(SPEC)
sys.modules['korea_marcap_v2_core_for_v3'] = v2
SPEC.loader.exec_module(v2)
core = v2.core

ROOT = Path('/tmp/korea_marcap_v3')
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp('2023-12-31')
CAL_START = pd.Timestamp('2024-01-01')
CAL_END = pd.Timestamp('2024-12-31')
VALID_START = pd.Timestamp('2025-01-01')
VALID_END = pd.Timestamp('2025-12-31')
FINAL_START = pd.Timestamp('2026-01-01')
FINAL_END = pd.Timestamp('2026-08-20')
INITIAL_CAPITAL = 2_000_000.0

TAIL_EDGES = np.array([0.0, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 1.000001])


def rank_band(values: pd.Series) -> pd.Series:
    return pd.Series(np.digitize(values.to_numpy(float), TAIL_EDGES[1:-1], right=False) + 1, index=values.index)


def tail_calibration(frame: pd.DataFrame, suffix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for market in ('KOSPI', 'KOSDAQ'):
        out[market] = {}
        base = frame[(frame['Market'] == market) & frame['eligible']].copy()
        for h in (1, 2):
            rank_col = f'pred_rank_h{h}_{suffix}'
            target_col = f'fwd{h}'
            q = base.dropna(subset=[rank_col, target_col]).copy()
            q['band'] = rank_band(q[rank_col]).astype(int)
            q['net_target'] = q[target_col].clip(-0.25, 0.25) - core.BASE_COST
            global_mean = float(q['net_target'].mean()) if len(q) else -core.BASE_COST
            by_band = q.groupby('band')['net_target'].agg(['mean', 'count']).reset_index()
            band_map: dict[int, float] = {}
            counts: dict[int, int] = {}
            for _, row in by_band.iterrows():
                b = int(row['band'])
                n = float(row['count'])
                counts[b] = int(n)
                band_map[b] = float((n * row['mean'] + 80.0 * global_mean) / (n + 80.0))
            regime_map: dict[str, dict[int, float]] = {}
            by_regime = q.groupby(['regime', 'band'])['net_target'].agg(['mean', 'count']).reset_index()
            for _, row in by_regime.iterrows():
                r = str(row['regime'])
                b = int(row['band'])
                n = float(row['count'])
                prior = band_map.get(b, global_mean)
                regime_map.setdefault(r, {})[b] = float((n * row['mean'] + 60.0 * prior) / (n + 60.0))
            out[market][h] = {
                'global_mean': global_mean,
                'band_map': {str(k): v for k, v in band_map.items()},
                'counts': {str(k): v for k, v in counts.items()},
                'regime_map': {r: {str(k): v for k, v in vals.items()} for r, vals in regime_map.items()},
            }
    return out


def apply_tail_calibration(frame: pd.DataFrame, suffix: str, table: dict[str, Any]) -> pd.DataFrame:
    frame = frame.copy()
    for h in (1, 2):
        rank_col = f'pred_rank_h{h}_{suffix}'
        expected = np.full(len(frame), np.nan, dtype=float)
        for market in ('KOSPI', 'KOSDAQ'):
            mask = frame['Market'].eq(market) & frame[rank_col].notna()
            if not mask.any():
                continue
            ranks = frame.loc[mask, rank_col].astype(float)
            bands = rank_band(ranks).astype(int)
            regimes = frame.loc[mask, 'regime'].astype(str)
            t = table[market][h]
            vals: list[float] = []
            for regime, band in zip(regimes, bands):
                b = str(int(band))
                vals.append(float(t['regime_map'].get(regime, {}).get(b, t['band_map'].get(b, t['global_mean']))))
            expected[np.flatnonzero(mask.to_numpy())] = vals
        frame[f'exp_h{h}_{suffix}'] = expected
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


def predict_period(
    frame: pd.DataFrame,
    train_end: pd.Timestamp,
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
    suffix: str,
) -> pd.DataFrame:
    frame = frame.copy()
    for h in (1, 2):
        frame[f'pred_h{h}_{suffix}'] = np.nan
    for market in ('KOSPI', 'KOSDAQ'):
        train = frame[
            frame['Market'].eq(market)
            & frame['Date'].between(pd.Timestamp('2020-01-01'), train_end)
            & frame['eligible']
        ].copy()
        pred_mask = frame['Market'].eq(market) & frame['Date'].between(pred_start, pred_end)
        for h in (1, 2):
            target = f'fwd{h}_rank'
            t = train[train[target].notna()].copy()
            if len(t) > 750_000:
                t = t.sample(750_000, random_state=20260823 + h)
            model = v2.fit_rank_model(t, target, market)
            frame.loc[pred_mask, f'pred_h{h}_{suffix}'] = v2.predict(model, frame.loc[pred_mask])
    return v2.add_prediction_ranks(frame, suffix)


def tail_candidate_mask(frame: pd.DataFrame) -> pd.Series:
    r = frame['regime'].astype(str)
    liquid = frame['amount_rank'].ge(0.45) & frame['size_rank'].ge(0.25)
    not_extreme = frame['dist_ma20'].between(-0.22, 0.28) & frame['stocks_chg5'].abs().fillna(0).lt(0.10)
    bull = frame['rel20_rank'].ge(0.55) & frame['dd20'].ge(-0.22)
    neutral = frame['rel20_rank'].ge(0.70) & frame['close_loc'].ge(0.40)
    transition = (
        frame['ret60'].gt(0)
        & frame['rel60_rank'].ge(0.70)
        & frame['ret3'].between(-0.16, 0.05)
        & frame['close_loc'].ge(0.48)
    )
    bear = (
        frame['ret20'].gt(0)
        & frame['rel20_rank'].ge(0.94)
        & frame['beta20'].le(1.0)
        & frame['close_loc'].ge(0.50)
    )
    panic = (
        (frame['rel5_rank'].ge(0.985) & frame['ret5'].gt(0) & frame['close_loc'].ge(0.60))
        | (frame['ret3'].le(-0.10) & frame['event_rank'].ge(0.92) & frame['close_loc'].ge(0.75))
    )
    setup = (
        (r.eq('bull') & bull)
        | (r.eq('neutral') & neutral)
        | (r.eq('transition') & transition)
        | (r.eq('bear') & bear)
        | (r.eq('panic') & panic)
    )
    return liquid & not_extreme & setup


def candidate_score(frame: pd.DataFrame, suffix: str) -> pd.Series:
    expected = frame[f'expected_net_{suffix}'].fillna(-9.0)
    rank = frame[f'best_rank_{suffix}'].fillna(0.0)
    quality = (
        0.28 * frame['event_rank']
        + 0.24 * frame['rel20_rank']
        + 0.18 * frame['rel60_rank']
        + 0.16 * frame['close_loc_rank']
        + 0.14 * frame['amount_rank']
    )
    return expected + 0.010 * rank + 0.006 * quality


@dataclasses.dataclass(frozen=True)
class Config:
    kospi_rank: float
    kosdaq_rank: float
    min_expected: float
    stop_atr: float
    target_r: float
    solo_exposure: float
    max_entry_gap: float
    monthly_stop: float
    loss_pause: int

    @property
    def id(self) -> str:
        return (
            f'kr{self.kospi_rank:.3f}-qr{self.kosdaq_rank:.3f}-e{self.min_expected:.3f}'
            f'-st{self.stop_atr:.2f}-t{self.target_r:.1f}-sx{self.solo_exposure:.2f}'
            f'-g{self.max_entry_gap:.2f}-ms{abs(self.monthly_stop):.2f}-lp{self.loss_pause}'
        )


def configs() -> list[Config]:
    rank_pairs = [
        (0.970, 0.970),
        (0.985, 0.970),
        (0.985, 0.985),
        (0.995, 0.985),
    ]
    out: list[Config] = []
    for (kr, qr), expected, stop, target, solo in itertools.product(
        rank_pairs,
        [0.003, 0.006, 0.010],
        [1.50, 2.00],
        [4.0, 8.0],
        [0.80, 1.00],
    ):
        out.append(Config(kr, qr, expected, stop, target, solo, 0.08, -0.10, 3))
    return out


def market_threshold(cfg: Config, market: str, regime: str) -> tuple[float, float]:
    rank = cfg.kospi_rank if market == 'KOSPI' else cfg.kosdaq_rank
    expected = cfg.min_expected
    if regime == 'transition':
        rank = max(rank, 0.985)
        expected += 0.002
    elif regime == 'bear':
        rank = max(rank, 0.995)
        expected += 0.006
    elif regime == 'panic':
        rank = max(rank, 0.997)
        expected += 0.010
    return rank, expected


def risk_fraction(regime: str) -> float:
    return {
        'bull': 0.030,
        'neutral': 0.025,
        'transition': 0.018,
        'bear': 0.010,
        'panic': 0.006,
    }.get(regime, 0.018)


def exposure_multiplier(regime: str) -> float:
    return {
        'bull': 1.00,
        'neutral': 0.85,
        'transition': 0.65,
        'bear': 0.35,
        'panic': 0.20,
    }.get(regime, 0.60)


def prepare_candidates(frame: pd.DataFrame, suffix: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    x = frame[frame['Date'].between(start, end)].copy()
    x = x[x['eligible'] & x[f'expected_net_{suffix}'].notna() & x[f'best_rank_{suffix}'].notna()]
    x = x[tail_candidate_mask(x)].copy()
    x['candidate_score'] = candidate_score(x, suffix)
    return x.sort_values(['Date', 'Market', 'candidate_score', 'Amount'], ascending=[True, True, False, False])


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
        return v2.empty_metrics(start, end), pd.DataFrame(), pd.DataFrame()
    date_to_i = {d: i for i, d in enumerate(all_dates)}
    by_code = {c: g.set_index('Date').sort_index() for c, g in raw.groupby('Code')}
    by_signal = {pd.Timestamp(d): g for d, g in candidates.groupby('Date')}

    cash = INITIAL_CAPITAL
    equity_value = INITIAL_CAPITAL
    positions: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    last_close: dict[str, float] = {}
    loss_streak = {'KOSPI': 0, 'KOSDAQ': 0}
    block_until = {'KOSPI': -1, 'KOSDAQ': -1}
    month_key = None
    month_start_equity = INITIAL_CAPITAL
    month_blocked = False

    def close_trade(pos: dict[str, Any], current: pd.Timestamp, exit_price: float, reason: str) -> None:
        nonlocal cash
        proceeds = pos['qty'] * exit_price * (1 - (core.SELL_FEE_TAX + core.SLIPPAGE) * cost_mult)
        cash += proceeds
        pnl = proceeds - pos['cash_out']
        market = str(pos['Market'])
        if pnl < 0:
            loss_streak[market] += 1
            if loss_streak[market] >= 2:
                block_until[market] = date_to_i.get(current, 0) + cfg.loss_pause
        else:
            loss_streak[market] = 0
        trade_rows.append({
            **{k: v for k, v in pos.items() if k != 'best_high'},
            'exit_date': str(current.date()),
            'exit_price': float(exit_price),
            'reason': reason,
            'pnl': float(pnl),
            'return': float(pnl / pos['cash_out']),
        })

    for current in all_dates:
        idx = date_to_i[current]
        current_month = current.to_period('M')
        if month_key != current_month:
            month_key = current_month
            marked0 = sum(p['qty'] * last_close.get(p['Code'], p['entry_price']) for p in positions)
            month_start_equity = cash + marked0
            month_blocked = False

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
            if pos['best_high'] >= pos['entry_price'] + 2.0 * pos['risk']:
                managed_stop = max(managed_stop, pos['entry_price'] + 0.60 * pos['risk'])
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
        if month_start_equity > 0 and equity_value / month_start_equity - 1.0 <= cfg.monthly_stop:
            month_blocked = True

        signal_date = all_dates[idx - 1] if idx > 0 else None
        if signal_date is not None and not month_blocked:
            cand = by_signal.get(signal_date)
            if cand is not None and not cand.empty:
                held_markets = {str(p['Market']) for p in positions}
                valid_by_market: dict[str, pd.Series] = {}
                for market in ('KOSPI', 'KOSDAQ'):
                    if market in held_markets or idx <= block_until[market]:
                        continue
                    cm = cand[cand['Market'].eq(market)].copy()
                    if cm.empty:
                        continue
                    chosen = None
                    for _, row in cm.iterrows():
                        regime = str(row['regime'])
                        min_rank, min_expected = market_threshold(cfg, market, regime)
                        if float(row[f'best_rank_{suffix}']) < min_rank:
                            continue
                        if float(row[f'expected_net_{suffix}']) < min_expected:
                            continue
                        hist = by_code.get(row['Code'])
                        if hist is None or current not in hist.index:
                            continue
                        bar = hist.loc[current]
                        if isinstance(bar, pd.DataFrame):
                            bar = bar.iloc[-1]
                        gap = float(bar['Open']) / float(row['Close']) - 1.0
                        remaining_edge = float(row[f'expected_net_{suffix}']) - max(gap, 0.0)
                        if gap > cfg.max_entry_gap or remaining_edge < min_expected * 0.35:
                            continue
                        chosen = row
                        break
                    if chosen is not None:
                        valid_by_market[market] = chosen

                n_signals = len(valid_by_market)
                if n_signals:
                    base_alloc = cfg.solo_exposure if n_signals == 1 else 0.50
                    for market, row in valid_by_market.items():
                        hist = by_code.get(row['Code'])
                        bar = hist.loc[current]
                        if isinstance(bar, pd.DataFrame):
                            bar = bar.iloc[-1]
                        regime = str(row['regime'])
                        entry_price = float(bar['Open']) * (1 + (core.BUY_FEE + core.SLIPPAGE) * cost_mult)
                        atr = max(float(row['atr20']), entry_price * 0.005)
                        risk = max(cfg.stop_atr * atr, entry_price * 0.010)
                        alloc = base_alloc * exposure_multiplier(regime)
                        qty_cap = int((equity_value * alloc) // entry_price)
                        qty_risk = int((equity_value * risk_fraction(regime)) // risk)
                        qty_cash = int(cash // entry_price)
                        qty = max(0, min(qty_cap, qty_risk, qty_cash))
                        if qty < 1:
                            continue
                        horizon = int(row[f'best_h_{suffix}'])
                        cash_out = qty * entry_price
                        stop = entry_price - risk
                        target = entry_price + cfg.target_r * risk
                        pos = {
                            'signal_date': str(signal_date.date()),
                            'entry_date': str(current.date()),
                            'Code': row['Code'],
                            'Name': row['Name'],
                            'Market': market,
                            'regime': regime,
                            'candidate_score': float(row['candidate_score']),
                            'expected_net': float(row[f'expected_net_{suffix}']),
                            'model_rank': float(row[f'best_rank_{suffix}']),
                            'horizon': horizon,
                            'entry_price': entry_price,
                            'qty': qty,
                            'cash_out': cash_out,
                            'stop': stop,
                            'target': target,
                            'risk': risk,
                            'best_high': float(bar['High']),
                            'config_id': cfg.id,
                        }
                        cash -= cash_out
                        if horizon == 1:
                            open_px = float(bar['Open'])
                            high_px = float(bar['High'])
                            low_px = float(bar['Low'])
                            close_px = float(bar['Close'])
                            pos['planned_exit_date'] = str(current.date())
                            if low_px <= stop:
                                close_trade(pos, current, min(open_px, stop), 'same_day_stop')
                            elif high_px >= target:
                                close_trade(pos, current, target, 'same_day_target')
                            else:
                                close_trade(pos, current, close_px, 'same_day_time')
                        else:
                            exit_idx = min(idx + horizon - 1, len(all_dates) - 1)
                            pos['planned_exit_date'] = all_dates[exit_idx]
                            pos['entry_date'] = current
                            positions.append(pos)
                            last_close[row['Code']] = float(bar['Close'])

        marked = sum(p['qty'] * last_close.get(p['Code'], p['entry_price']) for p in positions)
        equity_value = cash + marked
        equity_rows.append({
            'date': str(current.date()),
            'equity': float(equity_value),
            'cash': float(cash),
            'open_positions': len(positions),
            'month_blocked': month_blocked,
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
        equity_rows.append({'date': str(current.date()), 'equity': float(cash), 'cash': float(cash), 'open_positions': 0, 'month_blocked': month_blocked})

    trades = pd.DataFrame(trade_rows)
    equity = pd.DataFrame(equity_rows)
    return v2.calc_metrics(trades, equity, start, end), trades, equity


def validation_gate(m: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    mp = m.get('market_pnl', {})
    mt = m.get('market_trades', {})
    return {
        'return_20pct': m['return'] >= 0.20,
        'monthly_geom_1_5pct': m['monthly_geom'] >= 0.015,
        'pf_1_25': m['profit_factor'] >= 1.25,
        'mdd_18pct': m['max_drawdown'] >= -0.18,
        'positive_months_58pct': m['positive_month_ratio'] >= 0.58,
        'trades_24': m['trades'] >= 24,
        'stress_positive': stress['return'] > 0,
        'kospi_nonnegative': mp.get('KOSPI', -1.0) >= 0 and mt.get('KOSPI', 0) >= 4,
        'kosdaq_nonnegative': mp.get('KOSDAQ', -1.0) >= 0 and mt.get('KOSDAQ', 0) >= 4,
        'winner_concentration_40pct': m['largest_winner_share'] <= 0.40,
    }


def final_gate(m: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    mp = m.get('market_pnl', {})
    return {
        'return_positive': m['return'] > 0,
        'monthly_geom_5pct': m['monthly_geom'] >= 0.05,
        'pf_1_40': m['profit_factor'] >= 1.40,
        'mdd_15pct': m['max_drawdown'] >= -0.15,
        'positive_months_75pct': m['positive_month_ratio'] >= 0.75,
        'trades_32': m['trades'] >= 32,
        'stress_positive': stress['return'] > 0,
        'both_markets_positive': mp.get('KOSPI', -1.0) > 0 and mp.get('KOSDAQ', -1.0) > 0,
        'winner_concentration_30pct': m['largest_winner_share'] <= 0.30,
    }


def config_score(m: dict[str, Any], stress: dict[str, Any]) -> float:
    mp = m.get('market_pnl', {})
    market_bonus = 0.08 * sum(v > 0 for v in mp.values())
    return (
        1.1 * m['return']
        + 3.0 * m['monthly_geom']
        + 0.12 * min(m['profit_factor'], 3.0)
        + 0.12 * m['positive_month_ratio']
        + market_bonus
        + 0.45 * min(0.0, stress['return'])
        + 0.45 * min(0.0, m['max_drawdown'] + 0.12)
        - 0.10 * max(0.0, m['largest_winner_share'] - 0.30)
    )


def main() -> None:
    raw = core.load_data()
    print('raw', len(raw), raw['Date'].min(), raw['Date'].max(), raw['Code'].nunique(), flush=True)
    frame, market = core.build_features(raw)
    frame = v2.add_regime_code(frame)
    print('features', len(frame), frame['Date'].min(), frame['Date'].max(), flush=True)

    frame = predict_period(frame, TRAIN_END, CAL_START, CAL_END, '2024')
    cal_2024 = tail_calibration(frame[frame['Date'].between(CAL_START, CAL_END)], '2024')
    (OUT / 'tail_calibration_2024.json').write_text(json.dumps(cal_2024, ensure_ascii=False, indent=2), encoding='utf-8')

    frame = predict_period(frame, CAL_END, VALID_START, VALID_END, '2025')
    frame = apply_tail_calibration(frame, '2025', cal_2024)
    cand_2025 = prepare_candidates(frame, '2025', VALID_START, VALID_END)
    print('candidates 2025', len(cand_2025), cand_2025.groupby('Market').size().to_dict(), flush=True)

    rows: list[dict[str, Any]] = []
    best_cfg: Config | None = None
    best_pack: dict[str, Any] | None = None
    best_any: tuple[float, Config, dict[str, Any], pd.DataFrame, pd.DataFrame] | None = None
    all_cfg = configs()
    print('configs', len(all_cfg), flush=True)
    for i, cfg in enumerate(all_cfg):
        m, tr, eq = simulate(frame, cand_2025, cfg, '2025', VALID_START, VALID_END, 1.0)
        st, _, _ = simulate(frame, cand_2025, cfg, '2025', VALID_START, VALID_END, 1.5)
        gates = validation_gate(m, st)
        score = config_score(m, st)
        pack = {'config': dataclasses.asdict(cfg), 'config_id': cfg.id, 'metrics': m, 'stress': st, 'gates': gates, 'passed': all(gates.values()), 'score': score}
        rows.append(pack)
        if best_any is None or score > best_any[0]:
            best_any = (score, cfg, pack, tr, eq)
        if pack['passed'] and (best_pack is None or score > best_pack['score']):
            best_cfg = cfg
            best_pack = pack
        if (i + 1) % 24 == 0:
            print(i + 1, '/', len(all_cfg), flush=True)

    assert best_any is not None
    best_any[3].to_csv(OUT / 'best_2025_trades.csv', index=False)
    best_any[4].to_csv(OUT / 'best_2025_equity.csv', index=False)
    pd.DataFrame([
        {
            'config_id': r['config_id'],
            'passed': r['passed'],
            'score': r['score'],
            **{f'c_{k}': v for k, v in r['config'].items()},
            **{f'm_{k}': v for k, v in r['metrics'].items() if not isinstance(v, dict)},
            **{f's_{k}': v for k, v in r['stress'].items() if not isinstance(v, dict)},
        }
        for r in rows
    ]).to_csv(OUT / 'config_summary.csv', index=False)

    result: dict[str, Any] = {
        'version': 'korea-marcap-v3-fine-tail-dual-market',
        'data': {
            'rows': len(raw),
            'min_date': str(raw['Date'].min().date()),
            'max_date': str(raw['Date'].max().date()),
            'symbols': int(raw['Code'].nunique()),
            'source': 'FinanceData/marcap',
        },
        'periods': {
            'train': ['2020-01-01', str(TRAIN_END.date())],
            'calibration': [str(CAL_START.date()), str(CAL_END.date())],
            'strategy_validation': [str(VALID_START.date()), str(VALID_END.date())],
            'final': [str(FINAL_START.date()), str(FINAL_END.date())],
        },
        'candidate_counts_2025': {str(k): int(v) for k, v in cand_2025.groupby('Market').size().to_dict().items()},
        'config_trials': len(rows),
        'passing_configs': sum(r['passed'] for r in rows),
        'best_2025': best_any[2],
        'strategy_gate_passed': best_cfg is not None,
        'final_opened': best_cfg is not None,
    }

    if best_cfg is None or best_pack is None:
        result['accepted'] = False
        result['reason'] = 'No fine-tail dual-market configuration passed the 2025 gate; 2026 remained unopened.'
    else:
        result['selected_config'] = best_pack
        frame = predict_period(frame, VALID_END, FINAL_START, FINAL_END, '2026')
        cal_2025 = tail_calibration(frame[frame['Date'].between(VALID_START, VALID_END)], '2025')
        (OUT / 'tail_calibration_2025.json').write_text(json.dumps(cal_2025, ensure_ascii=False, indent=2), encoding='utf-8')
        frame = apply_tail_calibration(frame, '2026', cal_2025)
        cand_2026 = prepare_candidates(frame, '2026', FINAL_START, FINAL_END)
        fm, tr, eq = simulate(frame, cand_2026, best_cfg, '2026', FINAL_START, FINAL_END, 1.0)
        fs, _, _ = simulate(frame, cand_2026, best_cfg, '2026', FINAL_START, FINAL_END, 1.5)
        gates = final_gate(fm, fs)
        result['candidate_counts_2026'] = {str(k): int(v) for k, v in cand_2026.groupby('Market').size().to_dict().items()}
        result['final'] = {'metrics': fm, 'stress': fs, 'gates': gates, 'accepted': all(gates.values())}
        result['accepted'] = all(gates.values())
        tr.to_csv(OUT / 'final_trades.csv', index=False)
        eq.to_csv(OUT / 'final_equity.csv', index=False)
        cand_2026[['Date', 'Code', 'Name', 'Market', 'regime', 'expected_net_2026', 'best_rank_2026', 'best_h_2026', 'candidate_score']].to_csv(OUT / 'final_candidates.csv', index=False)

    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('===KOREA_MARCAP_V3_RESULT_BEGIN===')
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print('===KOREA_MARCAP_V3_RESULT_END===')


if __name__ == '__main__':
    main()
