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
V3_PATH = HERE.parent / 'korea_marcap_v3' / 'run_v3.py'
SPEC = importlib.util.spec_from_file_location('korea_marcap_v3_core_for_v4', V3_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError('unable to import v3 core')
v3 = importlib.util.module_from_spec(SPEC)
sys.modules['korea_marcap_v3_core_for_v4'] = v3
SPEC.loader.exec_module(v3)
v2 = v3.v2
core = v3.core

ROOT = Path('/tmp/korea_marcap_v4')
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


@dataclasses.dataclass(frozen=True)
class Config:
    kospi_rank: float
    kosdaq_rank: float
    min_expected: float
    stop_atr: float
    sleeve_weight: float
    transition_mult: float
    bear_mult: float
    panic_mult: float
    max_gap: float
    monthly_stop: float
    pause_days: int

    @property
    def id(self) -> str:
        return (
            f'kr{self.kospi_rank:.2f}-qr{self.kosdaq_rank:.2f}-e{self.min_expected:.3f}'
            f'-st{self.stop_atr:.1f}-w{self.sleeve_weight:.2f}-tm{self.transition_mult:.2f}'
            f'-bm{self.bear_mult:.2f}-pm{self.panic_mult:.2f}-g{self.max_gap:.2f}'
            f'-ms{abs(self.monthly_stop):.2f}-p{self.pause_days}'
        )


def configs() -> list[Config]:
    out: list[Config] = []
    for kr, qr, expected, stop, weight, tx, gap in itertools.product(
        [0.95, 0.97, 0.98],
        [0.90, 0.95, 0.97],
        [-0.005, 0.000, 0.003],
        [2.0, 3.0],
        [0.45, 0.50],
        [0.60, 0.80],
        [0.10],
    ):
        out.append(Config(kr, qr, expected, stop, weight, tx, 0.30, 0.15, gap, -0.12, 3))
    return out


def exposure_mult(regime: str, cfg: Config) -> float:
    return {
        'bull': 1.00,
        'neutral': 0.90,
        'transition': cfg.transition_mult,
        'bear': cfg.bear_mult,
        'panic': cfg.panic_mult,
    }.get(regime, 0.75)


def rank_threshold(market: str, regime: str, cfg: Config) -> float:
    base = cfg.kospi_rank if market == 'KOSPI' else cfg.kosdaq_rank
    if regime == 'transition':
        return max(base, 0.95)
    if regime == 'bear':
        return max(base, 0.98)
    if regime == 'panic':
        return max(base, 0.99)
    return base


def prepare_candidates(frame: pd.DataFrame, suffix: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    x = frame[frame['Date'].between(start, end)].copy()
    x = x[
        x['eligible']
        & x[f'best_rank_{suffix}'].notna()
        & x[f'expected_net_{suffix}'].notna()
        & x['amount_rank'].ge(0.35)
        & x['size_rank'].ge(0.15)
        & x['stocks_chg5'].abs().fillna(0).lt(0.12)
        & x['dist_ma20'].between(-0.30, 0.35)
    ].copy()
    x['direct_score'] = (
        0.55 * x[f'best_rank_{suffix}']
        + 0.12 * x['rel20_rank']
        + 0.10 * x['rel60_rank']
        + 0.09 * x['event_rank']
        + 0.07 * x['close_loc_rank']
        + 0.07 * x['amount_rank']
        + 2.0 * x[f'expected_net_{suffix}'].clip(-0.03, 0.08)
    )
    return x.sort_values(['Date', 'Market', 'direct_score', 'Amount'], ascending=[True, True, False, False])


def simulate(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    cfg: Config,
    suffix: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_mult: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw = frame[frame['Date'].between(start, end + pd.Timedelta(days=10))].sort_values(['Code', 'Date'])
    dates = [pd.Timestamp(d) for d in sorted(raw[raw['Date'].between(start, end)]['Date'].unique())]
    if not dates:
        return v2.empty_metrics(start, end), pd.DataFrame(), pd.DataFrame(), {}
    date_to_i = {d: i for i, d in enumerate(dates)}
    by_code = {c: g.set_index('Date').sort_index() for c, g in raw.groupby('Code')}
    by_signal = {pd.Timestamp(d): g for d, g in candidates.groupby('Date')}

    cash = INITIAL_CAPITAL
    positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    last_close: dict[str, float] = {}
    pause_until = {'KOSPI': -1, 'KOSDAQ': -1}
    loss_streak = {'KOSPI': 0, 'KOSDAQ': 0}
    month = None
    month_start = INITIAL_CAPITAL
    month_blocked = False
    diagnostics = {
        'candidate_rows': int(len(candidates)),
        'rank_pass': 0,
        'expected_pass': 0,
        'gap_pass': 0,
        'entries': 0,
        'entries_by_market': {'KOSPI': 0, 'KOSDAQ': 0},
    }

    def close_pos(pos: dict[str, Any], current: pd.Timestamp, px: float, reason: str) -> None:
        nonlocal cash
        proceeds = pos['qty'] * px * (1 - (core.SELL_FEE_TAX + core.SLIPPAGE) * cost_mult)
        cash += proceeds
        pnl = proceeds - pos['cash_out']
        market = str(pos['Market'])
        if pnl < 0:
            loss_streak[market] += 1
            if loss_streak[market] >= 2:
                pause_until[market] = date_to_i[current] + cfg.pause_days
        else:
            loss_streak[market] = 0
        trades.append({
            **{k: v for k, v in pos.items() if k != 'best_high'},
            'exit_date': str(current.date()),
            'exit_price': float(px),
            'reason': reason,
            'pnl': float(pnl),
            'return': float(pnl / pos['cash_out']),
        })

    for current in dates:
        idx = date_to_i[current]
        mkey = current.to_period('M')
        if month != mkey:
            month = mkey
            marked = sum(p['qty'] * last_close.get(p['Code'], p['entry_price']) for p in positions)
            month_start = cash + marked
            month_blocked = False

        remaining: list[dict[str, Any]] = []
        for pos in positions:
            hist = by_code.get(pos['Code'])
            if hist is None or current not in hist.index:
                remaining.append(pos)
                continue
            bar = hist.loc[current]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            open_px, high_px, low_px, close_px = map(float, [bar['Open'], bar['High'], bar['Low'], bar['Close']])
            last_close[pos['Code']] = close_px
            stop = pos['stop']
            if pos['best_high'] >= pos['entry_price'] + 1.3 * pos['risk']:
                stop = max(stop, pos['entry_price'] * (1 + core.BASE_COST * cost_mult))
            if low_px <= stop:
                close_pos(pos, current, min(open_px, stop), 'stop')
            elif current >= pos['planned_exit_date']:
                close_pos(pos, current, close_px, 'time')
            else:
                pos['best_high'] = max(pos['best_high'], high_px)
                remaining.append(pos)
        positions = remaining

        marked = sum(p['qty'] * last_close.get(p['Code'], p['entry_price']) for p in positions)
        equity = cash + marked
        if month_start > 0 and equity / month_start - 1.0 <= cfg.monthly_stop:
            month_blocked = True

        signal_date = dates[idx - 1] if idx > 0 else None
        if signal_date is not None and not month_blocked:
            cand = by_signal.get(signal_date)
            if cand is not None and not cand.empty:
                held_markets = {str(p['Market']) for p in positions}
                for market in ('KOSPI', 'KOSDAQ'):
                    if market in held_markets or idx <= pause_until[market]:
                        continue
                    cm = cand[cand['Market'].eq(market)]
                    if cm.empty:
                        continue
                    selected = None
                    for _, row in cm.iterrows():
                        regime = str(row['regime'])
                        threshold = rank_threshold(market, regime, cfg)
                        if float(row[f'best_rank_{suffix}']) < threshold:
                            continue
                        diagnostics['rank_pass'] += 1
                        min_exp = cfg.min_expected + (0.004 if regime in {'bear', 'panic'} else 0.0)
                        if float(row[f'expected_net_{suffix}']) < min_exp:
                            continue
                        diagnostics['expected_pass'] += 1
                        hist = by_code.get(row['Code'])
                        if hist is None or current not in hist.index:
                            continue
                        bar = hist.loc[current]
                        if isinstance(bar, pd.DataFrame):
                            bar = bar.iloc[-1]
                        gap = float(bar['Open']) / float(row['Close']) - 1.0
                        if gap > cfg.max_gap:
                            continue
                        diagnostics['gap_pass'] += 1
                        selected = row
                        break
                    if selected is None:
                        continue
                    row = selected
                    hist = by_code[row['Code']]
                    bar = hist.loc[current]
                    if isinstance(bar, pd.DataFrame):
                        bar = bar.iloc[-1]
                    regime = str(row['regime'])
                    entry = float(bar['Open']) * (1 + (core.BUY_FEE + core.SLIPPAGE) * cost_mult)
                    alloc = cfg.sleeve_weight * exposure_mult(regime, cfg)
                    qty = min(int((equity * alloc) // entry), int(cash // entry))
                    if qty < 1:
                        continue
                    atr = max(float(row['atr20']), entry * 0.005)
                    risk = max(cfg.stop_atr * atr, entry * 0.012)
                    horizon = int(row[f'best_h_{suffix}'])
                    cash_out = qty * entry
                    cash -= cash_out
                    pos = {
                        'signal_date': str(signal_date.date()),
                        'entry_date': str(current.date()),
                        'Code': row['Code'],
                        'Name': row['Name'],
                        'Market': market,
                        'regime': regime,
                        'model_rank': float(row[f'best_rank_{suffix}']),
                        'expected_net': float(row[f'expected_net_{suffix}']),
                        'direct_score': float(row['direct_score']),
                        'horizon': horizon,
                        'entry_price': entry,
                        'qty': qty,
                        'cash_out': cash_out,
                        'stop': entry - risk,
                        'risk': risk,
                        'best_high': float(bar['High']),
                        'config_id': cfg.id,
                    }
                    diagnostics['entries'] += 1
                    diagnostics['entries_by_market'][market] += 1
                    if horizon == 1:
                        open_px, low_px, close_px = map(float, [bar['Open'], bar['Low'], bar['Close']])
                        pos['planned_exit_date'] = str(current.date())
                        if low_px <= pos['stop']:
                            close_pos(pos, current, min(open_px, pos['stop']), 'same_day_stop')
                        else:
                            close_pos(pos, current, close_px, 'same_day_time')
                    else:
                        exit_idx = min(idx + horizon - 1, len(dates) - 1)
                        pos['planned_exit_date'] = dates[exit_idx]
                        pos['entry_date'] = current
                        positions.append(pos)
                        last_close[row['Code']] = float(bar['Close'])

        marked = sum(p['qty'] * last_close.get(p['Code'], p['entry_price']) for p in positions)
        equity = cash + marked
        equity_rows.append({'date': str(current.date()), 'equity': float(equity), 'cash': float(cash), 'open_positions': len(positions), 'month_blocked': month_blocked})

    if positions:
        current = dates[-1]
        for pos in positions:
            hist = by_code.get(pos['Code'])
            if hist is None or current not in hist.index:
                continue
            bar = hist.loc[current]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            close_pos(pos, current, float(bar['Close']), 'end')
        equity_rows.append({'date': str(current.date()), 'equity': float(cash), 'cash': float(cash), 'open_positions': 0, 'month_blocked': month_blocked})

    tr = pd.DataFrame(trades)
    eq = pd.DataFrame(equity_rows)
    return v2.calc_metrics(tr, eq, start, end), tr, eq, diagnostics


def gate(m: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    mp = m.get('market_pnl', {})
    mt = m.get('market_trades', {})
    return {
        'return_15pct': m['return'] >= 0.15,
        'monthly_geom_1pct': m['monthly_geom'] >= 0.01,
        'pf_1_20': m['profit_factor'] >= 1.20,
        'mdd_20pct': m['max_drawdown'] >= -0.20,
        'positive_months_55pct': m['positive_month_ratio'] >= 0.55,
        'trades_24': m['trades'] >= 24,
        'stress_positive': stress['return'] > 0,
        'kospi_nonnegative': mp.get('KOSPI', -1) >= 0 and mt.get('KOSPI', 0) >= 4,
        'kosdaq_nonnegative': mp.get('KOSDAQ', -1) >= 0 and mt.get('KOSDAQ', 0) >= 4,
        'winner_share_40pct': m['largest_winner_share'] <= 0.40,
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
        'both_markets_positive': mp.get('KOSPI', -1) > 0 and mp.get('KOSDAQ', -1) > 0,
        'winner_share_30pct': m['largest_winner_share'] <= 0.30,
    }


def score(m: dict[str, Any], stress: dict[str, Any]) -> float:
    mp = m.get('market_pnl', {})
    return (
        1.0 * m['return']
        + 2.5 * m['monthly_geom']
        + 0.12 * min(m['profit_factor'], 3.0)
        + 0.10 * m['positive_month_ratio']
        + 0.06 * sum(v > 0 for v in mp.values())
        + 0.45 * min(0.0, stress['return'])
        + 0.40 * min(0.0, m['max_drawdown'] + 0.12)
    )


def main() -> None:
    raw = core.load_data()
    print('raw', len(raw), raw['Date'].min(), raw['Date'].max(), raw['Code'].nunique(), flush=True)
    frame, market = core.build_features(raw)
    frame = v2.add_regime_code(frame)
    print('features', len(frame), frame['Date'].min(), frame['Date'].max(), flush=True)

    frame = v3.predict_period(frame, TRAIN_END, CAL_START, CAL_END, '2024')
    cal_2024 = v3.tail_calibration(frame[frame['Date'].between(CAL_START, CAL_END)], '2024')
    frame = v3.predict_period(frame, CAL_END, VALID_START, VALID_END, '2025')
    frame = v3.apply_tail_calibration(frame, '2025', cal_2024)
    cand_2025 = prepare_candidates(frame, '2025', VALID_START, VALID_END)
    print('candidate rows', len(cand_2025), cand_2025.groupby('Market').size().to_dict(), flush=True)

    rows: list[dict[str, Any]] = []
    best_pass: tuple[float, Config, dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]] | None = None
    best_any: tuple[float, Config, dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]] | None = None
    all_cfg = configs()
    print('configs', len(all_cfg), flush=True)
    for i, cfg in enumerate(all_cfg):
        m, tr, eq, diag = simulate(frame, cand_2025, cfg, '2025', VALID_START, VALID_END, 1.0)
        st, _, _, _ = simulate(frame, cand_2025, cfg, '2025', VALID_START, VALID_END, 1.5)
        gates = gate(m, st)
        sc = score(m, st)
        pack = {'config': dataclasses.asdict(cfg), 'config_id': cfg.id, 'metrics': m, 'stress': st, 'diagnostics': diag, 'gates': gates, 'passed': all(gates.values()), 'score': sc}
        rows.append(pack)
        tup = (sc, cfg, pack, tr, eq, diag)
        if best_any is None or sc > best_any[0]:
            best_any = tup
        if pack['passed'] and (best_pass is None or sc > best_pass[0]):
            best_pass = tup
        if (i + 1) % 36 == 0:
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
            **{f'd_{k}': v for k, v in r['diagnostics'].items() if not isinstance(v, dict)},
        }
        for r in rows
    ]).to_csv(OUT / 'config_summary.csv', index=False)

    result: dict[str, Any] = {
        'version': 'korea-marcap-v4-direct-tail-sleeves',
        'data': {'rows': len(raw), 'min_date': str(raw['Date'].min().date()), 'max_date': str(raw['Date'].max().date()), 'symbols': int(raw['Code'].nunique()), 'source': 'FinanceData/marcap'},
        'periods': {'train': ['2020-01-01', str(TRAIN_END.date())], 'calibration': [str(CAL_START.date()), str(CAL_END.date())], 'strategy_validation': [str(VALID_START.date()), str(VALID_END.date())], 'final': [str(FINAL_START.date()), str(FINAL_END.date())]},
        'config_trials': len(rows),
        'passing_configs': sum(r['passed'] for r in rows),
        'best_2025': best_any[2],
        'strategy_gate_passed': best_pass is not None,
        'final_opened': best_pass is not None,
    }

    if best_pass is None:
        result['accepted'] = False
        result['reason'] = 'No direct-tail sleeve configuration passed 2025; 2026 remained unopened.'
    else:
        _, cfg, pack, _, _, _ = best_pass
        result['selected_config'] = pack
        frame = v3.predict_period(frame, VALID_END, FINAL_START, FINAL_END, '2026')
        cal_2025 = v3.tail_calibration(frame[frame['Date'].between(VALID_START, VALID_END)], '2025')
        frame = v3.apply_tail_calibration(frame, '2026', cal_2025)
        cand_2026 = prepare_candidates(frame, '2026', FINAL_START, FINAL_END)
        fm, tr, eq, diag = simulate(frame, cand_2026, cfg, '2026', FINAL_START, FINAL_END, 1.0)
        fs, _, _, _ = simulate(frame, cand_2026, cfg, '2026', FINAL_START, FINAL_END, 1.5)
        gates = final_gate(fm, fs)
        result['final'] = {'metrics': fm, 'stress': fs, 'diagnostics': diag, 'gates': gates, 'accepted': all(gates.values())}
        result['accepted'] = all(gates.values())
        tr.to_csv(OUT / 'final_trades.csv', index=False)
        eq.to_csv(OUT / 'final_equity.csv', index=False)

    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('===KOREA_MARCAP_V4_RESULT_BEGIN===')
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print('===KOREA_MARCAP_V4_RESULT_END===')


if __name__ == '__main__':
    main()
