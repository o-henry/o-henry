from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path('/tmp/korea_regime_v2')
CORE_PATH = ROOT / 'run_v1.py'
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location('korea_regime_core_v1', CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError('unable to import v1 core')
core = importlib.util.module_from_spec(spec)
sys.modules['korea_regime_core_v1'] = core
spec.loader.exec_module(core)

PRE_START = pd.Timestamp('2025-10-20')
PRE_END = pd.Timestamp('2026-04-30')
FINAL_START = pd.Timestamp('2026-05-01')
FINAL_END = pd.Timestamp('2026-08-04')
STRESS = 1.5


@dataclass(frozen=True)
class V2Config:
    bull_hold: int
    transition_hold: int
    bear_hold: int
    n_select: int
    bull_exposure: float
    transition_exposure: float
    bear_exposure: float
    overheat_sigma: float
    include_panic_reversal: bool
    bull_style: str
    transition_style: str
    panic_style: str

    @property
    def config_id(self) -> str:
        return (
            f'v2-n{self.n_select}-tx{self.transition_exposure:.2f}'
            f'-ex{self.bear_exposure:.2f}-oh{self.overheat_sigma:.1f}'
            f'-pr{int(self.include_panic_reversal)}'
            f'-b{self.bull_style[0]}-t{self.transition_style[0]}-p{self.panic_style[0]}'
        )


def candidate_scores_v2(day: pd.DataFrame, cfg: V2Config) -> pd.DataFrame:
    if day.empty:
        return day
    market = str(day['market'].iloc[0])
    min_adv = 2_000_000_000 if market == 'KOSPI' else 1_000_000_000
    d = day.copy()
    eligible = (
        (d['history_n'] >= 60)
        & (d['close'] >= 2000)
        & (d['adv20'] >= min_adv)
        & (d['max_abs_ret20'] <= 0.35)
        & d['vol20'].between(0.008, 0.12)
        & d['ma20'].notna()
        & d['ma60'].notna()
    )
    d = d[eligible].copy()
    if d.empty:
        return d

    regime = str(d['regime'].iloc[0])
    overheat_ok = d['dist_ma20_sigma'] <= cfg.overheat_sigma

    if regime == 'bull':
        if cfg.bull_style == 'continuous':
            cond = (
                (d['close'] > d['ma20'])
                & (d['ma20'] >= d['ma60'] * 0.99)
                & (d['ret20'] > 0.025)
                & (d['rel20'] > 0.0)
                & (d['rank_rel20'] >= 0.72)
                & d['ret3'].between(-0.06, 0.08)
                & overheat_ok
            )
        else:
            cond = (
                (d['close'] > d['ma20'])
                & (d['ma20'] >= d['ma60'] * 0.99)
                & (d['ret20'] > 0.03)
                & (d['rel20'] > 0.0)
                & d['ret3'].between(-0.08, 0.035)
                & d['drawdown20'].between(-0.14, -0.003)
                & overheat_ok
            )
        d['score'] = (
            0.30 * d['rank_rel20']
            + 0.20 * d['rank_rel60']
            + 0.16 * d['rank_ret20']
            + 0.12 * d['rank_ret60']
            + 0.10 * d['rank_volume_ratio']
            + 0.08 * d['rank_adv20']
            + 0.04 * d['rank_low_vol20']
        )
    elif regime in {'transition', 'neutral'}:
        if cfg.transition_style == 'defensive':
            cond = (
                (d['ret60'] > 0.0)
                & (d['rel20'] > 0.0)
                & (d['rank_rel20'] >= 0.80)
                & (d['beta20'] <= 0.95)
                & (d['close'] >= d['ma20'] * 0.98)
                & (d['ret5'] > -0.06)
                & (d['volume_ratio'] >= 0.70)
                & overheat_ok
            )
        else:
            cond = (
                (d['ret60'] > 0.0)
                & (d['rel20'] > 0.0)
                & (d['rank_rel20'] >= 0.75)
                & d['ret3'].between(-0.12, -0.003)
                & (d['ret1'] > 0.0)
                & (d['close'] >= d['ma10'] * 0.975)
                & (d['volume_ratio'] >= 0.75)
                & overheat_ok
            )
        d['score'] = (
            0.32 * d['rank_rel20']
            + 0.18 * d['rank_rel60']
            + 0.14 * d['rank_ret60']
            + 0.13 * d['rank_volume_ratio']
            + 0.13 * d['rank_low_vol20']
            + 0.10 * (1.0 - d['beta20'].clip(0, 2) / 2.0)
        )
    else:
        if cfg.panic_style == 'extreme':
            rs_cond = (
                (d['ret20'] > 0.08)
                & (d['rel20'] > 0.12)
                & (d['rank_rel20'] >= 0.96)
                & (d['close'] > d['ma20'])
                & (d['beta20'] <= 0.75)
                & (d['ret5'] > -0.01)
                & overheat_ok
            )
        else:
            rs_cond = (
                (d['ret20'] > 0.04)
                & (d['rel20'] > 0.08)
                & (d['rank_rel20'] >= 0.92)
                & (d['close'] > d['ma20'])
                & (d['ret5'] > -0.025)
                & ((d['beta20'] <= 0.85) | (d['volume_ratio'] >= 1.8))
                & overheat_ok
            )
        reversal_cond = (
            cfg.include_panic_reversal
            & (d['ret5'] <= -0.15)
            & (d['ret1'] >= 0.05)
            & (d['volume_ratio'] >= 2.0)
            & (d['rel1'] > 0.03)
            & (d['rank_adv20'] >= 0.85)
        )
        cond = rs_cond | reversal_cond
        d['score'] = (
            0.36 * d['rank_rel20']
            + 0.18 * d['rank_rel5']
            + 0.15 * d['rank_ret20']
            + 0.12 * d['rank_volume_ratio']
            + 0.10 * d['rank_low_vol20']
            + 0.09 * (1.0 - d['beta20'].clip(0, 2) / 2.0)
        )
        d.loc[reversal_cond, 'score'] += 0.08

    return d[cond & d['score'].notna()].sort_values(
        ['score', 'adv20'], ascending=[False, False]
    )


def build_configs() -> list[V2Config]:
    out: list[V2Config] = []
    for vals in itertools.product(
        [1, 2],
        [0.45, 0.65],
        [0.10, 0.20],
        [1.5, 2.5],
        [False, True],
        ['continuous', 'pullback'],
        ['defensive', 'reversal'],
        ['strict', 'extreme'],
    ):
        n, tx, ex, oh, pr, bs, ts, ps = vals
        out.append(V2Config(2, 1, 1, n, 1.0, tx, ex, oh, pr, bs, ts, ps))
    return out


def regime_trade_stats(trades: pd.DataFrame) -> dict[str, dict[str, float]]:
    groups = {
        'bull': {'bull'},
        'transition': {'transition', 'neutral'},
        'defensive': {'bear', 'panic'},
    }
    out: dict[str, dict[str, float]] = {}
    for label, regimes in groups.items():
        sub = trades[trades['regime'].isin(regimes)] if not trades.empty else pd.DataFrame()
        if sub.empty:
            out[label] = {'trades': 0, 'pnl': 0.0, 'pf': 0.0, 'win_rate': 0.0}
            continue
        pnl = sub['net_pnl'].astype(float)
        gp = float(pnl[pnl > 0].sum())
        gl = float(-pnl[pnl < 0].sum())
        pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
        out[label] = {
            'trades': int(len(sub)),
            'pnl': float(pnl.sum()),
            'pf': pf,
            'win_rate': float((pnl > 0).mean()),
        }
    return out


def pre_gate(m: dict[str, Any], stress: dict[str, Any], rs: dict[str, Any]) -> dict[str, bool]:
    return {
        'return_positive': m['return'] > 0,
        'monthly_geom_4pct': m['monthly_geom'] >= 0.04,
        'pf_1_25': m['profit_factor'] >= 1.25,
        'mdd_18pct': m['max_drawdown'] >= -0.18,
        'positive_months_60pct': m['positive_month_ratio'] >= 0.60,
        'trades_20': m['trades'] >= 20,
        'cost_stress': stress['return'] > 0,
        'alpha_positive': m['alpha_sum'] > 0,
        'beta_below_0_9': m['beta'] <= 0.90,
        'bull_positive': rs['bull']['trades'] < 5 or (rs['bull']['pnl'] > 0 and rs['bull']['pf'] >= 1.10),
        'transition_positive': rs['transition']['trades'] < 5 or (rs['transition']['pnl'] > 0 and rs['transition']['pf'] >= 1.10),
        'defensive_nonnegative': rs['defensive']['trades'] < 5 or (rs['defensive']['pnl'] >= 0 and rs['defensive']['pf'] >= 1.00),
    }


def final_gate(m: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    return {
        'return_positive': m['return'] > 0,
        'monthly_geom_5pct': m['monthly_geom'] >= 0.05,
        'pf_1_20': m['profit_factor'] >= 1.20,
        'mdd_20pct': m['max_drawdown'] >= -0.20,
        'positive_months_50pct': m['positive_month_ratio'] >= 0.50,
        'cost_stress': stress['return'] > 0,
        'alpha_positive': m['alpha_sum'] > 0,
        'beta_below_1': m['beta'] <= 1.00,
    }


def score(m: dict[str, Any], rs: dict[str, Any]) -> float:
    regime_bonus = 0.0
    for k in ['bull', 'transition', 'defensive']:
        if rs[k]['pnl'] > 0:
            regime_bonus += min(rs[k]['pnl'] / core.INITIAL_CAPITAL, 0.20)
    return (
        0.30 * m['return']
        + 1.00 * m['monthly_geom']
        + 0.45 * m['alpha_sum']
        + 0.25 * regime_bonus
        + 0.03 * min(m['profit_factor'], 3.0)
        + 0.10 * m['positive_month_ratio']
        + 0.50 * min(0.0, m['max_drawdown'] + 0.10)
    )


def main() -> None:
    core.candidate_scores = candidate_scores_v2
    core.OUT = OUT
    print('loading data')
    prices = core.load_prices()
    benchmarks = core.load_benchmarks(prices['date'].min(), prices['date'].max(), prices)
    features, bench = core.build_features(prices, benchmarks)
    configs = build_configs()
    print('configs', len(configs), 'rows', len(prices))

    result: dict[str, Any] = {
        'version': 'korea-regime-v2',
        'data_range': [str(prices['date'].min().date()), str(prices['date'].max().date())],
        'pre_final_period': [str(PRE_START.date()), str(PRE_END.date())],
        'final_locked_period': [str(FINAL_START.date()), str(FINAL_END.date())],
        'markets': {},
    }
    chosen: dict[str, V2Config] = {}

    for market in ['KOSPI', 'KOSDAQ']:
        rows = []
        for i, cfg in enumerate(configs):
            m, tr, _ = core.simulate_market(market, cfg, features, bench, PRE_START, PRE_END)
            st, _, _ = core.simulate_market(market, cfg, features, bench, PRE_START, PRE_END, stress=STRESS)
            rs = regime_trade_stats(tr)
            gates = pre_gate(m, st, rs)
            rows.append({
                'config': cfg,
                'metrics': m,
                'stress': st,
                'regime_trade_stats': rs,
                'gates': gates,
                'passed': all(gates.values()),
                'score': score(m, rs),
            })
            if (i + 1) % 64 == 0:
                print(market, i + 1, '/', len(configs))
        passing = [x for x in rows if x['passed']]
        pool = passing if passing else rows
        best = max(pool, key=lambda x: x['score'])
        cfg = best['config']
        chosen[market] = cfg
        result['markets'][market] = {
            'selected_config': asdict(cfg),
            'selected_config_id': cfg.config_id,
            'pre_final': {k: v for k, v in best.items() if k != 'config'},
            'pre_final_passed': bool(best['passed']),
            'passing_configs': len(passing),
        }

    research_gate = all(result['markets'][m]['pre_final_passed'] for m in ['KOSPI', 'KOSDAQ'])
    result['research_gate_passed'] = research_gate
    result['final_locked_opened'] = research_gate

    if research_gate:
        for market in ['KOSPI', 'KOSDAQ']:
            cfg = chosen[market]
            m, tr, eq = core.simulate_market(market, cfg, features, bench, FINAL_START, FINAL_END)
            st, _, _ = core.simulate_market(market, cfg, features, bench, FINAL_START, FINAL_END, stress=STRESS)
            gates = final_gate(m, st)
            result['markets'][market]['final'] = {
                'metrics': m,
                'stress': st,
                'gates': gates,
                'passed': all(gates.values()),
                'regime_trade_stats': regime_trade_stats(tr),
            }
            tr.to_csv(OUT / f'{market.lower()}_v2_final_trades.csv', index=False)
            eq.to_csv(OUT / f'{market.lower()}_v2_final_equity.csv', index=False)
        result['accepted'] = all(result['markets'][m]['final']['passed'] for m in ['KOSPI', 'KOSDAQ'])
    else:
        result['accepted'] = False

    (OUT / 'v2_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('===KOREA_REGIME_V2_RESULT_BEGIN===')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('===KOREA_REGIME_V2_RESULT_END===')


if __name__ == '__main__':
    main()
