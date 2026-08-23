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

ROOT = Path('/tmp/kospi_v3')
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location('korea_core_kospi_v3', ROOT / 'run_v1.py')
if spec is None or spec.loader is None:
    raise RuntimeError('unable to import core')
core = importlib.util.module_from_spec(spec)
sys.modules['korea_core_kospi_v3'] = core
sys.modules['korea_regime_core_v1'] = core
spec.loader.exec_module(core)

v2spec = importlib.util.spec_from_file_location('korea_v2_kospi_v3', ROOT / 'v2.py')
if v2spec is None or v2spec.loader is None:
    raise RuntimeError('unable to import v2')
v2 = importlib.util.module_from_spec(v2spec)
sys.modules['korea_v2_kospi_v3'] = v2
v2spec.loader.exec_module(v2)

PRE_START = pd.Timestamp('2025-10-20')
PRE_END = pd.Timestamp('2026-04-30')
FINAL_START = pd.Timestamp('2026-05-01')
FINAL_END = pd.Timestamp('2026-08-04')


@dataclass(frozen=True)
class OverlayConfig:
    stock: v2.V2Config
    bull_index_weight: float
    transition_index_weight: float

    @property
    def config_id(self) -> str:
        return f'{self.stock.config_id}-ib{self.bull_index_weight:.2f}-it{self.transition_index_weight:.2f}'


def configs() -> list[OverlayConfig]:
    out: list[OverlayConfig] = []
    for n, oh, bs, ts, tx, ib, it in itertools.product(
        [1, 2],
        [1.5, 2.5],
        ['continuous', 'pullback'],
        ['defensive', 'reversal'],
        [0.45, 0.65],
        [0.40, 0.60, 0.80],
        [0.00, 0.20],
    ):
        stock = v2.V2Config(
            bull_hold=2,
            transition_hold=1,
            bear_hold=1,
            n_select=n,
            bull_exposure=1.0,
            transition_exposure=tx,
            bear_exposure=0.0,
            overheat_sigma=oh,
            include_panic_reversal=False,
            bull_style=bs,
            transition_style=ts,
            panic_style='extreme',
        )
        out.append(OverlayConfig(stock, ib, it))
    return out


def overlay_equity(base_eq: pd.DataFrame, cfg: OverlayConfig, stress: float = 1.0) -> pd.DataFrame:
    eq = base_eq.copy().sort_values('date').reset_index(drop=True)
    weights = []
    for regime in eq['regime'].astype(str):
        if regime == 'bull':
            weights.append(cfg.bull_index_weight)
        elif regime in {'transition', 'neutral'}:
            weights.append(cfg.transition_index_weight)
        else:
            weights.append(0.0)
    eq['index_weight'] = weights
    eq['combined_return'] = (
        (1.0 - eq['index_weight']) * eq['daily_return']
        + eq['index_weight'] * eq['benchmark_daily_return']
    )
    # Approximate index-sleeve trading cost only when the sleeve weight changes.
    turnover = eq['index_weight'].diff().abs().fillna(eq['index_weight'])
    eq['combined_return'] -= turnover * 0.0010 * stress
    eq['equity'] = core.INITIAL_CAPITAL * (1.0 + eq['combined_return']).cumprod()
    eq['daily_return'] = eq['combined_return']
    return eq


def metrics(eq: pd.DataFrame, base_m: dict[str, Any], benchmark: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    ret = float(eq['equity'].iloc[-1] / core.INITIAL_CAPITAL - 1.0)
    dd = eq['equity'] / eq['equity'].cummax() - 1.0
    eq2 = eq.copy()
    eq2['month'] = eq2['date'].dt.to_period('M')
    monthly = eq2.groupby('month')['daily_return'].apply(lambda x: float((1+x).prod()-1))
    monthly_geom = float((1+monthly).prod() ** (1/len(monthly)) - 1) if len(monthly) else 0.0
    beta = float(eq['daily_return'].cov(eq['benchmark_daily_return']) / eq['benchmark_daily_return'].var()) if eq['benchmark_daily_return'].var() > 0 else 0.0
    alpha = float((eq['daily_return'] - beta * eq['benchmark_daily_return']).sum())
    b = benchmark[(benchmark['market'] == 'KOSPI') & benchmark['date'].between(start, end)]
    bret = float(b['bench_close'].iloc[-1] / b['bench_close'].iloc[0] - 1.0) if len(b) >= 2 else 0.0
    regimes = {}
    for label, names in {
        'bull': {'bull'},
        'transition': {'transition', 'neutral'},
        'defensive': {'bear', 'panic'},
    }.items():
        sub = eq[eq['regime'].isin(names)]
        regimes[label] = {
            'days': int(len(sub)),
            'strategy_return': float((1+sub['daily_return']).prod()-1) if len(sub) else 0.0,
            'benchmark_return': float((1+sub['benchmark_daily_return']).prod()-1) if len(sub) else 0.0,
        }
    return {
        'return': ret,
        'final_equity': float(eq['equity'].iloc[-1]),
        'max_drawdown': float(dd.min()),
        'monthly_geom': monthly_geom,
        'monthly_median': float(monthly.median()) if len(monthly) else 0.0,
        'positive_month_ratio': float((monthly > 0).mean()) if len(monthly) else 0.0,
        'monthly_returns': {str(k): float(v) for k,v in monthly.items()},
        'benchmark_return': bret,
        'beta': beta,
        'alpha_sum': alpha,
        'stock_sleeve_profit_factor': base_m['profit_factor'],
        'stock_sleeve_trades': base_m['trades'],
        'regimes': regimes,
    }


def pre_gate(m: dict, stress: dict) -> dict[str, bool]:
    r = m['regimes']
    return {
        'return_positive': m['return'] > 0,
        'monthly_geom_5pct': m['monthly_geom'] >= 0.05,
        'positive_months_60pct': m['positive_month_ratio'] >= 0.60,
        'mdd_18pct': m['max_drawdown'] >= -0.18,
        'alpha_positive': m['alpha_sum'] > 0,
        'beta_below_0_85': m['beta'] <= 0.85,
        'transition_positive': r['transition']['strategy_return'] > 0,
        'defensive_nonnegative': r['defensive']['strategy_return'] >= 0,
        'stock_pf_1_05': m['stock_sleeve_profit_factor'] >= 1.05,
        'cost_stress': stress['return'] > 0,
    }


def final_gate(m: dict, stress: dict) -> dict[str, bool]:
    r = m['regimes']
    return {
        'return_positive': m['return'] > 0,
        'monthly_geom_5pct': m['monthly_geom'] >= 0.05,
        'positive_months_50pct': m['positive_month_ratio'] >= 0.50,
        'mdd_20pct': m['max_drawdown'] >= -0.20,
        'alpha_positive': m['alpha_sum'] > 0,
        'beta_below_0_90': m['beta'] <= 0.90,
        'transition_nonnegative': r['transition']['strategy_return'] >= 0,
        'defensive_nonnegative': r['defensive']['strategy_return'] >= 0,
        'cost_stress': stress['return'] > 0,
    }


def score(m: dict) -> float:
    return (
        0.35 * m['return']
        + 0.8 * m['monthly_geom']
        + 0.45 * m['alpha_sum']
        + 0.10 * m['positive_month_ratio']
        + 0.10 * max(0.0, m['regimes']['transition']['strategy_return'])
        + 0.10 * max(0.0, m['regimes']['defensive']['strategy_return'])
        + 0.50 * min(0.0, m['max_drawdown'] + 0.10)
    )


def run_one(cfg: OverlayConfig, features: pd.DataFrame, bench: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, stress_mult: float = 1.0):
    core.candidate_scores = v2.candidate_scores_v2
    base_m, trades, base_eq = core.simulate_market('KOSPI', cfg.stock, features, bench, start, end, stress=stress_mult)
    eq = overlay_equity(base_eq, cfg, stress=stress_mult)
    return metrics(eq, base_m, bench, start, end), trades, eq


def main() -> None:
    prices = core.load_prices()
    benchmarks = core.load_benchmarks(prices['date'].min(), prices['date'].max(), prices)
    features, bench = core.build_features(prices, benchmarks)
    rows = []
    all_cfg = configs()
    print('configs', len(all_cfg), 'rows', len(prices))
    for i, cfg in enumerate(all_cfg):
        m, _, _ = run_one(cfg, features, bench, PRE_START, PRE_END)
        st, _, _ = run_one(cfg, features, bench, PRE_START, PRE_END, 1.5)
        gates = pre_gate(m, st)
        rows.append({'config': cfg, 'metrics': m, 'stress': st, 'gates': gates, 'passed': all(gates.values()), 'score': score(m)})
        if (i+1) % 48 == 0:
            print(i+1, '/', len(all_cfg))
    passing = [x for x in rows if x['passed']]
    pool = passing if passing else rows
    best = max(pool, key=lambda x: x['score'])
    cfg = best['config']
    result: dict[str, Any] = {
        'version': 'kospi-regime-v3-explicit-beta-alpha',
        'pre_final_period': [str(PRE_START.date()), str(PRE_END.date())],
        'final_locked_period': [str(FINAL_START.date()), str(FINAL_END.date())],
        'selected_config': {
            'stock': asdict(cfg.stock),
            'bull_index_weight': cfg.bull_index_weight,
            'transition_index_weight': cfg.transition_index_weight,
        },
        'pre_final': {k:v for k,v in best.items() if k != 'config'},
        'passing_configs': len(passing),
        'research_gate_passed': bool(best['passed']),
        'final_locked_opened': bool(best['passed']),
    }
    if best['passed']:
        fm, tr, eq = run_one(cfg, features, bench, FINAL_START, FINAL_END)
        fs, _, _ = run_one(cfg, features, bench, FINAL_START, FINAL_END, 1.5)
        gates = final_gate(fm, fs)
        result['final'] = {'metrics': fm, 'stress': fs, 'gates': gates, 'passed': all(gates.values())}
        result['accepted'] = all(gates.values())
        tr.to_csv(OUT / 'kospi_v3_final_trades.csv', index=False)
        eq.to_csv(OUT / 'kospi_v3_final_equity.csv', index=False)
    else:
        result['accepted'] = False
    (OUT / 'kospi_v3_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('===KOSPI_V3_RESULT_BEGIN===')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('===KOSPI_V3_RESULT_END===')


if __name__ == '__main__':
    main()
