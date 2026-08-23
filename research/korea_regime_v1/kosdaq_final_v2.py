from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

ROOT = Path('/tmp/kosdaq_final_v2')
CORE_PATH = ROOT / 'run_v1.py'
V2_PATH = ROOT / 'v2.py'
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location('korea_regime_core_final', CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError('unable to import core')
core = importlib.util.module_from_spec(spec)
sys.modules['korea_regime_core_final'] = core
sys.modules['korea_regime_core_v1'] = core
spec.loader.exec_module(core)

v2spec = importlib.util.spec_from_file_location('korea_regime_v2_rules', V2_PATH)
if v2spec is None or v2spec.loader is None:
    raise RuntimeError('unable to import v2 rules')
v2 = importlib.util.module_from_spec(v2spec)
sys.modules['korea_regime_v2_rules'] = v2
v2spec.loader.exec_module(v2)

FINAL_START = pd.Timestamp('2026-05-01')
FINAL_END = pd.Timestamp('2026-08-04')

# Frozen before final data are opened. This is the exact KOSDAQ pre-final winner.
CFG = v2.V2Config(
    bull_hold=2,
    transition_hold=1,
    bear_hold=1,
    n_select=2,
    bull_exposure=1.0,
    transition_exposure=0.65,
    bear_exposure=0.20,
    overheat_sigma=1.5,
    include_panic_reversal=False,
    bull_style='pullback',
    transition_style='reversal',
    panic_style='strict',
)


def gate(metrics: dict, stress: dict, regime_stats: dict) -> dict[str, bool]:
    defensive = regime_stats['defensive']
    return {
        'return_positive': metrics['return'] > 0,
        'monthly_geom_5pct': metrics['monthly_geom'] >= 0.05,
        'profit_factor_1_20': metrics['profit_factor'] >= 1.20,
        'max_drawdown_20pct': metrics['max_drawdown'] >= -0.20,
        'positive_months_50pct': metrics['positive_month_ratio'] >= 0.50,
        'cost_stress_positive': stress['return'] > 0,
        'alpha_positive': metrics['alpha_sum'] > 0,
        'beta_below_1': metrics['beta'] <= 1.0,
        'defensive_nonnegative': defensive['trades'] < 5 or defensive['pnl'] >= 0,
    }


def main() -> None:
    core.candidate_scores = v2.candidate_scores_v2
    prices = core.load_prices()
    benchmarks = core.load_benchmarks(prices['date'].min(), prices['date'].max(), prices)
    features, bench = core.build_features(prices, benchmarks)

    metrics, trades, equity = core.simulate_market(
        'KOSDAQ', CFG, features, bench, FINAL_START, FINAL_END
    )
    stress, _, _ = core.simulate_market(
        'KOSDAQ', CFG, features, bench, FINAL_START, FINAL_END, stress=1.5
    )
    regime_stats = v2.regime_trade_stats(trades)

    # Predeclared sensitivity checks; never used to choose the configuration.
    cfg_low = replace(
        CFG,
        transition_exposure=CFG.transition_exposure * 0.85,
        bear_exposure=CFG.bear_exposure * 0.85,
        overheat_sigma=CFG.overheat_sigma * 0.85,
    )
    cfg_high = replace(
        CFG,
        transition_exposure=min(1.0, CFG.transition_exposure * 1.15),
        bear_exposure=min(0.5, CFG.bear_exposure * 1.15),
        overheat_sigma=CFG.overheat_sigma * 1.15,
    )
    low_m, _, _ = core.simulate_market('KOSDAQ', cfg_low, features, bench, FINAL_START, FINAL_END)
    high_m, _, _ = core.simulate_market('KOSDAQ', cfg_high, features, bench, FINAL_START, FINAL_END)

    gates = gate(metrics, stress, regime_stats)
    gates['sensitivity_minus_15pct_positive'] = low_m['return'] > 0
    gates['sensitivity_plus_15pct_positive'] = high_m['return'] > 0

    result = {
        'version': 'kosdaq-regime-v2-locked-final',
        'period': [str(FINAL_START.date()), str(FINAL_END.date())],
        'initial_capital_krw': core.INITIAL_CAPITAL,
        'config': asdict(CFG),
        'metrics': metrics,
        'cost_stress_1_5x': stress,
        'regime_trade_stats': regime_stats,
        'sensitivity': {
            'minus_15pct': low_m,
            'plus_15pct': high_m,
        },
        'gates': gates,
        'accepted': all(gates.values()),
        'integrity': {
            'configuration_selected_from': '2025-10-20 through 2026-04-30 only',
            'final_used_for_selection': False,
            'final_open_count': 1,
        },
    }
    (OUT / 'kosdaq_final_result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    trades.to_csv(OUT / 'kosdaq_final_trades.csv', index=False)
    equity.to_csv(OUT / 'kosdaq_final_equity.csv', index=False)
    print('===KOSDAQ_LOCKED_FINAL_BEGIN===')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('===KOSDAQ_LOCKED_FINAL_END===')


if __name__ == '__main__':
    main()
