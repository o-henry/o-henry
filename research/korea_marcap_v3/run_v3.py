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
    raise RuntimeError('unable to import v2')
v2 = importlib.util.module_from_spec(SPEC)
sys.modules['korea_marcap_v2_core_for_v3'] = v2
SPEC.loader.exec_module(v2)

OUT = Path('/tmp/korea_marcap_v2/outputs')

_ORIG_SIMULATE = v2.simulate


@dataclasses.dataclass(frozen=True)
class V3Config:
    kospi_rank: float
    kospi_expected: float
    kosdaq_rank: float
    kosdaq_expected: float
    stop_atr: float
    kosdaq_transition: bool
    target_r: float = 3.0
    bull_exp: float = 0.80
    neutral_exp: float = 0.45
    transition_exp: float = 0.30
    bear_exp: float = 0.15
    panic_exp: float = 0.0
    monthly_stop: float = -0.06
    loss_pause: int = 4

    @property
    def id(self) -> str:
        return (
            f'kr{self.kospi_rank:.2f}-ke{self.kospi_expected:.4f}'
            f'-qr{self.kosdaq_rank:.2f}-qe{self.kosdaq_expected:.3f}'
            f'-st{self.stop_atr:.2f}-qt{int(self.kosdaq_transition)}'
        )


def configs() -> list[V3Config]:
    out: list[V3Config] = []
    for kr, ke, qr, qe, stop, qt in itertools.product(
        [0.94, 0.97],
        [0.0025, 0.0045],
        [0.97, 0.99],
        [0.012, 0.018],
        [1.75, 2.25],
        [False, True],
    ):
        out.append(V3Config(kr, ke, qr, qe, stop, qt))
    return out


def market_normalized_score(frame: pd.DataFrame, suffix: str) -> pd.Series:
    expected = frame[f'expected_net_{suffix}'].fillna(-9.0)
    horizon = frame[f'best_h_{suffix}'].fillna(1.0).astype(float)
    risk = (frame['vol20'].clip(0.007, 0.20) * np.sqrt(horizon)).clip(0.01, 0.30)
    edge_ir = expected / risk
    edge_rank = edge_ir.groupby([frame['Date'], frame['Market']]).rank(pct=True)
    model_rank = frame[f'best_rank_{suffix}'].fillna(0.0)
    r = frame['regime'].astype(str)
    trend = 0.35 * frame['rel20_rank'] + 0.25 * frame['ret60_rank'] + 0.20 * frame['event_rank'] + 0.20 * frame['close_loc_rank']
    pullback = 0.35 * frame['rel60_rank'] + 0.25 * (1.0 - frame.groupby(['Date', 'Market'])['ret3'].rank(pct=True)) + 0.20 * frame['close_loc_rank'] + 0.20 * frame['event_rank']
    defensive = 0.40 * frame['rel20_rank'] + 0.25 * (1.0 - frame['beta20_rank']) + 0.20 * frame['ret5_rank'] + 0.15 * frame['event_rank']
    tech = trend.copy()
    tech = np.where(r.eq('transition'), pullback, tech)
    tech = np.where(r.isin(['bear', 'panic']), defensive, tech)
    return 0.50 * model_rank + 0.30 * edge_rank + 0.20 * pd.Series(tech, index=frame.index)


def simulate_v3(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    cfg: V3Config,
    suffix: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_mult: float = 1.0,
):
    rank_col = f'best_rank_{suffix}'
    exp_col = f'expected_net_{suffix}'
    r = candidates['regime'].astype(str)
    kospi = (
        candidates['Market'].eq('KOSPI')
        & candidates[rank_col].ge(cfg.kospi_rank)
        & candidates[exp_col].ge(cfg.kospi_expected)
        & ~r.eq('panic')
    )
    kosdaq_core = (
        candidates['Market'].eq('KOSDAQ')
        & candidates[rank_col].ge(cfg.kosdaq_rank)
        & candidates[exp_col].ge(cfg.kosdaq_expected)
        & r.eq('bull')
    )
    kosdaq_transition = (
        candidates['Market'].eq('KOSDAQ')
        & candidates[rank_col].ge(max(cfg.kosdaq_rank, 0.985))
        & candidates[exp_col].ge(cfg.kosdaq_expected + 0.006)
        & r.eq('transition')
    )
    filtered = candidates[kospi | kosdaq_core | (kosdaq_transition if cfg.kosdaq_transition else False)].copy()
    filtered = filtered.sort_values(['Date', 'candidate_score', exp_col, 'Amount'], ascending=[True, False, False, False])
    base = v2.Config(
        min_rank=0.0,
        min_expected=-9.0,
        stop_atr=cfg.stop_atr,
        target_r=cfg.target_r,
        bull_exp=cfg.bull_exp,
        neutral_exp=cfg.neutral_exp,
        transition_exp=cfg.transition_exp,
        bear_exp=cfg.bear_exp,
        panic_exp=cfg.panic_exp,
        monthly_stop=cfg.monthly_stop,
        loss_pause=cfg.loss_pause,
    )
    metrics, trades, equity = _ORIG_SIMULATE(frame, filtered, base, suffix, start, end, cost_mult)
    if not trades.empty:
        trades = trades.copy()
        trades['v3_config_id'] = cfg.id
    return metrics, trades, equity


def v3_score(metrics: dict[str, Any], stress: dict[str, Any]) -> float:
    markets = metrics.get('market_pnl', {})
    positive_markets = sum(1 for x in markets.values() if x >= 0)
    return (
        2.2 * metrics['monthly_geom'] + 0.9 * metrics['return']
        + 0.13 * min(metrics['profit_factor'], 3.0)
        + 0.15 * metrics['positive_month_ratio']
        + 0.10 * positive_markets
        + 0.35 * min(0.0, metrics['max_drawdown'] + 0.10)
        + 0.50 * min(0.0, stress['return'])
    )


def v3_gate(metrics: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    pnl = metrics.get('market_pnl', {})
    n = metrics.get('market_trades', {})
    return {
        'return_5pct': metrics['return'] >= 0.05,
        'monthly_geom_0_4pct': metrics['monthly_geom'] >= 0.004,
        'profit_factor_1_15': metrics['profit_factor'] >= 1.15,
        'max_drawdown_12pct': metrics['max_drawdown'] >= -0.12,
        'positive_months_50pct': metrics['positive_month_ratio'] >= 0.50,
        'trades_25': metrics['trades'] >= 25,
        'cost_stress_positive': stress['return'] > 0,
        'kospi_nonnegative': pnl.get('KOSPI', -1) >= 0,
        'kosdaq_nonnegative': pnl.get('KOSDAQ', -1) >= 0,
        'kospi_trades_5': n.get('KOSPI', 0) >= 5,
        'kosdaq_trades_5': n.get('KOSDAQ', 0) >= 5,
        'largest_winner_below_35pct': metrics['largest_winner_share'] <= 0.35,
    }


def final_gate(metrics: dict[str, Any], stress: dict[str, Any]) -> dict[str, bool]:
    pnl = metrics.get('market_pnl', {})
    n = metrics.get('market_trades', {})
    return {
        'return_positive': metrics['return'] > 0,
        'monthly_geom_5pct': metrics['monthly_geom'] >= 0.05,
        'profit_factor_1_40': metrics['profit_factor'] >= 1.40,
        'max_drawdown_15pct': metrics['max_drawdown'] >= -0.15,
        'positive_months_75pct': metrics['positive_month_ratio'] >= 0.75,
        'trades_35': metrics['trades'] >= 35,
        'cost_stress_positive': stress['return'] > 0,
        'kospi_positive': pnl.get('KOSPI', -1) > 0,
        'kosdaq_positive': pnl.get('KOSDAQ', -1) > 0,
        'kospi_trades_5': n.get('KOSPI', 0) >= 5,
        'kosdaq_trades_5': n.get('KOSDAQ', 0) >= 5,
        'largest_winner_below_30pct': metrics['largest_winner_share'] <= 0.30,
    }


v2.Config = V3Config
v2.configs = configs
v2.candidate_score = market_normalized_score
v2.simulate = simulate_v3
v2.strategy_score = v3_score
v2.strategy_gate = v3_gate
v2.final_gate = final_gate

if __name__ == '__main__':
    v2.main()
    result_path = OUT / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        result['version'] = 'korea-marcap-v3-market-normalized'
        result['design_changes'] = [
            'market-normalized candidate score',
            'lower-volatility KOSPI-specific expected-return threshold',
            'KOSDAQ restricted to bull and optional extreme transition signals',
            'panic entries disabled',
            'four-session market circuit breaker after two losses',
        ]
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('===KOREA_MARCAP_V3_RESULT_BEGIN===')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print('===KOREA_MARCAP_V3_RESULT_END===')
