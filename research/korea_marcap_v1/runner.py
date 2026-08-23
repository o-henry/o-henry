from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('korea_marcap_v1_core', HERE / 'run.py')
if SPEC is None or SPEC.loader is None:
    raise RuntimeError('unable to import run.py')
core = importlib.util.module_from_spec(SPEC)
sys.modules['korea_marcap_v1_core'] = core
SPEC.loader.exec_module(core)


def bounded_configs():
    out = []
    for hold, topn, score, stop, tx in itertools.product(
        [2, 3],
        [1, 2],
        [0.82, 0.88],
        [1.5, 2.0],
        [0.50, 0.70],
    ):
        out.append(core.Config(hold, topn, score, stop, 4.0, 1.0, 0.80, tx, 0.15, 0.10, 0.40))
    return out


core.configs = bounded_configs
_orig_load_data = core.load_data
_orig_to_parquet = pd.DataFrame.to_parquet
_orig_apply = pd.DataFrame.apply


def liquid_point_in_time_data():
    df = _orig_load_data()
    df['mcap_rank_day'] = df.groupby(['Date', 'Market'])['Marcap'].rank(method='first', ascending=False)
    df['amount_rank_day'] = df.groupby(['Date', 'Market'])['Amount'].rank(method='first', ascending=False)
    df = df[(df['mcap_rank_day'] <= 550) | (df['amount_rank_day'] <= 250)].copy()
    return df.drop(columns=['mcap_rank_day', 'amount_rank_day'])


def sampled_to_parquet(self, path, *args, **kwargs):
    if str(path).endswith('feature_sample.parquet') and len(self) > 25_000:
        return _orig_to_parquet(self.sample(25_000, random_state=20260823), path, *args, **kwargs)
    return _orig_to_parquet(self, path, *args, **kwargs)


def fast_apply(self, func, *args, **kwargs):
    axis = kwargs.get('axis', args[0] if args else 0)
    if func is core.regime_filter and axis in (1, 'columns'):
        r = self['regime'].astype(str)
        bull = (self['dist_ma60'] > -0.03) & (self['rel20_rank'] >= 0.55) & (self['dd20'] >= -0.18)
        transition = (self['ret60'] > 0) & (self['rel20_rank'] >= 0.70) & self['dd20'].between(-0.18, -0.01) & (self['close_loc'] >= 0.45)
        neutral = (self['rel20_rank'] >= 0.65) & (self['dist_ma20'] > -0.05)
        bear = (self['ret20'] > 0) & (self['rel20_rank'] >= 0.90) & (self['beta20'] <= 0.9) & (self['dist_ma20'] > -0.03)
        reversal = (self['ret3'] <= -0.08) & (self['close_loc'] >= 0.75) & (self['volume_ratio20'] >= 1.4)
        leader = (self['ret5'] > 0) & (self['rel5_rank'] >= 0.95) & (self['close_loc'] >= 0.60)
        panic = reversal | leader
        return (
            (r.eq('bull') & bull)
            | (r.eq('transition') & transition)
            | (r.eq('neutral') & neutral)
            | (r.eq('bear') & bear)
            | (r.eq('panic') & panic)
        )
    return _orig_apply(self, func, *args, **kwargs)


core.load_data = liquid_point_in_time_data
pd.DataFrame.to_parquet = sampled_to_parquet
pd.DataFrame.apply = fast_apply
core.main()
