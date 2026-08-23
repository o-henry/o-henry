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
    for hold, topn, score, stop, target, tx, ba, pa, pw in itertools.product(
        [2, 3],
        [1, 2],
        [0.82, 0.88],
        [1.5, 2.0],
        [3.0, 5.0],
        [0.50, 0.70],
        [0.15],
        [0.10],
        [0.30, 0.50],
    ):
        out.append(core.Config(hold, topn, score, stop, target, 1.0, 0.80, tx, ba, pa, pw))
    return out


core.configs = bounded_configs
_orig_to_parquet = pd.DataFrame.to_parquet


def sampled_to_parquet(self, path, *args, **kwargs):
    if str(path).endswith('feature_sample.parquet') and len(self) > 25_000:
        return _orig_to_parquet(self.sample(25_000, random_state=20260823), path, *args, **kwargs)
    return _orig_to_parquet(self, path, *args, **kwargs)


pd.DataFrame.to_parquet = sampled_to_parquet
core.main()
