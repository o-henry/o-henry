from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path('/tmp/korea_regime_v4_fast')
spec = importlib.util.spec_from_file_location('v4base', ROOT / 'v4_regime_overlay.py')
if spec is None or spec.loader is None:
    raise RuntimeError('unable to import v4 base')
v4 = importlib.util.module_from_spec(spec)
sys.modules['v4base'] = v4
spec.loader.exec_module(v4)


def fast_configs():
    out = []
    for n in [1, 2]:
        for bull_stock, bull_index in [(0.50, 0.45), (0.65, 0.30)]:
            for trans_stock, trans_index in [(0.0, 0.0), (0.25, 0.20)]:
                for rebound in [0.0, 0.15]:
                    for bull_style in ['bull_continuous', 'bull_pullback']:
                        out.append(v4.Config(
                            bull_hold=2,
                            transition_hold=1,
                            n_select=n,
                            bull_stock_weight=bull_stock,
                            bull_index_weight=bull_index,
                            transition_stock_weight=trans_stock,
                            transition_index_weight=trans_index,
                            rebound_stock_weight=rebound,
                            overheat_sigma=2.5,
                            bull_style=bull_style,
                            transition_style='transition_defensive',
                        ))
    return out

v4.configs = fast_configs
v4.OUT = ROOT / 'outputs'
v4.OUT.mkdir(parents=True, exist_ok=True)
v4.main()
