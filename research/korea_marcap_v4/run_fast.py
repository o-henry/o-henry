from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('korea_marcap_v4_fast_core', HERE / 'run_v4.py')
if SPEC is None or SPEC.loader is None:
    raise RuntimeError('unable to import run_v4.py')
core = importlib.util.module_from_spec(SPEC)
sys.modules['korea_marcap_v4_fast_core'] = core
SPEC.loader.exec_module(core)


def bounded_configs():
    out = []
    for (kr, qr), expected, stop, tx in itertools.product(
        [(0.95, 0.90), (0.97, 0.95), (0.98, 0.97)],
        [-0.005, 0.000, 0.003],
        [2.0, 3.0],
        [0.60, 0.80],
    ):
        out.append(core.Config(kr, qr, expected, stop, 0.50, tx, 0.30, 0.15, 0.10, -0.12, 3))
    return out


core.configs = bounded_configs
core.main()
