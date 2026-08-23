from __future__ import annotations

from pathlib import Path

path = Path(__file__).resolve().parent / 'run_v2.py'
text = path.read_text(encoding='utf-8')

old_grid = """    for rank, expected, stop, target, transition, panic in itertools.product(\n        [0.88, 0.92, 0.95],\n        [0.003, 0.006, 0.009],\n        [1.25, 1.75],\n        [2.5, 4.0],\n        [0.30, 0.50],\n        [0.00, 0.10],\n    ):"""
new_grid = """    for rank, expected, stop, target, transition, panic in itertools.product(\n        [0.90, 0.94],\n        [0.004, 0.007, 0.010],\n        [1.25, 1.75],\n        [2.5, 4.0],\n        [0.30, 0.50],\n        [0.00],\n    ):"""
if old_grid not in text:
    raise RuntimeError('config grid source block not found')
text = text.replace(old_grid, new_grid)

marker = "\ndef simulate(\n"
if marker not in text:
    raise RuntimeError('simulate marker not found')
text = text.replace(marker, "\n_SIM_CONTEXT_CACHE: dict[tuple[str, str], tuple[list[pd.Timestamp], dict[pd.Timestamp, int], dict[str, pd.DataFrame]]] = {}\n_SIGNAL_CACHE: dict[tuple[int, str, str], dict[pd.Timestamp, pd.DataFrame]] = {}\n\n\ndef simulate(\n", 1)

old_context = """    raw = frame[frame['Date'].between(start, end + pd.Timedelta(days=10))].sort_values(['Code', 'Date'])\n    all_dates = [pd.Timestamp(d) for d in sorted(raw[raw['Date'].between(start, end)]['Date'].unique())]\n    if not all_dates:\n        return empty_metrics(start, end), pd.DataFrame(), pd.DataFrame()\n    date_to_i = {d: i for i, d in enumerate(all_dates)}\n    by_code = {c: g.set_index('Date').sort_index() for c, g in raw.groupby('Code')}\n    by_signal = {pd.Timestamp(d): g for d, g in candidates.groupby('Date')}\n"""
new_context = """    context_key = (str(start.date()), str(end.date()))\n    context = _SIM_CONTEXT_CACHE.get(context_key)\n    if context is None:\n        raw = frame[frame['Date'].between(start, end + pd.Timedelta(days=10))].sort_values(['Code', 'Date'])\n        all_dates = [pd.Timestamp(d) for d in sorted(raw[raw['Date'].between(start, end)]['Date'].unique())]\n        if not all_dates:\n            return empty_metrics(start, end), pd.DataFrame(), pd.DataFrame()\n        date_to_i = {d: i for i, d in enumerate(all_dates)}\n        by_code = {c: g.set_index('Date').sort_index() for c, g in raw.groupby('Code')}\n        _SIM_CONTEXT_CACHE[context_key] = (all_dates, date_to_i, by_code)\n    else:\n        all_dates, date_to_i, by_code = context\n    signal_key = (id(candidates), str(start.date()), str(end.date()))\n    by_signal = _SIGNAL_CACHE.get(signal_key)\n    if by_signal is None:\n        by_signal = {pd.Timestamp(d): g for d, g in candidates.groupby('Date')}\n        _SIGNAL_CACHE[signal_key] = by_signal\n"""
if old_context not in text:
    raise RuntimeError('simulation context source block not found')
text = text.replace(old_context, new_context, 1)

path.write_text(text, encoding='utf-8')
print('patched', path)
