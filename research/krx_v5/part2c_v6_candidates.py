# V6 candidate generator. Loaded after part2.py.
# It replaces the pooled ML input with interpretable, pattern-specific signals
# and multiple coarse execution profiles. Every label is formed from t+1..t+3 only.

V6_EXECUTION_PROFILES: dict[str, tuple[float, float, int]] = {
    'h1_s025_t050': (0.025, 0.050, 1),
    'h1_s035_t070': (0.035, 0.070, 1),
    'h2_s030_t065': (0.030, 0.065, 2),
    'h2_s040_t085': (0.040, 0.085, 2),
    'h3_s035_t080': (0.035, 0.080, 3),
    'h3_s045_t110': (0.045, 0.110, 3),
    'h3_s050_time': (0.050, 0.500, 3),
}

V6_BASE_COLUMNS = [
    'date', 'code', 'name', 'market', 'regime',
    'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume', 'marcap', 'shares',
    'ret1', 'ret2', 'ret3', 'ret5', 'ret10', 'ret20', 'ret60', 'ret120',
    'rel1', 'rel5', 'rel20', 'rel60', 'gap', 'intraday', 'range_pct', 'close_loc',
    'volume_ratio', 'turnover_ratio', 'adv20', 'adv60', 'vol10', 'vol20',
    'drawdown20', 'drawdown60', 'dist_ma20_atr', 'atr_pct', 'beta20',
    'rank_rel5', 'rank_rel20', 'rank_rel60', 'rank_ret20', 'rank_volume_ratio',
    'rank_adv20', 'rank_low_vol20', 'breadth20', 'bench_ret5', 'bench_ret20',
    'bench_vol20', 'bench_dist_ma20', 'market_gap', 'market_intraday',
    'ma10', 'ma20', 'ma60', 'high20_prev', 'high60_prev', 'low20_prev',
    'prev_high', 'prev_low', 'prev_close', 'history_n', 'max_abs_ret20',
    'corp_event_recent', 'bad_adjustment',
    'f1_adj_open', 'f1_adj_high', 'f1_adj_low', 'f1_adj_close', 'f1_date',
    'f2_adj_open', 'f2_adj_high', 'f2_adj_low', 'f2_adj_close', 'f2_date',
    'f3_adj_open', 'f3_adj_high', 'f3_adj_low', 'f3_adj_close', 'f3_date',
]


def _v6_quality_clip(s: pd.Series, low: float = 0.0, high: float = 1.0) -> pd.Series:
    return s.clip(low, high).fillna(0.0)


def _v6_slim(x: pd.DataFrame, pattern: str, quality: pd.Series) -> pd.DataFrame:
    cols = [c for c in V6_BASE_COLUMNS if c in x.columns]
    y = x[cols].copy()
    y['pattern'] = pattern
    y['raw_quality'] = pd.to_numeric(quality.reindex(x.index), errors='coerce').fillna(0.0).to_numpy()
    return y


def _v6_label_one(r: pd.Series) -> dict[str, Any]:
    entry = float(r['f1_adj_open'])
    stop = entry * (1.0 - float(r['stop_pct']))
    target = entry * (1.0 + float(r['target_pct']))
    hold = int(r['hold_days'])
    exit_price = np.nan
    exit_date = pd.NaT
    reason = 'missing'
    mae = 0.0
    mfe = 0.0
    for k in range(1, hold + 1):
        o = r.get(f'f{k}_adj_open')
        h = r.get(f'f{k}_adj_high')
        l = r.get(f'f{k}_adj_low')
        cl = r.get(f'f{k}_adj_close')
        dt = r.get(f'f{k}_date')
        if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(cl) or pd.isna(dt):
            break
        o = float(o); h = float(h); l = float(l); cl = float(cl)
        mae = min(mae, l / entry - 1.0)
        mfe = max(mfe, h / entry - 1.0)
        # Conservative ordering: adverse gap/stop is assumed before target when both are possible.
        if o <= stop:
            exit_price, exit_date, reason = o, pd.Timestamp(dt), 'gap_stop'
            break
        if o >= target:
            exit_price, exit_date, reason = o, pd.Timestamp(dt), 'gap_target'
            break
        if l <= stop:
            exit_price, exit_date, reason = stop, pd.Timestamp(dt), 'stop'
            break
        if h >= target:
            exit_price, exit_date, reason = target, pd.Timestamp(dt), 'target'
            break
        if k == hold:
            exit_price, exit_date, reason = cl, pd.Timestamp(dt), 'time'
    if pd.isna(exit_price) or pd.isna(exit_date) or entry <= 0:
        return {'label_valid': False}
    net = (float(exit_price) * (1.0 - SELL_COST)) / (entry * (1.0 + BUY_COST)) - 1.0
    return {
        'label_valid': True,
        'entry_date': pd.Timestamp(r['f1_date']),
        'entry_price': entry,
        'exit_date': pd.Timestamp(exit_date),
        'exit_price': float(exit_price),
        'exit_reason': reason,
        'net_return': float(net),
        'mae': float(mae),
        'mfe': float(mfe),
    }


def generate_candidates_v6(p: pd.DataFrame) -> pd.DataFrame:
    d = p.copy().sort_values(['code', 'date']).reset_index(drop=True)
    # An unexplained >45% adjusted move contaminates long lookbacks. Exclude the following 120 sessions.
    d['anomaly_recent120'] = (
        d.groupby('code')['bad_adjustment']
        .transform(lambda s: s.rolling(120, min_periods=1).max())
        .fillna(0.0)
    )
    min_adv = np.where(d['market'].eq('KOSPI'), 2_500_000_000.0, 1_200_000_000.0)
    base = (
        (d['date'] >= pd.Timestamp('2020-01-01'))
        & (d['history_n'] >= 120)
        & d['adj_close'].between(1500, 700000)
        & (d['marcap'] >= 120_000_000_000)
        & (d['adv20'] >= min_adv)
        & d['vol20'].between(0.007, 0.125)
        & (d['max_abs_ret20'] <= 0.38)
        & (d['corp_event_recent'] == 0)
        & (d['anomaly_recent120'] == 0)
        & d['f1_adj_open'].notna()
        & (d['ret1'].abs() < 0.285)
        & (d['range_pct'] <= 0.30)
    )
    d = d[base].copy()
    if d.empty:
        raise RuntimeError('V6: no eligible rows')

    pieces: list[pd.DataFrame] = []

    m = (
        d['regime'].isin(['bull', 'transition', 'neutral'])
        & (d['rank_rel20'] >= 0.90)
        & (d['rank_rel60'] >= 0.82)
        & (d['adj_close'] > d['ma20'])
        & (d['ma20'] >= d['ma60'] * 0.985)
        & d['ret20'].between(0.035, 0.45)
        & d['ret1'].between(-0.015, 0.105)
        & (d['close_loc'] >= 0.58)
        & (d['volume_ratio'] >= 0.65)
        & (d['dist_ma20_atr'] <= 3.0)
    )
    x = d[m]
    q = (
        0.28 * x['rank_rel20'] + 0.20 * x['rank_rel60'] + 0.12 * x['rank_ret20']
        + 0.12 * _v6_quality_clip(x['close_loc'])
        + 0.10 * _v6_quality_clip(np.log1p(x['volume_ratio']) / np.log(4.0))
        + 0.10 * x['rank_adv20'] + 0.08 * x['rank_low_vol20']
    )
    pieces.append(_v6_slim(x, 'leader_continuation', q))

    m = (
        d['regime'].isin(['bull', 'transition'])
        & (d['rank_rel20'] >= 0.84)
        & (d['ret60'] >= 0.07)
        & d['ret3'].between(-0.105, -0.003)
        & (d['adj_close'] >= d['ma20'] * 0.968)
        & (d['adj_close'] >= d['ma60'])
        & (d['close_loc'] >= 0.56)
        & (d['intraday'] >= -0.006)
        & (d['volume_ratio'] >= 0.55)
        & (d['dist_ma20_atr'] <= 2.1)
    )
    x = d[m]
    q = (
        0.30 * x['rank_rel20'] + 0.17 * x['rank_rel60'] + 0.12 * x['rank_adv20']
        + 0.13 * _v6_quality_clip(x['close_loc'])
        + 0.10 * _v6_quality_clip((0.11 + x['ret3']) / 0.11)
        + 0.10 * x['rank_low_vol20'] + 0.08 * _v6_quality_clip(np.log1p(x['volume_ratio']) / np.log(3.0))
    )
    pieces.append(_v6_slim(x, 'leader_pullback', q))

    m = (
        d['regime'].isin(['bull', 'transition', 'neutral'])
        & (d['adj_close'] >= d['high20_prev'] * 1.001)
        & d['ret1'].between(0.014, 0.135)
        & (d['rank_rel20'] >= 0.68)
        & (d['volume_ratio'] >= 1.18)
        & (d['close_loc'] >= 0.70)
        & (d['gap'] <= 0.085)
        & (d['dist_ma20_atr'] <= 3.0)
    )
    x = d[m]
    q = (
        0.24 * x['rank_rel20'] + 0.14 * x['rank_rel5'] + 0.15 * x['rank_volume_ratio']
        + 0.14 * x['rank_adv20'] + 0.15 * _v6_quality_clip(x['close_loc'])
        + 0.10 * _v6_quality_clip(x['intraday'] / 0.10 + 0.25)
        + 0.08 * x['rank_low_vol20']
    )
    pieces.append(_v6_slim(x, 'fresh_breakout', q))

    m = (
        d['regime'].isin(['bull', 'transition', 'neutral'])
        & d['gap'].between(0.018, 0.125)
        & d['ret1'].between(0.026, 0.19)
        & (d['intraday'] >= 0.0)
        & (d['close_loc'] >= 0.72)
        & (d['volume_ratio'] >= 1.55)
        & (d['rank_adv20'] >= 0.70)
        & (d['dist_ma20_atr'] <= 3.4)
    )
    x = d[m]
    q = (
        0.18 * x['rank_rel20'] + 0.18 * x['rank_volume_ratio'] + 0.18 * x['rank_adv20']
        + 0.18 * _v6_quality_clip(x['close_loc']) + 0.12 * _v6_quality_clip(x['intraday'] / 0.12)
        + 0.08 * _v6_quality_clip(1.0 - (x['gap'] - 0.018) / 0.107)
        + 0.08 * x['rank_low_vol20']
    )
    pieces.append(_v6_slim(x, 'event_gap_hold', q))

    m = (
        d['regime'].isin(['transition', 'neutral'])
        & (d['ret60'] >= 0.06)
        & (d['rank_rel20'] >= 0.82)
        & d['ret5'].between(-0.14, 0.015)
        & (d['intraday'] >= 0.010)
        & (d['adj_close'] >= d['prev_high'] * 0.995)
        & (d['close_loc'] >= 0.68)
        & (d['volume_ratio'] >= 0.85)
        & (d['dist_ma20_atr'] <= 2.2)
    )
    x = d[m]
    q = (
        0.30 * x['rank_rel20'] + 0.16 * x['rank_rel60'] + 0.14 * x['rank_volume_ratio']
        + 0.12 * x['rank_adv20'] + 0.14 * _v6_quality_clip(x['close_loc'])
        + 0.08 * _v6_quality_clip(x['intraday'] / 0.10)
        + 0.06 * x['rank_low_vol20']
    )
    pieces.append(_v6_slim(x, 'transition_reclaim', q))

    m = (
        d['regime'].isin(['bear', 'panic'])
        & (d['rank_rel20'] >= 0.965)
        & (d['rel20'] >= 0.09)
        & (d['ret20'] > 0.0)
        & (d['adj_close'] > d['ma20'])
        & (d['ret1'] >= 0.0)
        & (d['close_loc'] >= 0.75)
        & (d['volume_ratio'] >= 1.15)
        & (d['beta20'] <= 1.05)
        & (d['gap'] > -0.06)
    )
    x = d[m]
    q = (
        0.34 * x['rank_rel20'] + 0.17 * x['rank_rel5'] + 0.13 * x['rank_volume_ratio']
        + 0.12 * x['rank_adv20'] + 0.13 * _v6_quality_clip(x['close_loc'])
        + 0.07 * x['rank_low_vol20'] + 0.04 * _v6_quality_clip(1.0 - x['beta20'] / 1.2)
    )
    pieces.append(_v6_slim(x, 'panic_absolute_strength', q))

    m = (
        d['regime'].eq('panic')
        & (d['ret5'] <= -0.16)
        & (d['gap'] <= -0.018)
        & (d['intraday'] >= 0.055)
        & (d['close_loc'] >= 0.88)
        & (d['volume_ratio'] >= 2.2)
        & (d['rank_adv20'] >= 0.80)
        & (d['adj_close'] >= d['low20_prev'] * 1.025)
    )
    x = d[m]
    q = (
        0.20 * x['rank_rel5'] + 0.20 * x['rank_volume_ratio'] + 0.18 * x['rank_adv20']
        + 0.20 * _v6_quality_clip(x['close_loc']) + 0.14 * _v6_quality_clip(x['intraday'] / 0.15)
        + 0.08 * _v6_quality_clip((-x['ret5'] - 0.16) / 0.20)
    )
    pieces.append(_v6_slim(x, 'panic_reversal', q))

    m = (
        d['regime'].isin(['transition', 'neutral', 'panic'])
        & d['ret5'].between(-0.18, -0.065)
        & (d['intraday'] >= 0.030)
        & (d['close_loc'] >= 0.78)
        & (d['volume_ratio'] >= 1.35)
        & (d['rank_adv20'] >= 0.70)
        & (d['gap'] > -0.10)
        & (d['adj_close'] >= d['low20_prev'] * 1.02)
    )
    x = d[m]
    q = (
        0.18 * x['rank_rel5'] + 0.18 * x['rank_volume_ratio'] + 0.16 * x['rank_adv20']
        + 0.20 * _v6_quality_clip(x['close_loc']) + 0.18 * _v6_quality_clip(x['intraday'] / 0.14)
        + 0.10 * _v6_quality_clip((-x['ret5'] - 0.065) / 0.115)
    )
    pieces.append(_v6_slim(x, 'oversold_reclaim', q))

    base_candidates = pd.concat([x for x in pieces if not x.empty], ignore_index=True)
    base_candidates = base_candidates.drop_duplicates(['date', 'code', 'pattern']).reset_index(drop=True)
    base_candidates['quality_rank'] = (
        base_candidates.groupby(['market', 'date', 'pattern'])['raw_quality'].rank(pct=True, method='average')
    )

    expanded: list[pd.DataFrame] = []
    for exec_id, (stop, target, hold) in V6_EXECUTION_PROFILES.items():
        x = base_candidates.copy()
        x['execution_id'] = exec_id
        x['stop_pct'] = stop
        x['target_pct'] = target
        x['hold_days'] = hold
        expanded.append(x)
    candidates = pd.concat(expanded, ignore_index=True)
    labels = pd.DataFrame([_v6_label_one(r) for _, r in candidates.iterrows()], index=candidates.index)
    candidates = pd.concat([candidates, labels], axis=1)
    candidates = candidates[candidates['label_valid'].fillna(False)].copy()
    candidates['target_positive'] = (candidates['net_return'] > 0.0).astype(int)
    candidates['signal_year'] = candidates['date'].dt.year
    candidates['exit_year'] = candidates['exit_date'].dt.year
    return candidates.reset_index(drop=True)
