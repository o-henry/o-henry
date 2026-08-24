# V6 robust walk-forward strategy core. This file replaces pooled ML ranking.

from dataclasses import dataclass as _v6_dataclass, asdict as _v6_asdict


@_v6_dataclass(frozen=True)
class V6Config:
    min_n: int
    min_pf: float
    min_edge: float
    min_recent_pf: float
    min_quality: float
    min_positive_year_ratio: float
    top_n: int
    risk_per_trade: float
    index_weight: float
    transition_index_weight: float
    enable_panic: bool

    @property
    def config_id(self) -> str:
        return (
            f'n{self.min_n}-pf{self.min_pf:.2f}-e{self.min_edge:.4f}'
            f'-rpf{self.min_recent_pf:.2f}-q{self.min_quality:.2f}'
            f'-yr{self.min_positive_year_ratio:.2f}-top{self.top_n}'
            f'-risk{self.risk_per_trade:.3f}-iw{self.index_weight:.2f}'
            f'-tw{self.transition_index_weight:.2f}-panic{int(self.enable_panic)}'
        )


V6_ETFS = {
    'KOSPI': {'code': '069500', 'name': 'KODEX 200'},
    'KOSDAQ': {'code': '229200', 'name': 'KODEX 코스닥150'},
}
V6_ETF_BUY_COST = 0.0006
V6_ETF_SELL_COST = 0.0006
V6_MAX_STOCK_POSITIONS = 2
V6_MAX_GROSS = 0.98
V6_MAX_STOCK_WEIGHT = 0.72
V6_MONTHLY_LOSS_CUT = -0.14


def _v6_pf(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors='coerce').dropna()
    gp = float(v[v > 0].sum())
    gl = float(-v[v < 0].sum())
    return gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)


def _v6_group_stats(x: pd.DataFrame) -> pd.Series:
    r = x['net_return'].astype(float)
    yearly = x.assign(_year=x['exit_date'].dt.year).groupby('_year')['net_return'].mean()
    return pd.Series({
        'n': int(len(x)),
        'sum_return': float(r.sum()),
        'mean_return': float(r.mean()),
        'median_return': float(r.median()),
        'win_rate': float((r > 0).mean()),
        'pf': float(_v6_pf(r)),
        'downside_mean': float(r[r < 0].mean()) if (r < 0).any() else 0.0,
        'upside_mean': float(r[r > 0].mean()) if (r > 0).any() else 0.0,
        'years': int(len(yearly)),
        'positive_year_ratio': float((yearly > 0).mean()) if len(yearly) else 0.0,
        'year_mean_std': float(yearly.std(ddof=0)) if len(yearly) > 1 else 0.0,
        'worst_year_mean': float(yearly.min()) if len(yearly) else 0.0,
    })


def build_v6_profile_stats(candidates: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.DateOffset(years=5)
    h = candidates[(candidates['exit_date'] <= cutoff) & (candidates['exit_date'] >= start)].copy()
    if h.empty:
        raise RuntimeError(f'V6: no history through {cutoff.date()}')
    keys = ['market', 'pattern', 'regime', 'execution_id']
    full = h.groupby(keys, dropna=False).apply(_v6_group_stats, include_groups=False).reset_index()

    recent_start = cutoff - pd.DateOffset(days=550)
    recent = h[h['exit_date'] >= recent_start]
    if recent.empty:
        recent_stats = pd.DataFrame(columns=keys + ['recent_n', 'recent_mean', 'recent_pf', 'recent_win'])
    else:
        recent_stats = (
            recent.groupby(keys, dropna=False)
            .agg(
                recent_n=('net_return', 'size'),
                recent_mean=('net_return', 'mean'),
                recent_win=('target_positive', 'mean'),
            )
            .reset_index()
        )
        recent_pf = recent.groupby(keys, dropna=False)['net_return'].apply(_v6_pf).rename('recent_pf').reset_index()
        recent_stats = recent_stats.merge(recent_pf, on=keys, how='left')
    s = full.merge(recent_stats, on=keys, how='left')
    for c in ['recent_n', 'recent_mean', 'recent_win', 'recent_pf']:
        s[c] = pd.to_numeric(s[c], errors='coerce').fillna(0.0)

    # Conservative empirical-Bayes shrinkage toward zero, not toward the in-sample average.
    s['shrunk_mean'] = s['sum_return'] / (s['n'] + 30.0)
    s['recent_shrunk_mean'] = s['recent_mean'] * s['recent_n'] / (s['recent_n'] + 18.0)
    s['stability_penalty'] = 0.20 * s['year_mean_std'].clip(lower=0.0) + 0.20 * (-s['worst_year_mean']).clip(lower=0.0)
    s['robust_edge'] = (
        0.58 * s['shrunk_mean']
        + 0.27 * s['recent_shrunk_mean']
        + 0.15 * s['median_return']
        - s['stability_penalty']
    )
    s['profile_score'] = (
        100.0 * s['robust_edge']
        + 0.08 * s['pf'].clip(0, 3)
        + 0.05 * s['recent_pf'].clip(0, 3)
        + 0.06 * s['positive_year_ratio']
        + 0.03 * s['win_rate']
    )
    return s


def select_v6_profiles(stats: pd.DataFrame, cfg: V6Config) -> pd.DataFrame:
    s = stats.copy()
    defensive = s['regime'].isin(['bear', 'panic'])
    effective_min_n = np.where(defensive, np.maximum(12, np.floor(cfg.min_n * 0.65)), cfg.min_n)
    eligible = (
        (s['n'] >= effective_min_n)
        & (s['pf'] >= cfg.min_pf)
        & (s['robust_edge'] >= cfg.min_edge)
        & (s['positive_year_ratio'] >= cfg.min_positive_year_ratio)
        & ((s['recent_n'] < 8) | (s['recent_pf'] >= cfg.min_recent_pf))
        & ((s['recent_n'] < 8) | (s['recent_mean'] >= -0.0025))
        & (s['worst_year_mean'] >= -0.025)
    )
    if not cfg.enable_panic:
        eligible &= ~defensive
    s = s[eligible].copy()
    if s.empty:
        return s
    cell = ['market', 'pattern', 'regime']
    s = s.sort_values(cell + ['profile_score', 'n'], ascending=[True, True, True, False, False])
    return s.drop_duplicates(cell, keep='first').reset_index(drop=True)


def prepare_v6_stock_signals(
    candidates: pd.DataFrame,
    cutoff: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: V6Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = build_v6_profile_stats(candidates, cutoff)
    selected = select_v6_profiles(stats, cfg)
    if selected.empty:
        return candidates.iloc[0:0].copy(), selected
    keys = ['market', 'pattern', 'regime', 'execution_id']
    keep_cols = keys + [
        'n', 'pf', 'recent_n', 'recent_pf', 'robust_edge', 'profile_score',
        'positive_year_ratio', 'worst_year_mean', 'mean_return', 'median_return',
    ]
    x = candidates[candidates['date'].between(start, end)].merge(selected[keep_cols], on=keys, how='inner')
    x = x[
        (x['quality_rank'] >= cfg.min_quality)
        & x['entry_date'].between(start, end)
        & x['exit_date'].between(start, end)
    ].copy()
    if x.empty:
        return x, selected
    x['signal_score'] = (
        110.0 * x['robust_edge']
        + 0.035 * x['quality_rank']
        + 0.012 * x['rank_rel20'].fillna(0.0)
        + 0.008 * x['rank_adv20'].fillna(0.0)
        + 0.006 * x['close_loc'].fillna(0.0)
    )
    # One setup per stock/day. A daily cross-market ranking is applied by the simulator.
    x = x.sort_values(['date', 'code', 'signal_score'], ascending=[True, True, False])
    x = x.drop_duplicates(['date', 'code'], keep='first')
    return x.reset_index(drop=True), selected


def load_v6_etfs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for market, meta in V6_ETFS.items():
        code = meta['code']
        cache_file = SOURCE_CACHE / f'fdr_KRX_p1d_{code}.csv'
        if cache_file.exists() and cache_file.stat().st_size > 200:
            content = cache_file.read_bytes()
        else:
            url = urljoin(CSV_BASE, f'fdr_KRX_p1d_{code}.csv')
            r = requests.get(url, timeout=180, headers={'User-Agent': 'Mozilla/5.0 KRX-research-audit/1.0'})
            r.raise_for_status()
            content = r.content
            cache_file.write_bytes(content)
        d = pd.read_csv(io.BytesIO(content))
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
        for c in ['open', 'high', 'low', 'close', 'volume']:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        d = d[d['date'].between(START_DATE, END_DATE)].dropna(subset=['date', 'open', 'high', 'low', 'close'])
        d['market'] = market
        d['code'] = code
        d['name'] = meta['name']
        frames.append(d[['date', 'market', 'code', 'name', 'open', 'high', 'low', 'close', 'volume']])
    return pd.concat(frames, ignore_index=True).sort_values(['market', 'date']).reset_index(drop=True)


def build_v6_index_signals(etfs: pd.DataFrame, bench: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    bcols = [
        'date', 'market', 'regime', 'bench_close', 'bench_ma20', 'bench_ma60',
        'bench_ret5', 'bench_ret20', 'bench_slope20', 'bench_vol20', 'breadth20',
    ]
    x = etfs.merge(bench[bcols], on=['market', 'date'], how='left').sort_values(['market', 'date'])
    g = x.groupby('market', group_keys=False)
    for k in [1, 2, 3]:
        for c in ['open', 'high', 'low', 'close']:
            x[f'f{k}_{c}'] = g[c].shift(-k)
        x[f'f{k}_date'] = g['date'].shift(-k)
    x['trend_score'] = (
        x['bench_ret20'].fillna(0.0)
        + 0.55 * x['bench_ret5'].fillna(0.0)
        + 0.20 * x['bench_slope20'].fillna(0.0)
        + 0.10 * (x['breadth20'].fillna(0.5) - 0.5)
        - 0.25 * x['bench_vol20'].fillna(0.0)
    )
    bull = (
        x['regime'].eq('bull')
        & (x['bench_close'] > x['bench_ma20'])
        & (x['bench_ma20'] > x['bench_ma60'])
        & (x['bench_ret5'] > -0.012)
        & (x['breadth20'] >= 0.52)
    )
    transition = (
        x['regime'].isin(['transition', 'neutral'])
        & (x['bench_close'] > x['bench_ma20'])
        & (x['bench_slope20'] > 0.0)
        & (x['bench_ret5'] > 0.0)
        & (x['breadth20'] >= 0.50)
    )
    x = x[(bull | transition) & x['date'].between(start, end) & x['f1_open'].notna()].copy()
    x['sleeve_regime'] = np.where(bull.reindex(x.index).fillna(False), 'bull', 'transition')
    records: list[dict[str, Any]] = []
    for _, r in x.iterrows():
        entry = float(r['f1_open'])
        stop = entry * 0.965
        target = entry * 1.080
        exit_price = np.nan; exit_date = pd.NaT; reason = 'missing'; mae = 0.0; mfe = 0.0
        for k in [1, 2, 3]:
            o = r.get(f'f{k}_open'); h = r.get(f'f{k}_high'); l = r.get(f'f{k}_low'); cl = r.get(f'f{k}_close'); dt = r.get(f'f{k}_date')
            if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(cl) or pd.isna(dt):
                break
            o=float(o); h=float(h); l=float(l); cl=float(cl)
            mae=min(mae,l/entry-1); mfe=max(mfe,h/entry-1)
            if o <= stop:
                exit_price,exit_date,reason=o,pd.Timestamp(dt),'gap_stop'; break
            if o >= target:
                exit_price,exit_date,reason=o,pd.Timestamp(dt),'gap_target'; break
            if l <= stop:
                exit_price,exit_date,reason=stop,pd.Timestamp(dt),'stop'; break
            if h >= target:
                exit_price,exit_date,reason=target,pd.Timestamp(dt),'target'; break
            if k == 3:
                exit_price,exit_date,reason=cl,pd.Timestamp(dt),'time'
        if pd.isna(exit_price) or pd.isna(exit_date):
            continue
        records.append({
            'asset_type':'index','market':r['market'],'code':r['code'],'name':r['name'],
            'pattern':'index_sleeve','regime':r['sleeve_regime'],'date':r['date'],
            'entry_date':pd.Timestamp(r['f1_date']),'entry_price':entry,
            'exit_date':pd.Timestamp(exit_date),'exit_price':float(exit_price),'exit_reason':reason,
            'stop_pct':0.035,'target_pct':0.080,'hold_days':3,'signal_score':float(r['trend_score']),
            'mae':mae,'mfe':mfe,
        })
    return pd.DataFrame(records)
