# Runtime overrides loaded after part1.py and before downstream feature code.
SOURCE_CACHE = Path('/tmp/krx_v5_cache')
SOURCE_CACHE.mkdir(parents=True, exist_ok=True)


def _download_one(rec: tuple[str, str, str, str], retries: int = 3) -> tuple[pd.DataFrame | None, str | None]:
    code, name, market, url = rec
    cache_file = SOURCE_CACHE / f'fdr_KRX_p1d_{code}.csv'
    last: Exception | None = None
    for attempt in range(retries):
        try:
            if cache_file.exists() and cache_file.stat().st_size > 200:
                content = cache_file.read_bytes()
            else:
                r = requests.get(url, timeout=180, headers={'User-Agent': 'Mozilla/5.0 KRX-research-audit/1.0'})
                r.raise_for_status()
                content = r.content
                tmp = cache_file.with_suffix('.tmp')
                tmp.write_bytes(content)
                tmp.replace(cache_file)
            df = pd.read_csv(io.BytesIO(content))
            required = {'date', 'open', 'high', 'low', 'close', 'volume', 'marcap', 'shares'}
            if not required.issubset(df.columns):
                cache_file.unlink(missing_ok=True)
                raise ValueError(f'missing columns {required - set(df.columns)}')
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df[df['date'].between(START_DATE, END_DATE)].copy()
            if df.empty:
                return None, f'{code}: no rows in date range'
            for c in ['open', 'high', 'low', 'close', 'volume', 'marcap', 'shares']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df['code'] = code
            df['name'] = name
            df['market'] = market
            return df[['date', 'code', 'name', 'market', 'open', 'high', 'low', 'close', 'volume', 'marcap', 'shares']], None
        except Exception as exc:
            last = exc
            if cache_file.exists() and attempt == 0:
                cache_file.unlink(missing_ok=True)
            time.sleep(0.4 * (attempt + 1))
    return None, f'{code}: {last!r}'


def adjust_splits(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Detect only split-like discontinuities. Broad same-date changes are treated as
    # vendor/share-count revisions rather than corporate actions.
    work = panel.sort_values(['code', 'date']).copy()
    g = work.groupby('code', group_keys=False)
    work['_price_ratio'] = work['close'] / g['close'].shift(1)
    work['_share_ratio'] = work['shares'] / g['shares'].shift(1)
    work['_marcap_ratio'] = work['marcap'] / g['marcap'].shift(1)
    work['_inverse_error'] = (work['_price_ratio'] * work['_share_ratio'] - 1.0).abs()

    common_ratios = np.array([0.1, 0.2, 0.25, 1/3, 0.5, 2.0, 3.0, 4.0, 5.0, 10.0], dtype=float)
    sr = work['_share_ratio'].to_numpy(dtype=float)
    ratio_distance = np.full(len(work), np.inf, dtype=float)
    finite = np.isfinite(sr) & (sr > 0)
    if finite.any():
        ratio_distance[finite] = np.min(np.abs(np.log(sr[finite, None] / common_ratios[None, :])), axis=1)
    work['_ratio_distance'] = ratio_distance

    raw_candidate = (
        finite
        & (work['_ratio_distance'] <= np.log(1.06))
        & (work['_inverse_error'] <= 0.16)
        & work['_marcap_ratio'].between(0.70, 1.35)
        & ((work['_price_ratio'] <= 0.72) | (work['_price_ratio'] >= 1.38))
    )
    work['_raw_split'] = raw_candidate.astype(int)
    per_date = work.groupby('date')['_raw_split'].transform('sum')
    # More than 8 simultaneous candidates is almost certainly a source-field reset.
    work['_split'] = ((work['_raw_split'] == 1) & (per_date <= 8)).astype(int)

    out: list[pd.DataFrame] = []
    events: list[dict[str, Any]] = []
    rejected_vendor_dates = sorted(
        str(pd.Timestamp(d).date()) for d, n in work.groupby('date')['_raw_split'].sum().items() if n > 8
    )
    for code, x in work.groupby('code', sort=False):
        x = x.sort_values('date').copy()
        event_ratio = pd.Series(np.where(x['_split'].eq(1), x['_share_ratio'], 1.0), index=x.index, dtype=float)
        factor = event_ratio.shift(-1, fill_value=1.0).iloc[::-1].cumprod().iloc[::-1]
        factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for c in ['open', 'high', 'low', 'close']:
            x[f'adj_{c}'] = x[c] / factor
        x['adj_shares'] = x['shares'] * factor
        x['split_event'] = x['_split'].astype(int)
        x['split_factor'] = factor
        x['corp_event_recent'] = x['split_event'].astype(float).rolling(6, min_periods=1).max().fillna(0.0).astype(int)
        if x['split_event'].any():
            for r in x.loc[x['split_event'].eq(1), ['date', 'code', 'name', 'market', '_share_ratio', '_price_ratio']].itertuples(index=False):
                events.append({
                    'date': str(r.date.date()), 'code': r.code, 'name': r.name, 'market': r.market,
                    'share_ratio': float(r._4), 'price_ratio': float(r._5),
                })
        out.append(x)
    p = pd.concat(out, ignore_index=True).sort_values(['market', 'code', 'date']).reset_index(drop=True)
    p['adj_ret1_check'] = p.groupby('code')['adj_close'].pct_change()
    p['bad_adjustment'] = (p['adj_ret1_check'].abs() > 0.45).astype(int)
    p = p.drop(columns=[c for c in p.columns if c.startswith('_')], errors='ignore')
    quality = {
        'split_events': len(events),
        'split_event_sample': events[:100],
        'rejected_vendor_reset_dates': rejected_vendor_dates[:100],
        'bad_adjustment_rows': int(p['bad_adjustment'].sum()),
    }
    return p, quality
