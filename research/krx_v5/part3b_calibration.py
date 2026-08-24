# Runtime overrides loaded after part3.py. The prior hard -999 gate could make
# every candidate tie and leave `best` unset. Use a broad grid and continuous
# penalties so the experiment always completes and exposes poor calibration.

def threshold_grid(scored_2024: pd.DataFrame) -> list[tuple[ThresholdConfig, float]]:
    valid_edges = scored_2024['edge_score'].replace([np.inf, -np.inf], np.nan).dropna()
    if valid_edges.empty:
        raise RuntimeError('No finite 2024 edge scores')
    out: list[tuple[ThresholdConfig, float]] = []
    for q in [0.35, 0.50, 0.62, 0.72, 0.80, 0.87, 0.92, 0.96]:
        edge = float(valid_edges.quantile(q))
        for p in [0.38, 0.44, 0.50, 0.56, 0.62]:
            for n in [1, 2]:
                for panic_add in [0.00, 0.05]:
                    out.append(ThresholdConfig(q, p, n, panic_add))
                    # preserve one edge per config through the tuple contract
                    out[-1] = (out[-1], edge)
    return out


def calibration_score(m: dict[str, Any]) -> float:
    trades = int(m.get('trades', 0))
    pf = float(m.get('profit_factor', 0.0))
    mdd = float(m.get('max_drawdown', 0.0))
    ret = float(m.get('return', 0.0))
    geom = float(m.get('monthly_geom', 0.0))
    positive = float(m.get('positive_month_ratio', 0.0))
    penalty = 0.0
    if trades < 12:
        penalty += (12 - trades) * 0.025
    if pf < 1.0:
        penalty += (1.0 - pf) * 0.55
    if mdd < -0.18:
        penalty += abs(mdd + 0.18) * 2.0
    return (
        2.2 * ret
        + 1.25 * geom
        + 0.14 * min(max(pf, 0.0), 3.0)
        + 0.10 * positive
        + 0.9 * min(0.0, mdd + 0.12)
        - penalty
    )
