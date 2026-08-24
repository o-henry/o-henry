def _v6_signal_key(cfg: V6Config) -> tuple[Any,...]:
    return (
        cfg.min_n,cfg.min_pf,cfg.min_edge,cfg.min_recent_pf,cfg.min_quality,
        cfg.min_positive_year_ratio,cfg.enable_panic,
    )


def _v6_dev_gate(year_metrics: list[dict[str,Any]]) -> dict[str,bool]:
    r=np.array([m['return'] for m in year_metrics],dtype=float)
    trades=sum(int(m['trades']) for m in year_metrics)
    return {
        'compound_positive': float(np.prod(1+r)-1)>0,
        'positive_years_2_of_3': int((r>0).sum())>=2,
        'worst_year_above_minus_15pct': float(r.min())>=-0.15,
        'total_trades_30': trades>=30,
        'median_pf_1_0': float(np.median([m['profit_factor'] for m in year_metrics]))>=1.0,
    }


def _v6_validation_gate(m:dict[str,Any],stress:dict[str,Any])->dict[str,bool]:
    stock=m.get('asset_stats',{}).get('stock',{})
    return {
        'return_12pct':m['return']>=0.12,
        'monthly_geom_1pct':m['monthly_geom']>=0.01,
        'profit_factor_1_15':m['profit_factor']>=1.15,
        'mdd_18pct':m['max_drawdown']>=-0.18,
        'positive_months_half':m['positive_month_ratio']>=0.50,
        'trades_24':m['trades']>=24,
        'cost_stress_positive':stress['return']>0,
        'stock_component_positive':float(stock.get('pnl',0.0))>0,
    }


def _v6_stress_gate(m:dict[str,Any],stress:dict[str,Any])->dict[str,bool]:
    regime=m.get('regime_stats',{})
    stock=m.get('asset_stats',{}).get('stock',{})
    transition=float(regime.get('transition',{}).get('pnl',0.0))+float(regime.get('neutral',{}).get('pnl',0.0))
    defensive=float(regime.get('bear',{}).get('pnl',0.0))+float(regime.get('panic',{}).get('pnl',0.0))
    return {
        'return_50pct':m['return']>=0.50,
        'monthly_geom_5pct':m['monthly_geom']>=0.05,
        'profit_factor_1_25':m['profit_factor']>=1.25,
        'mdd_20pct':m['max_drawdown']>=-0.20,
        'positive_months_5_of_8':m['positive_month_ratio']>=0.625,
        'trades_20':m['trades']>=20,
        'transition_nonnegative':transition>=0,
        'defensive_nonnegative':defensive>=0,
        'cost_stress_positive':stress['return']>0,
        'stock_component_positive':float(stock.get('pnl',0.0))>0,
    }


def _v6_user_target(m:dict[str,Any])->dict[str,bool]:
    return {
        'monthly_geom_20pct':m['monthly_geom']>=0.20,
        'positive_months_6_of_8':m['positive_month_ratio']>=0.75,
        'mdd_25pct':m['max_drawdown']>=-0.25,
    }


def _v6_profile_audit(candidates:pd.DataFrame,cutoff:pd.Timestamp,path:Path)->pd.DataFrame:
    s=build_v6_profile_stats(candidates,cutoff)
    s.sort_values(['profile_score','n'],ascending=[False,False]).to_csv(path,index=False)
    return s


def main()->None:
    print('V6 STEP 1: long OHLCV panel')
    panel_raw,download_quality=download_panel()
    panel,adjustment_quality=adjust_splits(panel_raw)
    print(json.dumps({'download':download_quality,'adjustment':adjustment_quality},ensure_ascii=False,indent=2))

    print('V6 STEP 2: features and multi-profile candidates')
    benchmarks=download_benchmarks(panel['date'].min(),panel['date'].max())
    features,bench=build_features(panel,benchmarks)
    candidates=generate_candidates_v6(features)
    print('v6 candidates',len(candidates),'base signals',candidates[['date','code','pattern']].drop_duplicates().shape[0])
    print(candidates.groupby(['pattern','regime']).size().to_string())
    etfs=load_v6_etfs()

    # Save raw, pre-selection evidence before any optimization.
    raw_year_pattern=(
        candidates.assign(year=candidates['exit_date'].dt.year)
        .groupby(['year','market','pattern','regime','execution_id'])
        .agg(n=('net_return','size'),mean_return=('net_return','mean'),median_return=('net_return','median'),win_rate=('target_positive','mean'))
        .reset_index()
    )
    raw_pf=(
        candidates.assign(year=candidates['exit_date'].dt.year)
        .groupby(['year','market','pattern','regime','execution_id'])['net_return']
        .apply(_v6_pf).rename('pf').reset_index()
    )
    raw_year_pattern.merge(raw_pf,on=['year','market','pattern','regime','execution_id'],how='left').to_csv(OUT/'raw_candidate_yearly_stats.csv',index=False)

    development_years=[2022,2023,2024]
    year_context:dict[int,dict[str,Any]]={}
    stats_by_cutoff:dict[int,pd.DataFrame]={}
    for year in development_years+[2025,2026]:
        cutoff=pd.Timestamp(f'{year-1}-12-31')
        start=pd.Timestamp(f'{year}-01-01')
        end=pd.Timestamp(f'{year}-12-31') if year<2026 else END_DATE
        stats_by_cutoff[year]=_v6_profile_audit(candidates,cutoff,OUT/f'profile_stats_for_{year}.csv')
        index_signals=build_v6_index_signals(etfs,bench,start,end)
        year_context[year]={'cutoff':cutoff,'start':start,'end':end,'index_signals':index_signals}

    configs=v6_config_grid()
    signal_cache:dict[tuple[int,tuple[Any,...]],tuple[pd.DataFrame,pd.DataFrame]]={}

    def get_signals(year:int,cfg:V6Config)->tuple[pd.DataFrame,pd.DataFrame]:
        key=(year,_v6_signal_key(cfg))
        if key not in signal_cache:
            ctx=year_context[year]
            selected=select_v6_profiles(stats_by_cutoff[year],cfg)
            if selected.empty:
                signals=candidates.iloc[0:0].copy()
            else:
                keys=['market','pattern','regime','execution_id']
                keep=keys+['n','pf','recent_n','recent_pf','robust_edge','profile_score','positive_year_ratio','worst_year_mean','mean_return','median_return']
                signals=candidates[candidates['date'].between(ctx['start'],ctx['end'])].merge(selected[keep],on=keys,how='inner')
                signals=signals[
                    (signals['quality_rank']>=cfg.min_quality)
                    & signals['entry_date'].between(ctx['start'],ctx['end'])
                    & signals['exit_date'].between(ctx['start'],ctx['end'])
                ].copy()
                if not signals.empty:
                    signals['signal_score']=(
                        110.0*signals['robust_edge']+0.035*signals['quality_rank']
                        +0.012*signals['rank_rel20'].fillna(0.0)+0.008*signals['rank_adv20'].fillna(0.0)
                        +0.006*signals['close_loc'].fillna(0.0)
                    )
                    signals=signals.sort_values(['date','code','signal_score'],ascending=[True,True,False]).drop_duplicates(['date','code'])
            signal_cache[key]=(signals.reset_index(drop=True),selected)
        return signal_cache[key]

    print('V6 STEP 3: anchored walk-forward config selection on 2022-2024')
    grid_rows=[]
    best_cfg=None;best_score=-1e18;best_year_metrics=None
    for i,cfg in enumerate(configs,1):
        yearly=[]
        for year in development_years:
            signals,_=get_signals(year,cfg)
            ctx=year_context[year]
            m,_,_=run_v6_portfolio(signals,ctx['index_signals'],features,etfs,bench,ctx['start'],ctx['end'],cfg)
            yearly.append(m)
        score=score_v6_development(yearly)
        gate=_v6_dev_gate(yearly)
        compound=float(np.prod([1+m['return'] for m in yearly])-1.0)
        grid_rows.append({
            'config_id':cfg.config_id,'score':score,'compound_return':compound,
            'positive_years':sum(m['return']>0 for m in yearly),'worst_year':min(m['return'] for m in yearly),
            'total_trades':sum(m['trades'] for m in yearly),'median_pf':float(np.median([m['profit_factor'] for m in yearly])),
            'all_dev_gates':all(gate.values()),
            **{f'return_{y}':yearly[j]['return'] for j,y in enumerate(development_years)},
            **{f'pf_{y}':yearly[j]['profit_factor'] for j,y in enumerate(development_years)},
            **{f'mdd_{y}':yearly[j]['max_drawdown'] for j,y in enumerate(development_years)},
        })
        # Prefer passing configurations; if none pass, retain the highest continuous score for diagnosis.
        adjusted=score+(100.0 if all(gate.values()) else 0.0)
        if adjusted>best_score:
            best_score=adjusted;best_cfg=cfg;best_year_metrics=yearly
        if i%20==0:
            print('configs',i,'/',len(configs))
    if best_cfg is None or best_year_metrics is None:
        raise RuntimeError('V6: no configuration selected')
    grid_df=pd.DataFrame(grid_rows).sort_values(['all_dev_gates','score'],ascending=[False,False])
    grid_df.to_csv(OUT/'development_grid_2022_2024.csv',index=False)
    dev_gate=_v6_dev_gate(best_year_metrics)
    print('selected',best_cfg.config_id,'dev gate',dev_gate)

    print('V6 STEP 4: fixed-rule 2025 validation')
    sig25,profiles25=get_signals(2025,best_cfg)
    ctx25=year_context[2025]
    m25,tr25,eq25=run_v6_portfolio(sig25,ctx25['index_signals'],features,etfs,bench,ctx25['start'],ctx25['end'],best_cfg)
    s25,_,_=run_v6_portfolio(sig25,ctx25['index_signals'],features,etfs,bench,ctx25['start'],ctx25['end'],best_cfg,cost_mult=1.5)
    g25=_v6_validation_gate(m25,s25)

    print('V6 STEP 5: unchanged architecture 2026 Jan-Aug stress')
    sig26,profiles26=get_signals(2026,best_cfg)
    ctx26=year_context[2026]
    m26,tr26,eq26=run_v6_portfolio(sig26,ctx26['index_signals'],features,etfs,bench,ctx26['start'],ctx26['end'],best_cfg)
    s26,_,_=run_v6_portfolio(sig26,ctx26['index_signals'],features,etfs,bench,ctx26['start'],ctx26['end'],best_cfg,cost_mult=1.5)
    g26=_v6_stress_gate(m26,s26)
    target=_v6_user_target(m26)

    accepted=bool(all(dev_gate.values()) and all(g25.values()) and all(g26.values()) and all(target.values()))
    result={
        'version':'krx-v6-empirical-walk-forward-multi-profile',
        'status':'ACCEPTED' if accepted else 'REJECTED',
        'accepted':accepted,
        'user_target_met':bool(all(target.values())),
        'integrity':{
            'configuration_selected_from':'anchored 2022, 2023, 2024 only',
            '2025_used_for_config_selection':False,
            '2026_used_for_config_selection':False,
            'profiles_updated_annually_using_only_prior completed exits':True,
            'signal_to_fill':'signal close t -> entry open t+1',
            'same_bar_ambiguity':'stop first',
            'integer_shares':True,'long_only':True,'leverage':False,'max_holding_sessions':3,
            '2026_warning':'2026 is a repeatedly inspected stress interval in this conversation, not pristine unseen OOS',
        },
        'data_quality':{'download':download_quality,'adjustment':adjustment_quality},
        'candidate_rows':int(len(candidates)),
        'base_signal_rows':int(candidates[['date','code','pattern']].drop_duplicates().shape[0]),
        'selected_config':_v6_asdict(best_cfg),'selected_config_id':best_cfg.config_id,
        'development_2022_2024':{
            'year_metrics':{str(y):best_year_metrics[i] for i,y in enumerate(development_years)},
            'gates':dev_gate,'passed':all(dev_gate.values()),
        },
        'validation_2025':{'metrics':m25,'cost_stress_1_5x':s25,'gates':g25,'passed':all(g25.values())},
        'stress_2026_01_08_21':{'metrics':m26,'cost_stress_1_5x':s26,'gates':g26,'passed':all(g26.values())},
        'user_monthly_target':{'target':0.20,'gates':target,'passed':all(target.values())},
        'selected_profile_count_2025':int(len(profiles25)),
        'selected_profile_count_2026':int(len(profiles26)),
        'limitations':[
            'The mapping adds only securities identifiable from currently available public listings; residual survivorship bias can remain.',
            'Split adjustment is algorithmic and not an exchange-certified total-return adjustment factor.',
            'No complete timestamped news archive is used; price/volume reaction is a catalyst proxy.',
            'ETF sleeve uses KODEX 200 and KODEX KOSDAQ150 historical OHLC, with explicit non-stock transaction costs.',
        ],
    }
    (OUT/'final_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    profiles25.to_csv(OUT/'selected_profiles_2025.csv',index=False)
    profiles26.to_csv(OUT/'selected_profiles_2026.csv',index=False)
    sig25[['date','entry_date','exit_date','market','code','name','pattern','regime','execution_id','quality_rank','robust_edge','pf','n','signal_score']].to_csv(OUT/'eligible_signals_2025.csv',index=False)
    sig26[['date','entry_date','exit_date','market','code','name','pattern','regime','execution_id','quality_rank','robust_edge','pf','n','signal_score']].to_csv(OUT/'eligible_signals_2026.csv',index=False)
    tr25.to_csv(OUT/'trades_2025.csv',index=False);eq25.to_csv(OUT/'equity_2025.csv',index=False)
    tr26.to_csv(OUT/'trades_2026.csv',index=False);eq26.to_csv(OUT/'equity_2026.csv',index=False)
    # Compact audit CSV is always provided. Pine is deliberately withheld unless every gate, including the user's target, passes.
    candidates[['date','entry_date','exit_date','market','code','name','pattern','regime','execution_id','quality_rank','net_return','mae','mfe']].to_parquet(OUT/'candidate_audit_v6.parquet',index=False)
    if accepted:
        create_pine_template(OUT/'KRX_v6_selected_stock_execution.txt')
    print('===KRX_V6_RESULT_BEGIN===')
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('===KRX_V6_RESULT_END===')


if __name__=='__main__':
    main()
