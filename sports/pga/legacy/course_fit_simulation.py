from __future__ import annotations
import numpy as np
import pandas as pd


def _z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors='coerce')
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((x-x.mean())/sd).fillna(0.0)


def resimulate_with_course_fit(mc: pd.DataFrame, fit: pd.DataFrame, sims: int=50000, seed: int=42,
                               max_round_shift: float=0.45, fit_strength: float=0.22) -> pd.DataFrame:
    """Paired Monte Carlo repricing for V10.2.

    Reconstructs a neutral scoring distribution from V10's expected four-round score,
    then applies a bounded course-fit shift to round mean plus a small upside variance
    adjustment. It is intended for controlled comparison when the original raw feature
    pipeline is unavailable in the deployed app. Future full-pipeline simulations should
    apply the same adjustment inside pga_predictor_pro.simulate().
    """
    d=mc.merge(fit[['player','course_fit_ceiling','course_fit_coverage']],on='player',how='left')
    n=len(d); rng=np.random.default_rng(seed)
    fit_z=_z(d['course_fit_ceiling']).to_numpy(float)
    coverage=pd.to_numeric(d['course_fit_coverage'],errors='coerce').fillna(0).clip(0,1).to_numpy(float)
    # Better fit lowers expected strokes. Bound the mean adjustment to avoid hindsight forcing.
    round_shift=np.clip(-fit_strength*fit_z*coverage,-max_round_shift,max_round_shift)
    base4=pd.to_numeric(d.get('expected_4r_to_par_if_made_cut'),errors='coerce')
    if base4.isna().all():
        # Fallback rank-derived center if a future uploaded simulation lacks score expectation.
        base4=(_z(pd.to_numeric(d['expected_finish'],errors='coerce'))*4.0)
    mu=(base4.fillna(base4.median()).to_numpy(float)/4.0)+round_shift
    # Conservative dispersion estimate; strong positive fit gets only a small extra upper-tail range.
    sd=np.full(n,2.75)
    sd_mult=np.clip(1.0+0.025*np.maximum(fit_z,0)*coverage,1.0,1.06)
    sd=sd*sd_mult
    course_shock=rng.normal(0,.35,size=(sims,1,4))
    latent=rng.normal(0,1,size=(sims,n,1))
    noise=rng.normal(0,1,size=(sims,n,4))
    rounds=mu[None,:,None]+.45*sd[None,:,None]*latent+.89*sd[None,:,None]*noise+course_shock
    after2=rounds[:,:,:2].sum(2); cut_size=max(1,n//2)
    cutoff=np.partition(after2,cut_size-1,axis=1)[:,cut_size-1]
    made=after2<=cutoff[:,None]
    final=rounds.sum(2); scored=np.where(made,final,final+50)
    order=np.argsort(scored+rng.normal(0,1e-6,scored.shape),axis=1)
    ranks=np.empty_like(order); ranks[np.arange(sims)[:,None],order]=np.arange(1,n+1)[None,:]
    out=d.copy()
    out['win_pct']=(ranks==1).mean(0)*100
    out['top5_pct']=(ranks<=5).mean(0)*100
    out['top10_pct']=(ranks<=10).mean(0)*100
    out['top20_pct']=(ranks<=20).mean(0)*100
    out['make_cut_pct']=made.mean(0)*100
    out['expected_finish']=ranks.mean(0)
    out['expected_4r_to_par_if_made_cut']=final.mean(0)
    # Preserve V10 proxy scale, but use newly simulated probabilities (percent units).
    out['dk_points_proxy']=(8*out.make_cut_pct/100+4*(ranks<=30).mean(0)+6*out.top20_pct/100+8*out.top10_pct/100+8*out.top5_pct/100+10*out.win_pct/100)
    out['course_fit_round_shift']=round_shift
    out['simulation_version']='V10.2 Course-Fit Monte Carlo'
    return out


def _quantile_calibrate(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Map repriced ranks onto the frozen baseline marginal distribution.

    This preserves the baseline model's absolute scale/distribution while allowing
    course fit to change relative ordering. No actual tournament results are used.
    """
    x = pd.to_numeric(values, errors='coerce')
    ref = pd.to_numeric(reference, errors='coerce').dropna().sort_values().to_numpy(float)
    if len(ref) == 0:
        return x
    pct = x.rank(method='average', pct=True).fillna(.5).to_numpy(float)
    idx = np.clip(np.rint(pct * (len(ref)-1)).astype(int), 0, len(ref)-1)
    return pd.Series(ref[idx], index=values.index)


def resimulate_with_course_fit_calibrated(mc: pd.DataFrame, fit: pd.DataFrame, sims: int=50000,
                                           seed: int=42, max_round_shift: float=0.45,
                                           fit_strength: float=0.22) -> pd.DataFrame:
    """V10.2c: upstream course-fit repricing with baseline-scale calibration.

    Step 1 runs the bounded course-fit Monte Carlo. Step 2 rank/quantile maps each
    simulated output back to the frozen V10 marginal distribution. This retains
    course-fit-driven relative movement while preventing a synthetic simulation
    from changing the field-wide probability or DK-proxy scale.
    """
    raw = resimulate_with_course_fit(mc, fit, sims=sims, seed=seed,
                                     max_round_shift=max_round_shift,
                                     fit_strength=fit_strength)
    out = raw.copy()
    # Higher-is-better probability / fantasy metrics.
    for col in ['win_pct','top5_pct','top10_pct','top20_pct','make_cut_pct','dk_points_proxy']:
        if col in raw.columns and col in mc.columns:
            out[col] = _quantile_calibrate(raw[col], mc[col])
    # Lower-is-better finish/score metrics: ordinary ascending quantiles already
    # preserve the correct ordering because low raw values map to low reference values.
    for col in ['expected_finish','expected_4r_to_par_if_made_cut']:
        if col in raw.columns and col in mc.columns:
            out[col] = _quantile_calibrate(raw[col], mc[col])
    out['simulation_version'] = 'V10.2c Calibrated Course-Fit Monte Carlo'
    out['calibration_method'] = 'baseline marginal quantile mapping'
    return out
