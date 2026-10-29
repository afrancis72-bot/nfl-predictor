
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

FINISH_POINTS = {
1:45,2:42,3:41,4:40,5:39,6:38,7:37,8:36,9:35,10:34,
11:32,12:31,13:30,14:29,15:28,16:27,17:26,18:25,19:24,20:23,
21:21,22:20,23:19,24:18,25:17,26:16,27:15,28:14,29:13,30:12,
31:10,32:9,33:8,34:7,35:6,36:5,37:4,38:3,39:2,40:1
}

@dataclass
class Config:
    sims: int = 100000
    seed: int = 42
    salary_cap: int = 50000
    roster_size: int = 6
    laps: int = 267
    max_overlap: int = 4

def _z(s, higher_better=True):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=s.index)
    fill = s.median()
    x = s.fillna(fill)
    sd = x.std(ddof=0)
    z = (x-x.mean())/(sd if sd > 1e-9 else 1.0)
    return z if higher_better else -z

def _rank_signal(df):
    cols = ["avg20_rank","avg15_rank","avg10_rank","avg5_rank","single_lap_rank"]
    weights = [0.30,0.25,0.22,0.15,0.08]
    vals = []
    wts = []
    for c,w in zip(cols,weights):
        if c in df:
            # Missing ranks are neutral. Do not let median-filled z-scores leak into
            # the numerator when a driver did not run that practice length.
            observed = df[c].notna()
            component = _z(df[c], higher_better=False).where(observed, 0.0)
            vals.append(component*w)
            wts.append(observed.astype(float)*w)
    if not vals:
        return pd.Series(0.0,index=df.index)
    num = sum(vals)
    den = sum(wts).replace(0,np.nan)
    return (num/den).fillna(0)


def _safe_signal(s, higher_is_better=True, min_obs=6, cap=2.25):
    x=pd.to_numeric(s,errors="coerce")
    obs=x.notna()
    n=int(obs.sum())
    out=pd.Series(0.0,index=x.index,dtype=float)
    if n < min_obs:
        return out
    vals=x[obs]
    sd=float(vals.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-9:
        return out
    z=(vals-float(vals.mean()))/sd
    if not higher_is_better:
        z=-z
    reliability=min(1.0, n/max(min_obs*2,12))
    out.loc[obs]=z.clip(-cap,cap)*reliability
    return out

def construct_track_dna(track):
    """Translate one track-profile row into an explainable race archetype."""
    w = track.iloc[0]
    def g(name, default):
        try:
            v = getattr(w, name)
            return float(v) if pd.notna(v) else float(default)
        except Exception:
            return float(default)

    length = g("track_length", 1.5)
    banking = g("banking", 18)
    tire = g("tire_wear", 0.55)
    passing = g("passing_difficulty", 0.50)
    volatility = g("volatility", 0.95)
    restart = g("restart_volatility", 0.55)
    pit = g("pit_importance", 0.55)

    if length >= 2.0 and banking >= 25:
        archetype = "superspeedway"
    elif length <= 1.1:
        archetype = "short_track"
    elif g("road_course", 0) >= 0.5:
        archetype = "road_course"
    elif 1.3 <= length <= 1.7:
        archetype = "intermediate"
    else:
        archetype = "oval"

    dna = {
        "track": str(getattr(w, "track", "Current race")),
        "archetype": archetype,
        "long_run_speed": min(1.0, 0.45 + 0.40*tire),
        "short_run_speed": min(1.0, 0.45 + 0.35*restart),
        "track_position": min(1.0, 0.35 + 0.45*passing),
        "place_differential": max(0.20, 0.75 - 0.45*passing),
        "dominator": min(1.0, 0.45 + 0.25*(1-passing) + 0.20*(1/ max(volatility,0.55))),
        "pit_execution": min(1.0, 0.30 + 0.55*pit),
        "incident_variance": min(1.0, 0.35 + 0.35*volatility + 0.20*restart),
        "tire_management": min(1.0, 0.25 + 0.65*tire),
    }
    return dna


def _first_signal(df, candidates, higher_better=True, min_obs=6):
    """Use the first available column from a generic/legacy alias list."""
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= min_obs:
            return _safe_signal(df[c], higher_better, min_obs=min_obs)
    return pd.Series(0.0, index=df.index, dtype=float)


def build_features(dk, priors, track, updates=None):
    """
    V2.0 race-agnostic feature builder.
    Generic weekly columns are preferred; legacy Las Vegas columns remain aliases
    so old data can still be inspected without making Las Vegas part of the model.
    """
    df = dk.copy()
    df = df.merge(priors.drop(columns=["Salary","AvgPointsPerGame"],errors="ignore"), on="Name", how="left")
    if updates is not None:
        df = df.merge(updates, on="Name", how="left")
    for c in ["single_lap_rank","avg5_rank","avg10_rank","avg15_rank","avg20_rank","qualifying_position"]:
        if c not in df.columns:
            df[c] = np.nan

    if "AvgPointsPerGame" not in df.columns:
        df["AvgPointsPerGame"] = pd.to_numeric(df.get("avg_points", 0), errors="coerce").fillna(0)
    if "manual_speed_rating" not in df.columns:
        df["manual_speed_rating"] = np.nan

    df["dk_base_z"] = _z(df["AvgPointsPerGame"], True)
    df["season_finish_z"] = _first_signal(df, ["season_avg_finish","season_2026_avg_finish"], False)
    df["season_led_z"] = _first_signal(df, ["season_laps_led","season_2026_laps_led"], True)
    df["season_speed_z"] = _first_signal(df, ["season_speed_rank"], False)
    df["recent_rating_z"] = _first_signal(df, ["recent_driver_rating","chase_driver_rating"], True)
    df["recent_finish_z"] = _first_signal(df, ["recent_avg_finish","chase_avg_finish"], False)

    df["track_type_finish_z"] = _first_signal(
        df, ["track_type_avg_finish","intermediate_2026_avg_finish"], False)
    df["track_type_led_z"] = _first_signal(
        df, ["track_type_laps_led","intermediate_2026_laps_led","nextgen_1p5_laps_led"], True)
    df["track_history_finish_z"] = _first_signal(
        df, ["track_history_avg_finish","vegas_nextgen_avg_finish","vegas_career_avg_finish","vegas_spring_2026_finish"], False)
    df["track_history_led_z"] = _first_signal(
        df, ["track_history_laps_led","vegas_nextgen_laps_led","vegas_career_laps_led","vegas_spring_2026_laps_led"], True)

    df["manual_speed_z"] = _z(df["manual_speed_rating"], True)
    df["practice_z"] = _rank_signal(df)

    # V2.9: NASCAR Reference DNA is now upstream of the race model, not display-only.
    # Component scores are cross-field percentiles (0-100), converted to z-scores.
    # V2.12 COMPONENT WIRING FIX
    # These scored columns are proven present at the app -> build_features boundary.
    # Normalize them DIRECTLY instead of routing them through _first_signal(), whose
    # availability rules were treating valid scored percentiles as unavailable.
    def _direct_scored_z(col, higher_is_better=True):
        if col not in df.columns:
            return pd.Series(0.0,index=df.index)
        raw=pd.to_numeric(df[col],errors="coerce")
        if raw.notna().sum()<2 or raw.nunique(dropna=True)<2:
            return pd.Series(0.0,index=df.index)
        z=_z(raw,higher_is_better)
        return pd.to_numeric(z,errors="coerce").fillna(0.0)

    df["reference_dna_z"]=_direct_scored_z("driver_dna_score",True)
    df["reference_form_z"]=_direct_scored_z("recent_form_score",True)
    df["reference_pd_z"]=_direct_scored_z("place_diff_score",True)
    df["reference_track_z"]=_direct_scored_z("comparable_track_score",True)
    df["reference_elo_z"]=_direct_scored_z("track_type_elo_score",True)
    df["reference_reliability_z"]=_direct_scored_z("reliability_score",True)

    # Observed-data fallback only when a scored component genuinely has no spread.
    if df["reference_form_z"].abs().sum()<1e-9:
        df["reference_form_z"]=_direct_scored_z("recent_avg_finish",False)
    if df["reference_pd_z"].abs().sum()<1e-9:
        df["reference_pd_z"]=_direct_scored_z("recent_place_diff",True)
    if df["reference_track_z"].abs().sum()<1e-9:
        df["reference_track_z"]=_direct_scored_z("comparable_track_avg_finish",False)
    if df["reference_elo_z"].abs().sum()<1e-9:
        df["reference_elo_z"]=_direct_scored_z("superspeedway_elo",True)
    if df["reference_reliability_z"].abs().sum()<1e-9:
        df["reference_reliability_z"]=_direct_scored_z("season_avg_finish",False)

    # Broad race strength: current-season results + recent form + reliability + DK prior.
    # Missing Reference signals are neutral (0 z), never converted to poor performance.
    current = (0.22*df["dk_base_z"].clip(-2,2) +
               0.26*df["season_finish_z"].clip(-2,2) +
               0.22*df["recent_finish_z"].clip(-2,2) +
               0.16*df["reference_form_z"].clip(-2,2) +
               0.08*df["reference_reliability_z"].clip(-2,2) +
               0.06*df["reference_pd_z"].clip(-2,2))

    # Track-type fit is explicitly driven by comparable-track average finish and
    # track-type Elo. Legacy laps-led inputs can supplement when a trusted file exists.
    track_type = (0.48*df["track_type_finish_z"].clip(-2,2) +
                  0.27*df["reference_track_z"].clip(-2,2) +
                  0.20*df["reference_elo_z"].clip(-2,2) +
                  0.05*df["track_type_led_z"].clip(-2,2))
    track_history = (0.70*df["track_history_finish_z"].clip(-2,2) +
                     0.30*df["track_history_led_z"].clip(-2,2))

    practice_available = df[["single_lap_rank","avg5_rank","avg10_rank","avg15_rank","avg20_rank"]].notna().any(axis=1)
    qual_available = df["qualifying_position"].notna()
    qual_z = _z(df["qualifying_position"], False) if qual_available.any() else pd.Series(0.0,index=df.index)

    w = track.iloc[0]
    dna = construct_track_dna(track)
    # Track DNA changes the balance between broad ability, track-type fit and venue history.
    hist_w = 0.10 + 0.08*dna["track_position"]
    type_w = 0.20 + 0.10*dna["long_run_speed"]
    base_w = max(0.35, 1.0-hist_w-type_w)
    raw = base_w*current + type_w*track_type + hist_w*track_history + 0.10*df["dk_base_z"]

    practice_weight = float(getattr(w,"weight_practice",0.08))
    qualifying_weight = float(getattr(w,"weight_qualifying",0.06))
    if practice_available.any():
        raw += practice_weight*df["practice_z"]
    else:
        raw += practice_weight*(0.55*current+0.45*track_type)
    if qual_available.any():
        raw += qualifying_weight*qual_z
    else:
        raw += qualifying_weight*(0.60*current+0.40*track_history)

    df["race_strength"] = raw
    df["track_volatility"] = float(getattr(w, "volatility", 0.95))
    df["projected_start"] = df["qualifying_position"].fillna((len(df)+1)/2)
    start_front = _z(df["projected_start"], False)

    ds = float(getattr(w,"dominator_weight_speed",0.35))
    dt = float(getattr(w,"dominator_weight_intermediate",0.30))
    dh = float(getattr(w,"dominator_weight_vegas",0.20))
    dstart = float(getattr(w,"dominator_weight_start",0.15))
    df["dominator_strength"] = (
        ds*(0.65*current+0.35*df["practice_z"]) +
        dt*(0.55*track_type+0.45*df["track_type_led_z"]) +
        dh*(0.55*track_history+0.45*df["season_led_z"]) +
        dstart*start_front
    )
    if qual_available.any():
        df["dominator_strength"] += (0.12 + 0.16*dna["track_position"])*start_front

    df["place_diff_room"] = df["projected_start"] - 1
    df["front_start_score"] = ((10-df["projected_start"]).clip(lower=0)/9.0)
    df["back_start_score"] = ((df["projected_start"]-20).clip(lower=0)/max(len(df)-20,1))

    evidence = [
        "season_avg_finish","season_2026_avg_finish","season_laps_led","season_2026_laps_led",
        "season_speed_rank","recent_driver_rating","chase_driver_rating","track_type_avg_finish",
        "intermediate_2026_avg_finish","track_history_avg_finish","vegas_nextgen_avg_finish"
    ]
    present=[c for c in evidence if c in df.columns]
    base_cov=df[present].notna().sum(axis=1) if present else pd.Series(0,index=df.index)
    df["data_coverage"]=base_cov + practice_available.astype(int)*2 + qual_available.astype(int)
    df["model_confidence"]=(0.45+0.045*df["data_coverage"]).clip(upper=0.95)

    df["audit_dk_prior"]=df["dk_base_z"].clip(-2,2)
    df["audit_track_history"]=track_history
    df["audit_current"]=current
    df["audit_track_type"]=track_type
    df["audit_reference_dna"]=df["reference_dna_z"]
    df["audit_reference_form"]=df["reference_form_z"]
    df["audit_reference_pd"]=df["reference_pd_z"]
    df["audit_reference_track"]=df["reference_track_z"]
    df["audit_reference_elo"]=df["reference_elo_z"]
    df["audit_reference_reliability"]=df["reference_reliability_z"]
    # Legacy aliases retained only for older UI/code compatibility.
    df["audit_vegas"]=track_history
    df["audit_intermediate"]=track_type
    df["track_dna_fit"]=(0.45*track_type + 0.25*track_history + 0.30*current)
    return df

def simulate(features, config=Config()):
    rng = np.random.default_rng(config.seed)
    n = len(features); sims = config.sims
    strength = features["race_strength"].to_numpy(float)
    dom = features["dominator_strength"].to_numpy(float)
    starts = features["projected_start"].to_numpy(float)

    # V2.14 confidence calibration: low-coverage drivers previously received extra
    # variance without enough shrinkage of their underlying strength. For deep
    # starters that could manufacture unrealistic PD/ceiling tails. Blend uncertain
    # strength toward the observable DK-market prior and shrink dominator strength
    # toward that conservative race-strength estimate. High-confidence drivers are
    # effectively unchanged.
    confidence = pd.to_numeric(features.get("model_confidence", 0.45), errors="coerce").fillna(0.45).clip(0.35,0.95).to_numpy(float)
    dk_anchor = pd.to_numeric(features.get("audit_dk_prior", 0.0), errors="coerce").fillna(0.0).clip(-2.5,2.5).to_numpy(float)
    shrink = np.clip((0.80-confidence)/0.45, 0.0, 1.0)
    strength = (1.0-0.55*shrink)*strength + (0.55*shrink)*dk_anchor
    dom = (1.0-0.65*shrink)*dom + (0.65*shrink)*np.minimum(strength, dom)
    dnf = pd.to_numeric(features.get("manual_dnf_risk", pd.Series(np.nan,index=features.index)), errors="coerce").fillna(0.11).clip(0.02,0.35).to_numpy()

    finish = np.empty((sims,n), dtype=np.int16)
    laps_led = np.zeros((sims,n), dtype=np.int16)
    fastest = np.zeros((sims,n), dtype=np.int16)
    dkpts = np.zeros((sims,n), dtype=np.float32)

    # Simulation: latent performance + incident tail. V1 intentionally transparent.
    for s in range(sims):
        # Keep uncertainty for thin-data drivers, but cap the variance bonus so
        # uncertainty does not itself become a DFS ceiling advantage.
        sigma = np.minimum(1.85 - 0.65*confidence, 1.48)
        # V1.4.1: modest track-position anchor. Starting position is predictive of finish,
        # so front starters retain some track-position value and deep starters must earn PD.
        # This changes finish probabilities, not DraftKings scoring.
        start_anchor = -((starts - starts.mean()) / (starts.std() if starts.std() > 1e-9 else 1.0))
        latent = strength + 0.22*start_anchor + rng.normal(0, sigma, n)
        incidents = rng.random(n) < dnf
        latent[incidents] -= rng.uniform(2.5,5.5,incidents.sum())
        order = np.argsort(-latent)
        f = np.empty(n,dtype=np.int16); f[order]=np.arange(1,n+1)
        finish[s]=f

        # Dominator events need race-to-race concentration variance. A fixed softmax
        # followed by a multinomial has the right mean but an unrealistically thin
        # right tail for laps led. Draw a race-level Dirichlet share around the same
        # pre-race expectation, then allocate laps from that share. This is symmetric:
        # no driver is named or boosted, and the expected share remains driven by
        # pre-race speed / practice / history / starting position.
        dscore = 0.52*dom + 0.48*latent
        p = np.exp(dscore - dscore.max()); p /= p.sum()
        track_vol = float(features["track_volatility"].iloc[0]) if "track_volatility" in features else 0.95
        concentration = 28.0 / max(track_vol, 0.35)
        alpha = np.maximum(p * concentration, 0.015)
        race_share = rng.dirichlet(alpha)
        ll = rng.multinomial(config.laps, race_share)
        fast_share = np.sqrt(race_share); fast_share /= fast_share.sum()
        flp = rng.multinomial(config.laps, fast_share)
        laps_led[s]=ll; fastest[s]=flp

        fp = np.array([FINISH_POINTS.get(int(x),1) for x in f])
        pdiff = starts - f
        dkpts[s] = fp + pdiff + 0.25*ll + 0.45*flp

    out = features[["Name","ID","Salary","AvgPointsPerGame","race_strength","dominator_strength",
                    "data_coverage","model_confidence","projected_start","place_diff_room",
                    "front_start_score","back_start_score","audit_dk_prior","audit_vegas","audit_current","audit_intermediate",
                    "audit_reference_dna","audit_reference_form","audit_reference_pd","audit_reference_track",
                    "audit_reference_elo","audit_reference_reliability"]].copy()
    out["proj_dk"] = dkpts.mean(axis=0)
    out["floor_p20"] = np.quantile(dkpts,0.20,axis=0)
    out["median_p50"] = np.quantile(dkpts,0.50,axis=0)
    out["ceiling_p90"] = np.quantile(dkpts,0.90,axis=0)
    out["ceiling_p95"] = np.quantile(dkpts,0.95,axis=0)
    out["ceiling_p99"] = np.quantile(dkpts,0.99,axis=0)
    top6 = np.argsort(dkpts,axis=1)[:,-min(6,n):]
    top6_hits = np.zeros(n,dtype=float)
    for j in range(n): top6_hits[j] = np.mean(np.any(top6==j,axis=1))
    out["sim_top6_pct"] = top6_hits
    out["win_pct"] = (finish==1).mean(axis=0)
    out["top5_pct"] = (finish<=5).mean(axis=0)
    out["top10_pct"] = (finish<=10).mean(axis=0)
    out["exp_finish"] = finish.mean(axis=0)
    out["exp_laps_led"] = laps_led.mean(axis=0)
    out["exp_fastest_laps"] = fastest.mean(axis=0)
    out["value_per_1k"] = out["proj_dk"]/(out["Salary"]/1000)
    # The optimizer indexes the simulation matrix by projection-row order.
    # Sorting projections without sorting columns silently attributes one
    # driver's simulation outcome to another driver's salary/identity.
    order = np.argsort(-out["proj_dk"].to_numpy(float), kind="stable")
    return out.iloc[order].reset_index(drop=True), dkpts[:, order]

def optimize_lineups(proj, n_lineups=10, salary_cap=50000, roster_size=6, max_overlap=4, objective="ceiling", max_exposure=1.0):
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except Exception as e:
        raise RuntimeError("SciPy with scipy.optimize.milp is required for lineup optimization.") from e
    score_col = {"median":"proj_dk","ceiling":"ceiling_p90","value":"value_per_1k"}.get(objective,"ceiling_p90")
    scores = proj[score_col].to_numpy(float)
    sal = proj["Salary"].to_numpy(float)
    n=len(proj); lineups=[]
    max_count=max(1, int(np.floor(n_lineups*max_exposure + 1e-9)))
    # V2.14: driver-specific exposure protection for thin/low-confidence data.
    # User max exposure remains the absolute ceiling; uncertain drivers get a
    # lower cap automatically instead of being allowed to hit that ceiling.
    conf=pd.to_numeric(proj.get("model_confidence", pd.Series(0.90,index=proj.index)),errors="coerce").fillna(0.45).to_numpy(float)
    cov=pd.to_numeric(proj.get("data_coverage", pd.Series(1.0,index=proj.index)),errors="coerce").fillna(0.15).to_numpy(float)
    driver_caps=np.full(n,max_count,dtype=int)
    low=(conf<0.55)|(cov<0.35)
    med=((conf<0.75)|(cov<0.60)) & ~low
    driver_caps[low]=np.minimum(driver_caps[low],max(1,int(np.floor(n_lineups*0.30+1e-9))))
    driver_caps[med]=np.minimum(driver_caps[med],max(1,int(np.floor(n_lineups*0.45+1e-9))))
    exposure_counts=np.zeros(n,dtype=int)
    constraints=[LinearConstraint(np.ones((1,n)),[roster_size],[roster_size]),
                 LinearConstraint(sal.reshape(1,-1),[-np.inf],[salary_cap])]
    for _ in range(n_lineups):
        cons=list(constraints)
        capped=np.where(exposure_counts>=driver_caps)[0]
        for i in capped:
            a=np.zeros(n); a[i]=1
            cons.append(LinearConstraint(a.reshape(1,-1),[0],[0]))
        for prev in lineups:
            a=np.zeros(n); a[prev]=1
            cons.append(LinearConstraint(a.reshape(1,-1),[-np.inf],[max_overlap]))
        res=milp(c=-scores, integrality=np.ones(n), bounds=Bounds(np.zeros(n),np.ones(n)), constraints=cons)
        if not res.success: break
        idx=np.where(res.x>0.5)[0].tolist()
        lineups.append(idx)
        exposure_counts[idx]+=1
    rows=[]
    for k,idx in enumerate(lineups,1):
        sub=proj.iloc[idx]
        rows.append({
            "Lineup":k,
            "Drivers":" | ".join(sub["Name"]),
            "Salary":int(sub["Salary"].sum()),
            "Projected":round(sub["proj_dk"].sum(),2),
            "CeilingScore":round(sub["ceiling_p90"].sum(),2),
            "IDs":" | ".join(sub["ID"].astype(str))
        })
    return pd.DataFrame(rows)


def optimize_race_sim_lineups(proj, sim_matrix, n_lineups=10, salary_cap=50000,
                              roster_size=6, max_overlap=4, candidate_pool=2500,
                              seed=42, risk_mode="gpp"):
    """
    V1.1 optimizer. Generates legal candidate rosters, then evaluates each roster by
    summing its six drivers inside EACH SAME simulated race. This preserves NASCAR
    race-level correlation and makes P90/P95 true lineup outcomes rather than sums
    of individual percentiles.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    rng=np.random.default_rng(seed)
    n=len(proj)
    salary=proj["Salary"].to_numpy(float)
    # Candidate generation: perturb mean projection to create diverse, strong legal rosters.
    base=proj["proj_dk"].to_numpy(float)
    candidates=set()
    attempts=max(candidate_pool*3,3000)
    for attempt in range(attempts):
        # Candidate diversity must come from coherent simulated race outcomes as
        # well as independent projection noise. Otherwise a front-running driver
        # with a lower median can have a genuine dominator tail yet never enter
        # the candidate bank. Sample the actual race matrix on 40% of attempts;
        # this is driver-agnostic and uses only pre-race simulation information.
        if attempt % 5 in (0, 1):
            score = sim_matrix[int(rng.integers(0, len(sim_matrix)))]
        else:
            noise=rng.normal(0,7.5,n)
            score=base+noise
        cons=[
            LinearConstraint(np.ones((1,n)),[roster_size],[roster_size]),
            LinearConstraint(salary.reshape(1,-1),[-np.inf],[salary_cap])
        ]
        res=milp(c=-score,integrality=np.ones(n),
                 bounds=Bounds(np.zeros(n),np.ones(n)),constraints=cons)
        if res.success:
            idx=tuple(np.where(res.x>0.5)[0].tolist())
            candidates.add(idx)
        if len(candidates)>=candidate_pool: break

    # Always include mean-optimal lineup.
    cons=[LinearConstraint(np.ones((1,n)),[roster_size],[roster_size]),
          LinearConstraint(salary.reshape(1,-1),[-np.inf],[salary_cap])]
    res=milp(c=-base,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=cons)
    if res.success:candidates.add(tuple(np.where(res.x>0.5)[0].tolist()))

    rows=[]
    for idx in candidates:
        scores=sim_matrix[:,idx].sum(axis=1)
        mean=float(scores.mean()); p10=float(np.quantile(scores,.10))
        p90=float(np.quantile(scores,.90)); p95=float(np.quantile(scores,.95))
        # GPP objective rewards coherent upper-tail outcomes without ignoring mean.
        gpp=0.40*mean+0.25*p90+0.35*p95
        cash=0.65*mean+0.35*p10
        rows.append((idx,mean,p10,p90,p95,gpp,cash))
    rows.sort(key=lambda x:x[5] if risk_mode=="gpp" else x[6],reverse=True)

    selected=[]
    for r in rows:
        idx=r[0]
        if all(len(set(idx)&set(s[0]))<=max_overlap for s in selected):
            selected.append(r)
            if len(selected)>=n_lineups: break

    out=[]
    for k,(idx,mean,p10,p90,p95,gpp,cash) in enumerate(selected,1):
        sub=proj.iloc[list(idx)]
        out.append({
            "Lineup":k,
            "Drivers":" | ".join(sub["Name"]),
            "Salary":int(sub["Salary"].sum()),
            "Mean":round(mean,2),"P10":round(p10,2),
            "P90":round(p90,2),"P95":round(p95,2),
            "GPP_Score":round(gpp,2),
            "IDs":" | ".join(sub["ID"].astype(str))
        })
    return pd.DataFrame(out)


def portfolio_exposure(lineups, proj):
    if lineups is None or len(lineups)==0:
        return pd.DataFrame()
    counts={}
    for s in lineups["Drivers"]:
        for name in [x.strip() for x in str(s).split("|")]:
            counts[name]=counts.get(name,0)+1
    n=len(lineups)
    out=pd.DataFrame({"Name":list(counts.keys()),"Lineups":list(counts.values())})
    out["Exposure"]=out["Lineups"]/n
    return out.merge(proj[["Name","Salary","proj_dk","ceiling_p90","model_confidence"]],
                     on="Name",how="left").sort_values(["Exposure","proj_dk"],ascending=[False,False])


def classify_race_scripts(features, sim_matrix):
    """Classify coherent simulated races using CURRENT dynamic dominator candidates.

    V1.5 removes driver-name hard-coding. The two highest dominator-strength
    drivers after any live practice/qualifying update define the concentrated
    dominator scripts for that slate.
    """
    dom=features["dominator_strength"].to_numpy(float)
    dom_order=np.argsort(dom)[::-1]
    primary_idx=int(dom_order[0]) if len(dom_order) else None
    secondary_idx=int(dom_order[1]) if len(dom_order)>1 else None

    order=np.argsort(sim_matrix,axis=1)[:,::-1]
    top1=sim_matrix[np.arange(len(sim_matrix)),order[:,0]]
    top2=sim_matrix[np.arange(len(sim_matrix)),order[:,1]]
    top3=sim_matrix[np.arange(len(sim_matrix)),order[:,2]]
    spread=top1-top3

    labels=np.full(len(sim_matrix),"split_dominator",dtype=object)
    labels[spread < np.quantile(spread,.22)]="chaos_attrition"
    labels[spread > np.quantile(spread,.78)]="track_position"

    if primary_idx is not None:
        x=sim_matrix[:,primary_idx]
        labels[(x >= np.quantile(x,.82)) & (x >= top2)]="primary_dominator"
    if secondary_idx is not None:
        x=sim_matrix[:,secondary_idx]
        labels[(x >= np.quantile(x,.82)) & (x >= top2)]="secondary_dominator"
    return labels

def _candidate_lineup_matrix(lineups, features):
    name_to_idx={n:i for i,n in enumerate(features["Name"])}
    mats=[]
    valid=[]
    for ridx,row in lineups.iterrows():
        ns=[x.strip() for x in str(row["Drivers"]).split("|")]
        if len(ns)==6 and all(n in name_to_idx for n in ns):
            mats.append([name_to_idx[n] for n in ns])
            valid.append(ridx)
    return np.asarray(mats,dtype=int), valid

def optimize_scenario_portfolio(features, sim_matrix, n_lineups=10, salary_cap=50000,
                                roster_size=6, max_overlap=4, candidate_pool=1800, seed=42,
                                max_exposure=1.0):
    """
    V1.3: build many legal candidates, score them inside coherent races, then
    reward portfolios that cover empirically occurring race scripts. No driver
    is forced in/out. V1.4.2 adds a user-controlled hard portfolio exposure cap.
    """
    base=optimize_race_sim_lineups(features,sim_matrix,n_lineups=max(250,min(candidate_pool,900)),
                                  salary_cap=salary_cap,roster_size=roster_size,
                                  max_overlap=roster_size,candidate_pool=candidate_pool,
                                  seed=seed,risk_mode="gpp")
    # Seed the candidate bank with a guaranteed-feasible portfolio under the
    # requested exposure/overlap rules. The scenario search can improve on it,
    # but will not get trapped with fewer than the requested number of lineups.
    seed_port=optimize_lineups(features,n_lineups=n_lineups,salary_cap=salary_cap,
                              roster_size=roster_size,max_overlap=max_overlap,
                              objective="ceiling",max_exposure=max_exposure)
    if len(seed_port):
        base=pd.concat([base,seed_port],ignore_index=True,sort=False).drop_duplicates("Drivers").reset_index(drop=True)
    idxs, valid=_candidate_lineup_matrix(base,features)
    if len(valid)==0: return base.head(n_lineups)

    labels=classify_race_scripts(features,sim_matrix)
    scripts=["primary_dominator","secondary_dominator","split_dominator","chaos_attrition","track_position"]
    probs={s:float(np.mean(labels==s)) for s in scripts}

    # Candidate score distribution from the SAME simulated races.
    scores=np.stack([sim_matrix[:,ix].sum(axis=1) for ix in idxs],axis=1)
    mean=scores.mean(0)
    p90=np.quantile(scores,.90,axis=0)
    p95=np.quantile(scores,.95,axis=0)

    script_scores={}
    for s in scripts:
        mask=labels==s
        if mask.sum() >= 20:
            script_scores[s]=np.quantile(scores[mask],.90,axis=0)
        else:
            script_scores[s]=p90.copy()

    # Normalize within candidate pool so a rare script matters in proportion
    # to how often the simulator actually produces it.
    def z(a):
        sd=np.std(a)
        return (a-np.mean(a))/(sd if sd>1e-9 else 1)
    scenario_component=np.zeros(len(valid))
    for s in scripts:
        scenario_component += probs[s]*z(script_scores[s])

    overall=0.30*z(mean)+0.25*z(p90)+0.20*z(p95)+0.25*scenario_component

    # V2.14 viability gate: diversification may choose a different race story,
    # but not a lineup whose simulated quality is far below the candidate bank.
    raw_gpp=0.40*mean+0.25*p90+0.35*p95
    mean_floor=0.82*float(np.max(mean))
    gpp_floor=0.85*float(np.max(raw_gpp))
    viable=(mean>=mean_floor) & (raw_gpp>=gpp_floor)

    # Driver-specific dominator tails for portfolio coverage. These are not
    # projection boosts: they score a lineup only inside simulated races where
    # that pre-race dominator candidate reaches his own upper tail.
    dom_rank=np.argsort(features["dominator_strength"].to_numpy(float))[::-1]
    tail_candidates=list(dom_rank[:min(6,len(dom_rank))])
    dom_tail_scores={}
    for d in tail_candidates:
        threshold=np.quantile(sim_matrix[:,d],0.85)
        mask=sim_matrix[:,d] >= threshold
        dom_tail_scores[d]=np.quantile(scores[mask],0.90,axis=0) if mask.sum()>=20 else p90.copy()

    # V1.4.2 portfolio construction starts from the guaranteed-feasible seed
    # portfolio, then locally upgrades lineups with stronger scenario candidates
    # while preserving the hard exposure and pairwise-overlap constraints.
    max_count=max(1, int(np.floor(n_lineups*max_exposure + 1e-9)))
    conf=pd.to_numeric(features.get("model_confidence", pd.Series(0.90,index=features.index)),errors="coerce").fillna(0.45).to_numpy(float)
    cov=pd.to_numeric(features.get("data_coverage", pd.Series(1.0,index=features.index)),errors="coerce").fillna(0.15).to_numpy(float)
    driver_caps=np.full(len(features),max_count,dtype=int)
    low=(conf<0.55)|(cov<0.35)
    med=((conf<0.75)|(cov<0.60)) & ~low
    driver_caps[low]=np.minimum(driver_caps[low],max(1,int(np.floor(n_lineups*0.30+1e-9))))
    driver_caps[med]=np.minimum(driver_caps[med],max(1,int(np.floor(n_lineups*0.45+1e-9))))
    cand_sets=[set(x.tolist()) for x in idxs]
    driver_tuple_to_j={tuple(sorted(x.tolist())):j for j,x in enumerate(idxs)}
    seed_idxs, _seed_valid=_candidate_lineup_matrix(seed_port,features)
    chosen=[]
    for x in seed_idxs:
        j=driver_tuple_to_j.get(tuple(sorted(x.tolist())))
        if j is not None: chosen.append(j)
    if len(chosen)<n_lineups:
        raise RuntimeError(f"Could not seed {n_lineups} legal lineups under max exposure {max_exposure:.0%} and max overlap {max_overlap}.")
    chosen=chosen[:n_lineups]
    # Replace non-viable seed/fallback builds before scenario optimization.
    for pos in range(n_lineups):
        if viable[chosen[pos]]: continue
        for j in [int(x) for x in np.argsort(overall)[::-1] if viable[int(x)]]:
            if j in chosen: continue
            trial=chosen.copy(); trial[pos]=j
            # temporary legality check inline because legal_portfolio is defined below
            sets=[cand_sets[x] for x in trial]
            counts=np.zeros(len(features),dtype=int)
            for ss in sets: counts[list(ss)]+=1
            if np.any(counts>driver_caps): continue
            if any(len(sets[a]&sets[b])>max_overlap for a in range(len(sets)) for b in range(a)): continue
            chosen=trial; break

    # Portfolio-level dominator coverage. In multi-entry GPP play, a plausible
    # front-running car should not be allowed to disappear solely because its
    # median projection trails place-differential values. For portfolios of 10+
    # lineups, require one lineup of coverage for each of the top six pre-race
    # dominator-strength candidates. This is symmetric and driver-agnostic.
    dom_order=np.argsort(features["dominator_strength"].to_numpy(float))[::-1]
    coverage_drivers=list(dom_order[:min(6,len(dom_order))]) if n_lineups>=10 else []

    def legal_portfolio(js, require_coverage=False):
        if any(not viable[int(j)] for j in js): return False
        counts=np.zeros(len(features),dtype=int)
        sets=[cand_sets[j] for j in js]
        for ss in sets: counts[list(ss)]+=1
        if np.any(counts>driver_caps): return False
        for a in range(len(sets)):
            for b in range(a):
                if len(sets[a] & sets[b])>max_overlap: return False
        if require_coverage and coverage_drivers:
            covered=set().union(*sets) if sets else set()
            if any(i not in covered for i in coverage_drivers): return False
        return True

    # Coordinate-ascent replacements: scenario score improves only when all
    # portfolio risk constraints remain satisfied.
    order=[int(j) for j in np.argsort(overall)[::-1] if viable[int(j)]]
    if not order:
        order=list(np.argsort(overall)[::-1])
    improved=True; passes=0
    while improved and passes<3:
        improved=False; passes+=1
        for pos in range(n_lineups):
            current=chosen[pos]
            for j in order:
                if j in chosen or overall[j] <= overall[current]+1e-12: continue
                trial=chosen.copy(); trial[pos]=j
                if legal_portfolio(trial):
                    chosen=trial; improved=True; break
    # V1.4.3: pair-swap rescue pass. A hard exposure cap can make a weak seed
    # lineup impossible to improve with a one-at-a-time replacement even when
    # two coordinated replacements produce a materially stronger legal portfolio.
    # Search coordinated two-lineup swaps and maximize the portfolio's weak end
    # first, then total scenario quality.
    top_order=order[:min(len(order),500)]
    def portfolio_key(js):
        vals=np.asarray([overall[j] for j in js],dtype=float)
        return (float(np.min(vals)), float(np.quantile(vals,.20)), float(np.sum(vals)))
    best_key=portfolio_key(chosen)
    pair_improved=True; pair_passes=0
    while pair_improved and pair_passes<2:
        pair_improved=False; pair_passes+=1
        # Start with the weakest slots; that is where coordinated swaps matter.
        weak_positions=list(np.argsort([overall[j] for j in chosen]))
        for aa in range(min(5,len(weak_positions))):
            a=weak_positions[aa]
            for b in range(n_lineups):
                if b==a: continue
                keep=set(chosen); keep.discard(chosen[a]); keep.discard(chosen[b])
                for j in top_order:
                    if j in keep: continue
                    for k in top_order:
                        if k==j or k in keep: continue
                        trial=chosen.copy(); trial[a]=j; trial[b]=k
                        if not legal_portfolio(trial): continue
                        key=portfolio_key(trial)
                        if key > best_key:
                            chosen=trial; best_key=key; pair_improved=True
                            break
                    if pair_improved: break
                if pair_improved: break
            if pair_improved: break

    # Final dominator-coverage repair: replace the weakest legal lineup when a
    # top-six dominator candidate is absent. Candidate lineups still must satisfy
    # salary, exposure, and overlap rules; among legal repairs choose the highest
    # scenario score.
    if coverage_drivers:
        for d in coverage_drivers:
            if any(d in cand_sets[j] for j in chosen):
                continue
            best=None
            for pos in np.argsort([overall[j] for j in chosen]):
                for j in order:
                    if j in chosen or d not in cand_sets[j]: continue
                    trial=chosen.copy(); trial[int(pos)]=j
                    if legal_portfolio(trial):
                        # For a missing dominator script, choose the lineup that
                        # performs best when that driver's simulated ceiling hits,
                        # while retaining a small overall-quality tiebreaker.
                        repair_score=float(z(dom_tail_scores[d])[j] + 0.15*overall[j])
                        if best is None or repair_score>best[0]: best=(repair_score,int(pos),j)
                if best is not None: break
            if best is not None:
                _,pos,j=best; chosen[pos]=j

    rows=[]
    for rank,j in enumerate(chosen,1):
        r=base.iloc[valid[j]].copy()
        r["Lineup"]=rank
        r["Mean"]=round(float(mean[j]),2)
        r["P10"]=round(float(np.quantile(scores[:,j],.10)),2)
        r["P90"]=round(float(p90[j]),2)
        r["P95"]=round(float(p95[j]),2)
        r["GPP_Score"]=round(float(raw_gpp[j]),2)
        r["ScenarioScore"]=round(float(scenario_component[j]),3)
        r["ViabilityMeanFloor"]=round(float(mean_floor),2)
        r["ViabilityGPPFloor"]=round(float(gpp_floor),2)
        for s in scripts:
            r[s+"_P90"]=round(float(script_scores[s][j]),2)
        rows.append(r)
    out=pd.DataFrame(rows)
    out.attrs["script_probabilities"]=probs
    return out

def scenario_summary(features, sim_matrix):
    labels=classify_race_scripts(features,sim_matrix)
    order=["primary_dominator","secondary_dominator","split_dominator","chaos_attrition","track_position"]
    return pd.DataFrame({
        "Race script":order,
        "Simulation probability":[float(np.mean(labels==s)) for s in order]
    })
