"""Four-source, field-normalized PGA strength blend feeding V11 tournament simulation.

Independent field-weighted components:
- strength-of-field-adjusted (weighted) SG OTT/APP/ARG/PUTT from a user export;
- standard SG OTT/APP/ARG/PUTT already joined in PGA weekly inputs;
- OTIS published Course Fit (native composite, higher is better);
- RunGood published Course Fit or fit Rank (native composite, higher better / rank lower better).

No second weighting is applied inside native vendor Course Fit composites.
Within each SG source, z-score EACH measured category across this field before
applying the user's adjustable category weights. All source groups are then
field-standardized, blended at user weights, and translated into the same
provisional strokes/round scale as the incumbent PGA model. Explicit coverage
and per-golfer component contributions make the optimizer's inputs auditable.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pga_input_integrity import align_to_dk

SOURCE_DEFAULTS={"weighted_sg":20., "standard_sg":20., "otis_course_fit":30., "rrg_course_fit":30., "advanced_stats":0.}
# Explicit FIFTH group, optional and zero by default. Never mislabel it as a
# hidden addition to the four requested source percentages.
ADVANCED_DEFAULTS={
    "birdie_rate":0.,"driving_distance":0.,"driving_accuracy":0.,
    "gir_rate":0.,"scrambling":0.,"bogey_avoidance":0.,
    "scoring_before_cut":0.,"par3_scoring":0.,"par4_scoring":0.,
    "par5_scoring":0.,"approach_175_200":0.,"approach_200_225":0.,
}
# Lower is better for scoring averages and proximity / approach distance.
ADVANCED_DIRECTION={k:(-1. if k in (
    "scoring_before_cut","par3_scoring","par4_scoring","par5_scoring",
    "approach_175_200","approach_200_225") else 1.)
    for k in ADVANCED_DEFAULTS}
CATEGORY_DEFAULTS={"sg_ott":25., "sg_approach":25., "sg_arg":25., "sg_putting":25.}
CATEGORIES=tuple(CATEGORY_DEFAULTS)
SOURCE_LABELS={
    "weighted_sg":"RunGood weighted strokes gained",
    "standard_sg":"RunGood regular strokes gained",
    "otis_course_fit":"OTIS native Course Fit",
    "rrg_course_fit":"RunGood native Course Fit",
    "advanced_stats":"Individually weighted PGA TOUR advanced statistics",
}

def _canon(x):
    return re.sub(r"[^a-z0-9]","",str(x).casefold())

def _player_col(df):
    pairs={_canon(col):col for col in df.columns}
    for term in ("player","name","golfer","playername","golfername"):
        if term in pairs:return pairs[term]
    raise ValueError("Upload needs a Player, Name or Golfer column")

def _numeric(df,col):
    if col not in df:
        return pd.Series(np.nan,index=df.index,dtype=float)
    return pd.to_numeric(df[col],errors="coerce").replace([np.inf,-np.inf],np.nan)

def _z(x):
    v=pd.to_numeric(x,errors="coerce").replace([np.inf,-np.inf],np.nan)
    std=v.std(ddof=0)
    if pd.isna(std) or std<1e-12:
        return pd.Series(np.nan,index=v.index,dtype=float)
    return (v-v.mean())/std

def _weights(raw,known,label):
    src=raw if raw is not None else {}
    if set(src)-set(known):
        raise ValueError(f"{label}: unknown keys: {sorted(set(src)-set(known))}")
    w={k:float(src.get(k,0.)) for k in known}
    if any(not np.isfinite(v) or v<0 for v in w.values()) or sum(w.values())<=0:
        raise ValueError(f"{label}: all weights must be finite, nonnegative, with a positive total.")
    total=sum(w.values())
    return {k:v/total for k,v in w.items()}

def normalize_weighted_sg(raw):
    """Parse common RunGood weighted SG exports. Require recognized categories.

    The source has already applied RunGood's tour/field adjustment; this function
    does NOT pretend raw cross-tour SG is comparable or try to re-adjust it.
    """
    if raw is None or raw.empty:
        raise ValueError("RunGood weighted SG CSV is required when its weight is positive.")
    df=raw.copy()
    who=_player_col(df)
    result=pd.DataFrame({"player":df[who].astype(str).str.strip()})
    aliases={
        "sg_ott":("sgottw","sgott","sgoffthetee","offthetee","ott","wsgott","weightedsgott","weightedoffthetee","sgottweighted"),
        "sg_approach":("sgappw","sgapp","sgapproach","sgapproachthegreen","approach","app","wsgapp","weightedsgapp","weightedapproach","sgappweighted"),
        "sg_arg":("sgargw","sgarg","sgaroundthegreen","sgaroundgreen","aroundthegreen","arg","wsgarg","weightedsgarg","weightedaroundgreen","sgargweighted"),
        "sg_putting":("sgputtw","sgputt","sgputting","putting","putt","wsgputt","weightedsgputt","weightedputting","sgputtweighted"),
    }
    for stat,keys in aliases.items():
        found=[col for col in df if col!=who and _canon(col) in keys]
        if len(found)>1:
            # Never silently prefer an ordinary column over a separate weighted column.
            raise ValueError(f"Weighted SG contains competing {stat} columns: {found}")
        if found:
            result["w_"+stat]=pd.to_numeric(df[found[0]],errors="coerce").replace([np.inf,-np.inf],np.nan)
    if len(result.columns)==1:
        raise ValueError("Weighted SG upload has no recognized SG OTT/APP/ARG/PUTT categories.")
    result=result[result["player"].ne("") & ~result["player"].str.casefold().isin(["nan","none"])].copy()
    return result.reset_index(drop=True)

def normalize_rrg_course_fit(raw):
    """Keep native RunGood fit as a single signal, or reverse an explicitly ranked score."""
    if raw is None or raw.empty:
        raise ValueError("RunGood Course Fit CSV is required when its weight is positive.")
    df=raw.copy()
    who=_player_col(df)
    scores=("coursefit","rrgcoursefit","coursefitscore","fitrating","fitscore","coursefitrating","fit")
    ranks=("coursefitrank","rrgcoursefitrank","fitrank","rank","ranking")
    score_cols=[c for c in df if c!=who and _canon(c) in scores]
    rank_cols=[c for c in df if c!=who and _canon(c) in ranks]
    if len(score_cols)>1 or (not score_cols and len(rank_cols)>1):
        raise ValueError("RunGood Course Fit file has ambiguous score/rank columns; use one.")
    if score_cols:
        column=score_cols[0]; direction=1.; interpretation="native fit score; higher = better"
    elif rank_cols:
        column=rank_cols[0]; direction=-1.; interpretation="native fit rank; lower = better"
    else:
        raise ValueError("RunGood Course Fit needs Course Fit, Course Fit Rank or Rank column.")
    out=pd.DataFrame({"player":df[who].astype(str).str.strip(),
                      "rrg_course_fit":direction*pd.to_numeric(df[column],errors="coerce")})
    out["rrg_course_fit"]=out["rrg_course_fit"].replace([np.inf,-np.inf],np.nan)
    out=out[out.player.ne("") & ~out.player.str.casefold().isin(["nan","none"])]
    return out.reset_index(drop=True),{"input_column":column,"direction":interpretation}

def _sg_group(frame,cols,category_weights):
    score=pd.Series(0.,index=frame.index,dtype=float)
    sufficient=pd.Series(True,index=frame.index,dtype=bool)
    contributions={}
    for category,weight in category_weights.items():
        if weight<=0:continue
        column=cols[category]
        z=_z(_numeric(frame,column))
        if z.notna().sum()<3:
            raise ValueError(f"Input '{column}' has insufficient measured variation for an active category weight.")
        contributions[category]=z*weight
        sufficient&=z.notna()
        score=score.add(contributions[category].fillna(0))
    score=score.where(sufficient,np.nan)
    return _z(score),contributions

def _advanced_group(frame,weights):
    """Within-field standardized advanced categories with explicit direction."""
    result=pd.Series(0.,index=frame.index,dtype=float)
    eligible=pd.Series(True,index=frame.index)
    contrib={}
    for stat,weight in weights.items():
        if weight<=0:continue
        z=_z(_numeric(frame,stat)*ADVANCED_DIRECTION[stat])
        if z.notna().sum()<3:
            raise ValueError(f"Active PGA TOUR statistic {stat!r} lacks enough valid player measurements. Adjust weights or supply a valid stat file.")
        contrib[stat]=z*weight
        result+=contrib[stat].fillna(0.)
        eligible&=z.notna()
    return _z(result.where(eligible)),contrib

def blend_four_sources(projections,weighted_sg=None,rrg_course_fit=None,
                       source_weights=None,category_weights=None,advanced_weights=None,
                       minimum_source_coverage=.80,round_strength_scale=.58):
    """Return (predictions, audit). All nonzero sources must be available.

    Per-golfer missing sources can only be renormalized when at least the
    explicit minimum active source weight is present; below that golfer is
    INELIGIBLE for the four-source optimizer. No synthetic SG is inserted.
    """
    if projections is None or projections.empty or "player" not in projections:
        raise ValueError("Build PGA tournament predictions before four-source blending.")
    if not (0.0<minimum_source_coverage<=1.0):
        raise ValueError("Minimum source coverage must be in (0, 1].")
    sw=_weights(source_weights or SOURCE_DEFAULTS,SOURCE_DEFAULTS,"Source weights")
    cw=_weights(category_weights or CATEGORY_DEFAULTS,CATEGORY_DEFAULTS,"SG category weights")
    aw=_weights(advanced_weights or ADVANCED_DEFAULTS,ADVANCED_DEFAULTS,"Advanced stat weights") if sw["advanced_stats"]>0 else {k:0. for k in ADVANCED_DEFAULTS}
    d=projections.copy().reset_index(drop=True)
    if d.player.astype(str).duplicated().any():
        raise ValueError("PGA projection golfers must be unique.")
    source_reports={}
    if sw["weighted_sg"]>0:
        measured=normalize_weighted_sg(weighted_sg)
        aligned,report=align_to_dk(measured,d[["player"]],"RunGood weighted SG")
        if report["matched"]<3:raise ValueError("RunGood weighted SG has fewer than three matched current-field golfers.")
        extra=aligned.set_index("player")
        for col in extra:
            d[col]=d["player"].map(extra[col])
        source_reports["weighted_sg"]=report
    if sw["rrg_course_fit"]>0:
        fit,fit_meta=normalize_rrg_course_fit(rrg_course_fit)
        aligned,report=align_to_dk(fit,d[["player"]],"RunGood Course Fit")
        if report["matched"]<3:raise ValueError("RunGood Course Fit has fewer than three matched current-field golfers.")
        d["rrg_course_fit"]=d.player.map(aligned.set_index("player")["rrg_course_fit"])
        source_reports["rrg_course_fit"]={**report,**fit_meta}
    signals={}
    if sw["weighted_sg"]>0:
        signals["weighted_sg"],_= _sg_group(d,{k:"w_"+k for k in CATEGORIES},cw)
    if sw["standard_sg"]>0:
        signals["standard_sg"],_= _sg_group(d,{k:k for k in CATEGORIES},cw)
    if sw["otis_course_fit"]>0:
        signals["otis_course_fit"]=_z(_numeric(d,"otis_course_fit"))
    if sw["rrg_course_fit"]>0:
        signals["rrg_course_fit"]=_z(_numeric(d,"rrg_course_fit"))
    if sw["advanced_stats"]>0:
        signals["advanced_stats"],advanced_contributions=_advanced_group(d,aw)
        # _sg_group is a z-score-weighted statistical composite. Adjust
        # low-is-good stats BEFORE standardizing, so lower scoring averages
        # improve the ranking. Do this here explicitly, with source audit.
        for stat,weight in aw.items():
            if weight<=0:continue
            if ADVANCED_DIRECTION[stat]<0:
                d[stat+"_lower_is_better"]=True
        for stat,contribution in advanced_contributions.items():
            d["advanced_contribution_"+stat]=contribution
    for key,z in signals.items():
        if z.notna().sum()<3:
            raise ValueError(f"{SOURCE_LABELS[key]} is missing or has insufficient score variation in the field.")
        d["four_source_"+key+"_z"]=z
        d["four_source_"+key+"_contribution"]=z*sw[key]
    coverage=pd.Series(0.,index=d.index,dtype=float)
    numerator=pd.Series(0.,index=d.index,dtype=float)
    for key,z in signals.items():
        present=z.notna().astype(float)
        coverage+=sw[key]*present
        numerator+=sw[key]*z.fillna(0)
    d["four_source_source_coverage"]=coverage
    d["four_source_source_count"]=sum(signals[key].notna().astype(int) for key in signals)
    ok=coverage.ge(float(minimum_source_coverage)-1e-9)
    combined=(numerator/coverage.replace(0,np.nan)).where(ok)
    normalized=_z(combined)
    if normalized.notna().sum()<3:
        raise ValueError("Fewer than three field golfers have enough four-source signal coverage.")
    d["four_source_blend_raw"]=combined
    d["four_source_blend_z"]=normalized
    d["four_source_rank"]=normalized.rank(ascending=False,method="min")
    d["four_source_eligible"]=ok & normalized.notna()
    d["four_source_status"]=np.select([
        ~d.four_source_eligible,coverage.lt(1.-1e-9)],
        ["INSUFFICIENT SOURCE COVERAGE","PARTIAL — AVAILABLE SOURCES RENORMALIZED"],default="ALL SOURCES PRESENT")
    old=_numeric(d,"expected_round")
    if old.isna().any():raise ValueError("Four-source input has invalid expected_round.")
    if "strength_z" not in d:
        raise ValueError("Incumbent model strength_z is required to preserve the original score scale.")
    original_z=_numeric(d,"strength_z")
    if original_z[d.four_source_eligible].isna().any():
        raise ValueError("Original field strength missing for an otherwise eligible golfer.")
    # The incumbent PGA mean follows expected_round=-.58*strength_z.
    # Replace only that player-strength component; retain cut / current form
    # adjustments added in the next pipeline stage. No fifth secret weight.
    shift=-float(round_strength_scale)*(normalized-original_z)
    d["four_source_delta_strokes"]=shift.where(d.four_source_eligible,0.)
    d["expected_round_pre_four_source"]=old
    d["expected_round"]=old+d["four_source_delta_strokes"]
    if "round_sd" in d:
        prior_sd=_numeric(d,"round_sd")
        d["round_sd"]=np.where(d.four_source_eligible,
                                np.clip(prior_sd-.10*(normalized-original_z),2.0,3.7),
                                prior_sd)
    audit={
        "weights":sw,"sg_category_weights":cw,
        "advanced_stat_weights":aw,"source_details":source_reports,
        "minimum_coverage":minimum_source_coverage,
        "eligible_golfers":int(d.four_source_eligible.sum()),
        "total_golfers":len(d),
        "all_sources_golfers":int(coverage.ge(1.-1e-9).sum()),
        "partial_golfers":int((d.four_source_eligible&coverage.lt(1.-1e-9)).sum()),
        "max_abs_delta_strokes":float(d.four_source_delta_strokes.abs().max()),
        "mean_abs_delta_strokes":float(d.four_source_delta_strokes.abs().mean()),
        "scoring":"Within-field standardized SG categories; weighted four-source z-score; incumbent -.58 strokes/round scale (uncalibrated).",
        "course_fits":"OTIS and RunGood native course-fit composites unchanged internally.",
        "advanced_stat_policy":"Fifth optional weighted source (0% by default); lower-is-better stats reversed; no fabricated unavailable measurements.",
    }
    return d,audit
