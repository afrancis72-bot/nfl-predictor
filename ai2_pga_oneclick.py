"""One-page PGA V11 adapter for Team AI² Pro, with explicit data provenance.

The original PGA predictor was CSV/OTIS driven. The unified UI accepts a
current DraftKings field alone for market-only research; optional genuine
observed golfer statistics unlock the full historical/course-fit model.
Missing strokes gained or official hole pars are never fabricated.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import inspect
import re
import unicodedata
import numpy as np
import pandas as pd

from pga_predictor_pro import Config as PGAConfig, predict_from_dataframes
from golf_first_v11 import simulate_golf_first
from portfolio_optimizer import PortfolioSettings, optimize_portfolio
from round_parlay import (build_round_ratings,simulate_groups,build_six_leg_tickets,
                          normalize_live_results,normalize_groupings_upload)
from round_showdown import build_showdown_pool,optimize_showdown

def key(value):
    s=unicodedata.normalize("NFKD",str(value)).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]","",s)

def parse_dk(upload):
    """Require genuine current-week golfers and legal salaries; preserve DK IDs."""
    df=upload.copy() if isinstance(upload,pd.DataFrame) else pd.read_csv(upload)
    columns={key(c):c for c in df.columns}
    def lookup(*names):
        return next((columns[key(name)] for name in names if key(name) in columns),None)
    name=lookup("Name","player","Player Name","golfer")
    salary=lookup("Salary","DK Salary")
    if not name or not salary:
        raise ValueError("DraftKings PGA salary CSV needs Name (golfer) and Salary.")
    status=lookup("Status","Player Status")
    if status is not None:
        df=df.loc[~df[status].astype(str).str.strip().str.lower().isin(["out","withdrawn","wd"])].copy()
    out=pd.DataFrame({"player":df[name].astype(str).str.strip(),
                      "salary":pd.to_numeric(df[salary].astype(str).str.replace("$","",regex=False)
                                .str.replace(",","",regex=False),errors="coerce")})
    id_col=lookup("ID","Player ID")
    if id_col:
        out["dk_id"]=df[id_col].astype(str).str.replace(r"\.0$","",regex=True).str.strip()
    else:
        out["dk_id"]=""
    out=out.loc[~out["player"].str.lower().isin(["nan","none",""])].copy()
    if out.empty or len(out)<6:
        raise ValueError("Need at least six valid golfers from the tournament slate.")
    if out["salary"].isna().any() or (~out["salary"].between(3000,20000)).any():
        raise ValueError("Invalid or missing DraftKings golf salary.")
    if out["player"].map(key).duplicated().any():
        raise ValueError("Duplicate golfer names in DraftKings slate.")
    if out["dk_id"].ne("").any() and (out["dk_id"].eq("").any() or out["dk_id"].duplicated().any()):
        raise ValueError("DraftKings golfer IDs are missing or duplicated.")
    out["salary"]=out["salary"].astype(int)
    return out.reset_index(drop=True)

def normalize_otis(raw,dk):
    """Map Advanced Course-Fit columns to original PGA predictor contracts.

    OTIS 'rank' and 'model' are audit-only, never direct forecast inputs.
    """
    if raw is None:
        return pd.DataFrame(),0
    d=raw.copy()
    src={key(c):c for c in d.columns}
    def find(n):
        return src.get(key(n))
    required={"player":"Player","otis_true_skill":"True Skill",
              "otis_course_fit":"Course Fit","otis_form":"Form"}
    for label,col in required.items():
        if not find(col):
            raise ValueError("OTIS Advanced Course-Fit CSV needs Player, True Skill, Course Fit, and Form.")
    optional={"otis_fit_app":"Fit: APP","otis_fit_ott":"Fit: OTT",
        "otis_fit_arg":"Fit: ARG","otis_fit_putt":"Fit: PUTT",
        "otis_fit_history":"Fit: History","otis_form_rds":"Form rds",
        "otis_venue_rds":"Venue rds","otis_rank":"Rank",
        "otis_model_audit":"Model"}
    fields={**required,**optional}
    out=pd.DataFrame({label:d[find(original)] for label,original in fields.items() if find(original)})
    out["player"]=out["player"].astype(str).str.strip()
    if out["player"].map(key).duplicated().any():
        raise ValueError("Duplicate golfer in OTIS Course-Fit CSV.")
    out["merge_key"]=out["player"].map(key)
    names=dk[["player"]].copy()
    names["merge_key"]=names["player"].map(key)
    out=names.merge(out.drop(columns=["player"]),on="merge_key",how="left",validate="one_to_one").drop(columns="merge_key")
    numeric=[c for c in out.columns if c!="player"]
    for c in numeric:
        out[c]=pd.to_numeric(out[c],errors="coerce")
    matched=int(out["otis_true_skill"].notna().sum())
    if matched/max(1,len(dk))<.70:
        raise ValueError(f"Only {matched}/{len(dk)} DraftKings golfers matched OTIS (70% required). Check event and name spelling.")
    # Missing OTIS rows remain genuinely absent; the original engine shrinks
    # missing values to neutral contribution rather than poor performances.
    return out,matched

def normalize_independent_stats(raw,dk):
    """Optional measured RRG/SG Stats Bank export; independent of OTIS ranking.

    Accepts published strokes-gained columns; missing measurements remain NaN.
    DK salaries or salary rank are never misrepresented as observed SG.
    """
    if raw is None:
        return pd.DataFrame(),0
    frame=raw.copy()
    columns={key(c):c for c in frame.columns}
    def choose(names):
        return next((columns[key(n)] for n in names if key(n) in columns),None)
    name=choose(("player","name","golfer","player name","golfer name"))
    if not name:
        raise ValueError("Independent golfer stats CSV needs a player/name column.")
    aliases={
        "sg_total":["sg_total","SG: Total","SG Total","SG TOT","Strokes Gained Total","SGT"],
        "sg_approach":["sg_approach","SG: APP","SG APP","SG Approach","Strokes Gained Approach","SG: Approach"],
        "sg_ott":["sg_ott","SG: OTT","SG OTT","SG Off the Tee","Strokes Gained Off the Tee","SG Off Tee"],
        "sg_arg":["sg_arg","SG: ARG","SG ARG","SG Around the Green","Strokes Gained Around the Green"],
        "sg_putting":["sg_putting","SG: PUTT","SG PUTT","SG Putting","Strokes Gained Putting"],
        "driving_distance":["driving_distance","Driving Distance"],
        "driving_accuracy":["driving_accuracy","Driving Accuracy"],
        "birdie_rate":["birdie_rate","Birdie Rate","Birdie Percentage"],
        "bogey_avoidance":["bogey_avoidance","Bogey Avoidance"],
        "measured_rounds":["measured_rounds","Measured Rounds","Rounds"],
    }
    found={k:choose(v) for k,v in aliases.items()}
    sg_keys=("sg_total","sg_approach","sg_ott","sg_arg","sg_putting")
    if not any(found[k] for k in sg_keys):
        raise ValueError("Independent stats CSV has no measured SG Total, APP, OTT, ARG, or PUTT column.")
    vals=pd.DataFrame({"player":frame[name].astype(str).str.strip()})
    vals["match_key"]=vals["player"].map(key)
    if vals["match_key"].duplicated().any():
        raise ValueError("Duplicate golfer in independent stats CSV.")
    for k,source in found.items():
        if source:
            vals[k]=pd.to_numeric(frame[source],errors="coerce")
    names=dk[["player"]].copy()
    names["match_key"]=names["player"].map(key)
    aligned=names.merge(vals.drop(columns=["player"]),on="match_key",how="left",validate="one_to_one").drop(columns=["match_key"])
    covered=int(aligned[list(k for k in sg_keys if k in aligned)].notna().any(axis=1).sum())
    if covered/max(1,len(dk))<.70:
        raise ValueError(f"Only {covered}/{len(dk)} golfers have measured SG data in the independent export; need 70% or more.")
    return aligned,covered


# RRG Player Fit is a course-correlation signal, NOT raw strokes gained.
# We use our own category weighting instead of blindly copying RRG's FIT score.
RRG_FIT_WEIGHTS={"rrg_fit_app":.35,"rrg_fit_ott":.25,
                 "rrg_fit_putt":.14,"rrg_fit_arg":.08,"rrg_fit_scoring":.18}

def normalize_rrg_course_fit(raw,dk):
    """Match RRG Course Key Stats / Player Fit exports to the current DK field."""
    empty_meta={"matched":0,"components":0,"overall_only":0,"used":[]}
    if raw is None:
        return pd.DataFrame(),empty_meta
    if raw.empty:
        raise ValueError("RickRunGood Course Fit CSV is empty.")
    frame=raw.copy()
    columns={key(c):c for c in frame.columns}
    def find(*names):
        return next((columns[key(n)] for n in names if key(n) in columns),None)
    name=find("Player","Name","Golfer","Golfer Name","Player Name")
    if name is None:
        raise ValueError("RRG Course Fit needs a Player/Name column.")
    aliases={
        "rrg_fit_app":("APP","Approach","Fit APP","APP Fit","Fit: APP","Course APP","Approach Fit"),
        "rrg_fit_ott":("OTT","Driving","Fit OTT","OTT Fit","Fit: OTT","Off the Tee","Driving Fit"),
        "rrg_fit_arg":("ARG","Around Green","Fit ARG","ARG Fit","Fit: ARG","Around the Green"),
        "rrg_fit_putt":("PUTT","Putting","Fit PUTT","PUTT Fit","Fit: PUTT","Putting Fit"),
        "rrg_fit_scoring":("SCORING","Scoring","Scoring Fit","Fit: SCORING"),
        "rrg_fit_overall":("FIT","Course Fit","Course Fit Score","Player Fit","Player Fit Score"),
        "rrg_fit_rank_audit":("Fit Rank","Course Fit Rank","Rank"),
        "rrg_fit_rounds":("Rounds","Measured Rounds","Sample Rounds"),
    }
    sources={k:find(*v) for k,v in aliases.items()}
    evidence=list(RRG_FIT_WEIGHTS)+["rrg_fit_overall"]
    if not any(sources[x] for x in evidence):
        raise ValueError("RRG Course Fit needs FIT or category columns: OTT, APP, ARG, PUTT, SCORING. Rank or salary is not enough.")
    values=pd.DataFrame({"player":frame[name].astype(str).str.strip()})
    values["_key"]=values["player"].map(key)
    if values["_key"].duplicated().any() or values["_key"].isin(["","nan","none"]).any():
        raise ValueError("RRG Course Fit golfer names must be nonblank and unique.")
    for target,original in sources.items():
        if original:
            cleaned=(frame[original].astype(str).str.strip()
                    .str.replace(",","",regex=False)
                    .str.replace("%","",regex=False)
                    .str.replace("−","-",regex=False))
            values[target]=pd.to_numeric(cleaned,errors="coerce")
    names=dk[["player"]].copy()
    names["_key"]=names.player.map(key)
    aligned=names.merge(values.drop(columns=["player"]),on="_key",
                        how="left",validate="one_to_one").drop(columns="_key")
    cols=[c for c in evidence if c in aligned]
    matched=int(aligned[cols].notna().any(axis=1).sum())
    if matched/len(dk)<.70:
        raise ValueError(f"Only {matched}/{len(dk)} DK golfers have usable RRG Course Fit; require >=70% matching the current tournament.")
    comps=[c for c in RRG_FIT_WEIGHTS if c in aligned]
    comp=int(aligned[comps].notna().any(axis=1).sum()) if comps else 0
    return aligned,{"matched":matched,"components":comp,
                    "overall_only":matched-comp,
                    "used":[x for x in cols if aligned[x].notna().any()]}

def apply_rrg_course_fit(base,fit):
    """Conservative, reproducible within-field RRG fit tilt, capped in strokes.

    Per-skill components take priority. Vendor-composite FIT only is half-weight,
    to prevent blindly trusting a separate weighting system.
    """
    result=base.copy()
    result["rrg_fit_adjustment_strokes"]=0.
    result["rrg_fit_signal_z"]=np.nan
    result["rrg_fit_evidence"]="not provided"
    if fit is None or fit.empty:
        return result
    result=result.merge(fit,on="player",how="left",validate="one_to_one")
    values=np.zeros(len(result))
    weights=np.zeros(len(result))
    for c,w in RRG_FIT_WEIGHTS.items():
        if c not in result:
            continue
        col=pd.to_numeric(result[c],errors="coerce")
        present=col.notna().to_numpy()
        if int(present.sum())<2:
            continue
        avg=float(col[present].mean())
        std=float(col[present].std(ddof=0))
        z=np.zeros(len(result)) if std<1e-10 else np.clip(
            (col.fillna(avg).to_numpy(float)-avg)/std,-2.5,2.5)
        values[present]+=w*z[present]
        weights[present]+=w
    has_comp=weights>0
    score=np.divide(values,weights,out=np.zeros(len(result)),where=has_comp)
    overall_known=np.zeros(len(result),bool)
    overall_z=np.zeros(len(result))
    if "rrg_fit_overall" in result:
        col=pd.to_numeric(result["rrg_fit_overall"],errors="coerce")
        overall_known=col.notna().to_numpy()
        if int(overall_known.sum())>=2:
            avg=float(col[overall_known].mean())
            std=float(col[overall_known].std(ddof=0))
            if std>1e-10:
                overall_z=np.clip((col.fillna(avg).to_numpy(float)-avg)/std,-2.5,2.5)
    fallback=~has_comp & overall_known
    chosen=np.where(has_comp,score,np.where(fallback,overall_z,0.))
    # Missing measured-round sample => shrink to 65% of nominal influence.
    reliability=np.full(len(result),.65)
    if "rrg_fit_rounds" in result:
        rounds=pd.to_numeric(result["rrg_fit_rounds"],errors="coerce")
        available=rounds.notna() & rounds.ge(0)
        reliability[available.to_numpy()]=np.sqrt(
            (rounds[available].clip(0,48)/48).to_numpy(float))
    change=np.clip(.12*chosen,-.22,.22)*reliability
    change[fallback]*=.5
    result["rrg_fit_signal_z"]=np.where(has_comp|fallback,chosen,np.nan)
    result["rrg_fit_adjustment_strokes"]=np.where(has_comp|fallback,-change,0.)
    result["rrg_fit_evidence"]=np.select([has_comp,fallback],
        ["Our weighting of RRG skill-fit categories","RRG overall FIT only (reduced weight)"],
        default="no measured course-fit input")
    result["expected_round_before_rrg_fit"]=pd.to_numeric(result.expected_round,errors="coerce")
    result["expected_round"]=result["expected_round_before_rrg_fit"]+result["rrg_fit_adjustment_strokes"]
    return result

def market_baseline(dk):
    """Observable salary ranking only; *not* measured SG or official world rank."""
    d=dk[["player","salary"]].copy()
    s=d["salary"].astype(float)
    sigma=max(float(s.std(ddof=0)),1.)
    z=np.clip((s-s.mean())/sigma,-2.5,2.5)
    d["expected_round"]=-0.48*z
    d["round_sd"]=2.85
    d["model_source"]="DK salary market-only baseline; no independent golfer history"
    d["model_data_confidence"]=0.10
    d["otis_fit_app"]=50.
    return d

def validated_holes(hole_frame,course_par):
    if hole_frame is None or hole_frame.empty:
        return None
    for c in ("par","Par","hole_par"):
        if c in hole_frame:
            vals=pd.to_numeric(hole_frame[c],errors="coerce")
            if len(vals)!=18 or vals.isna().any() or not vals.isin([3,4,5]).all() or int(vals.sum())!=int(course_par):
                raise ValueError("Verified hole-par CSV must contain exactly 18 valid pars summing to course par.")
            return vals.astype(int).tolist()
    raise ValueError("Hole-par CSV requires a par column for the 18 holes.")

def _ensure_coverage_for_optimizer(df):
    d=df.copy()
    d["dk_points_proxy"]=d["v11_mean_dk"]
    d["dk_value_per_1000"]=d["v11_salary_value"]
    d["make_cut_pct"]=d["v11_make_cut_pct"]/100.
    d["win_pct"]=d["v11_win_pct"]/100.
    d["top10_pct"]=d["v11_top10_pct"]/100.
    if "model_data_confidence" not in d:
        d["model_data_confidence"]=0.1
    return d

def group_ratings_source(model):
    """Market-only fallbacks are visibly flagged and not disguised as OTIS."""
    otis=model["otis"]
    if model.get("metadata",{}).get("rrg_fit_match",0):
        p=model["projections"][["player","expected_round","rrg_fit_signal_z"]].copy()
        z=(p.expected_round-p.expected_round.mean())/max(float(p.expected_round.std(ddof=0)),.01)
        p["otis_true_skill"]=(50.-9.*z).clip(20,80)
        p["otis_course_fit"]=(50.+8.*p.rrg_fit_signal_z.fillna(0)).clip(20,80)
        p["otis_form"]=50.
        return p[["player","otis_true_skill","otis_course_fit","otis_form"]], "Independent model + RRG skill-fit course tilt; estimated, not calibrated"
    if int(model.get("metadata",{}).get("independent_sg_match",0))>0:
        p=model["projections"][["player","expected_round"]].copy()
        z=(p["expected_round"]-p["expected_round"].mean())/max(float(p["expected_round"].std(ddof=0)),.01)
        p["otis_true_skill"]=(50.-9.*z).clip(20,80)
        p["otis_course_fit"]=50.
        p["otis_form"]=50.
        return p[["player","otis_true_skill","otis_course_fit","otis_form"]], "independent observed-SG model-derived ratings (not OTIS ranks)"
    if not otis.empty:
        return otis,"observed OTIS True Skill, Course Fit and Form; uncalibrated round rating"
    p=model["projections"][["player","expected_round"]].copy()
    z=(p["expected_round"]-p["expected_round"].mean())/max(float(p["expected_round"].std(ddof=0)),.01)
    p["otis_true_skill"]=(50.-9.*z).clip(20,80)
    p["otis_course_fit"]=50.
    p["otis_form"]=50.
    return p[["player","otis_true_skill","otis_course_fit","otis_form"]], "DK salary-derived proxy ONLY, not observed SG or independent course fit"

def run_tournament(dk,tournament,course_par,no_cut,cut_size=65,hole_frame=None,
                   otis_raw=None,independent_raw=None,course_fit_raw=None,weather_raw=None,sims=3000,seed=42,lineup_count=10,
                   max_exposure=.35,salary_floor=46500,min_unique=3,
                   strategy="Balanced / Single Entry",locks=(),excludes=()):
    if not str(tournament).strip():
        raise ValueError("Enter the actual current tournament.")
    if int(course_par) not in (70,71,72,73):
        raise ValueError("Confirm actual course par (70–73).")
    if int(sims) not in (1000,3000,5000):
        raise ValueError("Supported golf-first simulation runs: 1000, 3000, 5000.")
    dk=parse_dk(dk)
    if int(lineup_count)<1 or int(lineup_count)>30:
        raise ValueError("Lineup count must be 1–30.")
    if not .1 <=float(max_exposure)<=1:
        raise ValueError("Exposure cap outside 10–100%.")
    holes=validated_holes(hole_frame,int(course_par))
    otis,matched=normalize_otis(otis_raw,dk)
    independent,observed_match=normalize_independent_stats(independent_raw,dk)
    course_fit,course_fit_meta=normalize_rrg_course_fit(course_fit_raw,dk)
    weather=(weather_raw.copy() if isinstance(weather_raw,pd.DataFrame) else pd.DataFrame())
    if not independent.empty:
        base=predict_from_dataframes(
            PGAConfig(tournament=str(tournament),sims=int(sims),seed=int(seed)),
            dk[["player","salary"]],independent,
            results=pd.DataFrame(),history=pd.DataFrame(),holes=pd.DataFrame(),
            weather=weather,
        )
        mode="Independent observed strokes-gained stats + V10 golfer/course model; no OTIS model weighting"
    elif not otis.empty:
        # No silent use of stale repo results/course-history files.
        base=predict_from_dataframes(
            PGAConfig(tournament=str(tournament),sims=int(sims),seed=int(seed)),
            dk[["player","salary"]],otis,
            results=pd.DataFrame(),history=pd.DataFrame(),holes=pd.DataFrame(),
            weather=weather,
        )
        mode="OTIS observed field strength/form/fit + V10 model"
    else:
        base=market_baseline(dk)
        mode="DK salary ONLY; limited-data provisional, not independently predictive"
    # Match to actual DK contest field, including real DK IDs.
    base=dk[["player","salary","dk_id"]].merge(
        base.drop(columns=["salary","dk_id"],errors="ignore"),
        on="player",how="left",validate="one_to_one"
    )
    if base[["expected_round","round_sd"]].isna().any().any():
        raise ValueError("PGA forecast contains missing golfer baseline. No portfolio exported.")
    base=apply_rrg_course_fit(base,course_fit)
    predicted,draws,diagnostics=simulate_golf_first(
        base,sims=int(sims),seed=int(seed),course_par=int(course_par),
        no_cut=bool(no_cut),cut_size=int(cut_size),hole_pars=holes
    )
    predicted=_ensure_coverage_for_optimizer(predicted)
    settings=PortfolioSettings(lineup_count=int(lineup_count),salary_cap=50000,
        salary_floor=int(salary_floor),min_player_salary=3000,roster_size=6,
        max_exposure=float(max_exposure),min_unique=int(min_unique),
        strategy=strategy,candidate_samples=15000,target_candidates=1100,
        candidate_pool_size=min(len(predicted),80),seed=int(seed))
    portfolio,summary,exposure=optimize_portfolio(
        predicted,settings,locks=locks,excludes=excludes
    )
    if len(summary)!=int(lineup_count):
        raise ValueError("Incomplete PGA portfolio; no partial export.")
    if (summary["salary"]<salary_floor).any() or (summary["salary"]>50000).any():
        raise ValueError("Invalid lineup salary.")
    if (exposure["exposure_pct"]>100*float(max_exposure)+1e-6).any():
        raise ValueError("PGA portfolio violates exposure cap.")
    if not np.allclose(draws.mean(axis=0),predicted["v11_mean_dk"].to_numpy(float),atol=.01):
        raise ValueError("PGA simulated points not aligned to golfers.")
    # Actual same-scenario lineup distribution, NOT sum of individual P90s.
    ix={n:i for i,n in enumerate(predicted["player"])}
    same_world=[]
    for no,g in portfolio.groupby("lineup"):
        picks=[ix[n] for n in g["player"]]
        v=draws[:,picks].sum(axis=1)
        same_world.append({"lineup":int(no),
            "sim_lineup_mean":round(float(v.mean()),2),
            "sim_lineup_p10":round(float(np.quantile(v,.10)),2),
            "sim_lineup_p90":round(float(np.quantile(v,.90)),2),
            "sim_lineup_p95":round(float(np.quantile(v,.95)),2)})
    summary=summary.merge(pd.DataFrame(same_world),on="lineup",how="left",validate="one_to_one")
    return {"projections":predicted,"lineups":portfolio,"summary":summary,
        "exposure":exposure,"otis":otis,"independent":independent,"course_fit":course_fit,"raw_draws":draws,
        "diagnostics":diagnostics,
        "metadata":{"tournament":str(tournament),"course_par":int(course_par),
            "hole_pars_verified":holes is not None,"field_size":len(dk),
            "simulations":int(sims),"seed":int(seed),
            "otis_match":matched,"independent_sg_match":observed_match,
            "rrg_fit_match":course_fit_meta["matched"],
            "rrg_fit_components":course_fit_meta["components"],
            "rrg_fit_overall_only":course_fit_meta["overall_only"],
            "rrg_fit_columns":course_fit_meta["used"],
            "weather_hours":len(weather),"weather_wind_mph":(
                round(float(pd.to_numeric(weather["wind_mph"],errors="coerce").mean()),1)
                if "wind_mph" in weather and weather["wind_mph"].notna().any() else None),
            "source_mode":mode,
            "created_utc":datetime.now(timezone.utc).isoformat()}}

def dk_roster_ids(portfolio,projections):
    if not projections["dk_id"].ne("").all():
        raise ValueError("The DK slate lacks real unique golfer IDs. Download a names-only review instead.")
    ids=projections.set_index("player")["dk_id"].to_dict()
    fields=[]
    for _,g in portfolio.groupby("lineup"):
        chosen=g["player"].astype(str).tolist()
        if len(chosen)!=6 or len(set(chosen))!=6:
            raise ValueError("PGA DK roster must contain six distinct golfers.")
        fields.append([ids[n] for n in chosen])
    return pd.DataFrame(fields,columns=["G"]*6).to_csv(index=False).encode("utf-8-sig")

def evaluate_groups(groups,model,round_no,live=None,seed=42,sims=25000,tickets=5,
                    two_ball_tie_rule="loss"):
    if groups.empty:
        raise ValueError("No verified round groups available.")
    if round_no>1 and (live is None or live.empty):
        raise ValueError("Round 2–4 requires actual previous-round scores.")
    rates_input,provenance=group_ratings_source(model)
    ratings=build_round_ratings(rates_input,live,round_no,
        tournament_name=model["metadata"]["tournament"],use_course_dna=True)
    if round_no>1:
        covered=int((ratings["live_rounds"]>0).sum())
        if covered < max(3,int(len(ratings)*.5)):
            raise ValueError(f"Only {covered}/{len(ratings)} golfers matched completed-round scoring evidence.")
    # Streamlit can retain a shadowed or stale round_parlay import across reruns.
    # Preserve the selected sportsbook tie settlement; never silently drop the
    # keyword or revert to the legacy default when an old function is loaded.
    group_simulator = simulate_groups
    if "two_ball_tie_rule" not in inspect.signature(group_simulator).parameters:
        local_module_path = Path(__file__).resolve().with_name("round_parlay.py")
        spec = importlib.util.spec_from_file_location(
            "_ai2_pga_local_round_parlay", local_module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("PGA group simulator source could not be loaded.")
        local_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(local_module)
        group_simulator = local_module.simulate_groups
        if "two_ball_tie_rule" not in inspect.signature(group_simulator).parameters:
            raise RuntimeError(
                "PGA 2-ball settlement simulation is out of date. "
                "Redeploy the preview app before running matchups.")
    probs,_=group_simulator(groups,ratings,n_sims=int(sims),seed=int(seed),
                           two_ball_tie_rule=two_ball_tie_rule)
    summary,legs=build_six_leg_tickets(probs,ticket_count=int(tickets))
    return {"ratings":ratings,"probabilities":probs,"ticket_summary":summary,
        "ticket_legs":legs,"source":provenance,
        "two_ball_tie_rule":two_ball_tie_rule}

def bookmaker_filtered_tickets(probs,groups,offered_ids,ticket_count=3):
    """Build tickets only from user-confirmed, actually offered sportsbook groups.

    Tee-time availability DOES NOT establish sportsbook market availability.
    A golfer from an unavailable group must never be offered as a parlay leg.
    Unknown IDs are errors, not silently ignored; no picks until market
    availability has been explicitly provided for this event and round.
    """
    if groups is None or groups.empty or 'group' not in groups:
        raise ValueError("Verified tee groups are required before sportsbook filtering.")
    available={int(x) for x in groups['group']}
    chosen=set()
    for raw in offered_ids or []:
        if isinstance(raw,bool):
            raise ValueError("Invalid offered group ID.")
        gid=int(raw)
        if gid not in available:
            raise ValueError(f"Sportsbook group {gid} is not in this round's official tee groups.")
        chosen.add(gid)
    subset=probs.loc[probs['group'].isin(chosen)].copy()
    if len(chosen)<6:
        return subset,pd.DataFrame(),pd.DataFrame()
    summary,legs=build_six_leg_tickets(subset,ticket_count=int(ticket_count))
    if not legs.empty and not set(legs['group']).issubset(chosen):
        raise ValueError("A generated pick was not in the sportsbook availability list.")
    return subset,summary,legs

def build_round_showdown(dk,model,round_no,live,seed=42,count=4,exposure=.5,salary_floor=46500):
    if int(round_no) not in (2,3,4):
        raise ValueError("Showdown requires round 2, 3, or 4.")
    if live is None or live.empty:
        raise ValueError("Provide real completed-round scores.")
    input_stats,provenance=group_ratings_source(model)
    ratings=build_round_ratings(input_stats,live,int(round_no),
        tournament_name=model["metadata"]["tournament"],use_course_dna=True)
    d=parse_dk(dk)
    pool=build_showdown_pool(d,ratings,int(round_no))
    covered=int((pool["live_rounds"]>0).sum())
    if covered<max(3,int(len(pool)*.5)):
        raise ValueError("Fewer than half of Showdown golfers have verified previous-round data.")
    lineups,exposures=optimize_showdown(pool,int(count),float(exposure),int(salary_floor),seed=int(seed))
    return {"lineups":lineups,"exposure":exposures,"pool":pool,"source":provenance}
