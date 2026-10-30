"""One-page PGA V11 adapter for Team AI² Pro, with explicit data provenance.

The original PGA predictor was CSV/OTIS driven. The unified UI accepts a
current DraftKings field alone for market-only research; optional genuine
observed golfer statistics unlock the full historical/course-fit model.
Missing strokes gained or official hole pars are never fabricated.
"""
from __future__ import annotations
from datetime import datetime, timezone
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
    if not otis.empty:
        return otis,"observed OTIS inputs; independent model course-adjusted"
    p=model["projections"][["player","expected_round"]].copy()
    z=(p["expected_round"]-p["expected_round"].mean())/max(float(p["expected_round"].std(ddof=0)),.01)
    p["otis_true_skill"]=(50.-9.*z).clip(20,80)
    p["otis_course_fit"]=50.
    p["otis_form"]=50.
    return p[["player","otis_true_skill","otis_course_fit","otis_form"]], "DK salary-derived proxy ONLY, not observed SG or independent course fit"

def run_tournament(dk,tournament,course_par,no_cut,cut_size=65,hole_frame=None,
                   otis_raw=None,sims=3000,seed=42,lineup_count=10,
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
    if not otis.empty:
        # No silent use of stale repo results/course-history files.
        base=predict_from_dataframes(
            PGAConfig(tournament=str(tournament),sims=int(sims),seed=int(seed)),
            dk[["player","salary"]],otis,
            results=pd.DataFrame(),history=pd.DataFrame(),holes=pd.DataFrame(),
            weather=pd.DataFrame(),
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
        "exposure":exposure,"otis":otis,"raw_draws":draws,
        "diagnostics":diagnostics,
        "metadata":{"tournament":str(tournament),"course_par":int(course_par),
            "hole_pars_verified":holes is not None,"field_size":len(dk),
            "simulations":int(sims),"seed":int(seed),
            "otis_match":matched,"source_mode":mode,
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

def evaluate_groups(groups,model,round_no,live=None,seed=42,sims=25000,tickets=5):
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
    probs,_=simulate_groups(groups,ratings,n_sims=int(sims),seed=int(seed))
    summary,legs=build_six_leg_tickets(probs,ticket_count=int(tickets))
    return {"ratings":ratings,"probabilities":probs,"ticket_summary":summary,
        "ticket_legs":legs,"source":provenance}

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
