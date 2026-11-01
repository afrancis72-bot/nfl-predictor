"""NASCAR one-click adapter: strict DK post-qualifying data + V2.14 race engine."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

from nascar_reference_dna import build_reference_dna, blend_driver_dna
from nascar_predictor_pro import (Config, build_features, simulate,
    optimize_scenario_portfolio, portfolio_exposure, scenario_summary)

ROOT=Path(__file__).resolve().parent
PRACTICE_COLS=("single_lap_rank","avg5_rank","avg10_rank","avg15_rank","avg20_rank")

def track_catalog():
    return pd.read_csv(ROOT/"track_catalog.csv")

def _colkey(s):
    return re.sub("[^a-z0-9]","",str(s).lower().strip())

def _namekey(s):
    return re.sub("[^a-z0-9]","",unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode().lower())

# Strict, known driver-name aliases only; do NOT fuzzy-match unrelated drivers.
_PRACTICE_ALIASES={
    "johnhnemechek":"johnhunternemechek",
    "jhnemechek":"johnhunternemechek",
    "rickystenhousejr":"rickystenhouse",
    "rickystenhousejunior":"rickystenhouse",
    "chrisbell":"christopherbell",
    "willbyron":"williambyron",
}

def _practice_namekey(s):
    key=_namekey(s)
    return _PRACTICE_ALIASES.get(key,key)

def load_dk_slate(data):
    """Accept actual driver IDs, salaries and FULL qualifying grid only."""
    raw=data.copy() if isinstance(data,pd.DataFrame) else pd.read_csv(data)
    if raw.empty:
        raise ValueError("The DraftKings salary CSV is empty.")
    cols={_colkey(c):c for c in raw.columns}
    def find(names):
        return next((cols[_colkey(n)] for n in names if _colkey(n) in cols),None)
    mapping={
        "Name":find(("Name","Driver","Driver Name","Player Name")),
        "ID":find(("ID","Player ID","Driver ID")),
        "Salary":find(("Salary","DK Salary")),
        "Starting":find(("Starting","Starting Position","Start Pos","Start","Start Position","Grid","Qualifying Position")),
        "AvgPointsPerGame":find(("AvgPointsPerGame","AvgPoints","FPPG","Fantasy Points Per Game")),
    }
    for key in ("Name","ID","Salary","Starting"):
        if not mapping[key]:
            raise ValueError("Missing "+key+". Upload the DraftKings driver SALARY CSV AFTER qualifying with a complete starting grid.")
    dk=pd.DataFrame({k:raw[v] for k,v in mapping.items() if v is not None})
    if "AvgPointsPerGame" not in dk:
        dk["AvgPointsPerGame"]=0.0
    dk["Name"]=dk["Name"].astype(str).str.strip()
    dk["ID"]=dk["ID"].astype(str).str.replace(r"\.0$","",regex=True).str.strip()
    for k in ("Salary","Starting","AvgPointsPerGame"):
        dk[k]=pd.to_numeric(dk[k],errors="coerce")
    n=len(dk)
    if n<12 or n>45:
        raise ValueError(f"Unusual NASCAR driver field of {n}; verify the correct series and DK file.")
    if dk["Name"].isin(("","nan","None")).any() or dk["Name"].map(_namekey).duplicated().any():
        raise ValueError("Driver names must be nonempty and unique.")
    if dk["ID"].isin(("","nan","None")).any() or dk["ID"].duplicated().any():
        raise ValueError("Each driver needs a unique real DraftKings player ID.")
    if dk["Salary"].isna().any() or (dk["Salary"]<=0).any():
        raise ValueError("All DraftKings salaries must be positive numbers.")
    start=dk["Starting"]
    if start.isna().any() or (~start.between(1,n)).any() or ((start%1)!=0).any():
        raise ValueError("DK starting positions must be whole numbers in the range 1 to field size.")
    expected=set(range(1,n+1))
    if set(start.astype(int))!=expected:
        raise ValueError(f"DraftKings starting positions must form a unique complete 1–{n} qualifying grid.")
    dk["Starting"]=start.astype(int)
    dk["AvgPointsPerGame"]=dk["AvgPointsPerGame"].fillna(0.0)
    return dk.reset_index(drop=True)

def _speed_to_rank(df, source_col):
    speeds=pd.to_numeric(df[source_col],errors="coerce")
    if (speeds.dropna()<0).any():
        raise ValueError("Practice speed cannot be negative.")
    return speeds.rank(ascending=False,method="min")

def practice_updates(raw,dk,return_report=False):
    """Match published practice ranks to DK drivers, without changing qualifying.

    Ignore clearly unmatched extra rows (e.g. a driver not in the DK slate),
    but reject ambiguous/mostly wrong-series uploads rather than inventing data.
    The optional report makes every omission transparent in the unified app.
    """
    if raw is None:
        result=pd.DataFrame({"Name":dk["Name"],"qualifying_position":dk["Starting"]})
        report={"practice_rows":0,"matched_practice_rows":0,"unmatched_practice_names":[]}
        return (result,0,report) if return_report else (result,0)
    x=raw.copy()
    columns={_colkey(c):c for c in x.columns}
    name_col=next((columns.get(c) for c in ("name","driver","drivername") if c in columns),None)
    if not name_col:
        raise ValueError("Practice file needs a driver Name column.")
    x["key"]=x[name_col].map(_practice_namekey)
    if x["key"].eq("").any():
        raise ValueError("Practice driver names cannot be blank.")
    dk_keys=dk["Name"].map(_practice_namekey)
    if dk_keys.duplicated().any():
        raise ValueError("DraftKings driver names are ambiguous after standard name normalization.")
    unmatched=x.loc[~x["key"].isin(set(dk_keys)),name_col].astype(str).tolist()
    matched=x.loc[x["key"].isin(set(dk_keys))].copy()
    if matched["key"].duplicated().any():
        duplicated=matched.loc[matched["key"].duplicated(keep=False),name_col].astype(str).tolist()
        raise ValueError("Duplicate practice drivers after name normalization: "+", ".join(duplicated[:8]))
    if not len(matched):
        raise ValueError("No practice drivers matched the DraftKings slate. Verify the race/series and driver names.")
    if len(matched)/len(x)<0.70:
        raise ValueError(
            f"Practice file matched only {len(matched)}/{len(x)} listed drivers. "
            "Verify this is the SAME race and series as the DK slate. "
            "Unmatched: "+", ".join(unmatched[:8])
        )
    frame=dk[["Name","Starting"]].copy()
    frame["key"]=dk_keys
    frame=frame.merge(matched.drop(columns=[name_col]),on="key",how="left",validate="one_to_one")
    frame=frame.drop(columns=["key"]).rename(columns={"Starting":"qualifying_position"})
    speeds={
       "single_lap_rank":("single_lap_speed","single lap speed","best lap speed"),
       "avg5_rank":("avg5_speed","5 lap average speed"),
       "avg10_rank":("avg10_speed","10 lap average speed"),
       "avg15_rank":("avg15_speed","15 lap average speed"),
       "avg20_rank":("avg20_speed","20 lap average speed"),
    }
    for target in PRACTICE_COLS:
        hit=next((columns.get(_colkey(c)) for c in (target,target.replace("_"," ")) if _colkey(c) in columns),None)
        if hit:
            frame[target]=pd.to_numeric(frame[hit],errors="coerce")
        else:
            hit=next((columns.get(_colkey(c)) for c in speeds[target] if _colkey(c) in columns),None)
            frame[target]=_speed_to_rank(frame,hit) if hit else np.nan
    present=int(frame[list(PRACTICE_COLS)].notna().any(axis=1).sum())
    if not present:
        raise ValueError("No usable actual practice rank or speed columns found; remove the practice upload or correct its headers.")
    report={"practice_rows":len(x),"matched_practice_rows":present,
            "unmatched_practice_names":unmatched,
            "dk_without_practice":frame.loc[~frame[list(PRACTICE_COLS)].notna().any(axis=1),"Name"].astype(str).tolist()}
    update=frame[["Name","qualifying_position"]+list(PRACTICE_COLS)]
    return (update,present,report) if return_report else (update,present)

def _baseline(dk):
    salary=dk["Salary"].rank(pct=True)
    fppg=dk["AvgPointsPerGame"].rank(pct=True)
    return pd.DataFrame({"Name":dk["Name"],
        "manual_speed_rating":100*(.45*salary+.55*fppg),
        "manual_dnf_risk":np.nan})

def reference_signals(dk,selected_track):
    """Fetch only published driver records. Failure means unknown, never zero."""
    ref=build_reference_dna(dk,selected_track)
    blended=blend_driver_dna(dk,ref)
    return ref,blended

def build_input_features(dk,track_name,laps,practice=None,reference=None):
    dk=load_dk_slate(dk)
    catalog=track_catalog()
    track=catalog.loc[catalog["track"]==track_name].copy()
    if len(track)!=1:
        raise ValueError("Select a track from the NASCAR track catalog.")
    if not 25<=int(laps)<=700:
        raise ValueError("Race laps must be between 25 and 700; confirm scheduled race distance.")
    track.loc[:,"laps"]=int(laps)
    update,practice_drivers,practice_report=practice_updates(practice,dk,return_report=True)
    prior=_baseline(dk)
    ref_count=0
    ref_coverage=0
    if reference is not None:
        ref,blended=reference
        ref_count=int(ref["reference_ok"].sum())
        ref_coverage=int((ref["reference_coverage"].fillna(0)>=.4).sum())
        ref_by_name=blended.set_index("Name")
        for col in ("season_avg_finish","recent_avg_finish","recent_avg_start",
                    "recent_place_diff","comparable_track_avg_finish",
                    "recent_form_score","place_diff_score","comparable_track_score",
                    "track_type_elo_score","reliability_score","driver_dna_score"):
            if col in blended:
                prior[col]=prior["Name"].map(ref_by_name[col])
        if "comparable_track_avg_finish" in prior:
            prior["track_type_avg_finish"]=prior["comparable_track_avg_finish"]
    features=build_features(dk,prior,track,update)
    if reference is not None:
        ref,blended=reference
        coverage=ref.set_index("Name")["reference_coverage"]
        features["data_coverage"]=features["Name"].map(coverage).fillna(0.0).clip(0,1)
        features["model_confidence"]=(.45+.45*features["data_coverage"]).clip(.45,.90)
    else:
        features["data_coverage"]=0.0
        features["model_confidence"]=0.45
    return features,{"practice_drivers":practice_drivers,
         "practice_report":practice_report,
         "reference_drivers":ref_count,"reference_coverage_drivers":ref_coverage,
         "verified_grid":True,"track":track_name,"laps":int(laps)}

def validate_portfolio(lineups,projections,count,exposure,overlap,floor=47000):
    if len(lineups)!=count:
        raise ValueError(f"Only {len(lineups)}/{count} lineups produced. No partial portfolio can be exported; reduce count or relax constraints.")
    roster={str(r.ID):r for r in projections.itertuples(index=False)}
    groups=[]
    for v in lineups["IDs"]:
        ids=[x.strip() for x in str(v).split("|")]
        if len(ids)!=6 or len(set(ids))!=6 or not set(ids).issubset(roster):
            raise ValueError("A generated lineup contains invalid, duplicate, or unknown DraftKings IDs.")
        salary=sum(int(roster[x].Salary) for x in ids)
        if not floor<=salary<=50000:
            raise ValueError(f"Generated lineup outside legal salary range: {salary}.")
        group=set(ids)
        if group in groups:
            raise ValueError("Duplicate NASCAR lineup generated.")
        if any(len(group&g)>overlap for g in groups):
            raise ValueError("Lineup overlap exceeded user limit.")
        groups.append(group)
    cap=int(np.floor(count*exposure+1e-9))
    if any(sum(x in group for group in groups)>cap for x in roster):
        raise ValueError("Driver exposure exceeded the portfolio cap.")
    return True

def dk_roster_csv(lineups):
    values=[[part.strip() for part in str(ids).split("|")]
            for ids in lineups["IDs"]]
    return pd.DataFrame(values,columns=["D"]*6).to_csv(index=False).encode("utf-8-sig")

def run_oneclick(dk,track,laps,sims=25000,seed=42,portfolio_size=10,
                 max_exposure=.35,max_overlap=4,salary_floor=47000,
                 practice=None,reference=None,excluded_drivers=()):
    if sims not in (1000,10000,25000,50000,100000):
        raise ValueError("Unsupported simulation count.")
    if not 1<=int(portfolio_size)<=50:
        raise ValueError("Portfolio size must be 1–50.")
    if not .15<=float(max_exposure)<=1:
        raise ValueError("Exposure cap must be between 15% and 100%.")
    f,meta=build_input_features(dk,track,laps,practice,reference)
    cfg=Config(sims=int(sims),seed=int(seed),laps=int(laps),salary_cap=50000,
               roster_size=6,max_overlap=int(max_overlap))
    projections,draws=simulate(f,cfg)
    if draws.shape!=(int(sims),len(projections)) or not np.allclose(
        draws.mean(0),projections["proj_dk"],atol=.005):
        raise ValueError("Simulation draw/driver alignment failed; no lineups exported.")
    full_scripts=scenario_summary(projections,draws)
    skipped={str(name).strip() for name in excluded_drivers}
    unknown=skipped-set(projections["Name"].astype(str))
    if unknown:
        raise ValueError("Unknown excluded NASCAR drivers: "+", ".join(sorted(unknown)[:5]))
    allowed=~projections["Name"].astype(str).isin(skipped)
    if int(allowed.sum())<6:
        raise ValueError("Driver exclusions leave fewer than six eligible racers.")
    optimizer_pool=projections.loc[allowed].reset_index(drop=True)
    optimizer_draws=draws[:,allowed.to_numpy()]
    # Excluded racers still compete in the 40-car simulated races.
    # They are removed from the DFS optimizer only, preserving race correlations.
    lineups=optimize_scenario_portfolio(
        optimizer_pool,optimizer_draws,n_lineups=int(portfolio_size),max_overlap=int(max_overlap),
        max_exposure=float(max_exposure),salary_cap=50000,roster_size=6,
        candidate_pool=500,seed=int(seed),salary_floor=int(salary_floor))
    validate_portfolio(lineups,optimizer_pool,int(portfolio_size),
                       float(max_exposure),int(max_overlap),int(salary_floor))
    return {"projections":projections,"lineups":lineups,
        "exposure":portfolio_exposure(lineups,optimizer_pool),
        "scripts":full_scripts,
        "roster_csv":dk_roster_csv(lineups),
        "metadata":{**meta,"sims":int(sims),"seed":int(seed),
                    "drivers":len(projections),"excluded":sorted(skipped),
                    "built_at":datetime.now(timezone.utc).isoformat()}}
