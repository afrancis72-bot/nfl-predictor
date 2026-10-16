"""Experimental DraftKings classic lineup optimizer; DST scores are manual assumptions."""
import math
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint

def build_classic_lineups(salaries, projections, *, dst_points, count=5, salary_floor=47500,
                          locked_dst=None, excluded_names=(), max_overlap=7,
                          max_exposure=1.0, upside_weight=0.15, variance_penalty=0.0):
    pool=salaries.copy()
    pool["Position"]=pool["Position"].astype(str).str.upper().str.strip().replace({"D":"DST"})
    pool["Salary"]=pd.to_numeric(pool["Salary"],errors="raise")
    pool["ID"]=pool["ID"].astype(str)
    pool=pool[pool["Position"].isin(["QB","RB","WR","TE","DST"]) & ~pool["Name"].isin(excluded_names)].copy()
    if pool["ID"].duplicated().any(): raise ValueError("Duplicate player IDs in DK salaries")
    proj=projections[["ID","dk_points_mean","dk_points_p90"]].copy()
    proj["ID"]=proj["ID"].astype(str)
    pool=pool.merge(proj,on="ID",how="left",validate="one_to_one")
    pool=pool[(pool["Position"]=="DST") | pool["dk_points_mean"].notna()].copy()
    pool["score"]=np.where(pool["Position"]=="DST",pool["TeamAbbrev"].map(dst_points),pool["dk_points_mean"])
    pool=pool[pool["score"].notna()].reset_index(drop=True)
    if locked_dst and not ((pool.Position=="DST")&(pool.TeamAbbrev==locked_dst)).any():
        raise ValueError(f"Locked DST {locked_dst} not found in DK salary file")
    if len(pool)<9: raise ValueError("Insufficient eligible players to build lineups")
    n=len(pool)
    rows=[]; lower=[]; upper=[]
    def constraint(mask,lo,hi):
        rows.append(np.asarray(mask,dtype=float));lower.append(lo);upper.append(hi)
    pos=pool.Position.to_numpy()
    teams=pool.TeamAbbrev.astype(str).to_numpy()
    salary=pool.Salary.to_numpy(dtype=float)
    constraint(np.ones(n),9,9)
    for p,lo,hi in [("QB",1,1),("DST",1,1),("RB",2,3),("WR",3,4),("TE",1,2)]:
        constraint(pos==p,lo,hi)
    constraint(salary,salary_floor,50000)
    for t in sorted(set(teams)):
        constraint((pos=="RB")&(teams==t),0,1)
    if locked_dst: constraint((pos=="DST")&(teams==locked_dst),1,1)
    # Avoid QB versus opposing DST using the DK matchup string.
    if "Game Info" in pool:
        matchup=pool["Game Info"].astype(str).str.extract(r"^([A-Z]{2,3})@([A-Z]{2,3})")
        opponents={}
        for a,b in matchup.dropna().itertuples(index=False,name=None):
            opponents[a]=b;opponents[b]=a
        for t in sorted(set(teams)):
            opp=opponents.get(t)
            if opp:
                constraint(((pos=="QB")&(teams==t))|((pos=="DST")&(teams==opp)),0,1)
    if not 0 < max_exposure <= 1: raise ValueError("Exposure must be between 0 and 100%")
    cap=max(1,math.floor(int(count)*max_exposure+1e-9))
    # A locked defense is intentionally exempt from the general exposure cap.
    exposure_counts={}
    results=[]
    for k in range(int(count)):
        # Later lineups reward p90 upside modestly and penalize repeated players.
        scores=pool.score.to_numpy(dtype=float).copy()
        ceiling=pool.dk_points_p90.fillna(pool.score).to_numpy(dtype=float)
        spread=np.maximum(0,ceiling-scores)
        if k%4==3:
            scores+=upside_weight*spread
        else:
            scores-=variance_penalty*spread
        for prior in results:
            ids=set(prior.ID.astype(str))
            scores-=0.65*pool.ID.isin(ids).to_numpy(dtype=float)
        # Enforce actual player exposure, not merely a soft diversification penalty.
        remaining=np.ones(n)
        for idx, player_id in enumerate(pool.ID):
            if exposure_counts.get(player_id,0)>=cap and not (locked_dst and pos[idx]=="DST" and teams[idx]==locked_dst):
                remaining[idx]=0
        matrix=np.stack(rows)
        result=milp(c=-scores,integrality=np.ones(n),bounds=Bounds(np.zeros(n),remaining),
                    constraints=LinearConstraint(matrix,np.array(lower),np.array(upper)),
                    options={"time_limit":20})
        if result.x is None: 
            if not results: raise ValueError("No legal lineup found. Check salary floor, DST lock, and eligible players.")
            raise ValueError(f"Only {len(results)} of {count} lineups feasible at {max_exposure:.0%} exposure; try a higher cap or lower salary floor.")
        chosen=pool.loc[result.x>0.5].copy()
        if len(chosen)!=9: raise ValueError("Optimizer returned an incomplete lineup")
        results.append(chosen)
        for player_id in chosen.ID: exposure_counts[player_id]=exposure_counts.get(player_id,0)+1
        constraint(pool.ID.isin(chosen.ID).to_numpy(dtype=float),0,max_overlap)
    if not results: raise ValueError("No feasible lineup")
    output=[]
    for i,chosen in enumerate(results,1):
        for p in ["QB","RB","WR","TE","DST"]:
            subset=chosen[chosen.Position==p].copy()
            needed={"QB":1,"RB":2,"WR":3,"TE":1,"DST":1}[p]
            for _,r in subset.iloc[:needed].iterrows():
                output.append({"lineup":i,"slot":p,"Name":r.Name if False else r["Name"],"ID":r["ID"],
                               "team":r.TeamAbbrev,"salary":int(r.Salary),"projected_points":float(r.score)})
        used=output[-8:]
        used_ids={x["ID"] for x in used}
        flex=chosen[~chosen.ID.isin(used_ids)]
        if len(flex)!=1: raise ValueError("Invalid FLEX allocation")
        r=flex.iloc[0]
        output.append({"lineup":i,"slot":"FLEX","Name":r["Name"],"ID":r.ID,
                       "team":r.TeamAbbrev,"salary":int(r.Salary),"projected_points":float(r.score)})
    return pd.DataFrame(output)
