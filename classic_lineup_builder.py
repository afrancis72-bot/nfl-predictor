"""Experimental DraftKings classic portfolio optimizer with automatic DST inputs."""
import math
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint

def build_classic_lineups(salaries, projections, *, dst_points, count=5, salary_floor=47500,
                          locked_dst=None, excluded_names=(), max_overlap=6,
                          max_exposure=1.0, upside_weight=0.15, variance_penalty=0.0, ownership=None, leverage_weight=0.0):
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
    if ownership is not None:
        own=ownership.copy()
        if not {"ID","ownership_pct"}.issubset(own.columns): raise ValueError("Ownership CSV needs ID and ownership_pct columns")
        own["ID"]=own["ID"].astype(str).str.replace(r"\.0$","",regex=True).str.strip()
        own["ownership_pct"]=pd.to_numeric(own["ownership_pct"],errors="raise")
        if own["ID"].duplicated().any() or own["ownership_pct"].isna().any() or (~own["ownership_pct"].between(0,100)).any():
            raise ValueError("Ownership IDs must be unique and percentages between 0 and 100")
        pool=pool.merge(own[["ID","ownership_pct"]],on="ID",how="left",validate="one_to_one")
    else:
        pool["ownership_pct"]=np.nan
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
    # Two-stage portfolio optimizer: generate independently, then select ALL
    # lineups together. This prevents early lineups consuming chalk exposure.
    from scipy import sparse
    rng=np.random.default_rng(5100)
    base=pool.score.to_numpy(dtype=float)
    ceiling=pool.dk_points_p90.fillna(pool.score).to_numpy(dtype=float)
    spread=np.maximum(0,ceiling-base)
    own=pool.ownership_pct.fillna(20).to_numpy(dtype=float)
    known=pool.ownership_pct.notna().to_numpy()
    matrix=sparse.csr_matrix(np.stack(rows))
    lineup_constraints=LinearConstraint(matrix,np.asarray(lower),np.asarray(upper))
    optimum=milp(c=-base,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
                 constraints=lineup_constraints,options={"time_limit":30,"mip_rel_gap":0.0})
    if optimum.x is None or optimum.status!=0:
        raise ValueError("Could not certify the highest-projected legal lineup.")
    optimal_ids=frozenset(np.flatnonzero(optimum.x>0.5).tolist())
    if len(optimal_ids)!=9: raise ValueError("Exact optimum returned an incomplete lineup")
    results=[pool.iloc[sorted(optimal_ids)].copy()] if int(count)==1 else None
    candidate_bank={optimal_ids:(float(base[list(optimal_ids)].sum()),
                                 float((base+float(upside_weight)*spread)[list(optimal_ids)].sum()))}
    recent=[]
    best_mean=None
    target=max(180,int(count)*16)
    for attempt in range(target):
        upside_candidate=attempt%4==3
        # Small perturbations diversify credible alternatives; ownership
        # never replaces fantasy-point projection as the primary signal.
        scores=base.copy()
        if upside_candidate:
            scores+=float(upside_weight)*spread
            if ownership is not None:
                scores+=np.where(known,0.15*float(leverage_weight)*(20-own)/20,0)
        else:
            scores-=float(variance_penalty)*spread
        scores+=rng.normal(0,1.5+(attempt%7)*0.55,size=n)
        if recent:
            frequency=np.zeros(n)
            for ids in recent[-12:]:
                frequency[list(ids)]+=1
            scores-=0.30*frequency
        res=milp(c=-scores,integrality=np.ones(n),
                 bounds=Bounds(np.zeros(n),np.ones(n)),
                 constraints=lineup_constraints,options={"time_limit":1.5,"mip_rel_gap":0.03})
        if res.x is None: continue
        ids=frozenset(np.flatnonzero(res.x>0.5).tolist())
        if len(ids)!=9: continue
        mean=float(base[list(ids)].sum())
        high=float((base+float(upside_weight)*spread)[list(ids)].sum())
        recent.append(ids)
        if best_mean is None or mean>best_mean: best_mean=mean
        # A candidate can be eligible for either allocation; classify
        # based on the selection variable, not its generation pass.
        if ids not in candidate_bank:
            candidate_bank[ids]=(mean,high)
    if results is None:
        if not candidate_bank:
            raise ValueError("No legal candidate lineups. Check salary floor and eligible players.")
        candidates=list(candidate_bank)
        m=len(candidates)
        if m<int(count):
            raise ValueError(f"Only {m} distinct candidate lineups found; cannot select {count}.")
        incidence=sparse.lil_matrix((n,m),dtype=float)
        for j,ids in enumerate(candidates):
            incidence[list(ids),j]=1
        incidence=incidence.tocsr()
        # Variables: first m are balanced candidates; next m are upside
        # candidates. A candidate cannot be chosen twice.
        upside_count=int(round(int(count)*0.25))
        balanced_count=int(count)-upside_count
        cap=max(1,math.floor(int(count)*max_exposure+1e-9))
        cap_vec=np.full(n,cap,dtype=float)
        if locked_dst:
            cap_vec[(pos=="DST")&(teams==locked_dst)]=int(count)
        means=np.array([candidate_bank[c][0] for c in candidates])
        highs=np.array([candidate_bank[c][1] for c in candidates])
        # Reject extremely weak pivots relative to the best candidate.
        # Thresholds relax only if no legal portfolio exists.
        from scipy.sparse import vstack,hstack
        unique_rows=[]
        for j in range(m):
            unique_rows.append((j,j+m))
        overlaps=incidence.T@incidence
        conflict_pairs=[]
        coo=sparse.triu(overlaps,k=1).tocoo()
        for j,k,v in zip(coo.row,coo.col,coo.data):
            if v>max_overlap: conflict_pairs.append((int(j),int(k)))
        def portfolio_select(floor_ratio):
            constraints_rows=[];lo=[];hi=[]
            def add(indices,values,minimum,maximum):
                constraints_rows.append(sparse.csr_matrix((values,([0]*len(indices),indices)),shape=(1,2*m)))
                lo.append(minimum);hi.append(maximum)
            add(list(range(m)),[1.]*m,balanced_count,balanced_count)\n        add([0,m],[1,1],1,1)  # The exact optimum must be in the portfolio.
            add(list(range(m,2*m)),[1.]*m,upside_count,upside_count)
            for j,k in unique_rows:
                add([j,k],[1,1],0,1)
            for j,k in conflict_pairs:
                add([j,k,j+m,k+m],[1,1,1,1],0,1)
            # Exposure across both balanced and upside slots.
            exp=sparse.hstack([incidence,incidence],format="csr")
            constraints_rows.append(exp)
            lo.extend([0.]*n);hi.extend(cap_vec.tolist())
            permitted_balanced=means>=max(means)*floor_ratio
            permitted_upside=means>=max(means)*max(0.70,floor_ratio-0.07)
            upper_bounds=np.concatenate([permitted_balanced,permitted_upside]).astype(float)
            utility=np.concatenate([means,highs])
            res=milp(c=-utility,integrality=np.ones(2*m),
                     bounds=Bounds(np.zeros(2*m),upper_bounds),
                     constraints=LinearConstraint(sparse.vstack(constraints_rows,format="csr"),
                         np.asarray(lo),np.asarray(hi)),
                     options={"time_limit":35,"mip_rel_gap":0.04})
            return res
        selected=None
        for threshold in (0.88,0.84,0.80,0.75,0.0):
            solution=portfolio_select(threshold)
            if solution.x is not None and np.count_nonzero(solution.x>0.5)==int(count):
                selected=solution
                break
        if selected is None:
            raise ValueError(f"Could not select {count} diverse lineups under {max_exposure:.0%} exposure from {m} candidates. Try reducing lineup count or salary floor.")
        selected_balanced=np.flatnonzero(selected.x[:m]>0.5)
        selected_upside=np.flatnonzero(selected.x[m:]>0.5)
        chosen_indices=[0]+[j for j in list(selected_balanced)+list(selected_upside) if j!=0]
        results=[pool.iloc[sorted(candidates[j])].copy() for j in chosen_indices]
        # Verify global constraints before exporting.
        exposure_counts={}
        for chosen in results:
            for player_id in chosen.ID:
                exposure_counts[player_id]=exposure_counts.get(player_id,0)+1
        for player_id,frequency in exposure_counts.items():
            if frequency>cap and not (locked_dst and
                ((pool.ID==player_id)&(pool.Position=="DST")&(pool.TeamAbbrev==locked_dst)).any()):
                raise ValueError("Portfolio exposure validation failed")
        for i in range(len(results)):
            for j in range(i+1,len(results)):
                if len(set(results[i].ID)&set(results[j].ID))>max_overlap:
                    raise ValueError("Portfolio overlap validation failed")
    output=[]
    for i,chosen in enumerate(results,1):
        for p in ["QB","RB","WR","TE","DST"]:
            subset=chosen[chosen.Position==p].copy()
            needed={"QB":1,"RB":2,"WR":3,"TE":1,"DST":1}[p]
            for _,r in subset.iloc[:needed].iterrows():
                output.append({"lineup":i,"slot":p,"Name":r.Name if False else r["Name"],"ID":r["ID"],
                               "team":r.TeamAbbrev,"salary":int(r.Salary),"projected_points":float(r.score),"ownership_pct":float(r.ownership_pct) if pd.notna(r.ownership_pct) else float("nan")})
        used=output[-8:]
        used_ids={x["ID"] for x in used}
        flex=chosen[~chosen.ID.isin(used_ids)]
        if len(flex)!=1: raise ValueError("Invalid FLEX allocation")
        r=flex.iloc[0]
        output.append({"lineup":i,"slot":"FLEX","Name":r["Name"],"ID":r.ID,
                       "team":r.TeamAbbrev,"salary":int(r.Salary),"projected_points":float(r.score),
                       "ownership_pct":float(r.ownership_pct) if pd.notna(r.ownership_pct) else float("nan")})
    return pd.DataFrame(output)
