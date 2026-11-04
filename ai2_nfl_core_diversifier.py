"""One-click NFL Classic multi-tier selection that diversifies *offensive cores*.

The model, not the user, must supply independently stat-simulated legal
candidate lineups. This reducer never invents new player forecasts and never
asserts estimated ownership. A candidate universe can combine separate seeds
of the SAME current unified NFL engine. Missing pivot candidates cause a
visible infeasibility result, not relaxed salary/role/quality constraints.

The $3 / $0.50 / $0.10 tiers are the user's requested contest-entry mapping,
not forecasts of profit or contest-specific ROI.
"""
from collections import defaultdict
from copy import deepcopy
from math import floor
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from ai2_nfl_portfolio_risk import (
    maximum_player_appearances, uncertain_running_back_reason,
    MAX_SHARED_PLAYERS,
)
from ai2_nfl_qb_audit import qb_and_partners
from ai2_nfl_lineup import check_classic_lineup


TIERS = (("$3", 4), ("$0.50", 16), ("$0.10", 20))
COUNT = 40
ROLE_FLAG_CAP = 8
HIGH_SALARY_CORE_CAP = 9
DST_CAP = 9


def _id(roster):
    return tuple(sorted(str(p["ID"]) for p in roster))


def _top_salary_core(roster):
    skill = [p for p in roster if str(p["Position"]).upper() != "DST"]
    return {str(p["ID"]) for p in sorted(skill,
               key=lambda p: float(p["Salary"]), reverse=True)[:5]}


def diversify_40_from_stat_simulation(candidate_portfolios, *, role_anomalies=(),
                                      milp_time_limit=45.):
    """Return exactly 4/16/20 distinct entries from verified simulated lineups.

    Enforces total-player 28% ceilings and stricter observed RB role limits.
    A repeat high-salary offensive core is NOT rescued by swapping DST/FLEX.
    Premium QB and DST must be unique and high-stakes rosters are dissimilar.
    """
    all_items = []
    for portfolio in candidate_portfolios:
        all_items.extend(portfolio)
    if len(all_items) < COUNT:
        raise ValueError("Need at least 40 stat-simulated candidates for 4/16/20.")
    # Separate seeds can yield the exact same roster. Treat one physical
    # nine-player roster as ONE opportunity, not two independent candidates.
    unique = {}
    for item in all_items:
        r = item["candidate"]["players"]
        if len(r) != 9 or check_classic_lineup(r,salary_floor=49600,
                                               salary_cap=50000):
            raise ValueError("Unverified or illegal candidate in diverse NFL pool")
        key = _id(r)
        if key not in unique or float(item["simulated_p90"]) > float(
                unique[key]["simulated_p90"]):
            unique[key] = item
    scored = list(unique.values())
    n = len(scored)
    if n < COUNT:
        raise ValueError("Candidate sets have fewer than 40 unique legal rosters")
    rosters = [item["candidate"]["players"] for item in scored]
    roster_ids = [set(_id(r)) for r in rosters]
    core = [_top_salary_core(r) for r in rosters]
    players = {str(p["ID"]):p for r in rosters for p in r}
    appearances = defaultdict(list)
    for i,ids in enumerate(roster_ids):
        for p in ids:
            appearances[p].append(i)
    p_ids = sorted(players)
    role_names = {str(x).casefold().strip()
                  for x in role_anomalies if str(x).strip()}
    flagged = {pid for pid in p_ids if
               str(players[pid]["Name"]).casefold().strip() in role_names}
    expensive = {pid for pid in p_ids if
                 str(players[pid]["Position"]).upper() in {"RB","WR","TE"}
                 and float(players[pid]["Salary"]) >= 8000}
    premium_role_risk = [
        i for i,r in enumerate(rosters)
        if any(uncertain_running_back_reason(p) for p in r)]
    tsize = len(TIERS)
    num_vars = tsize*n
    row_index, columns, coeff, lower, upper = [], [], [], [], []
    def constraint(terms, lo=0., hi=np.inf):
        r = len(lower)
        lower.append(float(lo));upper.append(float(hi))
        for c,v in terms:
            row_index.append(r); columns.append(int(c)); coeff.append(float(v))
    # Four high-stakes, sixteen balanced, twenty lower-stakes entries.
    for tier,(_,quota) in enumerate(TIERS):
        constraint([(tier*n+i,1) for i in range(n)],quota,quota)
    for i in range(n):
        constraint([(tier*n+i,1) for tier in range(tsize)],0,1)
    max_any = floor(COUNT * .28)
    for pid in p_ids:
        p = players[pid]
        cap = min(max_any,maximum_player_appearances(p,COUNT,.28))
        if pid in flagged:
            cap = min(cap,ROLE_FLAG_CAP)
        if pid in expensive:
            cap = min(cap,HIGH_SALARY_CORE_CAP)
        if str(p["Position"]).upper() == "DST":
            cap = min(cap,DST_CAP)
        constraint([(tier*n+i,1) for tier in range(tsize)
                    for i in appearances[pid]],0,cap)
        # Four premium entries need separate paths to winning, including
        # a maximum of one exposed role-anomaly player.
        constraint([(i,1) for i in appearances[pid]],
                   0,1 if pid in flagged else 2)
    # Premium entries must have a genuine quarterback receiving stack.
    for i,roster in enumerate(rosters):
        qb,partners=qb_and_partners(roster)
        if not qb or not partners:
            constraint([(i,1)],0,0)
        # A premium lineup should not rely on an inflated RB workload
        # *together with* one expensive WR, increasing joint dependency.
        if roster_ids[i]&flagged and roster_ids[i]&expensive:
            constraint([(i,1)],0,0)
    # A 2-game RB role or returning-starter dependency is not verified
    # enough for the four highest-stakes entries: allow such upside in the
    # $0.50 / $0.10 portfolios under strict 8%-12% exposure caps instead.
    if premium_role_risk:
        constraint([(i,1) for i in premium_role_risk],0,0)
    # Avoid four premium entries using one favorite QB or cheap defense.
    for pid in p_ids:
        if str(players[pid]["Position"]).upper() in {"QB","DST"}:
            constraint([(i,1) for i in appearances[pid]],0,1)
    for i in range(n):
        for j in range(i+1,n):
            shared=len(roster_ids[i]&roster_ids[j])
            shared_core=len(core[i]&core[j])
            if shared>3 or shared_core>2:
                constraint([(i,1),(j,1)],0,1)
            # Four of the same five highest-salaried offensive players
            # isn't meaningful differentiation just because DST changed.
            if shared_core>=4:
                constraint([(tier*n+i,1) for tier in range(tsize)]+
                           [(tier*n+j,1) for tier in range(tsize)],0,1)
    # Repeated anomaly + star core pairing across ALL contests has
    # compounded failure risk even when each exposure is individually legal.
    for a in flagged:
        for b in expensive:
            overlapping = list(set(appearances[a]) & set(appearances[b]))
            if overlapping:
                constraint([(tier*n+i,1) for tier in range(tsize)
                            for i in overlapping],0,3)
    # Work exclusively from independently simulated player-stat outputs,
    # with conservative premiums that still value ceiling.
    metrics=np.asarray([
        [float(item[field]) for item in scored]
        for field in ("simulated_mean","simulated_p90","simulated_p99")])
    spread=np.maximum(metrics.max(axis=1)-metrics.min(axis=1),1.)
    normalized=(metrics-metrics.min(axis=1)[:,None])/spread[:,None]
    mixes=((.20,.35,.45),(.50,.35,.15),(.25,.30,.45))
    tier_value=(2.2,1.,.65)
    costs=[]
    for tier in range(tsize):
        scores=(np.asarray(mixes[tier])@normalized)*tier_value[tier]
        for i,ids in enumerate(roster_ids):
            scores[i]-=(.28,.12,.03)[tier]*len(ids&flagged)
            if tier==0 and ids&flagged and ids&expensive:
                scores[i]-=.25
        costs.extend(-scores)
    matrix=coo_matrix((coeff,(row_index,columns)),
                      shape=(len(lower),num_vars)).tocsr()
    solution=milp(c=np.asarray(costs),
                  integrality=np.ones(num_vars,dtype=int),
                  bounds=Bounds(np.zeros(num_vars),np.ones(num_vars)),
                  constraints=LinearConstraint(matrix,lower,upper),
                  options={"time_limit":float(milp_time_limit),"mip_rel_gap":.02})
    if solution.x is None:
        raise ValueError(
            f"Insufficient independent modeled candidates for honest 4/16/20: "
            f"solver status {solution.status}, {solution.message}. "
            "Generate additional seeded portfolios; do NOT pad with punts.")
    x=np.rint(solution.x).astype(int)
    actual=matrix@x
    if (not np.all(np.abs(solution.x-x)<1e-5)
            or np.any(actual<np.array(lower)-1e-5)
            or np.any(actual>np.array(upper)+1e-5)):
        raise ValueError("Diversified MILP yielded an invalid integer portfolio")
    selected=[]
    for tier,(label,quota) in enumerate(TIERS):
        chosen=[i for i in range(n) if x[tier*n+i]==1]
        if len(chosen)!=quota:
            raise ValueError("Incomplete tier assignment; refusing export")
        for i in chosen:
            result=deepcopy(scored[i])
            result["contest_tier"]=label
            result["source_candidate_id"]=i
            selected.append(result)
    max_premium_shared=max(
        (len(set(_id(a["candidate"]["players"])) &
             set(_id(b["candidate"]["players"])))
         for i,a in enumerate(selected[:4]) for b in selected[i+1:4]),default=0)
    return selected, {
        "source_lineup_count":len(all_items),
        "unique_candidate_count":n,
        "num_selected":len(selected),
        "premium_max_shared":max_premium_shared,
        "milp_solver_status":solution.status,
        "milp_message":str(solution.message),
        "role_anomalies":sorted(role_names),
        "real_ownership_data_available":False,
        "method":"Actual stat-simulation means/P90/P99 with caps on offensive core repetitions",
        "note":"No current-week news ingestion / opponent or ownership calibration added by this selector",
    }
