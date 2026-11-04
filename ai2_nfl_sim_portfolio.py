"""Score complete legal NFL Classic lineups on shared game simulations.

Scenario percentiles are not contest ROI or predicted cash rates.
"""
from collections import Counter
from math import ceil
import numpy as np
import pandas as pd
from ai2_nfl_lineup import check_classic_lineup
from ai2_nfl_role_evidence import player_opportunity_evidence
from ai2_nfl_qb_audit import qb_and_partners
from ai2_nfl_qb_counterfactual import quarterback_counterfactuals
from ai2_nfl_stack_intelligence import score_stack_lift
from ai2_nfl_correlation_guard import (opposing_pass_catcher_dst_pairs,
                                       opposing_dst_lineup_budget)
from ai2_nfl_portfolio_risk import (
    maximum_player_appearances, uncertain_running_back_reason,
    overlap_groups, overlap_conflicts, MAX_SHARED_PLAYERS,
)


from collections import defaultdict


def _solve_joint_portfolio(scored, *, count, premium_count, max_exposure,
                           upside_share, value_salary_ceiling,
                           value_max_appearances, premium_value_max_appearances,
                           premium_excluded_ids=frozenset()):
    """Find a feasible *whole* portfolio when sequential greedy selection dead-ends.

    Binary regular/premium decisions are optimized jointly, preserving hard
    DK eligibility, maximum player exposures and premium-value constraints.
    """
    try:
        from scipy.optimize import milp, Bounds, LinearConstraint
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise ValueError("SciPy MILP optimizer unavailable; no partial portfolio exported") from exc
    n = len(scored)
    if not n:
        raise ValueError("No legal candidates; no partial portfolio exported")
    appearances = defaultdict(set)
    player_records = {}
    cheap = set()
    for i, item in enumerate(scored):
        for p in item["candidate"]["players"]:
            ident = str(p["ID"])
            appearances[ident].add(i)
            player_records.setdefault(ident, p)
            if (str(p.get("Position", "")).upper() in {"RB", "WR"}
                    and float(p["Salary"]) < value_salary_ceiling):
                cheap.add(ident)
    rows, cols, data, lows, highs = [], [], [], [], []

    def add_constraint(terms, lower, upper):
        idx = len(lows)
        for col, coefficient in terms:
            rows.append(idx)
            cols.append(col)
            data.append(coefficient)
        lows.append(float(lower))
        highs.append(float(upper))

    add_constraint([(i, 1) for i in range(2 * n)], count, count)
    add_constraint([(n + i, 1) for i in range(n)], premium_count, premium_count)
    for i in range(n):
        add_constraint([(i, 1), (n + i, 1)], 0, 1)
    max_player = max(1, int(np.floor(count * max_exposure)))
    for ident in premium_excluded_ids:
        for i in appearances.get(str(ident), ()):
            add_constraint([(n + i, 1)], 0, 0)
    for ident, indices in appearances.items():
        role_cap = maximum_player_appearances(
            player_records[ident], count, max_exposure)
        cap = min(role_cap, value_max_appearances) if ident in cheap else role_cap
        add_constraint([(i, 1) for i in indices] +
                       [(n + i, 1) for i in indices], 0, cap)
        if ident in cheap:
            add_constraint([(n + i, 1) for i in indices],
                           0, premium_value_max_appearances)
    # No two final Classic lineups may share seven or more of nine players.
    # Repeated 7-player sets enforce overlap without enumerating all N^2 pairs.
    for group in overlap_groups(scored):
        add_constraint([(i, 1) for i in group] +
                       [(n + i, 1) for i in group], 0, 1)
    # Cross-lineup restriction: opposing WR/TE against a rostered DST can
    # occasionally be viable, but may not dominate the portfolio. This is a
    # whole-portfolio constraint, enforced even if greedy selection fails.
    conflicted = [i for i, item in enumerate(scored)
                  if opposing_pass_catcher_dst_pairs(item["candidate"]["players"])]
    if conflicted:
        add_constraint([(i, 1) for i in conflicted] +
                       [(n + i, 1) for i in conflicted],
                       0, opposing_dst_lineup_budget(count))
    # Premium entries carry more money. The global 35% player limit does
    # not protect a five-entry premium subportfolio by itself.
    premium_qb_limit = max(1, ceil(premium_count * 0.35))
    premium_pass_team_limit = max(1, ceil(premium_count * 0.60))
    quarterback_ids = defaultdict(set)
    passing_team_indices = defaultdict(set)
    for i, item in enumerate(scored):
        roster = item["candidate"]["players"]
        for p in roster:
            position = str(p.get("Position", "")).upper()
            if position == "QB":
                quarterback_ids[str(p["ID"])].add(i)
            if position in {"QB", "WR", "TE"}:
                passing_team_indices[str(p.get("TeamAbbrev", "")).upper()].add(i)
    for indices in quarterback_ids.values():
        add_constraint([(n + i, 1) for i in indices], 0, premium_qb_limit)
    for team, indices in passing_team_indices.items():
        if team:
            # A lineup with 2+ receivers from the same team counts once.
            add_constraint([(n + i, 1) for i in indices], 0, premium_pass_team_limit)
    means = np.array([v["simulated_mean"] for v in scored])
    p90 = np.array([v["simulated_p90"] for v in scored])
    p99 = np.array([v["simulated_p99"] for v in scored])

    def scaled(x):
        span = float(x.max() - x.min())
        return (x - x.min()) / max(span, 1e-6)

    # Balanced entries favor mean and P90; premium entries emphasize ceiling.
    stack_lifts = np.array([
        float(item.get("stack_lift",0.)) for item in scored], dtype=float)
    # This is an actual conditional upper-tail association in simulated
    # worlds, not a categorical bonus for simply having teammates.
    # Simulated co-ceiling frequency is NOT historically verified DFS ROI.
    opposing_dst = np.array([
        bool(opposing_pass_catcher_dst_pairs(item["candidate"]["players"]))
        for item in scored], dtype=float)
    regular = (.65 * scaled(means) + .25 * scaled(p90) +
               .10 * scaled(p99) + .02 * stack_lifts - .35 * opposing_dst)
    premium = (.15 * scaled(means) + .25 * scaled(p90) +
               .60 * scaled(p99) + .12 * stack_lifts - .45 * opposing_dst)
    matrix = coo_matrix((data, (rows, cols)),
                        shape=(len(lows), 2 * n)).tocsr()
    result = milp(c=-np.concatenate((regular, premium)),
                  integrality=np.ones(2 * n, dtype=int),
                  bounds=Bounds(np.zeros(2 * n), np.ones(2 * n)),
                  constraints=LinearConstraint(matrix, lows, highs),
                  options={"time_limit": 35.0, "mip_rel_gap": .03})
    if result.x is None:
        # SciPy status 1 is often a solver time limit, NOT proof that
        # the lineup rules are mathematically impossible.
        status = getattr(result,"status",None)
        message = getattr(result,"message","unknown solver termination")
        raise ValueError(
            f"Complete lineup portfolio not found: solver_status={status}; "
            f"solver_message={message}; retained_candidates={n}; "
            f"seven-player overlap groups={len(overlap_groups(scored))}; "
            "no partial portfolio exported")
    x = np.rint(result.x).astype(int)
    sums = matrix @ x
    if (not np.all(np.abs(result.x - x) < 1e-5) or
            not np.all(sums >= np.array(lows) - 1e-6) or
            not np.all(sums <= np.array(highs) + 1e-6)):
        raise ValueError("Integer optimization returned an invalid portfolio; no partial portfolio exported")
    chosen_ids = [i for i in range(n) if x[i] + x[n + i] == 1]
    if len(chosen_ids) != count:
        raise ValueError("Optimizer returned fewer than 20 lineups; no partial portfolio exported")
    upside_quota = round(count * upside_share)
    upside_ids = set(sorted(chosen_ids,
                            key=lambda i: scored[i]["simulated_p99"], reverse=True)[:upside_quota])
    chosen = []
    premium_ids = set()
    for i in chosen_ids:
        if x[n + i] == 1:
            premium_ids.add(len(chosen))
        chosen.append(dict(scored[i],
            portfolio_role="upside" if i in upside_ids else "balanced",
            contest_tier="$3" if x[n + i] else "$0.50",
            p90_per_1000=round(
                scored[i]["simulated_p90"] /
                (float(scored[i]["candidate"]["salary"]) / 1000), 4)))
    if len(premium_ids) != premium_count:
        raise ValueError("Premium portfolio allocation failed; no partial portfolio exported")
    return chosen, premium_ids

def select_simulated_portfolio(candidates, players, draws, count=20,
                               max_exposure=0.35, upside_share=0.25,
                               value_salary_ceiling=5000, value_max_appearances=2,
                               premium_count=5, premium_value_max_appearances=1,
                               min_wr_salary=4000, min_rb_salary=4500,
                               require_opportunity=True, p90_efficiency_retention=0.95,
                               premium_excluded_ids=frozenset(),
                               diagnostics=None, validation_draws=None):
    premium_excluded_ids = frozenset(str(x) for x in premium_excluded_ids)
    if not 1 <= count <= 150 or not 0 < max_exposure <= 1 or not 0 <= upside_share <= 1:
        raise ValueError("Invalid portfolio parameters")
    if not 0 < p90_efficiency_retention <= 1:
        raise ValueError("P90 efficiency retention must be in (0,1]")
    if not 0 <= premium_count <= count or not 1 <= premium_value_max_appearances <= value_max_appearances:
        raise ValueError("Invalid premium or value exposure limits")
    if not {"ID"}.issubset(players.columns):
        raise ValueError("Simulation player IDs are required")
    matrix = np.asarray(draws, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 100 or matrix.shape[1] != len(players) or not np.isfinite(matrix).all():
        raise ValueError("Invalid simulation draws")
    holdout=None
    if validation_draws is not None:
        holdout=np.asarray(validation_draws,dtype=float)
        if (holdout.ndim!=2 or holdout.shape[0]<100
                or holdout.shape[1]!=len(players)
                or not np.isfinite(holdout).all()):
            raise ValueError("Invalid disjoint NFL evaluation worlds")
        if np.shares_memory(matrix,holdout):
            raise ValueError("NFL evaluation worlds overlap selection worlds")
    ids = players["ID"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate simulation IDs")
    index = {key: i for i, key in enumerate(ids)}
    scored = []
    signatures = set()
    for candidate in candidates:
        roster = candidate["players"]
        errors = check_classic_lineup(roster, salary_floor=49600, salary_cap=50000)
        if errors:
            raise ValueError("Illegal candidate: " + "; ".join(errors))
        for p in roster:
            pos = str(p.get("Position", "")).upper()
            salary = float(p["Salary"])
            if (pos == "WR" and salary < min_wr_salary) or (pos == "RB" and salary < min_rb_salary):
                raise ValueError("Candidate violates RB/WR salary minimum")
            if require_opportunity and p.get("opportunity_verified") is not True:
                raise ValueError("Candidate lacks verified weekly opportunity")
            if require_opportunity and not player_opportunity_evidence(p)[0]:
                raise ValueError("Candidate lacks documented low-cost opportunity evidence")
        keys = [str(p["ID"]) for p in roster]
        if any(key not in index for key in keys):
            raise ValueError("Every roster member, including DST, needs simulated outcomes")
        signature = tuple(sorted(keys))
        if signature in signatures:
            continue
        signatures.add(signature)
        scores = matrix[:, [index[key] for key in keys]].sum(axis=1)
        scored.append(dict(candidate=candidate, simulated_mean=float(scores.mean()),
                           simulated_p90=float(np.percentile(scores, 90)),
                           simulated_p99=float(np.percentile(scores, 99)),
                           stack_lift=score_stack_lift(roster,index,matrix),
                           opposing_dst_conflicts=len(opposing_pass_catcher_dst_pairs(roster))))
    if not scored:
        raise ValueError("No simulated candidates available")
    all_scored=list(scored)  # Preserve pre-efficiency counterfactual opportunities.
    by_qb_before=Counter()
    best_qb_p99={}
    for item in scored:
        qb,_=qb_and_partners(item["candidate"]["players"])
        if qb is not None:
            ident=str(qb["ID"])
            by_qb_before[ident]+=1
            best_qb_p99[ident]=max(best_qb_p99.get(ident,float("-inf")),
                                   item["simulated_p99"])
    best_p90_efficiency = max(x["simulated_p90"] / (float(x["candidate"]["salary"]) / 1000) for x in scored)
    efficiency_floor = best_p90_efficiency * p90_efficiency_retention
    def efficient(item):
        return item["simulated_p90"] / (float(item["candidate"]["salary"]) / 1000)
    primary=[x for x in scored if efficient(x)>=efficiency_floor]
    # Ranking all roster combinations by a single global P90/$1K cutoff
    # can erase *every* stack for a legitimately viable QB. Recover the
    # strongest stacked alternatives per QB, but only if they have credible
    # fantasy ceiling AND are reasonably close to the global efficiency floor.
    # This is CANDIDATE PRESERVATION, not forcing QB exposure or a stack.
    retained_qbs={str(qb["ID"]) for item in primary
                  for qb,partners in [qb_and_partners(item["candidate"]["players"])]
                  if qb is not None and partners}
    rescued=[]
    pool_by_qb={}
    best_p99=max(item["simulated_p99"] for item in scored)
    for item in scored:
        qb,partners=qb_and_partners(item["candidate"]["players"])
        if not qb or not partners or str(qb["ID"]) in retained_qbs:
            continue
        # Both tests matter: cheap but no-upside combinations are not rescued.
        # The QB audit lists cases rejected by these thresholds.
        if (efficient(item)>=efficiency_floor*.85
                and item["simulated_p99"]>=best_p99*.88):
            pool_by_qb.setdefault(str(qb["ID"]),[]).append(item)
    for qb_id,items in pool_by_qb.items():
        rescued.extend(sorted(items,key=lambda item:(
            item["simulated_p99"],item["simulated_p90"]),reverse=True)[:3])
    # Global P90 ranking must not destroy an exposure-feasible portfolio.
    # Every retained option is already a legal and pregame-verified lineup;
    # preserve alternatives at RB before the 28% exposure cap is applied.
    current=primary+rescued
    current_keys={tuple(sorted(str(p["ID"]) for p in item["candidate"]["players"]))
                  for item in current}
    rb_id_to_items=defaultdict(list)
    for item in all_scored:
        for p in item["candidate"]["players"]:
            if str(p["Position"]).upper()=="RB":
                rb_id_to_items[str(p["ID"])].append(item)
    initial_rb_counts=Counter(
        str(p["ID"]) for item in current for p in item["candidate"]["players"]
        if str(p.get("Position","")).upper()=="RB")
    rb_rescued=[]
    for rb_id,items in sorted(rb_id_to_items.items(),key=lambda kv:(
        initial_rb_counts[kv[0]],kv[0])):
        need=max(0,10-initial_rb_counts[rb_id])
        if need<=0:
            continue
        already_for_rb=[
            {str(p["ID"]) for p in item["candidate"]["players"]}
            for item in current if any(str(p["ID"])==rb_id
                                      for p in item["candidate"]["players"])]
        pool=sorted(items,key=lambda item:(
            item["simulated_p90"],item["simulated_p99"],
            item["simulated_mean"]),reverse=True)
        for item in pool:
            if need<=0:
                break
            if (efficient(item)<efficiency_floor*.75
                    or item["simulated_p99"]<best_p99*.70):
                continue
            keys={str(p["ID"]) for p in item["candidate"]["players"]}
            signature=tuple(sorted(keys))
            if signature in current_keys:
                continue
            if any(len(keys & previous)>MAX_SHARED_PLAYERS
                   for previous in already_for_rb):
                continue
            current.append(item)
            rb_rescued.append(item)
            current_keys.add(signature)
            already_for_rb.append(keys)
            need-=1
    scored=current
    after=Counter()
    for item in scored:
        qb,_=qb_and_partners(item["candidate"]["players"])
        if qb is not None:
            after[str(qb["ID"])]+=1
    after_stacks=Counter()
    for item in scored:
        qb,partners=qb_and_partners(item["candidate"]["players"])
        if qb is not None and partners:
            after_stacks[str(qb["ID"])]+=1
    if diagnostics is not None:
        diagnostics.update({
            "stacked_qbs_after_efficiency":dict(after_stacks),
            "scored_qbs_before_efficiency":dict(by_qb_before),
            "scored_qbs_after_efficiency":dict(after),
            "qb_best_simulated_p99":best_qb_p99,
            "global_p90_efficiency_floor":round(float(efficiency_floor),4),
            "rescued_stacked_candidates":len(rescued),
            "rescued_verified_alternate_rb_candidates":len(rb_rescued),
            "distinct_running_backs_before_diversity_rescue":len(initial_rb_counts),
            "distinct_running_backs_after_diversity_rescue":len({
                str(p["ID"]) for item in scored for p in item["candidate"]["players"]
                if str(p.get("Position","")).upper()=="RB"}),
            "simulated_candidate_count":sum(by_qb_before.values()),
            "retained_candidate_count":len(scored),
        })
    limit = max(1, int(np.floor(count * max_exposure)))
    opposing_dst_budget = opposing_dst_lineup_budget(count)
    chosen, exposures, used = [], Counter(), set()
    chosen_rosters = []
    selected_opposing_dst = 0
    for role, quota in (("upside", round(count * upside_share)),
                        ("balanced", count-round(count * upside_share))):
        ordering = sorted(range(len(scored)),
                          key=lambda i: (
                              # Premium/upside gets a small, evidence-only
                              # QB/WR/TE stack preference, not a hard mandate.
                              (scored[i]["simulated_p99"] *
                               (1.0 + .06 * scored[i]["stack_lift"]) -
                               18.0 * bool(scored[i]["opposing_dst_conflicts"]))
                              if role == "upside" else
                              (scored[i]["simulated_mean"] +
                               .35 * scored[i]["stack_lift"] -
                               14.0 * bool(scored[i]["opposing_dst_conflicts"])),
                              scored[i]["simulated_p90"]), reverse=True)
        selected_role = 0
        for i in ordering:
            if selected_role >= quota:
                break
            if i in used:
                continue
            if scored[i]["opposing_dst_conflicts"] and selected_opposing_dst >= opposing_dst_budget:
                continue
            keys = [str(p["ID"]) for p in scored[i]["candidate"]["players"]]
            if overlap_conflicts(scored[i]["candidate"]["players"], chosen_rosters):
                continue
            is_premium = False
            def limit_reached(p):
                key = str(p["ID"])
                is_value = (str(p.get("Position", "")).upper() in {"RB", "WR"}
                            and float(p["Salary"]) < value_salary_ceiling)
                role_limit = maximum_player_appearances(p, count, max_exposure)
                if exposures[key] >= (min(role_limit, value_max_appearances)
                                     if is_value else role_limit):
                    return True
                return False
            if any(limit_reached(p) for p in scored[i]["candidate"]["players"]):
                continue
            chosen.append(dict(scored[i], portfolio_role=role,
                               contest_tier="$0.50",
                               p90_per_1000=round(scored[i]["simulated_p90"] /
                                   (float(scored[i]["candidate"]["salary"]) / 1000), 4)))
            exposures.update(keys)
            chosen_rosters.append(keys)
            used.add(i)
            selected_opposing_dst += bool(scored[i]["opposing_dst_conflicts"])
            selected_role += 1
    used_joint_solver = len(chosen) != count
    premium_ids = set()
    if used_joint_solver:
        chosen, premium_ids = _solve_joint_portfolio(
            scored, count=count, premium_count=premium_count,
            max_exposure=max_exposure, upside_share=upside_share,
            value_salary_ceiling=value_salary_ceiling,
            value_max_appearances=value_max_appearances,
            premium_value_max_appearances=premium_value_max_appearances,
            premium_excluded_ids=premium_excluded_ids)
    # Allocate premium entries AFTER selecting the complete exposure-constrained
    # portfolio: a candidate's position in the upside/balanced search is NOT
    # evidence that it deserves a more expensive contest entry.
    premium_value_counts = Counter()
    order = ([] if used_joint_solver else sorted(range(len(chosen)), key=lambda i: (
        chosen[i]["simulated_p99"], chosen[i]["simulated_p90"],
        chosen[i]["simulated_mean"]), reverse=True))
    for i in order:
        if len(premium_ids) == premium_count:
            break
        if any(str(p["ID"]) in premium_excluded_ids for p in chosen[i]["candidate"]["players"]):
            continue
        affordable = [p for p in chosen[i]["candidate"]["players"]
                      if str(p.get("Position", "")).upper() in {"RB", "WR"}
                      and float(p["Salary"]) < value_salary_ceiling]
        if any(premium_value_counts[str(p["ID"])] >= premium_value_max_appearances for p in affordable):
            continue
        premium_ids.add(i)
        premium_value_counts.update(str(p["ID"]) for p in affordable)
    if len(premium_ids) != premium_count:
        chosen, premium_ids = _solve_joint_portfolio(
            scored, count=count, premium_count=premium_count,
            max_exposure=max_exposure, upside_share=upside_share,
            value_salary_ceiling=value_salary_ceiling,
            value_max_appearances=value_max_appearances,
            premium_value_max_appearances=premium_value_max_appearances,
            premium_excluded_ids=premium_excluded_ids)

    # Even a complete greedy premium assignment can overconcentrate one QB
    # or one team's passing offense. Re-optimize jointly in that case.
    premium_qb_exposures = Counter()
    premium_pass_team_exposures = Counter()
    for i in premium_ids:
        roster = chosen[i]["candidate"]["players"]
        premium_qb_exposures.update(str(p["ID"]) for p in roster
                                    if str(p.get("Position", "")).upper() == "QB")
        premium_pass_team_exposures.update({
            str(p.get("TeamAbbrev", "")).upper() for p in roster
            if str(p.get("Position", "")).upper() in {"QB", "WR", "TE"}
            and str(p.get("TeamAbbrev", "")).strip()})
    if (premium_qb_exposures and max(premium_qb_exposures.values()) > max(1, ceil(premium_count * 0.35))
            or premium_pass_team_exposures and
            max(premium_pass_team_exposures.values()) > max(1, ceil(premium_count * 0.60))):
        chosen, premium_ids = _solve_joint_portfolio(
            scored, count=count, premium_count=premium_count,
            max_exposure=max_exposure, upside_share=upside_share,
            value_salary_ceiling=value_salary_ceiling,
            value_max_appearances=value_max_appearances,
            premium_value_max_appearances=premium_value_max_appearances,
            premium_excluded_ids=premium_excluded_ids)
    # Final portfolio checks are independent of the greedy/MILP decision path.
    # The whole-slate cap is floor(count * max_exposure), not a rounded-up
    # approximation. A single QB is also a player subject to this rule.
    final_exposure=Counter(str(p["ID"]) for item in chosen
                           for p in item["candidate"]["players"])
    final_records = {str(p["ID"]): p for item in chosen
                     for p in item["candidate"]["players"]}
    if len(chosen)!=count or any(
        value > maximum_player_appearances(final_records[ident],count,max_exposure)
        for ident,value in final_exposure.items()
    ):
        raise ValueError("Incomplete NFL portfolio or exposure/role-risk cap violated; no export")
    final_rosters = [[str(p["ID"]) for p in item["candidate"]["players"]]
                     for item in chosen]
    if any(len(set(final_rosters[i]) & set(final_rosters[j])) > MAX_SHARED_PLAYERS
           for i in range(len(final_rosters))
           for j in range(i+1,len(final_rosters))):
        raise ValueError("NFL portfolio has near-duplicate lineups; no export")
    final_opposing_dst = sum(bool(opposing_pass_catcher_dst_pairs(
        item["candidate"]["players"])) for item in chosen)
    if final_opposing_dst > opposing_dst_budget:
        raise ValueError("Opposing WR/TE versus DST portfolio safety cap violated; no export")
    final_qbs=Counter()
    final_stacks=Counter()
    for item in chosen:
        qb,partners=qb_and_partners(item["candidate"]["players"])
        if qb is not None:
            final_qbs[str(qb["ID"])]+=1
            if partners:
                final_stacks[str(qb["ID"])]+=1
    if diagnostics is not None:
        diagnostics.update({
            "final_qb_counts":dict(final_qbs),
            "final_qb_stacked_counts":dict(final_stacks),
            "final_stack_count":sum(final_stacks.values()),
            "final_qb_max_allowed":limit,
            "selection_method":"MILP joint" if used_joint_solver else "simulated greedy",
            "opposing_dst_conflicted_lineups":final_opposing_dst,
            "opposing_dst_max_lineups":opposing_dst_budget,
            "opposing_dst_candidates_before_filter":sum(
                bool(x["opposing_dst_conflicts"]) for x in all_scored),
            "max_shared_players":MAX_SHARED_PLAYERS,
            "selected_lineup_stack_lift": {
                str(i+1): float(x.get("stack_lift",0.))
                for i,x in enumerate(chosen)
            },
            "overall_max_exposure":max_exposure,
            "uncertain_running_back_exposures":{
                str(p.get("Name", p["ID"])): {
                    "lineups": final_exposure[str(p["ID"])],
                    "cap": maximum_player_appearances(p,count,max_exposure),
                    "reason": uncertain_running_back_reason(p),
                } for p in final_records.values()
                if uncertain_running_back_reason(p)
                and final_exposure[str(p["ID"])] > 0
            },
        })
    if holdout is not None and diagnostics is not None:
        counterfactual,validation=quarterback_counterfactuals(
            all_scored,chosen,ids,holdout)
        diagnostics["qb_counterfactuals"]=counterfactual
        diagnostics["independent_scenario_audit"]=validation
    if any(str(p["ID"]) in premium_excluded_ids
           for i in premium_ids for p in chosen[i]["candidate"]["players"]):
        raise ValueError("Premium player exclusion violated; refusing to export portfolio")
    for i, item in enumerate(chosen):
        item["contest_tier"] = "$3" if i in premium_ids else "$0.50"
        roster = item["candidate"]["players"]
        qb, direct_receivers=qb_and_partners(roster)
        teammates=[p.get("Name",str(p["ID"])) for p in direct_receivers]
        evidence = {}
        for p in roster:
            good, why = player_opportunity_evidence(p, value_ceiling=value_salary_ceiling)
            if not good:
                raise ValueError("Unjustified value player reached final portfolio")
            player_draws = matrix[:, index[str(p["ID"])]]
            player_mean = float(player_draws.mean())
            player_p90 = float(np.percentile(player_draws, 90))
            salary_eff = player_p90 / (float(p["Salary"]) / 1000)
            source = str(p.get("projection_source") or "unspecified").strip()
            basis = why if p["Position"] in {"WR", "RB"} and float(p["Salary"]) < value_salary_ceiling else "Weekly projection marked verified; low-cost role gate not applicable"
            evidence[str(p["ID"])] = (
                f"{basis}; simulated mean {player_mean:.1f}, P90 {player_p90:.1f}, "
                f"P90/$1K {salary_eff:.2f}; projection source: {source}. "
                "These statistics explain modeled opportunity, not a guaranteed outcome."
            )
        item["selection_explanation"] = {
            "contest_assignment": "Simulated P99/P90 and mean with modest receiving-stack preference; restricted low-salary exposure" if i in premium_ids else "Selected by portfolio simulation and exposure constraints",
            "qb_receiving_teammates": teammates,
            "qb_stack_note": ("No direct QB/WR-TE receiving stack; standalone QB ceiling was allowed"
                              if not teammates else
                              "Same-team QB/WR-TE stack considered; modeled ceiling evaluated on shared simulated worlds"),
            "player_opportunity_evidence": evidence,
            "simulated_p90_per_1000": item["p90_per_1000"],
            "simulated_qb_receiver_upper_tail_lift": item.get("stack_lift",0.),
            "opposing_pass_catchers_vs_dst": opposing_pass_catcher_dst_pairs(roster),
        }
    return sorted(chosen, key=lambda x: (x["contest_tier"] != "$3", -x["simulated_p99"]))
