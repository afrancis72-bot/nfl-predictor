"""Score complete legal NFL Classic lineups on shared game simulations.

Scenario percentiles are not contest ROI or predicted cash rates.
"""
from collections import Counter
import numpy as np
import pandas as pd
from ai2_nfl_lineup import check_classic_lineup
from ai2_nfl_role_evidence import player_opportunity_evidence


def select_simulated_portfolio(candidates, players, draws, count=20,
                               max_exposure=0.35, upside_share=0.25,
                               value_salary_ceiling=5000, value_max_appearances=2,
                               premium_count=5, premium_value_max_appearances=1,
                               min_wr_salary=4000, min_rb_salary=4500,
                               require_opportunity=True, p90_efficiency_retention=0.95):
    if not 1 <= count <= 20 or not 0 < max_exposure <= 1 or not 0 <= upside_share <= 1:
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
                           simulated_p99=float(np.percentile(scores, 99))))
    if not scored:
        raise ValueError("No simulated candidates available")
    best_p90_efficiency = max(x["simulated_p90"] / (float(x["candidate"]["salary"]) / 1000) for x in scored)
    efficiency_floor = best_p90_efficiency * p90_efficiency_retention
    scored = [x for x in scored if x["simulated_p90"] / (float(x["candidate"]["salary"]) / 1000) >= efficiency_floor]
    limit = max(1, int(np.floor(count * max_exposure)))
    chosen, exposures, used = [], Counter(), set()
    for role, quota in (("upside", round(count * upside_share)),
                        ("balanced", count-round(count * upside_share))):
        ordering = sorted(range(len(scored)),
                          key=lambda i: (scored[i]["simulated_p99"] if role == "upside"
                                         else scored[i]["simulated_mean"],
                                         scored[i]["simulated_p90"]), reverse=True)
        selected_role = 0
        for i in ordering:
            if selected_role >= quota:
                break
            if i in used:
                continue
            keys = [str(p["ID"]) for p in scored[i]["candidate"]["players"]]
            is_premium = False
            def limit_reached(p):
                key = str(p["ID"])
                is_value = (str(p.get("Position", "")).upper() in {"RB", "WR"}
                            and float(p["Salary"]) < value_salary_ceiling)
                if exposures[key] >= (min(limit, value_max_appearances) if is_value else limit):
                    return True
                return False
            if any(limit_reached(p) for p in scored[i]["candidate"]["players"]):
                continue
            chosen.append(dict(scored[i], portfolio_role=role,
                               contest_tier="$0.50",
                               p90_per_1000=round(scored[i]["simulated_p90"] /
                                   (float(scored[i]["candidate"]["salary"]) / 1000), 4)))
            exposures.update(keys)
            used.add(i)
            selected_role += 1
    if len(chosen) != count:
        raise ValueError(f"Only {len(chosen)} of {count} lineups meet role, exposure and P90/$1,000 retention ({p90_efficiency_retention:.0%}) constraints; no partial portfolio exported")
    # Allocate premium entries AFTER selecting the complete exposure-constrained
    # portfolio: a candidate's position in the upside/balanced search is NOT
    # evidence that it deserves a more expensive contest entry.
    premium_ids = set()
    premium_value_counts = Counter()
    order = sorted(range(len(chosen)), key=lambda i: (
        chosen[i]["simulated_p99"], chosen[i]["simulated_p90"],
        chosen[i]["simulated_mean"]), reverse=True)
    for i in order:
        if len(premium_ids) == premium_count:
            break
        affordable = [p for p in chosen[i]["candidate"]["players"]
                      if str(p.get("Position", "")).upper() in {"RB", "WR"}
                      and float(p["Salary"]) < value_salary_ceiling]
        if any(premium_value_counts[str(p["ID"])] >= premium_value_max_appearances for p in affordable):
            continue
        premium_ids.add(i)
        premium_value_counts.update(str(p["ID"]) for p in affordable)
    if len(premium_ids) != premium_count:
        raise ValueError("Could not assign all premium entries under low-cost exposure limits")

    for i, item in enumerate(chosen):
        item["contest_tier"] = "$3" if i in premium_ids else "$0.50"
        roster = item["candidate"]["players"]
        qb = next(p for p in roster if str(p.get("Position", "")).upper() == "QB")
        teammates = [p.get("Name", str(p["ID"])) for p in roster
                     if p is not qb and p.get("TeamAbbrev") == qb.get("TeamAbbrev")
                     and p.get("Position") in {"WR", "TE", "RB"}]
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
            "contest_assignment": "Ranked by simulated P99, then P90 and mean; restricted low-salary exposure" if i in premium_ids else "Selected by portfolio simulation and exposure constraints",
            "qb_receiving_teammates": teammates,
            "qb_stack_note": "No teammate receiving stack" if not teammates else "Same-team QB receiving correlation; strength not independently certified",
            "player_opportunity_evidence": evidence,
            "simulated_p90_per_1000": item["p90_per_1000"]
        }
    return sorted(chosen, key=lambda x: (x["contest_tier"] != "$3", -x["simulated_p99"]))
