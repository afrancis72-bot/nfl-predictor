"""Score complete legal NFL Classic lineups on shared game simulations.

Scenario percentiles are not contest ROI or predicted cash rates.
"""
from collections import Counter
import numpy as np
import pandas as pd
from ai2_nfl_lineup import check_classic_lineup


def select_simulated_portfolio(candidates, players, draws, count=20,
                               max_exposure=0.35, upside_share=0.25,
                               value_salary_ceiling=5000, value_max_appearances=2,
                               premium_count=5, premium_value_max_appearances=1,
                               min_wr_salary=4000, min_rb_salary=4500,
                               require_opportunity=True):
    if not 1 <= count <= 20 or not 0 < max_exposure <= 1 or not 0 <= upside_share <= 1:
        raise ValueError("Invalid portfolio parameters")
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
    limit = max(1, int(np.floor(count * max_exposure)))
    chosen, exposures, premium_exposures, used = [], Counter(), Counter(), set()
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
            is_premium = len(chosen) < premium_count
            def limit_reached(p):
                key = str(p["ID"])
                is_value = (str(p.get("Position", "")).upper() in {"RB", "WR"}
                            and float(p["Salary"]) < value_salary_ceiling)
                if exposures[key] >= (min(limit, value_max_appearances) if is_value else limit):
                    return True
                return is_premium and is_value and premium_exposures[key] >= premium_value_max_appearances
            if any(limit_reached(p) for p in scored[i]["candidate"]["players"]):
                continue
            chosen.append(dict(scored[i], portfolio_role=role,
                               contest_tier="$3" if is_premium else "$0.50",
                               p90_per_1000=round(scored[i]["simulated_p90"] /
                                   (float(scored[i]["candidate"]["salary"]) / 1000), 4)))
            exposures.update(keys)
            if is_premium:
                premium_exposures.update(keys)
            used.add(i)
            selected_role += 1
    if len(chosen) != count:
        raise ValueError(f"Only {len(chosen)} of {count} lineups satisfy the exposure and roster constraints; no partial portfolio exported")
    return chosen
