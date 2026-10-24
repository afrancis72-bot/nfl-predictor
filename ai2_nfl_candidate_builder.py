"""Transparent NFL Classic projection-based candidate builder (not a simulator).

Produces legal candidate lineups only from independently verified weekly
projection rows. No sportsbook inputs and no profitability guarantees.
"""
from collections import Counter
from random import Random
from ai2_nfl_lineup import check_classic_lineup

def build_classic_candidates(rows, *, count=5, attempts=30000, seed=7,
                             salary_floor=49600, salary_cap=50000,
                             max_exposure=0.60):
    if not 1 <= count <= 2000 or not 1 <= attempts <= 200000:
        raise ValueError("Invalid candidate count or search budget")
    if not 0 < max_exposure <= 1:
        raise ValueError("Invalid exposure limit")
    pools = {p: [] for p in ("QB", "RB", "WR", "TE", "DST")}
    for row in rows:
        position = str(row.get("Position", "")).upper()
        if position in pools:
            pools[position].append(row)
    if any(len(pools[p]) < n for p, n in (("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1), ("DST", 1))):
        return []
    rng = Random(seed)
    best = {}
    for _ in range(attempts):
        qb = rng.sample(pools["QB"], 1)
        rb = rng.sample(pools["RB"], 2)
        wr = rng.sample(pools["WR"], 3)
        te = rng.sample(pools["TE"], 1)
        dst = rng.sample(pools["DST"], 1)
        used = {str(p["ID"]) for p in qb + rb + wr + te + dst}
        flex_pool = [p for position in ("RB", "WR", "TE") for p in pools[position]
                     if str(p["ID"]) not in used]
        if not flex_pool:
            continue
        lineup = qb + rb + wr + te + dst + rng.sample(flex_pool, 1)
        if check_classic_lineup(lineup, salary_floor=salary_floor, salary_cap=salary_cap):
            continue
        key = tuple(sorted(str(p["ID"]) for p in lineup))
        score = sum(float(p["proj"]) for p in lineup)
        if key not in best or score > best[key][0]:
            best[key] = (score, lineup)
    # Improve the best legal random seeds with targeted same-position upgrades.
    # This discovers balanced salary trades that pure random sampling misses.
    seeds = sorted(best.values(), key=lambda x: x[0], reverse=True)[:min(count, 400)]
    for _, seed_lineup in seeds:
        roster = list(seed_lineup)
        used = {str(p["ID"]) for p in roster}
        for i, old in enumerate(roster):
            alternatives = sorted(pools[old["Position"]],
                                  key=lambda p: float(p["proj"]), reverse=True)[:20]
            for replacement in alternatives:
                if str(replacement["ID"]) in used:
                    continue
                revised = roster.copy()
                revised[i] = replacement
                if check_classic_lineup(revised, salary_floor=salary_floor, salary_cap=salary_cap):
                    continue
                key = tuple(sorted(str(p["ID"]) for p in revised))
                score = sum(float(p["proj"]) for p in revised)
                if key not in best or score > best[key][0]:
                    best[key] = (score, revised)
    selected = []
    exposures = Counter()
    limit = max(1, int(count * max_exposure + 0.999999))
    for score, lineup in sorted(best.values(), key=lambda x: x[0], reverse=True):
        if len(selected) >= count:
            break
        if any(exposures[str(p["ID"])] >= limit for p in lineup):
            continue
        selected.append({"players": lineup, "projected_points": round(score, 2),
                         "salary": int(sum(float(p["Salary"]) for p in lineup))})
        exposures.update(str(p["ID"]) for p in lineup)
    return selected
