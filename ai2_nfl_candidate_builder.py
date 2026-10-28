"""Transparent NFL Classic projection-based candidate builder (not a simulator).

Produces legal candidate lineups only from independently verified weekly
projection rows. No sportsbook inputs and no profitability guarantees.
"""
from collections import Counter
from ai2_nfl_role_evidence import player_opportunity_evidence
from random import Random
from ai2_nfl_lineup import check_classic_lineup

def build_classic_candidates(rows, *, count=5, attempts=30000, seed=7,
                             salary_floor=49600, salary_cap=50000,
                             max_exposure=0.60, min_wr_salary=4000, min_rb_salary=4500,
                             require_opportunity=True):
    if not 1 <= count <= 2000 or not 1 <= attempts <= 200000:
        raise ValueError("Invalid candidate count or search budget")
    if not 0 < max_exposure <= 1:
        raise ValueError("Invalid exposure limit")
    if min_wr_salary < 0 or min_rb_salary < 0:
        raise ValueError("Salary minimums must be nonnegative")
    pools = {p: [] for p in ("QB", "RB", "WR", "TE", "DST")}
    for row in rows:
        position = str(row.get("Position", "")).upper()
        if position not in pools:
            continue
        try:
            salary = float(row["Salary"])
        except (KeyError, TypeError, ValueError):
            continue
        if (position == "WR" and salary < min_wr_salary) or (position == "RB" and salary < min_rb_salary):
            continue
        if require_opportunity and row.get("opportunity_verified") is not True:
            continue
        if require_opportunity and not player_opportunity_evidence(row)[0]:
            continue
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
    # Preserve different quarterback and game-stack paths for the *simulator*.
    # Merely taking the top projected means repeatedly crowds out viable QBs,
    # including QBs in high-total games. These are candidates, NOT final picks.
    ordered = sorted(best.values(), key=lambda x: x[0], reverse=True)
    by_qb = {}
    for item in ordered:
        qb = next(p for p in item[1] if p["Position"] == "QB")
        qb_id = str(qb["ID"])
        has_pass_stack = any(
            p["Position"] in {"WR", "TE"} and p["TeamAbbrev"] == qb["TeamAbbrev"]
            for p in item[1] if p["ID"] != qb["ID"])
        by_qb.setdefault(qb_id, []).append((item, has_pass_stack))

    selected, selected_keys = [], set()
    exposures = Counter()
    limit = max(1, int(count * max_exposure + 0.999999))

    def add(item):
        score, lineup = item
        key = tuple(sorted(str(p["ID"]) for p in lineup))
        if key in selected_keys or len(selected) >= count:
            return False
        if any(exposures[str(p["ID"])] >= limit for p in lineup):
            return False
        selected.append({"players": lineup, "projected_points": round(score, 2),
                         "salary": int(sum(float(p["Salary"]) for p in lineup))})
        selected_keys.add(key)
        exposures.update(str(p["ID"]) for p in lineup)
        return True

    # Reserve up to half the search pool for alternative QBs. Prefer a QB +
    # receiving teammate where possible but retain nonstacked alternatives.
    # Fewer viable QBs does not create invented or illegal candidates.
    qb_ids = sorted(by_qb, key=lambda k: by_qb[k][0][0][0], reverse=True)
    target_diverse = min(count // 2, len(ordered))
    if qb_ids and target_diverse:
        quota = max(1, target_diverse // len(qb_ids))
        for qb_id in qb_ids:
            taken = 0
            for item, stacked in sorted(by_qb[qb_id], key=lambda x: (x[1], x[0][0]), reverse=True):
                if taken >= quota or len(selected) >= target_diverse:
                    break
                if add(item):
                    taken += 1
    # Remaining candidates compete on projected points, including previously
    # crowded-out stacked and unstacked combinations. Simulated portfolio
    # selection determines winners and contest assignment downstream.
    for item in ordered:
        if len(selected) >= count:
            break
        add(item)
    return selected
