"""Bounded NFL Classic roster-feasibility search, not lineup optimization.

Only an exhaustively searched pool may return feasible=False.
A search budget exhaustion returns indeterminate (None).
"""
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from ai2_nfl_lineup import check_classic_lineup

@dataclass(frozen=True)
class Feasibility:
    feasible: bool | None
    lineup: tuple
    examined: int
    reason: str

def find_classic_lineup(rows, *, salary_floor=47500, salary_cap=50000, max_checks=200000):
    if max_checks < 1:
        raise ValueError("max_checks must be positive")
    pools = {p: [] for p in ("QB", "RB", "WR", "TE", "DST")}
    seen = set()
    for row in rows:
        position = str(row.get("Position", "")).upper().strip()
        identifier = str(row.get("ID", "")).strip()
        if position not in pools or not identifier or identifier in seen:
            continue
        try:
            salary = float(row["Salary"])
        except (KeyError, TypeError, ValueError):
            continue
        if not isfinite(salary) or salary <= 0 or salary > salary_cap:
            continue
        if not str(row.get("TeamAbbrev", "")).strip():
            continue
        seen.add(identifier)
        pools[position].append(row)
    if any(len(pools[p]) < n for p, n in (("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1), ("DST", 1))):
        return Feasibility(False, (), 0, "Insufficient positional candidates")
    examined = 0
    for qb in pools["QB"]:
        for dst in pools["DST"]:
            if qb.get("opponent") and str(qb["opponent"]).upper() == str(dst["TeamAbbrev"]).upper():
                continue
            for rbs in combinations(pools["RB"], 2):
                if str(rbs[0]["TeamAbbrev"]).upper() == str(rbs[1]["TeamAbbrev"]).upper():
                    continue
                for wrs in combinations(pools["WR"], 3):
                    for te in pools["TE"]:
                        core = (qb, dst, *rbs, *wrs, te)
                        core_ids = {str(p["ID"]) for p in core}
                        for flex in (*pools["RB"], *pools["WR"], *pools["TE"]):
                            if str(flex["ID"]) in core_ids:
                                continue
                            lineup = (*core, flex)
                            examined += 1
                            if not check_classic_lineup(lineup, salary_cap=salary_cap, salary_floor=salary_floor):
                                return Feasibility(True, lineup, examined, "Legal lineup found")
                            if examined >= max_checks:
                                return Feasibility(None, (), examined, "Search budget exhausted; no conclusion")
    return Feasibility(False, (), examined, "Exhaustive search found no legal lineup")
