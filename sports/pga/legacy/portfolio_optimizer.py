from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Callable
import numpy as np
import pandas as pd


@dataclass
class PortfolioSettings:
    lineup_count: int = 10
    salary_cap: int = 50000
    salary_floor: int = 46500
    min_player_salary: int = 6500
    roster_size: int = 6
    max_exposure: float = 0.50
    min_unique: int = 3
    strategy: str = "GPP Ceiling"
    candidate_pool_size: int = 60
    candidate_samples: int = 45000
    target_candidates: int = 3500
    batch_size: int = 2500
    seed: int = 42


def _z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (x - x.mean()) / sd


def add_objectives(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    dk_col = "dk_points_proxy" if "dk_points_proxy" in d.columns else "dk_proxy"
    if dk_col not in d.columns:
        raise ValueError("Predictions must contain dk_points_proxy or dk_proxy.")
    for c in ["make_cut_pct", "top10_pct", "win_pct", dk_col, "salary"]:
        if c not in d.columns:
            raise ValueError(f"Missing required optimizer column: {c}")
    d["z_cut"] = _z(d["make_cut_pct"])
    d["z_top10"] = _z(d["top10_pct"])
    d["z_win"] = _z(d["win_pct"])
    d["z_dk"] = _z(d[dk_col])
    if "dk_value_per_1000" in d.columns:
        d["z_value"] = _z(d["dk_value_per_1000"])
    else:
        d["z_value"] = _z(d[dk_col] / (pd.to_numeric(d["salary"], errors="coerce") / 1000))
    if "model_data_confidence" in d.columns:
        d["z_conf"] = _z(d["model_data_confidence"])
    else:
        d["z_conf"] = 0.0
    d["obj_cut"] = .55*d.z_cut + .20*d.z_top10 + .15*d.z_dk + .10*d.z_conf
    d["obj_balanced"] = .25*d.z_cut + .30*d.z_top10 + .15*d.z_win + .20*d.z_dk + .10*d.z_value
    d["obj_gpp"] = .15*d.z_cut + .30*d.z_top10 + .30*d.z_win + .15*d.z_dk + .10*d.z_value
    # V11: optimize genuine simulated fantasy outcomes, not V10 rank proxies.
    # Keep legacy behavior only for older prediction files without V11 columns.
    if all(c in d.columns for c in ("v11_mean_dk","v11_p75_dk","v11_p90_dk")):
        d["obj_cut"] = _z(.85*d.v11_mean_dk + .15*d.v11_p75_dk)
        d["obj_balanced"] = _z(.80*d.v11_mean_dk + .20*d.v11_p75_dk)
        d["obj_gpp"] = _z(.45*d.v11_mean_dk + .55*d.v11_p90_dk)
    return d


def _objective_name(strategy: str) -> str:
    s = strategy.lower()
    if "cut" in s:
        return "obj_cut"
    if "balanced" in s or "single" in s:
        return "obj_balanced"
    return "obj_gpp"


def optimize_portfolio(
    predictions: pd.DataFrame,
    settings: PortfolioSettings | None = None,
    locks: Sequence[str] | None = None,
    excludes: Sequence[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = settings or PortfolioSettings()
    locks, excludes = set(locks or []), set(excludes or [])
    if locks & excludes:
        raise ValueError("A player cannot be both locked and excluded.")
    if len(locks) > settings.roster_size:
        raise ValueError("Too many locked golfers for the roster size.")

    d = add_objectives(predictions)
    d = d.dropna(subset=["player", "salary"]).copy()
    d["salary"] = pd.to_numeric(d["salary"], errors="coerce").astype(int)
    d = d[~d.player.isin(excludes)]
    d = d[d["salary"] >= settings.min_player_salary].copy()
    if not locks.issubset(set(d.player)):
        missing = sorted(locks - set(d.player))
        raise ValueError(f"Locked golfer(s) unavailable: {', '.join(missing)}")

    obj = _objective_name(settings.strategy)
    # Always retain locked players, then add best remaining candidates.
    locked_rows = d[d.player.isin(locks)]
    remaining = d[~d.player.isin(locks)].sort_values(obj, ascending=False)
    pool = pd.concat([locked_rows, remaining.head(settings.candidate_pool_size)]).drop_duplicates("player").reset_index(drop=True)

    rng = np.random.default_rng(settings.seed)
    values = pool[obj].to_numpy(float)
    salaries = pool["salary"].to_numpy(int)
    players_arr = pool["player"].astype(str).to_numpy()
    locked_idx = np.array(pool.index[pool.player.isin(locks)], dtype=int)
    locked_set = set(locked_idx.tolist())
    need = settings.roster_size - len(locked_idx)
    selectable = np.array([i for i in range(len(pool)) if i not in locked_set], dtype=int)
    candidates: dict[frozenset[str], tuple[float, int]] = {}

    if need < 0 or need > len(selectable):
        raise RuntimeError("Not enough eligible golfers to complete a lineup.")

    # Fast batched weighted-without-replacement generation.  Gumbel top-k gives
    # the same useful bias toward high-objective golfers as repeated weighted
    # choice, but evaluates thousands of candidate lineups per NumPy batch.
    temps = (1.10, 1.80, 3.00)
    total_budget = max(9000, int(settings.candidate_samples))
    per_temp_budget = max(3000, total_budget // len(temps))
    batch_size = max(250, int(settings.batch_size))
    target = max(settings.lineup_count * 180, int(settings.target_candidates))
    processed = 0

    if progress_callback:
        progress_callback(0.03, "Preparing DFS candidate pool…")

    locked_salary = int(salaries[locked_idx].sum()) if len(locked_idx) else 0
    locked_score = float(values[locked_idx].sum()) if len(locked_idx) else 0.0

    if need == 0:
        salary = locked_salary
        if settings.salary_floor <= salary <= settings.salary_cap:
            names = frozenset(players_arr[locked_idx])
            candidates[names] = (locked_score + 0.000010*(salary-settings.salary_floor), salary)
    else:
        base_values = values[selectable]
        for t_i, temp in enumerate(temps):
            logw = (base_values - np.nanmax(base_values)) / temp
            temp_done = 0
            while temp_done < per_temp_budget:
                n = min(batch_size, per_temp_budget - temp_done)
                # Gumbel perturbation + top-k = weighted sampling without replacement.
                g = rng.gumbel(size=(n, len(selectable)))
                keys = logw[None, :] + g
                local = np.argpartition(keys, -need, axis=1)[:, -need:]
                picks = selectable[local]
                batch_salary = salaries[picks].sum(axis=1) + locked_salary
                valid = (batch_salary >= settings.salary_floor) & (batch_salary <= settings.salary_cap)
                for row, salary in zip(picks[valid], batch_salary[valid]):
                    idx = np.concatenate((locked_idx, row))
                    names = frozenset(players_arr[idx])
                    score = float(values[idx].sum()) + 0.000010*(int(salary)-settings.salary_floor)
                    old = candidates.get(names)
                    if old is None or score > old[0]:
                        candidates[names] = (score, int(salary))
                temp_done += n
                processed += n
                if progress_callback:
                    frac = min(0.82, 0.05 + 0.77 * processed / max(1, total_budget))
                    progress_callback(frac, f"Building candidate lineups… {len(candidates):,} valid")
                # Once we have a deep diversified pool, extra random draws add
                # little lineup quality but can add substantial wait time.
                if len(candidates) >= target and temp_done >= min(5000, per_temp_budget):
                    break

    if progress_callback:
        progress_callback(0.86, f"Selecting {settings.lineup_count} portfolio lineups…")

    ranked = sorted([(v[0], v[1], k) for k, v in candidates.items()], reverse=True)
    if not ranked:
        raise RuntimeError("No feasible lineups found. Lower salary floor or relax locks/exclusions.")

    max_appearances = max(1, int(np.floor(settings.max_exposure * settings.lineup_count + 1e-9)))
    selected: list[frozenset[str]] = []
    exposure = {p: 0 for p in d.player.astype(str)}
    max_overlap = settings.roster_size - settings.min_unique

    for slot in range(settings.lineup_count):
        best = None
        best_adjusted = -1e18
        for score, salary, names in ranked:
            if names in selected:
                continue
            if any(len(names & old) > max_overlap for old in selected):
                continue
            if any(exposure.get(p, 0) + 1 > max_appearances for p in names):
                continue
            penalty = sum((exposure.get(p, 0) / max_appearances) ** 2 for p in names)
            adjusted = score - (0.30 + .025*(slot+1))*penalty
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best = names
        if best is None:
            raise RuntimeError(
                f"Could only construct {len(selected)} of {settings.lineup_count} lineups. "
                "Relax max exposure, minimum unique golfers, salary floor, or locks."
            )
        selected.append(best)
        for p in best:
            exposure[p] = exposure.get(p, 0) + 1

    dk_col = "dk_points_proxy" if "dk_points_proxy" in d.columns else "dk_proxy"
    rows, summaries = [], []
    for number, names in enumerate(selected, 1):
        lu = d[d.player.isin(names)].sort_values("salary", ascending=False)
        summaries.append({
            "lineup": number,
            "salary": int(lu.salary.sum()),
            "salary_left": settings.salary_cap - int(lu.salary.sum()),
            "sum_win_pct": float(lu.win_pct.sum()),
            "sum_top10_pct": float(lu.top10_pct.sum()),
            "avg_make_cut_pct": float(lu.make_cut_pct.mean()),
            "sum_dk_proxy": float(lu[dk_col].sum()),
        })
        for _, r in lu.iterrows():
            rows.append({
                "lineup": number, "player": r.player, "salary": int(r.salary),
                "win_pct": r.win_pct, "top10_pct": r.top10_pct,
                "make_cut_pct": r.make_cut_pct, "dk_points_proxy": r[dk_col],
            })
    portfolio = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    expo = portfolio.groupby("player").size().sort_values(ascending=False).rename("lineups").reset_index()
    expo["exposure_pct"] = 100 * expo.lineups / settings.lineup_count
    if progress_callback:
        progress_callback(1.0, "DFS portfolio complete.")
    return portfolio, summary, expo
