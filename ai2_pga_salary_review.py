"""PGA pre-entry salary-band review for the LIVE unified Team AI² Pro engine.

No model forecasts are invented or replaced. Recommendations explain the actual
V11 simulation result, measured golfer evidence and data quality. Separately
reviewed portfolios reuse the exact original simulated tournament worlds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from portfolio_optimizer import PortfolioSettings, optimize_portfolio

STAT_LABELS = {
    "sg_approach": "Approach SG",
    "sg_ott": "Off-the-tee SG",
    "sg_arg": "Around-green SG",
    "sg_putting": "Putting SG",
    "birdie_rate": "Birdie rate",
    "bogey_avoidance": "Bogey avoidance",
    "driving_distance": "Driving distance",
    "driving_accuracy": "Driving accuracy",
}
TIERS = ("$6K", "$7K", "$8K", "$9K", "$10K+")


def _numeric(data: pd.DataFrame, name: str) -> pd.Series:
    if name not in data:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return pd.to_numeric(data[name], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _z(data: pd.Series) -> pd.Series:
    std = data.std(ddof=0)
    if not np.isfinite(std) or std < 1e-10:
        return pd.Series(0.0, index=data.index)
    return ((data - data.mean()) / std).fillna(0.0)


def _tier(salary) -> str:
    if salary >= 10000: return "$10K+"
    if salary >= 9000: return "$9K"
    if salary >= 8000: return "$8K"
    if salary >= 7000: return "$7K"
    return "$6K"


def build_review(projections: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Rank only actual simulated PGA golfers, with traceable, nonfabricated reasons."""
    required = {"player", "salary", "v11_mean_dk", "v11_p90_dk", "make_cut_pct"}
    if not required.issubset(projections):
        raise ValueError("Run the PGA tournament simulation to create golfer reviews.")
    p = projections.copy().reset_index(drop=True)
    p["salary"] = _numeric(p, "salary")
    p = p.loc[p.salary.ge(6000) & p.salary.notna()].copy()
    if p.empty: return p
    p["tier"] = p.salary.map(_tier)
    p["mean_dk"] = _numeric(p, "v11_mean_dk")
    p["p90_dk"] = _numeric(p, "v11_p90_dk")
    p["cut_pct"] = 100 * _numeric(p, "make_cut_pct")
    p["mean_per_1k"] = p.mean_dk / (p.salary / 1000)
    p["p90_per_1k"] = p.p90_dk / (p.salary / 1000)
    if p[["mean_dk","p90_dk","cut_pct"]].isna().any().any():
        raise ValueError("Simulation contains missing mean, P90 or cut probabilities.")
    p["priority"] = 0.0
    for tier, ix in p.groupby("tier").groups.items():
        rows=p.loc[ix]
        p.loc[ix, "priority"] = (
            .35*_z(rows.mean_per_1k) + .35*_z(rows.p90_per_1k) +
            .30*_z(rows.cut_pct)
        )
    ranks={}
    for col in STAT_LABELS:
        vals = _numeric(p,col)
        if vals.notna().sum() >= 8 and float(vals.std(ddof=0)) > 1e-9:
            ranks[col] = (vals.rank(ascending=False, method="min"), int(vals.notna().sum()))
    market_only = not int(metadata.get("independent_sg_match",0)) and not int(metadata.get("otis_match",0))
    reasons=[]; concerns=[]; evidence=[]
    for ix,r in p.iterrows():
        why=[f"Simulated {r.mean_dk:.1f} mean / {r.p90_dk:.1f} P90 DraftKings points",
             f"{r.mean_per_1k:.1f} mean DK per $1,000 salary",
             f"{r.cut_pct:.1f}% modeled cut chance (uncalibrated)"]
        warnings=[]
        strengths=[]
        weaknesses=[]
        for col,(rank,available) in ranks.items():
            if pd.isna(r.get(col)): continue
            position=int(rank.loc[ix])
            share=position/available
            evidence_text=f"{STAT_LABELS[col]} #{position}/{available} (measured)"
            if share <= .30: strengths.append((share,evidence_text))
            if share >= .75: weaknesses.append((share,evidence_text))
        why.extend("Strength: "+e for _,e in sorted(strengths)[:2])
        warnings.extend("Weakness: "+e for _,e in sorted(weaknesses,reverse=True)[:2])
        if market_only:
            warnings.append("Salary-only baseline; no independent golfer skill measured")
        elif not strengths:
            warnings.append("Insufficient measured standout skill evidence for this golfer")
        fit_strokes=r.get("rrg_fit_adjustment_strokes",np.nan)
        if pd.notna(fit_strokes) and abs(float(fit_strokes)) > .005:
            why.append(f"RRG course-fit input moved baseline {float(fit_strokes):+.2f} strokes/round")
            warnings.append("Course-fit adjustment is experimental, not historical proof")
        if r.cut_pct < 60:
            warnings.append("Elevated modeled missed-cut risk")
        rounds=r.get("measured_rounds",np.nan)
        if pd.notna(rounds) and float(rounds)<16:
            warnings.append(f"Only {int(rounds)} observed SG rounds")
        if r.salary < 8000 and (market_only or r.cut_pct<60):
            warnings.append("Low-salary selection needs manual justification")
        reasons.append("; ".join(why))
        concerns.append("; ".join(dict.fromkeys(warnings)) or "Golfer outcomes remain volatile")
        evidence.append("Market only" if market_only else
                        ("Measured skill + projection" if strengths else "Projection; limited skill detail"))
    p["why"] = reasons
    p["concerns"] = concerns
    p["evidence"] = evidence
    p = p.sort_values(["tier","priority","player"], ascending=[True,False,True]).reset_index(drop=True)
    p["tier_rank"] = p.groupby("tier").cumcount()+1
    p["overall_mean_rank"] = p.mean_dk.rank(ascending=False,method="min").astype(int)
    return p


def attach_comparison(review: pd.DataFrame, otis: pd.DataFrame | None,
                      rrg_fit: pd.DataFrame | None) -> pd.DataFrame:
    """Benchmark external *ranks*, never insert them into our modeled probabilities."""
    x=review.copy()
    from ai2_pga_oneclick import key
    x["_key"]=x.player.map(key)
    for frame,field,target in (
        (otis,"otis_rank","OTIS rank"),
        (rrg_fit,"rrg_fit_rank_audit","RRG course-fit rank"),
    ):
        if frame is None or frame.empty or field not in frame: continue
        f=frame[["player",field]].copy()
        f["_key"]=f.player.map(key)
        if f["_key"].duplicated().any(): continue
        f[target]=pd.to_numeric(f[field],errors="coerce")
        x=x.merge(f[["_key",target]],on="_key",how="left",validate="one_to_one")
        x[target+" minus our mean DK rank"]=x[target]-x.overall_mean_rank
    return x.drop(columns=["_key"])


def rebuild_reviewed(result: dict, *, vetoes=(), locks=(), lineup_count=10,
                     max_exposure=.20, salary_floor=46500, min_unique=3,
                     strategy="Balanced / Single Entry", seed=None) -> dict:
    """Reoptimize a reviewed portfolio without rerunning or editing simulated golf."""
    predictions=result["projections"].copy()
    available=set(predictions.player)
    vetoes=set(vetoes)&available
    locks=set(locks)&available
    if vetoes&locks:
        raise ValueError("A locked golfer is vetoed. Resolve the conflict before building.")
    sim_players=predictions.player.astype(str).tolist()
    draws=np.asarray(result["raw_draws"])
    if draws.ndim!=2 or draws.shape[1]!=len(sim_players):
        raise ValueError("PGA simulated golfer worlds are missing/misaligned.")
    if not np.allclose(draws.mean(axis=0),predictions["v11_mean_dk"].to_numpy(float),atol=.015):
        raise ValueError("PGA simulation is stale; cannot safely rebuild.")
    settings=PortfolioSettings(lineup_count=int(lineup_count),salary_cap=50000,
        salary_floor=int(salary_floor),min_player_salary=3000,roster_size=6,
        max_exposure=float(max_exposure),min_unique=int(min_unique),
        strategy=strategy,candidate_samples=15000,target_candidates=1100,
        candidate_pool_size=min(len(predictions),80),
        seed=int(seed if seed is not None else result["metadata"]["seed"]))
    portfolio,summary,exposure=optimize_portfolio(
        predictions,settings,locks=sorted(locks),excludes=sorted(vetoes))
    if len(summary)!=lineup_count or bool(set(portfolio.player)&vetoes):
        raise ValueError("Reviewed portfolio could not meet lineup and veto constraints.")
    if (exposure["exposure_pct"] > 100*float(max_exposure)+1e-6).any():
        raise ValueError("Golfer exposure limit violated.")
    if not summary.salary.between(salary_floor,50000).all():
        raise ValueError("Lineup salary limits violated.")
    lookup={name:i for i,name in enumerate(sim_players)}
    worlds=[]
    for no,g in portfolio.groupby("lineup"):
        subset=draws[:,[lookup[name] for name in g.player]]
        scores=subset.sum(axis=1)
        worlds.append({"lineup":int(no),
            "sim_lineup_mean":round(float(scores.mean()),2),
            "sim_lineup_p10":round(float(np.quantile(scores,.10)),2),
            "sim_lineup_p90":round(float(np.quantile(scores,.90)),2),
            "sim_lineup_p95":round(float(np.quantile(scores,.95)),2)})
    summary=summary.merge(pd.DataFrame(worlds),on="lineup",how="left",validate="one_to_one")
    return {"lineups":portfolio,"summary":summary,"exposure":exposure,
            "vetoes":sorted(vetoes),"locks":sorted(locks)}
