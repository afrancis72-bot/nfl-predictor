"""Pregame-safe QB-vs-Miami study for the 2026 NFL season.

One primary opposing QB per game (highest attempts, >=12).
Compares QB actual to HIS OWN per-game history strictly PRIOR to facing Miami.
For Week 1 only, 2025's latest eight games supply a transparently labelled
fallback; 2026 season-to-date is the normal reference for Weeks 2-10.
Regular-season QB partial DK points omit fumbles lost, returns and 2-pt scores.
Do not replace any production projections based on these descriptive results.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from props_data import fetch_weekly, clean_weekly, dk_classic_points
from ai2_nfl_matchups import canonical_team

FIELDS = ("attempts","passing_yards","passing_tds","interceptions",
          "carries","rushing_yards","rushing_tds")

def _score(row):
    return float(dk_classic_points(row, "QB"))

def _frame(raw):
    h=clean_weekly(raw)
    h=h[h.position.astype(str).str.upper().eq("QB")].copy()
    h=h[(h.season >= 2025) & (h.season<=2026) &
        (h.week >=1) & (h.week<=18)]
    h["defense"]=h.opponent_team.map(
        lambda x: canonical_team(x) if pd.notna(x) else "")
    h["dk_partial"]=h.apply(_score, axis=1)
    h["qb_key"]=h["player_id"].astype("string") if "player_id" in h else h["player_name"].astype("string")
    valid=h.qb_key.notna() & h.qb_key.str.len().gt(2)
    h.loc[~valid,"qb_key"]="NAME:"+h.loc[~valid,"player_name"].astype(str).str.casefold()
    return h.sort_values(["season","week","player_name"])

def _numbers(g):
    vals={x:float(pd.to_numeric(g[x],errors="coerce").mean()) for x in FIELDS}
    vals["dk_partial"]=float(g["dk_partial"].mean())
    vals["games"]=int(len(g))
    return vals

def run(raw):
    h=_frame(raw)
    games=h[(h.season.eq(2026)) & (h.week.lt(11)) &
            (h.defense.eq("MIA")) & h.attempts.ge(12)].copy()
    games=games.sort_values(["week","attempts"],ascending=[True,False])
    games=games.drop_duplicates(["season","week"],keep="first").sort_values("week")
    if len(games)<8:
        raise ValueError(f"Expected nine completed Miami opponents before 2026 W11, observed {len(games)}")
    reports=[]
    for _,g in games.iterrows():
        prior=h[(h.qb_key.eq(g["qb_key"])) &
                ((h.season.lt(2026)) |
                 ((h.season.eq(2026)) & h.week.lt(g.week)))].copy()
        prior_season=prior[prior.season.eq(2026)]
        prior_eight=prior.sort_values(["season","week"]).tail(8)
        if prior_season.empty:
            baseline=prior[prior.season.eq(2025)].tail(8)
            label="2025 final games (W1 fallback)"
        else:
            baseline=prior_season
            label="2026 season-to-date before game"
        if baseline.empty:  # New NFL rookie or missing history; do not invent
            reports.append({"week":int(g.week),"qb":g.player_name,
                            "offense":g.recent_team,
                            "error":"No pregame player history"})
            continue
        b=_numbers(baseline)
        actual=_numbers(g.to_frame().T)
        row={
            "week":int(g.week),"qb":str(g.player_name),
            "team":str(g.recent_team),"vs":"MIA",
            "baseline_scope":label,"baseline_games":b["games"],
            "pass_yd_baseline":round(b["passing_yards"],1),
            "pass_yd_mia":round(actual["passing_yards"],1),
            "pass_yd_delta":round(actual["passing_yards"]-b["passing_yards"],1),
            "pass_td_baseline":round(b["passing_tds"],2),
            "pass_td_mia":int(actual["passing_tds"]),
            "rush_yd_baseline":round(b["rushing_yards"],1),
            "rush_yd_mia":round(actual["rushing_yards"],1),
            "rush_td_baseline":round(b["rushing_tds"],2),
            "rush_td_mia":int(actual["rushing_tds"]),
            "dk_baseline":round(b["dk_partial"],2),
            "dk_mia":round(actual["dk_partial"],2),
            "dk_delta":round(actual["dk_partial"]-b["dk_partial"],2),
            "dk_pct_vs_norm":round(100*(actual["dk_partial"]/b["dk_partial"]-1),1)
                              if b["dk_partial"]>0 else None,
            "pass_attempts_baseline":round(b["attempts"],1),
            "pass_attempts_mia":int(actual["attempts"])
        }
        reports.append(row)
    matched=[r for r in reports if "dk_delta" in r]
    return {
        "method":"Before-game 2026 QB season-to-date average; for W1 use final eight completed 2025 games. Primary QB per opponent game, highest pass attempts >=12. Source nflverse weekly stats. Points exclude fumbles lost, 2pt conversions, returns.",
        "qb_games":len(games),"matched":len(matched),"rows":reports,
        "mean_dk_delta":round(float(np.mean([r["dk_delta"] for r in matched])),2),
        "median_dk_delta":round(float(np.median([r["dk_delta"] for r in matched])),2),
        "qb_above_own_average":sum(r["dk_delta"]>0 for r in matched),
        "mean_pass_yd_delta":round(float(np.mean([r["pass_yd_delta"] for r in matched])),1),
        "mean_pass_td_delta":round(float(np.mean([r["pass_td_mia"]-r["pass_td_baseline"] for r in matched])),2),
        "mean_attempt_delta":round(float(np.mean([r["pass_attempts_mia"]-r["pass_attempts_baseline"] for r in matched])),1),
        "notes":["Different opposing QBs each week; opponent quality and game script confound results.",
                 "Averages before games do not include Miami matchup outcomes or later weeks.",
                 "Week 2 has only one current-season prior-game sample; report baseline_games."]
    }

if __name__=="__main__":
    print(json.dumps(run(fetch_weekly([2025,2026])),indent=2))
