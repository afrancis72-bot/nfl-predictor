"""Research-only role target forecasts conditioned on prior-game teammate co-usage.

Only *previously observed* player/team games can enter training. A player
missing from a nflverse stat file is NOT claimed officially OUT. Here,
"co_observed" simply means both teammates have a usable weekly stat row.
The historical reconstructed DK pool is NOT a verified game-day active list.

Use a pregame model-pool proxy ONLY in retrospective research with explicit
research_proxy=True; never enable this for live lineups without independent
official status validation, provenance and new predictive acceptance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team
from props_data import clean_weekly

SOURCE="nflverse same-team prior-week co-observed WR targets (not verified active)"
POSITIONS=frozenset(("WR",))


def conditional_wr_targets(history, rates, pool, *, season, week,
                           strength=.5, min_co_games=2,
                           min_nonco_games=2, research_proxy=False):
    """Return unchanged player universe + adjusted WR targets and audit ledger.

    For each WR, choose a distinct pregame WR teammate with the largest
    historical target expectation among current modeled WRs. Compare the focal
    player's prior games where the anchor WAS recorded with those where he was
    NOT recorded (which does *not* establish medical absence). Require both
    groups, shrink conditional mean to the individual pregame baseline, and
    cap per-player changes at 20%. Never add or remove player opportunity on
    the basis of current-week box scores.
    """
    if (not np.isfinite(float(strength)) or not 0<=float(strength)<=1
        or min_co_games<2 or min_nonco_games<2):
        raise ValueError("Invalid pregame conditional target experiment settings")
    if strength>0 and research_proxy is not True:
        raise ValueError("Co-observation is NOT verified ACTIVE; retrospective research only")
    if not isinstance(rates,pd.DataFrame) or not isinstance(pool,pd.DataFrame):
        raise ValueError("Pregame player rates and selected player pool required")
    needed={"player","team","position","targets_pg","games_played",
            "latest_observed_season","latest_observed_week"}
    if not needed.issubset(rates.columns):
        raise ValueError("Incomplete pregame rates for cohort experiment")
    if not {"Name","TeamAbbrev","Position"}.issubset(pool.columns):
        raise ValueError("Incomplete pregame player slate for cohort experiment")
    season,week=int(season),int(week)
    if season<2020 or not 1<=week<=22:
        raise ValueError("Invalid target NFL season/week")
    result=rates.copy()
    ledger=[]
    if strength==0:
        return result,ledger

    # Cross-feed normalization resolves full-name weekly player identities.
    histories=clean_weekly(history)
    histories=histories[
        histories["season"].eq(season)&histories["week"].lt(week)&
        histories["position"].astype(str).str.upper().eq("WR")
    ].copy()
    histories["_name"]=histories["player_name"].map(_norm_name)
    histories["_team"]=histories["recent_team"].map(canonical_team)
    # Same team-week-player duplicates are an identity error; don't silently
    # sum one player's targets twice or allow future-week contamination.
    if histories.duplicated(["season","week","_team","_name"]).any():
        raise ValueError("Duplicate historical WR in a same-team same-week box score")
    histories["targets"]=pd.to_numeric(histories["targets"],errors="coerce")
    histories=histories[histories["targets"].notna()&
                        histories["targets"].ge(0)].copy()
    result["_role_cohort_key"]=result["player"].map(_norm_name)
    result["_role_cohort_team"]=result["team"].map(canonical_team)
    team_wr={}
    for row in pool.to_dict("records"):
        if str(row["Position"]).upper()!="WR":
            continue
        team=canonical_team(row["TeamAbbrev"]);name=_norm_name(row["Name"])
        if name:
            team_wr.setdefault(team,set()).add(name)
    for team,names in sorted(team_wr.items()):
        own=result[
            result["_role_cohort_team"].eq(team)&
            result["position"].astype(str).str.upper().eq("WR")&
            result["_role_cohort_key"].isin(names)].copy()
        if len(own)<2:
            continue
        if own.duplicated(["_role_cohort_key"]).any():
            raise ValueError("Ambiguous same-team WR rates in cohort study")
        team_h=histories[histories["_team"].eq(team)]
        by_player={}
        for name,group in team_h.groupby("_name"):
            by_player[name]=group.sort_values(["season","week"]).tail(8)
        anchors=sorted(own.to_dict("records"),key=lambda row:(
            -float(row["targets_pg"]),row["_role_cohort_key"]))
        for player in anchors:
            name=player["_role_cohort_key"]
            possible=[p for p in anchors if p["_role_cohort_key"]!=name]
            if not possible or name not in by_player:
                continue
            anchor=possible[0]["_role_cohort_key"]
            # Only observed contemporaneous rows; missing means
            # "not in weekly WR box score", not automatically "injured OUT".
            rows=by_player[name]
            present=set(by_player.get(anchor,pd.DataFrame()).get(
                "week",pd.Series(dtype=int)).astype(int).tolist())
            co=rows[rows["week"].isin(present)]
            nonco=rows[~rows["week"].isin(present)]
            base=float(player["targets_pg"])
            games=int(player["games_played"])
            if (not np.isfinite(base) or base<=0 or games<4 or
                len(co)<min_co_games or len(nonco)<min_nonco_games):
                continue
            if (int(player["latest_observed_season"])!=season or
                int(player["latest_observed_week"])>=week):
                raise ValueError("Target-week usage leaked into cohort forecast")
            # More recent co-observations receive greater weight; a four-game
            # prior shrinks noisy two-game teammate splits toward baseline.
            wg=.87**np.arange(len(co)-1,-1,-1)
            conditional=float(np.average(co["targets"].to_numpy(float),weights=wg))
            reliability=len(co)/(len(co)+4.)
            shrunk=base+reliability*(conditional-base)
            forecast=max(.8*base,min(1.2*base,
                        base+float(strength)*(shrunk-base)))
            mask=(result["_role_cohort_team"].eq(team)&
                  result["_role_cohort_key"].eq(name)&
                  result["position"].astype(str).str.upper().eq("WR"))
            if mask.sum()!=1:
                raise ValueError("Ambiguous target row for conditional WR")
            result.loc[mask,"targets_pg"]=forecast
            ledger.append({
                "team":team,"player":player["player"],"anchor":anchor,
                "anchor_is_pregame_pool_proxy_not_certified_active":True,
                "co_observed_games":int(len(co)),
                "not_co_observed_games":int(len(nonco)),
                "baseline_targets_pg":base,"co_observed_targets_pg":conditional,
                "forecast_targets_pg":forecast,
                "delta_targets_pg":forecast-base,
                "source":SOURCE,"research_only":True,
                "verified_game_day_active":False,
                "target_season":season,"target_week":week,
            })
    return result.drop(columns=["_role_cohort_key","_role_cohort_team"]),ledger
