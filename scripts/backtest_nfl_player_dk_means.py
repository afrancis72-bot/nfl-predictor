"""Frozen pregame NFL player-DK mean / tail validation (not a live-score patch).

Build player pools from PRIOR player-stat games and published matchups only.
Select completed games *after* forecasts, for outcome comparison only.
No current-week box-score data, postgame odds, or outcomes enter predictions.
This benchmark deliberately does not infer missing-player OUT statuses from
their future absence. It is a scoring accuracy test, not a full contest ROI.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from ai2_nfl_matchups import canonical_team
from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_team_environment import predict_team_environment
from ai2_nfl_snap_roles import predict_snap_roles, fetch_snap_counts
from ai2_nfl_game_market import market_totals_for_slate
from ai2_nfl_matchup_diagnostics import defensive_allowance
from ai2_nfl_defensive_efficiency import build_defense_profiles
from dfs_stats_engine import simulate_stat_driven_dfs
from props_data import fetch_weekly, clean_weekly, make_rates, dk_classic_points

GAMES_SOURCE = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"


def pregame_game_pool(rates, away, home, season, week):
    """Every roster member comes from BEFORE week, never from future box scores."""
    away,home=canonical_team(away),canonical_team(home)
    sides=(away,home)
    selections=[]
    for team in sides:
        candidates=[]
        for r in rates.to_dict("records"):
            if canonical_team(r["team"])!=team:
                continue
            pos=str(r["position"]).upper()
            if pos not in {"QB","RB","WR","TE"}:
                continue
            if (int(r.get("games_played") or 0)<2
                or int(r.get("latest_observed_season") or 0)!=season
                or int(r.get("latest_observed_week") or 0)>=week
                or week-int(r.get("latest_observed_week") or 0)>3):
                continue
            def value(field):
                try:
                    v=float(r.get(field,0))
                    return v if np.isfinite(v) else 0.
                except (TypeError,ValueError):
                    return 0.
            p=value("pass_attempts_pg")
            touches=value("carries_pg")+value("targets_pg")
            if pos=="QB" and p<18:
                continue
            if pos=="RB" and touches<6:
                continue
            if pos=="WR" and value("targets_pg")<3.5:
                continue
            if pos=="TE" and value("targets_pg")<2.3:
                continue
            candidates.append(r)
        qbs=sorted((r for r in candidates if str(r["position"])=="QB"),
                   key=lambda r:-float(r.get("pass_attempts_pg") or 0))
        if not qbs:
            return pd.DataFrame()
        candidates=[qbs[0]]+[r for r in candidates if str(r["position"])!="QB"]
        for r in candidates:
            pos=str(r["position"]).upper()
            name=str(r["player"])
            selections.append({
                "ID":f"{team}-{pos}-{_norm_name(name)}","Name":name,
                "TeamAbbrev":team,"Position":pos,"Salary":5000,
                "game":f"{away}@{home}","proj":10.,"optimizer_eligible":True
            })
        selections.append({"ID":f"{team}-DST","Name":f"{team} DST",
                           "TeamAbbrev":team,"Position":"DST",
                           "Salary":3000,"game":f"{away}@{home}",
                           "proj":6.5,"optimizer_eligible":True})
    return pd.DataFrame(selections)


def actual_partial(row,position):
    return dk_classic_points(row,position)


def evaluate(weekly,schedules,snaps,season,weeks,*,games_per_week=2,n_sims=320,
             role_share_concentration=None,market_weight=0.,receiving_td_weight=0.,
             defense_weight=0.,other_yard_strength=0.):
    h=clean_weekly(weekly)
    results=[]
    failures=[]
    for week in weeks:
        rates=make_rates(weekly,season,week)
        environment=predict_team_environment(weekly,season,week)
        snap=predict_snap_roles(weekly,snaps,season,week,team_environment=environment)
        market_totals=market_totals_for_slate(schedules,season,week) if market_weight else {}
        defensive=build_defense_profiles(
            defensive_allowance(weekly,season,week),season,week) if defense_weight else {}
        week_games=schedules[
            (schedules["season"].eq(season))&
            (schedules["week"].eq(week))&
            schedules["game_type"].astype(str).eq("REG")&
            schedules["home_score"].notna()&schedules["away_score"].notna()
        ].copy()
        if "weekday" in week_games:
            week_games=week_games[week_games["weekday"].eq("Sunday")].copy()
        week_games=week_games.sort_values("game_id").head(games_per_week)
        for _,match in week_games.iterrows():
            away,home=canonical_team(match["away_team"]),canonical_team(match["home_team"])
            pool=pregame_game_pool(rates,away,home,season,week)
            if pool.empty:
                failures.append(f"{season} W{week} {away}@{home}: no eligible prior-week QB")
                continue
            try:
                players,draws,_=simulate_stat_driven_dfs(
                    pool,rates,n_sims=n_sims,seed=season*100+week,
                    allow_role_proxies=False,team_environment=environment,
                    snap_profiles=snap,injury_presence={},
                    role_share_concentration=role_share_concentration,
                    market_game_totals=market_totals,market_weight=market_weight,
                    receiving_td_weight=receiving_td_weight,
                    defense_profiles=defensive,defense_weight=defense_weight,
                    other_yard_strength=other_yard_strength)
            except (KeyError,ValueError,TypeError,IndexError) as e:
                failures.append(f"{season} W{week} {away}@{home}: {str(e)[:150]}")
                continue
            observations=h[(h["season"].eq(season))&
                           (h["week"].eq(week))&
                           h["recent_team"].map(canonical_team).isin((away,home))].copy()
            lookup={}
            for a in observations.to_dict("records"):
                k=(_norm_name(a["player_name"]),canonical_team(a["recent_team"]),
                   str(a["position"]).upper())
                lookup.setdefault(k,[]).append(a)
            rate_lookup={
                (_norm_name(r["player"]),canonical_team(r["team"]),
                 str(r["position"]).upper()):r
                for r in rates.to_dict("records")
            }
            for i,p in players.iterrows():
                if p["Position"]=="DST":
                    continue
                k=(_norm_name(p["Name"]),p["TeamAbbrev"],p["Position"])
                observed=lookup.get(k,[])
                if len(observed)!=1:
                    continue
                actual=actual_partial(observed[0],p["Position"])
                if not np.isfinite(actual):
                    continue
                pre=rate_lookup.get(k)
                if not pre:
                    continue
                hist=dk_classic_points({
                    "passing_yards":pre.get("pass_yards_pg"),
                    "passing_tds":pre.get("passing_tds_pg"),
                    "interceptions":pre.get("interceptions_pg"),
                    "rushing_yards":pre.get("rushing_yards_pg"),
                    "rushing_tds":pre.get("rushing_tds_pg"),
                    "receiving_yards":pre.get("receiving_yards_pg"),
                    "receptions":pre.get("receptions_pg"),
                    "receiving_tds":pre.get("receiving_tds_pg")
                },p["Position"])
                if not np.isfinite(hist):
                    continue
                scores=draws[:,i]
                results.append({
                    "season":season,"week":week,"player":p["Name"],
                    "team":p["TeamAbbrev"],"position":p["Position"],
                    "actual_partial_dk":float(actual),
                    "history_mean_partial_dk":float(hist),
                    "simulated_mean_dk":float(np.mean(scores)),
                    "simulated_p90_dk":float(np.percentile(scores,90))
                })
    return pd.DataFrame(results),failures


def summary(rows):
    if rows.empty:
        return {"n":0}
    y=rows["actual_partial_dk"].to_numpy(float)
    base=rows["history_mean_partial_dk"].to_numpy(float)
    sim=rows["simulated_mean_dk"].to_numpy(float)
    p90=rows["simulated_p90_dk"].to_numpy(float)
    return {"n":len(rows),
            "history_mean_mae":round(float(np.abs(base-y).mean()),3),
            "sim_mean_mae":round(float(np.abs(sim-y).mean()),3),
            "sim_mean_signed_bias":round(float((sim-y).mean()),3),
            "simulated_p90_actual_coverage_pct":round(float(100*np.mean(y<=p90)),1),
            "sim_mean_30plus_count":int((sim>=30).sum()),
            "sim_mean_30plus_actual_mean":round(float(y[sim>=30].mean()),2)
               if (sim>=30).any() else None}


def run():
    historical=fetch_weekly([2024,2025,2026])
    schedule=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    output={}
    for season,weeks in [(2025,[10,11,12]),(2026,[4,5,6,7,8])]:
        records,issues=evaluate(historical,schedule,snaps,season,weeks)
        if records.empty:
            raise ValueError(f"No matched actual-vs-pregame scoring observations for {season}: {issues[:3]}")
        output[str(season)]={
            "period":f"{season} weeks {min(weeks)}-{max(weeks)}; two Sunday games per week, when completed",
            "summary":summary(records),
            "by_position":{pos:summary(group) for pos,group in records.groupby("position")},
            "Puka_or_Bijan":[rec for rec in records.to_dict("records")
                              if _norm_name(rec["player"]) in {"puka nacua","bijan robinson"}],
            "skipped_games":len(issues),"skip_examples":issues[:4]}
    output["limitations"]=(
        "Partial DraftKings scoring lacks fumbles/returns; pregame pool based on"
        " prior-week usage, evaluated only when player played; historical injured"
        " absences not modeled because actual Sunday injury evidence was not frozen."
        " This is a scoring distribution benchmark, not lineup ROI.")
    return output


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
