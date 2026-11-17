"""Research-only NFL drive -> pass attempts conservation and causal-agnostic accounting.

Game outcomes are read ONLY after simulate_stat_driven_dfs returns. The
decomposition is an exact accounting identity, not evidence of causal effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ai2_nfl_auto_baselines import _norm_name
from ai2_nfl_matchups import canonical_team
from ai2_nfl_pbp_drives import engine_team


def observed_pbp_team_games(pbp, drives, schedules):
    """Convert actual completed nflverse scrimmage events to team-game truth.

    PBP pass_attempt, sack and rush_attempt mark mutually exclusive counted
    scrimmage plays. The eligible fixed-drive count is a PROXY and excludes
    ambiguous/unverified possessions in the existing extractor.
    """
    needed={"season","week","season_type","game_id","play_id","posteam",
            "pass_attempt","rush_attempt","sack"}
    if missing:=needed-set(pbp.columns):
        raise ValueError(f"Cannot count observed scrimmage plays: missing {sorted(missing)}")
    games=schedules.copy()
    games=games[(games["game_type"].astype(str).eq("REG")) &
                (games["home_score"].notna())&(games["away_score"].notna())]
    games["season"]=pd.to_numeric(games["season"],errors="coerce")
    games["week"]=pd.to_numeric(games["week"],errors="coerce")
    game_info={}
    for g in games.to_dict("records"):
        key=(int(g["season"]),int(g["week"]),str(g["game_id"]))
        name=f'{canonical_team(g["away_team"])}@{canonical_team(g["home_team"])}'
        if key in game_info and game_info[key]!=name:
            raise ValueError("Ambiguous nflverse schedule game identity")
        game_info[key]=name
    f=pbp[pbp["season_type"].astype(str).eq("REG")].copy()
    f=f[f["posteam"].notna() & f["play_id"].notna()].copy()
    for k in ("season","week","pass_attempt","rush_attempt","sack"):
        f[k]=pd.to_numeric(f[k],errors="coerce").fillna(0.)
    f["team"]=f["posteam"].map(canonical_team)
    f["pass"]=f["pass_attempt"].gt(0)
    f["rush"]=f["rush_attempt"].gt(0)
    f["sacked"]=f["sack"].gt(0)
    f=f[f[["pass","rush","sacked"]].any(axis=1)].copy()
    overlaps=f[f[["pass","rush","sacked"]].sum(axis=1)>1]
    if len(overlaps):
        example_cols=[x for x in ("season","week","game_id","play_id","play_type",
                    "pass_attempt","rush_attempt","sack") if x in overlaps.columns]
        counts=overlaps.groupby(["pass","rush","sacked"],dropna=False).size().to_dict()
        raise ValueError(f"PBP ambiguous scrimmage flags: n={len(overlaps)}, "
                         f"flag combinations={counts}, "
                         f"first examples={overlaps[example_cols].head(8).to_dict('records')}")
    if f.duplicated(["season","week","game_id","play_id"]).any():
        raise ValueError("Duplicate PBP game play IDs")
    counts=(f.groupby(["season","week","game_id","team"],sort=True)
              .agg(actual_pass_attempts=("pass","sum"),
                   actual_sacks=("sacked","sum"),
                   actual_rushes=("rush","sum")).reset_index())
    dc=drives.copy()
    dc["offense"]=dc["offense"].map(canonical_team)
    n_drives=(dc.groupby(["season","week","game_id","offense"],sort=True)
                .size().rename("eligible_observed_drives").reset_index()
                .rename(columns={"offense":"team"}))
    counts=counts.merge(n_drives,on=["season","week","game_id","team"],
                        how="left",validate="one_to_one")
    output={}
    for r in counts.to_dict("records"):
        game_id=(int(r["season"]),int(r["week"]),str(r["game_id"]))
        game=game_info.get(game_id)
        if game is None:
            continue
        team=canonical_team(r["team"])
        if team not in game.split("@"):
            raise ValueError("PBP offensive team not in scheduled NFL matchup")
        a=int(r["actual_pass_attempts"])
        s=int(r["actual_sacks"])
        rushing=int(r["actual_rushes"])
        drives_count=r["eligible_observed_drives"]
        if a<=0 or s<0 or rushing<0 or pd.isna(drives_count):
            raise ValueError("Missing PBP passing/rushing counts or eligible drives")
        key=(game_id[0],game_id[1],game,team)
        if key in output:
            raise ValueError("Duplicate PBP team-game")
        output[key]=dict(season=game_id[0],week=game_id[1],game=game,team=team,
            actual_pass_attempts=a,actual_sacks=s,actual_rushes=rushing,
            actual_dropbacks=a+s,actual_plays=a+s+rushing,
            eligible_observed_drives=int(drives_count),
            observed_game_id=str(r["game_id"]))
    return output


def _draws(obj, key, n_sims):
    arr=np.asarray(obj.get(key),dtype=float)
    if arr.shape!=(n_sims,) or not np.isfinite(arr).all() or (arr<0).any():
        raise ValueError("Missing or invalid sampled team accounting array: "+key)
    return arr


def component_row(*, season, week, game, team, players, stat_draws,
                  actual, observations, pregame_roles, n_sims, engine_name):
    """One team-game, forecast made before reading postgame PBP and WR outcomes."""
    label=engine_team(team)
    team=canonical_team(team)
    td=stat_draws.get(f"__TEAM_PLAYS__|{label}")
    env=stat_draws.get(f"__TEAM_ENVIRONMENT__|{label}")
    opp=stat_draws.get(f"__OPPORTUNITY__|{label}")
    other=stat_draws.get(f"__OTHER_RECEIVER__|{label}")
    if any(not isinstance(x,dict) for x in (td,env,opp,other)):
        raise ValueError("Missing team play/pass/receiver budget diagnostics")
    plays=_draws(td,"simulated_plays",n_sims)
    drops=_draws(td,"simulated_dropbacks",n_sims)
    rushes=_draws(td,"simulated_rushes",n_sims)
    attempts=_draws(td,"simulated_pass_attempts",n_sims)
    sacks=_draws(td,"simulated_sacks",n_sims)
    if (not np.array_equal(plays,drops+rushes) or
        not np.array_equal(drops,attempts+sacks)):
        raise ValueError("Failed simulation pass/run/sack conservation")
    k=("actual_pass_attempts","actual_sacks","actual_rushes","actual_dropbacks","actual_plays")
    if any(k0 not in actual or float(actual[k0])<0 for k0 in k):
        raise ValueError("Incomplete actual PBP accounting")
    act=float(actual["actual_pass_attempts"])
    act_sacks=float(actual["actual_sacks"])
    act_plays=float(actual["actual_plays"])
    act_drops=float(actual["actual_dropbacks"])
    act_rushes=float(actual["actual_rushes"])
    if (act_plays<=0 or act_drops<=0 or
        not np.isclose(act_plays,act_drops+act_rushes) or
        not np.isclose(act_drops,act+act_sacks)):
        raise ValueError("Actual PBP scrimmage conservation failed")

    sim_plays=float(plays.mean())
    sim_drops=float(drops.mean())
    sim_sacks=float(sacks.mean())
    sim_att=float(attempts.mean())
    observed_drop_fraction=act_drops/act_plays
    sim_drop_fraction=sim_drops/sim_plays
    # Exact telescoping identity:
    # (simAtt - actualAtt) = (simPlays - actualPlays)*actualDropFraction
    # + simPlays*(simDropFraction - actualDropFraction)
    # - (simSacks - actualSacks)
    plays_c=(sim_plays-act_plays)*observed_drop_fraction
    pass_run_c=sim_plays*(sim_drop_fraction-observed_drop_fraction)
    sacks_c=-(sim_sacks-act_sacks)
    total=sim_att-act
    if not np.isclose(plays_c+pass_run_c+sacks_c,total,rtol=0,atol=1e-8):
        raise ValueError("Pass attempt decomposition failed exact reconciliation")
    scores={"WR1":np.zeros(n_sims),"WR2":np.zeros(n_sims),
            "WR3":np.zeros(n_sims),"WR4+":np.zeros(n_sims),
            "WR_UNRANKED":np.zeros(n_sims),"RB":np.zeros(n_sims),
            "TE":np.zeros(n_sims),"OTHER":_draws(other,"targets",n_sims)}
    ranks={x["player_key"]:x["role"] for x in pregame_roles}
    selected=players[(players["TeamAbbrev"].map(canonical_team).eq(team)) &
                     (players["Position"].isin(["WR","RB","TE"]))]
    for player in selected.to_dict("records"):
        pos=str(player["Position"]).upper()
        key=(ranks.get(_norm_name(player["Name"]),"WR_UNRANKED")
             if pos=="WR" else pos)
        if key not in scores:raise ValueError("Unexpected pass target allocation group")
        record=stat_draws.get(str(player["Name"]))
        if not isinstance(record,dict):
            raise ValueError("Missing modeled player stat distribution")
        scores[key]+=_draws(record,"targets",n_sims)
    if not np.array_equal(sum(scores.values()),attempts):
        raise ValueError("All simulated receiver targets plus OTHER must equal attempts")

    obs=observations[observations["recent_team"].map(canonical_team).eq(team)]
    wr_truth={"WR1":0.,"WR2":0.,"WR3":0.,"WR4+":0.,
              "WR_UNRANKED":0.,"RB":0.,"TE":0.}
    for player in obs.to_dict("records"):
        pos=str(player.get("position","")).upper()
        if pos not in ("RB","TE","WR"):continue
        value=pd.to_numeric(player.get("targets"),errors="coerce")
        if not np.isfinite(value):continue
        role=ranks.get(_norm_name(player["player_name"]),"WR_UNRANKED") if pos=="WR" else pos
        wr_truth[role]+=float(value)

    prior=stat_draws.get(f"__TEAM_PASS_BASELINE__|{label}",{})
    poss=stat_draws.get(f"__POSSESSION_TEAM__|{label}")
    if engine_name=="possession" and not isinstance(poss,dict):
        raise ValueError("Possession engine did not produce simulated drives")
    if engine_name=="neutral" and poss is not None:
        raise ValueError("Neutral engine unexpectedly used possession worlds")
    pregame_pass=float(td["pregame_pass_attempts"])
    pregame_plays=float(td["expected_plays"])
    pregame_drops=float(td["pregame_expected_dropbacks"])
    pregame_fraction=float(td["pregame_pass_fraction"])
    pregame_sack_rate=float(td["pregame_sack_rate"])
    result={
        "season":int(season),"week":int(week),"game":str(game),"team":team,
        "engine":str(engine_name),
        "actual_plays":act_plays,"actual_dropbacks":act_drops,
        "actual_pass_attempts":act,"actual_sacks":act_sacks,
        "actual_rushes":act_rushes,
        "eligible_observed_drives":int(actual["eligible_observed_drives"]),
        "pregame_expected_plays":pregame_plays,
        "pregame_expected_dropbacks":pregame_drops,
        "pregame_pass_fraction":pregame_fraction,
        "pregame_sack_rate":pregame_sack_rate,
        "pregame_pass_attempts":pregame_pass,
        "pregame_team_env_pass_attempts":env.get("team_pass_attempts_prior"),
        "pregame_historical_qb_baseline":prior.get("team_pass_attempts_baseline"),
        "sim_plays":sim_plays,"sim_dropbacks":sim_drops,
        "sim_pass_attempts":sim_att,"sim_sacks":sim_sacks,"sim_rushes":float(rushes.mean()),
        "sim_drop_fraction":sim_drop_fraction,
        "actual_drop_fraction":observed_drop_fraction,
        "sim_sack_rate":sim_sacks/sim_drops,
        "actual_sack_rate":act_sacks/act_drops,
        "plays_error_component":plays_c,
        "pass_run_error_component":pass_run_c,
        "sacks_error_component":sacks_c,
        "total_attempts_error":total,
        "sim_drives":None if poss is None else float(_draws(poss,"simulated_drives",n_sims).mean()),
        "sim_plays_per_drive":None if poss is None else float(
            sim_plays/_draws(poss,"simulated_drives",n_sims).mean()),
        "observed_plays_per_eligible_drive":act_plays/int(actual["eligible_observed_drives"]),
        "simulated_target_allocation":{a:float(arr.mean()) for a,arr in scores.items()},
        "actual_known_position_and_rank_targets":wr_truth,
        "simulated_other_bucket_is_not_a_player_target_count":True,
    }
    return result


def report_metrics(frame):
    if frame.empty:
        raise ValueError("No game-level accounting data")
    keys=["season","week","game","team","engine"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate team-game accounting observations")
    def metrics(pred,actual):
        a=frame[pred].to_numpy(float)-frame[actual].to_numpy(float)
        return {"bias":round(float(a.mean()),3),
                "mae":round(float(np.abs(a).mean()),3),
                "underprojected":int((a<0).sum())}
    out={
        "n_team_games":int(len(frame)),
        "plays":metrics("sim_plays","actual_plays"),
        "dropbacks":metrics("sim_dropbacks","actual_dropbacks"),
        "pass_attempts":metrics("sim_pass_attempts","actual_pass_attempts"),
        "rushes":metrics("sim_rushes","actual_rushes"),
        "sacks":metrics("sim_sacks","actual_sacks"),
        "pregame_expected_plays":metrics("pregame_expected_plays","actual_plays"),
        "pregame_pass_attempts":metrics("pregame_pass_attempts","actual_pass_attempts"),
        "historical_qb_attempts":metrics("pregame_historical_qb_baseline","actual_pass_attempts"),
        "model_fraction_mean":round(float(frame["sim_drop_fraction"].mean()),4),
        "actual_fraction_mean":round(float(frame["actual_drop_fraction"].mean()),4),
        "model_sack_rate_mean":round(float(frame["sim_sack_rate"].mean()),4),
        "actual_sack_rate_mean":round(float(frame["actual_sack_rate"].mean()),4),
        "component_bias":{
            "plays":round(float(frame["plays_error_component"].mean()),3),
            "pass_run":round(float(frame["pass_run_error_component"].mean()),3),
            "sacks":round(float(frame["sacks_error_component"].mean()),3),
            "sum":round(float(frame[["plays_error_component","pass_run_error_component",
                                    "sacks_error_component"]].sum(axis=1).mean()),3),
        },
        "component_mean_abs":{
            "plays":round(float(frame["plays_error_component"].abs().mean()),3),
            "pass_run":round(float(frame["pass_run_error_component"].abs().mean()),3),
            "sacks":round(float(frame["sacks_error_component"].abs().mean()),3),
        },
    }
    if frame["sim_drives"].notna().all():
        out["possession_only"]={
            "sim_drives_mean":round(float(frame["sim_drives"].mean()),3),
            "eligible_observed_drives_proxy_mean":round(
                float(frame["eligible_observed_drives"].mean()),3),
            "sim_plays_per_drive_mean":round(float(frame["sim_plays_per_drive"].mean()),3),
            "observed_plays_per_eligible_drive_mean":round(
                float(frame["observed_plays_per_eligible_drive"].mean()),3),
            "warning":"eligible observed drives exclude some ambiguous/missing drives; not official count",
        }
    return out
