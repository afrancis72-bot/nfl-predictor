"""2026 NBA pregame team-constrained player-stat scenario bridge.

This corrects a major flaw in pure player-by-player bootstrap: all player
point draws in a matchup must come from the SAME plausible team game score,
and projected minutes may not exceed 240 regulation minutes for either team.
Team scores are sampled from ACTUAL historical full-team box-score totals;
market spreads/totals are NOT used to construct projections.

This is an empirical game/team scoring conservation model, not a
play-by-play possession engine, a verified rotation chart, or a guaranteed
profitable prop model. Incomplete player/team history fails closed.
"""
from __future__ import annotations

from datetime import date
import numpy as np
import pandas as pd

from nba_engine.predictor import MARKETS, STATS, simulate_stat_lines


def normalize_team_minutes(projections: pd.DataFrame, *, coverage_floor=195.0):
    """Make active, non-OUT player minutes fit one 240-minute team budget.

    Unmodeled small-minute bench players retain an explicit residual budget.
    We never fabricate a new player, and don't boost players simply because
    someone else is OUT unless their relative available-player share supports it.
    """
    required={"player_id","player","team","opponent","minutes","minutes_sd"}
    if not required.issubset(projections):
        raise ValueError("Incomplete NBA player rotation projections")
    if projections.empty or projections["player_id"].astype(str).duplicated().any():
        raise ValueError("No unique NBA players available for full-game simulation")
    x=projections.copy(deep=True)
    if not set(STATS).issubset(x):
        raise ValueError("Missing component NBA player-stat projections")
    audits=[]
    for team,rows in x.groupby("team",sort=True):
        indices=rows.index
        minutes=rows["minutes"].to_numpy(dtype=float)
        if len(rows)<7:
            raise ValueError(
                f"{team}: fewer than seven projected players; incomplete active rotation")
        if (not np.isfinite(minutes).all() or (minutes<0).any()
                or (minutes>48).any()):
            raise ValueError(f"{team}: invalid expected player minutes")
        before=float(minutes.sum())
        if before<coverage_floor:
            raise ValueError(
                f"{team}: only {before:.1f} documented rotation minutes. "
                "Insufficient roster coverage; do not simulate a full game.")
        scale=min(1.0,240.0/before)
        reduced=minutes*scale
        x.loc[indices,"minutes"]=reduced
        x.loc[indices,"minutes_sd"]=np.maximum(
            1.0,np.minimum(rows["minutes_sd"].to_numpy(dtype=float),
                           np.maximum(1.8,reduced*0.30)))
        for stat in STATS:
            x.loc[indices,stat]=rows[stat].to_numpy(dtype=float)*scale
        for market,parts in MARKETS.items():
            x.loc[indices,market]=x.loc[indices,list(parts)].sum(axis=1)
        after=float(reduced.sum())
        audits.append({
            "team":str(team),
            "players_modelled":len(rows),
            "projected_minutes_before":round(before,2),
            "projected_minutes_after":round(after,2),
            "unmodeled_bench_minutes":round(max(0.,240.-after),2),
            "minute_conservation_applied":bool(before>240.000001),
        })
        if after>240.00001:
            raise ValueError(f"{team}: impossible projected team minute budget")
    return x.reset_index(drop=True),pd.DataFrame(audits)


def observed_complete_team_games(logs: pd.DataFrame, when: date):
    """Only historical team-game scores with sufficient logged participants."""
    required={"date","team","opponent","player_id","minutes","pts"}
    if not required.issubset(logs):
        raise ValueError("Cannot verify historical complete NBA team games")
    x=logs.loc[logs["date"]<pd.Timestamp(when)].copy()
    if x.empty:
        raise ValueError("No pregame NBA team-boxscore history")
    g=x.groupby(["date","team","opponent"],as_index=False,observed=True).agg(
        pts=("pts","sum"), minutes=("minutes","sum"), players=("player_id","nunique"))
    # Low-minute/partial-source box scores cannot establish a team result.
    good=g.loc[
        (g["minutes"]>=195)&(g["minutes"]<=280)&
        (g["players"]>=7)&(g["pts"]>=65)&(g["pts"]<=175)
    ].sort_values("date")
    if good.empty:
        raise ValueError("No complete historical team scores for NBA scenarios")
    return good.reset_index(drop=True)


def _sample_team_scores(games,team,opponent,n,shared,rng):
    """Sample whole team points from offense AND the opponent's allowed totals."""
    recent=games.loc[games["team"]==team].tail(18)
    conceded=games.loc[games["opponent"]==opponent].tail(18)
    if len(recent)<5 or len(conceded)<5:
        raise ValueError(
            f"{team} vs {opponent}: insufficient verified complete team-game "
            "history to forecast a coherent game score")
    o=recent["pts"].to_numpy(dtype=float)
    d=conceded["pts"].to_numpy(dtype=float)
    # A prior full TEAM game is drawn (not the sum of individual forecasts).
    off=rng.choice(o,size=n,replace=True)
    defense=rng.choice(d,size=n,replace=True)
    expected=.70*float(o.mean())+.30*float(d.mean())
    # Shared matchup-level pace links opponents' game-score scenarios.
    total=np.rint(np.clip(
        (.70*off+.30*defense)*shared + rng.normal(0,2.2,n),65,180)
    ).astype(np.int32)
    return total,float(expected),len(o),len(d)


def _allocate_integer_points(total,weight,rng):
    """Exact, nonnegative scenario-by-scenario allocation to modeled players+OTHER."""
    if (weight.ndim!=2 or weight.shape[1]!=len(total)
            or not np.isfinite(weight).all() or (weight<0).any()):
        raise ValueError("Invalid NBA player share matrix")
    if np.any(weight.sum(axis=0)<=0):
        raise ValueError("No available NBA player scoring opportunity")
    fractional=weight/weight.sum(axis=0)[None,:]*total[None,:]
    floor=np.floor(fractional).astype(np.int32)
    remainder=total-floor.sum(axis=0)
    # Fair randomized largest-remainder tie-breaking, still exact conservation.
    rank=np.argsort(
        -(fractional-floor)+rng.uniform(0,1e-7,floor.shape),axis=0)
    cols=np.arange(len(total))
    for k in range(floor.shape[0]):
        which=remainder>k
        if which.any():
            floor[rank[k,which],cols[which]]+=1
    if np.any(floor<0) or not np.array_equal(floor.sum(axis=0),total):
        raise ValueError("NBA simulated player scoring did not conserve team points")
    return floor


def simulate_conserved_nba_games(projections: pd.DataFrame,logs: pd.DataFrame,
                                 *,when: date,trials=10000,seed=42):
    """Return (player-market draws, matchup/team scoring audit).

    Keep observed player within-game covariance for non-point stats, but couple
    all modeled player scoring to exactly one sampled TEAM score each trial.
    A residual OTHER player bucket absorbs small-minute players not in source.
    """
    if not 1000<=int(trials)<=50000:
        raise ValueError("NBA game scenarios must be 1,000-50,000")
    if projections.empty:
        raise ValueError("Cannot simulate without reviewed NBA rotations")
    if (projections["minutes"].astype(float)<0).any():
        raise ValueError("Negative NBA player minutes")
    if any(float(g["minutes"].sum())>240.0001
           for _,g in projections.groupby("team")):
        raise ValueError(
            "NBA projected team minutes exceed 240; normalize rotation first")
    # Actual observed history must have current-game totals, not player means.
    historical=observed_complete_team_games(logs,when)
    pairs={str(t):str(o) for t,o in zip(
        projections["team"],projections["opponent"])}
    for team,opp in pairs.items():
        if pairs.get(opp)!=team:
            raise ValueError(f"NBA opponent mapping incomplete: {team} vs {opp}")
    samples=simulate_stat_lines(projections,logs,trials=trials,seed=seed)
    rng=np.random.default_rng(int(seed)+7301)
    games=sorted({tuple(sorted([team,opp])) for team,opp in pairs.items()})
    records=[]
    for team_a,team_b in games:
        # One shared environment factor makes both teams' point totals
        # respond to the same pace/noise rather than unrelated solo draws.
        shared=np.clip(rng.normal(1.0,0.027,int(trials)),.90,1.10)
        for team,opp in ((team_a,team_b),(team_b,team_a)):
            team_rows=projections.loc[projections["team"]==team]
            expected_players=team_rows["player_id"].astype(str).tolist()
            if len(expected_players)<7:
                raise ValueError(f"{team}: incomplete simulated player rotation")
            target,mean,n_off,n_def=_sample_team_scores(
                historical,team,opp,int(trials),shared,rng)
            old=np.vstack([
                np.asarray(samples[(pid,"Points")],dtype=float)
                for pid in expected_players])
            extra_minutes=max(0.,240.-float(team_rows["minutes"].sum()))
            # Give missing bench players their own scoring share instead of
            # accidentally allocating the entire score to 7-8 main players.
            other_weight=np.full(int(trials),extra_minutes*(mean/240.))
            weights=np.vstack([old+.08,other_weight])
            assigned=_allocate_integer_points(target,weights,rng)
            player_total=np.zeros(int(trials),dtype=np.int32)
            for i,pid in enumerate(expected_players):
                points=assigned[i].astype(np.int16)
                samples[(pid,"Points")]=points
                player_total+=points.astype(np.int32)
                for combo in ("PRA","PR","PA"):
                    parts=MARKETS[combo]
                    samples[(pid,combo)]=sum(
                        (np.asarray(samples[(pid,next(
                            market for market,stats in MARKETS.items()
                            if stats==(part,)))],dtype=np.int32)
                         for part in parts))
            other=assigned[-1].astype(np.int32)
            if not np.array_equal(player_total+other,target):
                raise ValueError(f"{team}: player points exceed team total")
            records.append({
                "game":f"{team_a} vs {team_b}",
                "team":team,"opponent":opp,
                "mean_team_points":round(float(target.mean()),2),
                "p10_team_points":round(float(np.quantile(target,.10)),2),
                "p90_team_points":round(float(np.quantile(target,.90)),2),
                "mean_modeled_player_points":round(float(player_total.mean()),2),
                "mean_unmodeled_bench_points":round(float(other.mean()),2),
                "historical_team_games":n_off,
                "historical_opp_allowed_games":n_def,
                "mean_projected_minutes":round(
                    float(team_rows["minutes"].sum()),2),
                "mean_unmodeled_bench_minutes":round(extra_minutes,2),
                "team_score_conservation_passed":True,
                "market_total_used_to_fit":False,
            })
    audit=pd.DataFrame(records)
    return samples,audit
