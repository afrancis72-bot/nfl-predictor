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


def _weighted_budget(initial, floor, priority, budget):
    """Conserve a hard team budget while minimizing changes to core roles.

    This deterministic bounded water-fill is generic for player minutes and
    scoring opportunity. No player names, betting prices or market lines.
    """
    initial=np.asarray(initial,dtype=float)
    floor=np.asarray(floor,dtype=float)
    priority=np.asarray(priority,dtype=float)
    if (initial.ndim!=1 or len(initial)==0 or not
            (np.isfinite(initial).all() and np.isfinite(floor).all() and
             np.isfinite(priority).all())):
        raise ValueError("Invalid NBA player budget assumptions")
    if ((initial<0).any() or (floor<0).any() or
            (priority<=0).any() or budget<0):
        raise ValueError("Invalid NBA player budget constraints")
    if float(initial.sum())<=budget:
        return initial.copy()
    floor=np.minimum(floor,initial)
    if floor.sum()>budget+1e-6:
        raise ValueError(
            "Too many full-minute players for the available NBA team budget; "
            "rotation needs independent availability review")
    lo=0.
    hi=float(np.max((initial-floor)*priority))+1.
    for _ in range(80):
        midpoint=(lo+hi)/2.
        proposal=np.maximum(
            floor,initial-midpoint/priority)
        if proposal.sum()>budget:
            lo=midpoint
        else:
            hi=midpoint
    planned=np.maximum(floor,initial-hi/priority)
    if not np.isclose(planned.sum(),budget,atol=.005):
        raise ValueError("Could not conserve NBA player opportunity budget")
    return planned


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
        # The previous proportional shrink reduced 35-minute starters to
        # ~22 whenever a 15-player season roster summed above 350.
        # Recent full-game rotation evidence must be selected upstream.
        # Here we protect established high-minute roles *generically*,
        # and remove overbooked opportunity primarily from reserves.
        floor=np.where(
            minutes>=30,minutes*.93,
            np.where(minutes>=24,minutes*.84,
                     np.where(minutes>=18,minutes*.55,0.)))
        priority=np.maximum(.15,(minutes/18.)**3+.15)
        reduced=_weighted_budget(minutes,floor,priority,240.)
        ratios=np.divide(
            reduced,minutes,out=np.zeros(len(minutes)),where=minutes>0)
        x.loc[indices,"minutes"]=reduced
        x.loc[indices,"minutes_sd"]=np.maximum(
            1.,np.minimum(
                rows["minutes_sd"].to_numpy(dtype=float)*ratios,
                np.maximum(1.8,reduced*0.30)))
        for stat in STATS:
            x.loc[indices,stat]=rows[stat].to_numpy(dtype=float)*ratios
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
            "rotation_method":"recent-participants/core-role-protected",
            "core_players_30plus":int((minutes>=30).sum()),
            "core_minutes_prior":round(
                float(minutes[minutes>=30].sum()),2),
            "core_minutes_after":round(
                float(reduced[minutes>=30].sum()),2),
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


def calibrate_team_scoring_baselines(projections:pd.DataFrame,
                                     logs:pd.DataFrame,when:date):
    """Adjust ONLY point scoring baselines to plausible complete-team totals.

    The current minute-capped player's independent historic points-per-minute
    rate is just an opportunity WEIGHT, not a free-standing team score.
    Deterministically reconcile every active player's expected PTS to the
    team's recent full-score environment before random simulations; this
    keeps prop baseline outputs aligned with the actual scoring model.
    No bookmaker total is read or used.
    """
    games=observed_complete_team_games(logs,when)
    x=projections.copy(deep=True)
    if "historical_unconstrained_points" not in x:
        x["historical_unconstrained_points"]=x["pts"].astype(float)
    audit=[]
    for team,rows in x.groupby("team",sort=True):
        opps=set(rows["opponent"].astype(str))
        if len(opps)!=1:
            raise ValueError(f"{team}: multiple opponent score environments")
        opponent=next(iter(opps))
        offense=games.loc[games["team"]==team].tail(18)
        defense=games.loc[games["opponent"]==opponent].tail(18)
        if len(offense)<5 or len(defense)<5:
            raise ValueError(
                f"{team} vs {opponent}: insufficient complete historical "
                "team game scores for a baseline")
        expected=.70*float(offense["pts"].mean())+.30*float(defense["pts"].mean())
        modeled=float(rows["pts"].sum())
        residual=max(0.,240.-float(rows["minutes"].sum()))
        bench=(residual/240.)*expected
        if not np.isfinite([modeled,bench,expected]).all() or modeled<=0:
            raise ValueError(f"{team}: no plausible active team scoring shares")
        baseline=rows["pts"].to_numpy(dtype=float)
        minutes=rows["minutes"].to_numpy(dtype=float)
        score_budget=expected-bench
        if score_budget<0:
            raise ValueError(f"{team}: invalid team-score projection")
        contested_role_baseline=False
        if modeled>score_budget+1e-8:
            # A uniform second haircut to every scorer would force
            # 25-point players down just because reserves were overcounted.
            # Keep the primary scorers near empirically observed usage;
            # use the lower-priority reserve opportunity to reconcile
            # team points. No individual sportsbook lines are referenced.
            floors=np.where(
                (minutes>=28)&(baseline>=12),baseline*.88,
                np.where((minutes>=22)&(baseline>=8),
                         baseline*.67,0.))
            priority=.20+(minutes/23.)**3*(1+baseline/18.)
            if floors.sum()>score_budget+1e-7:
                # This roster claims more independently protected scoring
                # than the TEAM can actually score. It is still physically
                # possible to reconcile points, but the conflicting role
                # assumptions must be flagged rather than silently failing
                # every game or claiming all players reach their floors.
                contested_role_baseline=True
                floors*=score_budget/max(1e-8,float(floors.sum()))*.99
            adjusted=_weighted_budget(
                baseline,floors,priority,score_budget)
        else:
            # Upside after a teammate is removed is distributed by observed
            # remaining-player scoring opportunity, never invented punts.
            weights=np.maximum(.01,baseline)
            adjusted=baseline+(score_budget-modeled)*weights/weights.sum()
        x.loc[rows.index,"pts"]=adjusted
        for market,parts in MARKETS.items():
            x.loc[rows.index,market]=x.loc[rows.index,list(parts)].sum(axis=1)
        audit.append({
            "team":team,"opponent":opponent,
            "expected_team_points_from_complete_history":round(expected,2),
            "unconstrained_player_baseline_points":round(modeled,2),
            "constrained_player_baseline_points":round(
                float(x.loc[rows.index,"pts"].sum()),2),
            "reserve_bench_points":round(bench,2),
            "points_scaling_factor":round(
                float(score_budget/max(modeled,1e-8)),4),
            "scoring_method":"core-usage-protected/team-conserved",
            "scoring_role_conflict":contested_role_baseline,
            "uses_sportsbook_total":False,
        })
    return x.reset_index(drop=True),pd.DataFrame(audit)


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
