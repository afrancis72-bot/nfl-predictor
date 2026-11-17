"""Experimental NFL *sequential possession* worlds from prior nflverse drives.

This is an initial drive-level game model, not a validated release. It samples
real observed drive lengths and drive-result categories, with pregame offense
and defense probabilities shrunk to a measured league prior.  The two clubs
SHARE one game clock, score differential, and alternating possessions. Passing
propensity responds to actual simulated trailing/leading situations.

No player fantasy points are generated here; the stat-driven simulator's
existing target/carry/QB and team/DST allocation receives the resulting team
plays, dropbacks, pass/rush TDs, made FGs and interceptions.  The live one-click
path does not enable possession worlds.
"""
from __future__ import annotations

import math
import numpy as np

from ai2_nfl_matchups import canonical_team
from ai2_nfl_pbp_drives import OUTCOMES, check_profile_freshness


def _probabilities(base, offense, defense, strength, field_factor,
                   score_diff, seconds_remaining):
    """Log-pool both teams' observed rate evidence; drive outcome sums to one."""
    p = np.asarray(base,float)
    off = np.asarray(offense,float)
    deff = np.asarray(defense,float)
    if not all(np.isfinite(x).all() and x.shape==(5,) for x in (p,off,deff)):
        raise ValueError("Missing measured drive outcome distribution")
    if (p<=0).any() or (off<=0).any() or (deff<=0).any():
        raise ValueError("Drive smoothing must supply strictly positive probabilities")
    logits=np.log(p)
    logits+=float(strength)*(
        .60*np.log(off/p)+.60*np.log(deff/p))
    # Field-position adjustment estimated from previous scored drives by
    # starting-yardline bucket, not from the target game's result.
    logits[:3]+=math.log(float(np.clip(field_factor,.5,2.)))
    if seconds_remaining<900 and score_diff<=-7:
        # Trailing teams become more TD aggressive; fewer short FGs.
        logits[:2]+=.12
        logits[2]-=.18
        logits[3]+=.08
    if seconds_remaining<360 and score_diff>=7:
        # Fourth-quarter leading teams are less inclined to force scoring.
        logits[:3]-=.12
    # Softmax is normalized and cannot invent additional terminal outcomes.
    x=np.exp(logits-logits.max())
    return x/x.sum()


def _field_index(yards_to_endzone, bins):
    for i,(lo,hi) in enumerate(bins):
        if float(lo)<=float(yards_to_endzone)<=float(hi):
            return i
    raise ValueError("Invalid sampled field position")


def simulate_possession_worlds(game, profiles, *, season, week,
                                n_sims=400, seed=42,
                                strength=1., offense_pass_fraction=None):
    """Simulate real, scored/clocked football possessions for both teams.

    Returns {TEAM: {plays, dropbacks, pass_tds, rush_tds, field_goals,
    interceptions, fumbles_lost, drives, outcome_counts}} with arrays of
    length n_sims. Team totals are sum of the same explicit sequential drives.
    """
    if not np.isfinite(strength) or not 0<=strength<=1:
        raise ValueError("Possession-model strength must be in [0,1]")
    if strength<=0:
        raise ValueError("Zero possession strength must use untouched existing simulator")
    check_profile_freshness(profiles,season,week)
    teams=[canonical_team(s) for s in str(game).split()[0].split("@")]
    if len(teams)!=2 or any(not t for t in teams) or teams[0]==teams[1]:
        raise ValueError("Possession model needs complete away@home matchup")
    if not isinstance(n_sims,int) or n_sims<1 or n_sims>100000:
        raise ValueError("Invalid possession simulation count")
    sample=profiles["outcomes"]
    start=np.asarray(profiles["starting_field_positions"],dtype=float)
    if len(start)<100 or not np.isfinite(start).all():
        raise ValueError("No measured prior-game starting field-position distribution")
    league=np.asarray(profiles["league_probs"],float)
    pace_mean=float(profiles["overall_mean_drives_per_team"])
    if not 5<=pace_mean<=17:
        raise ValueError("Unrealistic historic drives per team")
    rng=np.random.default_rng(seed)
    output={}
    keys=("plays","dropbacks","pass_tds","rush_tds","field_goals",
          "interceptions","fumbles_lost","drives")
    for t in teams:
        output[t]={k:np.zeros(n_sims,dtype=np.int16) for k in keys}
        output[t]["outcomes"]={x:np.zeros(n_sims,dtype=np.int16) for x in OUTCOMES}
        output[t]["points"]=np.zeros(n_sims,dtype=np.int16)
        output[t]["starting_field_100_mean"]=np.zeros(n_sims,dtype=np.float32)
    pass_fracs = offense_pass_fraction or {}
    params={}
    for t in teams:
        profile=profiles["teams"].get(t,{})
        params[t]={
            "off":np.asarray(profile.get("offense",league),float),
            "allowed":np.asarray(profile.get("allowed",league),float),
            "pass_frac":float(np.clip(pass_fracs.get(t,.57),.30,.79)),
        }
    bins=profiles["field_bins"]
    factors=profiles["field_scoring_multiplier"]
    if len(factors)!=len(bins) or any(not np.isfinite(x) for x in factors):
        raise ValueError("Invalid pregame field-position evidence")
    for scenario in range(n_sims):
        sec=3600
        current=teams[int(rng.integers(0,2))]
        score={teams[0]:0,teams[1]:0}
        drives_count=0
        next_start=None
        while sec>20 and drives_count<36:
            other=teams[1] if current==teams[0] else teams[0]
            if next_start is None:
                field=float(rng.choice(start))
            else:
                field=next_start
            field=float(np.clip(field,1,99))
            next_start=None
            score_diff=score[current]-score[other]
            factor=factors[_field_index(field,bins)]
            probs=_probabilities(
                league,params[current]["off"],params[other]["allowed"],
                strength,factor,score_diff,sec)
            result=OUTCOMES[int(rng.choice(len(OUTCOMES),p=probs))]
            evidence=sample.get(result)
            if not isinstance(evidence,dict) or not evidence.get("plays"):
                raise ValueError("No prior possession lengths for drive outcome")
            j=int(rng.integers(0,len(evidence["plays"])))
            play_count=max(1,min(18,int(evidence["plays"][j])))
            duration=int(evidence["seconds"][j])
            # A single game clock constrains BOTH teams. Count offensive plays,
            # not kickoffs/punts/FG attempts as phantom scrimmage snaps.
            duration=max(35,min(500,duration))
            time_left=min(sec,duration)
            if time_left<duration:
                # End-of-half/game limits snap opportunities but does not
                # make the two offenses' possessions overlap.
                play_count=max(1,min(play_count,int(time_left//15)+1))
                if time_left<55:
                    result="other"  # No invented last-second scoring.
            frac=params[current]["pass_frac"]
            if score_diff<=-7:
                frac+=.055
            elif score_diff>=7:
                frac-=.04
            if sec<900 and score_diff<=-7:
                frac+=.08
            frac=float(np.clip(frac,.25,.89))
            drops=int(rng.binomial(play_count,frac))
            if result=="pass_td":
                drops=max(1,drops)
            elif result=="rush_td":
                drops=min(drops,play_count-1)
            # A pass/dropback event and a rush are mutually exclusive plays.
            if drops>play_count or drops<0:
                raise ValueError("Possession generated invalid scrimmage plays")
            team=output[current]
            team["plays"][scenario]+=play_count
            team["dropbacks"][scenario]+=drops
            team["drives"][scenario]+=1
            team["outcomes"][result][scenario]+=1
            team["starting_field_100_mean"][scenario]+=field
            if result=="pass_td":
                team["pass_tds"][scenario]+=1
                score[current]+=7
            elif result=="rush_td":
                team["rush_tds"][scenario]+=1
                score[current]+=7
            elif result=="field_goal":
                team["field_goals"][scenario]+=1
                score[current]+=3
            elif result=="turnover":
                # The fraction of interceptions versus lost fumbles is still
                # a research prior, not a fully reconstructed play sequence.
                if rng.random()<.72:
                    team["interceptions"][scenario]+=1
                else:
                    team["fumbles_lost"][scenario]+=1
                # Turnover can plausibly give opponent shorter field. This
                # initial field-position change is an explicit heuristic and
                # must be calibrated in a future drive-transition study.
                next_start=float(np.clip(
                    field-rng.integers(12,33),15,85))
            # Add a brief between-drive/possession transition, same clock.
            sec-=time_left+min(sec-time_left,int(rng.integers(6,18)))
            drives_count+=1
            current=other
        for t in teams:
            observed=output[t]["drives"][scenario]
            if observed:
                output[t]["starting_field_100_mean"][scenario]/=observed
            output[t]["points"][scenario]=score[t]
    for t in teams:
        o=output[t]
        if not np.array_equal(
                o["pass_tds"]+o["rush_tds"]+o["field_goals"]+
                o["interceptions"]+o["fumbles_lost"]+
                o["outcomes"]["other"],
                o["drives"]):
            raise ValueError("Possession event-count conservation failed")
        if (o["dropbacks"]>o["plays"]).any() or (o["plays"]<0).any():
            raise ValueError("Possession pass/rush conservation failed")
        if not np.array_equal(
                o["points"],7*(o["pass_tds"]+o["rush_tds"]) +
                3*o["field_goals"]):
            raise ValueError("Possession touchdown/field goal scoring mismatch")
    return output
