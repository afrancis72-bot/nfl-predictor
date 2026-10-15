"""Research-only snap-level NFL Showdown simulator.

Not calibrated to historical play-by-play. Never treat generated prop probabilities
as sportsbook-ready estimates. Designed for reproducible engineering tests.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST", "D"}

def simulate_snaps(players: pd.DataFrame, teams, n_games=1000, seed=316, defense_profiles=None):
    """Return (DK points [games, players], individual per-game stats).

    Stats are long-form: game, player_index, team, name, position, pass_att,
    pass_cmp, pass_yd, pass_td, interceptions, rush_att, rush_yd, rush_td,
    targets, receptions, rec_yd, rec_td, fg_made, xp_made, sacks, turnovers.
    Plays have explicit run/pass decisions, assigned rushers/targets, and outcomes.
    This prototype lacks penalties, substitutions, clock precision and drive state.
    """
    if len(teams) != 2:
        raise ValueError("Exactly two teams required")
    if not 1 <= int(n_games) <= 100000:
        raise ValueError("n_games must be 1..100000")
    p = players.reset_index(drop=True).copy()
    if p.empty or not {"Name", "TeamAbbrev", "Position"}.issubset(p.columns):
        raise ValueError("Player columns missing")
    p["Position"] = p["Position"].astype(str).str.upper()
    p["TeamAbbrev"] = p["TeamAbbrev"].astype(str)
    if not set(p["TeamAbbrev"]).issubset(set(teams)):
        raise ValueError("Players outside matchup")
    if "FLEX Salary" not in p:
        raise ValueError("FLEX Salary required")
    salary = pd.to_numeric(p["FLEX Salary"], errors="coerce").fillna(0).to_numpy()
    if (salary < 0).any():
        raise ValueError("Invalid salary")
    use_history = {"rush_weight", "target_weight", "usage_verified"}.issubset(p.columns)
    if use_history:
        if not p["usage_verified"].fillna(False).all():
            raise ValueError("Unverified player usage: resolve identities first")
        for col in ("rush_weight", "target_weight"):
            p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0).clip(lower=0)
    pos = p["Position"].to_numpy()
    team = p["TeamAbbrev"].to_numpy()
    names = p["Name"].astype(str).to_numpy()
    n = len(p)
    rng = np.random.default_rng(seed)
    fields = ["pass_att","pass_cmp","pass_yd","pass_td","interceptions",
              "rush_att","rush_yd","rush_td","targets","receptions","rec_yd",
              "rec_td","fg_made","xp_made","sacks","turnovers"]
    stats = np.zeros((int(n_games),n,len(fields)), dtype=np.float32)
    f = {name:i for i,name in enumerate(fields)}
    team_ids = {t:np.where(team==t)[0] for t in teams}
    qbs = {}
    for t, ids in team_ids.items():
        qs = ids[pos[ids]=="QB"]
        if len(qs)!=1:
            raise ValueError(f"{t}: require exactly one confirmed starting QB")
        qbs[t]=int(qs[0])
    # Pregame QB history drives distinct team passing volume and efficiency.
    qb_rates={}
    for t,qb in qbs.items():
        def rate(column, fallback):
            if column not in p: return fallback
            v=pd.to_numeric(pd.Series([p.iloc[qb][column]]),errors="coerce").iloc[0]
            return float(v) if np.isfinite(v) and v>0 else fallback
        attempts=rate("attempts_per_game",35.)
        completion=np.clip(rate("qb_completion_rate",.62),.40,.82)
        ypa=np.clip(rate("qb_yards_per_attempt",7.),4.,11.)
        # Approximately 60 offensive plays per team; game state still modifies mix.
        qb_rates[t]={"base_pass_prob":float(np.clip(attempts/60.,.36,.76)),
                     "completion":float(completion),"yards_per_completion":float(ypa/completion)}
    def pick(ids, power=1.0, usage_col=None):
        if use_history and usage_col:
            w=p[usage_col].to_numpy(dtype=float)[ids]
            if w.sum()<=0:
                raise ValueError("No verified "+usage_col+" opportunities")
        else:
            w=np.maximum(salary[ids],100.)**power
        return int(rng.choice(ids,p=w/w.sum()))
    # Explicit, optional opponent profiles. Neutral defaults are disclosed, not
    # inferred from sportsbook prices. Values should be measured from pregame data.
    defense_profiles = defense_profiles or {}
    defense = {}
    for t in teams:
        d = defense_profiles.get(t, {})
        defense[t] = {
            "pass_yards_factor": float(np.clip(d.get("pass_yards_factor", 1.), .65, 1.35)),
            "run_yards_factor": float(np.clip(d.get("run_yards_factor", 1.), .65, 1.35)),
            "completion_delta": float(np.clip(d.get("completion_delta", 0.), -.15, .15)),
        }
    for g in range(int(n_games)):
        # One shared game-level environment induces correlated player outcomes.
        # Separate team-level form lets good defenses occasionally have bad games.
        pace = float(np.clip(rng.lognormal(0, .09), .78, 1.28))
        form = {t: float(np.clip(rng.normal(1., .10), .72, 1.30)) for t in teams}
        defense_form = {t: float(np.clip(rng.normal(1., .10), .72, 1.30)) for t in teams}
        score={t:0 for t in teams}
        offense=teams[g%2]
        elapsed=0
        # Ball possession alternates at drive boundaries. Drives end on punts,
        # turnovers, scores or fourth-down failures; do not grant 14 free plays.
        possession_count=0
        while elapsed < 3600 and possession_count < 45:
            possession_count += 1
            opponent=teams[1] if offense==teams[0] else teams[0]
            ids=team_ids[offense]
            qb=qbs[offense]
            rushers=ids[np.isin(pos[ids],["RB","QB","WR"])]
            receivers=ids[np.isin(pos[ids],["WR","TE","RB"])]
            kickers=ids[pos[ids]=="K"]
            if len(rushers)==0 or len(receivers)==0:
                raise ValueError(f"{offense}: missing rushers or receivers")
            # Starting field position measured in yards from own goal.
            field=float(np.clip(rng.normal(29,9), 8, 50))
            down=1; to_go=10.
            for _ in range(24):
                if elapsed>=3600: break
                margin=score[offense]-score[opponent]
                remaining=3600-elapsed
                trailing=margin<0
                # Playcalling changes with down, distance, margin and late clock.
                urgency=(remaining<600 and margin<0)
                pass_prob=np.clip(qb_rates[offense]["base_pass_prob"]
                    +(.11 if down==3 else 0)+(.08 if to_go>=9 else -.07 if to_go<=2 else 0)
                    +(.14 if urgency else .05 if trailing else -.06 if margin>=9 else 0),.22,.90)
                # Fourth down is a decision, not an automatic offensive snap.
                if down==4:
                    if field>=64 and len(kickers) and remaining>80:
                        if rng.random()<.78:
                            stats[g,pick(kickers),f["fg_made"]]+=1
                            score[offense]+=3
                        break
                    go=(to_go<=2 and field>=45) or (remaining<240 and margin<0)
                    if not go:
                        break
                is_pass=rng.random()<pass_prob
                clock_base=rng.integers(23,43) if not urgency else rng.integers(9,25)
                elapsed+=max(5,int(clock_base/pace))
                if is_pass:
                    stats[g,qb,f["pass_att"]]+=1
                    target=pick(receivers,.9,"target_weight")
                    stats[g,target,f["targets"]]+=1
                    completion = np.clip(qb_rates[offense]["completion"]
                                         + defense[opponent]["completion_delta"]
                                         + .09*(form[offense]-defense_form[opponent]), .30, .90)
                    if rng.random()<completion:
                        avg=qb_rates[offense]["yards_per_completion"]
                        yards=max(-2.,float(rng.normal(avg*defense[opponent]["pass_yards_factor"]*form[offense]/defense_form[opponent],max(4.,avg*.65))))
                        stats[g,qb,f["pass_cmp"]]+=1
                        stats[g,qb,f["pass_yd"]]+=yards
                        stats[g,target,f["receptions"]]+=1
                        stats[g,target,f["rec_yd"]]+=yards
                    elif rng.random()<.055:
                        stats[g,qb,f["interceptions"]]+=1
                        # Defensive turnover belongs to a team DST, not an arbitrary offensive player.
                        # DST scoring is intentionally not modeled in this research simulator.
                        break
                    else: yards=0.
                else:
                    runner=pick(rushers,.65,"rush_weight")
                    yards=float(np.clip(rng.normal(4.4*defense[opponent]["run_yards_factor"]*form[offense]/defense_form[opponent],5.0),-5,35))
                    stats[g,runner,f["rush_att"]]+=1
                    stats[g,runner,f["rush_yd"]]+=yards
                # A completed play cannot gain more than the distance to goal.
                if field+yards>=100:
                    excess=field+yards-100
                    if is_pass and yards>0:
                        stats[g,qb,f["pass_yd"]]-=excess
                        stats[g,target,f["rec_yd"]]-=excess
                    elif not is_pass and yards>0:
                        stats[g,runner,f["rush_yd"]]-=excess
                    yards-=excess
                field+=yards
                if field>=100:
                    score[offense]+=7
                    if is_pass:
                        stats[g,qb,f["pass_td"]]+=1
                        stats[g,target,f["rec_td"]]+=1
                    else:
                        stats[g,runner,f["rush_td"]]+=1
                    if len(kickers):
                        stats[g,pick(kickers),f["xp_made"]]+=1
                    break
                to_go-=yards
                if to_go<=0:
                    down=1;to_go=10.
                else:
                    down+=1
                if down>4:
                    break
            offense=opponent
    pts=np.zeros((int(n_games),n),dtype=np.float32)
    s={key:stats[:,:,i] for key,i in f.items()}
    pts+=.04*s["pass_yd"]+4*s["pass_td"]-s["interceptions"]
    pts+=.1*(s["rush_yd"]+s["rec_yd"])+6*(s["rush_td"]+s["rec_td"])
    pts+=s["receptions"]+3*s["fg_made"]+s["xp_made"]
    pts+=3*(s["pass_yd"]>=300)+3*(s["rush_yd"]>=100)+3*(s["rec_yd"]>=100)
    # No defensive points awarded to offensive players; DST scoring is out of scope.
    long=pd.DataFrame(stats.reshape(-1,len(fields)),columns=fields)
    long.insert(0,"game",np.repeat(np.arange(int(n_games)),n))
    long.insert(1,"player_index",np.tile(np.arange(n),int(n_games)))
    long.insert(2,"name",np.tile(names,int(n_games)))
    long.insert(3,"team",np.tile(team,int(n_games)))
    long.insert(4,"position",np.tile(pos,int(n_games)))
    long["dk_points"]=pts.ravel()
    return pts,long

def summarize_player_props(stats: pd.DataFrame):
    """Per-player stat means and quantiles; not calibrated probabilities."""
    cols=["pass_yd","rush_yd","rec_yd","receptions","targets","rush_att","pass_att","fg_made","dk_points"]
    out=[]
    for (name,team),group in stats.groupby(["name","team"],sort=False):
        row={"Name":name,"Team":team}
        for c in cols:
            row[c+"_mean"]=round(float(group[c].mean()),2)
            row[c+"_median"]=round(float(group[c].median()),2)
            row[c+"_p90"]=round(float(group[c].quantile(.9)),2)
        out.append(row)
    return pd.DataFrame(out)
