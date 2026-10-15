"""Research-only snap-level NFL Showdown simulator.

Not calibrated to historical play-by-play. Never treat generated prop probabilities
as sportsbook-ready estimates. Designed for reproducible engineering tests.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST", "D"}

def simulate_snaps(players: pd.DataFrame, teams, n_games=1000, seed=316):
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
    def pick(ids, power=1.0, usage_col=None):
        if use_history and usage_col:
            w=p[usage_col].to_numpy(dtype=float)[ids]
            if w.sum()<=0:
                raise ValueError("No verified "+usage_col+" opportunities")
        else:
            w=np.maximum(salary[ids],100.)**power
        return int(rng.choice(ids,p=w/w.sum()))
    for g in range(int(n_games)):
        score={t:0 for t in teams}
        offense=teams[g%2]
        elapsed=0
        while elapsed < 3600:
            defense=teams[1] if offense==teams[0] else teams[0]
            ids=team_ids[offense]
            qb=qbs[offense]
            rushers=ids[np.isin(pos[ids],["RB","QB","WR"])]
            receivers=ids[np.isin(pos[ids],["WR","TE","RB"])]
            kickers=ids[pos[ids]=="K"]
            if len(rushers)==0 or len(receivers)==0:
                raise ValueError(f"{offense}: missing rushers or receivers")
            # Starting field position measured in yards from own goal.
            field=25.; down=1; to_go=10.
            for _ in range(14):
                if elapsed>=3600: break
                trailing=score[offense]<score[defense]
                pass_prob=np.clip(.56+(.10 if trailing else -.04)+(.08 if down>=3 else 0),.35,.83)
                is_pass=rng.random()<pass_prob
                elapsed+=int(rng.integers(19,42))
                if is_pass:
                    stats[g,qb,f["pass_att"]]+=1
                    target=pick(receivers,.9,"target_weight")
                    stats[g,target,f["targets"]]+=1
                    if rng.random()<.61:
                        yards=max(-2.,float(rng.normal(11,9)))
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
                    yards=float(np.clip(rng.normal(4.4,5.0),-5,35))
                    stats[g,runner,f["rush_att"]]+=1
                    stats[g,runner,f["rush_yd"]]+=yards
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
                    if field>=62 and len(kickers) and rng.random()<.72:
                        kicker=pick(kickers)
                        if rng.random()<.79:
                            stats[g,kicker,f["fg_made"]]+=1
                            score[offense]+=3
                    break
            offense=defense
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
