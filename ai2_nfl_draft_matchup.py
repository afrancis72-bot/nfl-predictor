"""Draft-only opponent role adjustment; preserve the neutral scoring benchmark."""
import numpy as np

def review_strength(strength, draft):
    if isinstance(strength,bool) or not isinstance(strength,(int,float)) or strength not in (0.,.25):
        raise ValueError("Choose neutral 0 or experimental .25")
    if strength and draft is not True:
        raise ValueError("Role adjustments are allowed only for manual-review drafts")
    return float(strength)

def paired_projection_rows(players, draws, neutral_players, neutral_draws, *, seed, season, week):
    a=players.ID.astype(str).tolist()
    b=neutral_players.ID.astype(str).tolist()
    if len(a)!=len(set(a)) or len(b)!=len(set(b)) or set(a)!=set(b):
        raise ValueError("Paired forecast player IDs differ or are duplicated")
    x=np.asarray(draws,float); y=np.asarray(neutral_draws,float)
    if x.shape!=(10000,len(a)) or y.shape!=(10000,len(b)) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Incomplete paired forecast")
    idx={ident:i for i,ident in enumerate(b)}
    rows=[]
    for i,player in enumerate(players.to_dict("records")):
        j=idx[a[i]]
        # Same selection window as the lineup optimizer; retain full-window P90 too.
        n=float(y[:7000,j].mean()); m=float(x[:7000,i].mean())
        rows.append(dict(season=int(season),week=int(week),seed=int(seed),
                         player_id=a[i],player=str(player["Name"]),team=str(player["TeamAbbrev"]),
                         position=str(player["Position"]),neutral_mean=n,adjusted_mean=m,
                         delta_mean=m-n,neutral_p90=float(np.percentile(y[:,j],90)),
                         adjusted_p90=float(np.percentile(x[:,i],90))))
    return rows
