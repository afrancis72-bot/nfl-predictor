"""Audit NBA rotation/prop compression with real pregame source logs + official status."""
from datetime import date
from io import BytesIO
import requests
import numpy as np
from nba_engine.data_sources import load_public_logs
from nba_engine.predictor import build_projections
from nba_engine.rotation_model import select_plausible_rotation
from nba_engine.injury_report import parse_official_injury_pdf,injury_exclusions,match_excluded_ids
from nba_engine.game_conservation import (
    normalize_team_minutes,calibrate_team_scoring_baselines,
    simulate_conserved_nba_games)

WHEN=date(2026,11,23)
GAMES={"POR@PHI","CLE@CHI","OKC@IND","MIL@TOR","BKN@MIN"}
STARS=["Tyrese Maxey","Jaylen Brown","Donovan Mitchell",
       "Shai Gilgeous-Alexander","Anthony Edwards","Giannis Antetokounmpo"]
def main():
    history,meta=load_public_logs(WHEN)
    print("PUBLIC LOGS",meta)
    url="https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-11-23_02_45PM.pdf"
    pdf=requests.get(url,timeout=20)
    pdf.raise_for_status()
    statuses=parse_official_injury_pdf(pdf.content,target_games=GAMES,
                                      document_date=WHEN,allow_partial=True)
    print("OFFICIAL source",url,"complete games",statuses.attrs["submitted_games"])
    team_map={}
    for game in GAMES:
        a,b=game.split("@")
        team_map[a]=b
        team_map[b]=a
    for game in sorted(GAMES):
        away,home=game.split("@")
        pair={away:home,home:away}
        excluded,uncertain=injury_exclusions(statuses.loc[statuses.game.eq(game)],pair)
        excluded_ids,_=match_excluded_ids(history,excluded,pair,WHEN)
        p=build_projections(history,pair,WHEN,min_minutes=8,excluded=excluded_ids)
        raw=p.copy()
        p,rotation=select_plausible_rotation(p,history,WHEN,uncertain_status=uncertain)
        active=p.copy()
        p,minutes=normalize_team_minutes(p)
        normalized=p.copy()
        p,baseline=calibrate_team_scoring_baselines(p,history,WHEN)
        draw,game_audit=simulate_conserved_nba_games(
            p,history,when=WHEN,trials=10000,seed=47)
        print("\nMATCHUP",game)
        print("Raw total minutes",raw.groupby("team").minutes.sum().to_dict(),
              "recent active",active.groupby("team").minutes.sum().to_dict())
        print("AFTER",minutes[["team","projected_minutes_after","core_minutes_after"]].to_string(index=False))
        print("SCORE",baseline[["team","points_scaling_factor"]].to_string(index=False))
        print("GAME",game_audit[["team","mean_team_points"]].to_string(index=False))
        for star in STARS:
            q=p.loc[p.player.eq(star)]
            if len(q)==0:
                continue
            r=raw.loc[raw.player.eq(star)]
            o=normalized.loc[normalized.player.eq(star)]
            hist=history.loc[history.player.eq(star)].sort_values("date")
            pid=str(q.iloc[0].player_id)
            print("STAR",star,"last15",round(hist.pts.tail(15).mean(),2),
                  "raw minutes",round(float(r.iloc[0].minutes),2),
                  "recent minutes",round(float(active.loc[active.player.eq(star),"minutes"].iloc[0]),2),
                  "after minute",round(float(o.iloc[0].minutes),2),
                  "after baseline",round(float(q.iloc[0].Points),2),
                  "sim mean",round(float(draw[(pid,"Points")].mean()),2),
                  "sim p90",round(float(np.quantile(draw[(pid,"Points")],.9)),2))
        print("REMOVED FROM ROTATION",len(rotation))
        print(rotation[["player","reason"]].head(14).to_string(index=False))
if __name__=="__main__":
    main()
