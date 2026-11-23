"""Diagnose NBA model compression on actual recent published player game logs.
No fabricated stats. Printed numbers are pre-game retrospective source diagnostics.
"""
from datetime import date
import pandas as pd
from nba_engine.data_sources import load_public_logs
from nba_engine.predictor import build_projections
from nba_engine.game_conservation import (
    normalize_team_minutes,calibrate_team_scoring_baselines)


def diagnose(when,teams,star):
    history,meta=load_public_logs(when)
    print("\nGAME",when,teams,"LATEST_HISTORY",meta["latest_game_date"],"ROWS",meta["rows"])
    p=build_projections(history,teams,when,min_minutes=8)
    roster=history.loc[history.player.eq(star)].sort_values("date").tail(30)
    print("STAR HISTORY",star,"n",len(roster),
          "last5 PTS",round(roster.pts.tail(5).mean(),2),
          "last15 PTS",round(roster.pts.tail(15).mean(),2),
          "last5 MIN",round(roster.minutes.tail(5).mean(),2),
          "last15 MIN",round(roster.minutes.tail(15).mean(),2))
    print("INITIAL MIN/PTS",p.loc[p.player.eq(star),["player","minutes","Points","current_season_games"]].to_string(index=False))
    print("TEAM UNCONSTRAINED",p.groupby("team").agg(players=("player","count"),minutes=("minutes","sum"),points=("pts","sum")).to_string())
    p,m=normalize_team_minutes(p)
    print("AFTER ROTATION",p.loc[p.player.eq(star),["player","minutes","Points"]].to_string(index=False))
    p,b=calibrate_team_scoring_baselines(p,history,when)
    print("AFTER SCORING",p.loc[p.player.eq(star),["player","minutes","Points"]].to_string(index=False))
    print("ROTATION",m.to_string(index=False))
    print("SCORING",b.to_string(index=False))
    team=next((t for t in teams if t in set(p.loc[p.player.eq(star),"team"])),None)
    if team:
        print("PLAYERS",p.loc[p.team.eq(team),["player","minutes","Points"]].sort_values("minutes",ascending=False).to_string(index=False))


if __name__=="__main__":
    diagnose(date(2026,11,23),{"PHI":"POR","POR":"PHI"},"Tyrese Maxey")
    diagnose(date(2026,11,24),{"BOS":"WAS","WAS":"BOS"},"Jaylen Brown")
    diagnose(date(2026,11,23),{"CLE":"CHI","CHI":"CLE"},"Donovan Mitchell")
    diagnose(date(2026,11,23),{"OKC":"IND","IND":"OKC"},"Shai Gilgeous-Alexander")
