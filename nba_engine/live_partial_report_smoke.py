"""Reproduce the Nov 23 2:45 PM NBA report: one team pending, others run.

Reads the *actual NBA official PDF* and real pregame historical player logs.
This is a data-path and game-model integration test, NOT forecast calibration.
"""
from datetime import date,datetime
from zoneinfo import ZoneInfo
from nba_engine.data_sources import load_public_logs
from nba_engine.injury_report import fetch_current_official_injuries
from nba_engine.slate_runner import run_verified_slate

WHEN=date(2026,11,23)
GAMES={
    "POR@PHI","MEM@CHA","BKN@MIN","OKC@IND","MIL@TOR",
    "CLE@CHI","PHX@NOP","LAC@HOU","UTA@LAL","SAS@SAC"
}


def main():
    now=datetime(2026,11,23,14,47,tzinfo=ZoneInfo("America/New_York"))
    report,meta=fetch_current_official_injuries(
        WHEN,target_games=GAMES,now=now)
    print("Actual NBA report",meta)
    print("Official status row count",len(report))
    print("Verified games",meta["games_reported"])
    print("Pending games",meta["games_pending"])
    assert "UTA@LAL" in meta["games_pending"],meta
    assert "UTA@LAL" not in meta["games_reported"],meta
    assert len(meta["games_reported"])>=7,meta
    assert not (report.game=="UTA@LAL").any()
    team_map={}
    for game in GAMES:
        away,home=game.split("@")
        team_map[away]=home
        team_map[home]=away
    history,provenance=load_public_logs(WHEN)
    print("Actual pregame history",provenance)
    result=run_verified_slate(
        history,team_map,report,meta["games_reported"],
        when=WHEN,trials=10000,seed=47)
    print("Simulated matchup results",result["games_completed"])
    print("Unavailable due to other player data",result["games_failed"])
    print(result["game_audit"][
        ["game","team","mean_team_points",
         "mean_projected_minutes","team_score_conservation_passed"]
    ].to_string(index=False))
    assert len(result["games_completed"])>=6, result["games_failed"]
    assert "UTA@LAL" not in result["games_completed"]
    assert "UTA" not in result["teams_map"]
    assert result["game_audit"].team_score_conservation_passed.all()
    assert result["game_audit"].mean_projected_minutes.max()<=240.000001
    assert len(result["samples"])>50
    print("PASS: one unsubmitted NBA game did not block the other games")


if __name__=="__main__":
    main()
