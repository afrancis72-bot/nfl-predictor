"""Live data end-to-end NBA IND/CLE regression from 2026-11-20 as-of report.

This is real official NBA injury-source and real public player-game logs.
Not a retrospective proof of predictive accuracy or profit.
"""
from datetime import date,datetime
from zoneinfo import ZoneInfo
from io import BytesIO
import requests

from nba_engine.data_sources import load_public_logs
from nba_engine.injury_report import (
    parse_official_injury_pdf,fetch_current_official_injuries,
    injury_exclusions,match_excluded_ids)
from nba_engine.predictor import build_projections
from nba_engine.game_conservation import (
    normalize_team_minutes,calibrate_team_scoring_baselines,
    simulate_conserved_nba_games)


def main():
    when=date(2026,11,20)
    url=("https://ak-static.cms.nba.com/referee/injury/"
         "Injury-Report_2026-11-20_02_15PM.pdf")
    response=requests.get(url,timeout=20)
    response.raise_for_status()
    status=parse_official_injury_pdf(
        response.content,target_games={"IND@CLE"},document_date=when)
    now=datetime(2026,11,20,14,17,tzinfo=ZoneInfo("America/New_York"))
    realtime,status_meta=fetch_current_official_injuries(
        when,target_games={"IND@CLE"},now=now)
    assert set(realtime.player_key)==set(status.player_key)
    assert status_meta["source_url"]==url
    print("OFFICIAL injury names",status[
        ["team","player","status"]].to_string(index=False))
    assert (status.team.eq("IND")&
            status.player_key.eq("tyresehaliburton")&
            status.status.eq("Out")).any()
    assert (status.team.eq("CLE")&
            status.player_key.eq("evanmobley")&
            status.status.eq("Out")).any()
    teams={"IND":"CLE","CLE":"IND"}
    excluded,_=injury_exclusions(status,teams)
    logs,meta=load_public_logs(when)
    ids,unmatched=match_excluded_ids(logs,excluded,teams,when)
    print("Historical stats source",meta)
    print("Injury-out matched IDs",ids,"unmatched official OUT",unmatched)
    p=build_projections(
        logs,teams,when,min_minutes=8,excluded=ids)
    p,minute_audit=normalize_team_minutes(p)
    p,baseline_audit=calibrate_team_scoring_baselines(p,logs,when)
    draws,score_audit=simulate_conserved_nba_games(
        p,logs,when=when,trials=10000,seed=42)
    print("BASELINE POINT CONSERVATION\n",baseline_audit.to_string(index=False))
    print("MINUTES\n",minute_audit.to_string(index=False))
    print("TEAM SCORE\n",score_audit.to_string(index=False))
    print("COMBINED MEAN",score_audit.mean_team_points.sum())
    assert not (p.player.str.contains("Haliburton|Mobley",
                                      regex=True,case=False)).any()
    assert minute_audit.projected_minutes_after.max()<=240.000001
    assert score_audit.team_score_conservation_passed.all()
    assert not baseline_audit.uses_sportsbook_total.any()
    assert (baseline_audit.constrained_player_baseline_points<=
            baseline_audit.expected_team_points_from_complete_history+0.10).all()
    assert score_audit.mean_team_points.sum()<300
    assert len(draws)>50
    print("PASS: actual-source NBA game and injury constraints used.")


if __name__=="__main__":
    main()
