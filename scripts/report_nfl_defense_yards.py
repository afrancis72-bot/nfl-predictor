"""Show plain historical defensive yards allowed by PRE-GAME player role.

No lineup optimization. No simulator or arbitrary matchup multipliers.
Research point-in-time data: 2026 week 12 means completed weeks <=11.
Output ALL measured defense/role rows and per-game audit details, plus
a filtered short "watch" list. The watch flag is descriptive, not predictive
skill, causal certainty, or proof of official current-week status.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from props_data import fetch_weekly
from ai2_nfl_defense_yardage_report import (
    observed_defense_yards,summarize_defense_yards)


def run_report(history,season,week):
    events=observed_defense_yards(history,season,week)
    summary=summarize_defense_yards(events)
    flagged=summary[summary["repeated_above_baseline"]].copy()
    return events,summary,flagged


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season",type=int,default=2026)
    p.add_argument("--week",type=int,default=12)
    p.add_argument("--out-dir",default="defense_role_yards")
    a=p.parse_args(argv)
    history=fetch_weekly([a.season-1,a.season])
    events,summary,flagged=run_report(history,a.season,a.week)
    root=Path(a.out_dir);root.mkdir(parents=True,exist_ok=True)
    events.to_csv(root/"all_observed_player_games.csv",index=False)
    summary.to_csv(root/"defense_role_yards_matrix.csv",index=False)
    flagged.sort_values("extra_yards_vs_player_average",ascending=False
                        ).to_csv(root/"repeated_yardage_weaknesses.csv",index=False)
    manifest={
        "season":a.season,"target_week_excluded":a.week,
        "source":"nflverse weekly player stats; 2025/2026 observed",
        "role_assignment":"Prior teammate target/carry/attempt ranking only",
        "player_baseline":"Prior 8 observed games, minimum 3",
        "pass":"QB passing; RB rushing and receiving; WR and TE receiving",
        "descriptive_consistency_rule":">=4 player-games, >=67% of games above player's pregame mean, >=10% mean yards above pregame own-player average",
        "games_are_observed_not_verified_current_player_status":True,
        "game_count_by_position_not_equal":True,
        "retrospective_not_prospective":True,
        "n_rows":len(summary),"n_player_games":len(events),
        "flagged_patterns":len(flagged),
        "production_matchup_multiplier_applied":False,
    }
    (root/"source_and_limits.json").write_text(json.dumps(
        manifest,indent=2),encoding="utf-8")
    print("Historical DEFENSE × ROLE receiving/rushing/passing yardage report")
    print("Target week excluded",a.week,"observed player games",len(events),
          "defense/role measured cells",len(summary),flush=True)
    print("Repeated measured yardage above player's baseline:",
          len(flagged),flush=True)
    print(flagged.sort_values("extra_yards_vs_player_average",
                              ascending=False).head(30).to_string(index=False),
          flush=True)
    print("Every defense/role metric, including weaker or insufficient samples, is saved.")
    print("No simulator, no lineup unlock or forecast multiplier.",flush=True)


if __name__=="__main__":
    main()
