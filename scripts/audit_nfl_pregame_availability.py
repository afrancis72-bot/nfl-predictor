"""Synthetic strict pregame availability contract audit, NOT a real NFL source.

The synthetic NFL URLs in this fixture are never fetched. Replays confirm
absence of evidence means no expanded football player universe, and dated
positive-active claims are required to create additional internal players.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from test_nfl_full_player_universe import fixture
from test_nfl_pregame_availability import snapshot,evidence
from ai2_nfl_pregame_availability import ACTIVE,OUT,UNKNOWN,classify_game_players
from ai2_nfl_full_player_universe import build_football_only_players
from dfs_stats_engine import simulate_stat_driven_dfs


def score(pool,rates,extra=None,status=None,n=10000):
    result=simulate_stat_driven_dfs(
        pool,rates,n_sims=n,seed=20261117,allow_role_proxies=False,
        wr_role_team_rng_isolated=True,
        football_only_players=extra,
        football_universe_target_season=(2026 if extra is not None else None),
        football_universe_target_week=(7 if extra is not None else None),
        football_availability_snapshot=status)
    players,points,stats=result
    assert players["ID"].astype(str).str.startswith("FOOTBALL_ONLY").sum()==0
    for team in ("AAA","BBB"):
        names=pool.loc[(pool["TeamAbbrev"]==team) &
                       pool["Position"].isin(("WR","RB","TE")),"Name"].tolist()
        if extra is not None:
            names+=extra.loc[extra["TeamAbbrev"]==team,"Name"].tolist()
        modeled=sum((np.asarray(stats[name]["targets"]) for name in names),
                    np.zeros(n,dtype=int))
        accounted=modeled+stats["__OTHER_RECEIVER__|"+team]["targets"]
        assert np.array_equal(accounted,stats["__TEAM_PLAYS__|"+team]["simulated_pass_attempts"])
    return result


def run(n):
    pool,rates=fixture()
    # ALL source URLs below are TEST STRINGS, never verified NFL content.
    uncertain=snapshot(evidence("Missing Salary WR",UNKNOWN,
                                 "NFL_OFFICIAL_GAME_STATUS_REPORT"))
    active=snapshot(evidence("Missing Salary WR",ACTIVE,
                              "NFL_GAME_DAY_ACTIVE_ROSTER"))
    ruled_out=snapshot(evidence("Missing Salary WR",OUT,
                                "NFL_OFFICIAL_INACTIVES"))
    empty=build_football_only_players(pool,rates,season=2026,week=7,
                require_verified_active=True)
    unknown=build_football_only_players(pool,rates,season=2026,week=7,
                require_verified_active=True,availability_snapshot=uncertain)
    activate=build_football_only_players(pool,rates,season=2026,week=7,
                require_verified_active=True,availability_snapshot=active)
    out=build_football_only_players(pool,rates,season=2026,week=7,
                require_verified_active=True,availability_snapshot=ruled_out)
    assert empty.empty and unknown.empty and out.empty
    assert activate.Name.tolist()==["Missing Salary WR"]
    basic=score(pool,rates,None,n=n)
    no_status=score(pool,rates,empty,n=n)
    questionable=score(pool,rates,unknown,uncertain,n=n)
    out_world=score(pool,rates,out,ruled_out,n=n)
    expanded=score(pool,rates,activate,active,n=n)
    for case in (no_status,questionable,out_world,expanded):
        assert basic[0].ID.tolist()==case[0].ID.tolist()
        assert basic[0].columns.tolist()==case[0].columns.tolist()
        assert basic[1].shape==case[1].shape
    for case in (no_status,questionable,out_world):
        assert np.array_equal(basic[1],case[1])
    assert not np.array_equal(basic[1],expanded[1])
    for team in ("AAA","BBB"):
        k="__TEAM_PLAYS__|"+team
        for metric in ("simulated_pass_attempts","simulated_plays",
                       "simulated_dropbacks","simulated_sacks"):
            assert np.array_equal(basic[2][k][metric],expanded[2][k][metric])
    return {
        "audit":"synthetic status gate and 10k-simulation football conservation",
        "source_content_independently_checked":False,
        "published_kickoff_are_synthetic":True,
        "is_nfl_real_world_availability_data":False,
        "status_model_release_approved":False,
        "independent_holdout":False,
        "full_universe_live_enabled":False,
        "n_sims":n,
        "no_evidence_no_extra_players":len(empty)==0,
        "questionable_remains_uncertain":len(unknown)==0,
        "explicit_out_does_not_activate_player":len(out)==0,
        "verified_source_claim_positive_active_adds_player":activate.Name.tolist(),
        "all_original_DK_players_untouched":True,
        "unknown_and_out_hold_opportunity_in_OTHER_without_free_gifting":True,
        "same_team_pass_attempts_and_conserved_player_targets":True,
        "original_salary_pool_scoring_bitwise_unchanged_with_no_verifiable_evidence":True,
        "source_claim_cannot_be_promoted_without_source_fetched_and_audited":True,
        "blocked_future_steps":[
            "Fetch, verify and freeze actual official pregame game-day source records",
            "Do not treat historical postgame weekly roster records as pregame data",
            "Register exact NFL game kickoff, player/team identity and capture timestamp",
            "Archive untouched pregame forecast snapshots for real future validation",
            "Require calibrated actual fantasy-point MAE and P90 improvement in independent games",
        ],
        "summary_counts":{
            "known_historical_extra_wr_in_fixture":2,
            "official_positive_active_rows_in_test":1,
            "indeterminate_player_rows_excluded_from_research_model":1,
            "synthetic_expanded_case_internal_players":expanded[2][
                "__FOOTBALL_PLAYER_UNIVERSE__"]["historical_football_only_players"],
        },
    }


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--sims",type=int,default=10000)
    p.add_argument("--out",default="nfl_pregame_availability_gate/validation.json")
    a=p.parse_args(argv)
    if a.sims<100 or a.sims>100000:
        p.error("sims must be within 100-100000")
    result=run(a.sims)
    outfile=Path(a.out)
    outfile.parent.mkdir(parents=True,exist_ok=True)
    outfile.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,sort_keys=True),flush=True)
    print("SYNTHETIC CONTRACT TEST PASS; NO NFL SOURCE INGESTION OR PRODUCTION PROMOTION")

if __name__=="__main__":
    main()
