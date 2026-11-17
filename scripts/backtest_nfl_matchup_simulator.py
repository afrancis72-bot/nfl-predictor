"""Research-only paired opponent-aware-vs-neutral *full football* simulation.

The football-matchup experiment affects actual passing/rushing/receiving
yardage distributions, not additive fantasy scoring. Test on *the same*
player-game identities, using exactly the same seeded 400-scenario game worlds.

Choose strength only from 2025 Weeks 9-12, then compare on 2026 Weeks 4-10.
WARNING: 2026 games have already informed prior AI2 experiments; this is
retrospective confirmation, NOT untouched prospective or contest validation.
No production simulator flags, lineups, accepted portfolios are changed.

Artifacts in --out-dir: paired_player_outcomes.csv, test_results.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from props_data import fetch_weekly
from ai2_nfl_snap_roles import fetch_snap_counts
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE, evaluate
from scripts.audit_nfl_full_slates import metric_summary, game_cluster_lift_ci


KEYS = ["season", "week", "game", "player", "team", "position"]
WEIGHTS = (0., .5, 1.)


def match_results(neutral: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Identical observed player-games; missing players cannot improve results."""
    if neutral.empty or candidate.empty:
        raise ValueError("Empty historical simulator output cannot validate model")
    for source in (neutral, candidate):
        if source.duplicated(KEYS).any():
            raise ValueError("Duplicate player-game within historical result")
    pair = neutral.merge(candidate, on=KEYS, how="outer",
                         validate="one_to_one",
                         suffixes=("_neutral", "_matchup"), indicator=True)
    if not pair["_merge"].eq("both").all():
        raise ValueError("Neutral and opponent-aware runs have different player pools")
    for col in ("actual_partial_dk", "history_mean_partial_dk"):
        if not np.allclose(pair[f"{col}_neutral"],
                           pair[f"{col}_matchup"], atol=1e-6):
            raise ValueError("Opponent matchup experiment changed target labels")
    return pair.drop(columns=["_merge"])


def paired_metrics(pair: pd.DataFrame) -> dict:
    if pair.empty:
        return {"n": 0}
    actual = pair["actual_partial_dk_neutral"].to_numpy(float)
    old = pair["simulated_mean_dk_neutral"].to_numpy(float)
    new = pair["simulated_mean_dk_matchup"].to_numpy(float)
    neutral_error = np.abs(actual-old)
    matchup_error = np.abs(actual-new)
    if not np.isfinite([neutral_error, matchup_error]).all():
        raise ValueError("Non-finite paired predictions")
    ref = pair[[
        "season", "week", "game", "player", "team", "position",
        "actual_partial_dk_neutral", "simulated_mean_dk_neutral",
        "simulated_mean_dk_matchup",
        "simulated_p10_dk_matchup", "simulated_p90_dk_matchup"
    ]].rename(columns={
        "actual_partial_dk_neutral": "actual_partial_dk",
        "simulated_mean_dk_neutral": "history_mean_partial_dk",
        "simulated_mean_dk_matchup": "simulated_mean_dk",
        "simulated_p10_dk_matchup": "simulated_p10_dk",
        "simulated_p90_dk_matchup": "simulated_p90_dk",
    })
    metrics = metric_summary(ref)
    metrics["baseline_name"] = "Identical-seed opponent-neutral simulated mean"
    metrics["baseline_simulator_mae"] = metrics.pop("historical_baseline_mae")
    metrics["matchup_simulator_mae"] = metrics.pop("simulator_mae")
    metrics["matchup_mae_improvement_pct"] = metrics.pop("simulator_improvement_pct")
    metrics["paired_game_cluster_lift_ci"] = game_cluster_lift_ci(ref)
    return metrics


def run_comparison(history, schedules, snaps, *,
                   train_weeks=(9,10,11,12),
                   test_weeks=(4,5,6,7,8,9,10),
                   n_sims=400) -> tuple[dict, pd.DataFrame]:
    training = {}
    neutral_train, errors = evaluate(
        history, schedules, snaps, 2025, train_weeks,
        games_per_week=None, n_sims=n_sims, component_strength=0.)
    if neutral_train.empty:
        raise ValueError(f"No 2025 neutral scores: {errors[:5]}")
    for strength in WEIGHTS:
        if strength == 0.:
            candidate, issues = neutral_train, errors
        else:
            candidate, issues = evaluate(
                history, schedules, snaps, 2025, train_weeks,
                games_per_week=None, n_sims=n_sims,
                component_strength=strength)
        paired = match_results(neutral_train,candidate)
        training[str(strength)] = {
            "metrics": paired_metrics(paired),
            "simulator_errors": issues[:15],
            "simulator_skipped_count": len(issues),
        }
    # Training-only choice. A marginal reduction that could be noise stays
    # neutral; avoiding hardcoded tuning on already-seen 2026 games.
    baseline_mae = training["0.0"]["metrics"]["baseline_simulator_mae"]
    qualified = [
        s for s in WEIGHTS if s > 0 and
        (baseline_mae-training[str(s)]["metrics"]["matchup_simulator_mae"])
        / max(baseline_mae,1e-9) >= .01
    ]
    selected = min(
        qualified,
        key=lambda strength:(
            training[str(strength)]["metrics"]["matchup_simulator_mae"],
            strength)
    ) if qualified else 0.

    neutral, issues0 = evaluate(
        history, schedules, snaps, 2026, test_weeks,
        games_per_week=None, n_sims=n_sims, component_strength=0.)
    candidate, issues1 = (
        evaluate(history, schedules, snaps, 2026, test_weeks,
                 games_per_week=None, n_sims=n_sims,
                 component_strength=selected)
        if selected > 0 else (neutral, issues0)
    )
    paired = match_results(neutral, candidate)
    overall = paired_metrics(paired)
    by_pos = {
        p: paired_metrics(group)
        for p, group in paired.groupby("position")
    }
    per_week = {
        str(w): paired_metrics(group)
        for w, group in paired.groupby("week")
    }
    ci = overall["paired_game_cluster_lift_ci"]["lift_pct_95_ci"]
    improvement = overall["matchup_mae_improvement_pct"]
    qualifications = {
        "2025_training_selected_nonzero_component_strength": selected > 0,
        "2026_retrospective_mae_lower_than_neutral": (
            improvement is not None and improvement > 0),
        "2026_cluster_bootstrap_95pct_lower_bound_positive": (
            ci is not None and ci[0] > 0),
        "2026_p90_coverage_at_least_85pct": (
            85 <= overall["actual_below_p90_pct"] <= 95),
        "no_position_mae_regression_over_3pct": all(
            v["matchup_mae_improvement_pct"] is not None
            and v["matchup_mae_improvement_pct"] >= -3
            for v in by_pos.values()),
    }
    output = {
        "name": "Experimental pregame opponent-adjusted football-stat simulator vs neutral",
        "training_period": "2025 W9-12; all completed Sunday games",
        "retrospective_test_period": "2026 W4-10; all completed Sunday games",
        "n_simulations_per_game": n_sims,
        "selected_strength_from_2025_only": selected,
        "training": training,
        "retrospective_2026": overall,
        "by_position_2026": by_pos,
        "by_week_2026": per_week,
        "skipped_games": {
            "neutral": issues0[:60],
            "component": issues1[:60],
        },
        "research_component_pass": all(qualifications.values()),
        "requirements": qualifications,
        "production_approved": False,
        "limitations": [
            "2026 weeks have been used in prior AI2 model research, so not truly untouched",
            "Historical player pools reconstructed from earlier usage and include played players only",
            "Only YPA, YPC and relative receiver YPT are adjusted; defense impact on volume, TD and possessions not yet modeled",
            "Historical salary, verified full injury roster, DFS contest and bookmaker information not replayed",
            "Actual player fantasy points are partial DraftKings scores",
            "Research is not production validation or a guaranteed betting edge",
        ],
    }
    return output, paired


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="nfl_matchup_simulator_audit")
    p.add_argument("--n-sims", type=int, default=400)
    args=p.parse_args(argv)
    if not 100 <= args.n_sims <= 10000:
        p.error("n-sims must be 100 to 10000")
    history=fetch_weekly([2024,2025,2026])
    schedules=pd.read_csv(GAMES_SOURCE)
    snaps=fetch_snap_counts([2025,2026])
    report, rows=run_comparison(history,schedules,snaps,n_sims=args.n_sims)
    dest=Path(args.out_dir)
    dest.mkdir(parents=True,exist_ok=True)
    rows.to_csv(dest/"paired_player_outcomes.csv",index=False)
    (dest/"test_results.json").write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({
        "selected_strength":report["selected_strength_from_2025_only"],
        "2026":report["retrospective_2026"],
        "requirements":report["requirements"],
        "research_component_pass":report["research_component_pass"],
        "production_approved":False,
        "artifact_directory":str(dest),
    },indent=2))


if __name__=="__main__":
    main()
