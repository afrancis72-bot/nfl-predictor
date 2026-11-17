"""Research-stage NFL *complete Sunday-game* player forecast benchmark.

This is NOT a DraftKings production-slate or contest backtest. The evaluation
pool is reconstructed from prior-week nflverse player usage, because dated
historical DK salary files and complete pre-kickoff inactive snapshots are not
available for every week. This omits some active and late-emerging players.

The production-neutral stat simulator is compared with its own simple
pregame historical-stat baseline; no actual target-week stats enter features.
An out-of-time-looking split is *not* independent if model choices already
used the 2026 results. Report that limitation prominently.

Run:
 PYTHONPATH=. python scripts/audit_nfl_full_slates.py --out-dir nfl_audit
Artifacts: player_predictions.csv, weekly_summary.csv, validation.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ai2_nfl_matchups import canonical_team
from ai2_nfl_snap_roles import fetch_snap_counts
from props_data import fetch_weekly
from scripts.backtest_nfl_player_dk_means import GAMES_SOURCE, evaluate

# Frozen *research* windows. 2026 W4-8 has already informed previous model
# experiments; it must never be described as previously untouched.
WINDOWS = {
    "2025_reference": (2025, [9, 10, 11, 12]),
    "2026_retrospective": (2026, [4, 5, 6, 7, 8, 9, 10]),
}
REQUIRED = {
    "season", "week", "game", "player", "team", "position",
    "actual_partial_dk", "history_mean_partial_dk",
    "simulated_mean_dk", "simulated_p10_dk", "simulated_p90_dk",
}


def metric_summary(frame: pd.DataFrame) -> dict:
    """Every prediction is a player-game, not a distinct person/lineup."""
    if frame.empty:
        return {"n": 0}
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing benchmark columns: {sorted(missing)}")
    y = frame["actual_partial_dk"].to_numpy(dtype=float)
    neutral = frame["history_mean_partial_dk"].to_numpy(dtype=float)
    sim = frame["simulated_mean_dk"].to_numpy(dtype=float)
    p10 = frame["simulated_p10_dk"].to_numpy(dtype=float)
    p90 = frame["simulated_p90_dk"].to_numpy(dtype=float)
    if not all(np.isfinite(a).all() for a in (y, neutral, sim, p10, p90)):
        raise ValueError("Non-finite forecast/actual data: do not silently drop")
    if (p10 > p90).any():
        raise ValueError("Simulator P10 is above P90")
    err = np.abs(sim - y)
    base_err = np.abs(neutral - y)
    baseline_mae = float(np.mean(base_err))
    sim_mae = float(np.mean(err))
    lift = (100 * (baseline_mae - sim_mae) / baseline_mae
            if baseline_mae > 0 else None)
    return {
        "n": len(frame),
        "historical_baseline_mae": round(baseline_mae, 4),
        "simulator_mae": round(sim_mae, 4),
        "simulator_rmse": round(float(np.sqrt(np.mean((sim-y)**2))), 4),
        "simulator_bias_predicted_minus_actual": round(float(np.mean(sim-y)), 4),
        "simulator_improvement_pct": round(lift, 3) if lift is not None else None,
        "within_3_dk_pct": round(100 * float(np.mean(err <= 3)), 2),
        "within_5_dk_pct": round(100 * float(np.mean(err <= 5)), 2),
        "within_10_dk_pct": round(100 * float(np.mean(err <= 10)), 2),
        "actual_below_p90_pct": round(100 * float(np.mean(y <= p90)), 2),
        "actual_below_p10_pct": round(100 * float(np.mean(y <= p10)), 2),
        "actual_inside_p10_p90_pct": round(
            100 * float(np.mean((y >= p10) & (y <= p90))), 2),
    }


def game_cluster_lift_ci(frame: pd.DataFrame, *, seed: int = 42,
                         resamples: int = 1500) -> dict:
    """Paired bootstrap by GAME: correlated players cannot be independent."""
    if frame.empty:
        return {"games": 0, "lift_pct_95_ci": None}
    df = frame.copy()
    df["_game_id"] = (df["season"].astype(str) + "-" +
                      df["week"].astype(str) + "-" + df["game"].astype(str))
    df["_base_error"] = (df["history_mean_partial_dk"] -
                         df["actual_partial_dk"]).abs()
    df["_sim_error"] = (df["simulated_mean_dk"] -
                        df["actual_partial_dk"]).abs()
    grouped = df.groupby("_game_id")[["_base_error", "_sim_error"]].sum()
    n_games = len(grouped)
    if n_games < 2:
        return {"games": n_games, "lift_pct_95_ci": None}
    values = grouped.to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_games, size=(resamples, n_games))
    aggregate = values[idx].sum(axis=1)
    base = aggregate[:, 0]
    lifts = np.divide(100 * (base - aggregate[:, 1]), base,
                      out=np.full(resamples, np.nan), where=base > 0)
    good = lifts[np.isfinite(lifts)]
    if len(good) < resamples * .95:
        raise ValueError("Insufficient nonzero baseline error for bootstrap")
    lo, hi = np.percentile(good, [2.5, 97.5])
    return {
        "games": n_games,
        "resamples": resamples,
        "lift_pct_95_ci": [round(float(lo), 3), round(float(hi), 3)],
    }


def provisional_gates(frame: pd.DataFrame, *, skipped_games: int = 0) -> dict:
    """Strict *diagnostic* gates, never automatic approval for paid entry.

    These thresholds are proposed engineering requirements, not validated
    profitability criteria. An independent prospective freeze and true
    historical salary/availability audit are still mandatory for production.
    """
    m = metric_summary(frame)
    ci = game_cluster_lift_ci(frame)
    n = m["n"]
    total_games = ci["games"] + skipped_games
    skip_pct = 100 * skipped_games / total_games if total_games else 100
    by_position = {
        pos: metric_summary(df)
        for pos, df in frame.groupby("position")
    } if n else {}
    tests = {
        "at_least_400_matched_player_games": n >= 400,
        "at_least_35_simulated_games": ci["games"] >= 35,
        "skipped_games_under_10pct": skip_pct <= 10,
        "simulator_mae_improvement_at_least_5pct": (
            m.get("simulator_improvement_pct") is not None and
            m["simulator_improvement_pct"] >= 5),
        "game_cluster_bootstrap_lower_lift_gt_zero": (
            ci["lift_pct_95_ci"] is not None and
            ci["lift_pct_95_ci"][0] > 0),
        "p90_below_rate_85_to_95pct": (
            85 <= m.get("actual_below_p90_pct", -1) <= 95),
        "p10_p90_interval_75_to_85pct": (
            75 <= m.get("actual_inside_p10_p90_pct", -1) <= 85),
        "all_roles_at_least_25_and_no_gt_3pct_regression": all(
            by_position.get(p, {}).get("n", 0) >= 25 and
            by_position[p].get("simulator_improvement_pct") is not None and
            by_position[p]["simulator_improvement_pct"] >= -3
            for p in ("QB", "RB", "WR", "TE")
        ),
    }
    return {
        "research_diagnostic_pass": all(tests.values()),
        "checks": tests,
        "matched_player_games": n,
        "simulated_games": ci["games"],
        "skipped_games": skipped_games,
        "skipped_game_pct": round(skip_pct, 2),
        "game_cluster_lift_confidence": ci,
        "position_results": by_position,
        "production_approved": False,
        "production_blockers": [
            "No independent forward/frozen pre-kickoff validation period",
            "Synthetic historical player pools omit dated DK salaries and verified inactive reports",
            "Partial DK scoring omits some fumbles, return scores and two-point events",
            "Production one-click data pipeline and 10,000-sim optimizer not replayed end-to-end",
            "Matchup-specific defense effects remain disabled in live scoring",
        ],
    }


def _scheduled_games(schedules: pd.DataFrame, season: int, weeks: list[int]) -> int:
    mask = (schedules["season"].eq(season) &
            schedules["week"].isin(weeks) &
            schedules["game_type"].astype(str).eq("REG") &
            schedules["home_score"].notna() & schedules["away_score"].notna())
    ss = schedules[mask]
    if "weekday" in ss:
        ss = ss[ss["weekday"].eq("Sunday")]
    return int(len(ss))


def build_report(history: pd.DataFrame, schedules: pd.DataFrame,
                 snaps: pd.DataFrame, *, n_sims: int = 400) -> tuple[dict, pd.DataFrame]:
    """Run every completed Sunday game in each predefined week."""
    blocks = []
    windows = {}
    for label, (season, weeks) in WINDOWS.items():
        observations, problems = evaluate(
            history, schedules, snaps, season, weeks,
            games_per_week=None, n_sims=n_sims,
            # Explicitly match the current opponent-neutral production setting;
            # other production app ingestion/optimization details differ.
            defense_weight=0.0, market_weight=0.0)
        if not observations.empty:
            observations = observations.copy()
            observations["audit_window"] = label
            blocks.append(observations)
        schedule_total = _scheduled_games(schedules, season, weeks)
        simulated_games = int(
            observations[["week", "game"]].drop_duplicates().shape[0]
        ) if not observations.empty else 0
        windows[label] = {
            "season": season, "weeks": weeks, "scheduled_completed_sunday_games": schedule_total,
            "observed_matched_games": simulated_games,
            "unreported_or_skipped_games": schedule_total - simulated_games,
            "simulation_errors": problems[:60],
            "player_metrics": metric_summary(observations),
        }
    details = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    if details.empty:
        raise ValueError("No matched player forecasts; cannot claim validation")
    reference = details[details["audit_window"] == "2026_retrospective"]
    if reference.empty:
        raise ValueError("No 2026 retrospective results to evaluate")
    skipped = windows["2026_retrospective"]["unreported_or_skipped_games"]
    gates = provisional_gates(reference, skipped_games=skipped)
    week_rows = [
        {"window": label, "season": int(year), "week": int(w),
         **metric_summary(df)}
        for (label, year, w), df in details.groupby(["audit_window", "season", "week"])
    ]
    report = {
        "title": "Team AI² Pro NFL full-Sunday research diagnostic (not contest ROI)",
        "source_branch": "feature/unified-ai2-platform",
        "source_commit": os.environ.get("GITHUB_SHA", "manual-run-unversioned"),
        "simulation_mode": "direct stat simulator, reconstructed prior-week player pools",
        "simulations_per_game": n_sims,
        "seed_rule": "season*100+week; fixed same seed within week",
        "windows": windows,
        "by_week": week_rows,
        "by_position_2026": {
            role: metric_summary(df)
            for role, df in reference.groupby("position")
        },
        "by_opponent_2026": {
            defense: metric_summary(df)
            for defense, df in reference.groupby("opponent")
        },
        "worst_15_absolute_misses_2026": (
            reference.assign(abs_error=(
                reference["simulated_mean_dk"] - reference["actual_partial_dk"]).abs())
            .sort_values("abs_error", ascending=False)
            [["week", "game", "player", "team", "position", "opponent",
              "simulated_mean_dk", "actual_partial_dk", "abs_error"]]
            .head(15).round(3).to_dict("records")
        ),
        "diagnostic_gates": gates,
        "limits": [
            "2026 Weeks 4-8 were previously used in research; not untouched holdout",
            "Only matched active player outcomes are scored: inactive players and non-modeled eligible players are omitted",
            "No frozen historical salary, final injuries or true DFS contest-result reconstruction",
            "Synthetic prior-week player pool differs from actual production DK player pool",
            "Actual scores use partial DK scoring from available nflverse box scores",
            "A pass on research gates does not certify real-world profitability or a ready production release",
        ],
    }
    return report, details


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="nfl_full_slate_audit")
    p.add_argument("--n-sims", type=int, default=400)
    args = p.parse_args(argv)
    if not 100 <= args.n_sims <= 10000:
        p.error("--n-sims must be between 100 and 10000")
    history = fetch_weekly([2024, 2025, 2026])
    schedule = pd.read_csv(GAMES_SOURCE)
    snaps = fetch_snap_counts([2025, 2026])
    report, details = build_report(history, schedule, snaps, n_sims=args.n_sims)
    dest = Path(args.out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    details.to_csv(dest / "player_predictions.csv", index=False)
    pd.DataFrame(report["by_week"]).to_csv(dest / "weekly_summary.csv", index=False)
    (dest / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    gates = report["diagnostic_gates"]
    print(json.dumps({
        "player_games": len(details),
        "2026_summary": report["windows"]["2026_retrospective"]["player_metrics"],
        "diagnostic_gate_results": gates["checks"],
        "research_diagnostic_pass": gates["research_diagnostic_pass"],
        "production_approved": False,
        "artifacts": str(dest),
    }, indent=2))


if __name__ == "__main__":
    main()
