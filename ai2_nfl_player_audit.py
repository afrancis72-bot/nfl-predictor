"""Read-only audit of each NFL player's simulated fantasy-point components.

Uses the SAME 10,000 football-stat worlds used for lineup selection.
Never substitutes salary-site averages or invented stats for actual draws.
"""
from collections import Counter
import numpy as np


def summarize_simulated_players(players, point_draws, stat_draws):
    """One compact audit record per simulated player, without retaining raw arrays."""
    n_sims = int(point_draws.shape[0])
    counts = Counter(players["Name"].astype(str))
    result = []

    for index, (_, player) in enumerate(players.iterrows()):
        name = str(player["Name"])
        team = str(player["TeamAbbrev"])
        position = str(player["Position"]).upper()
        scores = np.asarray(point_draws[:, index], dtype=float)
        item = {
            "Player": name,
            "Team": team,
            "Position": position,
            "DraftKings ID": str(player["ID"]),
            "Simulations": n_sims,
            "Sim mean DK": round(float(scores.mean()), 2),
            "Sim P90 DK": round(float(np.percentile(scores, 90)), 2),
            "Sim P99 DK": round(float(np.percentile(scores, 99)), 2),
            "Observed games": player.get("observed_games", ""),
            "Projection source": str(player.get("projection_source", "")),
            "Audit status": "No skill-player stat details (DST)",
        }
        if position not in {"RB", "WR", "TE", "QB"}:
            result.append(item)
            continue
        if counts[name] != 1:
            item["Audit status"] = "Ambiguous duplicated display name; no stat breakdown"
            result.append(item)
            continue
        detail = stat_draws.get(name)
        if not isinstance(detail, dict):
            item["Audit status"] = "No player stat draws in original simulation"
            result.append(item)
            continue

        def series(key, required=True):
            if key not in detail and not required:
                return np.zeros(n_sims, dtype=float)
            if key not in detail:
                raise ValueError(f"Simulation audit missing {key} for {name}")
            a = np.asarray(detail[key], dtype=float)
            if a.shape != (n_sims,) or not np.isfinite(a).all():
                raise ValueError(f"Simulation audit invalid {key} for {name}")
            return a

        if position == "QB":
            passing = series("passing_yards")
            pass_tds = series("passing_tds")
            interceptions = series("interceptions")
            rush_yards = series("rushing_yards")
            rush_tds = series("rushing_tds")
            pieces = {
                "Points: passing yards": 0.04 * passing,
                "Points: passing TD": 4.0 * pass_tds,
                "Points: interceptions": -interceptions,
                "Points: rushing yards": 0.1 * rush_yards,
                "Points: rushing TD": 6.0 * rush_tds,
                "Points: 300-yard passing bonus": 3.0 * (passing >= 300),
                "Points: 100-yard rushing bonus": 3.0 * (rush_yards >= 100),
            }
            item.update({
                "Mean passing yards": round(float(passing.mean()), 2),
                "Mean passing TD": round(float(pass_tds.mean()), 3),
                "Mean interceptions": round(float(interceptions.mean()), 3),
                "Mean rushing yards": round(float(rush_yards.mean()), 2),
                "Mean rushing TD": round(float(rush_tds.mean()), 3),
            })
        else:
            targets = series("targets")
            catches = series("receptions")
            yards = series("receiving_yards")
            rec_tds = series("receiving_tds")
            carries = series("carries")
            rush_yards = series("rushing_yards")
            rush_tds = series("rushing_tds")
            pieces = {
                "Points: receptions": catches,
                "Points: receiving yards": 0.1 * yards,
                "Points: receiving TD": 6.0 * rec_tds,
                "Points: 100-yard receiving bonus": 3.0 * (yards >= 100),
                "Points: rushing yards": 0.1 * rush_yards,
                "Points: rushing TD": 6.0 * rush_tds,
                "Points: 100-yard rushing bonus": 3.0 * (rush_yards >= 100),
            }
            item.update({
                "Mean targets": round(float(targets.mean()), 2),
                "Mean receptions": round(float(catches.mean()), 2),
                "Mean receiving yards": round(float(yards.mean()), 2),
                "Mean receiving TD": round(float(rec_tds.mean()), 3),
                "Chance 1+ receiving TD %": round(float(100 * (rec_tds >= 1).mean()), 1),
                "Chance 2+ receiving TD %": round(float(100 * (rec_tds >= 2).mean()), 1),
                "Chance 100+ receiving yards %": round(float(100 * (yards >= 100).mean()), 1),
                "Mean carries": round(float(carries.mean()), 2),
                "Mean rushing yards": round(float(rush_yards.mean()), 2),
                "Mean rushing TD": round(float(rush_tds.mean()), 3),
            })

            opportunity = stat_draws.get(f"__OPPORTUNITY__|{team}") or {}
            team_pass = stat_draws.get(f"__TEAM_PASS_BASELINE__|{team}") or {}
            scoring = stat_draws.get(f"__TEAM_SCORING__|{team}") or {}
            target_rates = opportunity.get("active_target_rates_pg") or {}
            rate = target_rates.get(name)
            budget = opportunity.get("target_budget_after")
            if rate is not None:
                item["Target allocation prior / game"] = round(float(rate), 2)
                if budget and float(budget) > 0:
                    item["Target allocation share %"] = round(100 * float(rate) / float(budget), 1)
            if team_pass.get("team_pass_attempts_baseline") is not None:
                item["Team pass attempts prior"] = round(float(team_pass["team_pass_attempts_baseline"]), 2)
            if scoring.get("expected_offensive_tds") is not None:
                item["Team offensive TD prior"] = round(float(scoring["expected_offensive_tds"]), 2)

        rebuilt = sum(pieces.values())
        if not np.allclose(rebuilt, scores, atol=0.02, rtol=0):
            raise ValueError(f"Simulated DraftKings points fail component reconciliation for {name}")
        item.update({key: round(float(values.mean()), 2) for key, values in pieces.items()})
        item["Rebuilt mean DK"] = round(float(rebuilt.mean()), 2)
        item["Audit status"] = "Verified: per-simulation DK points match components"
        result.append(item)
    return result
