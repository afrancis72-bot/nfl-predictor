"""Research-only opponent difficulty corrected for the caliber of past opponents.

The defense is measured by how many *additional* partial DraftKings points
players produced versus THEIR OWN pregame rolling baselines. Every baseline
uses earlier player-games, and every defense profile uses completed games
strictly BEFORE the target week. This is not a defender coverage/route model.

Never feed this into production scoring without chronological independent
validation; it is a fantasy-score residual, NOT a yard-per-opportunity factor.
"""
from __future__ import annotations

from collections import defaultdict, deque
import numpy as np
import pandas as pd

from ai2_nfl_matchups import canonical_team
from props_data import clean_weekly, dk_classic_points

POSITIONS = frozenset(("QB", "RB", "WR", "TE"))
SOURCE = "nflverse pregame own-player baseline versus completed opponent_team"
EMPTY = ("defense", "position", "observations", "defense_games",
         "raw_excess_dk", "shrunk_excess_dk", "mean_player_baseline_dk",
         "mean_actual_dk", "source", "target_season", "target_week")


def _identity(row):
    # Keep a traded player's new role separate from the old team's usage.
    team = canonical_team(row["recent_team"])
    pos = str(row["position"]).upper()
    ident = str(row.get("player_id", "") or "").strip()
    if ident.casefold() in {"", "nan", "none", "<na>", "null"}:
        ident = "NAME:" + " ".join(str(row["player_name"]).casefold().split())
    else:
        ident = "ID:" + ident
    return ident, team, pos


def _rolling_player_games(history, season, week, *, include_week=False,
                          prior_seasons=1, min_player_games=3, player_window=8):
    """One row per valid player-game, using only prior same-player team games.

    If include_week=True, the current game's actual enters the RESULT row,
    but NEVER that game's own baseline or any previous-week signal. This mode
    is exclusively for evaluating forecasts after games have completed.
    """
    if not 1 <= int(week) <= 23:
        raise ValueError("Invalid target week")
    if min_player_games < 2 or player_window < min_player_games:
        raise ValueError("Invalid prior-game sample requirement")
    h = clean_weekly(history)
    if "opponent_team" not in h:
        raise ValueError("Missing nflverse opponent_team: cannot verify opponent")
    h = h[(h["season"].ge(int(season)-int(prior_seasons))) &
          ((h["season"] < int(season)) |
           ((h["season"] == int(season)) &
            (h["week"] <= int(week) if include_week
             else h["week"] < int(week))))].copy()
    if h.empty:
        return pd.DataFrame()
    h = h.sort_values(["season", "week", "recent_team", "player_name"],
                      kind="stable")
    trailing = defaultdict(lambda: deque(maxlen=int(player_window)))
    seen = set()
    results = []
    for r in h.to_dict("records"):
        pos = str(r["position"]).upper()
        defense = canonical_team(r.get("opponent_team"))
        key = _identity(r)
        # One NFL player cannot play two games for the same team in a week.
        # Reject duplicates rather than letting a same-week outcome leak into
        # another supposed pregame baseline.
        unique = (int(r["season"]), int(r["week"]), key)
        if unique in seen:
            raise ValueError("Duplicate player/team/position week in NFL history")
        seen.add(unique)
        score = dk_classic_points(r, pos)
        if not np.isfinite(score):
            continue
        previous = trailing[key]
        if (pos in POSITIONS and defense and
                len(previous) >= min_player_games):
            vals = np.asarray(previous, dtype=float)
            weights = np.power(.87, np.arange(len(vals)-1, -1, -1))
            baseline = float(np.average(vals, weights=weights))
            results.append({
                "season": int(r["season"]), "week": int(r["week"]),
                "player": str(r["player_name"]), "team": key[1],
                "position": pos, "defense": defense,
                "player_prior_games": len(previous),
                "pregame_player_baseline_dk": baseline,
                "actual_partial_dk": float(score),
                "excess_dk": float(score - baseline),
            })
        previous.append(float(score))
    return pd.DataFrame(results)


def opponent_relative_profiles(history, season, week, *,
                               min_player_games=3, player_window=8,
                               defense_window=10, min_observations=8,
                               shrink_observations=30, max_residual=25.):
    """Build bounded, quality-of-opposition-adjusted positional defense signals.

    A strong player's usual 25-point game does not make a defense vulnerable
    simply because 25 points were conceded. Only excess OVER that player's
    own pregame baseline contributes. Defense has a max ten-game lookback;
    shrinkage is toward neutral excess 0, not raw fantasy points allowed.
    """
    if defense_window < 1 or min_observations < 1:
        raise ValueError("Invalid defense evidence threshold")
    if shrink_observations <= 0 or max_residual <= 0:
        raise ValueError("Invalid shrinkage or residual cap")
    obs = _rolling_player_games(
        history, season, week, prior_seasons=1,
        min_player_games=min_player_games, player_window=player_window)
    if obs.empty:
        return pd.DataFrame(columns=EMPTY)
    profiles = []
    for defense, team in obs.groupby("defense"):
        games = sorted(set(zip(team["season"].astype(int),
                               team["week"].astype(int))))
        use = set(games[-int(defense_window):])
        within = team[[ (int(x["season"]),int(x["week"])) in use
                       for x in team.to_dict("records")]]
        for pos, group in within.groupby("position"):
            n = len(group)
            if n < min_observations:
                continue
            raw_residual = group["excess_dk"].clip(
                lower=-max_residual, upper=max_residual)
            raw = float(raw_residual.mean())
            shrunk = float(n * raw / (n + shrink_observations))
            profiles.append({
                "defense": defense, "position": pos,
                "observations": int(n),
                "defense_games": int(group[["season", "week"]].drop_duplicates().shape[0]),
                "raw_excess_dk": round(raw, 4),
                "shrunk_excess_dk": round(shrunk, 4),
                "mean_player_baseline_dk": round(float(
                    group["pregame_player_baseline_dk"].mean()), 4),
                "mean_actual_dk": round(float(group["actual_partial_dk"].mean()), 4),
                "source": SOURCE,
                "target_season": int(season), "target_week": int(week),
            })
    return pd.DataFrame(profiles, columns=EMPTY)


def prior_player_baselines_for_grading(history, season, week, *,
                                      min_player_games=3, player_window=8):
    """Postgame grading only: actuals are present, baselines are pregame."""
    all_rows = _rolling_player_games(
        history, season, week, include_week=True, prior_seasons=1,
        min_player_games=min_player_games, player_window=player_window)
    if all_rows.empty:
        return all_rows
    return all_rows[
        all_rows["season"].eq(int(season)) &
        all_rows["week"].eq(int(week))
    ].reset_index(drop=True)
