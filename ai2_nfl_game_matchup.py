"""Research-only pregame opponent effects on team scoring and offensive plays.

Each past opposing offense is compared with ITS OWN pregame team baseline,
avoiding the trap of treating a defense as weak simply because it faced stars.
Game/week data are consumed in chronological order, with no target-week rows
eligible for the prediction.

Effects are bounded, shrunk toward zero, and never turned on by the unified
production one-click application without a separate paired historical audit.
This is a box-score proxy for team offense, NOT a true drive/play-by-play model.
"""
from __future__ import annotations

from collections import defaultdict, deque
import numpy as np
import pandas as pd

from ai2_nfl_matchups import canonical_team
from props_data import clean_weekly

SOURCE = "nflverse pregame offense-vs-own-history team TDs and play-volume residuals"
WINDOW = 8
RECENCY = .85


def _past_team_games(history, season: int, week: int) -> pd.DataFrame:
    frame = clean_weekly(history)
    if "opponent_team" not in frame:
        raise ValueError("Cannot assess game opponent without actual opponent_team history")
    frame = frame[((frame["season"] >= int(season) - 1) &
                   ((frame["season"] < int(season)) |
                    ((frame["season"] == int(season)) & (frame["week"] < int(week)))))]
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (yr, wk, off), grp in frame.groupby(["season", "week", "recent_team"],
                                            sort=True):
        off = canonical_team(off)
        defenses = {canonical_team(x) for x in grp["opponent_team"].dropna()
                    if str(x).upper() not in ("", "NAN", "NONE", "UNKNOWN")}
        if len(defenses) != 1:
            continue
        defense = defenses.pop()
        if not defense or defense == off:
            continue
        qbs = grp[grp["position"].astype(str).str.upper().eq("QB")]
        if qbs.empty:
            continue
        qb_attempts = pd.to_numeric(qbs["attempts"], errors="coerce")
        sacks = pd.to_numeric(qbs["sacks_suffered"], errors="coerce")
        rushes = pd.to_numeric(grp["carries"], errors="coerce")
        pass_tds = pd.to_numeric(qbs["passing_tds"], errors="coerce")
        rush_tds = pd.to_numeric(grp["rushing_tds"], errors="coerce")
        if (not qb_attempts.notna().any() or not rushes.notna().any() or
                not pass_tds.notna().any() or not rush_tds.notna().any()):
            continue
        n_plays = (float(qb_attempts.sum()) +
                   float(sacks.sum()) if sacks.notna().any() else
                   float(qb_attempts.sum()) + 2.0)
        n_plays += float(rushes.sum())
        tds = float(pass_tds.sum() + rush_tds.sum())
        if (not np.isfinite([n_plays, tds]).all() or
                not 35 <= n_plays <= 100 or not 0 <= tds <= 9):
            continue
        rows.append({
            "season": int(yr), "week": int(wk), "offense": off,
            "defense": defense, "plays": n_plays, "offensive_tds": tds,
        })
    return pd.DataFrame(rows)


def _weighted(values):
    a = np.asarray(values, float)
    w = RECENCY ** np.arange(len(a)-1, -1, -1)
    return float(np.average(a, weights=w))


def team_defense_game_profiles(history, season: int, week: int,
                               *, min_defense_games: int = 5):
    """Estimate defense's *residual* effect on another team's plays and TDs.

    A past game enters the defense record only if the *offense* had three
    earlier games from which its own scoring/play expectation can be computed.
    The target week is excluded at source. Profiles carry strict provenance.
    """
    if min_defense_games < 3:
        raise ValueError("Minimum observed opponent games must be >= 3")
    raw = _past_team_games(history, season, week)
    if raw.empty:
        return {}
    by_off = defaultdict(lambda: deque(maxlen=WINDOW))
    by_def = defaultdict(lambda: deque(maxlen=12))
    # Process all clubs' same-week observations against prior-week histories;
    # no intra-week leakage or use of the target Sunday's box scores.
    for (_season, _week), games in raw.groupby(["season", "week"], sort=True):
        pending = []
        for row in games.to_dict("records"):
            past = by_off[row["offense"]]
            if len(past) >= 3:
                mean_plays = _weighted([x["plays"] for x in past])
                mean_tds = _weighted([x["offensive_tds"] for x in past])
                # Adjust the touchdown expectation for this game's actual
                # volume, to separate efficiency from pace effects.
                normal_tds = mean_tds * row["plays"] / max(mean_plays, 1.)
                td_excess = float(np.clip(
                    row["offensive_tds"] - normal_tds, -2.5, 2.5))
                plays_excess = float(np.clip(
                    row["plays"] - mean_plays, -20., 20.))
                pending.append((row["defense"], td_excess, plays_excess,
                                int(row["season"]), int(row["week"])))
        for row in games.to_dict("records"):
            by_off[row["offense"]].append(row)
        for defense, td, plays, yr, wk in pending:
            by_def[defense].append((td, plays, yr, wk))
    output = {}
    for team, records in by_def.items():
        sample = list(records)
        if len(sample) < min_defense_games:
            continue
        n = len(sample)
        reliability = n / (n + 10.)
        tds = float(np.clip(
            reliability * _weighted([z[0] for z in sample]), -.7, .7))
        plays = float(np.clip(
            reliability * _weighted([z[1] for z in sample]), -6., 6.))
        output[team] = {
            "defense": team,
            "td_delta": round(tds, 5),
            "plays_delta": round(plays, 5),
            "sample_games": n,
            "last_observed_season": sample[-1][3],
            "last_observed_week": sample[-1][4],
            "target_season": int(season),
            "target_week": int(week),
            "source": SOURCE,
            "description": "shrunk prior-opponent own-baseline TD and plays residual",
        }
    return output


def validated_game_profile_lookup(profiles, *, season, week, strength=0.):
    """Reject spoofed, invalid or future-week team scoring assumptions."""
    if not np.isfinite(strength) or not 0. <= strength <= 1.:
        raise ValueError("Opponent game strength must be in [0,1]")
    if strength == 0:
        return {}
    if season is None or week is None:
        raise ValueError("Opponent game effects require pregame season and week")
    if not isinstance(profiles, dict):
        raise ValueError("Active opponent game effects require measured team profiles")
    output = {}
    for team, row in profiles.items():
        if not isinstance(row, dict) or row.get("source") != SOURCE:
            raise ValueError("Unsourced opponent team-game effect")
        if (row.get("target_season") != int(season) or
                row.get("target_week") != int(week)):
            raise ValueError("Wrong or future-week team defense evidence")
        defense = canonical_team(team)
        if not defense or defense != canonical_team(row.get("defense")):
            raise ValueError("Mismatched opponent profile team")
        n = float(row.get("sample_games", 0))
        td, plays = float(row.get("td_delta", np.nan)), float(
            row.get("plays_delta", np.nan))
        if (not np.isfinite([n, td, plays]).all() or n < 5 or
                abs(td) > .701 or abs(plays) > 6.01):
            raise ValueError("Invalid or unsupported defense scenario effect")
        if row.get("last_observed_season") == int(season) and (
                int(row.get("last_observed_week", 99)) >= int(week)):
            raise ValueError("Target-week defense results leaked into pregame")
        if int(row.get("last_observed_season", -1)) > int(season):
            raise ValueError("Future season defense data")
        output[defense] = (float(strength) * td, float(strength) * plays)
    return output


def get_game_effect(lookup, opponent):
    return lookup.get(canonical_team(opponent), (0., 0.))
