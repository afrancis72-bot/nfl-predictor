"""Observed NFL offensive snaps and pregame player-opportunity projections.

Pro Football Reference game-level offensive snap counts via nflverse.
Routes run are NOT provided by this feed. No missing snap measurements are
invented and no current/target-week game observations enter projections.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ai2_nfl_matchups import canonical_team
from ai2_nfl_auto_baselines import _norm_name
from props_data import clean_weekly

SNAP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
            "snap_counts/snap_counts_{season}.csv")
SKILL_POS = frozenset(("QB", "RB", "WR", "TE"))


def fetch_snap_counts(seasons, *, reader=None):
    reader = reader or pd.read_csv
    frames = []
    for season in sorted(set(int(s) for s in seasons)):
        data = reader(SNAP_URL.format(season=season))
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError(f"No real nflverse snap observations for {season}")
        frames.append(data.assign(season=season))
    return pd.concat(frames, ignore_index=True)


def normalize_snap_counts(raw, season, week):
    required = {"season", "week", "player", "team", "position", "offense_snaps", "offense_pct"}
    if not isinstance(raw, pd.DataFrame) or not required.issubset(raw.columns):
        raise ValueError("Snap feed missing observed player/week/offense participation fields")
    s = raw.copy()
    s["season"] = pd.to_numeric(s["season"], errors="coerce")
    s["week"] = pd.to_numeric(s["week"], errors="coerce")
    s = s[((s["season"] < season) | ((s["season"] == season) & (s["week"] < week))) &
          (s["season"] >= season - 1)].copy()
    if "game_type" in s.columns:
        s = s[s["game_type"].astype(str).str.upper().eq("REG")].copy()
    s["name"] = s["player"].map(_norm_name)
    s["team"] = s["team"].map(canonical_team)
    s["position"] = s["position"].astype(str).str.upper().str.strip()
    s["offense_snaps"] = pd.to_numeric(s["offense_snaps"], errors="coerce")
    s["offense_pct"] = pd.to_numeric(s["offense_pct"], errors="coerce")
    # PFR percentages may be 0-100 or 0-1 depending on reader/version.
    positive = s["offense_pct"].dropna()
    if not positive.empty and positive.max() <= 1.0:
        s["offense_pct"] *= 100.
    s = s[(s["position"].isin(SKILL_POS)) & (s["name"].ne("")) &
          s["offense_snaps"].between(1, 125) &
          s["offense_pct"].between(1, 100)].copy()
    key = ["season", "week", "name", "team", "position"]
    s = s.drop_duplicates(subset=key, keep=False)
    return s


def predict_snap_roles(history, snaps, season, week, team_environment=None,
                       *, min_games=4, weight=.35):
    """Estimate offensive opportunities from actual snaps * historical usage/snap.

    Projection is a blend of baseline opportunity and snap-conditioned role,
    not a fabricated route count.  weight 0 means no scoring adjustment.
    """
    if not (0 <= weight <= 1):
        raise ValueError("Snap-role blend weight must be [0,1]")
    h = clean_weekly(history)
    h = h[(h["season"].eq(int(season))) & h["week"].lt(int(week))].copy()
    h["name"] = h["player_name"].map(_norm_name)
    h["team"] = h["recent_team"].map(canonical_team)
    h["position"] = h["position"].astype(str).str.upper()
    s = normalize_snap_counts(snaps, int(season), int(week))
    join = ["season", "week", "name", "team", "position"]
    # Reject ambiguous identities, never allow a many-to-many merge.
    h = h.drop_duplicates(subset=join, keep=False)
    matched = h.merge(s[join+["offense_snaps", "offense_pct"]], on=join,
                      how="inner", validate="one_to_one")
    output = {}
    team_environment = team_environment or {}
    for (name, team, pos), group in matched.groupby(["name", "team", "position"]):
        g = group.sort_values("week").tail(8)
        if len(g) < min_games:
            continue
        snaps_used = g["offense_snaps"].to_numpy(float)
        if np.any(snaps_used < 1):
            continue
        pct = g["offense_pct"].to_numpy(float) / 100.
        recent_pct = float(np.mean(pct[-3:]))
        earlier_pct = float(np.mean(pct[:-3])) if len(pct) > 3 else float("nan")
        # Use the complete real offensive-snap count as the denominator
        # and historical individual play involvement as the numerator.
        # Stable when all records are from same team/current season.
        attempts = {}
        for kind, field in (("targets", "targets"), ("carries", "carries")):
            counts = pd.to_numeric(g[field], errors="coerce")
            if counts.notna().all() and not np.any(counts < 0):
                # Recent games weighted more, avoiding an arbitrary matchup boost.
                w = np.power(.87, np.arange(len(g)-1, -1, -1))
                usage_per_snap = float(np.average(counts.to_numpy(float) / snaps_used, weights=w))
                # Expected snaps from today's team pace and prior-game participation.
                team_plays = (team_environment.get(team) or {}).get("expected_plays")
                if team_plays is None:
                    estimated_team_plays = float(np.average(snaps_used / np.maximum(pct, .01), weights=w))
                else:
                    estimated_team_plays = float(team_plays)
                expected_snaps = float(np.clip(recent_pct * estimated_team_plays, 0, 90))
                attempts[kind] = max(0., usage_per_snap * expected_snaps)
        if not attempts:
            continue
        output[(name, team, pos)] = {
            "expected_targets": attempts.get("targets"),
            "expected_carries": attempts.get("carries"),
            "recent_snap_pct": round(recent_pct * 100., 2),
            "earlier_snap_pct": round(earlier_pct * 100., 2),
            "expected_offensive_snaps": round(
                recent_pct * float((team_environment.get(team) or {}).get(
                    "expected_plays", np.mean(snaps_used / np.maximum(pct, .01)))), 2),
            "snap_games": len(g), "last_observed_week": int(g.iloc[-1]["week"]),
            "last_observed_season": int(g.iloc[-1]["season"]),
            "source": "nflverse PFR prior-week offense snaps + observed targets/carries",
            "blend_weight": weight,
            "routes_verified": False}
    return output


def apply_snap_roles_to_rates(target_rates, carry_rates, skills, snap_profiles):
    """Change actual simulator per-player targets/carries, capped to avoid shocks.

    All changes happen *before* team budgets and OUT redistribution; the game
    simulator continues to conserve team opportunities in each scenario.
    """
    if not snap_profiles:
        return target_rates, carry_rates, []
    target = np.asarray(target_rates, dtype=float).copy()
    carry = np.asarray(carry_rates, dtype=float).copy()
    changes = []
    for i, player in enumerate(skills.to_dict("records")):
        key = (_norm_name(player.get("Name","")),
               canonical_team(player.get("TeamAbbrev","")),
               str(player.get("Position","")).upper())
        evidence = snap_profiles.get(key)
        if not evidence:
            continue
        weight=float(evidence.get("blend_weight",0))
        if not 0 <= weight <= 1 or int(evidence.get("snap_games",0)) < 4:
            continue
        change={"player":player.get("Name"),"team":key[1],
                "recent_snap_pct":evidence["recent_snap_pct"],
                "historical_snap_pct":evidence["earlier_snap_pct"],
                "games":evidence["snap_games"],"routes_verified":False,
                "target_before":float(target[i]),"carry_before":float(carry[i])}
        # Applying the same observed opportunity ratio to receiving targets
        # and rushing carries would double-count role changes. Instead use
        # per-snap actual historical target/carry rates separately.
        for rates,field in ((target,"expected_targets"),(carry,"expected_carries")):
            expected=evidence.get(field)
            if expected is not None and np.isfinite(float(expected)) and rates[i] > 0:
                blended=(1.-weight)*rates[i]+weight*float(expected)
                rates[i]=np.clip(blended,.75*rates[i],1.25*rates[i])
        change["target_after"]=float(target[i])
        change["carry_after"]=float(carry[i])
        changes.append(change)
    return target, carry, changes
