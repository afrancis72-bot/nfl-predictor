"""NFL Classic portfolio guard for negatively correlated DST/pass-catcher pairings.

A receiver facing the rostered DST is not an illegal DraftKings lineup, but
mass-producing that combination creates avoidable portfolio concentration.
Restrict it at final portfolio selection; retain rare lineup-level exceptions.
Unknown opponent metadata is not silently treated as a verified matchup.
"""
from ai2_nfl_matchups import canonical_team, opponent_from_game_info

PASS_CATCHERS = frozenset(("WR", "TE"))
MAX_OPPOSING_PASS_CATCHER_DST_SHARE = 0.04


def _opponent(player):
    raw = str(player.get("opponent") or "").strip()
    if raw and raw.upper() not in {"NAN", "NONE", "<NA>", "UNKNOWN"}:
        return canonical_team(raw)
    return opponent_from_game_info(player.get("TeamAbbrev"),
                                   player.get("Game Info") or player.get("game") or "")


def opposing_pass_catcher_dst_pairs(roster):
    """Return validated opposing WR/TE vs DST conflicts, with names and IDs."""
    dsts = [p for p in roster if str(p.get("Position", "")).upper() == "DST"]
    if not dsts:
        return []
    pairs = []
    for dst in dsts:
        defense = canonical_team(dst.get("TeamAbbrev"))
        opposing_team = _opponent(dst)
        if not defense:
            continue
        for player in roster:
            if str(player.get("Position", "")).upper() not in PASS_CATCHERS:
                continue
            offense = canonical_team(player.get("TeamAbbrev"))
            if not offense or offense == defense:
                continue
            if (opposing_team and offense == opposing_team) or (
                    _opponent(player) and _opponent(player) == defense):
                pairs.append({
                    "player_id": str(player.get("ID", "")),
                    "player": str(player.get("Name", player.get("ID", "Unknown"))),
                    "offense_team": offense,
                    "defense_team": defense,
                    "defense": str(dst.get("Name", dst.get("ID", "DST"))),
                })
    return pairs


def opposing_dst_lineup_budget(lineups):
    """No more than 4% of a Classic portfolio may have a DST/WR-TE conflict.

    For small portfolios (e.g. 15 or 20) that permits none; at 50, two.
    """
    return max(0, int(int(lineups) * MAX_OPPOSING_PASS_CATCHER_DST_SHARE))


def correlation_audit_rows(ranked):
    """One row per final lineup; expose conflicts instead of hiding them."""
    rows = []
    for position, entry in enumerate(ranked, 1):
        roster = entry.get("candidate", entry).get("players", [])
        pairs = opposing_pass_catcher_dst_pairs(roster)
        rows.append({
            "Lineup": position,
            "Conflicted": bool(pairs),
            "Opposing WR/TE vs DST pairs": len(pairs),
            "Players facing rostered DST": "; ".join(
                f"{p['player']} ({p['offense_team']}) vs {p['defense']} ({p['defense_team']})"
                for p in pairs),
        })
    return rows
