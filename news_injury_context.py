"""Single source of truth for NFL confirmed-out simulation identities."""
from auto_injury_feed import norm_name, norm_team

def merge_confirmed_out(existing, automatic, enabled=True):
    """Merge confirmed automatic statuses with reviewed scenario exclusions.

    Return stable team/name pairs, no duplicates. Automatic items are only
    included when the injury scenario is enabled.
    """
    merged={}
    for p in existing or []:
        team=norm_team(p["team"])
        player=str(p["player"]).strip()
        if team and player:
            merged[(team,norm_name(player))]={"team":team,"player":player}
    if enabled:
        for p in automatic or []:
            team=norm_team(p["team"])
            player=str(p["player"]).strip()
            if team and player:
                merged.setdefault((team,norm_name(player)),{"team":team,"player":player})
    return [merged[k] for k in sorted(merged)]
