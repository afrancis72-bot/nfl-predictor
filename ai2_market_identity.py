"""Explicit, reviewable athlete mapping for post-simulation props.

No fuzzy matching. Candidate mappings require exact sport/event and player
identity; ambiguous names are never silently resolved.
"""
import csv
import io

def parse_identity_csv(payload):
    if not isinstance(payload, bytes) or not payload or len(payload) > 2_000_000:
        raise ValueError("Identity CSV must be nonempty and at most 2 MB")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Identity CSV must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    expected = {"event_id", "player_name", "player_id"}
    if not reader.fieldnames or not expected.issubset(reader.fieldnames):
        raise ValueError("Identity CSV needs event_id, player_name, player_id")
    mapping = {}
    reverse = {}
    for index, row in enumerate(reader, 1):
        if index > 5000:
            raise ValueError("Identity CSV exceeds 5000 rows")
        event, name, player_id = (str(row.get(k) or "").strip()
                                  for k in ("event_id", "player_name", "player_id"))
        if not event or not name or not player_id:
            raise ValueError(f"Identity row {index} has empty fields")
        key = (event, name)
        if key in mapping:
            raise ValueError(f"Duplicate athlete mapping for event/name at row {index}")
        reverse_key = (event, player_id)
        if reverse_key in reverse:
            raise ValueError(f"Multiple names map to one athlete ID in event at row {index}")
        mapping[key] = player_id
        reverse[reverse_key] = name
    if not mapping:
        raise ValueError("Identity CSV has no mappings")
    return mapping

def identity_coverage(board, identity_map):
    """Report matched and unmapped athlete records without guessing."""
    total = matched = 0
    unmatched = set()
    for row in board:
        if not isinstance(row, dict):
            continue
        total += 1
        key = (str(row.get("event_id") or ""), str(row.get("player_name") or ""))
        if key in identity_map:
            matched += 1
        else:
            unmatched.add(key)
    return {"records": total, "matched": matched,
            "unmatched": total - matched, "unmapped_athletes": sorted(unmatched)}
