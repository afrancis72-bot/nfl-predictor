"""DraftKings Classic candidate export, with final legality validation.

Exports roster-slot ID columns for manual review; never submits entries.
"""
import csv
import io
from ai2_nfl_lineup import check_classic_lineup

SLOTS = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST")

def slot_classic_lineup(players):
    errors = check_classic_lineup(players)
    if errors:
        raise ValueError("Illegal NFL lineup: " + "; ".join(errors))
    by_position = {p: sorted(
        (row for row in players if str(row.get("Position", "")).upper() == p),
        key=lambda r: str(r["ID"])) for p in ("QB", "RB", "WR", "TE", "DST")}
    selected = []
    for position, amount in (("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1), ("DST", 1)):
        selected.extend(by_position[position][:amount])
    assigned = {str(p["ID"]) for p in selected}
    flex = [p for p in players if str(p["ID"]) not in assigned]
    if len(flex) != 1 or str(flex[0].get("Position", "")).upper() not in {"RB", "WR", "TE"}:
        raise ValueError("FLEX could not be assigned")
    return (by_position["QB"][0], *by_position["RB"][:2],
            *by_position["WR"][:3], by_position["TE"][0], flex[0],
            by_position["DST"][0])

def export_classic_candidates(candidates):
    if not candidates:
        raise ValueError("No candidates to export")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SLOTS)
    for candidate in candidates:
        writer.writerow([str(p["ID"]) for p in slot_classic_lineup(candidate["players"])])
    return buffer.getvalue()
