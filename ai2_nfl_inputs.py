"""Read saved NFL CSV inputs without executing the legacy optimizer."""
import csv
import io
from ai2_nfl_flags import verified_flag
from ai2_nfl_matchups import opponent_from_game_info
from datetime import datetime, timezone, timedelta

MAX_BYTES = 8 * 1024 * 1024

def parse_csv_payload(payload: bytes):
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError("CSV must be nonempty and at most 8 MiB")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be UTF-8 encoded") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("CSV header missing or duplicated")
    rows = list(reader)
    if len(rows) > 10000:
        raise ValueError("CSV row limit exceeded")
    return rows

def parse_timestamp(value: str):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid saved upload timestamp") from exc
    if stamp.tzinfo is None:
        raise ValueError("Upload timestamp lacks timezone")
    return stamp

def inspect_saved_nfl_inputs(salaries, projections):
    """Return rows and reasons; never certify data merely from upload time."""
    issues = []
    if not salaries:
        issues.append("DraftKings salaries not saved")
    if not projections:
        issues.append("Verified weekly projection file not saved")
    if issues:
        return [], issues
    salary_rows = parse_csv_payload(salaries[1])
    projection_rows = parse_csv_payload(projections[1])
    salary_ids = {str(r.get("ID", "")).strip() for r in salary_rows if str(r.get("ID", "")).strip()}
    if not salary_ids:
        issues.append("Salary CSV lacks nonempty DraftKings IDs")
    if not projection_rows:
        issues.append("Projection CSV has no player rows")
    if any(not str(r.get("ID", "")).strip() for r in projection_rows):
        issues.append("Projection rows must include DraftKings ID")
    if len({str(r.get("ID", "")).strip() for r in projection_rows}) != len(projection_rows):
        issues.append("Projection IDs are duplicated")
    unknown = [r.get("ID") for r in projection_rows if str(r.get("ID", "")).strip() not in salary_ids]
    if unknown:
        issues.append(f"{len(unknown)} projection IDs are absent from saved salary slate")
    if issues:
        return [], issues
    if len(salary_ids) != len([r for r in salary_rows if str(r.get("ID", "")).strip()]):
        return [], ["Salary CSV has duplicated DraftKings IDs"]
    by_id = {str(r["ID"]).strip(): r for r in salary_rows}
    merged = []
    for row in projection_rows:
        base = by_id[str(row["ID"]).strip()]
        merged.append({**row, **{k: base.get(k) for k in ("Name", "Position", "TeamAbbrev", "Salary", "Status", "Game Info")},
                       "dk_status": base.get("Status", "")})
    for row in merged:
        if str(row.get("Position", "")).upper().strip() in {"D/ST", "DEF"}:
            row["Position"] = "DST"
        game_info = str(row.get("Game Info") or "").strip()
        row["game"] = game_info.split()[0] if game_info else ""
        row["opponent"] = opponent_from_game_info(row.get("TeamAbbrev"), game_info)
        if row["opponent"] is None:
            row["game"] = ""
        row['weekly_projection_valid'] = verified_flag(row.get('weekly_projection_valid'))
        row['opportunity_verified'] = verified_flag(row.get('opportunity_verified'))
    return merged, []
