"""Strict, auditable PGA weekly joins for DK, OTIS and measured golfer statistics.

No future results and no silently discarded explicit stat uploads. DK display names
are authoritative for the current field; aliases are used for matching only.
"""
from __future__ import annotations

import re
import unicodedata
import pandas as pd

_ALIAS = {
    "benjaminjames": "benjames",
    "kristofferventura": "krisventura",
    "camerondavis": "camdavis",
    "jordanlsmith": "jordansmith",
    "zacharybauchou": "zachbauchou",
    # Official PGA TOUR profile / cross-provider field-name aliases.
    "richardhoey": "ricohoey",
    "samuelstevens": "samstevens",
}

_STAT_COLUMNS = {
    "sgott": "sg_ott", "sgapp": "sg_approach", "sgapproach": "sg_approach",
    "sgarg": "sg_arg", "sgputt": "sg_putting", "sgputting": "sg_putting",
    "sgtot": "sg_total", "sgtotal": "sg_total", "rounds": "measured_rounds",
    "measuredrounds": "measured_rounds",
    "drivingdistance": "driving_distance",
    "drivingaccuracy": "driving_accuracy",
    "birdierate": "birdie_rate", "birdieorbetterrate": "birdie_rate",
    "bogeyavoidance": "bogey_avoidance", "girrate": "gir_rate",
    "gir": "gir_rate", "girpct": "gir_rate",
    "approach175200": "approach_175_200",
    "approach200225": "approach_200_225",
    "par3": "par3", "par4": "par4", "par5": "par5",
    "formsgtotal": "form_sg_total",
    "formsgapproach": "form_sg_approach",
    "formsgott": "form_sg_ott",
    "formsgarg": "form_sg_arg",
    "formsgputting": "form_sg_putting",
    "formdrivingdistance": "form_driving_distance",
    "formdrivingaccuracy": "form_driving_accuracy",
}
SG_COLUMNS = ("sg_ott", "sg_approach", "sg_arg", "sg_putting", "sg_total")


def player_key(name: object) -> str:
    """Canonical identity for pre-event joins, including Last, First exports."""
    if name is None or pd.isna(name):
        return ""
    name = unicodedata.normalize("NFKD", str(name).strip())
    if "," in name:
        last, first = name.split(",", 1)
        name = first.strip() + " " + last.strip()
    name = "".join(c for c in name if not unicodedata.combining(c))
    key = "".join(c for c in name.casefold() if c.isalnum())
    return _ALIAS.get(key, key)


def _dk_lookup(players: pd.DataFrame) -> dict[str, str]:
    if "player" not in players:
        raise ValueError("DraftKings field requires player names")
    lookup = {}
    for name in players["player"].dropna().astype(str):
        key = player_key(name)
        if not key:
            continue
        if key in lookup and lookup[key] != name:
            raise ValueError(f"Ambiguous DraftKings player identity: {lookup[key]!r} / {name!r}")
        lookup[key] = name
    if not lookup:
        raise ValueError("Empty DraftKings player field")
    return lookup


def align_to_dk(frame: pd.DataFrame, players: pd.DataFrame, label: str):
    if frame is None or frame.empty or "player" not in frame:
        raise ValueError(f"{label}: nonempty player column required")
    source = frame.copy()
    source["player"] = source["player"].astype(str).str.strip()
    source["_match_key"] = source["player"].map(player_key)
    source = source[source["_match_key"].ne("")].copy()
    duplicate = source[source["_match_key"].duplicated(keep=False)]
    if not duplicate.empty:
        raise ValueError(f"{label}: duplicate/colliding player names: "
                         + ", ".join(duplicate["player"].astype(str).head(8)))
    lookup = _dk_lookup(players)
    source["_dk_name"] = source["_match_key"].map(lookup)
    matched = source[source["_dk_name"].notna()].copy()
    unmatched = source.loc[source["_dk_name"].isna(), "player"].tolist()
    matched["player"] = matched.pop("_dk_name")
    matched = matched.drop(columns=["_match_key"]).reset_index(drop=True)
    return matched, {"matched": len(matched), "unmatched": unmatched,
                     "source_rows": len(source)}


def normalize_measured_stats(raw: pd.DataFrame, label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError(f"{label}: uploaded data is empty")
    clean = raw.copy()
    cols = {re.sub(r"[^a-z0-9]", "", str(c).casefold()): c for c in clean.columns}
    who = next((cols[c] for c in ("player", "name", "playername", "golfer")
                if c in cols), None)
    if who is None:
        raise ValueError(f"{label}: missing golfer Name/Player column")
    out = pd.DataFrame({"player": clean[who].astype(str).str.strip()})
    for source_key, destination in _STAT_COLUMNS.items():
        if source_key not in cols:
            continue
        values = pd.to_numeric(clean[cols[source_key]], errors="coerce")
        if destination in out:
            out[destination] = out[destination].combine_first(values)
        else:
            out[destination] = values
    if len(out.columns) <= 1:
        raise ValueError(f"{label}: no measured statistic columns recognized; "
                         "expected SG APP, SG OTT, SG ARG, SG PUTT, SG TOT or equivalent")
    useful = out.drop(columns=["player"]).notna().any(axis=1)
    out = out.loc[useful & out["player"].ne("")].copy()
    if out.empty:
        raise ValueError(f"{label}: no usable numeric golfer measurements")
    return out.reset_index(drop=True)


def build_weekly_player_stats(
    players: pd.DataFrame,
    otis_fit: pd.DataFrame,
    measured_override: pd.DataFrame | None = None,
    rrg_last50: pd.DataFrame | None = None,
    min_otis_coverage: float = 0.70,
):
    """Combine rather than replace OTIS with measured SG; never silently discard uploads."""
    otis, otis_report = align_to_dk(otis_fit, players, "OTIS")
    required = len(_dk_lookup(players))
    if otis_report["matched"] / required < min_otis_coverage:
        raise ValueError(f"OTIS player match: {otis_report['matched']}/{required}; "
                         f"below required {min_otis_coverage:.0%}")
    baseline = otis.copy()
    audit = {"otis": otis_report, "sources": {}}

    for label, raw in (("RunGood last-50 SG", rrg_last50),
                       ("Manual measured golfer stats", measured_override)):
        if raw is None:
            continue
        measured = normalize_measured_stats(raw, label)
        matched, report = align_to_dk(measured, players, label)
        if matched.empty:
            raise ValueError(f"{label}: zero golfer matches against the current DraftKings field")
        if label == "RunGood last-50 SG":
            if not any(c in matched and matched[c].notna().any() for c in SG_COLUMNS):
                raise ValueError("RunGood last-50 file has no recognized numeric strokes-gained categories")
        audit["sources"][label] = report
        data = matched.set_index("player")
        for column in data.columns:
            values = baseline["player"].map(data[column])
            if column in baseline:
                baseline[column] = values.combine_first(baseline[column])
            else:
                baseline[column] = values
    audit["measured_sg_coverage"] = int(
        baseline[[c for c in SG_COLUMNS if c in baseline]].notna().any(axis=1).sum()
    ) if any(c in baseline for c in SG_COLUMNS) else 0
    audit["field_size"] = required
    return baseline.reset_index(drop=True), audit
