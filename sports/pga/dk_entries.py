"""Validate a PGA portfolio against a DraftKings Entries CSV before filling it.

Preserves the user's actual contest, entry IDs, and six roster-column positions.
Never invents DraftKings player IDs or entry metadata.
"""
import re

import pandas as pd


def fill_entries_template(portfolio: pd.DataFrame, dk_players: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    required = {"lineup", "player", "salary"}
    if not required.issubset(portfolio.columns):
        raise ValueError("Portfolio must contain lineup, player and salary columns.")
    if not {"player", "ID", "salary"}.issubset(dk_players.columns):
        raise ValueError("Upload the native DraftKings PGA salary file containing player IDs.")
    if "Entry ID" not in entries.columns:
        raise ValueError("Use your DraftKings Entries CSV with Entry ID and six G slots.")

    # Pandas changes repeated G headers to G.1, G.2, ... on read.
    slot_indices = [
        i for i, col in enumerate(entries.columns)
        if re.fullmatch(r"G(?:\.[1-9][0-9]*)?", str(col)) is not None
    ]
    if len(slot_indices) != 6:
        raise ValueError("Expected exactly six golfer slots (G) in the PGA Entries CSV.")

    dk = dk_players.copy()
    dk["__name"] = dk["player"].astype(str).str.strip().str.casefold()
    if dk["__name"].duplicated().any():
        raise ValueError("Duplicate golfer names in salary file; player ID lookup is ambiguous.")
    dk["__id"] = dk["ID"].astype(str).str.strip()
    if not dk["__id"].str.fullmatch(r"\d+(?:\.0)?").fillna(False).all():
        raise ValueError("DraftKings golfer ID is missing or malformed in salary file.")
    dk["__id"] = dk["__id"].str.replace(r"\.0$", "", regex=True)
    if dk["__id"].duplicated().any():
        raise ValueError("Duplicate DraftKings golfer IDs in salary file.")

    dk["__salary"] = pd.to_numeric(dk["salary"], errors="coerce")
    if dk["__salary"].isna().any() or (dk["__salary"] <= 0).any():
        raise ValueError("Current DraftKings salary file contains invalid salaries.")
    lookup = dict(zip(
        dk["__name"],
        zip(dk["player"].astype(str), dk["__id"], dk["__salary"])
    ))
    roster = portfolio.copy()
    roster["__name"] = roster["player"].astype(str).str.strip().str.casefold()
    roster["salary"] = pd.to_numeric(roster["salary"], errors="coerce")
    if roster["salary"].isna().any():
        raise ValueError("Portfolio includes an invalid salary.")
    lineups = []
    for _, group in roster.groupby("lineup", sort=True):
        if len(group) != 6 or group["__name"].nunique() != 6:
            raise ValueError("Every PGA lineup must contain exactly six distinct golfers.")
        if group["salary"].sum() > 50000 or group["salary"].sum() <= 0:
            raise ValueError("A PGA lineup violates the $50,000 salary cap.")
        unmatched = sorted(set(group["__name"]) - set(lookup))
        if unmatched:
            raise ValueError("Golfers missing from current DraftKings salary file: " + ", ".join(unmatched))
        for _, golfer in group.iterrows():
            actual_salary = lookup[golfer["__name"]][2]
            if int(golfer["salary"]) != int(actual_salary):
                raise ValueError(
                    "Saved lineup salary differs from this week's DraftKings salary "
                    f"for {golfer['player']}. Rebuild the PGA model using the current slate."
                )
        lineups.append([f"{lookup[n][0]} ({lookup[n][1]})" for n in group["__name"]])

    if not lineups:
        raise ValueError("No PGA lineups to export.")
    if len(lineups) > len(entries):
        raise ValueError("There are more lineups than available DraftKings entries.")
    if entries["Entry ID"].astype(str).duplicated().any():
        raise ValueError("Duplicate DraftKings Entry IDs in template.")
    if entries.iloc[:len(lineups)]["Entry ID"].isna().any():
        raise ValueError("DraftKings Entries CSV has blank Entry ID values.")
    result = entries.copy()
    for row_idx, golfers in enumerate(lineups):
        for col_idx, golfer in zip(slot_indices, golfers):
            result.iat[row_idx, col_idx] = golfer
    # Restore repeated DK slot headers if pandas suffixed them.
    columns = list(result.columns)
    for col_idx in slot_indices:
        columns[col_idx] = "G"
    result.columns = columns
    return result
