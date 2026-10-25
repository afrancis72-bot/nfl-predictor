"""Parse DraftKings Entries CSV files, including their appended salary-table rows."""
import csv
import io
import re

import pandas as pd


def parse_dk_entries(data: bytes | str) -> pd.DataFrame:
    if isinstance(data, bytes):
        text = data.decode("utf-8-sig")
    elif isinstance(data, str):
        text = data.lstrip("\ufeff")
    else:
        raise ValueError("DraftKings Entries input must be CSV text or bytes.")
    try:
        rows = csv.reader(io.StringIO(text, newline=""))
        header = next(rows)
    except (StopIteration, csv.Error) as exc:
        raise ValueError("DraftKings Entries CSV has no valid header.") from exc

    header = [str(c).strip() for c in header]
    if "Entry ID" not in header:
        raise ValueError("DraftKings Entries CSV is missing the Entry ID header.")
    if header.count("G") != 6:
        raise ValueError("DraftKings PGA Entries CSV must contain exactly six G roster columns.")
    if header.count("Entry ID") != 1:
        raise ValueError("DraftKings Entries CSV has duplicate Entry ID headers.")

    entry_idx = header.index("Entry ID")
    roster_start = [i for i, c in enumerate(header) if c == "G"]
    if len(header) <= max(roster_start):
        raise ValueError("Incomplete PGA roster columns.")

    # Original DraftKings exports can embed a separate salary table with more
    # columns below the entry list. These must never be treated as entries.
    selected = []
    for row in rows:
        if len(row) < max(roster_start) + 1:
            continue
        if entry_idx >= len(row):
            continue
        entry_id = str(row[entry_idx]).strip()
        if not re.fullmatch(r"\d+", entry_id):
            continue
        normalized = (list(row) + [""] * len(header))[:len(header)]
        selected.append(normalized)
    if not selected:
        raise ValueError("No valid entries found in the DraftKings Entries CSV.")

    # Maintain dataframe compatibility with pandas.read_csv() disambiguation.
    seen = {}
    unique_header = []
    for col in header:
        number = seen.get(col, 0)
        unique_header.append(f"{col}.{number}" if number else col)
        seen[col] = number + 1

    df = pd.DataFrame(selected, columns=unique_header, dtype=str)
    if df["Entry ID"].duplicated().any():
        raise ValueError("Duplicate Entry IDs in DraftKings Entries CSV.")
    return df
