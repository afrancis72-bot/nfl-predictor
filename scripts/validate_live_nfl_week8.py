"""Real Week 8 DraftKings acceptance: live nflverse + NFL official designations.

This workflow uses the actual uploaded DK salary slate. Unlike the offline
synthetic CI smoke tests, this must produce 20 validated *real-player* lineups.
It checks software function, not predictive accuracy, or contest profitability.
"""
from pathlib import Path
from collections import Counter
import csv
import io
import json
import hashlib
import traceback

from ai2_nfl_oneclick import build_oneclick_portfolio, _slate_week
from ai2_nfl_lineup import check_classic_lineup

SLATE = Path("tests/fixtures/2026_week8_user_dk_slate.csv")
OUTPUT = Path("artifacts/nfl_live_week8")

def main():
    contents = SLATE.read_bytes()
    saved = (SLATE.name, contents, "2026-10-28T22:00:00+00:00")
    rows = list(csv.DictReader(io.StringIO(contents.decode("utf-8-sig"))))
    print("Uploaded slate rows:",len(rows),
          "games:",len(set(r["Game Info"] for r in rows)),
          "year/week:",_slate_week(saved),
          "sha256:",hashlib.sha256(contents).hexdigest())
    try:
        result = build_oneclick_portfolio(saved, seed=42)
    except Exception:
        print("LIVE ACCEPTANCE FAILED - complete error follows")
        traceback.print_exc()
        raise
    assert len(result["ranked"]) == 20, "Must contain 20 complete lineups"
    assert len(result["summary"]) == 20
    official_ids = {str(row["ID"]) for row in rows}
    appearances = Counter()
    for lineup in result["ranked"]:
        players = lineup["candidate"]["players"]
        assert len(players) == 9
        assert check_classic_lineup(players, salary_floor=49600) == [], "Illegal roster"
        for p in players:
            assert str(p["ID"]) in official_ids, "Non-DraftKings player was selected"
            appearances[str(p["ID"])] += 1
    assert max(appearances.values()) <= 7, "Overexposure"
    roster_csv = list(csv.reader(io.StringIO(result["csv"])))
    assert len(roster_csv) == 21, "Invalid DraftKings roster CSV"
    assert sum(item["contest_tier"] == "$3" for item in result["ranked"]) == 5
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "real_week8_lineups_REVIEW_ONLY.csv").write_text(result["csv"])
    (OUTPUT / "real_week8_summary.json").write_text(json.dumps(result["summary"],indent=2))
    (OUTPUT / "real_week8_audit.json").write_text(json.dumps(result["audit"],indent=2))
    print("SUCCESS: 20/20 lineups, player IDs matched, 5 premium, salary and exposure checks passed")
    print("Audit:",result["audit"])
    print("Average simulated points:",
          round(sum(item["Mean"] for item in result["summary"]) / 20,2))

if __name__ == "__main__":
    main()
