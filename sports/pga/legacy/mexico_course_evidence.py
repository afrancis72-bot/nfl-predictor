"""Evidence ledger for the 2026 VidantaWorld Mexico Open course thesis.

Manually transcribed from the user's two RickRunGood course-stat screenshots
provided 2026-10-26. The screenshot reports an association-strength number
and a displayed relative rank ("#N/6"). Neither is an optimized model
coefficient, a player metric, a percentage, nor a probability. The underlying
data population and significance level were NOT provided.

Keep this distinct from RickRunGood golfer-level FIT rankings and from L50
observed golfer strokes-gained inputs.
"""
from __future__ import annotations

import pandas as pd

SOURCE = "RickRunGood course-stat chart, user screenshots received 2026-10-26"
PROFILE_ID = "vidantaworld_mexico_open_2026_rrg_evidence"

# (label, reported chart value, chart position among six, group,
#  observed golfer data currently available from L50 upload?)
_EVIDENCE = [
    ("SG: Total", .437, 2, "Overall", "Available, overlaps SG components"),
    ("Scoring Average", .427, 2, "Overall", "Not present in L50"),
    ("SG: Tee-to-Green", .381, 2, "Overall", "Available, overlaps SG components"),
    ("SG: Approach", .354, 2, "Approach", "Available"),
    ("Scoring Average (Actual)", .330, 3, "Overall", "Not present in L50"),
    ("Approach: 175–200 yards", .277, 1, "Approach buckets", "Not present in L50"),
    ("Birdie or Better %", .244, 4, "Scoring", "Not present in L50"),
    ("Approach: 200–225 yards", .240, 1, "Approach buckets", "Not present in L50"),
    ("Greens in Regulation %", .229, 2, "Ball striking", "Available"),
    ("Birdie Average", .227, 5, "Scoring", "Not present in L50"),
    ("Bogey Avoidance", .225, 3, "Scoring", "Not present in L50"),
    ("SG: Off-the-Tee", .206, 2, "Driving", "Available"),
    ("Putting: 15–20 ft", .183, 2, "Putting buckets", "Not present in L50"),
    ("Ball Speed", .165, 2, "Driving", "Not present in L50"),
    ("Approach: 150–175 yards", .144, 1, "Approach buckets", "Not present in L50"),
    ("Putting: 15–25 ft", .143, 4, "Putting buckets", "Not present in L50"),
    ("Driving Distance", .141, 3, "Driving", "Available"),
    ("Club Head Speed", .139, 2, "Driving", "Not present in L50"),
    ("SG: Putting", .133, 4, "Putting", "Available"),
    ("Approach: 100–125 yards", .126, 3, "Approach buckets", "Not present in L50"),
    ("Approach: 225–250 yards", .121, 1, "Approach buckets", "Not present in L50"),
    ("Three-Putt Avoidance", .117, 5, "Putting", "Not present in L50"),
    ("Putting: over 25 ft", .116, 4, "Putting buckets", "Not present in L50"),
    ("Sand Save %", .115, 1, "Short game", "Not present in L50"),
    ("Approach: 250–275 yards", .109, 1, "Approach buckets", "Not present in L50"),
    ("Scrambling", .106, 5, "Short game", "Not present in L50"),
    ("Approach: 125–150 yards", .051, 4, "Approach buckets", "Not present in L50"),
    ("SG: Around-the-Green", .063, 4, "Short game", "Available"),
    ("Good Drive %", .040, 6, "Driving", "Not present in L50"),
    ("Driving Accuracy %", .005, 6, "Driving", "Available"),
]

def mexico_course_evidence() -> pd.DataFrame:
    """Return the displayed external evidence, without inventing coefficients."""
    return pd.DataFrame(
        [
            {
                "Statistic": name,
                "Chart association": float(value),
                "Displayed comparison rank": f"#{rank}/6",
                "Category": category,
                "L50 golfer data": status,
            }
            for name, value, rank, category, status in _EVIDENCE
        ]
    ).sort_values(["Chart association","Statistic"],ascending=[False,True]).reset_index(drop=True)


def mexico_course_thesis() -> dict:
    """Qualitative thesis, not a fitted predictor or validated course coefficients."""
    return {
        "profile_id": PROFILE_ID,
        "source": SOURCE,
        "primary": "Broad ball-striking quality and SG: Approach",
        "distance_focus": "Approaches from 175–225 yards stand out in the chart",
        "supporting": "Scoring ability, GIR, measured off-tee play",
        "lower_emphasis": "Driving accuracy, around-green play, and overall putting",
        "missing": "Golfer approach-distance buckets and birdie/bogey rates are not in the L50 sheet",
        "guardrail": (
            "Chart association values are not direct weight percentages or independent evidence. "
            "SG Total, SG Tee-to-Green, SG Approach, and scoring average overlap. "
            "No extra forecast bonus is applied directly from this screenshot."
        ),
    }
