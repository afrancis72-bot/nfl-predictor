# PGA Predictor Pro V10.9.5 — Round Intelligence + Round Showdown

## Replace these three root files

1. `app.py`
2. `round_parlay.py`
3. `round_showdown.py` (new)

Keep all other files unchanged. In Streamlit, use Setup / Inputs to reload weekly inputs after deployment. Then select Round Parlays or Round Showdown.

## What changed

- Official PGA TOUR `LeaderboardCompressedV3` is the first choice for completed prior-round strokes. Only plausible numeric round totals are used. No in-progress or unscored round is treated as completed.
- R2/R3/R4 now block the parlay simulation if no verified prior-round scores exist, or if fewer than half the model golfers match the prior-round data. The page displays the actual evidence count and source.
- Course DNA ON/OFF audit remains same-seed, same inputs, and the existing 62% skill / 23% course / 15% form pre-round baseline is unchanged.
- A dedicated Round Showdown page has been added for R2/R3/R4, with round-specific DK PGA salaries, exposure controls, player ratings, and CSV lineup export.

## Important limitations

- Showdown outputs use a transparent **rating-based fantasy proxy**, not calibrated DK scoring projections or a verified historical PGA showdown scoring model. They are not official DraftKings upload-format CSVs; they list six golfers by name, so do not upload them directly as DK entry files.
- The round form layer uses completed prior-round *strokes* (field-relative fallback), or SG components if a manually uploaded file supplies them. PGA leaderboard does not automatically provide SG components here.
- Automatic official leaderboard retrieval is dependent on the public PGA TOUR endpoint and its frontend key remaining functional. If it fails, use the prior-round CSV fallback.
- This is a newly implemented Showdown page, not a recovered historical Showdown engine; none of the available PGA archive builds contained that engine.
