# PGA Predictor Pro V10.9.1 — Independent Course DNA Integration

- Starts from V10.8.1 Unified Dashboard / V10.9.0 Course DNA draft.
- Preserves the approved unified navigation and automatic PGA TOUR tee-times ingestion.
- 2026 Baycurrent / Yokohama Course DNA remains pre-event and independent of this week’s outcomes.
- Tournament DFS now uses an independent Yokohama course-fit signal in the model-strength path.
- Round Parlays now use the same independent Yokohama course-fit signal in the 23% pre-round course-fit slot.
- Available OTIS components are weighted 22 APP / 15 OTT / 5 ARG / 5 PUTT and renormalized across available data.
- Missing par-4, driving-distance, driving-accuracy, bogey-avoidance and approach-distance-bucket data are not fabricated.
- True Skill remains the anchor; Course DNA is a bounded course-specific tilt, avoiding double counting.
- Weather remains a separate tournament/round environment adjustment.

### V10.9.2 DFS optimizer hotfix
- Replaced the slow one-lineup-at-a-time 120k candidate loop with vectorized batched weighted candidate generation.
- Added adaptive early stopping after a deep diversified valid candidate pool is available.
- Preserves Course-DNA-adjusted projections, strategy objective, salary rules, locks/exclusions, exposure caps, and minimum uniqueness.
- Added visible DFS generation progress/status so the app no longer appears frozen.
