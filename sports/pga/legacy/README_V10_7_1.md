# PGA Predictor V10.7.1 — PGA TOUR Grouping Ingestion Fix

- PGA TOUR public tee-time/grouping pages are attempted before Golf Channel.
- Parsed groupings must validate as exactly three distinct plausible player names.
- Golf Channel round-specific article parsing remains a fallback; its interactive tee-time widget is not trusted.
- CSV grouping fallback remains available.
- Course DNA, round simulation, live-form updating, push handling, and 6-leg ticket construction are unchanged from V10.7.0.
