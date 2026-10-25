# PGA Predictor V10.6.2 — Weekly Input Contract

## Production weekly workflow
1. Enter tournament, course, tournament-week weather start date.
2. Paste your free Golf Courses API key (password-masked in Streamlit).
3. Add a country/region hint for international events (Japan, Bermuda, Mexico, etc.).
4. Click **Find course online**, review the ranked match, and select the correct course.
5. Upload the current DraftKings field CSV. It must contain `player` and `salary`.
6. Verify course/holes and weather show as current/automatic before simulation.
7. Upload current player_stats/results/course_history when available. Repository fallbacks are testing fallbacks, not proof of current-week freshness.
8. Build projections, then optimize.

## International course source
Golf Courses API (`golfcoursesapi.com`) is used for course search, coordinates and scorecard data. A free API key is required. Course matching uses the entered course name plus an optional country/region hint and displays ranked candidates instead of silently accepting the first response.

## Weather
Open-Meteo is queried from the selected course coordinates for the selected tournament-week start date.

## Guardrails
- No PGA TOUR scraping.
- No silent course-search failures: HTTP/auth/no-match failures are shown in the sidebar.
- DraftKings upload is required only when building projections, not when testing course/weather ingestion.
- Manual `course_holes.csv` and `weather.csv` override internet sources.
- Unknown course-DNA fields remain neutral rather than being fabricated.
- Predictive model and portfolio optimizer are not intentionally changed by V10.6.2.


## OTIS Golf import (V10.6.5)

The app supports user-triggered OTIS Golf CSV exports; it does not scrape or automate requests to OTIS.

1. In OTIS Custom Data, export **Season stats** for the current field/current season. Recommended columns:
   `sg_total`, `sg_off_the_tee`, `sg_approach`, `sg_around_green`, `sg_putting`,
   `sg_tee_to_green`, `measured_rounds`, `driving_distance`, `driving_accuracy_pct`,
   `gir_pct`, `scrambling_pct`, `scoring_avg`, `birdie_avg`, `par3_scoring_avg`,
   `par4_scoring_avg`, `par5_scoring_avg`, `events_played`, `cuts_made`, `top10s`, `wins`.
2. Export **Event results** for the same players. The app uses this as recent-results input and derives current-course history when the export includes the current venue.
3. Blank OTIS values remain missing; they are never converted to zero.
4. Direct model-contract uploads override OTIS; OTIS overrides repository fallbacks.

## V10.6.6 — OTIS Advanced Course-Fit weekly player layer

Production weekly player input is the CSV exported from OTIS Golf: Model → Course fit → Advanced → Export CSV.

Required columns: Player, True Skill, Course Fit, Form.
Recommended advanced columns: Fit: APP, Fit: OTT, Fit: ARG, Fit: PUTT, Fit: History, Form rds, Venue rds.

The app deliberately does not use OTIS Rank or OTIS Model as predictive inputs. They may be retained for audit only. True Skill anchors player quality; Form and Course Fit enter as bounded, sample-aware tilts. Form rds and Venue rds are used for reliability shrinkage. Missing values remain missing rather than being converted to zero.

The current DraftKings salary CSV remains the authority for the contest field and salary. The app requires a full DK-to-OTIS name match before building production projections.


## V10.6.6a partial OTIS coverage
A full OTIS match is preferred but not required. If at least 70% of the DraftKings field matches, unmatched golfers remain in the simulation and receive a neutral OTIS Skill/Form/Fit contribution (missing OTIS values are not converted into poor-player scores). Coverage below 70% blocks the production build as a wrong-file/matching safety check.
