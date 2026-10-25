# PGA Predictor Pro V10.7.0 — Round 3-Ball / 6-Leg Builder

Preserves the V10.6.6b tournament projection and DraftKings optimizer logic and adds a separate round-specific parlay module.

## Round model
- Public grouping pull with Golf Channel article URL support and CSV fallback.
- R1: OTIS True Skill + Course Fit + pre-event Form.
- R2-R4: same baseline plus shrunk evidence from completed prior tournament rounds.
- Sustainable SG components are preferred for live form; putting is deliberately discounted and live adjustment is capped.
- 25K / 50K / 100K round simulations.
- Per 3-ball: win, push, loss and non-loss probability.
- Six-leg ticket portfolio with overlap control.
- Push-aware ticket diagnostics: strict 6/6 and all-legs-non-loss probability.
- Unmatched names are excluded, never guessed.

## Data integrity
OTIS Rank and OTIS Model remain audit-only and are not predictive inputs. Public leaderboard ingestion is best-effort because publisher markup can change; prior-round CSV upload is the safe fallback. Sportsbook odds are not assumed in V1.
