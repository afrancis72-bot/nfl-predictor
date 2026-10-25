# V10.9.4 — Round Parlay Course DNA Impact Audit

Adds a transparent DNA ON vs OFF audit to Round Parlays without changing the existing 62% True Skill / 23% Course Fit / 15% Form model.

- Runs a paired simulation with the same groups, live inputs, simulation count, and random seed.
- DNA ON uses independent Yokohama Course DNA; DNA OFF uses the prior generic OTIS Course Fit.
- Shows rating delta, win-probability delta, non-loss delta, and whether Course DNA changes the recommended golfer in each group.
- Adds downloadable round-level Course DNA audit CSV.
- No parlay weights were changed.
