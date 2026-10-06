# NFL DFS Engine V2.0.10 — Role-Aware Correlation

Built from the validated V2.0.9.1 base.

## Narrow change only
- Preserves current-week projections, variance/tail generator, 50K coherent simulations, scenario portfolio selection, attribution, and persistent downloads.
- Keeps QB+WR/TE as the strongest direct stack signal.
- Adds conservative QB+RB correlation credit based on the RB's current role/PPG proxy. This recognizes that RB receiving production can correlate with QB passing production without pretending the app has target-share data when it does not.
- Keeps opponent bring-back correlation separate.
- Adds `Correlation Detail` to lineup exports so the displayed score is explainable.
- Genuinely unpaired QBs remain labeled `No modeled QB/game-stack pairing` and can still be selected if simulations justify them; there is no mandatory-stack rule.

## Validation target
Compare Maye + Rhamondre against genuinely unpaired QB lineups such as the prior Shough construction. Maye/Rhamondre should now receive bounded QB+RB proxy credit while a truly unpaired QB remains at zero unless an opponent bring-back or another modeled pairing is present.

## V2.0.11 — Injury Eligibility Gate
- OUT and DOUBTFUL are hard-excluded before simulation and candidate generation.
- IR, inactive, suspended, PUP, and NFI remain hard-excluded.
- QUESTIONABLE remains eligible and is explicitly flagged for pre-lock review.
- Re-activating/reloading the slate invalidates prior simulations and portfolios so updated injury statuses require regeneration.
- Final Portfolio Injury QC independently rejects any lineup containing a blocked injury status and reports QUESTIONABLE players.
- Carries forward the V2.0.10.1 candidate-column compatibility safeguard; no projection, variance, correlation, or portfolio-selection methodology was changed.
