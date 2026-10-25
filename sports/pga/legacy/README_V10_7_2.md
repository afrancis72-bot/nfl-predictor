# PGA Predictor V10.7.2 — Official TeeTimes API

- Uses PGA TOUR `TeeTimesCompressedV2` structured API as the primary grouping source.
- Resolves tournament IDs from the official PGA TOUR schedule when possible; accepts an explicit tournament ID override.
- Baycurrent 2026 defaults to official tournament ID `R2026527`.
- Validates exactly three distinct golfers per group.
- Rendered PGA TOUR pages and Golf Channel round articles remain fallbacks.
- CSV remains the final emergency fallback.
- No Course DNA, simulation, live-form, push, or ticket-selection methodology changed.
