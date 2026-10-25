# Team AI² Pro — PGA / NASCAR consolidation checklist

Target: `afrancis72-bot/nfl-predictor`, branch `feature/unified-ai2-platform`.
Source PGA: `afrancis72-bot/pga-predictor` (main, dashboard V10.9.5).
Source NASCAR: `afrancis72-bot/nascar-predictor` (main, dashboard V2.14).

## Migration rules
- One unified application with NFL, PGA, NASCAR and NBA navigation.
- Keep each sport's engine separate; share branding, data freshness, validation, audits and exports.
- Do not replace proven simulation logic with a simplified placeholder.
- Do not display production-ready lineup actions before the imported engine passes parity tests.
- Never silently replace missing data with zeros or stale fallback data.
- Preserve the source repos as references; make migration changes on the unified feature branch.
- No changes to the production NFL deployment.

## PGA — first
- Port `pga_predictor_pro.py`, `golf_first_v11.py`, `portfolio_optimizer.py`, `round_parlay.py`, `international_ingestion.py` and their transitive dependencies.
- Port course-DNA inputs, OTIS/RickRunGood workflows, round splits, weather and parlay tools.
- Remove source-app `st.set_page_config` and top-level Streamlit execution when adapting UI.
- Gate lineup export on current field, DK salary, source freshness and simulation validation.
- Regression test strokes-gained totals, player matches, score distributions and lineup exports against standalone PGA.

## NASCAR — second
- Port `nascar_predictor_pro.py`, `nascar_auto_intelligence.py`, `nascar_reference_dna.py`, `source_diagnostic.py` and their transitive dependencies.
- Port track catalog, driver DNA, verified starting grid, practice/qualifying update and 100k simulation support.
- Gate final lineup export on a complete, unique qualifying grid and non-stale salary data.
- Regression test per-driver projections, score distributions, portfolio exposure and CSV output against standalone V2.14.

## Release gates
1. Inventory files and dependencies from each source repository.
2. Copy engines and non-code assets into sport-namespaced modules.
3. Add unified UI entry points; no cross-sport state collisions.
4. Run syntax/import and dependency checks.
5. Run source-versus-unified parity tests on identical saved input fixtures.
6. Enable the corresponding sport only after parity and export tests pass.

Status: migration inventory drafted; no claim of working integrated PGA/NASCAR engines yet.
