"""Fail-closed, source-of-truth release criteria for the ONE unified NFL Classic app.

This gate deliberately does not mark a simulator ready from synthetic test
passes, a successful merge, or the existence of lineup CSV/XLSX files.

The criteria must be deliberately re-reviewed, with dated evidence and
prospective holdouts, before modifying their statuses.  Until then the NFL
consumer UI must not generate, approve, or advertise new entry-ready lineups.
The underlying simulator may still be exercised by offline QA/regression tests.
"""

CONTRACT_ID = "TEAM_AI2_NFL_BENTLEY_CLASSIC_V1"
STATUS_NOT_READY = "NOT READY FOR ENTRY"

# A PASS must require linked, independent evidence, not only a green unit test.
# These statements are intentionally explicit and conservative, based on the
# implementation actually wired to the unified production scoring path.
REQUIREMENTS = (
    {
        "id": "POSSESSION_WORLD",
        "criterion": "Shared chronological possessions, game-script, play volume and scoring drive the stat-by-stat Classic forecast",
        "status": "NOT_PROVEN",
        "evidence": "Possession simulation research exists; not activated or validated in the one-click released scoring path.",
        "pass_proof": "Real-slate game accounting, conservation, scoring, and forward-tested forecast improvements.",
    },
    {
        "id": "ROLE_SPECIFIC_DEFENSE",
        "criterion": "Prior opposing QB/RB1/RB2/WR1/WR2/WR3/TE1/TE2 outcomes vs their own pregame baselines influence relevant stat channels when supported",
        "status": "EXPERIMENTAL_UNVALIDATED",
        "evidence": "Legacy broad defense_weight=0.0 remains correctly disabled. A source-dated WR1/WR2/WR3/RB1/RB2/TE1/TE2/QB1 per-stat input-rate engine is now wired to the locked simulation path, but real-slate and independent holdout gains are NOT established.",
        "pass_proof": "Leakage-free effect sizes for targets/carries/yard efficiency/TDs, opponent-quality control, sample shrinkage, and independent holdout gain.",
    },
    {
        "id": "CURRENT_PLAYER_ROLES",
        "criterion": "Verified current-week player teams, depth roles, routes, target shares, snaps, inactive replacement roles and credible minimum opportunity",
        "status": "PARTIAL_UNVERIFIED",
        "evidence": "Prior-game role/participation estimates exist, but current-week WR1/2/3 roles and shifts are not fully live-verified.",
        "pass_proof": "Provenance per role, roster transaction and injury change; no late-game/target-week data leakage.",
    },
    {
        "id": "LIVE_AVAILABILITY",
        "criterion": "Source-verified current-slate NFL pregame status and injury intelligence, with explicit freshness and unknown coverage",
        "status": "PARTIAL_UNVERIFIED",
        "evidence": "Verified-source collector is opt-in and can legitimately have zero game-day active claims; end-to-end live weekend coverage unproven.",
        "pass_proof": "Actual prekickoff source captures on multiple real slates, timestamps and missing-source fail-closed behavior.",
    },
    {
        "id": "STAT_OUTCOME_CALIBRATION",
        "criterion": "Per-player possession-linked stat lines, valid means/P90 tails and realistic target/carry/TD distributions",
        "status": "NOT_PROVEN",
        "evidence": "Stat draws are produced; no passing release-grade multi-position coverage/calibration study documented.",
        "pass_proof": "Pregame-frozen player and team stat calibration, P90 coverage, bias and distribution checks by position.",
    },
    {
        "id": "OUT_OF_SAMPLE_VALIDATION",
        "criterion": "Independent historical and forward NFL holdouts show acceptable predictive accuracy and improvement beyond the existing baseline",
        "status": "NOT_PASSED",
        "evidence": "Earlier simple defensive and naive teammate-universe holdouts slightly worsened results; 118 targeted source/build tests establish software behavior, not skill.",
        "pass_proof": "Reproducible no-lookahead train/test dates, per-stat MAE/bias/P90 and uncertainty intervals vs the baseline.",
    },
    {
        "id": "CONTEST_PORTFOLIO",
        "criterion": "Valid DK $50K Classic rosters, >=$49,600 target, 35% exposure, opportunity, stacking, diversification and true user-chosen contest tiers",
        "status": "PARTIAL_UNVERIFIED",
        "evidence": "Lineup smoke tests exist, but 40-entry default contest tiers are $3/$0.50/$0.10, not this week's $1/$0.50 plan; no final released real-slate portfolio QA.",
        "pass_proof": "Real uploaded slate, 20+ entry portfolio tests, user-defined contest tier assignments, exposure/overlap and role-warning audits.",
    },
    {
        "id": "PROVENANCE_AND_AUDIT",
        "criterion": "Immutable pregame snapshot of sources, version, inputs, projections, mean/P90 and later comparisons to actual performance",
        "status": "PARTIAL_UNVERIFIED",
        "evidence": "Source and version manifests exist; prospective immutable complete pregame/postgame independent audit not yet proven.",
        "pass_proof": "Actual-slate frozen manifest and post-slate grade with original predictions and approved lineup preservation.",
    },
    {
        "id": "FULL_REGRESSION_AND_DEPLOYMENT",
        "criterion": "No unresolved critical regression failures, real-slate end-to-end smoke and verification of the exact deployed Streamlit code",
        "status": "NOT_PROVEN",
        "evidence": "118 targeted tests passed after the availability integration; that is not the complete regression suite or real-live deployment acceptance.",
        "pass_proof": "Full test suite, production SHA identity, full browser-to-export acceptance and credible failed-test disposition.",
    },
)
# Showdown is a distinct, SECONDARY release with its own CPT/FLEX acceptance
# criteria; it must not masquerade as a completed Classic feature.


def nfl_classic_release_status():
    blockers = [dict(x) for x in REQUIREMENTS if x["status"] != "PASS"]
    return {
        "contract_id": CONTRACT_ID,
        "status": STATUS_NOT_READY if blockers else "READY FOR USER REVIEW",
        "ready_for_entries": not blockers,
        "checks_total": len(REQUIREMENTS),
        "checks_passed": len(REQUIREMENTS)-len(blockers),
        "blockers": blockers,
        "must_not_build_lineups": bool(blockers),
        "showdown_status": "SECONDARY - NOT YET RELEASED",
        "user_final_approval_required_even_if_ready": True,
    }


def assert_nfl_classic_ready_for_entries():
    report = nfl_classic_release_status()
    if not report["ready_for_entries"]:
        failures = ", ".join(x["id"] for x in report["blockers"])
        raise ValueError(
            f"{STATUS_NOT_READY}: {CONTRACT_ID} release gate blocked NFL lineup generation. "
            f"Unpassed acceptance criteria: {failures}. "
            "Use the app's release checklist; do not rely on a new seed or experimental label."
        )
    return report
