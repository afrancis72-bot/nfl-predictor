"""Independent game-world conservation audit for Team AI² NFL research.

No optimization, no additional scoring draws, no postgame information. Every
world must account for shared team plays, QB attempts/completions, simulated
receiver *attempt slots*, rushing attempts, yards and touchdown events.

IMPORTANT: The simulator currently allocates every PASS ATTEMPT either to an
explicit modeled player or an OTHER bucket. OTHER includes both unknown
receivers AND passes without a recorded receiver target. Neither the summed
modeled targets nor OTHER is asserted to be the actual NFL box-score target
population. This distinction is essential for the next opportunity model.
"""
from __future__ import annotations

import numpy as np

SKILL = frozenset(("RB", "WR", "TE"))


def audit_team_game_conservation(stat_draws, roster, team):
    """Raise on any non-conserved scenario, return a small source-honest ledger.

    roster is the full (including football-only) simulator player DataFrame,
    BEFORE removing entries that cannot be DraftKings selected.
    """
    def required(key):
        record = stat_draws.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"Conservation missing {key}")
        return record

    def arr(record, key, *, discrete=False, nonnegative=True):
        if key not in record:
            raise ValueError(f"Conservation missing {team}.{key}")
        v = np.asarray(record[key])
        if v.ndim != 1 or v.size != n:
            raise ValueError(f"Conservation invalid world dimensions: {team}.{key}")
        if not np.isfinite(v).all():
            raise ValueError(f"Conservation nonfinite {team}.{key}")
        if discrete and (not np.equal(v, np.floor(v)).all()):
            raise ValueError(f"Conservation noninteger {team}.{key}")
        if nonnegative and (v < 0).any():
            raise ValueError(f"Conservation negative {team}.{key}")
        return v

    def equal(label, left, right, *, approx=False):
        okay = np.isclose(left, right, rtol=0., atol=.001) if approx else (left == right)
        if not np.all(okay):
            bad = np.flatnonzero(~okay)
            i = int(bad[0])
            raise ValueError(f"Conservation {team} {label} failed world {i}: "
                             f"{left[i]} versus {right[i]}")

    plays = required(f"__TEAM_PLAYS__|{team}")
    n = int(np.asarray(plays.get("simulated_plays", [])).size)
    if n < 1:
        raise ValueError(f"Conservation missing team plays for {team}")
    scoring = required(f"__TEAM_SCORING__|{team}")
    other_rec = required(f"__OTHER_RECEIVER__|{team}")
    other_rush = required(f"__OTHER_RUSHER__|{team}")

    own = roster[roster["TeamAbbrev"].astype(str).eq(team)]
    qbs = own[own["Position"].astype(str).eq("QB")]
    available_qbs = [stat_draws[str(row["Name"])]
                     for _, row in qbs.iterrows()
                     if isinstance(stat_draws.get(str(row["Name"])), dict)
                     and "pass_attempts" in stat_draws[str(row["Name"])]]
    if len(available_qbs) > 1:
        raise ValueError(f"Conservation multiple team passers not represented: {team}")
    qb = (available_qbs[0] if available_qbs else
          required(f"__OTHER_QB_PASSING__|{team}"))
    skill_rows = own[own["Position"].astype(str).isin(SKILL)]
    receivers = []
    for _, row in skill_rows.iterrows():
        rec = required(str(row["Name"]))
        for field in ("targets", "receptions", "receiving_yards",
                      "receiving_tds", "carries", "rushing_tds"):
            if field not in rec:
                raise ValueError(f"Conservation missing player {row['Name']}.{field}")
        receivers.append(rec)

    def vector_sum(records, field, *, discrete=False, nonnegative=True):
        result = np.zeros(n, dtype=float)
        for record in records:
            result += arr(record, field, discrete=discrete,
                          nonnegative=nonnegative)
        return result

    team_plays = arr(plays, "simulated_plays", discrete=True)
    dropbacks = arr(plays, "simulated_dropbacks", discrete=True)
    rushes = arr(plays, "simulated_rushes", discrete=True)
    attempts = arr(plays, "simulated_pass_attempts", discrete=True)
    sacks = arr(plays, "simulated_sacks", discrete=True)
    qb_attempts = arr(qb, "pass_attempts", discrete=True)
    qb_completions = arr(qb, "completions", discrete=True)
    qb_yards = arr(qb, "passing_yards")
    qb_pass_tds = arr(qb, "passing_tds", discrete=True)
    qb_carries = arr(qb, "carries", discrete=True)
    qb_rush_tds = arr(qb, "rushing_tds", discrete=True)
    interceptions = arr(qb, "interceptions", discrete=True)

    other_attempt_slots = arr(other_rec, "targets", discrete=True)
    other_catches = arr(other_rec, "receptions", discrete=True)
    other_yards = arr(other_rec, "receiving_yards")
    other_rec_td = arr(other_rec, "receiving_tds", discrete=True)
    other_carries = arr(other_rush, "carries", discrete=True)
    other_rush_td = arr(other_rush, "rushing_tds", discrete=True)

    unidentified=stat_draws.get(f"__UNIDENTIFIED_TARGET_ATTEMPTS__|{team}")
    if unidentified is not None:
        if not isinstance(unidentified,dict):
            raise ValueError(f"Conservation malformed no-identified-target record {team}")
        no_identified_slots=arr(unidentified,"attempts",discrete=True)
        for field in ("receiving_targets","receptions","receiving_tds"):
            if np.any(arr(unidentified,field,discrete=True)!=0):
                raise ValueError(f"Conservation unidentified passes awarded {field}")
    else:
        no_identified_slots=np.zeros(n,dtype=float)
    modeled_slots = vector_sum(receivers, "targets", discrete=True)
    modeled_catches = vector_sum(receivers, "receptions", discrete=True)
    modeled_yards = vector_sum(receivers, "receiving_yards")
    modeled_rec_td = vector_sum(receivers, "receiving_tds", discrete=True)
    modeled_carries = vector_sum(receivers, "carries", discrete=True)
    modeled_rush_td = vector_sum(receivers, "rushing_tds", discrete=True)

    equal("team plays", team_plays, dropbacks + rushes)
    equal("dropbacks", dropbacks, attempts + sacks)
    equal("QB attempts", attempts, qb_attempts)
    equal("pass attempt slots, not certified targets",
          attempts, modeled_slots + other_attempt_slots + no_identified_slots)
    equal("completed passes", qb_completions, modeled_catches + other_catches)
    equal("passing yards", qb_yards, modeled_yards + other_yards, approx=True)
    equal("passing TDs", qb_pass_tds, modeled_rec_td + other_rec_td)
    equal("rush attempts", rushes, modeled_carries + qb_carries + other_carries)
    total_rush_tds = modeled_rush_td + qb_rush_tds + other_rush_td
    all_tds = arr(scoring, "simulated_offensive_tds", discrete=True)
    equal("offensive touchdowns", all_tds, qb_pass_tds + total_rush_tds)
    field_goals = arr(scoring, "simulated_field_goals", discrete=True)
    points = arr(scoring, "simulated_offensive_points", discrete=True)
    equal("offensive points", points, all_tds*7 + field_goals*3)

    if np.any(qb_completions > attempts) or np.any(interceptions > attempts):
        raise ValueError(f"Conservation impossible QB attempts for {team}")
    for record in receivers + [other_rec]:
        if (np.any(arr(record, "receptions", discrete=True) >
                   arr(record, "targets", discrete=True)) or
            np.any(arr(record, "receiving_tds", discrete=True) >
                   arr(record, "receptions", discrete=True))):
            raise ValueError(f"Conservation receiver exceeds target/catch capacity for {team}")
    for record in receivers + [other_rush]:
        if np.any(arr(record, "rushing_tds", discrete=True) >
                  arr(record, "carries", discrete=True)):
            raise ValueError(f"Conservation rushing TD exceeds carries for {team}")
    if np.any(qb_rush_tds > qb_carries):
        raise ValueError(f"Conservation QB rushing TD exceeds carries for {team}")

    return {
        "worlds_verified": n,
        "conserved_pass_attempt_slots": True,
        "mean_team_pass_attempts": round(float(attempts.mean()), 4),
        "mean_modeled_player_attempt_slots": round(float(modeled_slots.mean()), 4),
        "mean_other_or_no_target_attempt_slots":round(
            float((other_attempt_slots+no_identified_slots).mean()),4),
        "mean_unmodeled_receiver_attempt_slots":round(
            float(other_attempt_slots.mean()),4) if unidentified is not None else None,
        "mean_no_identified_target_attempt_slots":round(
            float(no_identified_slots.mean()),4) if unidentified is not None else None,
        "mean_team_rushes": round(float(rushes.mean()), 4),
        "mean_team_offensive_tds": round(float(all_tds.mean()), 4),
        "unmodeled_receiver_targets_separately_identified":bool(unidentified is not None),
        "no_target_passes_separately_identified":False,
        "modeled_no_identified_target_attempts_separately":bool(unidentified is not None),
        "audit_warning": (
            "Research split distinguishes unmodeled receivers from passes lacking "
            "identified RB/WR/TE target; latter are not certified throwaways"
            if unidentified is not None else
            "OTHER represents unknown receivers plus no-target pass attempts; "
            "it must not be compared directly with actual nonmodeled targets"),
        "research_only": True,
        "predictive_accuracy_validated": False,
    }
