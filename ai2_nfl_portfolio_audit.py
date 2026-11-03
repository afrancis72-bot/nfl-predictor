"""Portable, reproducible NFL Classic portfolio audit for the LIVE unified app."""
import csv
import io
from ai2_nfl_qb_audit import qb_and_partners
from ai2_nfl_correlation_guard import opposing_pass_catcher_dst_pairs, opposing_dst_lineup_budget
from ai2_nfl_portfolio_risk import (
    uncertain_running_back_reason, maximum_player_appearances,
    DEFAULT_MAX_EXPOSURE, MAX_SHARED_PLAYERS,
)


def portfolio_audit_csv(ranked, *, seed):
    """Generate audit bytes in memory; no fragile temporary download file."""
    if not ranked:
        raise ValueError("No NFL lineups available for audit export")
    count = len(ranked)
    conflicted = sum(bool(opposing_pass_catcher_dst_pairs(
        item["candidate"]["players"])) for item in ranked)
    if conflicted > opposing_dst_lineup_budget(count):
        raise ValueError("NFL audit refused: opposing DST/receiver safety cap exceeded")
    buff = io.StringIO(newline="")
    writer = csv.DictWriter(buff, fieldnames=[
        "Lineup", "Seed", "Contest", "Salary", "Mean", "P90", "P99",
        "QB", "QB team", "QB passing teammates", "DST", "DST team",
        "Opposing pass catchers vs DST", "Pass catcher/DST conflicts",
        "Portfolio conflicted lineup cap", "All players",
        "Role-limited RBs", "Maximum RB role-risk appearances",
        "Overall maximum exposure", "Maximum shared players across two lineups",
        "Simulated QB/WR-TE upper-tail lift", "Potential returning RB competition",
    ])
    writer.writeheader()
    for i, item in enumerate(ranked, 1):
        roster = item["candidate"]["players"]
        qb, partners = qb_and_partners(roster)
        dst = next((p for p in roster
                    if str(p.get("Position", "")).upper() == "DST"), {})
        pairs = opposing_pass_catcher_dst_pairs(roster)
        writer.writerow({
            "Lineup": i, "Seed": seed, "Contest": item.get("contest_tier", ""),
            "Salary": item["candidate"].get("salary", ""),
            "Mean": round(float(item["simulated_mean"]), 2),
            "P90": round(float(item["simulated_p90"]), 2),
            "P99": round(float(item["simulated_p99"]), 2),
            "QB": qb.get("Name", "") if qb else "",
            "QB team": qb.get("TeamAbbrev", "") if qb else "",
            "QB passing teammates": ", ".join(str(p.get("Name", p.get("ID", "")))
                                              for p in partners),
            "DST": dst.get("Name", ""),
            "DST team": dst.get("TeamAbbrev", ""),
            "Opposing pass catchers vs DST": ", ".join(
                str(p.get("player", "")) for p in pairs),
            "Pass catcher/DST conflicts": len(pairs),
            "Portfolio conflicted lineup cap": opposing_dst_lineup_budget(count),
            "All players": " | ".join(str(p.get("Name", p.get("ID", "")))
                                       for p in roster),
            "Role-limited RBs": ", ".join(
                str(p.get("Name", p.get("ID", ""))) for p in roster
                if uncertain_running_back_reason(p)),
            "Maximum RB role-risk appearances": ", ".join(
                f"{p.get('Name',p.get('ID'))}: {maximum_player_appearances(p,count,DEFAULT_MAX_EXPOSURE)}/{count}"
                for p in roster if uncertain_running_back_reason(p)),
            "Overall maximum exposure": f"{DEFAULT_MAX_EXPOSURE:.0%}",
            "Maximum shared players across two lineups": MAX_SHARED_PLAYERS,
            "Simulated QB/WR-TE upper-tail lift": round(
                float(item.get("stack_lift",0.)),4),
            "Potential returning RB competition": "; ".join(
                str(p.get("Name",p.get("ID"))) + ": " +
                str(p.get("return_competition_reason"))
                for p in roster if p.get("return_competition_reason")),
        })
    return buff.getvalue().encode("utf-8-sig")
