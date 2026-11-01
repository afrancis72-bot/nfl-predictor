"""Explain large changes from observed partial DK history to simulated means.

A REVIEW FLAG, never an arbitrary scoring reduction. Passing it does not
certify the NFL model as historically calibrated or a good GPP decision.
"""
import numpy as np


def projection_review_flags(audits, pool):
    historical = {
        str(p["ID"]): float(p["proj"])
        for p in pool.to_dict("records")
        if np.isfinite(float(p.get("proj") or 0)) and float(p.get("proj") or 0)>0
    }
    flagged = []
    for rec in audits or []:
        if not str(rec.get("Audit status") or "").startswith("Verified"):
            continue
        raw=historical.get(str(rec.get("DraftKings ID")))
        if raw is None:
            continue
        mean=float(rec["Sim mean DK"])
        delta=mean-raw
        excessive=(mean>=25 and delta>=7 and mean>=raw*1.35)
        huge=mean>=30
        if not (excessive or huge):
            continue
        # Explain the mean with the exact same 10,000 stat worlds used by
        # the optimizer. Do not imply a football component was measured if
        # source detail is missing or the scoring ledger is unavailable.
        component_keys = (
            "Points: receptions", "Points: receiving yards",
            "Points: receiving TD", "Points: 100-yard receiving bonus",
            "Points: rushing yards", "Points: rushing TD",
            "Points: 100-yard rushing bonus", "Points: passing yards",
            "Points: passing TD", "Points: interceptions",
            "Points: 300-yard passing bonus",
        )
        points = {
            k.replace("Points: ", ""): round(float(rec[k]), 2)
            for k in component_keys if k in rec
        }
        opportunities = {
            k: rec[k] for k in (
                "Mean targets", "Mean receptions", "Mean receiving yards",
                "Mean receiving TD", "Mean carries", "Mean rushing yards",
                "Mean rushing TD", "Target allocation prior / game",
                "Team pass attempts prior", "Team offensive TD prior",
                "Chance 1+ receiving TD %", "Chance 100+ receiving yards %",
            ) if k in rec
        }
        flagged.append({
            "Player":rec["Player"],"Team":rec["Team"],
            "Position":rec["Position"],
            "Historical observed partial DK":round(raw,2),
            "New simulated mean DK":round(mean,2),
            "Difference DK":round(delta,2),
            "Fantasy point breakdown":", ".join(
                f"{k}: {v:+.2f}" for k,v in points.items()),
            "Simulated opportunities":", ".join(
                f"{k}: {v}" for k,v in opportunities.items()),
            "Historical games":rec.get("Observed games",""),
            "Projection source":rec.get("Projection source",""),
            "Reason":("Very high projected mean; verify every offensive stat prior"
                      if huge else
                      "Large increase over observed performance; check rate and injury usage"),
            "Action":"REVIEW—not automatic exclusion or an untested score cap"
        })
    return sorted(flagged,key=lambda row:-row["New simulated mean DK"])
