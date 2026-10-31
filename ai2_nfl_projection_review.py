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
        flagged.append({
            "Player":rec["Player"],"Team":rec["Team"],
            "Position":rec["Position"],
            "Historical observed partial DK":round(raw,2),
            "New simulated mean DK":round(mean,2),
            "Difference DK":round(delta,2),
            "Reason":("Very high projected mean; explain each football-stat component"
                      if huge else
                      "Large increase over historical performance; inspect modeled opportunities"),
            "Action":"REVIEW—not automatic exclusion or an untested score cap"
        })
    return sorted(flagged,key=lambda row:-row["New simulated mean DK"])
