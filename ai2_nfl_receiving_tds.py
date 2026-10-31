"""Observed receiving-TD per reception with conservative prior smoothing.

Receiving touchdowns must occur on actual receptions and remain limited
by the *shared* team scoring total. This adjusts only the RELATIVE chance
among players who caught a pass, using pregame nflverse TD and catch history.
"""
from __future__ import annotations
import numpy as np


def receiving_td_factors(observed_rates,*, prior_rate=.07,prior_catches=40.):
    """Empirical Bayes per-reception TD frequency, capped and fail-neutral."""
    factors=[]
    for row in observed_rates:
        if row is None:
            factors.append(1.)
            continue
        try:
            games=float(row.get("games_played",float("nan")))
            receptions=float(row.get("receptions_pg",float("nan")))
            rec_td=float(row.get("receiving_tds_pg",float("nan")))
        except (TypeError,ValueError):
            factors.append(1.)
            continue
        if not (np.isfinite([games,receptions,rec_td]).all()
                and games>0 and receptions>=0 and rec_td>=0):
            factors.append(1.)
            continue
        count=float(max(0.,receptions*games))
        tds=float(max(0.,rec_td*games))
        if tds>count:
            factors.append(1.)
            continue
        shrunk=(tds+prior_rate*prior_catches)/(count+prior_catches)
        factors.append(float(np.clip(shrunk/prior_rate,.55,1.85)))
    return np.asarray(factors,dtype=float)
