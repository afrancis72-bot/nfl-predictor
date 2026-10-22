"""Receiver-specific defensive injury exposure (research mode, no automatic boosts).

A WR2/WR3 hypothesis is tested through actual alignment/coverage exposure,
not assumed from depth-chart rank. Requires documented opponent and measured
alignment exposure. No coverage data => neutral and no DFS score adjustment.
"""
import numpy as np
import pandas as pd

MIN_ROUTES=50
MIN_MATCHUP_ROUTES=15

def receiver_injury_exposure(receivers, vulnerabilities):
    """Estimate receiver matchup exposure without guessing shadow coverage.

    receivers columns: player_id, offense_team, opponent_team, receiver_rank,
      alignment, routes, matchup_routes
    vulnerabilities columns: team, weakness, factor, supported
    matchup_routes must be measured routes versus the impacted coverage
    alignment/personnel group, not inferred from WR rank.
    Return audited exposure and a bounded efficiency-only hypothesis factor.
    The caller must not treat this as a target-share change.
    """
    cols=["player_id","offense_team","opponent_team","receiver_rank",
          "alignment","routes","matchup_routes","exposure","efficiency_factor",
          "supported","reason"]
    if receivers is None or vulnerabilities is None or receivers.empty or vulnerabilities.empty:
        return pd.DataFrame(columns=cols)
    needed={"player_id","offense_team","opponent_team","receiver_rank","alignment","routes","matchup_routes"}
    if not needed.issubset(receivers.columns):
        raise ValueError("Missing receiver exposure fields: "+str(sorted(needed-set(receivers.columns))))
    vneeded={"team","weakness","factor","supported"}
    if not vneeded.issubset(vulnerabilities.columns):
        raise ValueError("Missing vulnerability fields: "+str(sorted(vneeded-set(vulnerabilities.columns))))
    output=[]
    for row in receivers.itertuples(index=False):
        routes=pd.to_numeric(pd.Series([row.routes]),errors="coerce").iloc[0]
        affected=pd.to_numeric(pd.Series([row.matchup_routes]),errors="coerce").iloc[0]
        candidate=vulnerabilities[
            (vulnerabilities.team.astype(str)==str(row.opponent_team))&
            (vulnerabilities.weakness.astype(str)==str(row.alignment))&
            (vulnerabilities.supported==True)]
        supported=(pd.notna(routes) and pd.notna(affected)
                   and routes>=MIN_ROUTES and affected>=MIN_MATCHUP_ROUTES
                   and 0<=affected<=routes and not candidate.empty)
        exposure=float(affected/routes) if supported else 0.
        if supported:
            valid=pd.to_numeric(candidate.factor,errors="coerce")
            valid=valid[np.isfinite(valid)&valid.between(.88,1.12)]
            supported=not valid.empty
        factor=float(1.+exposure*(valid.mean()-1.)) if supported else 1.
        output.append(dict(player_id=str(row.player_id),offense_team=str(row.offense_team),
            opponent_team=str(row.opponent_team),receiver_rank=str(row.receiver_rank),
            alignment=str(row.alignment),routes=routes,matchup_routes=affected,
            exposure=exposure if supported else 0.,efficiency_factor=factor,
            supported=bool(supported),
            reason="observed_alignment_exposure" if supported else "insufficient_or_unmatched_evidence"))
    return pd.DataFrame(output,columns=cols)
