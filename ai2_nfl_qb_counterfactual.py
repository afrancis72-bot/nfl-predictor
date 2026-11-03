"""NFL quarterback counterfactuals evaluated on independent-held-out game worlds.

A legal QB stack's simulated value is evidence for comparing optimizer choices.
It is NOT proof that a quarterback should be rostered, a market forecast, or
a historical backtest. The evaluation worlds must never affect optimization.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from ai2_nfl_qb_audit import qb_and_partners


def quarterback_counterfactuals(all_scored,chosen,player_ids,holdout):
    """Return per-QB best modeled lineup, its held-out performance and final
    portfolio benchmark. Model-eligible candidates are *already* fully legal.
    All IDs must align to the unmodified simulated golfer-free NFL player pool.
    """
    matrix=np.asarray(holdout,dtype=float)
    ids=[str(v) for v in player_ids]
    if matrix.ndim!=2 or matrix.shape[0]<100 or matrix.shape[1]!=len(ids):
        raise ValueError("QB holdout must contain separate simulated NFL worlds")
    if not np.isfinite(matrix).all() or len(set(ids))!=len(ids):
        raise ValueError("QB holdout player IDs or outcomes are invalid")
    lookup={v:i for i,v in enumerate(ids)}
    total_by_qb=Counter()
    stacked_by_qb=Counter()
    best_mean={}
    best_ceiling={}
    def score_holdout(entry):
        roster=entry["candidate"]["players"]
        columns=[lookup[str(p["ID"])] for p in roster]
        values=matrix[:,columns].sum(axis=1)
        return (round(float(values.mean()),2),
                round(float(np.percentile(values,90)),2),
                round(float(np.percentile(values,99)),2))
    for entry in all_scored:
        qb,partners=qb_and_partners(entry["candidate"]["players"])
        if qb is None:
            continue
        ident=str(qb["ID"])
        total_by_qb[ident]+=1
        if not partners:
            continue
        stacked_by_qb[ident]+=1
        if (ident not in best_mean or
                (entry["simulated_mean"],entry["simulated_p99"])>
                (best_mean[ident]["simulated_mean"],best_mean[ident]["simulated_p99"])):
            best_mean[ident]=entry
        if (ident not in best_ceiling or
                (entry["simulated_p99"],entry["simulated_mean"])>
                (best_ceiling[ident]["simulated_p99"],best_ceiling[ident]["simulated_mean"])):
            best_ceiling[ident]=entry
    selected_mean=np.asarray([float(x["simulated_mean"]) for x in chosen])
    selected_p99=np.asarray([float(x["simulated_p99"]) for x in chosen])
    selected_heldout=[score_holdout(x) for x in chosen]
    selected_heldout_mean=np.mean([x[0] for x in selected_heldout])
    final_counts=Counter()
    for entry in chosen:
        qb,_=qb_and_partners(entry["candidate"]["players"])
        if qb is not None:
            final_counts[str(qb["ID"])]+=1
    rows={}
    for qb_id in sorted(total_by_qb):
        top=best_mean.get(qb_id)
        upper=best_ceiling.get(qb_id)
        if top is None:
            rows[qb_id]={
                "Counterfactual status":"No legal direct QB/WR-TE stack in scored candidates",
                "Counterfactual candidates":total_by_qb[qb_id],
                "Counterfactual stacked candidates":0}
            continue
        roster=top["candidate"]["players"]
        qb,partners=qb_and_partners(roster)
        hold_mean,hold_p90,hold_p99=score_holdout(top)
        rows[qb_id]={
            "Counterfactual status":("Selected" if final_counts[qb_id]
                                     else "Not selected; compare documented lineup values"),
            "Counterfactual candidates":total_by_qb[qb_id],
            "Counterfactual stacked candidates":stacked_by_qb[qb_id],
            "Best legal stack train mean":round(float(top["simulated_mean"]),2),
            "Best legal stack train P90":round(float(top["simulated_p90"]),2),
            "Best legal stack train P99":round(float(top["simulated_p99"]),2),
            "Best legal stack holdout mean":hold_mean,
            "Best legal stack holdout P90":hold_p90,
            "Best legal stack holdout P99":hold_p99,
            "Best legal ceiling stack train P99":round(float(upper["simulated_p99"]),2),
            "Best legal stack salary":top["candidate"]["salary"],
            "Best legal stack receivers":", ".join(str(p.get("Name",p["ID"])) for p in partners),
            "Best legal stack roster":", ".join(str(p.get("Name",p["ID"])) for p in roster),
            "Mean gap vs selected portfolio median":round(
                float(top["simulated_mean"])-float(np.median(selected_mean)),2),
            "P99 gap vs selected portfolio median":round(
                float(upper["simulated_p99"])-float(np.median(selected_p99)),2),
            "Holdout mean gap vs selected portfolio mean":round(
                hold_mean-float(selected_heldout_mean),2),
        }
    return rows,{
        "holdout_worlds":int(len(matrix)),
        "selection_worlds":"disjoint from holdout; see one-click run configuration",
        "selected_holdout_avg_mean":round(float(selected_heldout_mean),2),
        "selected_training_avg_mean":round(float(selected_mean.mean()),2),
        "selected_holdout_mean_minus_training":round(
            float(selected_heldout_mean-selected_mean.mean()),2),
        "counterfactual_qbs_with_legal_stacks":len(best_mean),
        "counterfactual_qbs_without_legal_stacks":len(total_by_qb)-len(best_mean),
    }
