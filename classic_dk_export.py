"""DraftKings Classic roster export with strict lineup and template validation."""
import csv
import io
import pandas as pd

SLOTS=("QB","RB","RB","WR","WR","WR","TE","FLEX","DST")

def export_classic_ids(picks, salaries):
    required={"lineup","slot","ID"}
    if not required.issubset(picks.columns):
        raise ValueError("Classic picks require lineup, slot and ID")
    if not {"ID","Position","Salary","TeamAbbrev"}.issubset(salaries.columns):
        raise ValueError("Original DraftKings Classic salaries required")
    pool=salaries.copy()
    pool["ID"]=pool["ID"].astype(str).str.replace(r"\.0$","",regex=True).str.strip()
    if pool["ID"].duplicated().any(): raise ValueError("Duplicate DK salary IDs")
    if "Status" not in pool.columns:
        raise ValueError("DraftKings salary CSV is missing Status; cannot validate player eligibility for export.")
    blocked={"OUT","O","IR","INACTIVE","SUSPENDED","PUP","NFI"}
    pool["Status"]=pool["Status"].fillna("").astype(str).str.upper().str.strip()
    pool=pool.set_index("ID")
    result=[]
    for number, group in picks.groupby("lineup",sort=True):
        if len(group)!=9: raise ValueError(f"Lineup {number}: expected nine players")
        ids=group["ID"].astype(str).str.replace(r"\.0$","",regex=True).str.strip().tolist()
        if len(set(ids))!=9: raise ValueError(f"Lineup {number}: duplicate player")
        if any(i not in pool.index for i in ids): raise ValueError(f"Lineup {number}: unknown DK player ID")
        slots=group["slot"].astype(str).str.upper().tolist()
        if sorted(slots)!=sorted(SLOTS): raise ValueError(f"Lineup {number}: invalid roster slots")
        chosen=pool.loc[ids]
        if chosen["Status"].isin(blocked).any(): raise ValueError(f"Lineup {number}: contains inactive player")
        salary=int(pd.to_numeric(chosen["Salary"],errors="raise").sum())
        if not 47500<=salary<=50000: raise ValueError(f"Lineup {number}: salary outside permitted range")
        row={"Lineup":number}
        for position in ("QB","RB","WR","TE","DST"):
            subset=group[group["slot"].astype(str).str.upper()==position]
            for idx,(_,p) in enumerate(subset.iterrows(),1):
                pid=str(p["ID"]).removesuffix(".0").strip()
                actual=("DST" if str(pool.loc[pid,"Position"]).upper()=="D" else str(pool.loc[pid,"Position"]).upper())
                if actual!=position: raise ValueError(f"Lineup {number}: {pid} is not {position}")
                row[position+str(idx) if position in ("RB","WR") else position]=pid
        flex=group[group["slot"].astype(str).str.upper()=="FLEX"].iloc[0]
        fid=str(flex["ID"]).removesuffix(".0").strip()
        if str(pool.loc[fid,"Position"]).upper() not in ("RB","WR","TE"):
            raise ValueError(f"Lineup {number}: invalid FLEX")
        row["FLEX"]=fid
        row["Salary"]=salary
        result.append(row)
    return pd.DataFrame(result)

def fill_classic_reserved_entries(entries_csv, lineups):
    rows=list(csv.reader(io.StringIO(entries_csv)))
    if not rows: raise ValueError("Empty DK template")
    header=[h.strip().upper() for h in rows[0]]
    try: entry_idx=header.index("ENTRY ID")
    except ValueError: raise ValueError("Missing Entry ID")
    slot_idxs=[]
    for slot in SLOTS:
        prefix=slot if slot not in ("RB","WR") else slot[:2]
        matches=[i for i,h in enumerate(header) if h==prefix or (h.startswith(prefix) and h[len(prefix):].isdigit())]
        occurrence=(SLOTS[:len(slot_idxs)+1].count(slot)-1)
        if len(matches)<=occurrence: raise ValueError(f"Missing template slot {slot}")
        slot_idxs.append(matches[occurrence])
    available=[r for r in rows[1:] if len(r)>max(slot_idxs+[entry_idx]) and r[entry_idx].strip().isdigit()
               and all(not r[i].strip() for i in slot_idxs)]
    if len(available)<len(lineups): raise ValueError("Insufficient empty reserved entries")
    for row,(_,line) in zip(available,lineups.iterrows()):
        for slot,idx in zip(("QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST"),slot_idxs):
            row[idx]=str(line[slot])
    out=io.StringIO(newline="")
    csv.writer(out).writerows(rows)
    return out.getvalue()
