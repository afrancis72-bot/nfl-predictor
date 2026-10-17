"""Convert research Showdown lineups to validated DraftKings slot IDs.

Never infer player identity by name alone: match player + team and roster slot.
"""
import pandas as pd

SLOTS=("CPT","FLEX1","FLEX2","FLEX3","FLEX4","FLEX5")

def export_showdown_ids(lineups, salary):
    required={"Name","TeamAbbrev","ID","Salary","Roster Position"}
    if not required.issubset(salary.columns):
        raise ValueError("Native DraftKings Showdown salary CSV with ID, team and CPT/FLEX rows required")
    s=salary.copy()
    s["ID"]=s["ID"].astype(str).str.replace(r"\\.0$","",regex=True)
    s["Roster Position"]=s["Roster Position"].astype(str).str.upper().str.strip()
    s["TeamAbbrev"]=s["TeamAbbrev"].astype(str).str.upper().str.strip()
    lookup={}
    for _,r in s.iterrows():
        for slot in ("CPT","FLEX"):
            if slot in r["Roster Position"].split("/"):
                key=(str(r["Name"]).strip(),str(r["TeamAbbrev"]),slot)
                if key in lookup: raise ValueError(f"Ambiguous DK identity: {key}")
                lookup[key]=(str(r["ID"]),int(r["Salary"]))
    name_teams=s.groupby("Name")["TeamAbbrev"].nunique()
    ambiguous=set(name_teams[name_teams>1].index)
    out=[]
    for i,line in lineups.iterrows():
        selected=[str(line[k]).strip() for k in SLOTS]
        if len(set(selected))!=6: raise ValueError(f"Lineup {i+1}: duplicate player")
        if any(name in ambiguous for name in selected):
            raise ValueError(f"Lineup {i+1}: same player name occurs on multiple teams")
        row={"Lineup":i+1}
        total=0;teams=set()
        for slot,name in zip(SLOTS,selected):
            matches=[(key,v) for key,v in lookup.items() if key[0]==name and key[2]==("CPT" if slot=="CPT" else "FLEX")]
            if len(matches)!=1: raise ValueError(f"Lineup {i+1}: missing or ambiguous {slot} DK ID for {name}")
            key,(dkid,cost)=matches[0]
            row[slot]=dkid;row[slot+" Name"]=name
            teams.add(key[1]);total+=cost
        if len(teams)!=2: raise ValueError(f"Lineup {i+1}: must use both teams")
        if total>50000: raise ValueError(f"Lineup {i+1}: exceeds $50,000")
        row["Salary"]=total
        out.append(row)
    return pd.DataFrame(out)

def fill_reserved_entries(entries_csv, id_lineups):
    """Return DK upload CSV text, changing only empty reserved rows.

    The DK template's original column names and all extra metadata are retained.
    """
    import csv, io
    rows=list(csv.reader(io.StringIO(entries_csv)))
    if not rows: raise ValueError("Empty DK entries template")
    header=rows[0]
    slot_indices=[]
    for slot in SLOTS:
        key="CPT" if slot=="CPT" else "FLEX"
        choices=[i for i,c in enumerate(header) if c.strip().upper().startswith(key)]
        needed=0 if slot=="CPT" else int(slot[-1])-1
        if len(choices)<=needed: raise ValueError(f"Missing template column {slot}")
        slot_indices.append(choices[needed])
    available=[r for r in rows[1:] if len(r)>max(slot_indices) and all(not r[k].strip() for k in slot_indices)]
    if len(available)<len(id_lineups): raise ValueError(f"Only {len(available)} empty reserved entries for {len(id_lineups)} lineups")
    for r,(_,line) in zip(available,id_lineups.iterrows()):
        for slot,j in zip(SLOTS,slot_indices):r[j]=str(line[slot])
    output=io.StringIO(newline="")
    writer=csv.writer(output)
    writer.writerows(rows)
    return output.getvalue()
