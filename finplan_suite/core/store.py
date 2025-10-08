# finplan_suite/core/store.py
from __future__ import annotations
import os, json, uuid, datetime, csv
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

DATA_DIR = "data/clients"
os.makedirs(DATA_DIR, exist_ok=True)

@dataclass
class Client:
    client_id: str
    first_name: str
    last_name: str
    email: str = ""
    phone: str = ""
    birth_year: Optional[int] = None
    spouse_name: str = ""
    spouse_birth_year: Optional[int] = None
    filing_status: str = "Married Filing Jointly"
    state: str = "PA"
    # simple holdings snapshot for now
    accounts: List[Dict[str, Any]] = field(default_factory=list)
    # planning inputs
    retirement_age: Optional[int] = 65
    retirement_spending: Optional[float] = None
    notes: str = ""
    # derived/linked
    risk_score: Optional[int] = None
    risk_band: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.date.today().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.date.today().isoformat())

def _path(cid: str) -> str:
    return os.path.join(DATA_DIR, f"{cid}.json")

def new_client() -> Client:
    cid = str(uuid.uuid4())
    return Client(client_id=cid, first_name="", last_name="")

def save_client(c: Client):
    c.updated_at = datetime.date.today().isoformat()
    with open(_path(c.client_id), "w", encoding="utf-8") as f:
        json.dump(asdict(c), f, indent=2)

def load_client(cid: str) -> Optional[Client]:
    p = _path(cid)
    if not os.path.exists(p): return None
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return Client(**obj)

def list_clients() -> List[Client]:
    out = []
    for name in os.listdir(DATA_DIR):
        if not name.endswith(".json"): continue
        c = load_client(name[:-5])
        if c: out.append(c)
    # sort by updated desc
    out.sort(key=lambda x: x.updated_at, reverse=True)
    return out

def export_clients_csv(path: str = "data/clients_export.csv"):
    rows = [asdict(c) for c in list_clients()]
    if not rows:
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows: w.writerow(r)
