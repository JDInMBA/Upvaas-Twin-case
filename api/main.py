"""FastAPI server for the UPVAAS-Twin clinician dashboard.

    uvicorn api.main:app --reload --port 8000      (then open http://localhost:8000)

Interactive API docs (Swagger) are served at /docs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from upvaas import service  # noqa: E402

app = FastAPI(title="UPVAAS-Twin API", version="0.1.0",
              description="Festival- and fasting-aware metabolic digital twin for Indians with type 2 diabetes. "
                          "Research prototype on synthetic data - not a medical device.")


class WhatIf(BaseModel):
    day_type: str = "karva_chauth_nirjala"
    su_action: str = "usual"
    low_gi_swap: bool = False
    walk_after_break_fast: bool = False
    garba: bool = False
    basal_insulin_change: float = 0.0
    dpos: Optional[int] = None


def _guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except KeyError as e:
        raise HTTPException(404, f"not found: {e}")


@app.get("/api/patients")
def patients(dpos: Optional[int] = None):
    return service.patients(dpos)


@app.get("/api/patient/{pid}")
def patient(pid: str, dpos: Optional[int] = None):
    return _guard(service.patient, pid, dpos)


@app.get("/api/patient/{pid}/timeline")
def timeline(pid: str, dpos: Optional[int] = None):
    return _guard(service.timeline, pid, dpos)


@app.get("/api/patient/{pid}/risk")
def risk(pid: str, dpos: Optional[int] = None):
    return _guard(service.risk, pid, dpos)


@app.post("/api/patient/{pid}/whatif")
def whatif(pid: str, body: WhatIf):
    d = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    return _guard(service.whatif, pid, d.pop("dpos"), d)


@app.get("/api/patient/{pid}/fhir")
def fhir(pid: str, dpos: Optional[int] = None):
    return _guard(service.fhir, pid, dpos)


@app.get("/api/asha")
def asha(limit: int = 60):
    return service.asha(limit)


@app.get("/api/model-card")
def model_card():
    return service.model_card()


app.mount("/static", StaticFiles(directory=ROOT / "dashboard"), name="static")
app.mount("/figures", StaticFiles(directory=ROOT / "reports" / "figures"), name="figures")


@app.get("/")
def index():
    return FileResponse(ROOT / "dashboard" / "index.html")
