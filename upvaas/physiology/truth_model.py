"""The "real body": a structurally DIFFERENT model from the one the twin uses.

Why this file exists
--------------------
A digital twin that is validated against a simulator running *the twin's own equations* measures
only parameter error. It cannot measure the thing that actually breaks twins in the clinic:
**model misspecification** - the body doing something the model has no term for.

So the virtual patients in this project are integrated with the model below, while the twin
(`upvaas/physiology/model.py`) keeps its simpler SU-Bergman-Gut equations and never learns these
extra terms. Four pieces of real physiology are present in the body and *absent from the twin*:

  1. **Three-compartment gut** (stomach -> upper intestine -> absorptive pool). The twin assumes
     two. Glucose appearance is therefore later and more spread out than the twin can represent.
  2. **Interstitial insulin delay.** Insulin acts from a remote compartment that fills with a
     ~18 min lag (Dalla Man 2007 style). The twin drives its action term straight from plasma.
  3. **Circadian insulin sensitivity.** SI swings ~+-18% over the day, lowest around 06:00
     (morning insulin resistance), highest in the evening. The twin only knows a sleep-quality
     multiplier.
  4. **Renal glucose excretion** above a ~180 mg/dL threshold, and an **incretin (GLP-1-like)
     secretion boost** proportional to glucose appearance. The twin has neither term.

Set `SIM.model_mismatch = False` (CLI: `--matched-model`) to integrate the body with the twin's
own equations instead. The pipeline reports both, so the cost of misspecification is measured
rather than assumed.

State adds `Q3` (third gut compartment) and `Id` (interstitial insulin) to the twin's state keys.
"""
from __future__ import annotations

import numpy as np

from .model import E_EX, F_BIO, K_EMP, KA_SU, VG, circadian_dawn

# --- components the twin does not model ------------------------------------------------------
K_GUT2 = 0.045          # 1/min, transit out of the second gut compartment [assumption]
TAU_ID = 18.0           # min, plasma -> interstitial insulin equilibration (Dalla Man 2007 range)
SI_CIRC_AMP = 0.18      # +-18% circadian swing in insulin sensitivity [assumption; dawn IR is real]
RENAL_THR = 180.0       # mg/dL, renal glucose threshold (textbook ~180)
RENAL_K = 0.0021        # 1/min, glucose clearance above threshold [assumption]
INCRETIN = 0.45         # incretin boost on secretion, per unit Ra [assumption]


def circadian_si(minute_of_day: np.ndarray | float) -> np.ndarray:
    """Insulin sensitivity rhythm: trough ~06:00, peak ~18:00."""
    h = (np.asarray(minute_of_day) / 60.0) % 24
    return 1.0 + SI_CIRC_AMP * np.cos(2 * np.pi * (h - 18.0) / 24.0)


def init_state(shape, G0, I0=0.0, C0=0.0) -> dict:
    z = np.zeros(shape, dtype=np.float64)
    I = np.broadcast_to(I0, shape).astype(float).copy()
    return {"Q1": z.copy(), "Q2": z.copy(), "Q3": z.copy(), "kab": z.copy() + 0.03,
            "G": np.broadcast_to(G0, shape).astype(float).copy(), "X": z.copy(), "I": I,
            "Id": I.copy(), "A": z.copy(), "C": np.broadcast_to(C0, shape).astype(float).copy(),
            "tsm": z.copy() + 480.0}


def add_meal(s: dict, carbs_mg, kab_meal) -> None:
    """Insert a meal into the stomach, carb-weighting the absorption rate of gut contents."""
    carbs_mg = np.asarray(carbs_mg, dtype=float)
    tot = s["Q1"] + s["Q2"] + s["Q3"] + carbs_mg
    s["kab"] = np.where(carbs_mg > 0,
                        (s["kab"] * (s["Q1"] + s["Q2"] + s["Q3"]) + kab_meal * carbs_mg) / np.maximum(tot, 1e-9),
                        s["kab"])
    s["Q1"] = s["Q1"] + carbs_mg


def step(s: dict, p: dict, dt: float, minute_of_day, act=0.0, si_mult=1.0, gb_mult=1.0):
    """Advance the true body by dt minutes (explicit Euler).

    Same signature as `physiology.model.step` so the simulator can swap one for the other.
    """
    G = s["G"]
    fast_egp = 1.0 - 0.22 / (1.0 + np.exp(-(s["tsm"] - 660.0) / 90.0))
    gb_eff = p["Gb0"] * p["met"] * (1.0 + p["dawn"] * circadian_dawn(minute_of_day)) * gb_mult * fast_egp

    # (1) three-compartment gut -> later, flatter glucose appearance than the twin's two
    ra = F_BIO * s["kab"] * s["Q3"] / (VG * p["bw"])
    cr = p["cr"] * np.maximum(70.0 - G, 0.0)
    ex = E_EX * act / 100.0 * G
    # (4a) renal excretion above threshold
    renal = RENAL_K * np.maximum(G - RENAL_THR, 0.0) * G / 100.0

    dG = -(p["SG"] + s["X"]) * G + p["SG"] * gb_eff + ra + cr - ex - renal
    # (3) circadian SI and (2) action driven by INTERSTITIAL insulin, not plasma
    dX = -p["p2"] * s["X"] + p["p2"] * p["SI"] * si_mult * circadian_si(minute_of_day) * s["Id"]
    dId = (s["I"] - s["Id"]) / TAU_ID

    sec = p["beta"] * np.maximum(G - p["Gb0"] * p["met"], 0.0)
    su = p["su_gain"] * 1.5 * s["C"] / (s["C"] + 0.5) * (0.45 + 0.55 * G / (G + 90.0))
    # (4b) incretin: the gut itself drives secretion, independent of the glucose rise
    incretin = INCRETIN * ra * p["beta"] / 0.085
    dI = -p["n"] * s["I"] + sec + su + incretin + p["n"] * p["ins_basal"]

    dQ1 = -K_EMP * s["Q1"]
    dQ2 = K_EMP * s["Q1"] - K_GUT2 * s["Q2"]
    dQ3 = K_GUT2 * s["Q2"] - s["kab"] * s["Q3"]
    dA = -KA_SU * s["A"]
    dC = KA_SU * s["A"] - p["ke"] * s["C"]

    s["G"] = np.clip(G + dt * dG, 20.0, 600.0)
    s["X"] = s["X"] + dt * dX
    s["I"] = np.maximum(s["I"] + dt * dI, 0.0)
    s["Id"] = np.maximum(s["Id"] + dt * dId, 0.0)
    s["Q1"] = np.maximum(s["Q1"] + dt * dQ1, 0.0)
    s["Q2"] = np.maximum(s["Q2"] + dt * dQ2, 0.0)
    s["Q3"] = np.maximum(s["Q3"] + dt * dQ3, 0.0)
    s["A"] = np.maximum(s["A"] + dt * dA, 0.0)
    s["C"] = np.maximum(s["C"] + dt * dC, 0.0)
    s["tsm"] = np.where(s["Q1"] + s["Q2"] + s["Q3"] < 3000.0, s["tsm"] + dt, 0.0)
    return ra
