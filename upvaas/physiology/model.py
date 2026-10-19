"""SU-Bergman-Gut model: the mechanistic core shared by the "real body" simulator and the twin.

State (all arrays broadcast over patients x particles):
  Q1, Q2  : stomach / gut glucose (mg)
  kab     : carb-weighted absorption constant of current gut contents (1/min)
  G       : plasma glucose (mg/dL)
  X       : remote insulin action (1/min)                       (Bergman minimal model)
  I       : plasma insulin above fasting basal (uU/mL)
  A, C    : sulfonylurea in gut / plasma (dose-normalised units)
  tsm     : minutes since the gut was last carrying carbohydrate (fasting clock)

Equations (per minute):
  dQ1/dt = -k_emp*Q1 + meal(t)
  dQ2/dt =  k_emp*Q1 - kab*Q2
  Ra     =  f*kab*Q2 / (Vg*BW)                                   (meal glucose appearance)
  Gb_eff =  Gb0 * met * (1 + dawn*circ(t)) * env(t) * fastEGP(tsm)   (hepatic set-point)
  fastEGP=  1 - 0.22*sigmoid((tsm - 660)/90)     (glycogen depletion after ~11 h without food)
  dG/dt  = -(SG + X)*G + SG*Gb_eff + Ra + cr*max(70-G,0) - e_ex*act(t)*G/100
  dX/dt  = -p2*X + p2*SI_day*I                                   (SI = p3/p2)
  dI/dt  = -n*I + beta*max(G-Gb0*met,0)                      (glucose-stimulated secretion)
           + su_gain*Emax(C)*(0.45 + 0.55*G/(G+90))               (SU: secretion even at low G)
           Emax(C) = 1.5*C/(C+0.5)                               (saturating SU effect)
           + n*ins_basal                                          (long-acting basal insulin)
  dA/dt  = -ka*A + dose(t) ;  dC/dt = ka*A - ke*C,  ke = ln2/t_half(CYP2C9, eGFR)

The SU term is what makes a fasting day dangerous: secretion keeps going while no carbohydrate
arrives, and counter-regulation (cr) is blunted in long-standing diabetes / autonomic neuropathy.
"""
from __future__ import annotations

import numpy as np

K_EMP = 0.030      # gastric emptying 1/min [assumption, between Dalla Man kmin/kmax]
F_BIO = 0.90       # carbohydrate bioavailability (Dalla Man 2007 uses f = 0.9)
VG = 1.88          # glucose distribution volume dL/kg (Dalla Man 2007)
KA_SU = 0.012      # SU absorption 1/min (t_max ~2-3 h) [assumption]
E_EX = 0.0035      # insulin-independent uptake per 100 steps/min [assumption]


def circadian_dawn(minute_of_day: np.ndarray | float) -> np.ndarray:
    """Dawn phenomenon bump: 0 outside 03:00-09:00, peak ~06:00."""
    h = (np.asarray(minute_of_day) / 60.0) % 24
    return np.clip(1 - ((h - 6.0) / 3.0) ** 2, 0, None)


def init_state(shape, G0, I0=0.0, C0=0.0):
    z = np.zeros(shape, dtype=np.float64)
    return {"Q1": z.copy(), "Q2": z.copy(), "kab": z.copy() + 0.03, "G": np.broadcast_to(G0, shape).astype(float).copy(),
            "X": z.copy(), "I": np.broadcast_to(I0, shape).astype(float).copy(), "A": z.copy(),
            "C": np.broadcast_to(C0, shape).astype(float).copy(), "tsm": z.copy() + 480.0}


def add_meal(s, carbs_mg, kab_meal):
    """Insert a meal into the stomach and update the carb-weighted absorption rate."""
    carbs_mg = np.asarray(carbs_mg, dtype=float)
    tot = s["Q1"] + s["Q2"] + carbs_mg
    s["kab"] = np.where(carbs_mg > 0, (s["kab"] * (s["Q1"] + s["Q2"]) + kab_meal * carbs_mg) / np.maximum(tot, 1e-9), s["kab"])
    s["Q1"] = s["Q1"] + carbs_mg


def step(s: dict, p: dict, dt: float, minute_of_day, act=0.0, si_mult=1.0, gb_mult=1.0):
    """Advance the state by dt minutes (explicit Euler; stable for dt <= 5 with these constants).

    p keys: SI, beta, Gb0, SG, p2, n, su_gain, ke, cr, dawn, bw, ins_basal, met
    """
    G = s["G"]
    fast_egp = 1.0 - 0.22 / (1.0 + np.exp(-(s["tsm"] - 660.0) / 90.0))
    gb_eff = p["Gb0"] * p["met"] * (1.0 + p["dawn"] * circadian_dawn(minute_of_day)) * gb_mult * fast_egp
    ra = F_BIO * s["kab"] * s["Q2"] / (VG * p["bw"])
    cr = p["cr"] * np.maximum(70.0 - G, 0.0)
    ex = E_EX * act / 100.0 * G
    dG = -(p["SG"] + s["X"]) * G + p["SG"] * gb_eff + ra + cr - ex
    dX = -p["p2"] * s["X"] + p["p2"] * p["SI"] * si_mult * s["I"]
    sec = p["beta"] * np.maximum(G - p["Gb0"] * p["met"], 0.0)
    su = p["su_gain"] * 1.5 * s["C"] / (s["C"] + 0.5) * (0.45 + 0.55 * G / (G + 90.0))
    dI = -p["n"] * s["I"] + sec + su + p["n"] * p["ins_basal"]
    dQ1 = -K_EMP * s["Q1"]
    dQ2 = K_EMP * s["Q1"] - s["kab"] * s["Q2"]
    dA = -KA_SU * s["A"]
    dC = KA_SU * s["A"] - p["ke"] * s["C"]

    s["G"] = np.clip(G + dt * dG, 20.0, 600.0)
    s["X"] = s["X"] + dt * dX
    s["I"] = np.maximum(s["I"] + dt * dI, 0.0)
    s["Q1"] = np.maximum(s["Q1"] + dt * dQ1, 0.0)
    s["Q2"] = np.maximum(s["Q2"] + dt * dQ2, 0.0)
    s["A"] = np.maximum(s["A"] + dt * dA, 0.0)
    s["C"] = np.maximum(s["C"] + dt * dC, 0.0)
    s["tsm"] = np.where(s["Q1"] + s["Q2"] < 3000.0, s["tsm"] + dt, 0.0)
    return ra


def params_from_truth(truth_row: dict, on_met: bool) -> dict:
    return {
        "SI": truth_row["SI"], "beta": truth_row["beta"], "Gb0": truth_row["Gb0"], "SG": truth_row["SG"],
        "p2": truth_row["p2"], "n": truth_row["n"], "su_gain": truth_row["su_gain"],
        "ke": np.log(2) / truth_row["su_thalf_min"], "cr": truth_row["cr_gain"], "dawn": truth_row["dawn"],
        "bw": truth_row["bw"], "ins_basal": truth_row["ins_basal"], "met": 0.9 if on_met else 1.0,
    }


def stack_params(rows: list[dict]) -> dict:
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
