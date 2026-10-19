"""Service layer behind the clinician dashboard. Pure Python -> JSON-ready dicts.

Both API servers (FastAPI in api/main.py and the zero-dependency fallback in serve.py) call these
functions, so the logic lives in one place.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .calendar_india import load_calendar, upcoming_fasts, window_start
from .config import ARTIFACT_DIR, COHORT_DIR, REPORT_DIR, SIM, THR
from .foods import FOODS
from .interop.fhir import IST, patient_bundle
from .physiology.simulate import DAY_TYPES
from .twin.engine import BAND_H, STATE_KEYS, TWIN_FEATURES
from .twin.features import FEATURE_GROUPS
from .twin.predictor import apply_mondrian
from .twin.priors import PhysiologyPrior, cyp_activity
from .twin.whatif import SCENARIO_DAY_TYPES, Scenario, rehearse

DAY_LABEL = {"normal": "Normal day", "navratri_phalahar": "Navratri vrat", "ekadashi": "Ekadashi vrat",
             "karva_chauth_nirjala": "Karva Chauth (nirjala)", "ramadan_roza": "Ramadan roza", "feast": "Festival feast"}
EXPLAIN_GROUPS = {
    "Glucose trend (CGM)": FEATURE_GROUPS["glucose_stream"],
    "Fasting & festival calendar": FEATURE_GROUPS["calendar"],
    "Sulfonylurea on board & meals": FEATURE_GROUPS["meal_med"] + ["tw_su_conc", "tw_insulin", "tw_X"],
    "Wearables (HR, HRV, steps, sleep)": FEATURE_GROUPS["wearable"],
    "Twin physiology forecast": [f for f in FEATURE_GROUPS["twin"] if f not in ("tw_su_conc", "tw_insulin", "tw_X")],
    "EHR, kidney & genetics": FEATURE_GROUPS["ehr"],
}
DRIFT_THR = 1.5
DEFAULT_NOW_DAY, DEFAULT_NOW_MIN = 9, 11 * 60   # day 9 of the window = 13 Oct 2026 for the Navratri cohort


def _clean(o):
    """Make any nested structure strict-JSON safe (NaN/inf -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _json_safe(fn):
    def wrap(*a, **kw):
        return _clean(fn(*a, **kw))
    wrap.__name__, wrap.__doc__ = fn.__name__, fn.__doc__
    return wrap


def _f(x, nd=1):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def store():
    S = {}
    S["ehr"] = pd.read_csv(COHORT_DIR / "ehr.csv", index_col=0)
    S["ehr"]["su_drug"] = S["ehr"]["su_drug"].fillna("")
    S["st"] = dict(np.load(COHORT_DIR / "streams.npz"))
    S["tw"] = dict(np.load(ARTIFACT_DIR / "twin_A_full.npz"))
    S["pr"] = dict(np.load(ARTIFACT_DIR / "predictions.npz"))
    S["folds"] = json.load(open(ARTIFACT_DIR / "folds.json"))
    k = max(S["folds"].values()) + 1
    # one model per CV fold: each patient is always scored by the model that never saw them
    S["models"] = {t: [joblib.load(ARTIFACT_DIR / "models" / f"A_full_{t}_fold{f}.joblib") for f in range(k)]
                   for t in ("hypo", "hyper")}
    feats = pd.read_pickle(ARTIFACT_DIR / "features_A_full.pkl")
    S["feats"] = feats
    neg = feats[(feats.y_hypo == 0) & (feats.day_type == "normal")]
    S["neutral"] = neg.median(numeric_only=True)
    S["prior"] = PhysiologyPrior.load()
    S["prior_mu"], _ = S["prior"].predict(S["ehr"])
    S["cal"] = load_calendar()
    S["metrics"] = json.load(open(REPORT_DIR / "metrics.json"))
    S["meals"] = pd.read_csv(COHORT_DIR / "meals_log.csv")
    S["asha"] = pd.read_csv(REPORT_DIR / "asha_pre_festival_recall_list.csv")
    return S


def _pidx(pid):
    return int(store()["ehr"].index.get_loc(pid))


def _dpos_for(day: int, minute: int) -> int:
    ds = store()["tw"]["dsteps"]
    k = (day * 1440 + minute) // 5
    return int(np.argmin(np.abs(ds - k)))


def _clock(pid, k5: int) -> datetime:
    p = store()["ehr"].loc[pid].to_dict()
    d0 = window_start(p, store()["cal"])
    return datetime(d0.year, d0.month, d0.day, tzinfo=IST) + timedelta(minutes=int(k5) * 5)


def default_now() -> int:
    return _dpos_for(DEFAULT_NOW_DAY, DEFAULT_NOW_MIN)


# ------------------------------------------------------------------------------------------------
@_json_safe
def patients(dpos: int | None = None) -> dict:
    S = store()
    dpos = default_now() if dpos is None else int(dpos)
    ehr, pr = S["ehr"], S["pr"]
    k = int(S["tw"]["dsteps"][dpos])
    out = []
    for i, (pid, p) in enumerate(ehr.iterrows()):
        now = _clock(pid, k)
        ph, py = pr["A_full_hypo"][i, dpos], pr["A_full_hyper"][i, dpos]
        nxt = upcoming_fasts(p.to_dict(), now.date(), S["cal"], 14)
        dt = DAY_TYPES[int(S["st"]["day_type"][i, k * 5 // 1440])]
        # relative to the patient's own baseline: an absolute NIS cut flags whole cohorts at once
        nis_src = S["pr"]["nis_rel"] if "nis_rel" in S["pr"] else S["pr"]["nis_day"]
        nis = nis_src[i, max(0, k * 5 // 1440 - 1)]
        out.append({"id": pid, "name": p["name"], "age": int(p["age"]), "sex": p["sex"], "state": p["state"],
                    "tradition": p["tradition"], "on_su": bool(p["on_sulfonylurea"]), "su_drug": p["su_drug"],
                    "hba1c": p["hba1c"], "p_hypo": _f(ph, 3), "p_hyper": _f(py, 3), "today": DAY_LABEL[dt],
                    "now": now.strftime("%a %d %b %Y, %H:%M"),
                    "next_fast": nxt[0] if nxt else None, "drift_flag": bool(np.nan_to_num(nis) > DRIFT_THR)})
    out.sort(key=lambda r: -(r["p_hypo"] or 0) - 0.5 * (r["p_hyper"] or 0))
    return {"dpos": dpos, "n_decisions": int(len(S["tw"]["dsteps"])), "patients": out}


@_json_safe
def patient(pid: str, dpos: int | None = None) -> dict:
    S = store()
    i = _pidx(pid)
    dpos = default_now() if dpos is None else int(dpos)
    p = S["ehr"].loc[pid]
    f = S["tw"]["feats"][i, dpos]
    fi = {n: f[j] for j, n in enumerate(TWIN_FEATURES)}
    mu = S["prior_mu"].loc[pid]
    badges = []
    if p["bmi"] < 25 and p["whtr"] >= 0.5:
        badges.append({"label": "Thin-fat phenotype", "detail": f"BMI {p['bmi']} but waist/height {p['whtr']}"})
    if cyp_activity(p["cyp2c9"]) < 2:
        badges.append({"label": f"CYP2C9 {p['cyp2c9']}", "detail": "slower sulfonylurea clearance"})
    if p["egfr"] < 60:
        badges.append({"label": f"eGFR {p['egfr']}", "detail": "reduced kidney function - drug accumulation risk"})
    if p["neuropathy"]:
        badges.append({"label": "Autonomic neuropathy", "detail": "blunted hypo warning symptoms & counter-regulation"})
    if p["tcf7l2_rs7903146_T"] >= 1:
        badges.append({"label": f"TCF7L2 T x{p['tcf7l2_rs7903146_T']}", "detail": "lower beta-cell secretion prior"})
    if p["lpa"] >= 50:
        badges.append({"label": f"Lp(a) {p['lpa']}", "detail": "elevated - premature CAD risk"})
    k = int(S["tw"]["dsteps"][dpos])
    nxt = upcoming_fasts(p.to_dict(), _clock(pid, k).date(), S["cal"], 21)
    ehr_view = {c: (p[c].item() if hasattr(p[c], "item") else p[c]) for c in
                ["name", "sex", "age", "state", "urban", "language", "abha_id", "asha_ward", "tradition", "vrat_style",
                 "garba", "bmi", "waist_cm", "whtr", "diabetes_duration_y", "hba1c", "fpg", "tg", "hdl", "ldl", "tyg",
                 "egfr", "alt", "lpa", "sbp", "dbp", "resting_hr", "hypertension", "neuropathy", "family_history_t2d",
                 "prior_hypo_12m", "tcf7l2_rs7903146_T", "cyp2c9", "prs_t2d_z", "on_metformin", "metformin_mg",
                 "on_sulfonylurea", "su_drug", "su_dose_mg", "on_basal_insulin", "basal_insulin_u",
                 "su_fasting_behaviour", "pm25_annual"]}
    twin = {
        "insulin_sensitivity": {"value": _f(fi["tw_SI"], 2), "prior": _f(np.exp(mu["log_SI"]) * 1e4, 2),
                                "unit": "x1e-4 /min per uU/mL", "sd_log": _f(fi["tw_SI_sd"], 3)},
        "beta_cell_gain": {"value": _f(fi["tw_beta"], 3), "prior": _f(np.exp(mu["log_beta"]), 3), "unit": "uU/mL/min per mg/dL"},
        "hepatic_set_point": {"value": _f(fi["tw_Gb0"], 0), "prior": _f(mu["Gb0"], 0), "unit": "mg/dL", "sd": _f(fi["tw_Gb0_sd"], 1)},
        "su_half_life": {"value": _f(fi["tw_su_thalf"], 1) if p["on_sulfonylurea"] else None,
                         "prior": _f(np.exp(mu["log_su_thalf"]) / 60, 1) if p["on_sulfonylurea"] else None, "unit": "h"},
        "su_on_board": _f(fi["tw_su_conc"], 2), "fasting_clock_h": _f(fi["tw_fast_clock"] / 60, 1),
        "fidelity_nis": _f(fi["tw_nis"], 2),
    }
    nis_day = (S["pr"]["nis_rel"] if "nis_rel" in S["pr"] else S["pr"]["nis_day"])[i]
    return {"id": pid, "ehr": ehr_view, "badges": badges, "twin": twin, "upcoming_fasts": nxt[:6],
            "nis_daily": [_f(x, 2) for x in nis_day], "drift_threshold": DRIFT_THR,
            "window_start": window_start(p.to_dict(), S["cal"]).isoformat()}


@_json_safe
def timeline(pid: str, dpos: int | None = None, hours_before: int = 18, hours_after: int = 4) -> dict:
    S = store()
    i = _pidx(pid)
    dpos = default_now() if dpos is None else int(dpos)
    st, tw, pr = S["st"], S["tw"], S["pr"]
    ds = tw["dsteps"]
    k = int(ds[dpos])
    T5 = st["cgm"].shape[1]
    a, b = max(0, k - hours_before * 12), min(T5, k + hours_after * 12 + 1)
    times = [_clock(pid, x).strftime("%Y-%m-%dT%H:%M") for x in range(a, b)]
    hist = slice(a, k + 1)

    def arr(x, nd=1):
        return [None if (v is None or np.isnan(v)) else round(float(v), nd) for v in x]
    mk = (ds >= a) & (ds <= k)
    meals = S["meals"][(S["meals"].patient_id == pid) & (S["meals"].t_min >= a * 5) & (S["meals"].t_min <= k * 5)]
    su_idx = np.nonzero(st["su_log"][i, a:k + 1])[0] + a
    band_t = [_clock(pid, k + h).strftime("%Y-%m-%dT%H:%M") for h in BAND_H]
    band = tw["band"][i, dpos]
    # Mondrian margins: one per predicted-glucose stratum, looked up by THIS forecast's median, so
    # the band widens during a post-meal excursion instead of sitting at its average width.
    mg = pr["margins"]
    margins = np.array([float(apply_mondrian(dict(enumerate(mg[b])), band[1][b])) for b in range(len(BAND_H))]) \
        if mg.ndim == 2 else np.asarray(mg, float)
    day = k * 5 // 1440
    return {
        "id": pid, "now": _clock(pid, k).strftime("%Y-%m-%dT%H:%M"), "now_label": _clock(pid, k).strftime("%a %d %b, %H:%M"),
        "day_type": DAY_LABEL[DAY_TYPES[int(st["day_type"][i, day])]],
        "fast_window": [(_clock(pid, st["fast_start"][i, day] // 5).strftime("%Y-%m-%dT%H:%M")),
                        (_clock(pid, st["fast_end"][i, day] // 5).strftime("%Y-%m-%dT%H:%M"))] if st["fast_start"][i, day] >= 0 else None,
        "t": times[: k + 1 - a],
        "cgm": arr(st["cgm"][i, hist], 0), "truth": arr(st["g_true"][i, hist], 0),
        "fingerstick": arr(st["fingerstick"][i, hist], 0),
        "hr": arr(st["hr"][i, hist], 0), "rmssd": arr(st["rmssd"][i, hist], 1), "steps": arr(st["steps5"][i, hist], 0),
        "sleep_stage": [int(x) for x in st["stage5"][i, hist]],
        "twin_mean": arr(tw["twin_mean"][i, hist].astype(float), 0),
        "risk_t": [_clock(pid, x).strftime("%Y-%m-%dT%H:%M") for x in ds[mk]],
        "p_hypo": arr(pr["A_full_hypo"][i, mk], 3), "p_hyper": arr(pr["A_full_hyper"][i, mk], 3),
        "meals": [{"t": _clock(pid, r.t_min // 5).strftime("%Y-%m-%dT%H:%M"), "food": FOODS[r.food]["label"],
                   "carbs": r.carbs_g, "logged": bool(r.logged)} for r in meals.itertuples()],
        "su_doses": [_clock(pid, x).strftime("%Y-%m-%dT%H:%M") for x in su_idx],
        "band": {"t": band_t, "q10": arr(band[0] - margins), "q50": arr(band[1]), "q90": arr(band[2] + margins),
                 "coverage": 0.9},
        "future_truth": arr(st["g_true"][i, k + 1:b], 0), "future_t": times[k + 1 - a:],
        "sleep_eff_last_night": _f(st["sleep_eff_obs"][i, day], 2), "pm25_today": _f(st["pm25"][i, day], 0),
    }


@_json_safe
def risk(pid: str, dpos: int | None = None) -> dict:
    S = store()
    dpos = default_now() if dpos is None else int(dpos)
    i = _pidx(pid)
    row = S["feats"][(S["feats"].pidx == i) & (S["feats"].dpos == dpos)]
    out = {"id": pid, "dpos": dpos}
    for t, ms in S["models"].items():
        m = ms[S["folds"][pid]]
        if row.empty:
            continue
        base = float(m.predict(row)[0])
        drivers = []
        for g, cols in EXPLAIN_GROUPS.items():
            cols = [c for c in cols if c in m.features]
            if not cols:
                continue
            r2 = row.copy()
            for c in cols:
                r2[c] = S["neutral"].get(c, r2[c].values[0])
            drivers.append({"group": g, "delta": round(base - float(m.predict(r2)[0]), 3)})
        drivers.sort(key=lambda d: -abs(d["delta"]))
        level = "high" if base >= max(m.threshold * 2, 0.3) else "elevated" if base >= m.threshold else "low"
        out[t] = {"p": round(base, 3), "threshold": round(m.threshold, 3), "level": level, "drivers": drivers}
    return out


@_json_safe
def whatif(pid: str, dpos: int | None, scenario: dict) -> dict:
    S = store()
    i = _pidx(pid)
    dpos = default_now() if dpos is None else int(dpos)
    p = S["ehr"].loc[pid].to_dict()
    k = int(S["tw"]["dsteps"][dpos])
    day = min(k * 5 // 1440, S["tw"]["theta_daily"].shape[1] - 1)
    theta = S["tw"]["theta_daily"][i, max(day - 1, 0)].astype(float)
    state = S["tw"]["state_mean"][i, dpos].astype(float)
    cr = float(S["prior_mu"].loc[pid, "cr"])
    allowed = set(Scenario().__dict__)
    sc = Scenario(**{k_: v for k_, v in scenario.items() if k_ in allowed})
    base = Scenario(day_type=sc.day_type, su_action="usual")
    minute0 = (k * 5) % 1440
    t0 = _clock(pid, k)
    today = DAY_TYPES[int(S["st"]["day_type"][i, k * 5 // 1440])]
    if today == "navratri_phalahar" and p["vrat_style"] == "ek_samay":
        today = "navratri_ek_samay"
    res = {"baseline": rehearse(p, theta, state, minute0, base, S["prior"].fixed, cr, seed=1, today_type=today),
           "scenario": rehearse(p, theta, state, minute0, sc, S["prior"].fixed, cr, seed=1, today_type=today)}
    for r in res.values():
        r["t"] = [(t0 + timedelta(minutes=m - minute0)).strftime("%Y-%m-%dT%H:%M") for m in r.pop("minutes")]
    res["start"] = t0.strftime("%a %d %b, %H:%M")
    res["options"] = {"day_type": list(SCENARIO_DAY_TYPES), "su_action": ["usual", "half", "skip", "move_to_break_fast"]}
    return res


@_json_safe
def asha(limit: int = 60) -> dict:
    df = store()["asha"].head(limit)
    return {"rows": json.loads(df.to_json(orient="records")), "total": int(len(store()["asha"]))}


@_json_safe
def model_card() -> dict:
    return store()["metrics"]


@_json_safe
def fhir(pid: str, dpos: int | None = None) -> dict:
    S = store()
    i = _pidx(pid)
    dpos = default_now() if dpos is None else int(dpos)
    p = {"patient_id": pid, **S["ehr"].loc[pid].to_dict()}
    k = int(S["tw"]["dsteps"][dpos])
    pts = [(_clock(pid, x), int(S["st"]["cgm"][i, x])) for x in range(k - 11, k + 1, 3) if not np.isnan(S["st"]["cgm"][i, x])]
    return patient_bundle(p, risk={"p_hypo": float(np.nan_to_num(S["pr"]["A_full_hypo"][i, dpos])),
                                   "p_hyper": float(np.nan_to_num(S["pr"]["A_full_hyper"][i, dpos]))},
                          cgm_points=pts, now=_clock(pid, k))
