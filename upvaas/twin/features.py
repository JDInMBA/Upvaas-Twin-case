"""Feature fusion: EHR (static) + wearable streams (dynamic) + festival calendar + twin outputs.

One row = one patient at one decision time (every 30 min). Labels look 2 hours ahead:
  y_hypo  : true glucose < 70 mg/dL sustained >= 15 min starting within the next 120 min
  y_hyper : true glucose > 250 mg/dL sustained >= 15 min within the next 120 min
Rows where the event is already happening are excluded (we predict onsets, not ongoing events).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..config import THR
from ..physiology.simulate import DAY_TYPES
from .priors import cyp_activity

warnings.filterwarnings("ignore", category=RuntimeWarning)

FEATURE_GROUPS = {
    "glucose_stream": ["cgm_now", "cgm_slope15", "cgm_slope30", "cgm_min_2h", "cgm_max_2h", "cgm_mean_24h",
                       "cgm_cv_24h", "cgm_tbr_24h", "fs_last", "fs_age_min"],
    "wearable": ["hr_dev", "rmssd_ratio", "steps_30", "steps_60", "sleep_eff_last"],
    "meal_med": ["carbs_3h", "min_since_meal", "su_today", "min_since_su"],
    "calendar": [f"dt_{d}" for d in DAY_TYPES] + ["is_fast_day", "min_into_fast", "min_to_fast_break",
                                                 "plan_carbs_next2h", "plan_activity_next2h", "hour_sin", "hour_cos"],
    "ehr": ["age", "diabetes_duration_y", "hba1c", "egfr", "on_su", "su_dose_frac", "on_ins", "basal_insulin_u",
            "cyp2c9_activity", "tcf7l2", "neuropathy", "bmi", "whtr", "prior_hypo_12m", "on_met",
            "su_fast_skip", "su_fast_predawn"],
    "twin": ["tw_g_now", "tw_g30", "tw_g60", "tw_g120", "tw_q10_120", "tw_q90_120", "tw_min", "tw_max",
             "tw_p_hypo", "tw_p_hyper", "tw_t_min", "tw_su_conc", "tw_insulin", "tw_X", "tw_fast_clock",
             "tw_SI", "tw_beta", "tw_Gb0", "tw_su_thalf", "tw_SI_sd", "tw_Gb0_sd", "tw_nis"],
}

TIER_GROUPS = {
    "A_full": ["glucose_stream", "wearable", "meal_med", "calendar", "ehr", "twin"],
    "B_frugal": ["glucose_stream", "wearable", "meal_med", "calendar", "ehr", "twin"],
    "C_minimal": ["meal_med", "calendar", "ehr", "twin"],
}


def sustained(mask: np.ndarray, n: int) -> np.ndarray:
    """True where mask is part of a run of >= n consecutive True values (along axis 1)."""
    out = np.zeros_like(mask)
    run = np.zeros(mask.shape[0], int)
    for k in range(mask.shape[1]):
        run = np.where(mask[:, k], run + 1, 0)
        hit = run >= n
        if hit.any():
            for b in range(n):
                out[hit, k - b] = True
    return out


def labels(g_true: np.ndarray, dsteps: np.ndarray) -> dict:
    n = THR.min_event_minutes // 5
    H = THR.horizon_min // 5
    sh = sustained(g_true < THR.hypo, n)
    sy = sustained(g_true > THR.hyper, n)
    T5 = g_true.shape[1]
    csh = np.concatenate([np.zeros((len(g_true), 1)), np.cumsum(sh, 1)], 1)
    csy = np.concatenate([np.zeros((len(g_true), 1)), np.cumsum(sy, 1)], 1)
    k = dsteps
    valid = k + H < T5
    kk = np.minimum(k + H, T5 - 1)
    y_hypo = (csh[:, kk + 1] - csh[:, k + 1]) > 0
    y_hyper = (csy[:, kk + 1] - csy[:, k + 1]) > 0
    ok_hypo = valid[None, :] & (g_true[:, k] >= THR.hypo)
    ok_hyper = valid[None, :] & (g_true[:, k] <= THR.hyper)
    # time to onset (minutes) for lead-time analysis
    return {"y_hypo": y_hypo, "y_hyper": y_hyper, "ok_hypo": ok_hypo, "ok_hyper": ok_hyper,
            "sus_hypo": sh, "sus_hyper": sy}


def _last_valid(a: np.ndarray, k: int, lookback: int):
    seg = a[:, max(0, k - lookback + 1):k + 1]
    val = np.full(a.shape[0], np.nan)
    age = np.full(a.shape[0], np.nan)
    for j in range(seg.shape[1] - 1, -1, -1):
        m = np.isnan(val) & ~np.isnan(seg[:, j])
        val[m] = seg[m, j]
        age[m] = (seg.shape[1] - 1 - j) * 5
    return val, age


def build_features(ehr: pd.DataFrame, st: dict, twin: dict, tier: str) -> pd.DataFrame:
    N = len(ehr)
    ds = twin["dsteps"]
    D = len(ds)
    cols: dict[str, np.ndarray] = {}

    def put(name, arr):
        cols[name] = np.asarray(arr, dtype=np.float32).reshape(N, D)

    cgm, fs = st["cgm"], st["fingerstick"]
    use_cgm = tier == "A_full"
    use_fs = tier == "B_frugal"
    use_watch = tier in ("A_full", "B_frugal")
    meal_c = st["plan_c"] if tier == "C_minimal" else st["log_c"]
    su = st["plan_su"] if tier == "C_minimal" else st["su_log"]

    blocks = {k: np.full((N, D), np.nan, np.float32) for g in ("glucose_stream", "wearable", "meal_med")
              for k in FEATURE_GROUPS[g]}
    hr, rm, steps = st["hr"], st["rmssd"], st["steps5"]
    for j, k in enumerate(ds):
        if use_cgm:
            now, _ = _last_valid(cgm, k, 3)
            b15, _ = _last_valid(cgm, k - 3, 2)
            b30, _ = _last_valid(cgm, k - 6, 2)
            w2 = cgm[:, k - 23:k + 1]
            w24 = cgm[:, max(0, k - 287):k + 1]
            blocks["cgm_now"][:, j] = now
            blocks["cgm_slope15"][:, j] = (now - b15) / 15.0
            blocks["cgm_slope30"][:, j] = (now - b30) / 30.0
            blocks["cgm_min_2h"][:, j] = np.nanmin(w2, 1)
            blocks["cgm_max_2h"][:, j] = np.nanmax(w2, 1)
            mu24 = np.nanmean(w24, 1)
            blocks["cgm_mean_24h"][:, j] = mu24
            blocks["cgm_cv_24h"][:, j] = np.nanstd(w24, 1) / mu24
            blocks["cgm_tbr_24h"][:, j] = np.nanmean(w24 < THR.hypo, 1)
        if use_fs:
            v, a = _last_valid(fs, k, 288)
            blocks["fs_last"][:, j] = v
            blocks["fs_age_min"][:, j] = a
        if use_watch:
            h15 = hr[:, k - 2:k + 1].mean(1); h24 = hr[:, max(0, k - 287):k + 1].mean(1)
            blocks["hr_dev"][:, j] = h15 - h24
            blocks["rmssd_ratio"][:, j] = rm[:, k - 2:k + 1].mean(1) / rm[:, max(0, k - 287):k + 1].mean(1)
            blocks["sleep_eff_last"][:, j] = st["sleep_eff_obs"][:, (k * 5) // 1440]
        blocks["steps_30"][:, j] = steps[:, k - 5:k + 1].sum(1) * (1.0 if use_watch else 0.75)
        blocks["steps_60"][:, j] = steps[:, k - 11:k + 1].sum(1) * (1.0 if use_watch else 0.75)
        blocks["carbs_3h"][:, j] = meal_c[:, k - 35:k + 1].sum(1) / 1000.0
        last_meal = np.array([np.max(np.nonzero(r)[0]) if r.any() else -999 for r in (meal_c[:, max(0, k - 287):k + 1] > 0)])
        blocks["min_since_meal"][:, j] = np.where(last_meal >= 0, (min(k, 287) - last_meal) * 5, 1440)
        d0 = (k // 288) * 288
        blocks["su_today"][:, j] = su[:, d0:k + 1].sum(1)
        last_su = np.array([np.max(np.nonzero(r)[0]) if r.any() else -999 for r in (su[:, max(0, k - 575):k + 1] > 0)])
        blocks["min_since_su"][:, j] = np.where(last_su >= 0, (min(k, 575) - last_su) * 5, 2880)
    cols.update(blocks)

    # ---- calendar ------------------------------------------------------------------------------
    minute = ds * 5
    day = minute // 1440
    dt = st["day_type"][:, day]
    for i, name in enumerate(DAY_TYPES):
        put(f"dt_{name}", (dt == i).astype(float))
    put("is_fast_day", np.isin(dt, [1, 2, 3, 4]).astype(float))
    fs_, fe_ = st["fast_start"][:, day], st["fast_end"][:, day]
    in_fast = (fs_ >= 0) & (minute[None, :] >= fs_) & (minute[None, :] < fe_)
    put("min_into_fast", np.where(in_fast, minute[None, :] - fs_, -1))
    put("min_to_fast_break", np.where(in_fast, fe_ - minute[None, :], -1))
    H = THR.horizon_min // 5
    pc = np.cumsum(np.concatenate([np.zeros((N, 1)), st["plan_c"]], 1), 1)
    pa = np.cumsum(np.concatenate([np.zeros((N, 1)), st["plan_act"]], 1), 1)
    kk = np.minimum(ds + H, st["plan_c"].shape[1] - 1)
    put("plan_carbs_next2h", (pc[:, kk + 1] - pc[:, ds + 1]) / 1000.0)
    put("plan_activity_next2h", (pa[:, kk + 1] - pa[:, ds + 1]))
    hod = (minute % 1440) / 1440.0
    put("hour_sin", np.tile(np.sin(2 * np.pi * hod), (N, 1)))
    put("hour_cos", np.tile(np.cos(2 * np.pi * hod), (N, 1)))

    # ---- EHR (static, broadcast over time) ---------------------------------------------------------
    e = ehr
    static = {
        "age": e["age"], "diabetes_duration_y": e["diabetes_duration_y"], "hba1c": e["hba1c"], "egfr": e["egfr"],
        "on_su": e["on_sulfonylurea"], "su_dose_frac": e["su_dose_frac"], "on_ins": e["on_basal_insulin"],
        "basal_insulin_u": e["basal_insulin_u"], "cyp2c9_activity": e["cyp2c9"].map(cyp_activity),
        "tcf7l2": e["tcf7l2_rs7903146_T"], "neuropathy": e["neuropathy"], "bmi": e["bmi"], "whtr": e["whtr"],
        "prior_hypo_12m": e["prior_hypo_12m"], "on_met": e["on_metformin"],
        "su_fast_skip": (e["su_fasting_behaviour"] == "skip") & e["on_sulfonylurea"],
        "su_fast_predawn": (e["su_fasting_behaviour"] == "with_predawn_meal") & e["on_sulfonylurea"],
    }
    for name, v in static.items():
        put(name, np.repeat(v.values.astype(float)[:, None], D, 1))

    # ---- twin --------------------------------------------------------------------------------------
    for i, name in enumerate(twin["feature_names"]):
        put(name, twin["feats"][:, :, i])

    lab = labels(st["g_true"], ds)
    df = pd.DataFrame({k: v.ravel() for k, v in cols.items()})
    df.insert(0, "pidx", np.repeat(np.arange(N), D))
    df.insert(1, "patient_id", np.repeat(ehr.index.values, D))
    df.insert(2, "dpos", np.tile(np.arange(D), N))
    df.insert(3, "k", np.tile(ds, N))
    df["day_type"] = np.array(DAY_TYPES)[dt.ravel()]
    for key in ("y_hypo", "y_hyper", "ok_hypo", "ok_hyper"):
        df[key] = lab[key].ravel()
    return df


def feature_list(tier: str, drop_groups: tuple = ()) -> list[str]:
    out = []
    for g in TIER_GROUPS[tier]:
        if g in drop_groups:
            continue
        cols = FEATURE_GROUPS[g]
        if g == "glucose_stream":
            cols = [c for c in cols if (c.startswith("cgm") if tier == "A_full" else c.startswith("fs"))]
        if g == "twin" and tier == "C_minimal":
            cols = [c for c in cols if c != "tw_nis"]   # no observations -> no innovation to measure
        out += cols
    # Tier C has no watch, but a phone still counts steps.
    if tier == "C_minimal" and "wearable" not in drop_groups:
        out += ["steps_30", "steps_60"]
    return list(dict.fromkeys(out))
