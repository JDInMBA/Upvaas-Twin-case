"""Adapter: run the twin on a REAL (open / anonymised) CGM recording instead of the simulator.

Input CSV (one patient), any sampling rate - it is resampled to 5 minutes:
    timestamp, glucose_mgdl[, heart_rate][, steps][, carbs_g][, su_dose_frac]
Input EHR JSON with the fields used by the prior (see data/examples/example_patient.json).

Typical sources (download yourself; licences stay with the owners):
  * ShanghaiT2DM (figshare, Zhao et al. Sci Data 2023) - Libre CGM 15-min + diet + labs, 100 T2D
  * CGMacros (PhysioNet, CC BY-NC-SA 4.0) - Libre/Dexcom + Fitbit + meal macros, 45 people
  Map their columns to the names above (a 10-line pandas script) and call `replay()`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .twin.engine import TwinEngine
from .twin.priors import PhysiologyPrior

EHR_DEFAULTS = {"sex": "M", "age": 55, "bmi": 25.5, "whtr": 0.56, "tyg": 9.2, "hba1c": 7.5, "fpg": 150,
                "diabetes_duration_y": 8, "su_dose_frac": 0.0, "basal_insulin_u": 0.0, "prs_t2d_z": 0.0,
                "tcf7l2_rs7903146_T": 0, "alt": 28, "egfr": 85, "cyp2c9": "*1/*1", "neuropathy": False,
                "on_metformin": True, "on_sulfonylurea": False, "on_basal_insulin": False, "weight_kg": 68}


def load_ehr(path_or_dict) -> pd.DataFrame:
    d = json.load(open(path_or_dict)) if isinstance(path_or_dict, str) else dict(path_or_dict)
    row = {**EHR_DEFAULTS, **d}
    return pd.DataFrame([row], index=[row.get("patient_id", "REAL-0001")])


def streams_from_csv(path: str) -> tuple[dict, pd.DatetimeIndex]:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    start = df.index[0].normalize()
    end = df.index[-1].normalize() + pd.Timedelta(1, unit="D")
    idx = pd.date_range(start, end, freq="5min", inclusive="left")
    g = df["glucose_mgdl"].resample("5min").mean().reindex(idx)
    T5 = len(idx)
    n_days = T5 // 288
    z = np.zeros((1, T5), np.float32)

    def col(name, how="sum"):
        if name not in df:
            return z.copy()
        s = getattr(df[name].resample("5min"), how)().reindex(idx).fillna(0)
        return s.values[None, :].astype(np.float32)
    carbs = col("carbs_g") * 1000.0
    st = {"cgm": g.values[None, :].astype(np.float32), "fingerstick": np.full((1, T5), np.nan, np.float32),
          "log_c": carbs, "log_k": np.where(carbs > 0, 0.02, 0).astype(np.float32), "su_log": col("su_dose_frac"),
          "plan_c": carbs, "plan_k": np.where(carbs > 0, 0.02, 0).astype(np.float32), "plan_su": col("su_dose_frac"),
          "plan_act": z.copy(), "steps5": col("steps"), "hr": col("heart_rate", "mean"),
          "sleep_eff_obs": np.full((1, n_days), 0.88, np.float32), "pm25": np.full((1, n_days), 40.0, np.float32),
          "day_type": np.zeros((1, n_days), np.int8), "fast_start": np.full((1, n_days), -1, np.int32),
          "fast_end": np.full((1, n_days), -1, np.int32)}
    return st, idx


def replay(csv_path: str, ehr_json, prior: PhysiologyPrior | None = None, P: int = 128,
           detail: bool = False) -> dict:
    """Assimilate a real CGM trace and report the twin's forecast error vs persistence.

    The twin is NOT retrained here: the EHR sets the prior and the particle filter personalises it
    from the glucose trace alone, exactly as it would for a new patient in clinic.

    With `detail=True` the per-decision predictions, persistence values and realised future readings
    are returned as arrays too, so a caller validating a whole cohort can pool them before computing
    error and Clarke zones rather than averaging per-patient RMSEs.
    """
    prior = prior or PhysiologyPrior.load()
    ehr = load_ehr(ehr_json)
    st, idx = streams_from_csv(csv_path)
    out = TwinEngine(ehr, prior, st, "A_full", P=P, S=32).run(verbose=False)
    names, ds, cgm = out["feature_names"], out["dsteps"], st["cgm"][0]
    res: dict = {}
    arrays: dict = {}
    for h, nm in ((6, "tw_g30"), (12, "tw_g60"), (24, "tw_g120")):
        ok = ds + h < len(cgm)
        pred = out["feats"][0, ok, names.index(nm)]
        fut = cgm[ds[ok] + h]
        now = cgm[ds[ok]]
        m = ~np.isnan(fut) & ~np.isnan(now) & ~np.isnan(pred)
        res[f"{h * 5}min"] = {"twin_rmse": float(np.sqrt(np.mean((pred[m] - fut[m]) ** 2))),
                              "persistence_rmse": float(np.sqrt(np.mean((now[m] - fut[m]) ** 2))), "n": int(m.sum())}
        if detail:
            arrays[f"{h * 5}min"] = {"pred": pred[m], "persistence": now[m], "actual": fut[m]}
    th = out["feats"][0, -1]
    res["personalised"] = {"SI_x1e4": float(th[names.index("tw_SI")]), "Gb0": float(th[names.index("tw_Gb0")])}
    if detail:
        res["arrays"] = arrays
    return res
