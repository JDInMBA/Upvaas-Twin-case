"""Virtual-body simulator: turns the synthetic EHR + festival calendar into 28 days of
minute-level physiology and the wearable/IoT streams a patient would actually produce.

The body is integrated with `truth_model.py`, which is deliberately a DIFFERENT model from the one
the twin uses, so the evaluation measures model misspecification and not just parameter error.
Eating is deliberately messy too: wide meal-time jitter, large portion variability, skipped meals
and unplanned snacks, so the festival calendar is a useful prior rather than an oracle. Both
behaviours are switched by `SIM.model_mismatch` / `SIM.realistic_meals` and the pipeline reports
what each one costs.

Outputs (5-minute grid, shape [patients, T5]):
  g_true       true plasma glucose (hidden from the twin; used only for labels)
  cgm          Libre/Dexcom-like CGM: 8-min interstitial lag, AR(1) noise, bias, dropouts
  fingerstick  two capillary readings per day (07:30, 20:30) for the "frugal" tier
  hr, rmssd    smartwatch heart rate and HRV; both react to activity, sleep and hypoglycaemia
  steps5       steps per 5-minute bin
  stage5       sleep stage (0 wake, 1 light, 2 deep, 3 REM)
plus meal / medication logs, the twin's *planned* inputs (from the calendar) and day metadata.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..calendar_india import day_plan, hm, load_calendar, window_days
from ..config import COHORT_DIR, PM25_GLUCOSE_PER_UG, SIM
from ..foods import FOODS, SNACKS, absorption_rate
from . import model as twin_model
from . import truth_model
from .model import params_from_truth, stack_params

FASTING_TYPES = {"navratri_phalahar", "ekadashi", "karva_chauth_nirjala", "ramadan_roza"}
DAY_TYPES = ["normal", "navratri_phalahar", "ekadashi", "karva_chauth_nirjala", "ramadan_roza", "feast"]

# Unplanned eating. Real people eat things the calendar does not predict: office samosas, a
# neighbour's prasad, tea with biscuits. Without this the festival meal template is a near-perfect
# preview of future carbohydrate and the prediction task becomes a timetable lookup.
UNPLANNED_PER_DAY = 1.5        # Poisson mean extra eating occasions on a non-fasting day
MEAL_JITTER_MIN = 40.0         # SD of meal timing (min) on an ordinary day
PORTION_LOG_SD = 0.35          # lognormal SD of portion size
SKIP_MEAL_P, SKIP_SNACK_P = 0.12, 0.30


def _season_factor(region: str, d: date) -> float:
    """Post-monsoon / winter smog multiplier for PM2.5 [assumption]."""
    if region in ("north", "east") and d.month in (10, 11, 12, 1, 2):
        return 1.5
    return 1.0


def build_schedules(ehr: pd.DataFrame, cal: dict, n_days: int, rng: np.random.Generator) -> dict:
    """Realise meals, medication, activity and sleep for every patient-minute."""
    N, T = len(ehr), n_days * 1440
    T5 = T // 5
    meal_c = np.zeros((N, T), np.float32)
    meal_k = np.zeros((N, T), np.float32)
    su_dose = np.zeros((N, T), np.float32)
    act = np.zeros((N, T), np.float32)
    asleep = np.zeros((N, T), bool)
    stage = np.zeros((N, T), np.int8)
    gb_mult = np.ones((N, T), np.float32)
    # 5-minute arrays the twin is allowed to see
    log_c = np.zeros((N, T5), np.float32); log_k = np.zeros((N, T5), np.float32)
    plan_c = np.zeros((N, T5), np.float32); plan_k = np.zeros((N, T5), np.float32)
    plan_su = np.zeros((N, T5), np.float32); plan_act = np.zeros((N, T5), np.float32)
    su_log = np.zeros((N, T5), np.float32)
    day_type = np.zeros((N, n_days), np.int8)
    fast_start = np.full((N, n_days), -1, np.int32); fast_end = np.full((N, n_days), -1, np.int32)
    sleep_eff = np.full((N, n_days), 0.9, np.float32)
    pm25 = np.zeros((N, n_days), np.float32)
    meals_table = []

    for i, (pid, pt) in enumerate(ehr.iterrows()):
        p = pt.to_dict()
        days = window_days(p, cal, n_days)
        plans = [day_plan(d, p, cal) for d in days]
        walker = rng.random() < (0.45 if p["urban"] else 0.25)
        worker = p["age"] < 60 and rng.random() < (0.6 if p["urban"] else 0.4)
        for di, (d, pl) in enumerate(zip(days, plans)):
            base = di * 1440
            day_type[i, di] = DAY_TYPES.index(pl.day_type)
            is_fast = pl.day_type in FASTING_TYPES
            if pl.fast_window:
                fast_start[i, di] = base + pl.fast_window[0]; fast_end[i, di] = base + pl.fast_window[1]
            pm25[i, di] = p["pm25_annual"] * _season_factor(p["region"], d) * rng.lognormal(0, 0.3)
            gmul = 1 + PM25_GLUCOSE_PER_UG * (pm25[i, di] - 40) / 150.0
            gb_mult[i, base:base + 1440] *= gmul
            if pl.day_type == "karva_chauth_nirjala" and pl.fast_window:   # dehydration [low-confidence prior]
                gb_mult[i, base + pl.fast_window[0]: base + pl.fast_window[1]] *= 1.03

            # ---- meals: realised (truth) and logged (what the app receives) -------------
            real = SIM.realistic_meals
            jit_sd = MEAL_JITTER_MIN if real else 15.0
            port_sd = PORTION_LOG_SD if real else 0.18
            skip_m, skip_s = (SKIP_MEAL_P, SKIP_SNACK_P) if real else (0.04, 0.15)
            meals = list(pl.meals)
            in_hard_fast = bool(pl.fast_window and pl.day_type in ("karva_chauth_nirjala", "ramadan_roza"))
            if not in_hard_fast:
                if pl.day_type in ("normal", "ekadashi") and rng.random() < 0.3:
                    meals.append((hm("11:30") + int(rng.integers(-60, 60)), str(rng.choice(SNACKS))))
                if pl.day_type in ("navratri_phalahar", "feast") and rng.random() < 0.45 and \
                        p["vrat_style"] != "ek_samay":
                    meals.append((hm("18:30") + int(rng.integers(-60, 90)), str(rng.choice(["sweet_gulab_jamun", "sweet_jalebi", "kheer"]))))
                if real:
                    # eating occasions the calendar never predicted, at any waking hour
                    for _ in range(rng.poisson(UNPLANNED_PER_DAY)):
                        meals.append((int(rng.integers(hm("07:00"), hm("22:30"))),
                                      str(rng.choice(SNACKS + ["masala_chai_sugar", "samosa", "pakora_plate",
                                                               "sweet_jalebi", "banana"]))))
            for (mt, food) in meals:
                f = FOODS[food]
                is_snack = f["carbs"] <= 20 or food.startswith("sweet")
                if rng.random() < (skip_s if is_snack else skip_m) and food not in ("dates_x3", "sargi_platter"):
                    continue
                # the fast-breaking meal is socially fixed (moonrise / iftar); everything else drifts
                near_break = bool(pl.fast_window and abs(mt - pl.fast_window[1]) < 60)
                jitter = rng.normal(0, 8.0 if near_break else jit_sd)
                t_abs = int(np.clip(base + mt + jitter, 0, T - 1))
                carbs = f["carbs"] * rng.lognormal(0, port_sd) * 1000.0      # mg
                k = absorption_rate(f["gi"], f["fat"])
                meal_c[i, t_abs] += carbs; meal_k[i, t_abs] = k
                logged = rng.random() < ((0.50 if food.startswith("sweet") else 0.75) if real
                                         else (0.6 if food.startswith("sweet") else 0.88))
                if logged:
                    t5 = min(max(t_abs + int(rng.normal(0, 8)), 0) // 5, T5 - 1)
                    log_c[i, t5] += carbs * rng.lognormal(0, 0.25); log_k[i, t5] = k
                meals_table.append({"patient_id": pid, "t_min": t_abs, "food": food, "carbs_g": round(carbs / 1000, 1),
                                    "logged": bool(logged), "day": di})
            for (mt, food) in pl.meals:          # the twin's expectation (template, no noise)
                t5 = min((base + mt) // 5, T5 - 1)
                plan_c[i, t5] += FOODS[food]["carbs"] * 1000.0
                plan_k[i, t5] = absorption_rate(FOODS[food]["gi"], FOODS[food]["fat"])

            # ---- sulfonylurea --------------------------------------------------------------
            if p["on_sulfonylurea"]:
                plan_su[i, min((base + pl.su_time) // 5, T5 - 1)] = p["su_dose_frac"]
                if is_fast:
                    beh = p["su_fasting_behaviour"]
                    t_su = None if beh == "skip" else (pl.meals[0][0] if beh == "with_predawn_meal" else pl.su_time)
                else:
                    t_su = pl.su_time if rng.random() < 0.94 else None
                if t_su is not None:
                    t_abs = int(np.clip(base + t_su + rng.normal(0, 15), 0, T - 1))
                    su_dose[i, t_abs] = p["su_dose_frac"]
                    su_log[i, t_abs // 5] = p["su_dose_frac"]

            # ---- activity --------------------------------------------------------------------
            sl = slice(base, base + 1440)
            bouts = (rng.random(1440) < 0.12) * rng.uniform(15, 80, 1440)
            a = bouts.astype(np.float32)
            if walker and not (is_fast and rng.random() < 0.5):
                a[hm("06:45"):hm("07:30")] = rng.normal(100, 8)
            if worker and pl.day_type != "feast":
                a[hm("09:15"):hm("09:45")] = 75; a[hm("18:15"):hm("18:45")] = 75
            if pl.day_type == "ramadan_roza":
                a[hm("20:00"):hm("21:00")] = np.maximum(a[hm("20:00"):hm("21:00")], 25)
            a *= pl.activity_scale
            act[i, sl] = a
            if pl.night_activity:
                s0, s1, spm = pl.night_activity
                if rng.random() < 0.85:
                    e = min(base + s1, T)
                    act[i, base + s0:e] = spm * rng.uniform(0.8, 1.15)
                pa0, pa1 = (base + s0) // 5, min((base + s1) // 5, T5)
                plan_act[i, pa0:pa1] = spm

        # ---- sleep: night d runs from bed(d) to wake(d+1) -----------------------------------
        bed_prev_end = plans[0].wake + int(rng.normal(0, 20))
        asleep[i, 0:bed_prev_end] = True
        for di in range(n_days):
            pl = plans[di]
            nxt = plans[di + 1] if di + 1 < n_days else plans[di]
            b = di * 1440 + pl.bed + int(rng.normal(0, 30))
            w = (di + 1) * 1440 + nxt.wake + int(rng.normal(0, 20))
            if b >= T:
                continue
            w = min(w, T)
            poor = rng.random() < 0.15
            latency = int(rng.uniform(5, 30 if not poor else 60))
            asleep[i, b + latency:w] = True
            for _ in range(rng.poisson(5 if poor else 2)):
                s = int(rng.integers(b, max(b + 1, w - 10)))
                asleep[i, s:s + int(rng.uniform(3, 25 if poor else 12))] = False
            if nxt.extra_wake and di + 1 < n_days:
                ew0, ew1 = (di + 1) * 1440 + nxt.extra_wake[0], (di + 1) * 1440 + nxt.extra_wake[1]
                asleep[i, ew0:ew1] = False
            in_bed = max(w - b, 1)
            se = asleep[i, b:w].sum() / in_bed
            if di + 1 < n_days:
                sleep_eff[i, di + 1] = se
            # stages
            idx = np.arange(b, w)
            c = (idx - b) % 90
            first_half = (idx - b) < (w - b) / 2
            st = np.ones_like(idx, dtype=np.int8)
            st[first_half & (c >= 20) & (c < 50)] = 2
            st[~first_half & (c >= 25) & (c < 35)] = 2
            st[(first_half & (c >= 70)) | (~first_half & (c >= 55))] = 3
            st[~asleep[i, b:w]] = 0
            stage[i, b:w] = st
        act[i, asleep[i]] = 0.0

    return dict(meal_c=meal_c, meal_k=meal_k, su_dose=su_dose, act=act, asleep=asleep, stage=stage,
                gb_mult=gb_mult, log_c=log_c, log_k=log_k, plan_c=plan_c, plan_k=plan_k, plan_su=plan_su,
                plan_act=plan_act, su_log=su_log, day_type=day_type, fast_start=fast_start, fast_end=fast_end,
                sleep_eff=sleep_eff, pm25=pm25, meals_table=pd.DataFrame(meals_table))


def simulate_cohort(ehr: pd.DataFrame, truth: pd.DataFrame, n_days: int = SIM.n_days, seed: int = SIM.seed + 1,
                    save: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    cal = load_calendar()
    N, T = len(ehr), n_days * 1440
    T5 = T // 5
    sch = build_schedules(ehr, cal, n_days, rng)

    P = stack_params([params_from_truth(truth.loc[pid].to_dict(), bool(ehr.loc[pid, "on_metformin"]))
                      for pid in ehr.index])
    tr = truth.loc[ehr.index]
    si_day = 1.0 - 0.6 * tr["sleep_sens"].values[:, None] * np.maximum(0.0, 0.88 - sch["sleep_eff"])
    ill_start = tr["illness_start_day"].values
    ill = np.zeros((N, n_days), bool)
    for i, s in enumerate(ill_start):
        if s >= 0:
            ill[i, s:s + 4] = True
    si_day = np.where(ill, si_day * 0.65, si_day)
    gb_day = np.where(ill, 1.12, 1.0)

    # The body runs truth_model (four terms the twin has no equation for) unless the matched-model
    # ablation is requested, in which case it runs the twin's own equations.
    body = truth_model if SIM.model_mismatch else twin_model

    # warm start at each patient's fasting equilibrium
    s = body.init_state((N,), P["Gb0"] * P["met"], I0=P["ins_basal"])
    g_true = np.zeros((N, T5), np.float32); g_isf = np.zeros((N, T5), np.float32)
    ra5 = np.zeros((N, T5), np.float32); i5 = np.zeros((N, T5), np.float32); c5 = np.zeros((N, T5), np.float32)
    gi = s["G"].copy()                    # interstitial glucose (CGM site)
    for t in range(T):
        d = t // 1440
        mc = sch["meal_c"][:, t]
        if mc.any():
            body.add_meal(s, mc, sch["meal_k"][:, t])
        sd = sch["su_dose"][:, t]
        if sd.any():
            s["A"] = s["A"] + sd
        ra = body.step(s, P, 1.0, t % 1440, act=sch["act"][:, t], si_mult=si_day[:, d],
                       gb_mult=sch["gb_mult"][:, t] * gb_day[:, d])
        gi += (s["G"] - gi) / 8.0
        if t % 5 == 4:
            k = t // 5
            g_true[:, k] = s["G"]; g_isf[:, k] = gi; ra5[:, k] = ra; i5[:, k] = s["I"]; c5[:, k] = s["C"]

    # ---- sensors ----------------------------------------------------------------------------
    cgm = np.empty_like(g_isf)
    bias = rng.normal(1.0, 0.04, (N, 1))
    e = np.zeros(N)
    for k in range(T5):
        e = 0.7 * e + rng.normal(0, 5.0, N)
        cgm[:, k] = g_isf[:, k] * bias[:, 0] + e
    cgm = np.clip(np.round(cgm), 40, 400).astype(np.float32)
    for i in range(N):                     # signal dropouts (compression lows, Bluetooth loss)
        for _ in range(rng.poisson(4)):
            s0 = int(rng.integers(0, T5 - 30)); cgm[i, s0:s0 + int(rng.integers(2, 24))] = np.nan
    fs = np.full((N, T5), np.nan, np.float32)
    for dd in range(n_days):
        for hhmm in ("07:30", "20:30"):
            k = (dd * 1440 + hm(hhmm)) // 5
            fs[:, k] = g_true[:, k] * rng.normal(1.0, 0.07, N)

    act5 = sch["act"].reshape(N, T5, 5)
    steps5 = act5.sum(-1)
    asleep5 = sch["asleep"].reshape(N, T5, 5).mean(-1)
    stage5 = sch["stage"].reshape(N, T5, 5)[:, :, 2]
    rhr = ehr["resting_hr"].values[:, None].astype(float)
    hypo_sig = 1.0 / (1.0 + np.exp(np.clip((g_true - 68.0) / 4.0, -50, 50)))
    post = np.clip(ra5 / 3.0, 0, 1.5)
    hr = rhr - 9 * asleep5 + 0.33 * steps5 / 5 + 5 * post + tr["hr_hypo"].values[:, None] * hypo_sig
    hr = np.round(hr + rng.normal(0, 2.5, hr.shape)).astype(np.float32)
    rm = tr["rmssd_base"].values[:, None] * (1 + 0.35 * asleep5) * np.exp(-steps5 / 5 / 70.0) \
        * (1 - 0.35 * hypo_sig) * rng.lognormal(0, 0.15, hr.shape)
    rmssd = np.round(rm, 1).astype(np.float32)
    se_obs = np.clip(sch["sleep_eff"] + rng.normal(0, 0.03, sch["sleep_eff"].shape), 0.3, 1.0)

    out = dict(g_true=g_true, cgm=cgm, fingerstick=fs, hr=hr, rmssd=rmssd, steps5=steps5.astype(np.float32),
               stage5=stage5, asleep5=asleep5.astype(np.float32), insulin5=i5, su_conc5=c5,
               log_c=sch["log_c"], log_k=sch["log_k"], plan_c=sch["plan_c"], plan_k=sch["plan_k"],
               plan_su=sch["plan_su"], plan_act=sch["plan_act"], su_log=sch["su_log"],
               day_type=sch["day_type"], fast_start=sch["fast_start"], fast_end=sch["fast_end"],
               sleep_eff_obs=se_obs.astype(np.float32), pm25=sch["pm25"], illness=ill)
    if save:
        COHORT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(COHORT_DIR / "streams.npz", **out)
        sch["meals_table"].to_csv(COHORT_DIR / "meals_log.csv", index=False)
    out["meals_table"] = sch["meals_table"]
    return out
