"""Counterfactual "Vrat Rehearsal" sandbox.

Before a fast, the doctor asks the twin: "If this patient keeps fasting the way she usually does
and takes her glimepiride as usual tomorrow, what happens? What if we halve it, or move it to the
fast-breaking meal, or swap the festive thali for a lower-GI plate?" The twin replays the next
36 hours on every posterior particle (= every physiology still consistent with her data) and
reports the probability of hypoglycaemia / hyperglycaemia with an uncertainty band.

This is decision SUPPORT: the output is phrased for a registered practitioner, who stays in the
loop (ICMR 2023 AI ethics guideline; keeps the tool on the "inform" side of CDSCO SaMD risk classes).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np

from ..calendar_india import DayPlan, day_plan
from ..config import THR
from ..foods import FOODS, absorption_rate
from ..physiology.model import add_meal, init_state, step
from .engine import STATE_KEYS, make_params

SCENARIO_DAY_TYPES = {
    "normal": None,
    "navratri_phalahar": ("navratri_phalahar", "phalahar"),
    "navratri_ek_samay": ("navratri_phalahar", "ek_samay"),
    "ekadashi": ("ekadashi", None),
    "karva_chauth_nirjala": ("karva_chauth_nirjala", None),
    "ramadan_roza": ("ramadan_roza", None),
}
LOW_GI_SWAPS = {"festive_thali": "roti_dal_sabzi", "sabudana_khichdi": "makhana_roasted",
                "sweet_gulab_jamun": "fruit_plate", "sweet_jalebi": "fruit_plate", "rooh_afza_sharbat": None,
                "pakora_plate": "chickpea_sundal", "singhara_halwa": "fruit_plate", "veg_biryani": "roti_dal_sabzi",
                "sargi_platter": "sehri_paratha_egg"}


@dataclass
class Scenario:
    day_type: str = "karva_chauth_nirjala"
    su_action: str = "usual"            # usual | half | skip | move_to_break_fast
    low_gi_swap: bool = False           # swap sugary / high-GI festival foods for lower-GI options
    walk_after_break_fast: bool = False # 20-min walk after breaking the fast
    garba: bool = False                 # 2 h of garba / dandiya at night
    basal_insulin_change: float = 0.0   # e.g. -0.2 = reduce basal insulin by 20 %


def scenario_plan(patient: dict, sc: Scenario, d: date) -> DayPlan:
    spec = SCENARIO_DAY_TYPES[sc.day_type]
    p = dict(patient)
    if spec is None:
        cal = {"events": []}
    else:
        typ, style = spec
        if style:
            p["vrat_style"] = style
        p["garba"] = sc.garba
        ev = {"name": "scenario", "type": typ, "start": d.isoformat(), "end": d.isoformat(),
              "traditions": [p["tradition"]]}
        cal = {"events": [ev]}
    return day_plan(d, p, cal)


def rehearse(patient: dict, theta: np.ndarray, state0: np.ndarray, minute0: int, sc: Scenario,
             fixed: dict, cr: float, hours: int = 36, seed: int = 0, today_type: str = "normal",
             model_error_sd: float = 6.0) -> dict:
    """Simulate the scenario from `minute0` (minute of day of the start) for `hours` hours.

    theta  : (P, 5) posterior particles for this patient
    state0 : (9,) twin mean state at the start time (order = STATE_KEYS)
    The scenario day is the calendar day AFTER the start time (i.e. "tomorrow").
    """
    rng = np.random.default_rng(seed)
    P = theta.shape[0]
    on_su = float(patient["on_sulfonylurea"])
    ins_basal = 0.65 * patient["basal_insulin_u"]
    par = make_params(theta[None, :, :], np.array([[on_su]]), np.array([[cr]]),
                      np.array([[patient["weight_kg"]]]), np.array([[ins_basal * (1 + sc.basal_insulin_change)]]),
                      np.array([[0.9 if patient["on_metformin"] else 1.0]]), fixed)
    s = init_state((1, P), state0[STATE_KEYS.index("G")])
    for i, key in enumerate(STATE_KEYS):
        s[key] = np.full((1, P), float(state0[i]))
    s["G"] = s["G"] + rng.normal(0, 4, (1, P))

    today = date(2026, 1, 1)
    tomorrow = today + timedelta(days=1)
    plan_today = scenario_plan(patient, Scenario(day_type=today_type if today_type in SCENARIO_DAY_TYPES else "normal"), today)
    plan_tom = scenario_plan(patient, sc, tomorrow)
    if sc.low_gi_swap:
        plan_tom.meals = [(t, LOW_GI_SWAPS.get(f, f)) for (t, f) in plan_tom.meals if LOW_GI_SWAPS.get(f, f)]

    n = hours * 12
    meal_c = np.zeros((n, P)); meal_k = np.zeros(n); su = np.zeros(n); act = np.zeros((n, P))
    def idx(day_offset, minute):  # 5-min index relative to start
        return (day_offset * 1440 + minute - minute0) // 5
    for (t, f) in plan_today.meals:
        j = idx(0, t)
        if 0 <= j < n:
            meal_c[j] += FOODS[f]["carbs"] * 1000 * rng.lognormal(0, 0.2, P)
            meal_k[j] = absorption_rate(FOODS[f]["gi"], FOODS[f]["fat"])
    for (t, f) in plan_tom.meals:
        j = idx(1, t)
        if 0 <= j < n:
            meal_c[j] += FOODS[f]["carbs"] * 1000 * rng.lognormal(0, 0.2, P)
            meal_k[j] = absorption_rate(FOODS[f]["gi"], FOODS[f]["fat"])
    dose = patient["su_dose_frac"] if on_su else 0.0
    j_today = idx(0, plan_today.su_time)
    if dose and 0 <= j_today < n:
        su[j_today] = dose
    if dose:
        if sc.su_action == "usual":
            su_t = plan_tom.su_time
        elif sc.su_action == "half":
            su_t, dose = plan_tom.su_time, dose / 2
        elif sc.su_action == "move_to_break_fast":
            su_t = plan_tom.fast_window[1] if plan_tom.fast_window else plan_tom.meals[-1][0]
        else:
            su_t = None
        if su_t is not None and 0 <= idx(1, su_t) < n:
            su[idx(1, su_t)] = dose
    if plan_tom.night_activity:
        a0, a1, spm = plan_tom.night_activity
        act[max(idx(1, a0), 0):max(idx(1, a1), 0)] = spm * 0.8
    if sc.walk_after_break_fast:
        bt = plan_tom.fast_window[1] if plan_tom.fast_window else plan_tom.meals[-1][0]
        j = idx(1, bt + 30)
        act[max(j, 0):max(j + 4, 0)] = 90
    nirjala_win = plan_tom.fast_window if plan_tom.day_type == "karva_chauth_nirjala" else None

    traj = np.zeros((n, P))
    for j in range(n):
        if meal_c[j].any():
            add_meal(s, meal_c[j][None, :], meal_k[j])
        if su[j]:
            s["A"] = s["A"] + su[j]
        minute = minute0 + j * 5
        gbm = 1.0
        if nirjala_win and 1440 + nirjala_win[0] <= minute < 1440 + nirjala_win[1]:
            gbm = 1.03
        step(s, par, 5.0, minute % 1440, act=act[j][None, :], gb_mult=gbm)
        s["G"] = np.clip(s["G"] + rng.normal(0, 1.0, (1, P)), 20, 600)
        traj[j] = s["G"][0]

    # The raw particle spread is over-confident (its 10-90% band covers well under half of outcomes,
    # see the model card), so we add a smooth AR(1) model-error process to avoid falsely certain
    # 0%/100% answers.
    #
    # HONESTY NOTE on this constant: 6 mg/dL is a fixed modelling choice, NOT fitted to the
    # evaluation it is scored against. An earlier version picked it by maximising the very
    # rehearsal AUROC it was then reported against, which is circular. The pipeline now selects it
    # on a held-out half of the fasters and scores the other half (see "rehearsal_validation" in
    # reports/metrics.json, which reports both halves and the number of events behind them).
    sd = model_error_sd
    if sd > 0:
        phi = 0.97
        err = np.zeros_like(traj)
        e = rng.normal(0, sd, P)
        for j in range(n):
            e = phi * e + rng.normal(0, sd * np.sqrt(1 - phi ** 2), P)
            err[j] = e
        traj = np.clip(traj + err, 20, 600)

    # risks are scored over the scenario day ("tomorrow", 00:00 until the end of the horizon)
    t_min = minute0 + np.arange(n) * 5
    tom = t_min >= 1440
    run = np.zeros(P, int); hypo = np.zeros(P, bool); severe = np.zeros(P, bool)
    for j in np.where(tom)[0]:
        run = np.where(traj[j] < THR.hypo, run + 1, 0)
        hypo |= run >= THR.min_event_minutes // 5
        severe |= traj[j] < THR.severe_hypo
    hyper = (traj[tom] > THR.hyper).sum(0) >= 3
    q = np.quantile(traj, [0.1, 0.5, 0.9], axis=1)
    nadir_j = int(np.where(tom)[0][np.argmin(q[1][tom])]) if tom.any() else int(np.argmin(q[1]))
    return {"scenario": asdict(sc), "minutes": t_min.tolist(),
            "q10": q[0].round(1).tolist(), "q50": q[1].round(1).tolist(), "q90": q[2].round(1).tolist(),
            "p_hypo": float(hypo.mean()), "p_severe_hypo": float(severe.mean()), "p_hyper": float(hyper.mean()),
            "nadir_mgdl": float(q[1][nadir_j]), "nadir_clock": f"{(t_min[nadir_j] // 60) % 24:02d}:{t_min[nadir_j] % 60:02d}",
            "nadir_day": "tomorrow" if t_min[nadir_j] >= 1440 else "today",
            "meals_tomorrow": [{"time": f"{t // 60:02d}:{t % 60:02d}", "food": FOODS[f]["label"]} for t, f in plan_tom.meals],
            "su_time_tomorrow": None if not dose or sc.su_action == "skip" else sc.su_action}
