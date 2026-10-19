"""Festival & fasting calendar engine.

The key idea of UPVAAS: in India, many high-risk physiological states are *scheduled*. Navratri,
Ekadashi, Karva Chauth and Ramadan change meal timing, meal composition, sleep and activity in a
way that is known days in advance. This module turns the calendar into day templates that both
the body simulator (truth) and the twin (expectation) use.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta

from .config import CALENDAR_FILE


def hm(s: str) -> int:
    """'HH:MM' -> minutes after midnight."""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


@dataclass
class DayPlan:
    day_type: str
    meals: list = field(default_factory=list)       # list of (minute_of_day, food_key)
    su_time: int | None = hm("08:30")                # planned sulfonylurea intake time
    fast_window: tuple | None = None                 # (start_min, end_min) of no-food period
    bed: int = hm("23:00")                           # bedtime (minute of day, may be > 1440)
    wake: int = hm("06:30")                          # wake time next morning
    extra_wake: tuple | None = None                  # e.g. pre-dawn sehri/sargi wake window
    night_activity: tuple | None = None              # (start, end, steps_per_min) e.g. garba
    activity_scale: float = 1.0


def load_calendar(path=CALENDAR_FILE) -> dict:
    with open(path) as f:
        return json.load(f)


def events_on(day: date, tradition: str, cal: dict) -> list[dict]:
    out = []
    for ev in cal["events"]:
        s, e = date.fromisoformat(ev["start"]), date.fromisoformat(ev["end"])
        if s <= day <= e and tradition in ev["traditions"]:
            out.append(ev)
    return out


def day_plan(day: date, patient: dict, cal: dict) -> DayPlan:
    """Expected (template) plan for one patient-day. The simulator adds noise on top."""
    tradition = patient["tradition"]
    evs = events_on(day, tradition, cal)
    types = {e["type"] for e in evs}
    region = patient.get("region", "north")
    breakfast = {"south": "idli_sambar", "west": "poha", "east": "poha", "north": "aloo_paratha"}[region]
    lunch = {"south": "rice_dal_sabzi", "west": "roti_dal_sabzi", "east": "rice_dal_sabzi",
             "north": "roti_dal_sabzi"}[region]
    if patient.get("millet_eater"):
        lunch = "ragi_mudde" if region == "south" else "jowar_roti_sabzi"

    # ----- Karva Chauth: nirjala (no food or water) from pre-dawn sargi to moonrise --------------
    if "karva_chauth_nirjala" in types:
        ev = next(e for e in evs if e["type"] == "karva_chauth_nirjala")
        brk = hm(ev.get("fast_break_time", "20:15"))
        return DayPlan("karva_chauth_nirjala",
                       meals=[(hm("04:30"), "sargi_platter"), (brk + 15, "festive_thali"),
                              (brk + 45, "sweet_gulab_jamun")],
                       su_time=hm("08:30"), fast_window=(hm("05:00"), brk),
                       extra_wake=(hm("04:15"), hm("05:00")), bed=hm("23:45"), activity_scale=0.85)

    # ----- Ramadan roza: sehri before dawn, iftar at sunset -------------------------------------
    if "ramadan_roza" in types:
        ev = next(e for e in evs if e["type"] == "ramadan_roza")
        sehri, iftar = hm(ev.get("sehri_time", "05:15")), hm(ev.get("iftar_time", "18:20"))
        return DayPlan("ramadan_roza",
                       meals=[(sehri - 30, "sehri_paratha_egg"), (iftar, "dates_x3"),
                              (iftar + 5, "rooh_afza_sharbat"), (iftar + 20, "pakora_plate"),
                              (hm("21:15"), "veg_biryani")],
                       su_time=hm("08:30"), fast_window=(sehri, iftar),
                       extra_wake=(sehri - 60, sehri + 10), bed=hm("23:30"), activity_scale=0.8)

    # ----- Navratri: phalahar (fruit/sabudana) or ek-samay (one meal after evening aarti) --------
    if "navratri_phalahar" in types:
        ev = next(e for e in evs if e["type"] == "navratri_phalahar")
        night = None
        if patient.get("garba"):
            night = (hm("21:30"), hm("23:45"), 85)
        if patient.get("vrat_style") == "ek_samay":
            meals = [(hm("07:00"), "masala_chai_sugar"), (hm("16:30"), "masala_chai_sugar"),
                     (hm("20:30"), "kuttu_puri_aloo"), (hm("20:50"), "singhara_halwa")]
            fw = (hm("07:15"), hm("20:30"))
        else:
            meals = [(hm("07:00"), "masala_chai_sugar"), (hm("11:00"), "fruit_plate"),
                     (hm("13:30"), "sabudana_khichdi"), (hm("17:00"), "makhana_roasted"),
                     (hm("21:00"), "kuttu_puri_aloo")]
            fw = None
        return DayPlan("navratri_phalahar", meals=meals, su_time=hm("08:30"), fast_window=fw,
                       bed=hm("00:15") + 1440 if night else hm("23:15"), night_activity=night,
                       activity_scale=0.9)

    if "ekadashi" in types:
        return DayPlan("ekadashi",
                       meals=[(hm("07:00"), "masala_chai_sugar"), (hm("11:30"), "fruit_plate"),
                              (hm("14:00"), "sabudana_khichdi"), (hm("20:30"), "kuttu_puri_aloo")],
                       su_time=hm("08:30"), fast_window=None, activity_scale=0.95)

    if "feast" in types:
        return DayPlan("feast",
                       meals=[(hm("07:00"), "masala_chai_sugar"), (hm("09:00"), breakfast),
                              (hm("14:00"), "festive_thali"), (hm("17:00"), "sweet_jalebi"),
                              (hm("21:30"), "festive_thali"), (hm("22:00"), "kheer")],
                       su_time=hm("09:00"), bed=hm("00:00") + 1440, activity_scale=1.1)

    # ----- ordinary day ---------------------------------------------------------------------
    return DayPlan("normal",
                   meals=[(hm("07:00"), "masala_chai_sugar"), (hm("08:45"), breakfast),
                          (hm("13:30"), lunch), (hm("17:15"), "marie_biscuits"),
                          (hm("21:15"), patient.get("dinner", "roti_dal_sabzi"))],
                   su_time=hm("08:45"))


def window_start(patient: dict, cal: dict) -> date:
    return date.fromisoformat(cal["observation_windows"][patient["tradition"]])


def window_days(patient: dict, cal: dict, n_days: int) -> list[date]:
    s = window_start(patient, cal)
    return [s + timedelta(days=i) for i in range(n_days)]


def upcoming_fasts(patient: dict, from_day: date, cal: dict, horizon_days: int = 14) -> list[dict]:
    """Fasting events this patient is likely to observe in the next `horizon_days`."""
    out = []
    for i in range(horizon_days):
        d = from_day + timedelta(days=i)
        for ev in events_on(d, patient["tradition"], cal):
            if ev["type"] != "feast":
                out.append({"date": d.isoformat(), "name": ev["name"], "type": ev["type"]})
    return out
