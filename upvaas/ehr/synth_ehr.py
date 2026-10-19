"""India-localised synthetic EHR generator ("Synthea-India-lite").

Each virtual patient has two layers:
  * a HIDDEN physiological truth (insulin sensitivity, beta-cell gain, drug clearance, ...)
  * an OBSERVABLE EHR (demographics, diagnoses, labs, medications, genetic markers)

Labs are generated *from* the hidden physiology with noise, so the EHR carries real but imperfect
information. The digital twin only ever sees the EHR + wearable streams and has to infer the
hidden physiology - exactly the situation a doctor's twin faces in real life.

Why not Synthea directly? Synthea's defaults are US demographics. Its modules can be localised
(the README explains how), but for a fast, reproducible PoC we generate the cohort here and export
FHIR R4 (see upvaas/interop/fhir.py) so it is interchangeable with Synthea output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import GENETICS, INDIAN_PRIORS, MED_MIX, SIM, SU_FASTING_BEHAVIOUR

# state, region, urban share, annual PM2.5 ug/m3 [assumption: rough CPCB-style annual levels],
# share of Ramadan-observing patients [assumption], language
STATES = [
    ("Delhi", "north", 0.97, 100, 0.13, "Hindi"),
    ("Punjab", "north", 0.40, 60, 0.02, "Punjabi"),
    ("Uttar Pradesh", "north", 0.23, 85, 0.19, "Hindi"),
    ("Gujarat", "west", 0.43, 50, 0.10, "Gujarati"),
    ("Maharashtra", "west", 0.45, 45, 0.12, "Marathi"),
    ("Karnataka", "south", 0.39, 30, 0.13, "Kannada"),
    ("Tamil Nadu", "south", 0.48, 30, 0.06, "Tamil"),
    ("Kerala", "south", 0.48, 25, 0.27, "Malayalam"),
    ("West Bengal", "east", 0.32, 60, 0.27, "Bengali"),
    ("Telangana", "south", 0.39, 35, 0.13, "Telugu"),
]
STATE_WEIGHTS = np.array([0.08, 0.06, 0.16, 0.10, 0.12, 0.10, 0.10, 0.08, 0.10, 0.10])

FIRST_M = ["Rajesh", "Suresh", "Arjun", "Imran", "Vikram", "Mohan", "Anil", "Farhan", "Ravi",
           "Harpreet", "Kiran", "Sanjay", "Abdul", "Prakash", "Thomas", "Gopal", "Naveen", "Manoj"]
FIRST_F = ["Sunita", "Priya", "Lakshmi", "Ayesha", "Meena", "Kavita", "Fatima", "Geeta", "Anjali",
           "Simran", "Deepa", "Rekha", "Nazia", "Asha", "Mary", "Savitri", "Pooja", "Shabnam"]
LAST = ["Sharma", "Iyer", "Patel", "Khan", "Reddy", "Singh", "Nair", "Das", "Gupta", "Kulkarni",
        "Menon", "Ansari", "Joshi", "Chatterjee", "Verma", "Pillai", "Shaikh", "Yadav"]
MUSLIM_FIRST_M = ["Imran", "Farhan", "Abdul", "Salman", "Irfan"]
MUSLIM_FIRST_F = ["Ayesha", "Fatima", "Nazia", "Shabnam", "Rukhsar"]
MUSLIM_LAST = ["Khan", "Ansari", "Shaikh", "Qureshi", "Siddiqui"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_cohort(n: int = SIM.n_patients, seed: int = SIM.seed) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ehr, truth) DataFrames indexed by patient_id."""
    rng = np.random.default_rng(seed)
    ehr_rows, truth_rows = [], []
    cyp_keys = list(GENETICS["cyp2c9_genotypes"].keys())
    cyp_p = np.array(list(GENETICS["cyp2c9_genotypes"].values()))
    cyp_p = cyp_p / cyp_p.sum()

    for i in range(n):
        pid = f"UPV-{i + 1:04d}"
        s_idx = rng.choice(len(STATES), p=STATE_WEIGHTS / STATE_WEIGHTS.sum())
        state, region, urban_p, pm25, ramadan_p, lang = STATES[s_idx]
        urban = bool(rng.random() < urban_p)
        sex = "F" if rng.random() < 0.48 else "M"
        age = int(np.clip(rng.normal(54, 10), 32, 80))

        # ---- fasting tradition ------------------------------------------------------------
        if rng.random() < ramadan_p:
            tradition = "ramadan"
        else:
            r = rng.random()
            if r < 0.30:
                tradition = "none"
            elif sex == "F" and region in ("north", "west") and r < 0.78:
                tradition = "karva_chauth"
            else:
                tradition = "navratri"
        if tradition == "ramadan":
            first = rng.choice(MUSLIM_FIRST_F if sex == "F" else MUSLIM_FIRST_M)
            last = rng.choice(MUSLIM_LAST)
        else:
            first = rng.choice([x for x in (FIRST_F if sex == "F" else FIRST_M)
                                if x not in MUSLIM_FIRST_F + MUSLIM_FIRST_M])
            last = rng.choice([x for x in LAST if x not in MUSLIM_LAST])

        # ---- latent physiology ------------------------------------------------------------
        z_ir = rng.normal()      # insulin resistance (visceral / liver fat)
        z_beta = rng.normal()    # beta-cell deficit
        tcf7l2 = int(rng.binomial(2, GENETICS["tcf7l2_T_allele_freq"]))
        cyp = str(rng.choice(cyp_keys, p=cyp_p))
        duration = float(np.clip(2 + 7 * _sigmoid(z_beta) + rng.normal(0, 3), 0.5, 30))

        height = rng.normal(171 if sex == "M" else 156, 6)
        whtr = 0.56 + 0.04 * z_ir + rng.normal(0, 0.02)            # waist-to-height ratio
        waist = whtr * height
        bmi = float(np.clip(INDIAN_PRIORS["bmi_mean_t2d"] + 1.2 * z_ir + rng.normal(0, 3.0), 18, 40))
        weight = bmi * (height / 100) ** 2
        tg = float(np.exp(np.log(160) + 0.25 * z_ir + rng.normal(0, 0.22)))
        hdl = float(np.clip(rng.normal(41 if sex == "M" else 46, 7) - 3 * z_ir, 22, 80))
        ldl = float(np.clip(rng.normal(110, 28), 50, 220))
        alt = float(np.exp(np.log(28) + 0.25 * z_ir + rng.normal(0, 0.3)))
        lpa = float(np.exp(rng.normal(np.log(20), 0.9)))           # mg/dL, median ~20 in Indians
        prs_z = float(0.3 * z_beta + rng.normal(0, 0.95))

        egfr = float(np.clip(rng.normal(96 - 0.6 * (age - 40), 16) - 4 * (duration > 12), 22, 125))
        neuropathy = bool(rng.random() < _sigmoid(-2.2 + 0.12 * duration))
        hypertension = bool(rng.random() < _sigmoid(-0.4 + 0.04 * (age - 50) + 0.4 * z_ir))
        family_hx = bool(rng.random() < _sigmoid(0.2 + 0.5 * prs_z))

        # ---- drug-free hepatic set-point (needed first: doctors titrate drugs to glucose) ----
        gb0 = float(np.clip(150 + 22 * z_beta + 10 * z_ir + rng.normal(0, 10), 110, 260))

        # ---- medications (titrated to glycaemia, as in routine care) -------------------------
        on_met = bool(rng.random() < MED_MIX["metformin"]) and egfr >= 30
        on_su = bool(rng.random() < _sigmoid(-0.2 + 0.035 * (gb0 - 145)))
        on_ins = bool(rng.random() < _sigmoid(-2.2 + 0.03 * (gb0 - 160) + 0.08 * (duration - 8)))
        su_drug = str(rng.choice(["glimepiride", "gliclazide_MR"], p=[0.6, 0.4])) if on_su else ""
        su_dose_frac = float(np.clip(0.25 * np.round((gb0 - 105) / 30 + rng.normal(0, 0.9)), 0.25, 1.0)) if on_su else 0.0
        su_dose_mg = {"glimepiride": 4.0, "gliclazide_MR": 120.0}.get(su_drug, 0) * su_dose_frac
        ins_units = float(np.round(rng.uniform(8, 20))) if on_ins else 0.0
        met_mg = float(rng.choice([1000, 1500, 2000])) if on_met else 0.0

        # ---- hidden physiology parameters (the "true body") --------------------------------
        si = float(np.exp(INDIAN_PRIORS["si_log_mean"] - 0.30 * z_ir + rng.normal(0, 0.15)))
        si *= 1.15 if on_met else 1.0
        beta = float(INDIAN_PRIORS["beta_gain_mean"] * np.exp(-0.30 * z_beta - 0.10 * tcf7l2
                                                            + rng.normal(0, 0.12)))
        sg = float(np.clip(rng.normal(0.012, 0.0025) - 0.0008 * z_ir, 0.006, 0.02))
        p2 = float(np.clip(rng.normal(0.025, 0.004), 0.015, 0.04))
        n_clear = float(np.clip(rng.normal(0.12, 0.015), 0.08, 0.16))
        t_half_su = 6.0 * 60 * GENETICS["cyp2c9_halflife_mult"][cyp] * (1.6 if egfr < 45 else 1.3 if egfr < 60 else 1.0)
        su_gain = float(7.0 * np.exp(rng.normal(0, 0.25))) if on_su else 0.0
        cr_gain = float(0.07 * np.exp(-0.03 * duration) * (0.55 if neuropathy else 1.0))
        hr_hypo = float(18 * (0.45 if neuropathy else 1.0))
        dawn = float(np.clip(rng.normal(0.07, 0.025), 0.0, 0.15))
        sleep_sens = float(np.clip(rng.normal(1.0, 0.3), 0.2, 2.0))
        ins_basal = 0.65 * ins_units
        illness = rng.random() < 0.12
        ill_start = int(rng.integers(13, 22)) if illness else -1

        rhr = float(np.clip(rng.normal(74 + 3 * z_ir + 4 * neuropathy, 6), 55, 100))
        rmssd = float(np.clip(np.exp(rng.normal(np.log(30) - 0.012 * (age - 50) - 0.35 * neuropathy, 0.25)), 8, 70))
        g_mean_expected = gb0 * (0.9 if on_met else 1.0) - 10 * su_dose_frac + 28
        fpg = float(np.clip(gb0 * (0.9 if on_met else 1.0) - 9 * su_dose_frac - 4 * on_ins + rng.normal(0, 10), 85, 300))
        hba1c = float(np.clip((g_mean_expected + 46.7) / 28.7 + rng.normal(0, 0.35), 5.7, 13.5))
        prior_hypos = int(rng.poisson(0.3 + 1.2 * on_su * (cyp in ("*2/*3", "*3/*3", "*1/*3")) + 0.8 * (egfr < 60)))
        sbp = float(np.clip(rng.normal(128 + 10 * hypertension + 0.3 * (age - 50), 12), 100, 190))
        dbp = float(np.clip(sbp * 0.62 + rng.normal(0, 6), 60, 115))

        garba = tradition in ("navratri", "karva_chauth") and region in ("west", "north") \
            and age < 62 and rng.random() < (0.55 if state == "Gujarat" else 0.2)
        vrat_style = str(rng.choice(["phalahar", "ek_samay"], p=[0.6, 0.4]))
        su_fast_behaviour = str(rng.choice(list(SU_FASTING_BEHAVIOUR),
                                           p=list(SU_FASTING_BEHAVIOUR.values())))

        ehr_rows.append({
            "patient_id": pid, "name": f"{first} {last}", "sex": sex, "age": age,
            "state": state, "region": region, "urban": urban, "language": lang,
            "abha_id": f"SYNTH-{rng.integers(10**13, 10**14)}",
            "asha_ward": f"{state[:3].upper()}-W{rng.integers(1, 40):02d}",
            "tradition": tradition, "vrat_style": vrat_style, "garba": bool(garba),
            "millet_eater": bool(rng.random() < 0.2), "dinner": str(rng.choice(["roti_dal_sabzi", "rice_dal_sabzi"])),
            "height_cm": round(height, 1), "weight_kg": round(weight, 1), "bmi": round(bmi, 1),
            "waist_cm": round(waist, 1), "whtr": round(whtr, 3),
            "diabetes_duration_y": round(duration, 1), "hba1c": round(hba1c, 1), "fpg": round(fpg),
            "tg": round(tg), "hdl": round(hdl), "ldl": round(ldl), "alt": round(alt),
            "tyg": round(float(np.log(tg * fpg / 2)), 2), "egfr": round(egfr), "lpa": round(lpa, 1),
            "sbp": round(sbp), "dbp": round(dbp), "resting_hr": round(rhr),
            "hypertension": hypertension, "neuropathy": neuropathy, "family_history_t2d": family_hx,
            "prior_hypo_12m": prior_hypos,
            "tcf7l2_rs7903146_T": tcf7l2, "cyp2c9": cyp, "prs_t2d_z": round(prs_z, 2),
            "on_metformin": on_met, "metformin_mg": met_mg,
            "on_sulfonylurea": on_su, "su_drug": su_drug, "su_dose_mg": su_dose_mg, "su_dose_frac": su_dose_frac,
            "on_basal_insulin": on_ins, "basal_insulin_u": ins_units,
            "su_fasting_behaviour": su_fast_behaviour,
            "pm25_annual": pm25,
            "phone": f"+91-9{rng.integers(10**8, 10**9)}",
        })
        truth_rows.append({
            "patient_id": pid, "SI": si, "beta": beta, "Gb0": gb0, "SG": sg, "p2": p2, "n": n_clear,
            "su_gain": su_gain, "su_thalf_min": t_half_su, "cr_gain": cr_gain, "hr_hypo": hr_hypo,
            "dawn": dawn, "sleep_sens": sleep_sens, "ins_basal": ins_basal, "bw": weight,
            "illness_start_day": ill_start, "rmssd_base": rmssd, "z_ir": z_ir, "z_beta": z_beta,
        })

    ehr = pd.DataFrame(ehr_rows).set_index("patient_id")
    truth = pd.DataFrame(truth_rows).set_index("patient_id")
    return ehr, truth
