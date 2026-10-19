"""Central configuration: paths, simulation settings, clinical thresholds and Indian priors.

Every number that comes from literature carries a short source tag so the jury can trace it.
Numbers marked [assumption] are modelling choices that a clinical partner should review.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
COHORT_DIR = DATA_DIR / "cohort"
CALENDAR_FILE = DATA_DIR / "calendar" / "india_fasting_calendar_2026_27.json"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"


@dataclass
class SimConfig:
    n_patients: int = 200            # virtual patients in the synthetic cohort
    n_days: int = 28                 # length of each patient's observation window
    dt_truth: float = 1.0            # minutes, "real body" integrator step
    dt_twin: float = 5.0             # minutes, twin integrator step (= CGM sampling)
    seed: int = 2026
    # --- honesty switches (both ON by default; the pipeline reports the cost of each) --------
    # Integrate the virtual body with upvaas/physiology/truth_model.py, which carries four terms
    # the twin has no equation for (3-compartment gut, interstitial insulin delay, circadian SI,
    # renal excretion + incretin). With this False the body runs the twin's own equations and the
    # evaluation measures parameter error only - the optimistic setting most twin papers report.
    model_mismatch: bool = True
    # Realistic eating: wide meal-time jitter, large portion variability, skipped meals and
    # genuinely unplanned snacks. With this False, meals follow the calendar template closely and
    # the festival plan becomes a near-oracle for future carbohydrate (plan vs realised r = 0.91).
    realistic_meals: bool = True
    # Share of patients by fasting tradition [assumption, loosely reflects India's religious mix
    # and the ~49% fasting rate in a North-Indian T2D clinic cohort (Gupta et al., touchENDO 2020)]
    tradition_mix: dict = field(default_factory=lambda: {
        "navratri": 0.38,     # Navratri phalahar + Ekadashi
        "karva_chauth": 0.14, # married women: Navratri + nirjala Karva Chauth
        "ramadan": 0.18,      # roza (simulated in the Ramadan 2027 window)
        "none": 0.30,         # do not fast -> natural controls
    })


@dataclass
class Thresholds:
    hypo: float = 70.0               # mg/dL, ADA level-1 hypoglycaemia
    severe_hypo: float = 54.0        # mg/dL, ADA level-2
    hyper: float = 250.0             # mg/dL, clinically significant spike for alerting
    min_event_minutes: int = 15      # an event must persist >= 15 min (CGM consensus)
    horizon_min: int = 120           # prediction horizon: 2 hours (challenge brief)
    decision_every_min: int = 30     # the twin re-scores risk every 30 minutes


SIM = SimConfig()
THR = Thresholds()

# Sensor tiers used for the graceful-degradation study --------------------------------------
TIERS = {
    "A_full": "CGM (5-min) + smartwatch (HR, HRV, steps, sleep) + EHR + meal log",
    "B_frugal": "2 fingersticks/day + smartwatch + EHR + meal log (no CGM)",
    "C_minimal": "EHR + phone step counter + festival calendar only",
}

# Indian-phenotype priors -------------------------------------------------------------------
# Thin-fat phenotype: diabetics at mean BMI 24.8 had ~2x visceral fat, 5x HOMA-IR and
# lower HOMA-B (Frontiers Endocrinol 2025). We therefore lower insulin sensitivity (SI) and
# beta-cell gain for a given BMI, and drive SI from waist-to-height and TyG index, not BMI.
INDIAN_PRIORS = {
    "bmi_mean_t2d": 25.5,
    "bmi_sd": 3.8,
    "waist_cm_male": (95, 9),
    "waist_cm_female": (90, 9),
    "si_log_mean": -8.25,     # ln(SI) ~ 2.4e-4 /min per uU/mL [assumption; Bergman range 2.1-18.2e-4]
    "si_log_sd": 0.35,
    "beta_gain_mean": 0.085,  # uU/mL/min per mg/dL above basal [assumption, tuned]
    "beta_gain_sd": 0.012,
}

# Genetics. TCF7L2 rs7903146-T lowers insulin secretion; CYP2C9 *2/*3 slow sulfonylurea
# clearance. Carriers of *2/*3 or *3/*3 had OR 5.2 for severe SU hypoglycaemia
# (Holstein et al., Br J Clin Pharmacol 2005, PMC1884896). Frequencies are placeholders until
# Indian genotype data are plugged in [assumption].
GENETICS = {
    "tcf7l2_T_allele_freq": 0.30,
    "cyp2c9_genotypes": {"*1/*1": 0.66, "*1/*2": 0.10, "*1/*3": 0.16, "*2/*2": 0.02,
                          "*2/*3": 0.03, "*3/*3": 0.03},
    # multiplier on sulfonylurea half-life by genotype [assumption based on reduced CYP2C9 activity]
    "cyp2c9_halflife_mult": {"*1/*1": 1.0, "*1/*2": 1.1, "*1/*3": 1.35, "*2/*2": 1.25,
                              "*2/*3": 1.6, "*3/*3": 2.0},
}

# Environment. +0.40 mg/dL fasting glucose per +10 ug/m3 monthly PM2.5 (CARRS, BMJ ODRC 2023)
PM25_GLUCOSE_PER_UG = 0.040

# Medication mix in North-Indian T2D clinic cohort: 93% metformin, 57% SU, 30% insulin
# (Gupta et al. 2020). Used to sample regimens.
MED_MIX = {"metformin": 0.93, "sulfonylurea": 0.57, "basal_insulin": 0.22}

# What patients do with their morning SU on a fasting day. Only 1.35% consulted a doctor before
# fasting (Gupta et al. 2020), so most keep their usual routine [assumption on split].
SU_FASTING_BEHAVIOUR = {"usual_time": 0.55, "with_predawn_meal": 0.30, "skip": 0.15}
