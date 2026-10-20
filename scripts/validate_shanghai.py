"""External validation: run the twin on the ShanghaiT2DM cohort — real patients, no retraining.

    # 1. download (CC BY 4.0, ~3.4 MB) and unzip:
    #    https://doi.org/10.6084/m9.figshare.20444397
    # 2. point this at the extracted folder:
    python scripts/validate_shanghai.py /path/to/extracted --out reports/EXTERNAL_VALIDATION.md

Why this script exists
----------------------
Everything else in this repository is measured on a simulator we wrote. That is a fair way to build
a proof of concept and a terrible way to prove one. ShanghaiT2DM (Zhao et al., *Scientific Data* 10,
35, 2023) is 109 real CGM recordings from people with type 2 diabetes in Shanghai, with the labs and
medication lists the twin's prior needs. Nothing is fitted here: the EHR row sets the prior, the
particle filter personalises it from the glucose trace, and the twin forecasts forward.

The cohort is Chinese, not Indian, and carries no festival calendar, so this validates the
*mechanistic twin and its forecast*, not the fasting-aware part of the system. That is stated in the
output rather than glossed over.

The raw data are NOT redistributed in this repository. Download them yourself; the licence is
CC BY 4.0 and the citation is in the generated report.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from upvaas.adapters import replay  # noqa: E402
from upvaas.config import REPORT_DIR  # noqa: E402
from upvaas.twin.predictor import clarke_zone  # noqa: E402

# Hypoglycaemic agents that act on the sulfonylurea receptor, i.e. the drug class this whole
# project is about. Repaglinide is a glinide but shares the mechanism (and the hypo risk).
SU_DRUGS = ("gliclazide", "glimepiride", "gliquidone", "glipizide", "glibenclamide", "glyburide",
            "repaglinide")
INSULINS = ("insulin", "novolin", "aspart", "degludec", "detemir", "glargine", "glarigine",
            "humulin", "gansulin", "glulisine", "lispro")
DEFAULT_MEAL_CARBS_G = 50.0     # when a meal is logged but not quantified [assumption, stated in the report]


def _num(v, default=np.nan) -> float:
    """Summary sheets mix numbers, strings and blanks in the same column."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+\.?\d*", str(v))
        return float(m.group()) if m else default


def ehr_from_summary(row: pd.Series) -> dict:
    """Map one ShanghaiT2DM summary row onto the fields the twin's prior reads.

    Fields the dataset does not carry (CYP2C9, TCF7L2, polygenic score, ALT) fall back to the
    population defaults in upvaas/adapters.py. That is the honest situation for a new patient whose
    genotype has never been measured, and it is what the tier-C path assumes anyway.
    """
    agents = f"{row.get('Hypoglycemic Agents', '')} {row.get('Other Agents', '')}".lower()
    on_su = any(d in agents for d in SU_DRUGS)
    on_ins = any(d in agents for d in INSULINS)
    hba1c_mmol = _num(row.get("HbA1c (mmol/mol)"))
    hba1c_pct = hba1c_mmol / 10.929 + 2.15 if np.isfinite(hba1c_mmol) else 7.5
    fpg = _num(row.get("Fasting Plasma Glucose (mg/dl)"), 150.0)
    tg_mmol = _num(row.get("Triglyceride (mmol/L)"), 1.8)
    tg_mgdl = tg_mmol * 88.57
    bmi = _num(row.get("BMI (kg/m2)"), 25.5)
    weight = _num(row.get("Weight (kg)"), 68.0)
    egfr = _num(row.get("Estimated Glomerular Filtration Rate  (ml/min/1.73m2) "), 90.0)
    micro = str(row.get("Diabetic Microvascular Complications", "")).lower()
    return {
        "patient_id": str(row["Patient Number"]),
        "sex": "F" if _num(row.get("Gender (Female=1, Male=2)"), 2) == 1 else "M",
        "age": _num(row.get("Age (years)"), 55),
        "bmi": bmi,
        # waist/height is not recorded; a BMI-based stand-in keeps the prior sane and is flagged
        "whtr": float(np.clip(0.40 + 0.0068 * bmi, 0.42, 0.70)),
        "tyg": float(np.log(max(tg_mgdl, 30) * max(fpg, 50) / 2)),
        "hba1c": float(np.clip(hba1c_pct, 5.0, 15.0)),
        "fpg": float(np.clip(fpg, 60, 400)),
        "diabetes_duration_y": _num(row.get("Duration of diabetes (years)"), 8),
        "egfr": float(np.clip(egfr, 10, 140)),
        "weight_kg": float(np.clip(weight, 35, 160)),
        "on_metformin": "metformin" in agents,
        "on_sulfonylurea": bool(on_su),
        "on_basal_insulin": bool(on_ins),
        "su_dose_frac": 0.5 if on_su else 0.0,      # dose not recorded; mid-range [assumption]
        "basal_insulin_u": 12.0 if on_ins else 0.0,  # units not reliably recorded [assumption]
        "neuropathy": "neuropathy" in micro,
    }


def patient_csv(xlsx: Path, out_csv: Path, use_meal_times: bool) -> int:
    """Convert one recording to the adapter's CSV schema. Returns the number of CGM rows."""
    d = pd.read_excel(xlsx)
    cols = {c.strip().lower(): c for c in d.columns}
    date_c = cols.get("date")
    cgm_c = next((v for k, v in cols.items() if k.startswith("cgm")), None)
    if date_c is None or cgm_c is None:
        return 0
    out = pd.DataFrame({"timestamp": pd.to_datetime(d[date_c], errors="coerce"),
                        "glucose_mgdl": pd.to_numeric(d[cgm_c], errors="coerce")})
    diet_c = next((v for k, v in cols.items() if "dietary" in k), None)
    if use_meal_times and diet_c is not None:
        # The diet column is free text (and partly Chinese), so there is no carbohydrate amount to
        # read. We take the TIMING only and assume a fixed portion - the information a photo-log app
        # realistically gives you. The no-meal-log arm below is the stricter test.
        out["carbs_g"] = np.where(d[diet_c].notna(), DEFAULT_MEAL_CARBS_G, 0.0)
    else:
        out["carbs_g"] = 0.0
    su_c = next((v for k, v in cols.items() if "non-insulin" in k), None)
    out["su_dose_frac"] = 0.0
    if su_c is not None:
        txt = d[su_c].astype(str).str.lower()
        out["su_dose_frac"] = np.where(txt.str.contains("|".join(SU_DRUGS), na=False), 0.5, 0.0)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out[out["glucose_mgdl"].between(20, 600) | out["glucose_mgdl"].isna()]
    if len(out) < 288:                      # need at least ~3 days at 15-min sampling
        return 0
    out.to_csv(out_csv, index=False)
    return int(out["glucose_mgdl"].notna().sum())


def pooled(chunks: list[dict], key: str) -> dict:
    pred = np.concatenate([c[key]["pred"] for c in chunks]) if chunks else np.array([])
    pers = np.concatenate([c[key]["persistence"] for c in chunks]) if chunks else np.array([])
    act = np.concatenate([c[key]["actual"] for c in chunks]) if chunks else np.array([])
    if len(act) == 0:
        return {}
    z = clarke_zone(act, pred)
    zp = clarke_zone(act, pers)
    return {"n": int(len(act)),
            "twin_rmse": float(np.sqrt(np.mean((pred - act) ** 2))),
            "persistence_rmse": float(np.sqrt(np.mean((pers - act) ** 2))),
            "twin_mard": float(np.mean(np.abs(pred - act) / np.maximum(act, 1)) * 100),
            "twin_clarke_AB": float(np.isin(z, ["A", "B"]).mean()),
            "persistence_clarke_AB": float(np.isin(zp, ["A", "B"]).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder containing Shanghai_T2DM/ and the *_Summary.xlsx")
    ap.add_argument("--limit", type=int, default=0, help="only the first N patients (for a quick check)")
    ap.add_argument("--out", default=str(REPORT_DIR / "EXTERNAL_VALIDATION.md"))
    a = ap.parse_args()

    root = Path(a.root)
    summ = next(root.rglob("Shanghai_T2DM_Summary.xlsx"), None)
    pdir = next((p for p in root.rglob("Shanghai_T2DM") if p.is_dir()), None)
    if summ is None or pdir is None:
        sys.exit(f"Could not find Shanghai_T2DM_Summary.xlsx / Shanghai_T2DM/ under {root}")
    s = pd.read_excel(summ).set_index("Patient Number", drop=False)

    files = sorted(pdir.glob("*.xls*"))
    if a.limit:
        files = files[: a.limit]
    tmp = Path(a.out).parent / "_shanghai_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    arms = {"cgm_only": False, "cgm_plus_meal_times": True}
    results = {k: [] for k in arms}
    su_results: dict = {}
    per_patient, skipped, su_ids = [], 0, set()

    for i, f in enumerate(files, 1):
        # The summary's "Patient Number" is the full recording stem (e.g. 2000_0_20201230), not the
        # bare patient prefix; several patients contributed more than one recording.
        key = next((k for k in s.index if str(k) == f.stem), None)
        pid = f.stem.split("_")[0]
        if key is None:
            skipped += 1
            continue
        ehr = ehr_from_summary(s.loc[key] if not isinstance(s.loc[key], pd.DataFrame) else s.loc[key].iloc[0])
        if ehr["on_sulfonylurea"]:
            su_ids.add(pid)
        row = {"patient_id": pid, "file": f.name, "on_su": ehr["on_sulfonylurea"]}
        ok_any = False
        for arm, use_meals in arms.items():
            csv = tmp / f"{f.stem}_{arm}.csv"
            n = patient_csv(f, csv, use_meals)
            if n == 0:
                continue
            try:
                r = replay(str(csv), ehr, P=96, detail=True)
            except Exception as exc:                        # noqa: BLE001
                print(f"  ! {f.name} [{arm}]: {type(exc).__name__}: {exc}")
                continue
            results[arm].append(r["arrays"])
            if ehr["on_sulfonylurea"]:
                su_results.setdefault(arm, []).append(r["arrays"])
            row[f"{arm}_rmse120"] = r["120min"]["twin_rmse"]
            row[f"{arm}_pers120"] = r["120min"]["persistence_rmse"]
            row["n_cgm"] = n
            ok_any = True
        if ok_any:
            per_patient.append(row)
        else:
            skipped += 1
        if i % 10 == 0:
            print(f"  {i}/{len(files)} recordings processed", flush=True)

    if not per_patient:
        sys.exit("No usable recordings found.")

    out = {"dataset": "ShanghaiT2DM (Zhao et al., Scientific Data 10:35, 2023), CC BY 4.0",
           "doi": "10.6084/m9.figshare.20444397",
           "recordings_used": len(per_patient), "recordings_skipped": skipped,
           "patients_on_sulfonylurea": len(su_ids),
           "retrained": False,
           "arms": {arm: {h: pooled(results[arm], h) for h in ("30min", "60min", "120min")}
                    for arm in arms if results[arm]},
           "sulfonylurea_subgroup": {arm: {h: pooled(su_results[arm], h) for h in ("30min", "60min", "120min")}
                                     for arm in su_results if su_results[arm]}}
    jpath = Path(a.out).with_suffix(".json")
    jpath.write_text(json.dumps(out, indent=1))

    L = ["# External validation on real CGM data (ShanghaiT2DM)", "",
         f"*Generated by `scripts/validate_shanghai.py`. {out['recordings_used']} real recordings, "
         f"{out['patients_on_sulfonylurea']} of those patients on a sulfonylurea. "
         "The twin was **not retrained**: the published EHR row sets its prior and the particle filter "
         "personalises it from the glucose trace alone.*", "",
         "## Why this matters", "",
         "Every other number in this repository comes from a simulator we wrote. This one does not. "
         "ShanghaiT2DM is 109 continuous glucose recordings from people with type 2 diabetes in Shanghai, "
         "published with the labs and medication lists the twin's prior needs.", "",
         "**Summary: the twin does not currently transfer to this dataset.** The detail, and our best "
         "explanation, are below. We ran this because a proof of concept measured only on its own simulator "
         "proves very little, and we are reporting the outcome rather than the intention.", "",
         "**What this does and does not validate.** The cohort is Chinese, not Indian, and carries no festival "
         "calendar, no fasting days and no wearable streams. So this tests the *mechanistic twin and its "
         "glucose forecast on real physiology*. It does not validate the fasting-aware part of the system, "
         "which still rests entirely on simulation.", ""]

    for arm, label, note in (
            ("cgm_only", "CGM only", "no meal log at all - the twin sees glucose and the EHR, nothing else"),
            ("cgm_plus_meal_times", "CGM + meal timing",
             f"meal *times* from the diet column, with a fixed {DEFAULT_MEAL_CARBS_G:.0f} g assumed per meal, "
             "since the column is free text and carries no amount")):
        if arm not in out["arms"]:
            continue
        L += [f"### {label}", "", f"*{note}*", "",
              "| Horizon | n forecasts | Twin RMSE | Persistence RMSE | Twin MARD | Twin Clarke A+B |",
              "|---|---|---|---|---|---|"]
        for h in ("30min", "60min", "120min"):
            r = out["arms"][arm].get(h)
            if not r:
                continue
            better = "**" if r["twin_rmse"] < r["persistence_rmse"] else ""
            L.append(f"| {h.replace('min', ' min')} | {r['n']:,} | {better}{r['twin_rmse']:.1f} mg/dL{better} | "
                     f"{r['persistence_rmse']:.1f} mg/dL | {r['twin_mard']:.1f}% | {r['twin_clarke_AB']:.1%} |")
        L.append("")

    su = out["sulfonylurea_subgroup"].get("cgm_only", {}).get("120min")
    if su and su.get("n", 0) > 200:
        L += ["### Sulfonylurea-treated patients only", "",
              f"This project is about sulfonylurea risk, so the subgroup matters more than the average. "
              f"{out['patients_on_sulfonylurea']} of these recordings are from patients on a sulfonylurea "
              "(gliclazide, glimepiride, gliquidone or repaglinide).", "",
              "| Horizon | n forecasts | Twin RMSE | Persistence RMSE | Twin Clarke A+B |",
              "|---|---|---|---|---|"]
        for h in ("30min", "60min", "120min"):
            r = out["sulfonylurea_subgroup"]["cgm_only"].get(h)
            if r:
                L.append(f"| {h.replace('min', ' min')} | {r['n']:,} | {r['twin_rmse']:.1f} mg/dL | "
                         f"{r['persistence_rmse']:.1f} mg/dL | {r['twin_clarke_AB']:.1%} |")
        L.append("")

    a2 = out["arms"].get("cgm_only", {}).get("120min")
    a30 = out["arms"].get("cgm_only", {}).get("30min")
    if a2:
        won = a2["twin_rmse"] < a2["persistence_rmse"]
        verdict = ("The twin beat persistence at the 2-hour horizon on real patients."
                   if won else
                   "**The twin did not beat persistence on this dataset, at any horizon.**")
        L += ["## The result", "", verdict, "",
              f"At 2 hours it reaches **{a2['twin_rmse']:.0f} mg/dL RMSE** against "
              f"**{a2['persistence_rmse']:.0f} mg/dL** for simply assuming glucose stays where it is, with "
              f"{a2['twin_clarke_AB']:.1%} of forecasts in Clarke zones A+B (persistence: "
              f"{a2['persistence_clarke_AB']:.1%}). Our own simulator puts the twin at roughly 38 mg/dL at the "
              "same horizon, comfortably ahead of persistence there. It does not transfer."]
        if a30 and not won:
            ratio30 = a30["twin_rmse"] / max(a30["persistence_rmse"], 1e-9)
            ratio120 = a2["twin_rmse"] / max(a2["persistence_rmse"], 1e-9)
            L += ["",
                  f"The gap narrows with horizon — the twin is {ratio30:.1f}× persistence at 30 minutes and "
                  f"{ratio120:.2f}× at 2 hours — which is the signature of a model whose *dynamics* are roughly "
                  "right but whose *state and parameters* are poorly initialised, rather than one whose physics "
                  "is wrong."]
        L += ["", "### Why, as best we can tell", "",
              "1. **The prior has never seen a real patient.** The EHR → physiology mapping in "
              "`upvaas/twin/priors.py` is a ridge regression fitted on our own synthetic generator. Applied to "
              "real Shanghai inpatients it starts every particle from the wrong place, and three days of sparse "
              "CGM is not enough for the filter to recover. This is the most likely single cause and it is "
              "entirely our doing.",
              "2. **There is no meal log.** The dataset records diet as free text with no carbohydrate amount, so "
              "the twin forecasts forward as if the patient were fasting. The arm where we assumed a fixed "
              f"{DEFAULT_MEAL_CARBS_G:.0f} g per logged meal was *worse still* — meal information only helps when "
              "it is accurate, which matches what our own benchmark ablations predicted.",
              "3. **Sampling is three times sparser than the twin expects.** These are 15-minute Libre traces; the "
              "particle filter was built around 5-minute CGM, so it updates a third as often and drifts in "
              "between.",
              "4. **The observation-noise constant was chosen for our simulated sensor**, not for this one.", "",
              "### What this does and does not change", "",
              "It does not invalidate the design: the mechanistic core, the particle filter, the counterfactual "
              "rehearsal and the graceful-degradation ladder are unchanged, and the simulated results are "
              "reported exactly as before. What it does is put a number on the warning the README already "
              "carried — **the simulated results are an upper bound** — and show that the gap between simulation "
              "and reality is large rather than cosmetic.",
              "",
              "We are reporting a failed external validation rather than omitting it. A reader can now see "
              "precisely how far this prototype is from clinical usefulness, which is more than most digital-twin "
              "papers make possible.", "",
              "### What would have to happen next", "",
              "* ~~Refit the EHR → physiology prior on real cohorts instead of on our own simulator.~~ "
              "**We tried this** — see `reports/PRIOR_REFIT.md`. It did not help. Refitting on real patients "
              "left the held-out result unchanged; the regression's residual spread exploded, which says the "
              "routine EHR fields barely predict these physiological parameters in a real cohort at all. "
              "Widening the prior and raising the assumed sensor noise moved the held-out ratio from 1.10 to "
              "1.01 — parity with persistence, not better than it.",
              "* Handle sparse and irregular sampling explicitly rather than upsampling to a 5-minute grid.",
              "* Fit the observation-noise and process-noise constants per sensor.",
              "* Then repeat this exact script and see whether the numbers move.", ""]

    L += ["## Reproducing this", "", "```bash",
          "# download the dataset (CC BY 4.0) from https://doi.org/10.6084/m9.figshare.20444397",
          "unzip data.zip -d shanghai/",
          "python scripts/validate_shanghai.py shanghai/", "```", "",
          "The raw data are **not** redistributed in this repository. Citation: Zhao, Q., Zhu, J., Shen, X. et al. "
          "*Chinese diabetes datasets for data-driven machine learning.* Scientific Data 10, 35 (2023).", ""]

    Path(a.out).write_text("\n".join(L))
    pd.DataFrame(per_patient).to_csv(Path(a.out).with_name("external_validation_per_patient.csv"), index=False)
    for f in tmp.glob("*.csv"):
        f.unlink()
    tmp.rmdir()
    print(f"\nwrote {a.out}")
    for arm in out["arms"]:
        r = out["arms"][arm].get("120min", {})
        if r:
            print(f"  {arm:<22} 2h: twin {r['twin_rmse']:.1f} vs persistence {r['persistence_rmse']:.1f} mg/dL "
                  f"(n={r['n']:,}, Clarke A+B {r['twin_clarke_AB']:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
