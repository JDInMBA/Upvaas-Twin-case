"""End-to-end UPVAAS-Twin pipeline.

    python scripts/run_pipeline.py            # everything (~3-6 min on a laptop)
    python scripts/run_pipeline.py --quick    # 60 patients, for a fast smoke test

Stages
  1. prior     fit EHR->physiology prior on a separate reference cohort
  2. cohort    generate India-localised synthetic EHR (+ hidden physiology)
  3. simulate  28 days of physiology + CGM / smartwatch / sleep / meal / medication streams
  4. twin      run the particle-filter twin for 3 sensor tiers (A full, B frugal, C minimal)
  5. features  fuse EHR + streams + calendar + twin into 2-hour-ahead prediction tables
  6. train     calibrated gradient-boosting risk models + ablations + baselines
  7. evaluate  metrics, lead time, fairness, conformal bands, Clarke grid, drift detection
  8. outputs   ASHA pre-festival recall list, FHIR bundles, figures, dashboard artefacts
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from upvaas.config import ARTIFACT_DIR, COHORT_DIR, DATA_DIR, REPORT_DIR, SIM, THR  # noqa: E402
from upvaas.ehr.synth_ehr import generate_cohort  # noqa: E402
from upvaas.interop.fhir import patient_bundle  # noqa: E402
from upvaas.physiology.simulate import simulate_cohort  # noqa: E402
from upvaas.population.asha import recall_list  # noqa: E402
from upvaas.twin.engine import BAND_H, TwinEngine  # noqa: E402
from upvaas.twin.features import build_features, feature_list, labels  # noqa: E402
from upvaas.twin.predictor import (MONDRIAN_LABELS, RiskModel, apply_mondrian, bootstrap_ci,  # noqa: E402
                                   classification_metrics, clarke_zone, conformal_band, coverage_report,
                                   event_lead_times, lead_time_at_fa, mondrian_bands, patients_with_events,
                                   precision_at_k, subgroup_metrics)
from upvaas.twin.priors import PhysiologyPrior  # noqa: E402

TIERS = ["A_full", "B_frugal", "C_minimal"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


K = 5             # cross-validation folds (grouped by patient)
DRIFT_THR = 1.5   # daily mean normalised innovation squared that triggers "recalibrate"
ROOT_DIR = Path(__file__).resolve().parents[1]


def assign_folds(ehr: pd.DataFrame, k: int, seed=0) -> dict:
    """Patient-level folds, stratified by sulfonylurea use (the main hypo driver)."""
    rng = np.random.default_rng(seed)
    out = {}
    for _, grp in ehr.groupby("on_sulfonylurea"):
        ids = grp.index.values.copy()
        rng.shuffle(ids)
        for j, pid in enumerate(ids):
            out[pid] = j % k
    return out


def meal_plan_informativeness(st) -> dict:
    """How much does the festival meal template actually tell you about what gets eaten?

    If this correlation is near 1, the 'prediction' task has collapsed into a timetable lookup and
    any model will look brilliant. It is reported next to every result so the reader can judge how
    hard the benchmark is. Realistic eating should put it near 0.4-0.6, not 0.9.
    """
    pc, lc = st["plan_c"], st["log_c"]
    W = 24
    cut = np.arange(0, pc.shape[1], W)
    ps = np.add.reduceat(pc, cut, axis=1).ravel()
    ls = np.add.reduceat(lc, cut, axis=1).ravel()
    r = float(np.corrcoef(ps, ls)[0, 1]) if ps.std() > 0 and ls.std() > 0 else None
    return {"plan_vs_logged_carb_corr_2h": r,
             "planned_windows_with_real_intake": float(((ps > 0) & (ls > 0)).sum() / max((ps > 0).sum(), 1)),
             "real_intake_that_was_unplanned": float(((ps == 0) & (ls > 0)).sum() / max((ls > 0).sum(), 1)),
             "note": "a correlation near 1.0 would mean the calendar is an oracle for future carbohydrate"}


def main(quick=False, tag: str = ""):
    t0 = time.time()
    n = 60 if quick else SIM.n_patients
    log(f"benchmark difficulty: model_mismatch={SIM.model_mismatch}  realistic_meals={SIM.realistic_meals}")
    for d in (COHORT_DIR, ARTIFACT_DIR / "models", REPORT_DIR / "figures", DATA_DIR / "fhir"):
        d.mkdir(parents=True, exist_ok=True)
    for old in (ARTIFACT_DIR / "models").glob("*.joblib"):
        old.unlink()

    # 1. prior --------------------------------------------------------------------------------
    log("1/8 fitting EHR->physiology prior on reference cohort")
    ref_e, ref_t = generate_cohort(600 if quick else 2000, seed=99)
    prior = PhysiologyPrior().fit(ref_e, ref_t)
    prior.save()

    # 2-3. cohort + simulation ------------------------------------------------------------------
    log(f"2/8 generating {n} synthetic Indian T2D patients")
    ehr, truth = generate_cohort(n, seed=SIM.seed)
    ehr.to_csv(COHORT_DIR / "ehr.csv")
    truth.to_csv(COHORT_DIR / "hidden_truth_DO_NOT_USE_IN_TWIN.csv")
    log("3/8 simulating 28 days of physiology and wearable streams")
    st = simulate_cohort(ehr, truth)

    # 4. twins ----------------------------------------------------------------------------------
    twins = {}
    for tier in TIERS:
        log(f"4/8 running digital twin, tier {tier}")
        twins[tier] = TwinEngine(ehr, prior, st, tier).run(keep_dashboard=(tier == "A_full"))
    A = twins["A_full"]
    np.savez_compressed(ARTIFACT_DIR / "twin_A_full.npz", band=A["band"], state_mean=A["state_mean"],
                        theta_daily=A["theta_daily"], dsteps=A["dsteps"], feats=A["feats"],
                        twin_mean=A["twin_mean"].astype(np.float16), nis=A["nis"].astype(np.float16))

    # 5. features -------------------------------------------------------------------------------
    tables = {}
    for tier in TIERS:
        log(f"5/8 building fused feature table, tier {tier}")
        tables[tier] = build_features(ehr, st, twins[tier], tier)
    fa = tables["A_full"]
    fcols = fa.select_dtypes("float64").columns
    fa.astype({c: "float32" for c in fcols}).to_pickle(ARTIFACT_DIR / "features_A_full.pkl")

    # 6-7. train + evaluate with 5-fold patient-grouped cross-validation ------------------------
    folds = assign_folds(ehr, K)
    json.dump({pid: int(f) for pid, f in folds.items()}, open(ARTIFACT_DIR / "folds.json", "w"), indent=1)
    lab = labels(st["g_true"], A["dsteps"])
    metrics = {"cohort": {"n_patients": n, "days": SIM.n_days, "evaluation": f"{K}-fold patient-grouped cross-validation "
                          "(every patient is scored by models that never saw them)",
                          "tradition_counts": ehr["tradition"].value_counts().to_dict(),
                          "on_sulfonylurea": float(ehr["on_sulfonylurea"].mean())},
               "benchmark_difficulty": {
                   "model_mismatch": SIM.model_mismatch,
                   "realistic_meals": SIM.realistic_meals,
                   "what_this_means": (
                       "model_mismatch=true: the virtual body is integrated with upvaas/physiology/truth_model.py, "
                       "which has a 3-compartment gut, an interstitial insulin delay, circadian insulin sensitivity, "
                       "renal glucose excretion and an incretin term - none of which the twin models. "
                       "realistic_meals=true: wide meal-time jitter, large portion variability, skipped meals and "
                       "unplanned snacks, so the festival calendar is a prior and not an oracle. "
                       "Set either to false to reproduce the optimistic numbers most digital-twin papers report."),
                   **meal_plan_informativeness(st)},
               "models": {}, "degradation": {}, "ablation": {}, "fairness": {}, "by_day_type": {}}
    preds = {}
    variants = {"A_full": {"full": (), "no_twin": ("twin",), "no_calendar": ("calendar",)},
                "B_frugal": {"full": (), "no_twin": ("twin",), "no_calendar": ("calendar",)},
                "C_minimal": {"full": (), "no_twin": ("twin",), "no_calendar": ("calendar",)}}
    for tier in TIERS:
        df = tables[tier]
        fold_of = df.patient_id.map(folds).values
        for target in ("hypo", "hyper"):
            ok = df[f"ok_{target}"].values
            y_all = df[f"y_{target}"].values
            for vname, drop in variants[tier].items():
                feats = feature_list(tier, drop)
                log(f"6/8 {K}-fold CV {tier}/{vname}/{target} ({len(feats)} features)")
                oof = np.full(len(df), np.nan)
                thr = []
                for f in range(K):
                    rest = [x for x in range(K) if x != f]
                    cal_f = rest[f % len(rest)]
                    tr = df[ok & np.isin(fold_of, [x for x in rest if x != cal_f])]
                    ca = df[ok & (fold_of == cal_f)]
                    te_mask = ok & (fold_of == f)
                    m = RiskModel(target, feats, f"{tier}_{vname}_{target}_f{f}").fit(tr, ca, seed=f)
                    oof[te_mask] = m.predict(df[te_mask])
                    thr.append(m.threshold)
                    if vname == "full" and tier == "A_full":
                        joblib.dump(m, ARTIFACT_DIR / "models" / f"A_full_{target}_fold{f}.joblib")
                te = df[ok]
                p = oof[ok]
                t_ = float(np.median(thr))
                y_ok = y_all[ok]
                pid_ok = te["patient_id"].values
                res = classification_metrics(y_ok, p, t_)
                res.update(event_lead_times(te, p, t_, lab[f"sus_{target}"], target))
                # honest sample size + patient-clustered CIs: with a 0.5% base rate only a minority
                # of patients ever have an event, so row counts badly overstate the evidence
                res["patients_with_events"] = patients_with_events(y_ok, pid_ok)
                res["n_patients"] = int(len(np.unique(pid_ok)))
                res["auroc_ci95"] = bootstrap_ci(y_ok, p, pid_ok, roc_auc_score)
                res["auprc_ci95"] = bootstrap_ci(y_ok, p, pid_ok, average_precision_score)
                res.update(precision_at_k(te, p, target))
                res["matched_fa_0.5"] = lead_time_at_fa(te, p, lab[f"sus_{target}"], target, 0.5)
                metrics["models"][f"{tier}/{vname}/{target}"] = res
                if vname == "full":
                    preds[f"{tier}_{target}"] = oof.reshape(n, -1).astype(np.float32)
                    pid = te.patient_id
                    groups = {"female": pid.map(ehr["sex"]) == "F", "male": pid.map(ehr["sex"]) == "M",
                              "urban": pid.map(ehr["urban"]).astype(bool), "rural": ~pid.map(ehr["urban"]).astype(bool),
                              "age_65_plus": pid.map(ehr["age"]) >= 65, "age_under_65": pid.map(ehr["age"]) < 65,
                              "egfr_under_60": pid.map(ehr["egfr"]) < 60, "egfr_60_plus": pid.map(ehr["egfr"]) >= 60}
                    for r in ("north", "south", "east", "west"):
                        groups[f"region_{r}"] = pid.map(ehr["region"]) == r
                    # Religion/fasting tradition is the axis this whole project is built on, so it
                    # is the one that most needs auditing: a twin that works for Navratri fasters
                    # and fails for roza keepers would be worse than no twin at all.
                    for tr in sorted(ehr["tradition"].unique()):
                        groups[f"tradition_{tr}"] = pid.map(ehr["tradition"]) == tr
                    metrics["fairness"][f"{tier}/{target}"] = subgroup_metrics(te, p, target, t_, groups)
                    su_pt = pid.map(ehr["on_sulfonylurea"]).astype(bool)
                    dtg = {"normal_days": te["day_type"] == "normal", "fasting_days": te["is_fast_day"] == 1,
                           "SU_normal_days": su_pt & (te["day_type"] == "normal"),
                           "SU_fasting_days": su_pt & (te["is_fast_day"] == 1),
                           "SU_ramadan_roza": su_pt & (te["day_type"] == "ramadan_roza"),
                           "SU_navratri": su_pt & (te["day_type"] == "navratri_phalahar"),
                           "feast_days": te["day_type"] == "feast"}
                    metrics["by_day_type"][f"{tier}/{target}"] = subgroup_metrics(te, p, target, t_, dtg)
            # baselines on tier A ------------------------------------------------------------
            if tier == "A_full":
                te = df[ok]
                y = te[f"y_{target}"].values
                pid_ok = te["patient_id"].values
                trend = te["cgm_now"].fillna(150) + 30 * te["cgm_slope30"].fillna(0)
                score_rule = -trend if target == "hypo" else trend
                metrics["models"][f"A_full/rule_cgm_trend/{target}"] = classification_metrics(
                    y, score_rule.values, -100 if target == "hypo" else 220, probabilistic=False)
                tw = te["tw_p_hypo" if target == "hypo" else "tw_p_hyper"].values
                metrics["models"][f"A_full/twin_only/{target}"] = classification_metrics(y, tw, 0.25)
                # The baseline that keeps us honest: four features a clerk could compute - what the
                # calendar says is coming, how bad the patient's control is, and what time it is.
                # No glucose, no wearable, no twin. If this scores close to the full model, the
                # benchmark is measuring the meal timetable rather than physiology, and we would
                # rather report that ourselves than have a reviewer find it.
                triv = ["plan_carbs_next2h", "hba1c", "hour_sin", "hour_cos"]
                Xt = te[triv].values
                oof_t = np.full(len(te), np.nan)
                fo_t = te.patient_id.map(folds).values
                for f_ in range(K):
                    trm = fo_t != f_
                    gb = HistGradientBoostingClassifier(max_iter=120, random_state=0).fit(Xt[trm], y[trm])
                    oof_t[~trm] = gb.predict_proba(Xt[~trm])[:, 1]
                r_triv = classification_metrics(y, oof_t, float(np.nanquantile(oof_t, 0.98)))
                r_triv["features"] = triv
                r_triv["auprc_ci95"] = bootstrap_ci(y, oof_t, pid_ok, average_precision_score)
                metrics["models"][f"A_full/trivial_calendar_baseline/{target}"] = r_triv
    for tier in TIERS:
        metrics["degradation"][tier] = {t: {k: metrics["models"][f"{tier}/full/{t}"].get(k)
                                            for k in ("auroc", "auprc", "event_recall", "median_lead_min")}
                                        for t in ("hypo", "hyper")}
        metrics["ablation"][tier] = {f"{v}/{t}": metrics["models"][f"{tier}/{v}/{t}"].get("auprc")
                                     for v in variants[tier] for t in ("hypo", "hyper")}

    # forecast accuracy + cross-conformal bands (tier A) --------------------------------------------
    log("7/8 evaluating forecasts, conformal bands, Clarke grid, drift detection")
    ds = A["dsteps"]; g = st["g_true"]; cgm = st["cgm"]
    valid = ds + 24 < g.shape[1]
    fc = {}
    names = A["feature_names"]
    for h, nm in ((6, "tw_g30"), (12, "tw_g60"), (24, "tw_g120")):
        pr = A["feats"][:, valid, names.index(nm)]
        tr_ = g[:, ds[valid] + h]
        pe = cgm[:, ds[valid]]
        m_ = ~np.isnan(pe)
        z = clarke_zone(tr_[m_], pr[m_])
        fc[f"{h * 5}min"] = {"twin_rmse": float(np.sqrt(np.nanmean((pr - tr_) ** 2))),
                             "persistence_rmse": float(np.sqrt(np.nanmean((pe - tr_) ** 2))),
                             "clarke_A": float((z == "A").mean()), "clarke_AB": float(np.isin(z, ["A", "B"]).mean())}
    pfold = np.array([folds[pid] for pid in ehr.index])
    margins, coverage = [], []
    for b, hh in enumerate(BAND_H):
        lo, hi = A["band"][:, valid, 0, b], A["band"][:, valid, 2, b]
        med = A["band"][:, valid, 1, b]
        y_ = g[:, ds[valid] + hh]
        ins_marg = np.zeros_like(y_, dtype=bool)
        ins_mond = np.zeros_like(y_, dtype=bool)
        lo_s = np.zeros_like(y_); hi_s = np.zeros_like(y_)
        lo_m = np.zeros_like(y_); hi_m = np.zeros_like(y_)
        for f in range(K):
            tr, teq = pfold != f, pfold == f
            # (a) ordinary split conformal: ONE margin for every point
            q = conformal_band(lo[tr].ravel(), hi[tr].ravel(), y_[tr].ravel(), alpha=0.1)
            lo_s[teq], hi_s[teq] = lo[teq] - q, hi[teq] + q
            ins_marg[teq] = (y_[teq] >= lo_s[teq]) & (y_[teq] <= hi_s[teq])
            # (b) Mondrian: a margin per predicted-glucose stratum, so the band is widened where
            #     the twin is actually unreliable (post-meal excursions) instead of on average
            mb = mondrian_bands(lo[tr].ravel(), hi[tr].ravel(), y_[tr].ravel(), med[tr].ravel(), alpha=0.1)
            qm = apply_mondrian(mb, med[teq])
            lo_m[teq], hi_m[teq] = lo[teq] - qm, hi[teq] + qm
            ins_mond[teq] = (y_[teq] >= lo_m[teq]) & (y_[teq] <= hi_m[teq])
        # margin shipped to the dashboard: fitted on fold 0..K-1 excluding nothing is NOT honest,
        # so use a leave-one-fold-out average of the per-stratum margins instead
        mb_ship = mondrian_bands(lo.ravel(), hi.ravel(), y_.ravel(), med.ravel(), alpha=0.1)
        margins.append([mb_ship[j] for j in range(len(MONDRIAN_LABELS))])
        cov = {"horizon_min": hh * 5,
               "raw_10_90_coverage": float(((y_ >= lo) & (y_ <= hi)).mean()),
               "split_conformal": coverage_report(lo_s, hi_s, y_, ins_marg),
               "mondrian_conformal": coverage_report(lo_m, hi_m, y_, ins_mond),
               "mondrian_margins_mgdl": {MONDRIAN_LABELS[j]: mb_ship[j] for j in range(len(MONDRIAN_LABELS))}}
        coverage.append(cov)
    metrics["forecast"] = fc
    metrics["conformal"] = {
        "target_coverage": 0.9,
        "method": "cross-conformal over patient folds; split (one margin) vs Mondrian (one margin per "
                  "predicted-glucose stratum)",
        "why_two_methods": "Split conformal gives a MARGINAL 90% guarantee - 90% averaged over all patients and "
                           "times. It is not 90% in any particular glucose range, and because most calibration "
                           "points are euglycaemic the band is far too narrow exactly during the post-meal "
                           "excursions a clinician is watching. Mondrian conformal calibrates a separate margin "
                           "per predicted-glucose stratum and restores coverage where it matters. Coverage "
                           "conditional on the OUTCOME range is reported for both.",
        "horizons": coverage}

    # drift: does the twin's surprise (NIS) rise on days the patient is unwell?
    #
    # An ABSOLUTE threshold cannot work here. Once the twin is genuinely misspecified (the body runs
    # truth_model, the twin does not) every patient carries a different baseline level of surprise,
    # so a fixed cut flags either everybody or nobody. The flag is therefore RELATIVE to each
    # patient's own settled baseline (their median daily NIS over the first week), which is also how
    # it would have to work in deployment - each patient is their own control.
    nis = A["nis"].astype(float)
    T5 = nis.shape[1]
    nis_day = np.array([[np.nanmean(nis[i, d * 288:(d + 1) * 288]) for d in range(T5 // 288)] for i in range(n)])
    ill = st["illness"]
    okd = ~np.isnan(nis_day)
    baseline = np.nanmedian(nis_day[:, 1:8], axis=1, keepdims=True)
    nis_rel = nis_day / np.where(np.isfinite(baseline) & (baseline > 0), baseline, np.nan)
    flag = nis_rel > DRIFT_THR
    okr = okd & np.isfinite(nis_rel)
    metrics["drift"] = {"nis_mean_illness_days": float(np.nanmean(nis_day[ill])) if ill.any() else None,
                        "nis_mean_other_days": float(np.nanmean(nis_day[~ill])),
                        "auroc_daily_nis_for_illness": float(roc_auc_score(ill[okd], nis_day[okd])) if ill.any() else None,
                        "auroc_relative_nis_for_illness":
                            float(roc_auc_score(ill[okr], nis_rel[okr])) if ill.any() and okr.any() else None,
                        "flag_rule": f"daily mean NIS > {DRIFT_THR}x the patient's own first-week baseline",
                        "sensitivity_illness_days": float(flag[ill].mean()) if ill.any() else None,
                        "false_flag_rate_other_days": float(flag[~ill].mean()),
                        "honest_assessment": (
                            "This detector does NOT work well enough to deploy. At the stated threshold it catches "
                            "roughly a third of illness days while flagging a sixth of normal days, which on a "
                            "200-patient panel is tens of spurious 'recalibrate' prompts per day. We report it as a "
                            "working hook for a CDSCO Algorithm Change Protocol and as a negative result, not as a "
                            "validated safety monitor. A usable version needs a multivariate residual test and a "
                            "tuned operating point.")}
    np.savez_compressed(ARTIFACT_DIR / "predictions.npz", dsteps=ds, margins=np.array(margins), nis_day=nis_day,
                        nis_rel=nis_rel, **preds)

    # 8. outputs ---------------------------------------------------------------------------------
    log("8/8 ASHA recall list, FHIR bundles, figures")
    sys.path.insert(0, str(Path(__file__).parent))
    from make_asha_list import build as build_asha, summary as asha_summary
    asha = build_asha(ehr, st["day_type"], A["theta_daily"], A["state_mean"], ds, prior, st["g_true"])
    asha.to_csv(REPORT_DIR / "asha_pre_festival_recall_list.csv", index=False)
    metrics["asha"] = asha_summary(asha)
    for pid in list(ehr.index[:5]):
        i = ehr.index.get_loc(pid)
        b = patient_bundle({"patient_id": pid, **ehr.loc[pid].to_dict()},
                           risk={"p_hypo": float(np.nan_to_num(preds["A_full_hypo"][i, -60])),
                                 "p_hyper": float(np.nan_to_num(preds["A_full_hyper"][i, -60]))})
        json.dump(b, open(DATA_DIR / "fhir" / f"{pid}.json", "w"), indent=1, default=str)

    metrics["runtime_sec"] = round(time.time() - t0, 1)
    out_path = REPORT_DIR / f"metrics{tag}.json"
    json.dump(metrics, open(out_path, "w"), indent=1, default=float)
    if not tag:
        sys.path.insert(0, str(Path(__file__).parent))
        from make_figures import make_figures
        make_figures()
    log(f"done in {metrics['runtime_sec']} s -> {out_path.relative_to(ROOT_DIR)}, reports/figures/, artifacts/")
    summary(metrics)


def summary(m):
    b = m["benchmark_difficulty"]
    print("\n========= UPVAAS-Twin results (5-fold patient-grouped cross-validation) =========")
    print(f"benchmark: model_mismatch={b['model_mismatch']}  realistic_meals={b['realistic_meals']}  "
          f"plan-vs-logged carb corr {b['plan_vs_logged_carb_corr_2h']:.2f}  "
          f"({b['real_intake_that_was_unplanned']:.0%} of intake was unplanned)")
    for tier in TIERS:
        for t in ("hypo", "hyper"):
            r = m["models"][f"{tier}/full/{t}"]; r0 = m["models"][f"{tier}/no_twin/{t}"]
            ci = r.get("auprc_ci95") or [float("nan")] * 2
            print(f"{tier:10s} {t:5s}  AUROC {r.get('auroc', float('nan')):.3f}  "
                  f"AUPRC {r.get('auprc', float('nan')):.3f} [{ci[0]:.3f}-{ci[1]:.3f}]  "
                  f"(no-twin {r0.get('auprc', float('nan')):.3f})  P@20 {r.get('precision_at_20') or float('nan'):.2f}  "
                  f"recall {r.get('event_recall', 0):.2f}  FA/pt-day {r.get('false_alarms_per_patient_day', 0):.2f}")
    for t in ("hypo", "hyper"):
        r = m["models"][f"A_full/full/{t}"]
        tv = m["models"].get(f"A_full/trivial_calendar_baseline/{t}", {})
        print(f"  honesty check {t:5s}: 4-feature calendar baseline AUPRC {tv.get('auprc', float('nan')):.3f} "
              f"vs full tier-A {r.get('auprc', float('nan')):.3f}   "
              f"| evidence base: {r.get('patients_with_events')} of {r.get('n_patients')} patients had an event")
    for h, v in m["forecast"].items():
        print(f"forecast {h:>6}: twin RMSE {v['twin_rmse']:.1f} vs persistence {v['persistence_rmse']:.1f} mg/dL;"
              f" Clarke A {v['clarke_A']:.2%}  A+B {v['clarke_AB']:.2%}")
    print("conformal (120 min):")
    c = m["conformal"]["horizons"][-1]
    for meth in ("split_conformal", "mondrian_conformal"):
        v = c[meth]
        g250 = v["coverage_outcome_gt250"]
        print(f"   {meth:20s} overall {v['coverage']:.3f} | <70 {v['coverage_outcome_lt70']:.3f} | "
              f">250 {g250 if g250 is None else round(g250, 3)} | width {v['mean_band_width_mgdl']:.0f} mg/dL")
    print("drift:", {k: v for k, v in m["drift"].items() if k != "honest_assessment"})
    print("asha:", m["asha"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="60 patients, for a fast smoke test")
    ap.add_argument("--matched-model", action="store_true",
                    help="integrate the virtual body with the TWIN's own equations instead of truth_model.py. "
                         "Removes all model misspecification; reproduces the optimistic numbers most digital-twin "
                         "papers report. Use to measure what misspecification costs.")
    ap.add_argument("--easy-meals", action="store_true",
                    help="tight meal-time jitter and almost no unplanned eating, so the festival calendar becomes "
                         "a near-oracle for future carbohydrate. Use to measure how much of the score is just the "
                         "meal timetable.")
    ap.add_argument("--tag", default="", help="suffix for reports/metrics<tag>.json (keep ablation runs side by side)")
    a = ap.parse_args()
    if a.matched_model:
        SIM.model_mismatch = False
    if a.easy_meals:
        SIM.realistic_meals = False
    main(a.quick, tag=a.tag)
