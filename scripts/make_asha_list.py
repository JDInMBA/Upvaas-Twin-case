"""Rebuild the ASHA pre-festival recall list from saved artifacts (no need to rerun the whole pipeline).

    python scripts/make_asha_list.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upvaas.config import ARTIFACT_DIR, COHORT_DIR, REPORT_DIR  # noqa: E402
from upvaas.population.asha import recall_list  # noqa: E402
from upvaas.twin.priors import PhysiologyPrior  # noqa: E402


def build(ehr, day_type, theta_daily, state_mean, dsteps, prior, g_true=None):
    theta_last, state_last, fast_day = [], [], []
    for i in range(len(ehr)):
        fast_days = np.where(np.isin(day_type[i], [1, 2, 3, 4]))[0]
        nd = int(fast_days[fast_days >= 2][0]) if (fast_days >= 2).any() else 10
        k_eve = (nd - 1) * 288 + 21 * 12
        j = int(np.argmin(np.abs(dsteps - k_eve)))
        theta_last.append(theta_daily[i, nd - 2]); state_last.append(state_mean[i, j]); fast_day.append(nd)
    asha = recall_list(ehr, np.array(theta_last), np.array(state_last), prior.fixed, prior.predict(ehr)[0]["cr"].values)
    if g_true is not None:   # did a hypo actually happen on the rehearsed fast day? (simulated ground truth)
        pos = {pid: i for i, pid in enumerate(ehr.index)}
        obs = []
        for pid in asha.patient_id:
            i = pos[pid]; nd = fast_day[i]
            gg = g_true[i, nd * 288:(nd + 1) * 288] < 70
            run, hit = 0, False
            for v in gg:
                run = run + 1 if v else 0
                hit |= run >= 3
            obs.append(hit)
        asha["observed_hypo_on_fast_day_SIMULATED"] = obs
    return asha


def summary(asha):
    hy = asha[asha.on_sulfonylurea]
    out = {}
    if "observed_hypo_on_fast_day_SIMULATED" in asha:
        from sklearn.metrics import brier_score_loss, roc_auc_score

        def _score(mask, label):
            y = asha.loc[mask, "observed_hypo_on_fast_day_SIMULATED"].astype(int).values
            p = asha.loc[mask, "p_hypo_usual"].values
            return {"split": label, "n_fast_days": int(len(y)), "n_events": int(y.sum()),
                    "observed_rate": float(y.mean()) if len(y) else None,
                    "mean_predicted": float(p.mean()) if len(p) else None,
                    "brier": float(brier_score_loss(y, p)) if len(y) else None,
                    "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None}

        # Split by a stable hash of patient_id so the two halves are reproducible and disjoint.
        half = asha["patient_id"].map(lambda s: int(s.split("-")[-1]) % 2 == 0).values
        y_all = asha["observed_hypo_on_fast_day_SIMULATED"].astype(int).values
        out["rehearsal_validation"] = {
            **_score(np.ones(len(asha), bool), "all"),
            "held_out_halves": [_score(half, "half_A"), _score(~half, "half_B")],
            "caveat": (
                "This is the weakest number in the submission and we flag it rather than lead with it. "
                f"Only {int(y_all.sum())} of {len(y_all)} rehearsed fast days actually ended in a hypoglycaemic "
                "event, so the AUROC has a very wide confidence interval; the two held-out halves are reported so "
                "the spread is visible. The rehearsal also over-predicts the event rate, which is the safe "
                "direction for a recall list but means the probabilities are not calibrated. Treat as indicative, "
                "not validated."),
        }
    return {**out, "patients_listed": int(len(asha)), "high_priority": int((asha.priority == "high").sum()),
            "mean_p_hypo_usual_su": float(hy.p_hypo_usual.mean()) if len(hy) else None,
            "mean_p_hypo_dose_moved_su": float(hy.p_hypo_if_dose_moved.mean()) if len(hy) else None}


if __name__ == "__main__":
    ehr = pd.read_csv(COHORT_DIR / "ehr.csv", index_col=0)
    ehr["su_drug"] = ehr["su_drug"].fillna("")
    st = np.load(COHORT_DIR / "streams.npz")
    tw = np.load(ARTIFACT_DIR / "twin_A_full.npz")
    asha = build(ehr, st["day_type"], tw["theta_daily"], tw["state_mean"], tw["dsteps"], PhysiologyPrior.load(), st["g_true"])
    asha.to_csv(REPORT_DIR / "asha_pre_festival_recall_list.csv", index=False)
    m = json.load(open(REPORT_DIR / "metrics.json"))
    m["asha"] = summary(asha)
    json.dump(m, open(REPORT_DIR / "metrics.json", "w"), indent=1)
    print(m["asha"])
