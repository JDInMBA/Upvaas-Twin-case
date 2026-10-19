"""Hybrid risk predictor: gradient boosting over fused features, isotonic calibration, conformal bands,
and the evaluation suite (discrimination, calibration, lead time, fairness, graceful degradation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


class RiskModel:
    """Calibrated event-risk model for one target ('hypo' or 'hyper') and one feature set.

    Early stopping is OFF on purpose. scikit-learn's internal `validation_fraction` splits *rows*
    at random, and each patient contributes ~1,344 highly autocorrelated rows, so the internal
    validation set would be full of near-duplicates of the training rows and the iteration count
    would be chosen on leaked data. The outer cross-validation is patient-grouped; the inner loop
    must be too, so we fix the iteration budget and rely on l2 + min_samples_leaf for control.
    """

    def __init__(self, target: str, features: list[str], name: str = ""):
        self.target, self.features, self.name = target, features, name
        self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
                                                  l2_regularization=1.0, min_samples_leaf=40,
                                                  early_stopping=False, random_state=0)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.threshold = 0.5

    def fit(self, train: pd.DataFrame, cal: pd.DataFrame, target_recall: float = 0.8, neg_frac: float = 0.35,
            seed: int = 0):
        # keep every positive, subsample negatives (speed); isotonic calibration on the untouched
        # calibration patients restores honest probabilities afterwards
        y = train[f"y_{self.target}"].values
        keep = (y == 1) | (np.random.default_rng(seed).random(len(y)) < neg_frac)
        self.clf.fit(train[self.features].values[keep], y[keep])
        raw = self.clf.predict_proba(cal[self.features].values)[:, 1]
        y = cal[f"y_{self.target}"].values
        self.iso.fit(raw, y)
        p = self.iso.predict(raw)
        # alarm threshold: highest threshold that still reaches the target recall on calibration data
        order = np.sort(np.unique(p))[::-1]
        thr = order[-1]
        for t in order:
            if (p[y == 1] >= t).mean() >= target_recall:
                thr = t
                break
        self.threshold = float(thr)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.iso.predict(self.clf.predict_proba(df[self.features].values)[:, 1])


def ece(y, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def classification_metrics(y: np.ndarray, p: np.ndarray, thr: float, probabilistic: bool = True) -> dict:
    y = y.astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "events": int(y.sum())}
    a = p >= thr
    tp, fp = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
    out = {"n": int(len(y)), "events": int(y.sum()), "prevalence": float(y.mean()),
           "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
           "recall": tp / max(int(y.sum()), 1), "precision": tp / max(tp + fp, 1), "threshold": float(thr)}
    if probabilistic:
        out.update(brier=float(brier_score_loss(y, p)), ece=ece(y, p))
    return out


def event_lead_times(df: pd.DataFrame, p: np.ndarray, thr: float, sus: np.ndarray, target: str) -> dict:
    """Event-level view: for each event onset, was there an alarm in the preceding 2 h, and how early?

    Lead time is capped at 120 min by construction (the horizon), and it is reported two ways,
    because the naive version flatters models that alarm constantly:
      * `median_lead_min`      - time from the FIRST alarm in the 2 h window. A model that alarms
                                 all day scores ~120 min here without predicting anything.
      * `median_lead_last_min` - time from the LAST alarm before onset, i.e. the warning the
                                 clinician actually acts on.
    Compare lead times across tiers only at a matched false-alarm rate (see `lead_time_at_fa`).
    """
    alarms = df.assign(alarm=p >= thr)
    leads, leads_last, caught, total = [], [], 0, 0
    false_alarm_days = 0.0
    for pidx, g in alarms.groupby("pidx"):
        s = sus[pidx]
        onsets = np.where(s[1:] & ~s[:-1])[0] + 1
        ks, al = g["k"].values, g["alarm"].values
        lo, hi = ks.min() + 1, ks.max() + 24
        for o in onsets[(onsets >= lo) & (onsets <= hi)]:
            total += 1
            win = (ks < o) & (ks >= o - 24) & al
            if win.any():
                caught += 1
                leads.append((o - ks[win].min()) * 5)
                leads_last.append((o - ks[win].max()) * 5)
        # false alarms: alarms whose next 2 h contain no event
        fa = al & ~g[f"y_{target}"].values.astype(bool)
        false_alarm_days += fa.sum()
    days = len(df) / 48.0
    return {"event_onsets": total, "event_recall": caught / max(total, 1),
            "median_lead_min": float(np.median(leads)) if leads else None,
            "median_lead_last_min": float(np.median(leads_last)) if leads_last else None,
            "false_alarms_per_patient_day": float(false_alarm_days / max(days, 1))}


def lead_time_at_fa(df: pd.DataFrame, p: np.ndarray, sus: np.ndarray, target: str,
                    fa_per_patient_day: float = 0.5) -> dict:
    """Re-score lead time and recall at a FIXED false-alarm budget.

    Without this, a tier that alarms six times a day appears to give the longest warning. Here the
    threshold is moved until every tier raises the same number of false alarms per patient-day, so
    the comparison is like for like.
    """
    y = df[f"y_{target}"].values.astype(bool)
    days = max(len(df) / 48.0, 1e-9)
    target_fa = fa_per_patient_day * days
    order = np.unique(p)[::-1]
    thr = order[0]
    for t in order[:: max(1, len(order) // 400)]:
        if ((p >= t) & ~y).sum() > target_fa:
            break
        thr = t
    out = event_lead_times(df, p, float(thr), sus, target)
    out["threshold_at_fa"] = float(thr)
    out["fa_budget_per_patient_day"] = fa_per_patient_day
    return out


def bootstrap_ci(y: np.ndarray, p: np.ndarray, patient: np.ndarray, metric, n_boot: int = 400,
                 seed: int = 0, alpha: float = 0.05) -> list:
    """Patient-clustered bootstrap CI.

    Resampling rows would badly understate the interval: rows within a patient are not independent,
    and only a minority of patients ever experience an event. Resampling PATIENTS gives the honest
    width, which for rare events is wide.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    y, p, patient = y[ok], p[ok], np.asarray(patient)[ok]
    pats = np.unique(patient)
    idx = {q: np.where(patient == q)[0] for q in pats}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[q] for q in rng.choice(pats, len(pats), replace=True)])
        if 0 < y[ii].sum() < len(ii):
            vals.append(metric(y[ii], p[ii]))
    if not vals:
        return [None, None]
    return [float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))]


def patients_with_events(y: np.ndarray, patient: np.ndarray) -> int:
    """How many distinct patients contribute at least one positive row (the real sample size)."""
    return int(len(np.unique(np.asarray(patient)[np.asarray(y).astype(bool)])))


def precision_at_k(df: pd.DataFrame, p: np.ndarray, target: str, ks=(10, 20, 40)) -> dict:
    """Worklist precision: of the k riskiest patient-timepoints on a given day, how many are real?

    This is the number a clinic actually feels. AUPRC answers "is the ranking good"; precision@k
    answers "if I open the top 20 names each morning, how many of them matter".
    """
    d = df.assign(_p=p, _day=(df["k"].values * 5) // 1440)
    y = f"y_{target}"
    out = {}
    for k in ks:
        prec = [g.nlargest(k, "_p")[y].values.astype(int).mean() for _, g in d.groupby("_day") if len(g) >= k]
        out[f"precision_at_{k}"] = float(np.mean(prec)) if prec else None
    out["prevalence"] = float(df[y].values.astype(int).mean())
    return out


def conformal_band(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal widening of the twin's particle band (CQR-style). Returns the margin Q.

    NOTE: this gives a MARGINAL guarantee - 90% coverage averaged over all patients and all times.
    It does not promise 90% in any particular glucose range, and in practice it does not deliver
    it: a single margin fitted mostly on euglycaemic points is far too narrow for post-meal
    excursions. Use `mondrian_bands` for coverage where the clinician is actually looking.
    """
    s = np.maximum(lo - y, y - hi)
    s = s[~np.isnan(s)]
    n = len(s)
    if n == 0:
        return float("nan")
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(s, min(q, 1.0)))


# Bands are calibrated separately inside each predicted-glucose stratum. The strata are chosen on
# the twin's own PREDICTION (which is available at run time), never on the outcome.
MONDRIAN_EDGES = (0.0, 90.0, 140.0, 200.0, np.inf)
MONDRIAN_LABELS = ("pred<90", "90-140", "140-200", "pred>200")


def mondrian_strata(pred: np.ndarray) -> np.ndarray:
    """Stratum index for each predicted value (see MONDRIAN_EDGES)."""
    return np.clip(np.digitize(np.asarray(pred, float), np.array(MONDRIAN_EDGES[1:-1])), 0,
                   len(MONDRIAN_LABELS) - 1)


def mondrian_bands(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, pred: np.ndarray,
                   alpha: float = 0.1, min_n: int = 200) -> dict:
    """Per-stratum conformal margins (Mondrian / class-conditional conformal prediction).

    Returns {stratum_index: margin}. Strata with too few calibration points fall back to the
    pooled marginal margin, so the method degrades to standard split conformal rather than
    producing a margin fitted on noise.
    """
    pooled = conformal_band(lo, hi, y, alpha)
    st = mondrian_strata(pred)
    out = {}
    for j in range(len(MONDRIAN_LABELS)):
        m = (st == j) & ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi)
        out[j] = conformal_band(lo[m], hi[m], y[m], alpha) if m.sum() >= min_n else pooled
    return out


def apply_mondrian(margins: dict, pred: np.ndarray) -> np.ndarray:
    """Margin per point, looked up by the stratum of its prediction."""
    st = mondrian_strata(pred)
    return np.array([margins[int(j)] for j in st.ravel()]).reshape(np.shape(pred))


THR_HYPO, THR_HYPER = 70.0, 250.0


def coverage_report(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, inside: np.ndarray) -> dict:
    """Marginal coverage plus coverage conditional on the OUTCOME range.

    A 90% band that covers 99% of euglycaemic points and 46% of hyperglycaemic ones is 90% overall
    and useless exactly where the clinician is looking, so both are reported.
    """
    def cov(mask):
        return float(inside[mask].mean()) if mask.sum() > 30 else None
    width = (hi - lo)
    return {"coverage": float(inside.mean()),
            "coverage_outcome_lt70": cov(y < THR_HYPO), "coverage_outcome_70_180": cov((y >= THR_HYPO) & (y <= 180)),
            "coverage_outcome_gt250": cov(y > THR_HYPER),
            "mean_band_width_mgdl": float(np.nanmean(width))}


CLARKE_ZONES = "ABCDE"


def clarke_zone(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Clarke error grid zone for each (reference, prediction) pair (mg/dL)."""
    r, p = np.asarray(ref, float), np.asarray(pred, float)
    z = np.full(r.shape, "B", dtype="<U1")
    a = ((r <= 70) & (p <= 70)) | ((p <= 1.2 * r) & (p >= 0.8 * r))
    e = ((r >= 180) & (p <= 70)) | ((r <= 70) & (p >= 180))
    c = ((r >= 70) & (r <= 290) & (p >= r + 110)) | ((r >= 130) & (r <= 180) & (p <= (7 / 5) * r - 182))
    d = ((r >= 240) & (p >= 70) & (p <= 180)) | ((r <= 175 / 3) & (p <= 180) & (p >= 70)) | \
        ((r >= 175 / 3) & (r <= 70) & (p >= (6 / 5) * r))
    z[d] = "D"; z[c] = "C"; z[e] = "E"; z[a] = "A"
    return z


def subgroup_metrics(df: pd.DataFrame, p: np.ndarray, target: str, thr: float, groups: dict) -> dict:
    out = {}
    y = df[f"y_{target}"].values
    for gname, mask in groups.items():
        m = mask.values if hasattr(mask, "values") else mask
        if m.sum() > 50:
            out[gname] = classification_metrics(y[m], p[m], thr)
    return out
