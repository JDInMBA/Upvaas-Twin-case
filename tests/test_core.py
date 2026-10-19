"""Fast unit tests for the physiology, calendar, twin utilities, metrics and FHIR export.

    python -m pytest -q tests
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upvaas.calendar_india import day_plan, load_calendar  # noqa: E402
from upvaas.config import SIM, THR  # noqa: E402
from upvaas.ehr.synth_ehr import generate_cohort  # noqa: E402
from upvaas.interop.fhir import patient_bundle  # noqa: E402
from upvaas.physiology.model import add_meal, init_state, params_from_truth, stack_params, step  # noqa: E402
from upvaas.physiology.simulate import simulate_cohort  # noqa: E402
from upvaas.twin.engine import TWIN_FEATURES, _sample_idx  # noqa: E402
from upvaas.twin.features import build_features, labels, sustained  # noqa: E402
from upvaas.twin.predictor import (apply_mondrian, clarke_zone, conformal_band,  # noqa: E402
                                   mondrian_bands)
from upvaas.twin.priors import cyp_activity  # noqa: E402


def _params(**over):
    base = {"SI": 3e-4, "beta": 0.08, "Gb0": 140.0, "SG": 0.012, "p2": 0.025, "n": 0.12, "su_gain": 7.0,
            "su_thalf_min": 360.0, "cr_gain": 0.04, "dawn": 0.0, "bw": 65.0, "ins_basal": 0.0}
    base.update(over)
    return stack_params([params_from_truth(base, on_met=True)])


def _fast_day(su_dose: float, hours=14):
    p = _params()
    s = init_state((1,), 126.0)
    s["A"] = s["A"] + su_dose
    for t in range(hours * 60):
        step(s, p, 1.0, (8 * 60 + t) % 1440)
    return float(s["G"][0])


def test_sulfonylurea_on_fasting_day_lowers_glucose():
    assert _fast_day(1.0) < _fast_day(0.0) - 15


def test_meal_raises_then_returns():
    p = _params(su_gain=0.0)
    s = init_state((1,), 126.0)
    add_meal(s, np.array([60000.0]), 0.02)
    peak = 0.0
    for t in range(360):
        step(s, p, 1.0, 13 * 60 + t)
        peak = max(peak, float(s["G"][0]))
    assert peak > 170 and float(s["G"][0]) < peak - 40


def test_calendar_karva_chauth_and_navratri():
    cal = load_calendar()
    kc = {"tradition": "karva_chauth", "region": "north", "vrat_style": "phalahar", "garba": False}
    assert day_plan(date(2026, 10, 29), kc, cal).day_type == "karva_chauth_nirjala"
    assert day_plan(date(2026, 10, 13), kc, cal).day_type == "navratri_phalahar"
    nofast = dict(kc, tradition="none")
    assert day_plan(date(2026, 10, 13), nofast, cal).day_type == "normal"


def test_cohort_is_reproducible_and_plausible():
    e1, t1 = generate_cohort(30, seed=1)
    e2, _ = generate_cohort(30, seed=1)
    assert e1.equals(e2)
    assert e1["hba1c"].between(5.7, 13.5).all()
    assert (t1["SI"] > 0).all()


def test_cyp_activity():
    assert cyp_activity("*1/*1") == 2.0 and cyp_activity("*3/*3") == 0.0


def test_systematic_resampling_follows_weights():
    w = np.array([[0.0, 0.0, 1.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    idx = _sample_idx(w, 8, np.random.default_rng(0))
    assert (idx[0] == 2).all() and set(idx[1]) == {0, 1, 2, 3}


def test_sustained_runs():
    m = np.array([[0, 1, 1, 1, 0, 1, 1, 0]], bool)
    assert sustained(m, 3).tolist() == [[False, True, True, True, False, False, False, False]]


def test_clarke_zones():
    z = clarke_zone(np.array([100, 60, 250, 200]), np.array([105, 200, 120, 60]))
    assert z.tolist() == ["A", "E", "D", "E"]


def test_conformal_margin_reaches_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    q = conformal_band(np.full(5000, -0.5), np.full(5000, 0.5), y, alpha=0.1)
    y2 = rng.normal(0, 1, 5000)
    cov = ((y2 >= -0.5 - q) & (y2 <= 0.5 + q)).mean()
    assert 0.88 < cov < 0.92


def test_fhir_bundle_contains_risk_and_consent():
    e, _ = generate_cohort(2, seed=3)
    pid = e.index[0]
    b = patient_bundle({"patient_id": pid, **e.loc[pid].to_dict()}, risk={"p_hypo": 0.4, "p_hyper": 0.1})
    types = {x["resource"]["resourceType"] for x in b["entry"]}
    assert {"Patient", "RiskAssessment", "Consent", "Observation"} <= types


# --------------------------------------------------------------------------- leakage regressions
# The unit tests above check that the physics is sane. These check the thing that would actually
# invalidate the results: a feature or a label quietly reading the future.

def test_labels_look_only_forward_and_exclude_ongoing_events():
    """y=1 must mean 'an event STARTS in the next 2 h', never 'an event is happening now'."""
    T5 = 600
    g = np.full((1, T5), 120.0)
    g[0, 300:310] = 50.0                      # a sustained hypo at k=300..309
    ds = np.arange(290, 330, 6)
    lab = labels(g, ds)
    H = THR.horizon_min // 5
    for j, k in enumerate(ds):
        if not lab["ok_hypo"][0, j]:
            continue                           # row excluded (event already running) - fine
        starts_ahead = lab["sus_hypo"][0, k + 1:min(k + 1 + H, T5)].any()
        assert bool(lab["y_hypo"][0, j]) == bool(starts_ahead)
    # a decision point taken while glucose is already below 70 must be excluded, not labelled
    assert not lab["ok_hypo"][0, list(ds).index(302)] if 302 in list(ds) else True


def test_no_feature_reads_future_truth():
    """Perturbing the future must not change any feature computed at the decision time.

    This is the regression that matters: if a window index is ever off by one, or a cumulative
    sum slices one step too far, this test fails and the headline numbers are invalid.
    """
    rng = np.random.default_rng(0)
    ehr, truth = generate_cohort(4, seed=5)
    st = simulate_cohort(ehr, truth, n_days=3, seed=7, save=False)
    cut = 288 + 60                              # decision index we will protect
    twin = {"dsteps": np.array([cut]), "feature_names": TWIN_FEATURES,
            "feats": np.zeros((len(ehr), 1, len(TWIN_FEATURES)), np.float32)}
    base = build_features(ehr, st, twin, "A_full")

    poisoned = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    for key in ("cgm", "g_true", "hr", "rmssd", "steps5", "fingerstick", "log_c", "su_log"):
        a = poisoned[key]
        a[:, cut + 1:] = a[:, cut + 1:] * 0 + rng.normal(300, 50, a[:, cut + 1:].shape)
    after = build_features(ehr, poisoned, twin, "A_full")

    # labels legitimately depend on the future; features must not.
    feat_cols = [c for c in base.columns if c not in
                 ("y_hypo", "y_hyper", "ok_hypo", "ok_hyper", "patient_id", "day_type")]
    for c in feat_cols:
        if c.startswith("plan_"):
            continue                            # the calendar plan is knowable in advance by design
        b, a = base[c].values.astype(float), after[c].values.astype(float)
        same = np.isclose(b, a, equal_nan=True)
        assert same.all(), f"feature '{c}' changed when only the FUTURE was perturbed -> look-ahead leak"


def test_twin_and_body_are_different_models():
    """The virtual body must not be integrated with the twin's own equations by default.

    If these ever become the same function again, every accuracy number in the repo silently
    becomes an upper bound that measures parameter error only.
    """
    from upvaas.physiology import model as twin_model
    from upvaas.physiology import truth_model

    assert SIM.model_mismatch, "SIM.model_mismatch must default to True"
    assert truth_model.step is not twin_model.step
    # the body carries state the twin has no equation for
    s_body = truth_model.init_state((1,), 120.0)
    s_twin = twin_model.init_state((1,), 120.0)
    assert {"Q3", "Id"} <= set(s_body) and not {"Q3", "Id"} & set(s_twin)


def test_mondrian_conformal_beats_split_in_the_tail():
    """A single margin under-covers where the spread is largest; per-stratum margins fix it."""
    rng = np.random.default_rng(0)
    pred = np.concatenate([rng.normal(110, 10, 4000), rng.normal(260, 10, 1000)])
    noise = np.where(pred > 200, rng.normal(0, 40, len(pred)), rng.normal(0, 8, len(pred)))
    y = pred + noise
    lo, hi = pred - 5, pred + 5
    q = conformal_band(lo, hi, y, alpha=0.1)
    split_tail = ((y > 200) & (y >= lo - q) & (y <= hi + q)).sum() / (y > 200).sum()
    mb = mondrian_bands(lo, hi, y, pred, alpha=0.1)
    qm = apply_mondrian(mb, pred)
    mond_tail = ((y > 200) & (y >= lo - qm) & (y <= hi + qm)).sum() / (y > 200).sum()
    assert mond_tail > split_tail + 0.2 and mond_tail > 0.8
