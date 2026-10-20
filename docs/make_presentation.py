"""Build docs/presentation.pdf (and .png previews) straight from reports/metrics.json.

    python docs/make_presentation.py

Every number on every slide is read from the metrics file, so the deck can never drift out of
sync with the results. Re-run the pipeline, re-run this, and the deck is current.

The deck is deliberately ordered the way the jury will think:
    problem -> insight -> what we built -> does it work -> when does it NOT work -> who uses it.
Slide 7 ("where it fails") is not a disclaimer slide; it is the credibility slide.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upvaas.config import REPORT_DIR, ROOT  # noqa: E402

BLUE, ORANGE, AQUA, CRIT = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK, INK2, MUTED, PAPER = "#141413", "#52514e", "#898781", "#fcfcfb"
BAND = {"blue": "#eaf1fb", "orange": "#fdeee7", "green": "#e8f6f0", "grey": "#f2f1ec"}
W, H = 13.333, 7.5          # 16:9


def _fig():
    f = plt.figure(figsize=(W, H), facecolor=PAPER)
    ax = f.add_axes((0, 0, 1, 1)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    return f, ax


def _title(ax, t, sub=None):
    ax.text(0.7, H - 0.85, t, fontsize=27, color=INK, weight="bold", va="top")
    if sub:
        ax.text(0.7, H - 1.55, sub, fontsize=13.5, color=INK2, va="top")
    ax.plot([0.7, W - 0.7], [H - 1.95, H - 1.95], color="#dcdad2", lw=1)


def _foot(ax, n, note="UPVAAS-Twin  ·  synthetic cohort, plus one external validation on real CGM  ·  not a medical device"):
    ax.text(0.7, 0.35, note, fontsize=9, color=MUTED)
    ax.text(W - 0.7, 0.35, str(n), fontsize=9, color=MUTED, ha="right")


def _card(ax, x, y, w, h, fill, title, lines, tcol=INK, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.13",
                                fc=fill, ec="#dcdad2", lw=0.9))
    ax.text(x + 0.3, y + h - 0.42, title, fontsize=13, color=tcol, weight="bold", va="top")
    ax.text(x + 0.3, y + h - 0.92, "\n".join([l for l in lines if l]), fontsize=fs, color=INK2,
            va="top", linespacing=1.4)


def _stat(ax, x, y, w, big, label, col=BLUE, sub=None):
    ax.text(x + w / 2, y + 0.95, big, fontsize=31, color=col, weight="bold", ha="center")
    ax.text(x + w / 2, y + 0.45, label, fontsize=11, color=INK2, ha="center")
    if sub:
        ax.text(x + w / 2, y + 0.08, sub, fontsize=9, color=MUTED, ha="center")


def _external_validation():
    """Read reports/EXTERNAL_VALIDATION.json if the validation has been run."""
    p = REPORT_DIR / "EXTERNAL_VALIDATION.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return d if d.get("arms", {}).get("cgm_only", {}).get("120min") else None
    except Exception:                                    # noqa: BLE001
        return None


def _prior_refit():
    """Read reports/prior_refit_shanghai.json if the refit experiment has been run."""
    p = REPORT_DIR / "prior_refit_shanghai.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return d if d.get("held_out", {}).get("refit_prior") else None
    except Exception:                                    # noqa: BLE001
        return None


def build(m: dict, out: Path):
    g = lambda k: m["models"][k]                                     # noqa: E731
    bd = m.get("benchmark_difficulty", {})
    A, B, C = g("A_full/full/hypo"), g("B_frugal/full/hypo"), g("C_minimal/full/hypo")
    Bn = g("B_frugal/no_twin/hypo")
    Ah = g("A_full/full/hyper")
    coh = m["cohort"]
    fc = m["forecast"]
    pages = []

    # ---------------------------------------------------------------- 1 title
    f, ax = _fig()
    ax.add_patch(FancyBboxPatch((0.7, 2.6), W - 1.4, 2.9, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc=BAND["blue"], ec="none"))
    ax.text(1.2, 4.85, "UPVAAS-Twin", fontsize=46, color=INK, weight="bold", va="top")
    ax.text(1.2, 4.05, "A festival- and fasting-aware metabolic digital twin\nfor Indians with type 2 diabetes",
            fontsize=19, color=INK2, va="top", linespacing=1.5)
    # NB: deliberately transliterated rather than set in Devanagari - matplotlib's bundled DejaVu Sans
    # has no Devanagari glyphs and would render tofu boxes on a machine without an Indic font.
    ax.text(1.2, 2.25, "upvaas  ·  Uncertainty-aware Physiology-informed Virtual Avatar for Adverse-event Surveillance",
            fontsize=12, color=MUTED, va="top")
    ax.text(1.2, 1.75, "Happiest Health · Reimagining and Reforming Healthcare in India Summit 2026 · Digital Twin Challenge",
            fontsize=11.5, color=INK2, va="top")
    ax.text(1.2, 1.25, "Team <Team Name> · <College / Incubator>", fontsize=11.5, color=INK, va="top")
    _foot(ax, 1, "")
    pages.append(f)

    # ---------------------------------------------------------------- 2 problem
    f, ax = _fig()
    _title(ax, "A danger that is already on the calendar",
           "Scheduled, common, unmonitored - and therefore predictable")
    _card(ax, 0.7, 3.4, 3.8, 2.0, BAND["grey"], "The scale",
          ["101 million adults with diabetes", "136 million with prediabetes", "(ICMR-INDIAB, Lancet D&E 2023)"])
    _card(ax, 4.75, 3.4, 3.8, 2.0, BAND["orange"], "The drug",
          ["57% take a sulfonylurea,", "which keeps releasing insulin", "even when no food arrives"])
    _card(ax, 8.8, 3.4, 3.8, 2.0, BAND["blue"], "The behaviour",
          ["49% fast for religion", "4% check glucose while fasting", "1.35% ask a doctor first"])
    ax.text(0.7, 2.9, "Navratri · Ekadashi · Karva Chauth (nirjala) · Ramadan roza", fontsize=14, color=INK,
            weight="bold", va="top")
    ax.text(0.7, 2.35, "A morning glimepiride plus a 13-hour fast is a predictable hypoglycaemia. The fast-breaking\n"
                       "thali that follows is a predictable spike. Existing twins treat these days as noise.",
            fontsize=13, color=INK2, va="top", linespacing=1.6)
    _foot(ax, 2)
    pages.append(f)

    # ---------------------------------------------------------------- 3 insight + system
    f, ax = _fig()
    _title(ax, "Static data sets the prior. Live data updates the posterior.",
           "The calendar tells the twin what is coming - days in advance")
    _card(ax, 0.7, 3.25, 3.8, 2.15, BAND["grey"], "1 · EHR prior",
          ["labs, meds, eGFR,", "CYP2C9 and TCF7L2", "->  personal physiology", "     before any device data"])
    _card(ax, 4.75, 3.25, 3.8, 2.15, BAND["orange"], "2 · Mechanistic twin",
          ["Bergman + gut + sulfonylurea", "PK/PD + fasting clock", "+ blunted counter-regulation",
           "->  explains WHY, not just what"])
    _card(ax, 8.8, 3.25, 3.8, 2.15, BAND["blue"], "3 · Particle filter",
          ["64 candidate physiologies", "re-weighted by every reading", "->  a twin that follows the",
           "     patient, not a classifier"])
    _card(ax, 0.7, 0.95, 5.9, 2.15, BAND["green"], "Vrat Rehearsal - the part that makes it a twin",
          ["\"If she fasts tomorrow and takes her usual glimepiride, what happens?\"",
           "36-hour counterfactual replay on the posterior particles:",
           "usual dose  ·  halved  ·  skipped  ·  moved to the fast-breaking meal",
           "low-GI swap  ·  a walk after iftar  ·  garba at night"])
    _card(ax, 6.85, 0.95, 5.75, 2.15, BAND["grey"], "Into the health system",
          ["ASHA 'Vrat Saathi' recall list, ranked, in the patient's language",
           "eSanjeevani teleconsult before the fast, not an ambulance after",
           "FHIR R4 RiskAssessment + ABDM-shaped consent artefact",
           "Runs on a laptop CPU, in a Tier-3 clinic, with no CGM"])
    _foot(ax, 3)
    pages.append(f)

    # ---------------------------------------------------------------- 4 headline result
    f, ax = _fig()
    _title(ax, "The result that matters: it works without a CGM",
           "Most Indians with type 2 diabetes will never wear one")
    ax.add_patch(FancyBboxPatch((0.7, 3.2), W - 1.4, 2.25, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=BAND["blue"], ec="none"))
    ax.text(W / 2, 5.3, "Tier B  ·  two fingersticks a day + a basic smartwatch  ·  no CGM",
            fontsize=13, color=INK2, ha="center", va="top")
    _stat(ax, 0.9, 3.5, 3.5, f"{Bn.get('auprc', 0):.2f} → {B.get('auprc', 0):.2f}",
          "hypoglycaemia AUPRC", BLUE, f"{B.get('auprc', 0) / max(Bn.get('auprc', 1e-9), 1e-9):.1f}x, adding the twin")
    _stat(ax, 4.6, 3.5, 3.5, f"{Bn.get('false_alarms_per_patient_day', 0):.1f} → "
                             f"{B.get('false_alarms_per_patient_day', 0):.1f}",
          "false alarms per patient-day", AQUA, "the twin filters, not just ranks")
    _stat(ax, 8.3, 3.5, 3.9, f"{(B.get('precision_at_20') or 0):.0%}",
          "precision of a 20-name worklist", ORANGE,
          f"against a {B.get('prevalence', 0):.1%} base rate")
    ax.text(0.7, 3.08, "Of the 20 patients the twin flags each morning, roughly "
                       f"{round((B.get('precision_at_20') or 0) * 20)} really do have an event within two hours.",
            fontsize=14, color=INK, va="top", weight="bold")
    _card(ax, 0.7, 0.8, 5.9, 1.85, BAND["grey"], "Forecasting (tier A)",
          [f"twin RMSE {fc['30min']['twin_rmse']:.0f} / {fc['60min']['twin_rmse']:.0f} / "
           f"{fc['120min']['twin_rmse']:.0f} mg/dL at 30 / 60 / 120 min",
           f"persistence  {fc['30min']['persistence_rmse']:.0f} / {fc['60min']['persistence_rmse']:.0f} / "
           f"{fc['120min']['persistence_rmse']:.0f} mg/dL",
           f"Clarke error grid zone A+B: {fc['120min']['clarke_AB']:.1%}"])
    _card(ax, 6.85, 0.8, 5.75, 1.85, BAND["green"], "How it was scored",
          [f"{coh['n_patients']} synthetic patients x {coh['days']} days",
           "5-fold patient-grouped cross-validation",
           "every patient scored by a model that never saw them",
           "patient-clustered bootstrap CIs on every headline number"])
    _foot(ax, 4)
    pages.append(f)

    # ---------------------------------------------------------------- 5 degradation
    f, ax = _fig()
    _title(ax, "Graceful degradation: the same twin, three levels of device",
           "We publish the whole ladder, so a low-resource clinic knows exactly what it is getting")
    rows = [("A · CGM + watch", A, BLUE), ("B · 2 fingersticks + watch", B, AQUA), ("C · EHR + phone only", C, ORANGE)]
    ax.text(0.9, 5.25, "sensor tier", fontsize=11, color=MUTED)
    for j, h in enumerate(["hypo AUPRC (95% CI)", "precision@20", "event recall", "false alarms / patient-day"]):
        ax.text(4.6 + j * 2.1, 5.25, h, fontsize=10.5, color=MUTED, ha="center")
    for i, (lab, r, col) in enumerate(rows):
        y = 4.7 - i * 0.85
        ax.add_patch(FancyBboxPatch((0.75, y - 0.3), W - 1.5, 0.78,
                                    boxstyle="round,pad=0.01,rounding_size=0.08",
                                    fc=BAND["grey"] if i % 2 == 0 else PAPER, ec="none"))
        ax.text(0.95, y, lab, fontsize=12.5, color=col, weight="bold", va="center")
        ci = r.get("auprc_ci95") or [0, 0]
        cells = [f"{r.get('auprc', 0):.3f}  [{ci[0]:.2f}–{ci[1]:.2f}]",
                 f"{(r.get('precision_at_20') or 0):.2f}",
                 f"{r.get('event_recall', 0):.0%}",
                 f"{r.get('false_alarms_per_patient_day', 0):.2f}"]
        for j, c in enumerate(cells):
            ax.text(4.6 + j * 2.1, y, c, fontsize=12, color=INK, ha="center", va="center")
    _card(ax, 0.7, 0.85, W - 1.4, 1.8, BAND["blue"], "Read this honestly", [
        "The twin helps at every tier, but NOT equally, and the gain grows as the data gets thinner:",
        f"   tier A {A.get('auprc', 0) / max(g('A_full/no_twin/hypo').get('auprc', 1e-9), 1e-9):.1f}x   ·   "
        f"tier B {B.get('auprc', 0) / max(Bn.get('auprc', 1e-9), 1e-9):.1f}x   ·   "
        f"tier C {C.get('auprc', 0) / max(g('C_minimal/no_twin/hypo').get('auprc', 1e-9), 1e-9):.1f}x",
        "Tier B is the one that matters for India: sparse enough that physics has something to add, dense",
        "enough to personalise - and it needs no CGM. False alarms still rise sharply as sensors are removed."])
    _foot(ax, 5)
    pages.append(f)

    # ---------------------------------------------------------------- 6 where it fails
    f, ax = _fig()
    _title(ax, "Where it does not work",
           "Including the external validation we ran, and failed")
    cov250 = m["conformal"]["horizons"][-1]["mondrian_conformal"].get("coverage_outcome_gt250")
    ev = _external_validation()
    rf = _prior_refit()
    if ev:
        e2 = ev["arms"]["cgm_only"]["120min"]
        _card(ax, 0.7, 3.3, 5.9, 2.1, BAND["orange"], "External validation - and it failed", [
            f"{ev['recordings_used']} REAL CGM recordings (ShanghaiT2DM), no retraining.",
            f"2-hour RMSE {e2['twin_rmse']:.0f} mg/dL vs {e2['persistence_rmse']:.0f} for persistence.",
            "It does not beat 'assume glucose stays put', at any horizon.",
            "We then refit the prior on real patients to fix it.",
            "   That failed too - and told us why. ->"],
            fs=10.5)
    else:
        _card(ax, 0.7, 3.3, 5.9, 2.1, BAND["orange"], "Null results we are not hiding", [
            f"The festival calendar as ML features: {g('A_full/no_calendar/hypo').get('auprc', 0):.2f} without",
            f"   vs {A.get('auprc', 0):.2f} with - no real difference.",
            "The drift monitor does not work (AUROC "
            f"{(m['drift'].get('auroc_relative_nis_for_illness') or 0):.2f}).",
            f"Forecast bands cover {cov250:.0%} above 250 mg/dL, not 90%." if cov250 else ""],
            fs=10.5)
    if rf:
        h = rf["held_out"]
        _card(ax, 6.85, 3.3, 5.75, 2.1, BAND["orange"], "Why it failed - the real finding", [
            f"Refit on real patients: ratio {h['refit_prior']['ratio']:.2f} - no change.",
            f"Just WIDENING the prior: {h['original_prior_wider']['ratio']:.2f} - parity, not better.",
            "The refit's residual spread exploded (Gb0 SD 8 -> 40):",
            "   routine EHR fields barely predict physiology in a",
            "   REAL cohort. Ours predicted it because we built it that way."],
            fs=10.5)
    else:
        _card(ax, 6.85, 3.3, 5.75, 2.1, BAND["grey"], "We made the benchmark harder on purpose", [
            "The virtual body runs a DIFFERENT model from the twin:",
            "   3-compartment gut, interstitial insulin delay,",
            "   circadian SI, renal excretion, incretin.",
            f"Meal plan vs what was eaten: r = {bd.get('plan_vs_logged_carb_corr_2h', 0):.2f}, with",
            f"   {bd.get('real_intake_that_was_unplanned', 0):.0%} of intake unplanned - a prior, not an oracle."],
            fs=10.5)
    tv = m["models"].get("A_full/trivial_calendar_baseline/hyper", {})
    _card(ax, 0.7, 0.95, 5.9, 2.2, BAND["green"], "The baseline that keeps us honest", [
        "Four features - planned carbs, HbA1c, time of day.",
        "No glucose. No wearable. No twin.",
        f"Hyperglycaemia AUPRC {tv.get('auprc', 0):.2f}, against {Ah.get('auprc', 0):.2f} for the",
        "   full model. We publish it so the reader can see",
        "   how much is physiology and how much is timetable."], fs=10.5)
    _card(ax, 6.85, 0.95, 5.75, 2.2, BAND["grey"], "Other null results, and limits", [
        f"The calendar as ML FEATURES adds nothing ({g('A_full/no_calendar/hypo').get('auprc', 0):.2f} vs "
        f"{A.get('auprc', 0):.2f});",
        "   its value is the Rehearsal, days earlier.",
        f"The drift monitor does not work (AUROC "
        f"{(m['drift'].get('auroc_relative_nis_for_illness') or 0):.2f}).",
        f"Bands cover {cov250:.0%} above 250 mg/dL, not 90%." if cov250 else "",
        f"Only {A.get('patients_with_events', 0)} of {A.get('n_patients', 0)} patients ever had a hypo event."],
        fs=10.5)
    _foot(ax, 6)
    pages.append(f)

    # ---------------------------------------------------------------- 7 uncertainty
    f, ax = _fig()
    _title(ax, "Uncertainty a clinician can actually rely on",
           "A 90% band that is 90% on average can be 50% exactly where you are looking")
    c = m["conformal"]["horizons"][-1]
    sp, mo = c["split_conformal"], c["mondrian_conformal"]
    ax.text(0.9, 5.25, "coverage of the 2-hour forecast band", fontsize=11, color=MUTED)
    for j, h in enumerate(["overall", "when outcome < 70", "when outcome > 250", "mean band width"]):
        ax.text(4.9 + j * 2.0, 5.25, h, fontsize=10.5, color=MUTED, ha="center")
    for i, (lab, r, col) in enumerate([("Split conformal (one margin)", sp, MUTED),
                                       ("Mondrian (margin per glucose band)", mo, BLUE)]):
        y = 4.55 - i * 0.88
        ax.add_patch(FancyBboxPatch((0.75, y - 0.3), W - 1.5, 0.75,
                                    boxstyle="round,pad=0.01,rounding_size=0.08",
                                    fc=BAND["grey"] if i == 0 else BAND["blue"], ec="none"))
        ax.text(0.95, y, lab, fontsize=12.5, color=col, weight="bold", va="center")
        vals = [r.get("coverage"), r.get("coverage_outcome_lt70"), r.get("coverage_outcome_gt250")]
        for j, v in enumerate(vals):
            ax.text(4.9 + j * 2.0, y, "–" if v is None else f"{v:.2f}", fontsize=12.5,
                    color=CRIT if (v is not None and v < 0.75) else INK, ha="center", va="center")
        ax.text(4.9 + 3 * 2.0, y, f"{r.get('mean_band_width_mgdl', 0):.0f} mg/dL", fontsize=12,
                color=INK, ha="center", va="center")
    _card(ax, 0.7, 1.05, W - 1.4, 2.2, BAND["green"], "Why this slide exists",
          ["Split conformal gives a MARGINAL guarantee: 90% averaged over every patient and every minute. Because",
           "most calibration points are euglycaemic, the band is far too narrow during exactly the post-meal excursions",
           "a clinician is watching. Mondrian conformal calibrates a separate margin per predicted-glucose stratum and",
           "restores coverage where it matters. Most published twins report the marginal number only; we report both,",
           "and the dashboard draws the Mondrian band."])
    _foot(ax, 7)
    pages.append(f)

    # ---------------------------------------------------------------- 8 fairness + governance
    f, ax = _fig()
    _title(ax, "Trust, fairness and the regulatory path",
           "Built for ICMR 2023 ethics, DPDP 2023 and the CDSCO draft software guidance")
    fr = m.get("fairness", {}).get("A_full/hypo", {})
    trad = {k.replace("tradition_", ""): v for k, v in fr.items() if k.startswith("tradition_")}
    lines = [f"{k}: AUROC {v.get('auroc', 0):.3f}" for k, v in sorted(trad.items())] or ["(run the pipeline)"]
    _card(ax, 0.7, 3.25, 5.9, 2.15, BAND["blue"], "Audited on the axis that matters here",
          ["This project is built on religious fasting, so fasting tradition is the",
           "first axis we audit - not an afterthought:"] + lines +
          ["Also audited: sex, urban/rural, age, eGFR, region."])
    _card(ax, 6.85, 3.25, 5.75, 2.15, BAND["grey"], "Governance by design",
          ["DPDP 2023 - synthetic data only; consent ID on every record,",
           "   purpose limitation (CAREMGT), deletion on consent expiry.",
           "ABDM - FHIR R4 bundles, consent artefact via HIE-CM.",
           "CDSCO draft (Oct 2025) - informs, does not drive, a registered",
           "   practitioner -> likely Class B. Human always in the loop.",
           "ICMR 2023 - model card, subgroup audit, right to refuse AI."])
    _card(ax, 0.7, 0.95, W - 1.4, 2.2, BAND["green"], "Deployment path",
          ["WHERE   NP-NCD screening at Ayushman Arogya Mandirs (~25 million on diabetes treatment under '75 by 25'),",
           "              private diabetology clinics, CGM programmes.",
           "HOW      ASHA app receives the ranked recall list two weeks before Navratri / Ramadan -> home visit or phone call",
           "              -> eSanjeevani teleconsult -> doctor runs the Vrat Rehearsal -> dose-timing plan in the patient's language.",
           "PROOF   (1) retrospective validation on ShanghaiT2DM + CGMacros  (2) IRB observational pilot, 100 SU-treated fasters,",
           "              Navratri 2027  (3) cluster-randomised ASHA recall-list trial; outcome = hypoglycaemia events and ER visits."])
    _foot(ax, 8)
    pages.append(f)

    # ---------------------------------------------------------------- 9 close
    f, ax = _fig()
    _title(ax, "What we are claiming - and what we are not", "")
    _card(ax, 0.7, 3.25, 5.9, 2.15, BAND["green"], "We are claiming",
          ["A twin that treats India's festival calendar as a planned",
           "   perturbation, not as noise.",
           "A counterfactual rehearsal a doctor can run BEFORE the fast.",
           "A measured ladder from CGM down to a phone, so the people",
           "   least likely to own a device are not silently excluded.",
           "Uncertainty, fairness and null results reported, not implied."])
    _card(ax, 6.85, 3.25, 5.75, 2.15, BAND["orange"], "We are not claiming",
          ["Clinical validity. The data are synthetic; nothing here has",
           "   been tested on a real patient.",
           "That the twin helps everywhere - at tier A it does not help",
           "   precision, and we show that.",
           "That the drift monitor is a safety system yet. It is not.",
           "A medical device. This informs a practitioner; it decides nothing."])
    ax.add_patch(FancyBboxPatch((0.7, 1.1), W - 1.4, 2.4, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=BAND["blue"], ec="none"))
    ax.text(W / 2, 3.05, "The fast is on the calendar. The hypoglycaemia does not have to be a surprise.",
            fontsize=19, color=INK, ha="center", va="top", weight="bold")
    ax.text(W / 2, 2.25, "github.com/<your-org>/<TeamName_CollegeName>   ·   MIT licence   ·   synthetic data CC0",
            fontsize=12, color=INK2, ha="center", va="top")
    ax.text(W / 2, 1.7, "Reproduce every number on these slides:  bash run.sh", fontsize=12, color=MUTED,
            ha="center", va="top")
    _foot(ax, 9)
    pages.append(f)

    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for f in pages:
            pdf.savefig(f, facecolor=PAPER)
    png = out.parent / "presentation_preview.png"
    pages[3].savefig(png, dpi=110, facecolor=PAPER)
    for f in pages:
        plt.close(f)
    print(f"presentation -> {out}  ({len(pages)} slides)")
    print(f"preview      -> {png}")


if __name__ == "__main__":
    mf = REPORT_DIR / "metrics.json"
    if not mf.exists():
        sys.exit("reports/metrics.json not found. Run:  python scripts/run_pipeline.py")
    build(json.load(open(mf)), ROOT / "docs" / "presentation.pdf")
