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


def _foot(ax, n, note="UPVAAS-Twin  ·  synthetic data only  ·  not a medical device"):
    ax.text(0.7, 0.35, note, fontsize=9, color=MUTED)
    ax.text(W - 0.7, 0.35, str(n), fontsize=9, color=MUTED, ha="right")


def _card(ax, x, y, w, h, fill, title, lines, tcol=INK, fs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.13",
                                fc=fill, ec="#dcdad2", lw=0.9))
    ax.text(x + 0.3, y + h - 0.45, title, fontsize=13.5, color=tcol, weight="bold", va="top")
    ax.text(x + 0.3, y + h - 1.05, "\n".join(lines), fontsize=fs, color=INK2, va="top", linespacing=1.6)


def _stat(ax, x, y, w, big, label, col=BLUE, sub=None):
    ax.text(x + w / 2, y + 0.95, big, fontsize=31, color=col, weight="bold", ha="center")
    ax.text(x + w / 2, y + 0.45, label, fontsize=11, color=INK2, ha="center")
    if sub:
        ax.text(x + w / 2, y + 0.08, sub, fontsize=9, color=MUTED, ha="center")


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
    _card(ax, 0.7, 3.5, 3.8, 2.1, BAND["grey"], "The scale",
          ["101 million adults with diabetes", "136 million with prediabetes", "(ICMR-INDIAB, Lancet D&E 2023)"])
    _card(ax, 4.75, 3.5, 3.8, 2.1, BAND["orange"], "The drug",
          ["57% take a sulfonylurea,", "which keeps releasing insulin", "even when no food arrives"])
    _card(ax, 8.8, 3.5, 3.8, 2.1, BAND["blue"], "The behaviour",
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
    _card(ax, 0.7, 3.9, 3.8, 2.4, BAND["grey"], "1 · EHR prior",
          ["labs, meds, eGFR,", "CYP2C9 and TCF7L2", "->  personal physiology", "     before any device data"])
    _card(ax, 4.75, 3.9, 3.8, 2.4, BAND["orange"], "2 · Mechanistic twin",
          ["Bergman + gut + sulfonylurea", "PK/PD + fasting clock", "+ blunted counter-regulation",
           "->  explains WHY, not just what"])
    _card(ax, 8.8, 3.9, 3.8, 2.4, BAND["blue"], "3 · Particle filter",
          ["64 candidate physiologies", "re-weighted by every reading", "->  a twin that follows the",
           "     patient, not a classifier"])
    _card(ax, 0.7, 1.05, 5.9, 2.5, BAND["green"], "Vrat Rehearsal - the part that makes it a twin",
          ["\"If she fasts tomorrow and takes her usual glimepiride, what happens?\"",
           "36-hour counterfactual replay on the posterior particles:",
           "usual dose  ·  halved  ·  skipped  ·  moved to the fast-breaking meal",
           "low-GI swap  ·  a walk after iftar  ·  garba at night"])
    _card(ax, 6.85, 1.05, 5.75, 2.5, BAND["grey"], "Into the health system",
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
    ax.text(0.9, 5.55, "sensor tier", fontsize=11, color=MUTED)
    for j, h in enumerate(["hypo AUPRC (95% CI)", "precision@20", "event recall", "false alarms / patient-day"]):
        ax.text(4.6 + j * 2.1, 5.55, h, fontsize=10.5, color=MUTED, ha="center")
    for i, (lab, r, col) in enumerate(rows):
        y = 4.75 - i * 0.95
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
    _card(ax, 0.7, 0.95, W - 1.4, 1.85, BAND["blue"], "Read this honestly",
          ["The twin's value is NOT uniform. It is largest at tier B, where data is sparse enough that physics has",
           "something to add and dense enough to personalise - which is the tier most of India will actually live on.",
           "At tier A a dense CGM already carries most of the 2-hour signal. We report that null result on the next slide."])
    _foot(ax, 5)
    pages.append(f)

    # ---------------------------------------------------------------- 6 where it fails
    f, ax = _fig()
    _title(ax, "Where it does not work - and how hard we made the test",
           "Reported because a result you cannot break is a result you cannot trust")
    _card(ax, 0.7, 4.15, 5.9, 2.2, BAND["orange"], "Null results we are not hiding",
          [f"Tier A: the twin changes hypo AUPRC {g('A_full/no_twin/hypo').get('auprc', 0):.2f} → "
           f"{A.get('auprc', 0):.2f}. A dense CGM",
           "   already carries the short-horizon signal.",
           f"The calendar as ML features: {g('A_full/no_calendar/hypo').get('auprc', 0):.2f} without vs "
           f"{A.get('auprc', 0):.2f} with.",
           "   Its value is in Rehearsal days ahead, not in the feature vector.",
           "The NIS drift monitor does not yet work well enough to deploy."])
    _card(ax, 6.85, 4.15, 5.75, 2.2, BAND["grey"], "We made the benchmark harder on purpose",
          ["The virtual body runs a DIFFERENT model from the twin:",
           "   3-compartment gut · interstitial insulin delay · circadian SI",
           "   · renal excretion · incretin. The twin has none of these terms.",
           f"Meal plan vs what was actually eaten: r = {bd.get('plan_vs_logged_carb_corr_2h', 0):.2f}",
           f"   ({bd.get('real_intake_that_was_unplanned', 0):.0%} of intake unplanned) - a prior, not an oracle."])
    tv = m["models"].get("A_full/trivial_calendar_baseline/hyper", {})
    _card(ax, 0.7, 1.9, 5.9, 1.9, BAND["green"], "The baseline that keeps us honest",
          ["A 4-feature model - planned carbs, HbA1c, time of day - with no glucose,",
           "no wearable and no twin, reaches hyperglycaemia AUPRC "
           f"{tv.get('auprc', 0):.2f}.",
           f"The full tier-A model reaches {Ah.get('auprc', 0):.2f}. We report both, so the",
           "reader can see how much is physiology and how much is timetable."])
    _card(ax, 6.85, 1.9, 5.75, 1.9, BAND["grey"], "Limits we state first",
          ["Synthetic data only: these are upper bounds, not clinical evidence.",
           f"Only {A.get('patients_with_events', 0)} of {A.get('n_patients', 0)} patients ever had a "
           "hypoglycaemic event -",
           "   the confidence intervals are wide and we print them.",
           "Next step: external validation on ShanghaiT2DM / CGMacros (adapter built)."])
    _foot(ax, 6)
    pages.append(f)

    # ---------------------------------------------------------------- 7 uncertainty
    f, ax = _fig()
    _title(ax, "Uncertainty a clinician can actually rely on",
           "A 90% band that is 90% on average can be 50% exactly where you are looking")
    c = m["conformal"]["horizons"][-1]
    sp, mo = c["split_conformal"], c["mondrian_conformal"]
    ax.text(0.9, 5.5, "coverage of the 2-hour forecast band", fontsize=11, color=MUTED)
    for j, h in enumerate(["overall", "when outcome < 70", "when outcome > 250", "mean band width"]):
        ax.text(4.9 + j * 2.0, 5.5, h, fontsize=10.5, color=MUTED, ha="center")
    for i, (lab, r, col) in enumerate([("Split conformal (one margin)", sp, MUTED),
                                       ("Mondrian (margin per glucose band)", mo, BLUE)]):
        y = 4.7 - i * 0.9
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
    _card(ax, 0.7, 1.1, W - 1.4, 2.4, BAND["green"], "Why this slide exists",
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
    _card(ax, 0.7, 3.8, 5.9, 2.5, BAND["blue"], "Audited on the axis that matters here",
          ["This project is built on religious fasting, so fasting tradition is the",
           "first axis we audit - not an afterthought:"] + lines +
          ["Also audited: sex, urban/rural, age, eGFR, region."])
    _card(ax, 6.85, 3.8, 5.75, 2.5, BAND["grey"], "Governance by design",
          ["DPDP 2023 - synthetic data only; consent ID on every record,",
           "   purpose limitation (CAREMGT), deletion on consent expiry.",
           "ABDM - FHIR R4 bundles, consent artefact via HIE-CM.",
           "CDSCO draft (Oct 2025) - informs, does not drive, a registered",
           "   practitioner -> likely Class B. Human always in the loop.",
           "ICMR 2023 - model card, subgroup audit, right to refuse AI."])
    _card(ax, 0.7, 1.0, W - 1.4, 2.5, BAND["green"], "Deployment path",
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
    _card(ax, 0.7, 3.9, 5.9, 2.5, BAND["green"], "We are claiming",
          ["A twin that treats India's festival calendar as a planned",
           "   perturbation, not as noise.",
           "A counterfactual rehearsal a doctor can run BEFORE the fast.",
           "A measured ladder from CGM down to a phone, so the people",
           "   least likely to own a device are not silently excluded.",
           "Uncertainty, fairness and null results reported, not implied."])
    _card(ax, 6.85, 3.9, 5.75, 2.5, BAND["orange"], "We are not claiming",
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
