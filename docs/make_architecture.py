"""Draws the architecture diagram -> docs/architecture.pdf and docs/architecture.png

    python docs/make_architecture.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
INK, INK2, LINE = "#141413", "#52514e", "#c3c2b7"
FILL = {"data": "#e8f1fb", "interop": "#f3eee6", "twin": "#fbe9dd", "ml": "#e6f5ee", "out": "#efe9f7", "gov": "#f6f5f1"}


def box(ax, x, y, w, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=FILL[kind], ec=LINE, lw=1))
    ax.text(x + 0.008, y + h - 0.018, title, fontsize=9.5, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.008, y + h - 0.045, "\n".join(lines), fontsize=7.6, color=INK2, va="top", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.1))


def main():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.02, 0.965, "UPVAAS-Twin - festival- & fasting-aware metabolic digital twin for Indians with type 2 diabetes",
            fontsize=14, fontweight="bold", color=INK)
    ax.text(0.02, 0.935, "Uncertainty-aware, Physiology-informed Virtual Avatar for Adverse-event Surveillance  ·  "
            "static EHR sets the prior, live wearables update the posterior, the festival calendar tells the twin what is coming",
            fontsize=9, color=INK2)
    cols = [0.02, 0.215, 0.41, 0.605, 0.80]
    W = 0.18
    ax.text(cols[0], 0.89, "1  DATA STREAMS", fontsize=8, color=INK2, fontweight="bold")
    box(ax, cols[0], 0.66, W, 0.21, "Static / historical (EHR)", [
        "Synthea-India-lite generator", "demographics, state, urban/rural, ABHA", "HbA1c, FPG, lipids, TyG, eGFR, ALT, Lp(a)",
        "diagnoses: T2D, HTN, neuropathy", "meds: metformin, SU dose, basal insulin", "genetics: CYP2C9, TCF7L2, PRS",
        "vrat questionnaire: tradition, style,", "SU habit when fasting"], "data")
    box(ax, cols[0], 0.40, W, 0.23, "Dynamic / real-time (IoT)", [
        "CGM 5-min (Libre/Dexcom-like, lag+noise)", "or 2 fingersticks/day (frugal tier)",
        "smartwatch: HR, HRV (RMSSD), steps", "sleep stages & efficiency", "meal photo log (carbs, GI), smart pillbox",
        "phone step counter (minimal tier)"], "data")
    box(ax, cols[0], 0.20, W, 0.17, "Context (India-specific)", [
        "Festival & fasting calendar engine", "Navratri, Ekadashi, Karva Chauth, Ramadan,", "Diwali/Eid feasts, garba nights",
        "CPCB-style PM2.5 (+0.40 mg/dL per 10 ug/m3)", "Indian food GI library"], "data")

    ax.text(cols[1], 0.89, "2  CONSENT & INTEROP", fontsize=8, color=INK2, fontweight="bold")
    box(ax, cols[1], 0.56, W, 0.31, "ABDM-ready ingestion", [
        "Twin registers as Health Information User", "consent artefact via HIE-CM (purpose CAREMGT)",
        "FHIR R4 bundles (NRCeS IG): Patient,", "Observation, MedicationStatement, Consent",
        "DPDP Act 2023 / Rules 2025:", "notice + consent, purpose limitation,", "erase on consent expiry, breach < 72 h",
        "Research/training: synthetic / anonymised", "data only (Rule 16 research carve-out)"], "interop")
    box(ax, cols[1], 0.20, W, 0.33, "Feature fusion", [
        "per patient, every 30 min:", "glucose trend (CGM or fingerstick)", "wearables (HR dev, HRV ratio, steps, sleep)",
        "meal / SU on board", "calendar: fast type, minutes into fast,", "minutes to fast-break, planned carbs,",
        "planned garba", "EHR + pharmacogenomics", "+ 22 twin features (from layer 3)"], "interop")

    ax.text(cols[2], 0.89, "3  DIGITAL TWIN CORE", fontsize=8, color=INK2, fontweight="bold")
    box(ax, cols[2], 0.73, W, 0.14, "EHR -> physiology prior", [
        "ridge maps labs+genes -> SI, beta-gain,", "hepatic set-point, SU effect, SU half-life",
        "(CYP2C9 & eGFR lengthen SU t1/2 prior)"], "twin")
    box(ax, cols[2], 0.45, W, 0.25, "SU-Bergman-Gut model", [
        "Bergman minimal model (G, X)", "+ 2-compartment gut, GI-dependent k_abs",
        "+ glucose-stimulated insulin secretion", "+ sulfonylurea PK/PD (secretes at low G)",
        "+ fasting clock: hepatic output falls >12 h", "+ dawn, sleep->SI, activity, PM2.5,",
        "  counter-regulation blunted by neuropathy"], "twin")
    box(ax, cols[2], 0.20, W, 0.22, "Particle-filter data assimilation", [
        "64 candidate physiologies per patient", "predict -> update on each glucose reading",
        "unannounced-meal particles", "2-h calendar-conditioned forecast ensemble",
        "self-check: normalised innovation (NIS)", "-> drift flag -> recalibration"], "twin")

    ax.text(cols[3], 0.89, "4  INTELLIGENCE", fontsize=8, color=INK2, fontweight="bold")
    box(ax, cols[3], 0.62, W, 0.25, "Hybrid risk model", [
        "gradient boosting on fused features", "targets: hypo <70 & hyper >250 mg/dL",
        "sustained >=15 min within next 120 min", "isotonic calibration", "alarm threshold for 80% recall",
        "ablations: no-twin, no-calendar, rule"], "ml")
    box(ax, cols[3], 0.40, W, 0.19, "Uncertainty & explanation", [
        "split-conformal widening of twin bands", "(90% coverage guarantee on held-out pts)",
        "counterfactual driver attribution", "(swap a feature group to neutral -> delta p)"], "ml")
    box(ax, cols[3], 0.20, W, 0.17, "Vrat Rehearsal (what-if)", [
        "36-h replay on posterior particles", "fast type x SU timing x low-GI swap",
        "x walk x garba x basal insulin", "-> P(hypo), P(severe), P(hyper), nadir"], "ml")

    ax.text(cols[4], 0.89, "5  WHO USES IT", fontsize=8, color=INK2, fontweight="bold")
    box(ax, cols[4], 0.62, W, 0.25, "Clinician console", [
        "roster ranked by 2-h risk", "CGM + twin forecast + conformal band",
        "risk cards with drivers", "virtual-patient physiology vs EHR prior",
        "Vrat Rehearsal sandbox", "twin fidelity chart, FHIR export"], "out")
    box(ax, cols[4], 0.42, W, 0.17, "ASHA 'Vrat Saathi' list", [
        "pre-festival rehearsal for every faster", "ranked by predicted hypo risk",
        "language, ward, action script", "-> eSanjeevani teleconsult"], "out")
    box(ax, cols[4], 0.20, W, 0.19, "Back into the health system", [
        "FHIR RiskAssessment to HIP / EMR", "NP-NCD / Ayushman Arogya Mandir",
        "model card: degradation by sensor tier,", "fairness by sex/region/age/eGFR"], "out")

    ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.11, boxstyle="round,pad=0.006,rounding_size=0.012", fc=FILL["gov"], ec=LINE))
    ax.text(0.03, 0.125, "GOVERNANCE RAIL", fontsize=8, color=INK2, fontweight="bold")
    ax.text(0.03, 0.10, "CDSCO draft medical-device-software guidance (Oct 2025): informs, does not drive -> likely Class B  ·  "
            "NIS drift monitor feeds a pre-declared Algorithm Change Protocol  ·  ICMR 2023 AI ethics: human in the loop, "
            "Indian-representative data,\nnon-discrimination audit, right to refuse AI  ·  DPDP: consent-tagged data, purpose limitation, "
            "deletion on expiry  ·  Sensor-tier degradation curve published so low-resource users are not silently under-served",
            fontsize=8, color=INK2, va="top", linespacing=1.5)

    for i in range(4):
        for y in (0.76, 0.52, 0.30):
            arrow(ax, cols[i] + W + 0.002, y, cols[i + 1] - 0.003, y)
    fig.savefig(OUT / "architecture.pdf"); fig.savefig(OUT / "architecture.png", dpi=150)
    print("wrote docs/architecture.pdf and docs/architecture.png")


if __name__ == "__main__":
    main()
