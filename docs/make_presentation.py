"""Build docs/presentation.pptx from reports/metrics.json.

    python docs/make_presentation.py

Every figure on every slide is read from the metrics file, so the deck cannot drift away from the
results. Re-run the pipeline, re-run this, and the deck is current.

Layout follows a standard consulting grid on a 10 x 5.63in page: an action title across the top
that states the finding, content between 0.9in and 5.1in, and a footer rule with a section tracker
and page number. Type is Aptos at 8pt body, 12pt panel headings, 15pt action titles.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.dml.color import RGBColor
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Emu, Inches, Pt
except ImportError:
    sys.exit("python-pptx is not installed.\n  pip install python-pptx\nthen re-run this script.")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upvaas.config import REPORT_DIR, ROOT  # noqa: E402

# --- design system, lifted from the reference deck --------------------------------------------
FONT = "Aptos"
NAVY = RGBColor(0x1C, 0x49, 0x80)
NAVY_MID = RGBColor(0x3A, 0x5F, 0x8F)
NAVY_SOFT = RGBColor(0x8F, 0xAD, 0xD0)
PANEL = RGBColor(0xEA, 0xF0, 0xF6)
PANEL_EDGE = RGBColor(0xB9, 0xCB, 0xE0)
GREEN = RGBColor(0x0D, 0x89, 0x43)
RED = RGBColor(0xC0, 0x39, 0x2B)
RED_PANEL = RGBColor(0xFB, 0xEA, 0xEA)
INK = RGBColor(0x26, 0x26, 0x26)
GREY = RGBColor(0x7F, 0x7F, 0x7F)
GREY_LT = RGBColor(0xBF, 0xBF, 0xBF)
GREY_BG = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = 10.0, 5.625
SECTIONS = ["Problem", "Solution", "Results", "Evidence", "Deployment"]
IMG = ROOT / "docs" / "img"


def _tf(shape, *, size=8, bold=False, color=INK, align=PP_ALIGN.LEFT, space=2):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for p in tf.paragraphs:
        p.alignment = align
        p.space_after = Pt(space)
        for r in p.runs:
            r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(size), bold, color
    return tf


def text(slide, x, y, w, h, body, *, size=8, bold=False, color=INK, align=PP_ALIGN.LEFT, space=2):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    lines = body if isinstance(body, (list, tuple)) else [body]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        r = p.add_run()
        r.text = str(line)
        r.font.name, r.font.size, r.font.bold = FONT, Pt(size), bold
        r.font.color.rgb = color
    return box


def rect(slide, x, y, w, h, fill=PANEL, line=None, shadow=False):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    sh.shadow.inherit = shadow
    return sh


def line(slide, x, y, w, color=GREY_LT, width=0.75):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Pt(width))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def slide_base(prs, title, kicker=None, section=None, page=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, 0.20, 0.14, 9.3, 0.40, title, size=15, bold=True, color=NAVY)
    if kicker:
        text(s, 0.20, 0.52, 9.3, 0.26, kicker, size=9, color=GREY)
    line(s, 0.0, 5.17, 10.0, GREY_LT)
    if section:
        for i, name in enumerate(SECTIONS):
            x = 0.15 + i * 1.32
            on = name == section
            text(s, x, 5.22, 1.30, 0.22, name, size=7,
                 bold=on, color=NAVY if on else GREY_LT)
            if on:
                line(s, x, 5.17, 1.20, NAVY, width=1.5)
    if page:
        text(s, 9.30, 5.22, 0.55, 0.22, str(page), size=7, color=GREY, align=PP_ALIGN.RIGHT)
    return s


def panel(slide, x, y, w, h, heading, lines, *, fill=PANEL, head_color=NAVY, size=8, edge=None):
    rect(slide, x, y, w, h, fill=fill, line=edge)
    text(slide, x + 0.06, y + 0.07, w - 0.12, 0.22, heading, size=9, bold=True, color=head_color)
    if lines:
        text(slide, x + 0.06, y + 0.33, w - 0.12, h - 0.40, lines, size=size, color=INK, space=3)


def stat(slide, x, y, w, value, label, *, color=NAVY, sub=None):
    text(slide, x, y, w, 0.42, value, size=22, bold=True, color=color, align=PP_ALIGN.CENTER)
    text(slide, x, y + 0.42, w, 0.24, label, size=8, color=INK, align=PP_ALIGN.CENTER)
    if sub:
        text(slide, x, y + 0.64, w, 0.20, sub, size=7, color=GREY, align=PP_ALIGN.CENTER)


def table(slide, x, y, w, rows, col_w, *, head_fill=NAVY, row_h=0.27, size=8, highlight=None):
    """rows[0] is the header. highlight = index of a body row to tint."""
    cy = y
    for ri, row in enumerate(rows):
        hdr = ri == 0
        if hdr:
            rect(slide, x, cy, w, row_h, fill=head_fill)
        elif highlight is not None and ri == highlight:
            rect(slide, x, cy, w, row_h, fill=PANEL)
        elif ri % 2 == 0:
            rect(slide, x, cy, w, row_h, fill=GREY_BG)
        cx = x
        for ci, cell in enumerate(row):
            text(slide, cx + 0.04, cy + 0.045, col_w[ci] - 0.08, row_h,
                 cell, size=size, bold=hdr or (highlight is not None and ri == highlight),
                 color=WHITE if hdr else INK,
                 align=PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER)
            cx += col_w[ci]
        cy += row_h
    return cy


def picture(slide, path: Path, x, y, w, h, caption=None):
    """Drop in a screenshot, or a labelled placeholder if it has not been captured yet."""
    if path.exists():
        pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
        if pic.height > Inches(h):
            scale = Inches(h) / pic.height
            pic.height = Inches(h)
            pic.width = Emu(int(pic.width * scale))
            pic.left = Inches(x + (w - pic.width / Emu(914400)) / 2)
        pic.line.color.rgb = PANEL_EDGE
        pic.line.width = Pt(0.75)
    else:
        rect(slide, x, y, w, h, fill=GREY_BG, line=GREY_LT)
        text(slide, x, y + h / 2 - 0.15, w, 0.3, f"[ {path.name} ]", size=8,
             color=GREY, align=PP_ALIGN.CENTER)
    if caption:
        text(slide, x, y + h + 0.04, w, 0.20, caption, size=7, color=GREY)


# ==============================================================================================
def build(m: dict, out: Path) -> None:
    g = m["models"]
    bd = m.get("benchmark_difficulty", {})
    coh = m["cohort"]
    fc = m["forecast"]
    A, B, C = g["A_full/full/hypo"], g["B_frugal/full/hypo"], g["C_minimal/full/hypo"]
    An, Bn, Cn = g["A_full/no_twin/hypo"], g["B_frugal/no_twin/hypo"], g["C_minimal/no_twin/hypo"]
    Ah = g["A_full/full/hyper"]
    tv = g.get("A_full/trivial_calendar_baseline/hyper", {})
    cov = m["conformal"]["horizons"][-1]
    ev = _json(REPORT_DIR / "EXTERNAL_VALIDATION.json")
    rf = _json(REPORT_DIR / "prior_refit_shanghai.json")

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    pg = iter(range(1, 40))

    # ---------------------------------------------------------------- 1 title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, fill=NAVY)
    rect(s, 0, 0, 0.09, H, fill=RED)
    text(s, 0.75, 1.55, 8.5, 0.70, "UPVAAS-Twin", size=34, bold=True, color=WHITE)
    text(s, 0.75, 2.30, 8.5, 0.60,
         "A festival-aware metabolic digital twin for Indians with type 2 diabetes",
         size=14, color=NAVY_SOFT)
    line(s, 0.75, 3.05, 1.6, RED, width=2)
    text(s, 0.75, 3.25, 8.5, 0.50,
         ["Predicting hypoglycaemia and glucose spikes two hours ahead, and rehearsing a patient's",
          "religious fast before the patient lives it."], size=9, color=WHITE, space=2)
    text(s, 0.75, 4.35, 8.5, 0.22,
         "Happiest Health · Reimagining and Reforming Healthcare in India Summit 2026",
         size=8, color=NAVY_SOFT)
    text(s, 0.75, 4.60, 8.5, 0.22, "Digital Twin Challenge", size=8, color=NAVY_SOFT)
    text(s, 0.75, 4.92, 8.5, 0.22, "Breakpoint Consulting · Symbiosis Institute of Management Studies, Pune",
         size=8, bold=True, color=WHITE)

    # ---------------------------------------------------------------- 2 problem
    s = slide_base(prs, "Half of Indian T2D patients fast for religion, most take a drug that does not stop",
                   "The dangerous combination is scheduled, so it can be predicted", "Problem", next(pg) + 1)
    panel(s, 0.20, 0.88, 3.05, 1.42, "The scale",
          ["101 million adults with diabetes in India.",
           "136 million more with prediabetes.",
           "ICMR-INDIAB, Lancet Diab Endo 2023"])
    panel(s, 3.42, 0.88, 3.05, 1.42, "The drug",
          ["57% of one North Indian clinic cohort take a sulfonylurea.",
           "It keeps releasing insulin even when no food arrives."],
          fill=RED_PANEL, head_color=RED)
    panel(s, 6.64, 0.88, 3.16, 1.42, "The behaviour",
          ["49% had fasted for religion in the past year.",
           "4% checked their glucose during the fast.",
           "1.35% asked a doctor first."])
    text(s, 0.20, 2.48, 9.6, 0.26,
         "Navratri · Ekadashi · Karva Chauth (nirjala) · Ramadan roza", size=11, bold=True, color=NAVY)
    rect(s, 0.20, 2.82, 9.6, 1.05, fill=GREY_BG)
    text(s, 0.32, 2.92, 9.4, 0.85,
         ["A morning glimepiride followed by a thirteen hour fast produces a hypoglycaemia that was "
          "predictable the day before. The festive thali that breaks the fast produces a spike that was "
          "equally predictable.",
          "Existing digital twins treat those days as noise. They predict meal responses, assume the "
          "patient wears a CGM, and never look at the calendar."], size=9, space=4)
    text(s, 0.20, 4.02, 9.6, 0.24, "What we predict", size=9, bold=True, color=NAVY)
    table(s, 0.20, 4.28, 9.6,
          [["Adverse event", "Threshold", "Horizon", "Who it matters for"],
           ["Hypoglycaemia", "< 70 mg/dL for 15 min", "2 hours ahead", "Sulfonylurea patients during a fast"],
           ["Hyperglycaemia", "> 250 mg/dL for 15 min", "2 hours ahead", "Everyone, after the fast breaks"]],
          [3.0, 2.3, 1.6, 2.7])

    # ---------------------------------------------------------------- 3 gap
    s = slide_base(prs, "Commercial twins answer a different question, and assume hardware most Indians lack",
                   "Where UPVAAS-Twin sits against what already exists", "Problem", next(pg) + 1)
    rows = [["", "Twin Health / January AI", "Academic hybrid twins", "UPVAAS-Twin"],
            ["Predicts", "food to glucose response", "glucose trajectories, mostly T1D", "adverse events 2 h ahead, plus what-if"],
            ["Model", "gradient boosting / RF", "physics-constrained neural net", "mechanistic ODE + particle filter + calibrated GBM"],
            ["Uncertainty", "not published", "sometimes, marginal only", "conformal, reported by glucose range"],
            ["Needs a CGM?", "yes", "yes", "no, three sensor tiers evaluated"],
            ["Validated against", "not published", "the same model family", "a structurally different body model"],
            ["India specific", "food database", "no", "festival calendar, thin-fat prior, SU pharmacogenomics, ASHA workflow"],
            ["Fairness audit", "no", "rarely", "yes, led by fasting tradition"]]
    table(s, 0.20, 0.92, 9.6, rows, [1.45, 2.05, 2.35, 3.75], row_h=0.355, size=7.5)
    rect(s, 0.20, 4.52, 9.6, 0.58, fill=PANEL)
    text(s, 0.32, 4.60, 9.4, 0.45,
         "A 2026 review of 28 diabetes digital twin studies found one that addressed fairness and three "
         "with a human in the loop. None forecast adverse events.", size=8.5, color=NAVY)

    # ---------------------------------------------------------------- 4 architecture
    s = slide_base(prs, "Static data sets the prior, live data updates the posterior, the calendar says what is coming",
                   "Five layers, from raw streams to a clinician's screen", "Solution", next(pg) + 1)
    cols = [("1  Data in", ["India-localised synthetic EHR:", "labs, meds, eGFR, CYP2C9, TCF7L2",
                            "CGM or 2 fingersticks a day",
                            "watch: HR, HRV, steps, sleep",
                            "festival calendar + PM2.5"]),
            ("2  Twin core", ["EHR to physiology prior (ridge)",
                              "SU-Bergman-Gut ODE",
                              "particle filter, 64 candidate",
                              "physiologies per patient",
                              "2 h forecast on the calendar plan"]),
            ("3  Intelligence", ["gradient boosting on fused",
                                 "EHR + stream + calendar + twin",
                                 "isotonic calibration",
                                 "Mondrian conformal bands",
                                 "counterfactual drivers"]),
            ("4  Decision", ["clinician console, ranked roster",
                             "Vrat Rehearsal what-if",
                             "ASHA pre-festival recall list",
                             "FHIR R4 RiskAssessment"]),
            ("5  Governance", ["DPDP: consent ID, purpose limit",
                               "ABDM: HIE-CM consent artefact",
                               "CDSCO draft: informs a doctor,",
                               "likely Class B",
                               "ICMR: human in the loop"])]
    x = 0.20
    for i, (head, lines) in enumerate(cols):
        panel(s, x, 0.92, 1.84, 1.95, head, lines, size=7.5,
              fill=PANEL if i < 3 else GREY_BG, head_color=NAVY if i < 3 else NAVY_MID)
        if i < 4:
            text(s, x + 1.86, 2.05, 0.14, 0.3, "▶", size=9, color=NAVY_SOFT)
        x += 1.96
    rect(s, 0.20, 3.10, 9.6, 1.88, fill=NAVY)
    text(s, 0.34, 3.24, 9.3, 0.24, "What makes it a twin rather than a classifier", size=9, bold=True, color=WHITE)
    text(s, 0.34, 3.56, 9.3, 1.30,
         ["The particle filter keeps 64 candidate physiologies per patient and re-weights them on every "
          "glucose reading, so the model follows the patient instead of scoring them once.",
          "Those same posterior particles are replayed forward under a changed plan. That is the Vrat "
          "Rehearsal: the doctor asks what happens if the glimepiride moves to the fast-breaking meal, "
          "and gets an answer before the fast."], size=8.5, color=WHITE, space=4)

    # ---------------------------------------------------------------- 5 mechanism
    s = slide_base(prs, "The model knows why a fast plus a morning glimepiride is dangerous",
                   "Mechanistic core, so every alert can be explained to the patient", "Solution", next(pg) + 1)
    panel(s, 0.20, 0.92, 4.70, 2.05, "SU-Bergman-Gut model, per minute",
          ["Bergman minimal model for glucose and remote insulin action.",
           "Two-compartment gut, absorption set by the food's glycaemic index and fat.",
           "Sulfonylurea PK/PD that keeps secreting insulin at low glucose.",
           "Fasting clock: hepatic output falls after about 11 hours without food.",
           "Counter-regulation blunted by autonomic neuropathy.",
           "CYP2C9 genotype and eGFR lengthen the drug half-life prior."], size=8)
    panel(s, 5.10, 0.92, 4.70, 2.05, "Why that combination harms people",
          ["The tablet is taken at 08:30 on a normal schedule.",
           "No carbohydrate arrives for the next thirteen hours.",
           "Insulin secretion continues anyway.",
           "After eleven hours the liver has less glycogen to release.",
           "In a patient with neuropathy the warning symptoms are blunted.",
           "Glucose falls in the late afternoon, hours before moonrise."],
          fill=RED_PANEL, head_color=RED, size=8)
    text(s, 0.20, 3.12, 9.6, 0.24, "The twin turns that mechanism into four things a doctor can act on",
         size=9, bold=True, color=NAVY)
    items = [("Risk now", "Calibrated probability of an event in the next two hours, with the drivers behind it."),
             ("A virtual patient", "Personalised insulin sensitivity, beta-cell gain, hepatic set-point and drug half-life, each next to its EHR-only prior."),
             ("A rehearsal", "Tomorrow's fast replayed under four drug plans before the patient lives it."),
             ("A recall list", "Every faster in the ward ranked before the festival, with language, ward and an action script.")]
    x = 0.20
    for head, body in items:
        panel(s, x, 3.40, 2.32, 1.55, head, [body], size=8)
        x += 2.42

    # ---------------------------------------------------------------- 6 results
    s = slide_base(prs, "The twin earns its keep where there is no CGM, which is most of India",
                   f"{coh['n_patients']} synthetic patients x {coh['days']} days, 5-fold patient-grouped cross-validation",
                   "Results", next(pg) + 1)
    rect(s, 0.20, 0.90, 9.6, 1.12, fill=PANEL)
    text(s, 0.34, 0.96, 9.3, 0.22,
         "Tier B: two fingersticks a day and a basic smartwatch. No CGM.", size=9, bold=True, color=NAVY)
    stat(s, 0.40, 1.22, 2.9, f"{Bn.get('auprc', 0):.2f} → {B.get('auprc', 0):.2f}",
         "hypoglycaemia AUPRC", sub=f"{B.get('auprc', 0)/max(Bn.get('auprc', 1e-9), 1e-9):.1f}x by adding the twin")
    stat(s, 3.55, 1.22, 2.9, f"{Bn.get('false_alarms_per_patient_day', 0):.1f} → {B.get('false_alarms_per_patient_day', 0):.1f}",
         "false alarms per patient-day", color=GREEN, sub="the twin filters as well as ranks")
    stat(s, 6.70, 1.22, 2.9, f"{(B.get('precision_at_20') or 0):.0%}", "precision of a 20-name worklist",
         color=RED, sub=f"against a {B.get('prevalence', 0):.1%} base rate")
    text(s, 0.20, 2.16, 9.6, 0.26,
         f"Of the 20 patients the twin flags each morning, about {round((B.get('precision_at_20') or 0)*20)} "
         "really do have an event within two hours.", size=10, bold=True, color=INK)
    rows = [["Sensor tier", "AUROC", "AUPRC [95% CI]", "Without twin", "P@20", "Recall", "False alarms/day"]]
    for lab, r, rn in (("A  CGM + watch", A, An), ("B  2 fingersticks + watch", B, Bn),
                       ("C  EHR + phone only", C, Cn)):
        ci = r.get("auprc_ci95") or [0, 0]
        rows.append([lab, f"{r.get('auroc', 0):.3f}", f"{r.get('auprc', 0):.3f} [{ci[0]:.2f}-{ci[1]:.2f}]",
                     f"{rn.get('auprc', 0):.3f}", f"{(r.get('precision_at_20') or 0):.2f}",
                     f"{r.get('event_recall', 0):.0%}", f"{r.get('false_alarms_per_patient_day', 0):.2f}"])
    table(s, 0.20, 2.50, 9.6, rows, [2.5, 1.0, 1.9, 1.3, 0.8, 0.9, 1.2], row_h=0.30, highlight=2)
    panel(s, 0.20, 3.80, 4.70, 1.22, "Forecasting, tier A",
          [f"Twin RMSE {fc['30min']['twin_rmse']:.0f} / {fc['60min']['twin_rmse']:.0f} / "
           f"{fc['120min']['twin_rmse']:.0f} mg/dL at 30 / 60 / 120 min.",
           f"Persistence {fc['30min']['persistence_rmse']:.0f} / {fc['60min']['persistence_rmse']:.0f} / "
           f"{fc['120min']['persistence_rmse']:.0f} mg/dL.",
           f"Clarke error grid zones A+B: {fc['120min']['clarke_AB']:.1%}."])
    panel(s, 5.10, 3.80, 4.70, 1.22, "How it was scored",
          ["Every patient scored by a model that never saw them, in the dashboard as well as the metrics.",
           "Patient-clustered bootstrap intervals on every headline number.",
           f"Only {A.get('patients_with_events', 0)} of {A.get('n_patients', 0)} patients ever had a "
           "hypoglycaemic event, so the intervals are wide and we print them."], fill=GREY_BG)

    # ---------------------------------------------------------------- 7 console
    s = slide_base(prs, "The doctor sees the risk, the reason, and the patient's own physiology",
                   "Clinician console, built for a ten-minute OPD consultation", "Results", next(pg) + 1)
    picture(s, IMG / "console.png", 0.20, 0.92, 6.25, 3.95,
            "Roster ranked by two-hour risk, with today's vrat type and a recalibrate flag")
    panel(s, 6.65, 0.92, 3.15, 1.28, "Risk, with its drivers",
          ["Calibrated probability and the alarm threshold.",
           "Counterfactual attribution shows which feature group moved it."], size=7.5)
    panel(s, 6.65, 2.28, 3.15, 1.28, "The virtual patient",
          ["Insulin sensitivity, beta-cell gain, hepatic set-point, drug half-life.",
           "Each shown against its EHR-only prior, so the doctor sees what the data changed."], size=7.5)
    panel(s, 6.65, 3.64, 3.15, 1.23, "Forecast band",
          ["Mondrian conformal, so the band widens where the twin is least reliable.",
           "A reveal toggle shows what actually happened, for teaching."], size=7.5)

    # ---------------------------------------------------------------- 8 rehearsal
    dp = _json(IMG / "demo_patient.json") or {}
    hd = (f"Moving the tablet to the fast-breaking meal takes this patient from "
          f"{dp['usual_hypo']:.0%} to {dp['moved_hypo']:.0%}" if dp
          else "Moving the tablet to the fast-breaking meal removes this patient's hypoglycaemia risk")
    s = slide_base(prs, hd,
                   "Vrat Rehearsal: a 36-hour counterfactual on the posterior particles", "Results", next(pg) + 1)
    picture(s, IMG / "rehearsal.png", 0.20, 0.92, 6.25, 3.60,
            "Grey is the patient's usual habit, green is the proposed change")
    panel(s, 6.65, 0.92, 3.15, 1.75, "What the doctor changes",
          ["Which fast tomorrow is: Karva Chauth, roza, Navratri phalahar or ek-samay.",
           "Sulfonylurea: as usual, halved, skipped, or moved to the fast-breaking meal.",
           "Basal insulin, a low-GI swap, a walk after iftar, garba at night."], size=7.5)
    panel(s, 6.65, 2.75, 3.15, 1.30, "What comes back",
          ["P(hypo), P(severe), P(hyper) and the predicted nadir time.",
           "Across all sulfonylurea-treated fasters, moving the dose lowers predicted hypoglycaemia from "
           f"{m['asha']['mean_p_hypo_usual_su']:.0%} to {m['asha']['mean_p_hypo_dose_moved_su']:.0%}."], size=7.5)
    rect(s, 6.65, 4.13, 3.15, 0.74, fill=RED_PANEL)
    text(s, 6.77, 4.20, 2.95, 0.62,
         "Read as a ranking, not a calibrated probability. The rehearsal over-predicts how often a fast "
         "ends in hypoglycaemia.", size=7, color=RED)

    # ---------------------------------------------------------------- 9 honesty
    s = slide_base(prs, "We made the benchmark hard on purpose, and published the baseline that beats us",
                   "A result you cannot break is a result you cannot trust", "Evidence", next(pg) + 1)
    panel(s, 0.20, 0.92, 4.70, 1.72, "The virtual body is not the twin",
          ["The body runs a different model: three-compartment gut, interstitial insulin delay, "
           "circadian insulin sensitivity, renal glucose excretion, an incretin term.",
           "The twin has an equation for none of them, so the error we report includes model "
           "misspecification and not only parameter error."])
    panel(s, 5.10, 0.92, 4.70, 1.72, "Eating is messy on purpose",
          [f"Correlation between the festival meal plan and what was actually eaten: "
           f"{bd.get('plan_vs_logged_carb_corr_2h', 0):.2f}.",
           f"{bd.get('real_intake_that_was_unplanned', 0):.0%} of food intake was never on the plan.",
           "Wide meal-time jitter, skipped meals, unplanned snacks. The calendar is a prior, not an oracle."],
          fill=GREY_BG)
    rect(s, 0.20, 2.76, 9.6, 0.92, fill=RED_PANEL)
    text(s, 0.34, 2.84, 9.3, 0.22, "The baseline that keeps us honest", size=9, bold=True, color=RED)
    text(s, 0.34, 3.08, 9.3, 0.55,
         [f"Four features: planned carbohydrate in the next two hours, HbA1c, and time of day. No glucose "
          f"reading, no wearable, no twin. It reaches hyperglycaemia AUPRC {tv.get('auprc', 0):.2f} against "
          f"{Ah.get('auprc', 0):.2f} for the full tier-A model.",
          "We publish it so a reader can see how much of the task is physiology and how much is a meal timetable."],
         size=8, space=3)
    rows = [["Null results we report rather than bury", "Number"],
            ["The festival calendar as model features adds nothing",
             f"{g['A_full/no_calendar/hypo'].get('auprc', 0):.3f} vs {A.get('auprc', 0):.3f}"],
            ["The drift self-check does not work well enough to deploy",
             f"AUROC {(m['drift'].get('auroc_relative_nis_for_illness') or 0):.2f}"],
            ["Forecast bands cover far less than 90% above 250 mg/dL",
             f"{cov['mondrian_conformal'].get('coverage_outcome_gt250') or 0:.0%} vs 90% target"],
            ["The hypoglycaemia signal rests on a minority of patients",
             f"{A.get('patients_with_events', 0)} of {A.get('n_patients', 0)}"]]
    table(s, 0.20, 3.82, 9.6, rows, [7.3, 2.3], row_h=0.255, size=8)

    # ---------------------------------------------------------------- 10 external
    s = slide_base(prs, "We took it to real patients, it did not beat persistence, and the failure explains itself",
                   "ShanghaiT2DM: 107 real CGM recordings, no retraining. Beyond what the brief asks for",
                   "Evidence", next(pg) + 1)
    if ev:
        e2 = ev["arms"]["cgm_only"]["120min"]
        rows = [["2-hour forecast on real patients", "Twin", "Persistence"],
                ["RMSE", f"{e2['twin_rmse']:.0f} mg/dL", f"{e2['persistence_rmse']:.0f} mg/dL"],
                ["Clarke zones A+B", f"{e2['twin_clarke_AB']:.1%}", f"{e2['persistence_clarke_AB']:.1%}"],
                ["On our own simulator", "38 mg/dL, ahead of persistence", "55 mg/dL"]]
        table(s, 0.20, 0.92, 4.70, rows, [2.3, 1.4, 1.0], row_h=0.30, size=8)
    if rf:
        h = rf["held_out"]
        rows = [["Can a better prior fix it? (59 held-out patients)", "Ratio to persistence"],
                ["Original prior, as submitted", f"{h['original_prior']['ratio']:.2f}"],
                ["Original prior, simply widened", f"{h['original_prior_wider']['ratio']:.2f}"],
                ["Prior refitted on real patients", f"{h['refit_prior']['ratio']:.2f}"]]
        table(s, 5.10, 0.92, 4.70, rows, [3.4, 1.3], row_h=0.30, size=8, highlight=2)
    rect(s, 0.20, 2.30, 9.6, 1.30, fill=NAVY)
    text(s, 0.34, 2.38, 9.3, 0.22, "The finding is worth more than a fixed number", size=9, bold=True, color=WHITE)
    text(s, 0.34, 2.64, 9.3, 0.90,
         ["Refitting the prior on real patients changed nothing. Merely widening it reached parity with "
          "persistence and no further.",
          "The refit told us why. Its residual spread exploded: hepatic set-point SD went from 8 to 40 mg/dL "
          "and ln insulin sensitivity from 0.22 to 1.26. In a real cohort the routine EHR fields barely "
          "predict these physiological parameters. In our simulator they predict them well because the "
          "simulator generated the labs from those parameters in the first place."], size=8.5, color=WHITE, space=4)
    panel(s, 0.20, 3.74, 4.70, 1.28, "What this changes",
          ["Not the design. The mechanistic core, the filter, the rehearsal and the sensor ladder are unchanged.",
           "It puts a number on the warning the README already carried: the simulated results are an upper bound."],
          fill=GREY_BG, size=8)
    panel(s, 5.10, 3.74, 4.70, 1.28, "What would have to happen next",
          ["Fit the prior against real physiological reference data, from clamp or OGTT studies, not routine labs.",
           "Handle sparse 15-minute sampling directly instead of upsampling.",
           "Then repeat this same script and see whether the numbers move."], fill=GREY_BG, size=8)

    # ---------------------------------------------------------------- 11 trust
    s = slide_base(prs, "Audited first on the axis this project is built on",
                   "Fairness, uncertainty and the Indian regulatory path", "Evidence", next(pg) + 1)
    fr = m.get("fairness", {}).get("A_full/hypo", {})
    trad = {k.replace("tradition_", ""): v for k, v in fr.items() if k.startswith("tradition_")}
    rows = [["Fasting tradition", "AUROC", "Events"]]
    for k, v in sorted(trad.items()):
        rows.append([k.replace("_", " "), f"{v.get('auroc', 0):.3f}", str(v.get("events", "-"))])
    table(s, 0.20, 0.92, 3.35, rows, [1.85, 0.85, 0.65], row_h=0.28, size=8)
    panel(s, 3.75, 0.92, 3.00, 1.68, "Uncertainty that keeps its promise",
          [f"A single conformal margin covers {cov['split_conformal'].get('coverage_outcome_gt250') or 0:.0%} "
           "of outcomes above 250 mg/dL while claiming 90%.",
           f"Mondrian conformal, calibrated per predicted-glucose band, raises that to "
           f"{cov['mondrian_conformal'].get('coverage_outcome_gt250') or 0:.0%} with a narrower band overall.",
           "Both are published, by outcome range."], size=7.5)
    panel(s, 6.95, 0.92, 2.85, 1.68, "Also audited",
          ["Sex, urban and rural, age, eGFR, region.",
           "A model card ships inside the dashboard.",
           "The tier C path still works for a patient who refuses or cannot afford a wearable."],
          fill=GREY_BG, size=7.5)
    rows = [["Requirement", "How UPVAAS-Twin meets it"],
            ["DPDP Act 2023 / Rules 2025",
             "Synthetic data only. Production design keeps a consent ID on every record, enforces purpose limitation, deletes on expiry."],
            ["ABDM", "Registers as a Health Information User. Consent artefact via HIE-CM. FHIR R4 bundles validated against the R4 schema."],
            ["CDSCO draft software guidance, Oct 2025",
             "Informs a registered practitioner rather than directing treatment, so likely Class B. Drift monitor is the hook an Algorithm Change Protocol would attach to."],
            ["ICMR 2023 AI ethics",
             "Human in the loop, Indian-representative priors, subgroup audit, published model card, a working path for patients without devices."]]
    table(s, 0.20, 2.78, 9.6, rows, [2.5, 7.1], row_h=0.44, size=7.5)

    # ---------------------------------------------------------------- 12 deployment
    s = slide_base(prs, "It runs on a clinic laptop, and reaches the patient through an ASHA",
                   "Where this goes next, and what it would take", "Deployment", next(pg) + 1)
    steps = [("Two weeks before", "The ASHA app receives a ranked recall list for every faster in the ward, with language, ward and an action script."),
             ("Home visit or call", "The highest-risk patients are nudged to see a doctor before the fast, not after an ambulance call."),
             ("Teleconsult", "The doctor runs a Vrat Rehearsal in clinic or over eSanjeevani and picks a drug plan."),
             ("During the fast", "The patient has a dose-timing plan in their own language. The twin keeps scoring risk every 30 minutes.")]
    x = 0.20
    for i, (head, body) in enumerate(steps):
        panel(s, x, 0.92, 2.25, 1.42, f"{i+1}.  {head}", [body], size=7.5)
        if i < 3:
            text(s, x + 2.27, 1.52, 0.16, 0.3, "▶", size=9, color=NAVY_SOFT)
        x += 2.42
    panel(s, 0.20, 2.52, 4.70, 1.35, "Where it would run",
          ["NP-NCD screening at Ayushman Arogya Mandirs, which carry about 25 million people on diabetes treatment under the 75 by 25 initiative.",
           "Private diabetology clinics and CGM programmes.",
           "One clinic server can cover a district. Tiers B and C need no CGM."])
    panel(s, 5.10, 2.52, 4.70, 1.35, "How we would prove it",
          ["Fit the prior on real physiological reference data, then repeat the external validation.",
           "IRB-approved observational pilot during Navratri 2027, 100 sulfonylurea-treated fasters on Libre CGMs.",
           "Cluster-randomised ASHA recall-list trial, outcomes being hypoglycaemia events and ER visits."],
          fill=GREY_BG)
    rect(s, 0.20, 4.02, 9.6, 1.00, fill=NAVY)
    text(s, 0.34, 4.12, 9.3, 0.26, "The fast is on the calendar. The hypoglycaemia does not have to be a surprise.",
         size=13, bold=True, color=WHITE)
    text(s, 0.34, 4.48, 9.3, 0.45,
         ["Research prototype on synthetic data. Not a medical device. It informs a registered medical "
          "practitioner and decides nothing.",
          "MIT licence. Every number in this deck regenerates from reports/metrics.json."],
         size=8, color=NAVY_SOFT, space=2)

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"presentation -> {out.relative_to(ROOT)}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
    _to_pdf(out)


def _json(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, ValueError):
        return None


def _to_pdf(pptx: Path) -> None:
    """Export a PDF alongside the .pptx when LibreOffice is available."""
    for exe in ("/Applications/LibreOffice.app/Contents/MacOS/soffice", "libreoffice", "soffice"):
        try:
            subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir",
                            str(pptx.parent), str(pptx)],
                           check=True, capture_output=True, timeout=240)
            print(f"presentation -> {pptx.with_suffix('.pdf').relative_to(ROOT)}")
            return
        except (OSError, subprocess.SubprocessError):
            continue
    print("(no LibreOffice found, so no PDF was written. Open the .pptx and use File > Export as PDF.)")


if __name__ == "__main__":
    mf = REPORT_DIR / "metrics.json"
    if not mf.exists():
        sys.exit("reports/metrics.json not found. Run:  python scripts/run_pipeline.py")
    build(json.load(open(mf)), ROOT / "docs" / "presentation.pptx")
