# UPVAAS-Twin: final solution

**UPVAAS** (उपवास, "fast") — **U**ncertainty-aware, **P**hysiology-informed **V**irtual **A**vatar for
**A**dverse-event **S**urveillance.

> A metabolic digital twin for Indians with type 2 diabetes that knows the festival calendar. It predicts
> hypoglycaemia and hyperglycaemia 2 hours ahead, lets a doctor rehearse a patient's fast **before the patient
> lives it**, and keeps working on phones and fingersticks rather than only on CGMs.

---

## 1. The problem we chose

| Fact | Source |
|---|---|
| India has **101 million adults with diabetes**, 136 million with prediabetes, 315 million with hypertension | ICMR-INDIAB, *Lancet Diabetes Endocrinol* 2023 |
| Indians develop diabetes at lower BMI ("thin-fat"). In one cohort, diabetics had mean BMI 24.8 but **~2× visceral fat, ~5× HOMA-IR** and lower beta-cell function | *Frontiers in Endocrinology* 2025 |
| **62% of energy comes from carbohydrate**; traditional foods are not low-GI (ragi mudde GI 98, jowar roti 84) | ICMR-INDIAB diet study (*Nature Medicine* 2025); IJMR |
| In a North-Indian T2D clinic, **49% had fasted for religion** in the past year and **57% took a sulfonylurea**. Only **4% checked glucose during the fast**; **1.35%** consulted a doctor first | Gupta et al., *touchENDOCRINOLOGY* 2020 |
| Carriers of slow **CYP2C9** variants had **5.2× the odds** of severe sulfonylurea hypoglycaemia. **Renal insufficiency** in 55% of cases vs 7% of controls | Holstein et al., *Br J Clin Pharmacol* 2005 (PMC1884896) |
| Commercial "digital twins" predict **food-to-glucose responses**, not adverse events. In a 2026 review of 28 diabetes twin studies, **1 addressed fairness** and **3 had a human in the loop** | *Frontiers* 2024; *JMIR Diabetes* 2026 |

**Chosen condition:** type 2 diabetes.
**Adverse events:** hypoglycaemia (<70 mg/dL for ≥15 min) and hyperglycaemia (>250 mg/dL for ≥15 min), each
predicted **2 hours in advance**.
**The Indian twist:** sulfonylurea-treated patients during religious fasts — common, unmonitored and
**scheduled**, therefore predictable.

## 2. The insight

In India, many dangerous metabolic states are **on the calendar**. Navratri, Karva Chauth and Ramadan change meal
timing, meal composition, sleep and activity (garba nights, sehri wake-ups), and all of it is known days in
advance. A generic twin treats those days as noise. UPVAAS treats them as **planned perturbations** to a
personalised physiological model, which turns a hard time-series problem into a tractable simulation problem —
and, crucially, makes a *rehearsal* possible before the day arrives.

## 3. Architecture

See `docs/architecture.pdf`.

```
 EHR (labs, meds, CYP2C9, TCF7L2, eGFR) ──► physiology PRIOR ─┐
 CGM / fingerstick / watch / meals ──────► particle filter ───┼─► personalised twin ─► 2-h forecast ensemble
 Festival calendar + PM2.5 ──────────────► planned inputs ────┘                               │
                                                                                               ▼
        calibrated hybrid risk model  ◄── fused features (EHR + streams + calendar + twin) ─────┤
        Mondrian conformal bands, counterfactual drivers                                        │
        Vrat Rehearsal what-if (36 h)  ◄── posterior particles ─────────────────────────────────┘
        ─► clinician console · ASHA recall list · FHIR RiskAssessment · model card
```

## 4. How it works (technical)

**Mechanistic core, per minute** (`upvaas/physiology/model.py` — what the *twin* believes):

```
dG/dt = −(SG + X)·G + SG·Gb_eff + Ra(t) + cr·max(70−G,0) − e·act(t)·G
dX/dt = −p2·X + p2·SI·sleep(t)·I
dI/dt = −n·I + β·max(G − Gb,0) + k_SU·Emax(C)·(0.45 + 0.55·G/(G+90)) + n·I_basal
Ra    = f·k_abs·Q2 /(V·BW),   Q1→Q2 gut, k_abs from the food's GI and fat
dC/dt = k_a·A − ln2/t½(CYP2C9, eGFR)·C                         (sulfonylurea)
Gb_eff = Gb0 · metformin · dawn(t) · PM2.5(t) · [1 − 0.22·σ((t_fast − 660)/90)]
```

**Twin = particle filter** over θ = (ln SI, ln β, Gb0, ln k_SU, ln t½). Particles start from the EHR prior (ridge
regression fitted on a *separate* reference cohort). Hourly random-walk jitter tracks slow drift. Systematic
resampling is vectorised across patients.

**Forecast.** Every 30 minutes, 24 posterior particles are rolled 2 hours ahead using the *calendar plan*
(template meals for today's vrat type, planned SU, planned garba), producing 22 twin features: G at +30/60/120,
quantiles, P(min<70), P(max>250), SU on board, fasting clock, θ and its uncertainty, and the normalised
innovation. Forecasts use only what is knowable at the decision time — a forecast crossing midnight holds today's
sleep efficiency and PM2.5 rather than reading tomorrow's row.

**Risk model.** HistGradientBoosting on fused features, isotonic calibration on held-out patients, alarm
threshold set for 80% recall on calibration data. Evaluation is **5-fold patient-grouped** cross-validation.
Early stopping is off on purpose: scikit-learn's internal `validation_fraction` splits rows at random, and with
~1,300 autocorrelated rows per patient that would pick the iteration count on leaked data.

## 5. Data: the sandbox rules

* **Static/historical (EHR).** "Synthea-India-lite" (`upvaas/ehr/synth_ehr.py`) generates adults across 10 states
  with state-weighted urban/rural status, fasting tradition, ABHA-format IDs, labs (HbA1c, FPG, lipids, TyG,
  eGFR, ALT, Lp(a)), diagnoses, medications (93% metformin; SU titrated to glycaemia), genetics (CYP2C9, TCF7L2,
  PRS) and a vrat questionnaire. Labs are generated **from** hidden physiology with noise, so the EHR is
  informative but imperfect.
* **Dynamic/real-time.** A minute-level body simulator produces 28 days of CGM (8-min interstitial lag, AR(1)
  noise, calibration bias, dropouts), fingersticks, smartwatch HR and HRV (reacting to activity, sleep, meals and
  hypoglycaemia), steps, sleep stages and efficiency, meal logs and pillbox logs.
* **Real-data path.** `scripts/replay_cgm_csv.py` runs the twin on any CGM CSV (ShanghaiT2DM, CGMacros, an
  exported Libre file). No real patient data are bundled — DPDP/HIPAA-safe by construction.

## 6. Making the benchmark hard enough to believe

This section exists because a simulation study is only as good as the difficulty of the simulation.

**6.1 The virtual body does not run the twin's equations.** `upvaas/physiology/truth_model.py` integrates the
"real" patient with four pieces of physiology the twin has **no term for**:

| In the body | Absent from the twin |
|---|---|
| Three-compartment gut (stomach → upper intestine → absorptive pool) | Twin assumes two, so glucose appearance is earlier and sharper than reality |
| Interstitial insulin delay (~18 min) driving the remote action compartment | Twin drives its action term straight from plasma insulin |
| Circadian insulin sensitivity (±18%, trough ~06:00) | Twin knows only a sleep-quality multiplier |
| Renal glucose excretion above ~180 mg/dL, and an incretin secretion boost | Twin has neither |

So the reported error includes **model misspecification**, which is what actually breaks twins in clinic — not
just parameter error. `make ablations` reruns everything with a matched model and writes
`reports/metrics_matched_model.json`, so the cost of misspecification is measured rather than assumed.

**6.2 Eating is messy on purpose.** Wide meal-time jitter, large portion variability, skipped meals and genuinely
unplanned snacks. The correlation between the festival meal plan and what was actually eaten is printed with
every run; it should look like a useful prior, not an oracle. `--easy-meals` reproduces the tight-schedule
version so the difference is visible.

**6.3 A trivial baseline sits next to the headline.** Four features — planned carbs in the next 2 h, HbA1c, time
of day — with no glucose, no wearable and no twin. If that approaches the full model, the benchmark is measuring
a timetable rather than physiology, and §7 says so.

**6.4 A leakage regression test.** `tests/test_core.py::test_no_feature_reads_future_truth` perturbs the future
and asserts that **no** feature computed at the decision time changes. An off-by-one window cannot silently
inflate any number in this repository.

## 7. Results

<!-- RESULTS:START -->
**200 synthetic patients × 28 days · 5-fold patient-grouped cross-validation (every patient is scored by models that never saw them).** Every number below is regenerated from [`reports/metrics.json`](reports/metrics.json) by `python scripts/update_readme.py`, so the tables and the code cannot drift apart.

### How hard is this benchmark?

Accuracy figures mean nothing without this, so it comes first.

| | |
|---|---|
| Virtual body uses a **different model** from the twin | `True` |
| Correlation between the festival meal plan and what was **actually eaten** (2-h windows) | **0.55** |
| Share of real food intake that the calendar did **not** predict | **23%** |
| 4-feature baseline (planned carbs + HbA1c + time of day; no glucose, no twin) — hypoglycaemia AUPRC | 0.037 vs **0.220** for the full model |
| 4-feature baseline (planned carbs + HbA1c + time of day; no glucose, no twin) — hyperglycaemia AUPRC | 0.305 vs **0.603** for the full model |

> The body is integrated with `upvaas/physiology/truth_model.py`, which has a three-compartment gut, an interstitial insulin delay, circadian insulin sensitivity, renal glucose excretion and an incretin term. **The twin models none of these**, so the evaluation measures model misspecification and not just parameter error. Run `make ablations` to see what we would have scored without this.

### Main result — predicting an adverse event 2 hours ahead

**Hypoglycaemia (&lt;70 mg/dL for ≥15 min) within the next 2 h** · base rate 0.9% · 624 event onsets in 40 of 200 patients

| Sensor tier | AUROC | AUPRC <sub>[95% CI]</sub> | AUPRC without the twin | precision@20 | event recall | false alarms / patient-day |
|---|---|---|---|---|---|---|
| A · CGM + watch | 0.959 | 0.220 <sub>[0.10–0.34]</sub> | 0.171 | 0.36 | 56% | 1.00 |
| **B · 2 fingersticks/day + watch** | 0.911 | **0.132** <sub>[0.05–0.23]</sub> | 0.080 | 0.26 | 58% | 2.69 |
| C · EHR + phone only | 0.792 | 0.071 <sub>[0.02–0.14]</sub> | 0.034 | 0.26 | 62% | 6.87 |

**Hyperglycaemia (&gt;250 mg/dL for ≥15 min) within the next 2 h** · base rate 6.2% · 3901 event onsets in 196 of 200 patients

| Sensor tier | AUROC | AUPRC <sub>[95% CI]</sub> | AUPRC without the twin | precision@20 | event recall | false alarms / patient-day |
|---|---|---|---|---|---|---|
| A · CGM + watch | 0.942 | 0.603 <sub>[0.58–0.63]</sub> | 0.567 | 0.99 | 98% | 4.69 |
| **B · 2 fingersticks/day + watch** | 0.935 | **0.573** <sub>[0.54–0.60]</sub> | 0.483 | 0.97 | 95% | 4.95 |
| C · EHR + phone only | 0.913 | 0.448 <sub>[0.41–0.48]</sub> | 0.397 | 0.85 | 90% | 6.20 |

> **The headline is tier B, not tier A.** Without a CGM — which is where almost every Indian patient actually lives — the twin lifts hypoglycaemia AUPRC from 0.080 to **0.132** (1.6×) and cuts false alarms from 5.39 to **2.69** per patient-day. Of the 20 patients it flags each morning, about **5 really do have an event within two hours**, against a base rate of 0.9%.

### Ablations, including the ones that do not flatter us

| Ablation | AUPRC without it | Full model | Reading |
|---|---|---|---|
| Tier A (CGM), drop the twin | 0.171 | 0.220 | full model better (1.3×) — even against a dense CGM the mechanistic forecast adds signal a feature-based model misses |
| Tier B (fingersticks), drop the twin | 0.080 | 0.132 | full model better (1.6×) — two readings a day are too sparse to extrapolate from; the twin's physics fills the gap |
| Tier C (no glucose), drop the twin | 0.034 | 0.071 | full model better (2.1×) — with no glucose measurement at all, the twin's simulation is most of the signal there is |
| Tier A, drop the calendar features | 0.218 | 0.220 | **no real difference** — the calendar already reaches the model through the twin's planned inputs, so it is double-counted as a feature |
| Replace everything with a CGM trend rule | 0.099 | 0.220 | full model better (2.2×) — the obvious clinical heuristic, for scale |
| Hyperglycaemia: 4-feature timetable baseline | 0.305 | 0.603 | full model better (2.0×) — no glucose, no wearable, no twin - how much of this task is just knowing the meal schedule |

> Differences inside 5% relative are noise at these event counts and are labelled as such. The calendar's contribution **as model features** is one of them: its real value is in the Vrat Rehearsal, which runs days before any of these features exist.

### Forecasting and uncertainty (tier A)

| Horizon | Twin RMSE | Persistence RMSE | Clarke A+B |
|---|---|---|---|
| 30 min | **27 mg/dL** | 29 mg/dL | 98.9% |
| 60 min | **33 mg/dL** | 44 mg/dL | 97.9% |
| 120 min | **38 mg/dL** | 55 mg/dL | 97.2% |

A 90% band that is 90% *on average* can be far less than that exactly where a clinician is looking, so coverage is reported conditional on the outcome (120-min horizon):

| Method | Overall | When outcome &lt;70 | When outcome &gt;250 | Mean band width |
|---|---|---|---|---|
| Split conformal (one margin) | 0.90 | 0.95 | 0.47 | 125 mg/dL |
| **Mondrian (margin per glucose band)** | 0.90 | 0.91 | 0.58 | 110 mg/dL |

### Fairness — audited first on the axis this project is built on

| Fasting tradition | AUROC | n rows | events |
|---|---|---|---|
| karva_chauth | 0.980 | 20,559 | 76 |
| navratri | 0.938 | 122,551 | 1245 |
| none | 0.981 | 72,900 | 673 |
| ramadan | 0.973 | 39,164 | 404 |

Across sex, urban/rural, age, eGFR and region, hypoglycaemia AUROC ranges from **0.934** (region_west) to **0.991** (egfr_under_60).

### Vrat Rehearsal — reported with its caveat, not as a headline

Run the evening before a fast, the rehearsal ranks who will actually have a hypoglycaemic event the next day: **AUROC 0.77** over 143 fast days — but on only **11 events**, so the interval is very wide. Held-out halves: half_A: AUROC 0.80 (5 events) · half_B: AUROC 0.76 (6 events). It over-predicts (15% predicted vs 7.7% observed), which is the safe direction for a recall list but means the probabilities are not calibrated. **Treat as indicative, not validated.**

Across sulfonylurea-treated fasters, moving the dose to the fast-breaking meal lowers predicted hypoglycaemia from **28%** to **10%**.

### Twin self-check (drift) — a negative result

The normalised-innovation monitor detects illness days with AUROC only **0.53** (33% sensitivity at a 29% false-flag rate). We ship it as the hook a CDSCO Algorithm Change Protocol would attach to, and report it as **not yet good enough to deploy**.

*Full pipeline runtime: 101 min on a laptop CPU (`--quick` builds 60 patients in about 5).*
<!-- RESULTS:END -->

## 8. Honest limits

1. **The data are synthetic.** Everything here is an upper bound on what the same method would achieve
   clinically. No real patient has been involved. External validation on ShanghaiT2DM and CGMacros is the
   immediate next step and the adapter is already written.
2. **The hypoglycaemia evidence rests on a minority of patients.** At a ~0.5% base rate only a fraction of the
   cohort ever has an event, which is why every headline number carries a patient-clustered bootstrap interval
   and why the count of patients-with-events is printed next to it.
3. **The twin is not uniformly useful.** At tier A, with a dense CGM, it does not improve precision. The calendar
   does not help as *features*. Both are reported in §7 rather than buried.
4. **The drift self-check does not work well enough to deploy.** It is the hook an Algorithm Change Protocol
   would attach to, and a negative result.
5. **The Vrat Rehearsal number is indicative, not validated** — it rests on a small number of actual events, it
   over-predicts the rate (the safe direction for a recall list, but not calibrated), and both held-out halves
   are reported so the spread is visible.
6. **Effect sizes for fasting, PM2.5 and genotype** come from small or non-Indian studies; parameters marked
   `[assumption]` in the code need clinical review before any real use.
7. **Karva Chauth is the flagship scenario and the smallest cell** in the cohort. Conclusions specific to nirjala
   fasting carry the widest uncertainty of anything here.

## 9. How a doctor uses it

1. **Roster** ranked by 2-h risk, with today's vrat type, drug chip and a "recalibrate" flag.
2. **Patient header:** ABHA ID, language, ASHA ward, phenotype badges (thin-fat, CYP2C9, eGFR, neuropathy,
   TCF7L2, Lp(a)) and upcoming fasts.
3. **Risk cards:** calibrated probability, alarm threshold and counterfactual drivers.
4. **Virtual patient card:** personalised insulin sensitivity, beta-cell gain, hepatic set-point and SU half-life,
   each beside its EHR-only prior.
5. **Glucose chart:** CGM, twin estimate, 2-h forecast with the Mondrian conformal band, meals (unlogged ones
   flagged), SU doses, fast-window shading, and a "reveal what actually happened" toggle.
6. **Vrat Rehearsal:** fast type × SU action × basal change × low-GI swap × walk × garba → baseline vs scenario
   bands, P(hypo), P(severe), P(hyper), nadir time.
7. **Twin fidelity** chart (relative to the patient's own baseline) and **FHIR R4 export**.
8. **ASHA tab:** pre-festival recall list. **Model card tab:** every metric, including the unflattering ones.

## 10. Regulation and ethics by design

| Requirement | How UPVAAS meets it |
|---|---|
| DPDP Act 2023 / Rules 2025 | Synthetic data only. Production design: consent ID on every record, purpose limitation (care management), deletion on consent expiry, breach logging within 72 h; research carve-out for training on anonymised data |
| ABDM | Health Information User registration; consent artefact via HIE-CM; FHIR R4 bundles following the NRCeS IG (validate before production) |
| CDSCO draft software guidance (Oct 2025) | Informs, does not drive, a registered practitioner → likely **Class B**. The drift monitor is the ACP hook; §8 states it is not yet fit for that purpose |
| ICMR 2023 AI ethics | Human in the loop, Indian-representative priors, subgroup fairness audit **led by fasting tradition**, published model card, and a tier-C path for patients who refuse or cannot afford wearables |

## 11. Deployment path and impact

* **Where:** NP-NCD screening at Ayushman Arogya Mandirs (~25 million on diabetes treatment under "75 by 25"),
  private diabetology clinics, CGM programmes.
* **How:** the ASHA app receives the recall list two weeks before Navratri and Ramadan; the doctor runs a Vrat
  Rehearsal in clinic or via eSanjeevani; patients get a dose-timing plan in their own language.
* **Pilot plan:** (1) retrospective validation on ShanghaiT2DM and CGMacros; (2) IRB-approved observational pilot
  at one centre during Navratri 2027 with Libre CGMs on 100 SU-treated fasters; (3) cluster-randomised ASHA
  recall-list study, outcomes = hypoglycaemia events and ER visits.
* **Cost:** runs on a laptop CPU; tiers B and C need no CGM. One clinic server can serve a district.

## 12. Submission checklist (brief → repo)

| Brief requirement | Where |
|---|---|
| Fusion of static EHR + dynamic wearable streams | `upvaas/ehr/`, `upvaas/physiology/simulate.py`, `upvaas/twin/features.py` |
| Working algorithm that predicts an adverse event | `scripts/run_pipeline.py` → `reports/metrics.json` |
| Conceptual UI dashboard for a doctor | `dashboard/` + `serve.py` |
| Synthetic / open data only | synthetic generator; real-data adapter for open datasets |
| README with team, title, problem, stack, video, licence | `README.md` |
| Architecture diagram (PDF) | `docs/architecture.pdf` (`make architecture`) |
| Presentation (PDF) | `docs/presentation.pdf` (`make presentation`) |
| ≥20-min video | link in `README.md` §1; shot list in `docs/VIDEO_SCRIPT.md` |
| Open-source licence | `LICENSE` (MIT; synthetic data CC0) |

## 13. Key sources

ICMR-INDIAB (*Lancet Diabetes Endocrinol* 2023); ICMR-INDIAB diet (*Nature Medicine* 2025); Gupta et al.
(*touchENDOCRINOLOGY* 2020); Holstein et al. (*Br J Clin Pharmacol* 2005); thin-fat phenotype (*Frontiers in
Endocrinology* 2025); CARRS PM2.5 (*BMJ Open Diab Res Care* 2023); Bergman minimal model; Dalla Man et al. 2007;
Vovk/Lei on Mondrian conformal prediction; GlucoBench (ICLR 2024); PREDICT sleep–glycaemia (*Diabetologia* 2022);
*JMIR Diabetes* 2026 scoping review; DPDP Rules 2025; NRCeS FHIR IG; CDSCO draft guidance (Oct 2025); ICMR
Ethical Guidelines for AI in Healthcare (2023). Full report with links: `docs/research_report.md`.
