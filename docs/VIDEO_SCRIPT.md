# 20-minute demo video: shot list

The brief requires a video of **at least 20 minutes**. This plan runs about 23–24 minutes. Record the screen with the
dashboard running (`python serve.py`) and `docs/presentation.pdf` open beside it.

**The single most important thing in this video is section 8.** Any team can show a dashboard and a high AUROC.
Almost none will stand up and say "here is where our model does *not* help, and here is the trivial baseline that
nearly matches us on one task." That is what a technical panel remembers, and it is what makes the rest of your
claims believable. Do not rush it or apologise through it — deliver it as a strength, because it is one.

| # | Time | Section | What to show | What to say |
|---|---|---|---|---|
| 1 | 0:00–1:30 | Hook | Slide 2. "On Karva Chauth, millions of Indian women fast without food **or water** until moonrise." | Half of patients in a North-Indian T2D clinic cohort fast for religion; 57% take a sulfonylurea, a drug that keeps releasing insulin when no food arrives; only 4% check their sugar during the fast. The danger is **on the calendar**. We built a twin that sees it coming. |
| 2 | 1:30–3:30 | Problem & gap | `docs/research_report.md` | Commercial twins predict meal responses, not emergencies. One of 28 reviewed twin studies looked at fairness. No open Indian CGM dataset exists. Western priors, an assumed CGM, and a blind spot for festivals. |
| 3 | 3:30–6:00 | Architecture | `docs/architecture.pdf`, slide 3 | Five layers plus the governance rail — one sentence each. Emphasise: **static data sets the prior, live data updates the posterior, the calendar says what is coming.** |
| 4 | 6:00–9:00 | The twin core | `upvaas/physiology/model.py`, then `upvaas/twin/engine.py` | Walk the SU term and the fasting clock: secretion continues while the gut is empty and hepatic output falls after ~11 hours. Show CYP2C9 and eGFR lengthening the SU half-life prior. Then the particle filter: 64 candidate physiologies re-weighted by every reading. |
| 5 | 9:00–13:00 | Live clinician demo | Dashboard → roster → pick an SU patient on a vrat day → risk cards and drivers → forecast band → tick *reveal what actually happened* → drag the time slider | Read the counterfactual drivers aloud. Point at the **Mondrian** conformal band and say why it is wider after a meal. Replay six hours so the viewer sees the risk rise before the event. |
| 6 | 13:00–15:30 | Vrat Rehearsal — the part that makes it a *twin* | Tomorrow = Karva Chauth. Run *usual*, then *move to the fast-breaking meal*, then add the low-GI swap | This is a counterfactual on the posterior particles, not a lookup. The doctor rehearses the fast **before the patient lives it**. Show P(hypo) falling as the dose moves. |
| 7 | 15:30–17:00 | ASHA "Vrat Saathi" list | ASHA tab | Ranked recall list with ward, language and an action script, escalating to an eSanjeevani teleconsult. This is how it reaches a patient with no smartphone. |
| 8 | 17:00–21:00 | **Evidence — including what does not work** | Model card tab + `reports/figures/benchmark_difficulty.png`, `ablation_hypo.png`, `conformal_coverage.png`, `precision_at_k.png` | Four beats, in this order: **(a) how hard the benchmark is** — the virtual body runs a *different* model from the twin (3-compartment gut, interstitial insulin delay, circadian SI, renal excretion, incretin), and eating is messy enough that the calendar is a prior, not an oracle. **(b) the headline** — tier B, no CGM: the twin multiplies hypoglycaemia AUPRC and cuts false alarms; of 20 flagged patients each morning, state how many are real. **(c) where it fails** — at tier A the twin does not improve precision; the calendar does not help as features; the drift monitor is a negative result. **(d) the trivial baseline** — four features, no glucose, no twin, and what it scores on hyperglycaemia. **(e) the external validation, and that it failed** — 107 real ShanghaiT2DM recordings, no retraining, 2-hour RMSE 52 vs 46 mg/dL for persistence; say plainly that it does not beat 'assume glucose stays put', and that the likeliest cause is ours (the prior was fitted on our own simulator and has never seen a real patient). Finish: "every number here regenerates from `reports/metrics.json`; `make ablations` shows what we would have claimed with an easier benchmark, and `reports/EXTERNAL_VALIDATION.md` shows what happened when we left the simulator." |
| 9 | 21:00–22:30 | Compliance & deployment | FHIR export button (open the JSON), the Consent resource, slide 8 | CDSCO Class B reasoning — informs, does not drive, a registered practitioner. DPDP: synthetic only, consent ID, purpose limitation, deletion on expiry. ABDM: FHIR R4 via HIE-CM. Then NP-NCD and Ayushman Arogya Mandirs. |
| 10 | 22:30–23:30 | Roadmap & ask | Slide 9 | External validation on ShanghaiT2DM / CGMacros (`scripts/replay_cgm_csv.py` is already written), then an IRB-approved observational pilot during Navratri 2027, then a cluster-randomised ASHA recall-list trial. |

## Before you record

- `bash run.sh` and leave the dashboard running; confirm `reports/metrics.json` is current
  (`python scripts/update_readme.py` should report "already current").
- Rebuild the deck so the slides match the metrics: `make presentation`.
- Pre-select a sulfonylurea patient whose roster chip shows a vrat today, and check that their risk curve
  actually rises before an event — scrub the slider once before recording.
- Keep *reveal what actually happened* switched **off** until you have finished explaining the forecast.
- Have `reports/metrics.json` open in a second tab so you can scroll to any number a viewer might question.

## Delivery notes

- Two people present; both must be able to answer questions on any section (two members travel to Bengaluru).
- Say "synthetic data" early and often. Never imply clinical validity.
- When you quote a number, quote its confidence interval too. The CIs are in `metrics.json` and on slide 5.
- If you only have time to rehearse one section, rehearse **section 8**.
