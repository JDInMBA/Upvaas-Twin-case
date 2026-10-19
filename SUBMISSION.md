# Submission checklist

Everything the brief asks for, and exactly what is still on you. Delete this file before submitting if you
prefer — nothing else links to it.

## Done in the repository

- [x] Working algorithm that fuses **static EHR** + **dynamic wearable streams** and predicts an adverse event
      2 hours ahead — `scripts/run_pipeline.py` → `reports/metrics.json`
- [x] Conceptual **clinician dashboard** — `dashboard/` + `serve.py` (`http://localhost:8000`)
- [x] **Synthetic / open data only** — generator in `upvaas/ehr/`, real-data adapter in `upvaas/adapters.py`,
      no patient data bundled
- [x] **README** with problem statement, healthcare use case, technical stack, AI/ML details, licence
- [x] **Architecture diagram** — `docs/architecture.pdf` (+ `.png`), rebuild with `make architecture`
- [x] **Presentation** — `docs/presentation.pdf`, 9 slides, rebuild with `make presentation`
- [x] **Open-source licence** — `LICENSE` (MIT); generated synthetic data CC0 1.0
- [x] Video **shot list** — `docs/VIDEO_SCRIPT.md` (~23 minutes of material)
- [x] Reproducibility — pinned deps, `OMP_NUM_THREADS=1`, CI on 3.10 and 3.12, 14 tests including
      leakage regressions

## Still on you — these are the ones that cost marks

1. **Record the ≥20-minute video.** Mandatory. Follow `docs/VIDEO_SCRIPT.md`; rehearse section 8 (the
   evidence/limitations section) most. Upload unlisted to YouTube and paste the link into README §1.
2. **Fill in the team table** in README §1 — team name, college/incubator, leader name + email + phone,
   members. There are four placeholder rows.
3. **Rename the folder** to `TeamName_CollegeName`. The brief is explicit about this.
4. **Create the public GitHub repository and push.** Verify in a logged-out browser window that the repo,
   the PDFs and the video link are all reachable.
5. **Submit** only the team leader's name, phone, email and the public repo link on the platform.
6. Replace the `<your-org>/<TeamName_CollegeName>` placeholder on the final slide of
   `docs/make_presentation.py`, then `make presentation`.

## Before you push

```bash
make test            # 14 tests, including the leakage regressions
make quick           # ~5 min end-to-end build, proves a fresh clone works
python scripts/update_readme.py --check   # fails if the README tables are stale
```

Then, in a **fresh clone in a new directory**, run `bash run.sh --quick` and confirm the dashboard opens.
That is the exact path a judge will take.

## If a judge asks you a hard question

The honest answers are already in the repository — know where they are:

| Question | Answer |
|---|---|
| "Isn't your simulator just your model, so of course it works?" | No. The body runs `upvaas/physiology/truth_model.py` with a 3-compartment gut, interstitial insulin delay, circadian insulin sensitivity, renal excretion and an incretin term. The twin models none of them. `make ablations` shows what we would have scored with a matched model. |
| "How much of this is just knowing the meal timetable?" | We report it: the plan-vs-reality carbohydrate correlation, and a 4-feature baseline with no glucose and no twin. Both are in README §3. |
| "What's your real sample size?" | The count of patients who actually had an event is printed next to every headline number, with a patient-clustered bootstrap CI. |
| "Does the calendar actually help?" | As *features*, no — we say so. Its value is the Vrat Rehearsal, which runs days before any of those features exist. |
| "Is the 90% uncertainty band really 90%?" | Marginally yes, conditionally no. We publish coverage by outcome range and ship the Mondrian band, which is better in the tail and narrower overall. |
| "Does your drift monitor work?" | Not yet. We report it as a negative result rather than claiming a safety feature. |
| "Have you validated on real data?" | No. The adapter is written (`scripts/replay_cgm_csv.py`) and ShanghaiT2DM / CGMacros are the immediate next step. We do not claim clinical validity anywhere. |
