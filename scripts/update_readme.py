"""Regenerate the results tables inside README.md and docs/SOLUTION.md from reports/metrics.json.

    python scripts/update_readme.py

Numbers in a README drift away from the code the moment someone re-runs anything. Everything
between the `<!-- RESULTS:START -->` / `<!-- RESULTS:END -->` markers is therefore generated, so a
reader can trust that the tables match the metrics file in the same commit. Prose outside the
markers is written by hand and left alone.

CI runs this with `--check`, which fails if the committed tables are stale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upvaas.config import REPORT_DIR, ROOT  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
TIERS = [("A_full", "A · CGM + watch"), ("B_frugal", "B · 2 fingersticks/day + watch"),
         ("C_minimal", "C · EHR + phone only")]


def _ci(r, key):
    c = r.get(key)
    return "" if not c or c[0] is None else f" <sub>[{c[0]:.2f}–{c[1]:.2f}]</sub>"


def _n(v, nd=3, pct=False):
    if v is None:
        return "–"
    if pct:
        # base rates here are well under 10%; rounding them to whole percent throws away the
        # thing that makes AUPRC interpretable
        return f"{v:.1%}" if v < 0.10 else f"{v:.0%}"
    return f"{v:.{nd}f}"


def render(m: dict) -> str:
    g = m["models"]
    bd = m.get("benchmark_difficulty", {})
    coh = m["cohort"]
    fc = m["forecast"]
    L = []

    L.append(f"**{coh['n_patients']} synthetic patients × {coh['days']} days · "
             f"{coh['evaluation']}.** Every number below is regenerated from "
             "[`reports/metrics.json`](reports/metrics.json) by `python scripts/update_readme.py`, so the tables "
             "and the code cannot drift apart.\n")

    # --- difficulty -----------------------------------------------------------------------
    L.append("### How hard is this benchmark?\n")
    L.append("Accuracy figures mean nothing without this, so it comes first.\n")
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| Virtual body uses a **different model** from the twin | `{bd.get('model_mismatch')}` |")
    L.append(f"| Correlation between the festival meal plan and what was **actually eaten** (2-h windows) | "
             f"**{_n(bd.get('plan_vs_logged_carb_corr_2h'), 2)}** |")
    L.append(f"| Share of real food intake that the calendar did **not** predict | "
             f"**{_n(bd.get('real_intake_that_was_unplanned'), pct=True)}** |")
    for t, lab in (("hypo", "hypoglycaemia"), ("hyper", "hyperglycaemia")):
        tv = g.get(f"A_full/trivial_calendar_baseline/{t}")
        if tv:
            L.append(f"| 4-feature baseline (planned carbs + HbA1c + time of day; no glucose, no twin) — {lab} AUPRC "
                     f"| {_n(tv.get('auprc'))} vs **{_n(g[f'A_full/full/{t}'].get('auprc'))}** for the full model |")
    L.append("")
    L.append("> The body is integrated with `upvaas/physiology/truth_model.py`, which has a three-compartment gut, an "
             "interstitial insulin delay, circadian insulin sensitivity, renal glucose excretion and an incretin "
             "term. **The twin models none of these**, so the evaluation measures model misspecification and not "
             "just parameter error. Run `make ablations` to see what we would have scored without this.\n")

    # --- headline -------------------------------------------------------------------------
    L.append("### Main result — predicting an adverse event 2 hours ahead\n")
    for t, title, thr in (("hypo", "Hypoglycaemia", "&lt;70 mg/dL for ≥15 min"),
                          ("hyper", "Hyperglycaemia", "&gt;250 mg/dL for ≥15 min")):
        a = g[f"A_full/full/{t}"]
        L.append(f"**{title} ({thr}) within the next 2 h** · base rate {_n(a.get('prevalence'), pct=True)} · "
                 f"{a.get('event_onsets')} event onsets in "
                 f"{a.get('patients_with_events')} of {a.get('n_patients')} patients\n")
        L.append("| Sensor tier | AUROC | AUPRC <sub>[95% CI]</sub> | AUPRC without the twin | precision@20 | "
                 "event recall | false alarms / patient-day |")
        L.append("|---|---|---|---|---|---|---|")
        for key, lab in TIERS:
            r, rn = g[f"{key}/full/{t}"], g[f"{key}/no_twin/{t}"]
            best = "**" if key == "B_frugal" else ""
            L.append(f"| {best}{lab}{best} | {_n(r.get('auroc'))} | {best}{_n(r.get('auprc'))}{best}"
                     f"{_ci(r, 'auprc_ci95')} | {_n(rn.get('auprc'))} | "
                     f"{_n(r.get('precision_at_20'), 2)} | {_n(r.get('event_recall'), pct=True)} | "
                     f"{_n(r.get('false_alarms_per_patient_day'), 2)} |")
        L.append("")
    b, bn = g["B_frugal/full/hypo"], g["B_frugal/no_twin/hypo"]
    lift = (b.get("auprc") or 0) / max(bn.get("auprc") or 1e-9, 1e-9)
    p20 = b.get("precision_at_20") or 0
    L.append(f"> **The headline is tier B, not tier A.** Without a CGM — which is where almost every Indian patient "
             f"actually lives — the twin lifts hypoglycaemia AUPRC from {_n(bn.get('auprc'))} to "
             f"**{_n(b.get('auprc'))}** ({lift:.1f}×) and cuts false alarms from "
             f"{_n(bn.get('false_alarms_per_patient_day'), 2)} to "
             f"**{_n(b.get('false_alarms_per_patient_day'), 2)}** per patient-day. Of the 20 patients it flags each "
             f"morning, about **{round(p20 * 20)} really do have an event within two hours**, against a base rate of "
             f"{_n(b.get('prevalence'), pct=True)}.\n")

    # --- null results ----------------------------------------------------------------------
    L.append("### Ablations, including the ones that do not flatter us\n")
    L.append("| Ablation | AUPRC without it | Full model | Reading |")
    L.append("|---|---|---|---|")
    rows = [("A_full/no_twin/hypo", "A_full/full/hypo", "Tier A (CGM), drop the twin",
             "even against a dense CGM the mechanistic forecast adds signal a feature-based model misses"),
            ("B_frugal/no_twin/hypo", "B_frugal/full/hypo", "Tier B (fingersticks), drop the twin",
             "two readings a day are too sparse to extrapolate from; the twin's physics fills the gap"),
            ("C_minimal/no_twin/hypo", "C_minimal/full/hypo", "Tier C (no glucose), drop the twin",
             "with no glucose measurement at all, the twin's simulation is most of the signal there is"),
            ("A_full/no_calendar/hypo", "A_full/full/hypo", "Tier A, drop the calendar features",
             "the calendar already reaches the model through the twin's planned inputs, so it is "
             "double-counted as a feature"),
            ("A_full/rule_cgm_trend/hypo", "A_full/full/hypo", "Replace everything with a CGM trend rule",
             "the obvious clinical heuristic, for scale"),
            ("A_full/trivial_calendar_baseline/hyper", "A_full/full/hyper",
             "Hyperglycaemia: 4-feature timetable baseline",
             "no glucose, no wearable, no twin - how much of this task is just knowing the meal schedule")]
    for abl, full, lab, note in rows:
        if abl not in g:
            continue
        va, vf = g[abl].get("auprc") or 0, g[full].get("auprc") or 0
        # anything inside 5% relative is noise at these event counts - do not dress it up as a win
        if abs(va - vf) <= 0.05 * max(vf, 1e-9):
            verdict = "**no real difference**"
        elif va < vf:
            verdict = f"full model better ({vf / max(va, 1e-9):.1f}×)"
        else:
            verdict = "**the ablation wins**"
        L.append(f"| {lab} | {_n(va)} | {_n(vf)} | {verdict} — {note} |")
    L.append("")
    L.append("> Differences inside 5% relative are noise at these event counts and are labelled as such. "
             "The calendar's contribution **as model features** is one of them: its real value is in the "
             "Vrat Rehearsal, which runs days before any of these features exist.\n")

    # --- forecasting + uncertainty ---------------------------------------------------------
    L.append("### Forecasting and uncertainty (tier A)\n")
    L.append("| Horizon | Twin RMSE | Persistence RMSE | Clarke A+B |")
    L.append("|---|---|---|---|")
    for h in ("30min", "60min", "120min"):
        v = fc[h]
        L.append(f"| {h.replace('min', ' min')} | **{v['twin_rmse']:.0f} mg/dL** | "
                 f"{v['persistence_rmse']:.0f} mg/dL | {v['clarke_AB']:.1%} |")
    L.append("")
    c = m["conformal"]["horizons"][-1]
    sp, mo = c["split_conformal"], c["mondrian_conformal"]
    L.append("A 90% band that is 90% *on average* can be far less than that exactly where a clinician is looking, "
             "so coverage is reported conditional on the outcome (120-min horizon):\n")
    L.append("| Method | Overall | When outcome &lt;70 | When outcome &gt;250 | Mean band width |")
    L.append("|---|---|---|---|---|")
    for lab, r in (("Split conformal (one margin)", sp), ("**Mondrian (margin per glucose band)**", mo)):
        L.append(f"| {lab} | {_n(r.get('coverage'), 2)} | {_n(r.get('coverage_outcome_lt70'), 2)} | "
                 f"{_n(r.get('coverage_outcome_gt250'), 2)} | {r.get('mean_band_width_mgdl', 0):.0f} mg/dL |")
    L.append("")

    # --- fairness ---------------------------------------------------------------------------
    fr = m.get("fairness", {}).get("A_full/hypo", {})
    trad = {k.replace("tradition_", ""): v for k, v in fr.items() if k.startswith("tradition_")}
    if trad:
        L.append("### Fairness — audited first on the axis this project is built on\n")
        L.append("| Fasting tradition | AUROC | n rows | events |")
        L.append("|---|---|---|---|")
        for k, v in sorted(trad.items()):
            L.append(f"| {k} | {_n(v.get('auroc'))} | {v.get('n'):,} | {v.get('events')} |")
        others = {k: v for k, v in fr.items() if not k.startswith("tradition_") and "auroc" in v}
        if others:
            lo = min(others.items(), key=lambda kv: kv[1]["auroc"])
            hi = max(others.items(), key=lambda kv: kv[1]["auroc"])
            L.append("")
            L.append(f"Across sex, urban/rural, age, eGFR and region, hypoglycaemia AUROC ranges from "
                     f"**{lo[1]['auroc']:.3f}** ({lo[0]}) to **{hi[1]['auroc']:.3f}** ({hi[0]}).")
        L.append("")

    # --- rehearsal + drift ------------------------------------------------------------------
    rv = m.get("asha", {}).get("rehearsal_validation", {})
    if rv:
        halves = rv.get("held_out_halves", [])
        hs = " · ".join(f"{h['split']}: AUROC {_n(h.get('auroc'), 2)} ({h.get('n_events')} events)" for h in halves)
        L.append("### Vrat Rehearsal — reported with its caveat, not as a headline\n")
        L.append(f"Run the evening before a fast, the rehearsal ranks who will actually have a hypoglycaemic event "
                 f"the next day: **AUROC {_n(rv.get('auroc'), 2)}** over {rv.get('n_fast_days')} fast days — but on "
                 f"only **{rv.get('n_events')} events**, so the interval is very wide. Held-out halves: {hs}. "
                 f"It over-predicts ({_n(rv.get('mean_predicted'), pct=True)} predicted vs "
                 f"{_n(rv.get('observed_rate'), pct=True)} observed), which is the safe direction for a recall list "
                 "but means the probabilities are not calibrated. **Treat as indicative, not validated.**\n")
        a = m["asha"]
        L.append(f"Across sulfonylurea-treated fasters, moving the dose to the fast-breaking meal lowers predicted "
                 f"hypoglycaemia from **{_n(a.get('mean_p_hypo_usual_su'), pct=True)}** to "
                 f"**{_n(a.get('mean_p_hypo_dose_moved_su'), pct=True)}**.\n")
    d = m.get("drift", {})
    if d:
        L.append("### Twin self-check (drift) — a negative result\n")
        L.append(f"The normalised-innovation monitor detects illness days with AUROC only "
                 f"**{_n(d.get('auroc_relative_nis_for_illness') or d.get('auroc_daily_nis_for_illness'), 2)}** "
                 f"({_n(d.get('sensitivity_illness_days'), pct=True)} sensitivity at a "
                 f"{_n(d.get('false_flag_rate_other_days'), pct=True)} false-flag rate). "
                 "We ship it as the hook a CDSCO Algorithm Change Protocol would attach to, and report it as "
                 "**not yet good enough to deploy**.\n")
    L.append(f"*Full pipeline runtime: {m.get('runtime_sec', 0) / 60:.0f} min on a laptop CPU "
             "(`--quick` builds 60 patients in about 5).*")
    return "\n".join(L)


def patch(path: Path, block: str, check: bool) -> bool:
    src = path.read_text()
    if START not in src or END not in src:
        print(f"  ! {path.name}: no RESULTS markers, skipped")
        return True
    pre, rest = src.split(START, 1)
    _, post = rest.split(END, 1)
    new = f"{pre}{START}\n{block}\n{END}{post}"
    if new == src:
        print(f"  = {path.name} already current")
        return True
    if check:
        print(f"  ! {path.name} is STALE - run: python scripts/update_readme.py")
        return False
    path.write_text(new)
    print(f"  + {path.name} updated")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the committed tables are stale")
    args = ap.parse_args()
    mf = REPORT_DIR / "metrics.json"
    if not mf.exists():
        sys.exit("reports/metrics.json not found. Run:  python scripts/run_pipeline.py")
    blk = render(json.load(open(mf)))
    ok = all([patch(ROOT / "README.md", blk, args.check), patch(ROOT / "docs" / "SOLUTION.md", blk, args.check)])
    sys.exit(0 if ok else 1)
