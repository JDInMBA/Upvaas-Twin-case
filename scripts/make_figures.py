"""Static figures for the README / slides (matplotlib). Reads reports/metrics.json + artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upvaas.config import ARTIFACT_DIR, COHORT_DIR, REPORT_DIR  # noqa: E402

# palette: categorical slots 1-3 + status critical (validated reference palette)
BLUE, ORANGE, AQUA, CRIT = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})
FIG = REPORT_DIR / "figures"


def _bars(ax, labels, series: dict, colors, ylabel):
    x = np.arange(len(labels))
    w = 0.8 / len(series)
    for i, (name, vals) in enumerate(series.items()):
        b = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w * 0.92, color=colors[i], label=name)
        ax.bar_label(b, fmt="%.2f", fontsize=8, color=INK2, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8)


def make_figures():
    m = json.load(open(REPORT_DIR / "metrics.json"))
    FIG.mkdir(parents=True, exist_ok=True)
    tiers = ["A_full", "B_frugal", "C_minimal"]
    tlab = ["A: CGM + watch", "B: fingerstick + watch", "C: EHR + phone only"]

    # 1. graceful degradation: with vs without the twin, per tier
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, t, title in zip(axes, ("hypo", "hyper"), ("Hypoglycaemia (<70) within 2 h", "Hyperglycaemia (>250) within 2 h")):
        full = [m["models"][f"{tr}/full/{t}"].get("auprc", 0) for tr in tiers]
        no = [m["models"][f"{tr}/no_twin/{t}"].get("auprc", 0) for tr in tiers]
        prev = [m["models"][f"{tr}/full/{t}"].get("prevalence", 0) for tr in tiers]
        x = np.arange(3)
        b1 = ax.bar(x - 0.2, full, 0.38, color=BLUE, label="Hybrid: twin + ML")
        b2 = ax.bar(x + 0.2, no, 0.38, color=ORANGE, label="ML without the twin")
        for b in (b1, b2):
            ax.bar_label(b, fmt="%.2f", fontsize=8, color=INK2, padding=2)
        ax.plot(x, prev, ls=":", color=MUTED, lw=1.5, marker="o", ms=3, label="chance level (prevalence)")
        ax.set_xticks(x, tlab, fontsize=8.5)
        ax.set_ylabel("AUPRC (out-of-fold)")
        ax.set_ylim(0, max(full + no) * 1.25)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Graceful degradation: the twin matters most when sensors are scarce", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94)); fig.savefig(FIG / "degradation.png", dpi=160); plt.close(fig)

    # 2. ablation on tier A, both targets, including the baseline that keeps us honest
    names = [("full", "Full UPVAAS"), ("no_calendar", "No festival calendar"), ("no_twin", "No twin features"),
             ("twin_only", "Twin physics only"), ("rule_cgm_trend", "CGM trend rule"),
             ("trivial_calendar_baseline", "4-feature calendar baseline")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for ax, t, ttl in zip(axes, ("hypo", "hyper"), ("2-h hypoglycaemia", "2-h hyperglycaemia")):
        vals, labs = [], []
        for k, lab in names:
            r = m["models"].get(f"A_full/{k}/{t}")
            if r is None:
                continue
            vals.append(r.get("auprc", 0)); labs.append(lab)
        cols = [BLUE if l == "Full UPVAAS" else (CRIT if l.startswith("4-feature") else MUTED) for l in labs]
        b = ax.barh(labs[::-1], vals[::-1], color=cols[::-1], height=0.62)
        ax.bar_label(b, fmt="%.3f", fontsize=8, color=INK2, padding=3)
        ax.set_xlim(0, max(vals) * 1.3)
        ax.set_xlabel(f"AUPRC, {ttl} (tier A, out-of-fold)")
        ax.set_title(ttl, loc="left", color=INK, fontsize=11)
    fig.suptitle("What each ingredient contributes - and what a 4-feature timetable already gets you",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "ablation_hypo.png", dpi=160); plt.close(fig)

    # 2b. benchmark difficulty: how much does the calendar really tell you about what gets eaten?
    bd = m.get("benchmark_difficulty")
    if bd and bd.get("plan_vs_logged_carb_corr_2h") is not None:
        fig, ax = plt.subplots(figsize=(7.5, 2.3))
        r = bd["plan_vs_logged_carb_corr_2h"]
        ax.barh([0], [r], color=BLUE, height=0.45)
        ax.axvline(0.91, color=CRIT, ls="--", lw=1.4)
        ax.text(0.91, 0.42, " oracle-like (earlier build)", color=CRIT, fontsize=8, va="center")
        ax.text(r, -0.42, f" this build: r = {r:.2f}, {bd['real_intake_that_was_unplanned']:.0%} of intake unplanned",
                color=INK2, fontsize=8, va="center")
        ax.set_yticks([]); ax.set_xlim(0, 1.05); ax.set_ylim(-0.8, 0.8)
        ax.set_xlabel("correlation between the festival meal plan and what was actually eaten (2-h windows)")
        ax.set_title("Benchmark difficulty: the calendar is a prior, not an oracle", loc="left", color=INK)
        fig.tight_layout(); fig.savefig(FIG / "benchmark_difficulty.png", dpi=160); plt.close(fig)

    # 3. forecast accuracy vs persistence
    fc = m["forecast"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    hs = list(fc.keys())
    _bars(ax, hs, {"Twin forecast": [fc[h]["twin_rmse"] for h in hs],
                   "Persistence (last CGM value)": [fc[h]["persistence_rmse"] for h in hs]}, [BLUE, ORANGE], "RMSE (mg/dL)")
    ax.set_title("Glucose forecast error by horizon (lower is better)", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "forecast_rmse.png", dpi=160); plt.close(fig)

    # 4. fairness
    fr = m["fairness"].get("A_full/hypo", {})
    if fr:
        keys = [k for k in fr if "auroc" in fr[k]]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        b = ax.barh(keys[::-1], [fr[k]["auroc"] for k in keys][::-1], color=BLUE, height=0.6)
        ax.bar_label(b, fmt="%.3f", fontsize=8, color=INK2, padding=3)
        ax.set_xlim(0.5, 1.0); ax.set_xlabel("AUROC, 2-h hypoglycaemia (tier A)")
        ax.set_title("Subgroup audit (ICMR 2023: non-discrimination)", loc="left", color=INK)
        fig.tight_layout(); fig.savefig(FIG / "fairness.png", dpi=160); plt.close(fig)

    # 4b. conformal coverage: marginal looks fine, conditional is where it bites
    cf = m.get("conformal", {}).get("horizons")
    if cf and "split_conformal" in cf[-1]:
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        rows = [("overall", "coverage"), ("outcome < 70 (hypo)", "coverage_outcome_lt70"),
                ("outcome 70-180", "coverage_outcome_70_180"), ("outcome > 250 (spike)", "coverage_outcome_gt250")]
        x = np.arange(len(rows)); w = 0.38
        for i, (meth, lab, col) in enumerate([("split_conformal", "Split conformal (one margin)", MUTED),
                                              ("mondrian_conformal", "Mondrian (margin per glucose band)", BLUE)]):
            v = [cf[-1][meth].get(k) or 0 for _, k in rows]
            b = ax.bar(x + (i - 0.5) * w, v, w * 0.92, color=col, label=lab)
            ax.bar_label(b, fmt="%.2f", fontsize=8, color=INK2, padding=2)
        ax.axhline(0.9, color=CRIT, ls="--", lw=1.3)
        ax.text(len(rows) - 0.4, 0.915, "target 90%", color=CRIT, fontsize=8)
        ax.set_xticks(x, [r for r, _ in rows], fontsize=9)
        ax.set_ylim(0, 1.12); ax.set_ylabel("coverage of the 2-h forecast band")
        ax.legend(loc="lower left", fontsize=8.5)
        ax.set_title("A 90% band is only 90% on average: coverage by outcome range (120-min horizon)",
                     loc="left", color=INK)
        fig.tight_layout(); fig.savefig(FIG / "conformal_coverage.png", dpi=160); plt.close(fig)

    # 4c. the metric a clinic actually feels: precision of the daily worklist
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, t, ttl in zip(axes, ("hypo", "hyper"), ("2-h hypoglycaemia", "2-h hyperglycaemia")):
        ks = [10, 20, 40]
        for tr, lab, col in zip(tiers, tlab, [BLUE, AQUA, ORANGE]):
            r = m["models"][f"{tr}/full/{t}"]
            ax.plot(ks, [r.get(f"precision_at_{k}") or 0 for k in ks], marker="o", color=col, lw=2, label=lab)
        prev = m["models"][f"A_full/full/{t}"].get("prevalence", 0)
        ax.axhline(prev, color=MUTED, ls=":", lw=1.5)
        ax.text(40, prev, f" base rate {prev:.1%}", color=MUTED, fontsize=8, va="bottom", ha="right")
        ax.set_xticks(ks); ax.set_xlabel("size of the daily worklist (k)")
        ax.set_ylabel("precision@k"); ax.set_ylim(0, 1.0)
        ax.set_title(ttl, loc="left", color=INK, fontsize=11)
    axes[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("If a doctor opens the k riskiest patients each morning, how many are real?",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "precision_at_k.png", dpi=160); plt.close(fig)

    # 5. example patient on a fasting day
    try:
        st = np.load(COHORT_DIR / "streams.npz")
        pr = np.load(ARTIFACT_DIR / "predictions.npz")
        ehr = pd.read_csv(COHORT_DIR / "ehr.csv", index_col=0)
        g, cgm, dt = st["g_true"], st["cgm"], st["day_type"]
        ds = pr["dsteps"]
        ph = pr["A_full_hypo"]
        cand = []
        for i in range(len(ehr)):
            if not ehr.iloc[i]["on_sulfonylurea"]:
                continue
            for d in range(1, dt.shape[1] - 1):
                if dt[i, d] not in (1, 3, 4):
                    continue
                low = np.where(g[i, d * 288:(d + 1) * 288] < 70)[0]
                if len(low) < 4 or low[0] < 30:
                    continue
                onset = d * 288 + low[0]
                pre = (ds >= onset - 24) & (ds < onset - 6)
                if pre.any() and np.nanmax(ph[i, pre]) > 0.3:
                    cand.append((float(np.nanmax(ph[i, pre])), i, d))
        cand = [(i, d) for _, i, d in sorted(cand, reverse=True)]
        if cand:
            i, d = cand[0]
            sl = slice(d * 288, (d + 1) * 288)
            hrs = np.arange(288) * 5 / 60
            ph = pr["A_full_hypo"][i]
            mk = (ds >= d * 288) & (ds < (d + 1) * 288)
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
            a1.axhspan(0, 70, color=CRIT, alpha=0.08)
            a1.plot(hrs, g[i, sl], color=MUTED, lw=1.2, label="true plasma glucose (hidden)")
            a1.plot(hrs, cgm[i, sl], color=BLUE, lw=2, label="CGM")
            a1.set_ylabel("mg/dL"); a1.set_ylim(40, max(260, np.nanmax(cgm[i, sl]) + 10))
            a1.legend(loc="upper right", fontsize=8)
            dtn = ["normal", "Navratri phalahar", "Ekadashi", "Karva Chauth (nirjala)", "Ramadan roza", "feast"][dt[i, d]]
            a1.set_title(f"{ehr.index[i]} ({ehr.iloc[i]['name']}, {ehr.iloc[i]['su_drug']}) - {dtn} day", loc="left", color=INK)
            a2.plot((ds[mk] - d * 288) * 5 / 60, ph[mk], color=CRIT, lw=2, marker="o", ms=3)
            a2.set_ylabel("P(hypo in 2 h)"); a2.set_ylim(0, 1); a2.set_xlabel("hour of day")
            fig.tight_layout(); fig.savefig(FIG / "example_fasting_day.png", dpi=160); plt.close(fig)
    except FileNotFoundError:
        pass
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    make_figures()
