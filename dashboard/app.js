/* UPVAAS-Twin clinician console - vanilla JS + Plotly. */
const C = { blue: "#2a78d6", orange: "#eb6834", aqua: "#1baf7a", crit: "#d03b3b", ink: "#141413",
  ink2: "#52514e", muted: "#898781", grid: "#e4e2da", accent: "#b4532a" };
const state = { dpos: null, n: 0, pid: null, roster: [], playing: null, tl: null };
const $ = (s) => document.querySelector(s);
const api = (p, opt) => fetch(p, opt).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
const pct = (p) => (p == null ? "–" : `${Math.round(p * 100)}%`);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const LAYOUT = (extra = {}) => Object.assign({
  margin: { l: 48, r: 16, t: 8, b: 30 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", size: 11, color: C.ink2 },
  xaxis: { gridcolor: C.grid, linecolor: "#c3c2b7", type: "date", tickformat: "%H:%M\n%d %b" },
  yaxis: { gridcolor: C.grid, zeroline: false }, showlegend: true,
  legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11 } }, hovermode: "x unified",
}, extra);
const CFG = { displayModeBar: false, responsive: true };

/* ---------------------------------------------------------------- tabs & clock */
document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === b.dataset.view));
  if (b.dataset.view === "asha") loadAsha();
  if (b.dataset.view === "card") loadCard();
  if (b.dataset.view === "clinic") window.dispatchEvent(new Event("resize"));
});
$("#slider").oninput = (e) => { state.dpos = +e.target.value; refresh(); };
$("#play").onclick = () => {
  if (state.playing) { clearInterval(state.playing); state.playing = null; $("#play").textContent = "▶"; return; }
  $("#play").textContent = "❚❚";
  state.playing = setInterval(() => {
    state.dpos = Math.min(state.n - 1, state.dpos + 1); $("#slider").value = state.dpos; refresh();
    if (state.dpos >= state.n - 1) $("#play").click();
  }, 1200);
};
["#search", "#fOnlySU", "#fFasting"].forEach((s) => $(s).oninput = renderRoster);

let pending = null;
function refresh() { clearTimeout(pending); pending = setTimeout(async () => { await loadRoster(); if (state.pid) loadPatient(state.pid); }, 120); }

/* ---------------------------------------------------------------- roster */
async function loadRoster() {
  const d = await api(`/api/patients${state.dpos != null ? `?dpos=${state.dpos}` : ""}`);
  state.dpos = d.dpos; state.n = d.n_decisions; state.roster = d.patients;
  $("#slider").max = d.n_decisions - 1; $("#slider").value = d.dpos;
  renderRoster();
  if (!state.pid) {
    const pick = d.patients.find((p) => p.on_su && p.next_fast && p.p_hypo > 0.05 && p.p_hypo < 0.6) ||
      d.patients.find((p) => p.on_su && p.next_fast) || d.patients[0];
    selectPatient(pick.id);
  }
}
function renderRoster() {
  const q = $("#search").value.toLowerCase();
  const rows = state.roster.filter((p) => (!q || p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q))
    && (!$("#fOnlySU").checked || p.on_su) && (!$("#fFasting").checked || /vrat|roza|nirjala/i.test(p.today)));
  $("#roster").innerHTML = rows.map((p) => `
    <div class="pt ${p.id === state.pid ? "sel" : ""}" data-id="${p.id}">
      <div class="row"><span class="nm">${esc(p.name)}</span><span class="meta">${p.id}</span></div>
      <div class="row meta"><span>${p.age}${p.sex} · ${esc(p.state)} · HbA1c ${p.hba1c}</span>
        ${p.on_su ? `<span class="chip">${esc(p.su_drug.replace("_MR", " MR"))}</span>` : ""}</div>
      <div class="row"><span class="chip ${/vrat|roza|nirjala/i.test(p.today) ? "fast" : ""}">${esc(p.today)}</span>
        ${p.drift_flag ? '<span class="chip flag" title="Twin no longer matches patient">⚠ recalibrate</span>' : ""}</div>
      <div class="bars">
        <span>Hypo</span><div class="bar"><i style="width:${(p.p_hypo || 0) * 100}%;background:${C.crit}"></i></div><span>${pct(p.p_hypo)}</span>
        <span>Hyper</span><div class="bar"><i style="width:${(p.p_hyper || 0) * 100}%;background:${C.orange}"></i></div><span>${pct(p.p_hyper)}</span>
      </div>
    </div>`).join("");
  document.querySelectorAll(".pt").forEach((el) => el.onclick = () => selectPatient(el.dataset.id));
  if (state.roster[0]) $("#clockLabel").textContent = state.roster.find((p) => p.id === state.pid)?.now || state.roster[0].now;
}
function selectPatient(id) { state.pid = id; renderRoster(); loadPatient(id); }

/* ---------------------------------------------------------------- patient */
async function loadPatient(id) {
  const q = `?dpos=${state.dpos}`;
  const [p, tl, rk] = await Promise.all([api(`/api/patient/${id}${q}`), api(`/api/patient/${id}/timeline${q}`), api(`/api/patient/${id}/risk${q}`)]);
  state.tl = tl;
  const e = p.ehr;
  $("#clockLabel").textContent = tl.now_label;
  $("#phead").innerHTML = `
    <div class="top"><span class="name">${esc(e.name)}</span>
      <span class="ids">${id} · ABHA ${esc(e.abha_id)} · ${e.age} y ${e.sex === "F" ? "female" : "male"} · ${esc(e.state)} (${e.urban ? "urban" : "rural"}) · ${esc(e.language)} · ASHA ward ${esc(e.asha_ward)}</span></div>
    <div class="upcoming"><b>Now:</b> ${esc(tl.now_label)} · <span class="chip ${/vrat|roza|nirjala/i.test(tl.day_type) ? "fast" : ""}">${esc(tl.day_type)}</span>
      · last-night sleep efficiency ${tl.sleep_eff_last_night ?? "–"} · PM2.5 today ${tl.pm25_today ?? "–"} µg/m³</div>
    <div class="badges">${p.badges.map((b) => `<span class="badge"><b>${esc(b.label)}</b> <span>${esc(b.detail)}</span></span>`).join("") || '<span class="hint">No special risk modifiers</span>'}</div>
    <div class="upcoming"><b>Upcoming fasts:</b> ${p.upcoming_fasts.length ? p.upcoming_fasts.map((f) => `<span class="chip fast">${esc(f.name)} · ${f.date.slice(5)}</span>`).join("") : "none in next 3 weeks"}</div>`;
  renderRisk("#riskHypo", "Hypoglycaemia in next 2 h", rk.hypo, C.crit, "<70 mg/dL ≥15 min");
  renderRisk("#riskHyper", "Hyperglycaemia in next 2 h", rk.hyper, C.orange, ">250 mg/dL ≥15 min");
  renderTwin(p.twin, e);
  plotGlucose(tl); plotRisk(tl); plotWear(tl); plotNis(p);
  renderEhr(e);
  const dt = /Karva/.test(tl.day_type) ? "karva_chauth_nirjala" : /roza/.test(tl.day_type) ? "ramadan_roza" : null;
  const nxt = p.upcoming_fasts[0]?.type;
  const map = { karva_chauth_nirjala: "karva_chauth_nirjala", ramadan_roza: "ramadan_roza", navratri_phalahar: e.vrat_style === "ek_samay" ? "navratri_ek_samay" : "navratri_phalahar", ekadashi: "ekadashi" };
  $("#wiDay").value = map[nxt] || dt || "karva_chauth_nirjala";
  $("#wiGarba").checked = !!e.garba;
  // The baseline is always "usual", so defaulting the scenario to "usual" too would open the panel
  // showing a counterfactual against itself - identical curves, identical probabilities. For a patient
  // on a sulfonylurea, open on the change a doctor would actually consider.
  $("#wiSU").value = e.on_sulfonylurea ? "move_to_break_fast" : "usual";
  runWhatIf();
}

function renderRisk(sel, title, r, color, def) {
  if (!r) { $(sel).innerHTML = `<div class="lbl">${title}</div><div class="hint">No estimate at this time point</div>`; return; }
  const icon = r.level === "high" ? "▲" : r.level === "elevated" ? "●" : "✓";
  const lvlColor = r.level === "low" ? "#2f6b2f" : r.level === "elevated" ? "#8a5a00" : "#9d2626";
  const maxAbs = Math.max(0.05, ...r.drivers.map((d) => Math.abs(d.delta)));
  $(sel).innerHTML = `<div class="lbl">${title}</div>
    <div class="big" style="color:${C.ink}">${pct(r.p)}</div>
    <div class="lvl" style="color:${lvlColor}">${icon} ${r.level.toUpperCase()} <span class="hint">alarm threshold ${pct(r.threshold)} · ${def}</span></div>
    <div class="drivers"><div class="hint">What is driving it (counterfactual, ± percentage points)</div>
      ${r.drivers.slice(0, 5).map((d) => {
        const w = Math.abs(d.delta) / maxAbs * 50, left = d.delta >= 0 ? 50 : 50 - w;
        return `<div class="drv"><span>${esc(d.group)}</span><div class="track"><i style="left:${left}%;width:${w}%;background:${d.delta >= 0 ? color : C.muted}"></i></div><span class="v">${d.delta >= 0 ? "+" : ""}${(d.delta * 100).toFixed(1)}</span></div>`;
      }).join("")}</div>`;
}

function renderTwin(t, e) {
  const row = (lbl, o, nd = 2) => o && o.value != null ? `<tr><td>${lbl}</td><td class="num"><b>${o.value}</b></td><td class="num hint">${o.prior ?? "–"}</td><td class="hint">${o.unit}</td></tr>` : "";
  $("#twinCard").innerHTML = `<h3>Virtual patient: personalised physiology</h3>
    <table><tr><td class="hint">parameter</td><td class="num hint">twin now</td><td class="num hint">EHR prior</td><td></td></tr>
      ${row("Insulin sensitivity", t.insulin_sensitivity)}${row("Beta-cell gain", t.beta_cell_gain)}
      ${row("Hepatic set-point", t.hepatic_set_point)}${row("SU half-life", t.su_half_life)}
      <tr><td>SU on board</td><td class="num"><b>${t.su_on_board ?? "–"}</b></td><td></td><td class="hint">dose units</td></tr>
      <tr><td>Fasting clock</td><td class="num"><b>${t.fasting_clock_h ?? "–"}</b></td><td></td><td class="hint">h since gut empty</td></tr>
    </table>
    <div class="cap">"EHR prior" comes from labs, genotype (CYP2C9, TCF7L2) and kidney function alone. "Twin now" has been updated by every CGM reading since.</div>`;
}

function shapesFor(tl) {
  const s = [];
  if (tl.fast_window) s.push({ type: "rect", xref: "x", yref: "paper", x0: tl.fast_window[0], x1: tl.fast_window[1], y0: 0, y1: 1, fillcolor: "#fbe9dd", opacity: 0.45, line: { width: 0 }, layer: "below" });
  s.push({ type: "line", xref: "x", yref: "paper", x0: tl.now, x1: tl.now, y0: 0, y1: 1, line: { color: C.accent, width: 1.5, dash: "dot" } });
  return s;
}

function plotGlucose(tl) {
  const yMax = Math.max(260, ...tl.cgm.filter((v) => v != null), ...tl.band.q90) + 20;
  const traces = [
    { x: tl.t, y: tl.cgm, name: "CGM", mode: "lines", line: { color: C.blue, width: 2 }, connectgaps: false },
    { x: tl.t, y: tl.twin_mean, name: "Twin estimate", mode: "lines", line: { color: C.muted, width: 1.5, dash: "dot" } },
    { x: [tl.now, ...tl.band.t], y: [tl.cgm.filter((v) => v != null).slice(-1)[0], ...tl.band.q90], mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: [tl.now, ...tl.band.t], y: [tl.cgm.filter((v) => v != null).slice(-1)[0], ...tl.band.q10], name: "Forecast 90% band (conformal)", fill: "tonexty", fillcolor: "rgba(42,120,214,0.15)", mode: "lines", line: { width: 0 } },
    { x: [tl.now, ...tl.band.t], y: [tl.twin_mean.slice(-1)[0], ...tl.band.q50], name: "Twin forecast (median)", mode: "lines", line: { color: C.blue, width: 2, dash: "dash" } },
    { x: tl.meals.map((m) => m.t), y: tl.meals.map(() => 50), name: "Meal", mode: "markers", marker: { symbol: "triangle-up", size: 10, color: C.orange, line: { color: "#fff", width: 1.5 } },
      text: tl.meals.map((m) => `${m.food} · ${m.carbs} g${m.logged ? "" : " (NOT logged)"}`), hovertemplate: "%{text}<extra></extra>" },
    { x: tl.su_doses, y: tl.su_doses.map(() => 50), name: "Sulfonylurea dose", mode: "markers", marker: { symbol: "diamond", size: 10, color: C.crit, line: { color: "#fff", width: 1.5 } } },
  ];
  if ($("#showTruth").checked) traces.push({ x: tl.future_t, y: tl.future_truth, name: "What actually happened", mode: "lines", line: { color: C.ink, width: 1.5 } });
  const layout = LAYOUT({ yaxis: { title: "mg/dL", gridcolor: C.grid, range: [40, yMax] }, shapes: [...shapesFor(tl),
    { type: "line", xref: "paper", x0: 0, x1: 1, y0: 70, y1: 70, line: { color: C.crit, width: 1, dash: "dot" } },
    { type: "line", xref: "paper", x0: 0, x1: 1, y0: 250, y1: 250, line: { color: C.orange, width: 1, dash: "dot" } }],
    annotations: [{ xref: "paper", x: 1, y: 70, text: "70", showarrow: false, xanchor: "left", font: { color: C.crit, size: 10 } },
      { xref: "paper", x: 1, y: 250, text: "250", showarrow: false, xanchor: "left", font: { color: C.orange, size: 10 } }] });
  layout.margin.r = 28;
  Plotly.react("chartGlucose", traces, layout, CFG);
}
$("#showTruth").onchange = () => state.tl && plotGlucose(state.tl);

function plotRisk(tl) {
  Plotly.react("chartRisk", [
    { x: tl.risk_t, y: tl.p_hypo, name: "P(hypo in 2 h)", mode: "lines", line: { color: C.crit, width: 2, shape: "hv" } },
    { x: tl.risk_t, y: tl.p_hyper, name: "P(hyper in 2 h)", mode: "lines", line: { color: C.orange, width: 2, shape: "hv" } },
  ], LAYOUT({ yaxis: { range: [0, 1], tickformat: ".0%", gridcolor: C.grid }, xaxis: { gridcolor: C.grid, type: "date", tickformat: "%H:%M", range: [tl.t[0], tl.future_t.slice(-1)[0] || tl.now] }, shapes: shapesFor(tl) }), CFG);
}

function plotWear(tl) {
  const sleep = tl.sleep_stage.map((s) => (s ? 1 : null));
  Plotly.react("chartWear", [
    { x: tl.t, y: tl.hr, name: "Heart rate (bpm)", mode: "lines", line: { color: C.accent, width: 1.5 } },
    { x: tl.t, y: tl.rmssd, name: "HRV RMSSD (ms)", mode: "lines", line: { color: C.aqua, width: 1.5 } },
    { x: tl.t, y: tl.steps.map((s) => s / 5), name: "Steps / min", type: "bar", marker: { color: "rgba(137,135,129,0.45)" } },
    { x: tl.t, y: sleep.map((v) => (v ? 3 : null)), name: "Asleep", mode: "lines", line: { color: C.blue, width: 6 } },
  ], LAYOUT({ yaxis: { gridcolor: C.grid }, xaxis: { gridcolor: C.grid, type: "date", tickformat: "%H:%M", range: [tl.t[0], tl.future_t.slice(-1)[0] || tl.now] }, shapes: shapesFor(tl), bargap: 0 }), CFG);
}

function plotNis(p) {
  const days = p.nis_daily.map((_, i) => i);
  Plotly.react("chartNis", [
    { x: days, y: p.nis_daily, type: "bar", name: "Daily NIS / own baseline", marker: { color: p.nis_daily.map((v) => (v > p.drift_threshold ? C.crit : C.blue)) } },
  ], LAYOUT({ showlegend: false, hovermode: "closest", xaxis: { title: "day of observation window", gridcolor: C.grid }, yaxis: { gridcolor: C.grid },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: p.drift_threshold, y1: p.drift_threshold, line: { color: C.crit, dash: "dot", width: 1 } }],
    annotations: [{ xref: "paper", x: 1, y: p.drift_threshold, text: "recalibrate", showarrow: false, xanchor: "right", yanchor: "bottom", font: { color: C.crit, size: 10 } }] }), CFG);
}

function renderEhr(e) {
  const f = [["HbA1c", `${e.hba1c} %`], ["FPG", `${e.fpg} mg/dL`], ["Duration", `${e.diabetes_duration_y} y`], ["BMI", e.bmi],
    ["Waist", `${e.waist_cm} cm`], ["TyG index", e.tyg], ["eGFR", e.egfr], ["ALT", e.alt], ["Lp(a)", `${e.lpa} mg/dL`],
    ["BP", `${e.sbp}/${e.dbp}`], ["TG / HDL / LDL", `${e.tg}/${e.hdl}/${e.ldl}`], ["Resting HR", e.resting_hr],
    ["Metformin", e.on_metformin ? `${e.metformin_mg} mg/d` : "no"], ["Sulfonylurea", e.on_sulfonylurea ? `${e.su_drug} ${e.su_dose_mg} mg` : "no"],
    ["Basal insulin", e.on_basal_insulin ? `${e.basal_insulin_u} U` : "no"], ["CYP2C9", e.cyp2c9], ["TCF7L2 T alleles", e.tcf7l2_rs7903146_T],
    ["PRS (z)", e.prs_t2d_z], ["Neuropathy", e.neuropathy ? "yes" : "no"], ["Hypos past 12 m", e.prior_hypo_12m],
    ["Fasting tradition", e.tradition], ["Vrat style", e.vrat_style], ["SU when fasting", e.on_sulfonylurea ? e.su_fasting_behaviour.replaceAll("_", " ") : "–"],
    ["Annual PM2.5", e.pm25_annual]];
  $("#ehr").innerHTML = f.map(([k, v]) => `<div><span>${k}</span><b>${esc(v)}</b></div>`).join("");
}
$("#fhirBtn").onclick = async () => {
  const b = await api(`/api/patient/${state.pid}/fhir?dpos=${state.dpos}`);
  const url = URL.createObjectURL(new Blob([JSON.stringify(b, null, 2)], { type: "application/fhir+json" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: `${state.pid}_fhir_bundle.json` }); a.click();
};

/* ---------------------------------------------------------------- what-if */
async function runWhatIf() {
  if (!state.pid) return;
  $("#wiSummary").innerHTML = '<div class="hint">Rehearsing on posterior particles…</div>';
  const body = { day_type: $("#wiDay").value, su_action: $("#wiSU").value, low_gi_swap: $("#wiGI").checked,
    walk_after_break_fast: $("#wiWalk").checked, garba: $("#wiGarba").checked, basal_insulin_change: +$("#wiIns").value, dpos: state.dpos };
  const r = await api(`/api/patient/${state.pid}/whatif`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const b = r.baseline, s = r.scenario;
  const band = (x, q10, q90, rgba, name) => [
    { x, y: q90, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x, y: q10, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: rgba, name, hoverinfo: "skip" }];
  Plotly.react("chartWhatIf", [
    ...band(b.t, b.q10, b.q90, "rgba(137,135,129,0.18)", "Usual habits: 10–90%"),
    { x: b.t, y: b.q50, mode: "lines", name: "Usual habits (median)", line: { color: C.muted, width: 2 } },
    ...band(s.t, s.q10, s.q90, "rgba(27,175,122,0.18)", "Scenario: 10–90%"),
    { x: s.t, y: s.q50, mode: "lines", name: "Scenario (median)", line: { color: C.aqua, width: 2.5 } },
  ], LAYOUT({ yaxis: { title: "mg/dL", gridcolor: C.grid }, shapes: [
    { type: "rect", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 70, fillcolor: "#fdeaea", opacity: 0.5, line: { width: 0 }, layer: "below" },
    { type: "line", xref: "paper", x0: 0, x1: 1, y0: 250, y1: 250, line: { color: C.orange, width: 1, dash: "dot" } }] }), CFG);
  const cmp = (lbl, a, c, lowerBetter = true) => {
    const cls = Math.abs(c - a) < 0.02 ? "" : (c < a) === lowerBetter ? "better" : "worse";
    return `<span>${lbl}</span><span>${pct(a)}</span><span class="${cls}">${pct(c)}</span>`;
  };
  $("#wiSummary").innerHTML = `
    <div class="cmp"><span class="h">tomorrow</span><span class="h">usual</span><span class="h">scenario</span>
      ${cmp("P(hypoglycaemia)", b.p_hypo, s.p_hypo)}${cmp("P(severe <54)", b.p_severe_hypo, s.p_severe_hypo)}${cmp("P(hyper >250)", b.p_hyper, s.p_hyper)}</div>
    <div class="note">Scenario nadir <b>${Math.round(s.nadir_mgdl)} mg/dL</b> at ${s.nadir_clock} ${s.nadir_day}.<br/>
      Tomorrow's meals: ${s.meals_tomorrow.map((m) => `${m.time} ${esc(m.food)}`).join("; ")}</div>
    <div class="hint">Starting point: ${esc(r.start)}. Bands = spread of the twin's posterior particles + meal variability.
      Decision support only - the prescriber decides.</div>`;
}
$("#wiRun").onclick = runWhatIf;

/* ---------------------------------------------------------------- ASHA */
async function loadAsha() {
  const d = await api("/api/asha?limit=200");
  const rows = d.rows, hi = rows.filter((r) => r.priority === "high").length, su = rows.filter((r) => r.on_sulfonylurea);
  const avg = (a, k) => a.length ? a.reduce((s, r) => s + r[k], 0) / a.length : 0;
  $("#ashaSummary").innerHTML = `
    <div class="kpi"><div class="v">${d.total}</div><div class="l">patients planning to fast</div></div>
    <div class="kpi"><div class="v">${hi}</div><div class="l">high priority (visit within 48 h)</div></div>
    <div class="kpi"><div class="v">${pct(avg(su, "p_hypo_usual"))} → ${pct(avg(su, "p_hypo_if_dose_moved"))}</div><div class="l">mean hypo risk on SU, if the dose is moved to the fast-breaking meal</div></div>
    <div class="kpi caveat"><div class="l"><b>Read this as a ranking, not a probability.</b> The rehearsal
      over-predicts how often a fast actually ends in hypoglycaemia — the safe direction for a recall list,
      but the numbers are not calibrated. Validated on a small number of events; see the Model card.</div></div>`;
  const pri = { high: "▲ High", medium: "● Medium", low: "✓ Low" };
  $("#ashaTable").className = "data";
  $("#ashaTable").innerHTML = `<thead><tr><th>#</th><th>Patient</th><th>Ward</th><th>Language</th><th>Fast</th><th>Drug</th>
    <th>eGFR</th><th>CYP2C9</th><th class="num">Hypo risk (usual)</th><th class="num">If dose moved</th><th>Nadir</th><th>Priority</th><th>ASHA action</th></tr></thead><tbody>
    ${rows.map((r) => `<tr><td>${r.rank}</td><td><b>${esc(r.name)}</b><br/><span class="hint">${r.patient_id} · ${r.age}${r.sex} · ${esc(r.phone)}</span></td>
      <td>${esc(r.asha_ward)}</td><td>${esc(r.language)}</td><td>${esc(r.fast_type.replaceAll("_", " "))}</td><td>${esc(r.su_drug || "–")}</td>
      <td class="num">${r.egfr}</td><td>${esc(r.cyp2c9)}</td><td class="num">${pct(r.p_hypo_usual)}</td><td class="num">${pct(r.p_hypo_if_dose_moved)}</td>
      <td>${r.predicted_nadir_mgdl} mg/dL @ ${r.nadir_time}</td><td><span class="pri ${r.priority}">${pri[r.priority]}</span></td><td class="hint">${esc(r.asha_action)}</td></tr>`).join("")}</tbody>`;
}

/* ---------------------------------------------------------------- model card */
async function loadCard() {
  const m = await api("/api/model-card");
  const f3 = (x) => (x == null ? "–" : (+x).toFixed(3));
  const tiers = [["A_full", "A · CGM + smartwatch"], ["B_frugal", "B · 2 fingersticks/day + smartwatch"], ["C_minimal", "C · EHR + phone steps only"]];
  const mrow = (key, lbl) => { const r = m.models[key] || {}; return `<tr><td>${lbl}</td><td class="num">${f3(r.auroc)}</td><td class="num">${f3(r.auprc)}${ci(r)}</td><td class="num">${f3(r.prevalence)}</td><td class="num">${r.precision_at_20 != null ? r.precision_at_20.toFixed(2) : "–"}</td><td class="num">${r.event_recall != null ? pct(r.event_recall) : "–"}</td><td class="num">${r.false_alarms_per_patient_day != null ? r.false_alarms_per_patient_day.toFixed(2) : "–"}</td><td class="num">${f3(r.ece)}</td></tr>`; };
  const head = "<tr><th>Model</th><th>AUROC</th><th>AUPRC <span class='hint'>[95% CI]</span></th><th>Prevalence</th><th>Precision@20</th><th>Event recall</th><th>False alarms / pt-day</th><th>ECE</th></tr>";
  const fc = m.forecast, cov = m.conformal.horizons;
  const bd = m.benchmark_difficulty || {}, hy = m.models["A_full/full/hypo"] || {};
  const ci = (r) => { const c = r && r.auprc_ci95; return c && c[0] != null ? ` <span class="hint">[${c[0].toFixed(2)}–${c[1].toFixed(2)}]</span>` : ""; };
  $("#cardBody").innerHTML = `
    <div class="panel prose"><h3>Intended use & honesty statement</h3>
      <ul><li><b>Intended user:</b> a registered medical practitioner (and, for the recall list, ASHAs under supervision). Outputs inform, they do not direct treatment. Under the 2025 CDSCO draft software guidance this is likely Class B decision-support software.</li>
      <li><b>Data:</b> ${m.cohort.n_patients} synthetic Indian adults with T2D, ${m.cohort.days} days each. No real patient data were used. Evaluation: ${m.cohort.evaluation}.</li>
      <li><b>How hard the test is:</b> the virtual body runs a <i>different</i> model from the twin (3-compartment gut, interstitial insulin delay, circadian insulin sensitivity, renal excretion, incretin — the twin models none of them), so these numbers include model misspecification, not just parameter error. Eating is deliberately messy: the correlation between the festival meal plan and what was actually eaten is ${(bd.plan_vs_logged_carb_corr_2h ?? 0).toFixed(2)}, with ${pct(bd.real_intake_that_was_unplanned)} of intake unplanned, so the calendar is a prior and not an oracle.</li>
      <li><b>Known limits:</b> synthetic data only — these are upper bounds, not clinical evidence. Only ${hy.patients_with_events ?? "–"} of ${hy.n_patients ?? "–"} patients ever had a hypoglycaemic event, so the confidence intervals are wide (shown below). The festival calendar adds little <i>as model features</i>; its value is the Vrat Rehearsal, days earlier. The drift monitor does not yet work well enough to deploy. Validation on open CGM data (ShanghaiT2DM, CGMacros) and then an Indian prospective cohort is required.</li></ul></div>
    <div class="panel"><h3>2-hour adverse-event prediction (out-of-fold, patient-grouped)</h3><table class="mtable">${head}
      ${tiers.map(([k, l]) => mrow(`${k}/full/hypo`, `Hypo · ${l}`)).join("")}
      ${tiers.map(([k, l]) => mrow(`${k}/full/hyper`, `Hyper · ${l}`)).join("")}
      ${mrow("A_full/no_twin/hypo", "Hypo · A without twin features")}${mrow("A_full/no_calendar/hypo", "Hypo · A without festival calendar")}
      ${mrow("A_full/twin_only/hypo", "Hypo · twin physics only")}${mrow("A_full/rule_cgm_trend/hypo", "Hypo · CGM trend rule (baseline)")}
      ${mrow("A_full/trivial_calendar_baseline/hypo", "Hypo · 4-feature timetable baseline (no glucose, no twin)")}
      ${mrow("A_full/trivial_calendar_baseline/hyper", "Hyper · 4-feature timetable baseline (no glucose, no twin)")}</table>
      <p class="hint">The last two rows are the baseline that keeps us honest: planned carbohydrate, HbA1c and time
      of day, with no glucose measurement, no wearable and no twin. If it came close to the full model, this
      benchmark would be measuring a meal timetable rather than physiology.</p></div>
    <div class="card-grid">
      <div class="panel"><h3>Graceful degradation across sensor tiers</h3><img src="/figures/degradation.png" alt="Degradation chart"/></div>
      <div class="panel"><h3>Ablation</h3><img src="/figures/ablation_hypo.png" alt="Ablation chart"/></div>
      <div class="panel"><h3>Forecast accuracy</h3><img src="/figures/forecast_rmse.png" alt="Forecast RMSE"/>
        <table class="mtable"><tr><th>Horizon</th><th>Twin RMSE</th><th>Persistence</th><th>Clarke A</th><th>Clarke A+B</th></tr>
        ${Object.entries(fc).map(([h, v]) => `<tr><td>${h}</td><td class="num">${v.twin_rmse.toFixed(1)}</td><td class="num">${v.persistence_rmse.toFixed(1)}</td><td class="num">${pct(v.clarke_A)}</td><td class="num">${pct(v.clarke_AB)}</td></tr>`).join("")}</table></div>
      <div class="panel"><h3>Uncertainty: marginal vs conditional coverage</h3>
        <p class="hint">A 90% band that is 90% <i>on average</i> can be far less than that in the range a clinician is
        actually watching. Both are reported; the chart above the table is the 2-hour horizon.</p>
        <img src="/figures/conformal_coverage.png" alt="Conformal coverage by outcome range"/>
        <table class="mtable"><tr><th>Horizon</th><th>Raw 10–90%</th><th>Split: overall</th><th>Split: &gt;250</th><th>Mondrian: overall</th><th>Mondrian: &gt;250</th></tr>
        ${cov.map((c) => { const s = c.split_conformal || {}, mo = c.mondrian_conformal || {};
          return `<tr><td>${c.horizon_min} min</td><td class="num">${pct(c.raw_10_90_coverage)}</td>
            <td class="num">${pct(s.coverage)}</td><td class="num">${s.coverage_outcome_gt250 != null ? pct(s.coverage_outcome_gt250) : "–"}</td>
            <td class="num">${pct(mo.coverage)}</td><td class="num">${mo.coverage_outcome_gt250 != null ? pct(mo.coverage_outcome_gt250) : "–"}</td></tr>`; }).join("")}</table>
        <h3 style="margin-top:14px">Twin self-check (drift) — a negative result</h3>
        <p class="hint">${esc(m.drift.honest_assessment || "")}</p>
        <table class="mtable"><tr><td>AUROC for illness days (relative NIS)</td><td class="num">${f3(m.drift.auroc_relative_nis_for_illness ?? m.drift.auroc_daily_nis_for_illness)}</td></tr>
        <tr><td>Flag rule</td><td>${esc(m.drift.flag_rule)}</td></tr>
        <tr><td>Illness days flagged</td><td class="num">${pct(m.drift.sensitivity_illness_days)}</td></tr>
        <tr><td>Other days flagged</td><td class="num">${pct(m.drift.false_flag_rate_other_days)}</td></tr></table></div>
      <div class="panel"><h3>Worklist precision — the number a clinic feels</h3>
        <img src="/figures/precision_at_k.png" alt="Precision at k"/></div>
      <div class="panel"><h3>How hard is the benchmark?</h3>
        <img src="/figures/benchmark_difficulty.png" alt="Benchmark difficulty"/></div>
      <div class="panel"><h3>Fairness audit</h3><img src="/figures/fairness.png" alt="Fairness chart"/></div>
      <div class="panel"><h3>Example: a fasting day</h3><img src="/figures/example_fasting_day.png" alt="Example fasting day"/></div>
    </div>`;
}

loadRoster();
