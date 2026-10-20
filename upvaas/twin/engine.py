"""The digital twin itself: a particle-filter data-assimilation engine around the SU-Bergman-Gut model.

For every patient the twin keeps P "candidate physiologies" (particles). Each particle carries
5 personal parameters (insulin sensitivity, beta-cell gain, hepatic set-point, SU effect, SU
half-life) plus the full physiological state. Every 5 minutes:

  1. PREDICT  - all particles are pushed through the mechanistic model with the inputs the twin
                knows about (logged meals, logged SU doses, watch steps, last night's sleep, PM2.5,
                fasting calendar).
  2. UPDATE   - when a glucose reading arrives (CGM in tier A, fingerstick in tier B) particles are
                re-weighted by how well they explain it; degenerate sets are resampled.
  3. FORECAST - every 30 minutes a subset of particles is rolled 2 hours ahead using the
                *calendar-expected* plan (meal template for today's vrat/roza type, planned SU,
                planned garba). The spread of particle futures is the twin's uncertainty.
  4. SELF-CHECK - normalised innovation squared (NIS) measures how surprised the twin is by
                reality; a sustained rise means the twin no longer matches the patient (illness,
                new drug, sensor fault) and must be re-calibrated (CDSCO "algorithm change" hook).

Tier C has no glucose readings at all: the twin then runs "open loop" on its EHR prior and the
calendar - that is the honest lower bound of what a twin can do for a patient with no devices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import THR
from ..physiology.model import add_meal, init_state, step
from .priors import PARAMS, PhysiologyPrior

STATE_KEYS = ["Q1", "Q2", "kab", "G", "X", "I", "A", "C", "tsm"]
TWIN_FEATURES = ["tw_g_now", "tw_g30", "tw_g60", "tw_g120", "tw_q10_120", "tw_q90_120", "tw_min", "tw_max",
                 "tw_p_hypo", "tw_p_hyper", "tw_t_min", "tw_su_conc", "tw_insulin", "tw_X", "tw_fast_clock",
                 "tw_SI", "tw_beta", "tw_Gb0", "tw_su_thalf", "tw_SI_sd", "tw_Gb0_sd", "tw_nis"]
BAND_H = [4, 8, 12, 16, 20, 24]          # band horizons in 5-min steps (20..120 min)
JITTER = np.array([0.012, 0.01, 0.6, 0.012, 0.006])  # hourly random-walk SD of the 5 parameters
SNACK_P, SNACK_MG = 0.01, 20000.0   # "unannounced meal" particles: 1%/step get +20 g carbs


def make_params(th, on_su, cr, bw, ins_basal, met, fixed) -> dict:
    """Map particle parameter vectors theta[..., 5] to the model parameter dict."""
    return {"SI": np.exp(th[..., 0]), "beta": np.exp(th[..., 1]), "Gb0": th[..., 2],
            "su_gain": np.exp(th[..., 3]) * on_su, "ke": np.log(2) / np.exp(th[..., 4]),
            "SG": fixed["SG"], "p2": fixed["p2"], "n": fixed["n"], "dawn": fixed["dawn"],
            "cr": cr, "bw": bw, "ins_basal": ins_basal, "met": met}


def decision_steps(T5: int) -> np.ndarray:
    return np.arange(288 + 5, T5, THR.decision_every_min // 5)


def _sample_idx(w: np.ndarray, S: int, rng) -> np.ndarray:
    """Systematic resampling, vectorised over rows. w: (M, P) normalised weights -> (M, S) indices."""
    M, P = w.shape
    c = np.cumsum(w, axis=1)
    c[:, -1] = 1.0
    u = (rng.random((M, 1)) + np.arange(S)[None, :]) / S
    off = np.arange(M)[:, None]
    idx = np.searchsorted((c + off).ravel(), (u + off).ravel()).reshape(M, S) - off * P
    return np.clip(idx, 0, P - 1)


class TwinEngine:
    def __init__(self, ehr: pd.DataFrame, prior: PhysiologyPrior, streams: dict, tier: str,
                 P: int = 64, S: int = 24, seed: int = 11, proc_sd: float = 3.0):
        self.ehr, self.st, self.tier, self.P, self.S = ehr, streams, tier, P, S
        # Process noise on glucose per 5-min step. Together with obs_sd this sets how much the
        # filter trusts its own model versus the sensor, and both are sensor-specific: a 15-minute
        # Libre trace needs a different setting from 5-minute CGM. _inputs() picks a default per
        # tier; callers working with other hardware override it.
        self.proc_sd = proc_sd
        self.rng = np.random.default_rng(seed)
        N = len(ehr)
        self.N = N
        mu, sd = prior.predict(ehr)
        self.prior_mu, self.prior_sd = mu, sd
        th = np.stack([mu[p].values for p in PARAMS], 1)[:, None, :] + \
            np.stack([sd[p].values for p in PARAMS], 1)[:, None, :] * self.rng.normal(size=(N, P, len(PARAMS)))
        self.theta = th
        self.fixed = {k: v for k, v in prior.fixed.items()}
        self.cr = mu["cr"].values[:, None]
        self.on_su = ehr["on_sulfonylurea"].values.astype(float)[:, None]
        self.bw = ehr["weight_kg"].values[:, None]
        self.ins_basal = 0.65 * ehr["basal_insulin_u"].values[:, None]
        self.met = np.where(ehr["on_metformin"].values, 0.9, 1.0)[:, None]
        self._inputs()

    # ------------------------------------------------------------------------------------------
    def _inputs(self):
        st, N = self.st, self.N
        T5 = st["cgm"].shape[1]
        self.T5 = T5
        if self.tier == "C_minimal":
            self.meal_c, self.meal_k, self.su = st["plan_c"], st["plan_k"], st["plan_su"]
            self.obs, self.obs_sd = None, None
            self.act = st["steps5"] / 5.0 * 0.75
            self.si_day = np.ones_like(st["sleep_eff_obs"])
        else:
            self.meal_c, self.meal_k, self.su = st["log_c"], st["log_k"], st["su_log"]
            self.obs = st["cgm"] if self.tier == "A_full" else st["fingerstick"]
            self.obs_sd = 14.0 if self.tier == "A_full" else 12.0
            self.act = st["steps5"] / 5.0
            self.si_day = 1.0 - 0.6 * np.maximum(0.0, 0.88 - st["sleep_eff_obs"])
        pm = np.repeat(st["pm25"], 288, axis=1)
        gb = 1 + 0.040 * (pm - 40) / 150.0
        k = np.arange(T5)[None, :] * 5
        nirjala = (st["day_type"] == 3)
        fs, fe = np.repeat(st["fast_start"], 288, 1), np.repeat(st["fast_end"], 288, 1)
        in_fast = (k >= fs) & (k < fe) & np.repeat(nirjala, 288, 1)
        self.gb5 = (gb * np.where(in_fast, 1.03, 1.0)).astype(np.float64)

    def _params(self, th):
        return make_params(th, self.on_su, self.cr, self.bw, self.ins_basal, self.met, self.fixed)

    # ------------------------------------------------------------------------------------------
    def run(self, keep_dashboard: bool = False, verbose: bool = True) -> dict:
        N, P, S, T5, rng = self.N, self.P, self.S, self.T5, self.rng
        fpg = self.ehr["fpg"].values[:, None].astype(float)
        s = init_state((N, P), fpg + rng.normal(0, 8, (N, P)), I0=self.ins_basal)
        w = np.full((N, P), 1.0 / P)
        par = self._params(self.theta)
        dsteps = decision_steps(T5)
        dpos = {k: j for j, k in enumerate(dsteps)}
        D = len(dsteps)
        feats = np.full((N, D, len(TWIN_FEATURES)), np.nan, np.float32)
        band = np.full((N, D, 3, len(BAND_H)), np.nan, np.float32) if keep_dashboard else None
        state_mean = np.zeros((N, D, len(STATE_KEYS)), np.float32) if keep_dashboard else None
        theta_daily = np.zeros((N, T5 // 288, P, len(PARAMS)), np.float32) if keep_dashboard else None
        nis_hist = np.full((N, T5), np.nan, np.float32)
        twin_mean = np.full((N, T5), np.nan, np.float32)
        hrz = THR.horizon_min // 5

        for k in range(T5):
            mod = (k * 5) % 1440
            d = (k * 5) // 1440
            mc = self.meal_c[:, k]
            if mc.any():
                add_meal(s, mc[:, None].astype(float), self.meal_k[:, k][:, None].astype(float))
            sd_ = self.su[:, k]
            if sd_.any():
                s["A"] = s["A"] + sd_[:, None]
            step(s, par, 5.0, mod, act=self.act[:, k][:, None], si_mult=self.si_day[:, d][:, None],
                 gb_mult=self.gb5[:, k][:, None])
            s["G"] = np.clip(s["G"] + rng.normal(0, self.proc_sd, (N, P)), 20, 600)
            if self.obs is not None and self.tier == "A_full":
                # unannounced-meal hypothesis: a few particles assume the patient ate something unlogged
                snack = (rng.random((N, P)) < SNACK_P) * SNACK_MG
                add_meal(s, snack, 0.02)
            m = (w * s["G"]).sum(1)
            twin_mean[:, k] = m

            # ---- UPDATE with observation -------------------------------------------------
            if self.obs is not None:
                y = self.obs[:, k]
                has = ~np.isnan(y)
                if has.any():
                    v = (w[has] * (s["G"][has] - m[has, None]) ** 2).sum(1)
                    nis_hist[has, k] = (y[has] - m[has]) ** 2 / (v + self.obs_sd ** 2)
                    ll = -0.5 * ((y[has, None] - s["G"][has]) / self.obs_sd) ** 2
                    ll -= ll.max(1, keepdims=True)
                    wn = w[has] * np.exp(ll)
                    wn /= wn.sum(1, keepdims=True)
                    w[has] = wn
                    ess = 1.0 / (w ** 2).sum(1)
                    rs = np.where(has & (ess < P / 2))[0]
                    if len(rs):
                        idx = _sample_idx(w[rs], P, rng)
                        for key in STATE_KEYS:
                            s[key][rs] = np.take_along_axis(s[key][rs], idx, 1)
                        self.theta[rs] = np.take_along_axis(self.theta[rs], idx[..., None], 1)
                        w[rs] = 1.0 / P
                        par = self._params(self.theta)
                # hourly parameter jitter (lets the twin track slow drift)
                if k % 12 == 11:
                    self.theta += rng.normal(size=self.theta.shape) * JITTER
                    par = self._params(self.theta)

            if keep_dashboard and k % 288 == 287:
                idx = _sample_idx(w, P, rng)
                theta_daily[:, k // 288] = np.take_along_axis(self.theta, idx[..., None], 1)

            # ---- FORECAST ---------------------------------------------------------------------
            if k in dpos:
                j = dpos[k]
                idx = _sample_idx(w, S, rng)
                fs = {key: np.take_along_axis(s[key], idx, 1) for key in STATE_KEYS}
                fth = np.take_along_axis(self.theta, idx[..., None], 1)
                fpar = self._params(fth)
                recent_meal = (self.meal_c[:, max(0, k - 11):k + 1] > 0).any(1)
                day0 = d * 288
                su_today = (self.su[:, day0:k + 1] > 0).any(1)
                traj = np.zeros((N, S, hrz), np.float64)
                # Forecasts must use only what is knowable NOW. Last night's sleep efficiency and
                # today's PM2.5 are known at time k; tomorrow's are not, so a forecast crossing
                # midnight holds today's values rather than reading the future row.
                si_now = self.si_day[:, d][:, None]
                gb_now = self.gb5[:, k][:, None]
                for h in range(1, hrz + 1):
                    kk = min(k + h, T5 - 1)
                    pc = self.st["plan_c"][:, kk] * ~(recent_meal & (h <= 12))
                    if pc.any():
                        add_meal(fs, pc[:, None].astype(float), self.st["plan_k"][:, kk][:, None].astype(float))
                    psu = self.st["plan_su"][:, kk] * ~su_today
                    if psu.any():
                        fs["A"] = fs["A"] + psu[:, None]
                    step(fs, fpar, 5.0, (kk * 5) % 1440, act=self.st["plan_act"][:, kk][:, None] * 0.8,
                         si_mult=si_now, gb_mult=gb_now)
                    traj[:, :, h - 1] = fs["G"]
                mins, maxs = traj.min(2), traj.max(2)
                th_mean = (w[..., None] * self.theta).sum(1)
                th_sd = np.sqrt((w[..., None] * (self.theta - th_mean[:, None, :]) ** 2).sum(1))
                lo = max(0, k - 71)
                nis_recent = np.nanmean(np.where(np.isnan(nis_hist[:, lo:k + 1]), np.nan, nis_hist[:, lo:k + 1]), 1) \
                    if self.obs is not None and np.isfinite(nis_hist[:, lo:k + 1]).any() else np.full(N, np.nan)
                f = np.column_stack([
                    m, traj[:, :, 5].mean(1), traj[:, :, 11].mean(1), traj[:, :, -1].mean(1),
                    np.quantile(traj[:, :, -1], 0.1, axis=1), np.quantile(traj[:, :, -1], 0.9, axis=1),
                    mins.mean(1), maxs.mean(1), (mins < THR.hypo).mean(1), (maxs > THR.hyper).mean(1),
                    traj.mean(1).argmin(1) * 5.0,
                    (w * s["C"]).sum(1), (w * s["I"]).sum(1), (w * s["X"]).sum(1), (w * s["tsm"]).sum(1),
                    np.exp(th_mean[:, 0]) * 1e4, np.exp(th_mean[:, 1]), th_mean[:, 2], np.exp(th_mean[:, 4]) / 60.0,
                    th_sd[:, 0], th_sd[:, 2], nis_recent])
                feats[:, j] = f
                if keep_dashboard:
                    for b, hh in enumerate(BAND_H):
                        band[:, j, :, b] = np.quantile(traj[:, :, hh - 1], [0.1, 0.5, 0.9], axis=1).T
                    state_mean[:, j] = np.column_stack([(w * s[key]).sum(1) for key in STATE_KEYS])
            if verbose and k % 1440 == 0:
                print(f"    [{self.tier}] twin day {k // 288:>2}/{T5 // 288}", flush=True)

        out = {"feats": feats, "dsteps": dsteps, "feature_names": TWIN_FEATURES, "nis": nis_hist,
               "twin_mean": twin_mean, "theta_final": self.theta, "w_final": w}
        if keep_dashboard:
            out.update(band=band, state_mean=state_mean, theta_daily=theta_daily)
        return out
