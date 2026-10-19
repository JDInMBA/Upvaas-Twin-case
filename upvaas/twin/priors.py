"""EHR -> physiology prior ("static data sets the prior, dynamic data updates the posterior").

The twin never sees the hidden physiology of the patients it serves. Instead it learns population
relationships between routine EHR fields and physiological parameters from a separate REFERENCE
cohort. In the real world this reference would come from clamp / OGTT / pharmacokinetic studies in
Indians (e.g. CURES, ICMR-INDIAB sub-studies); here it is a separate synthetic cohort generated
with a different random seed.

Pharmacogenomic hook: CYP2C9 genotype and eGFR shift the prior on sulfonylurea half-life, so a
"slow metaboliser with CKD-3" starts life in the twin with a longer drug tail before any CGM data
arrives. TCF7L2 risk alleles shift the prior on beta-cell gain.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..config import ARTIFACT_DIR

CYP_ACTIVITY = {"*1": 1.0, "*2": 0.5, "*3": 0.0}
PARAMS = ["log_SI", "log_beta", "Gb0", "log_su_gain", "log_su_thalf"]


def cyp_activity(genotype: str) -> float:
    a, b = genotype.split("/")
    return CYP_ACTIVITY[a] + CYP_ACTIVITY[b]


def ehr_design(ehr: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=ehr.index)
    X["age"] = ehr["age"]
    X["male"] = (ehr["sex"] == "M").astype(float)
    for c in ["bmi", "whtr", "tyg", "hba1c", "fpg", "diabetes_duration_y", "su_dose_frac",
              "basal_insulin_u", "prs_t2d_z", "tcf7l2_rs7903146_T"]:
        X[c] = ehr[c].astype(float)
    X["log_alt"] = np.log(ehr["alt"].astype(float))
    X["egfr_lt60"] = (ehr["egfr"] < 60).astype(float)
    X["egfr_lt45"] = (ehr["egfr"] < 45).astype(float)
    X["cyp2c9_activity"] = ehr["cyp2c9"].map(cyp_activity)
    X["neuropathy"] = ehr["neuropathy"].astype(float)
    X["on_met"] = ehr["on_metformin"].astype(float)
    X["on_su"] = ehr["on_sulfonylurea"].astype(float)
    X["on_ins"] = ehr["on_basal_insulin"].astype(float)
    return X


def truth_targets(truth: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "log_SI": np.log(truth["SI"]), "log_beta": np.log(truth["beta"]), "Gb0": truth["Gb0"],
        "log_su_gain": np.log(truth["su_gain"].clip(lower=1e-3)), "log_su_thalf": np.log(truth["su_thalf_min"]),
    }, index=truth.index)


class PhysiologyPrior:
    """One ridge regression per parameter; residual SD becomes the prior SD."""

    def __init__(self):
        self.models, self.sd, self.columns = {}, {}, None
        self.fixed = {}

    def fit(self, ehr: pd.DataFrame, truth: pd.DataFrame) -> "PhysiologyPrior":
        X = ehr_design(ehr)
        self.columns = list(X.columns)
        Y = truth_targets(truth)
        for p in PARAMS:
            mask = ehr["on_sulfonylurea"].values if p.startswith("log_su") else np.ones(len(X), bool)
            m = Ridge(alpha=1.0).fit(X[mask], Y.loc[mask, p])
            resid = Y.loc[mask, p] - m.predict(X[mask])
            self.models[p], self.sd[p] = m, float(np.std(resid) * 1.15)   # inflate: be humble
        # population constants for parameters the twin does not personalise
        self.fixed = {k: float(truth[k].mean()) for k in ["SG", "p2", "n", "dawn"]}
        # counter-regulation: population relationship with neuropathy & duration
        self.cr_model = Ridge(alpha=1.0).fit(X[["diabetes_duration_y", "neuropathy"]], np.log(truth["cr_gain"]))
        return self

    def predict(self, ehr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        X = ehr_design(ehr)[self.columns]
        mu = pd.DataFrame({p: self.models[p].predict(X) for p in PARAMS}, index=ehr.index)
        sd = pd.DataFrame({p: np.full(len(X), self.sd[p]) for p in PARAMS}, index=ehr.index)
        mu["cr"] = np.exp(self.cr_model.predict(X[["diabetes_duration_y", "neuropathy"]]))
        return mu, sd

    def save(self, path=None):
        path = path or ARTIFACT_DIR / "physiology_prior.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=None) -> "PhysiologyPrior":
        return joblib.load(path or ARTIFACT_DIR / "physiology_prior.joblib")
