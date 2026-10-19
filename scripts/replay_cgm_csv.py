"""Run the digital twin on a real CGM CSV (e.g. one ShanghaiT2DM or CGMacros patient).

    python scripts/replay_cgm_csv.py data/examples/example_cgm.csv data/examples/example_patient.json

If you have not downloaded a real dataset yet, create a demo CSV from the synthetic cohort:
    python scripts/replay_cgm_csv.py --make-example
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from upvaas.adapters import replay  # noqa: E402
from upvaas.config import COHORT_DIR, DATA_DIR  # noqa: E402


def make_example():
    st = np.load(COHORT_DIR / "streams.npz")
    ehr = pd.read_csv(COHORT_DIR / "ehr.csv", index_col=0)
    i = 0
    T5 = 7 * 288
    ts = pd.date_range("2026-10-04", periods=T5, freq="5min")
    df = pd.DataFrame({"timestamp": ts, "glucose_mgdl": st["cgm"][i, :T5], "heart_rate": st["hr"][i, :T5],
                       "steps": st["steps5"][i, :T5], "carbs_g": st["log_c"][i, :T5] / 1000.0,
                       "su_dose_frac": st["su_log"][i, :T5]})
    out = DATA_DIR / "examples"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "example_cgm.csv", index=False)
    p = ehr.iloc[i].to_dict()
    keep = ["sex", "age", "bmi", "whtr", "tyg", "hba1c", "fpg", "diabetes_duration_y", "su_dose_frac", "basal_insulin_u",
            "prs_t2d_z", "tcf7l2_rs7903146_T", "alt", "egfr", "cyp2c9", "neuropathy", "on_metformin",
            "on_sulfonylurea", "on_basal_insulin", "weight_kg"]
    json.dump({"patient_id": ehr.index[i], **{k: (v.item() if hasattr(v, "item") else v) for k, v in p.items() if k in keep}},
              open(out / "example_patient.json", "w"), indent=1)
    print(f"wrote {out/'example_cgm.csv'} and {out/'example_patient.json'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--make-example":
        make_example()
    else:
        csv, ehr = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else \
            (str(DATA_DIR / "examples/example_cgm.csv"), str(DATA_DIR / "examples/example_patient.json"))
        print(json.dumps(replay(csv, ehr), indent=1))
