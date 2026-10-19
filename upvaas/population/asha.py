"""Pre-festival ASHA recall list ("Vrat Saathi" list).

Before a fasting festival, the twin of every patient in a ward rehearses the fast under the
patient's usual habits. Patients are ranked by predicted hypoglycaemia probability so that the
ASHA / community health officer visits the riskiest first and nudges them to see the doctor
(teleconsult via eSanjeevani) for a dose review BEFORE the fast - not after an ambulance call.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..twin.whatif import Scenario, rehearse

TRADITION_FAST = {"karva_chauth": "karva_chauth_nirjala", "ramadan": "ramadan_roza",
                  "navratri": "navratri_phalahar"}

ACTION_TEXT = {
    "high": "Visit within 48 h. Ask the patient to see the doctor (or eSanjeevani teleconsult) for a "
            "sulfonylurea dose review before the fast; carry glucose tablets; check sugar at mid-fast.",
    "medium": "Phone call. Remind: do not take the morning tablet on an empty stomach; check sugar if "
              "sweaty, shaky or confused; break the fast with water + a small meal first.",
    "low": "Include in group SMS / WhatsApp festival-health message.",
}


def recall_list(ehr: pd.DataFrame, theta_last: np.ndarray, state_last: np.ndarray, fixed: dict,
                cr: np.ndarray, start_minute: int = 21 * 60) -> pd.DataFrame:
    rows = []
    for i, (pid, p) in enumerate(ehr.iterrows()):
        if p["tradition"] == "none":
            continue
        fast = TRADITION_FAST[p["tradition"]]
        if fast == "navratri_phalahar" and p["vrat_style"] == "ek_samay":
            fast = "navratri_ek_samay"
        pd_ = p.to_dict()
        r = rehearse(pd_, theta_last[i], state_last[i], start_minute, Scenario(day_type=fast, su_action="usual"),
                     fixed, float(cr[i]), hours=30, seed=i)
        alt = rehearse(pd_, theta_last[i], state_last[i], start_minute,
                       Scenario(day_type=fast, su_action="move_to_break_fast"), fixed, float(cr[i]), hours=30, seed=i) \
            if p["on_sulfonylurea"] else r
        risk = r["p_hypo"]
        tier = "high" if risk >= 0.35 else "medium" if risk >= 0.10 else "low"
        rows.append({"patient_id": pid, "name": p["name"], "age": p["age"], "sex": p["sex"],
                     "asha_ward": p["asha_ward"], "language": p["language"], "phone": p["phone"],
                     "fast_type": fast, "on_sulfonylurea": bool(p["on_sulfonylurea"]),
                     "su_drug": p["su_drug"], "egfr": p["egfr"], "cyp2c9": p["cyp2c9"],
                     "p_hypo_usual": round(risk, 3), "p_hypo_if_dose_moved": round(alt["p_hypo"], 3),
                     "predicted_nadir_mgdl": round(r["nadir_mgdl"]), "nadir_time": r["nadir_clock"],
                     "priority": tier, "asha_action": ACTION_TEXT[tier]})
    df = pd.DataFrame(rows).sort_values("p_hypo_usual", ascending=False)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df
