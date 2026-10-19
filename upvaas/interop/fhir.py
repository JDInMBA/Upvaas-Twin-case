"""FHIR R4 export, shaped for ABDM / NRCeS (India) exchange.

Produces a `collection` Bundle per patient with Patient, Condition, Observation (labs, CGM,
wearables, genotype), MedicationStatement, RiskAssessment (the twin's prediction) and a Consent
resource that mirrors an ABDM consent artefact (purpose CAREMGT, HIU = the twin service).

All identifiers are SYNTHETIC. Codes: LOINC 4548-4 (HbA1c), 2339-0 (glucose, blood), 33914-3
(eGFR), 8867-4 (heart rate), 80404-7 (HRV, SD of R-R interval), 41950-7 (steps in 24 h);
SNOMED CT 44054006 (type 2 diabetes), 302866003 (hypoglycaemia), 80394007 (hyperglycaemia).
Before production use, validate against the NRCeS FHIR IG profiles (https://nrces.in/ndhm/fhir/r4/).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))


def _ref(res):
    return {"reference": f"{res['resourceType']}/{res['id']}"}


def _obs(pid, code, system, display, value, unit, when, category="laboratory", method=None):
    o = {"resourceType": "Observation", "id": str(uuid.uuid4()), "status": "final",
         "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                   "code": category}]}],
         "code": {"coding": [{"system": system, "code": code, "display": display}], "text": display},
         "subject": {"reference": f"Patient/{pid}"}, "effectiveDateTime": when.isoformat(),
         "valueQuantity": {"value": value, "unit": unit}}
    if method:
        o["method"] = {"text": method}
    return o


def patient_bundle(p: dict, risk: dict | None = None, cgm_points: list | None = None,
                   now: datetime | None = None) -> dict:
    now = now or datetime.now(IST)
    pid = p["patient_id"]
    LOINC, SNOMED = "http://loinc.org", "http://snomed.info/sct"
    patient = {"resourceType": "Patient", "id": pid,
               "identifier": [{"system": "https://healthid.abdm.gov.in", "value": p["abha_id"],
                               "type": {"text": "ABHA number (SYNTHETIC)"}}],
               "name": [{"text": p["name"]}], "gender": "female" if p["sex"] == "F" else "male",
               "address": [{"state": p["state"], "country": "IN"}],
               "communication": [{"language": {"text": p["language"]}}]}
    res = [patient]
    res.append({"resourceType": "Condition", "id": str(uuid.uuid4()), "subject": _ref(patient),
                "code": {"coding": [{"system": SNOMED, "code": "44054006", "display": "Type 2 diabetes mellitus"}]},
                "onsetString": f"{p['diabetes_duration_y']} years ago"})
    res.append(_obs(pid, "4548-4", LOINC, "Hemoglobin A1c", p["hba1c"], "%", now))
    res.append(_obs(pid, "33914-3", LOINC, "eGFR", p["egfr"], "mL/min/1.73m2", now))
    res.append(_obs(pid, "8867-4", LOINC, "Heart rate (resting)", p["resting_hr"], "/min", now, "vital-signs"))
    res.append({"resourceType": "Observation", "id": str(uuid.uuid4()), "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}],
                "code": {"text": "CYP2C9 genotype (pharmacogenomic)"}, "subject": _ref(patient),
                "valueString": p["cyp2c9"]})
    res.append({"resourceType": "Observation", "id": str(uuid.uuid4()), "status": "final",
                "code": {"text": "TCF7L2 rs7903146 T-allele count"}, "subject": _ref(patient),
                "valueInteger": int(p["tcf7l2_rs7903146_T"])})
    for (t, v) in (cgm_points or []):
        res.append(_obs(pid, "2339-0", LOINC, "Glucose [Mass/volume] in Blood", v, "mg/dL", t, "vital-signs",
                        method="Continuous glucose monitor (interstitial)"))
    if p["on_sulfonylurea"]:
        res.append({"resourceType": "MedicationStatement", "id": str(uuid.uuid4()), "status": "active",
                    "subject": _ref(patient), "medicationCodeableConcept": {"text": p["su_drug"]},
                    "dosage": [{"text": f"{p['su_dose_mg']} mg once daily before breakfast"}]})
    if p["on_metformin"]:
        res.append({"resourceType": "MedicationStatement", "id": str(uuid.uuid4()), "status": "active",
                    "subject": _ref(patient), "medicationCodeableConcept": {"text": "metformin"},
                    "dosage": [{"text": f"{p['metformin_mg']} mg/day"}]})
    if risk:
        res.append({"resourceType": "RiskAssessment", "id": str(uuid.uuid4()), "status": "preliminary",
                    "subject": _ref(patient), "occurrenceDateTime": now.isoformat(),
                    "method": {"text": "UPVAAS-Twin hybrid digital twin v0.1 (research prototype, not a medical device)"},
                    "prediction": [
                        {"outcome": {"coding": [{"system": SNOMED, "code": "302866003", "display": "Hypoglycemia"}]},
                         "probabilityDecimal": round(risk["p_hypo"], 3),
                         "whenRange": {"high": {"value": 120, "unit": "min"}}},
                        {"outcome": {"coding": [{"system": SNOMED, "code": "80394007", "display": "Hyperglycemia"}]},
                         "probabilityDecimal": round(risk["p_hyper"], 3),
                         "whenRange": {"high": {"value": 120, "unit": "min"}}}],
                    "note": [{"text": "Decision support for a registered medical practitioner. "
                                      "Uncertainty band and drivers available in the clinician dashboard."}]})
    res.append({"resourceType": "Consent", "id": str(uuid.uuid4()), "status": "active",
                "scope": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/consentscope", "code": "patient-privacy"}]},
                "category": [{"text": "ABDM consent artefact (HIE-CM) - SYNTHETIC"}], "patient": _ref(patient),
                "dateTime": now.isoformat(),
                "provision": {"type": "permit",
                              "period": {"start": now.isoformat(), "end": (now + timedelta(days=90)).isoformat()},
                              "purpose": [{"system": "https://projecteka.in/fhir/purpose", "code": "CAREMGT",
                                           "display": "Care Management"}],
                              "data": [{"meaning": "related", "reference": _ref(r)} for r in res[1:4]]}})
    return {"resourceType": "Bundle", "id": str(uuid.uuid4()), "type": "collection",
            "timestamp": now.isoformat(), "entry": [{"fullUrl": f"urn:uuid:{r['id']}", "resource": r} for r in res]}
