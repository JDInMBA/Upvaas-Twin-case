"""Validate the exported FHIR R4 bundles against the real R4 schema.

    pip install "fhir.resources==7.1.0"
    python scripts/validate_fhir.py

Claiming "FHIR R4" is cheap; parsing every bundle with a conformant FHIR library is the thing that
makes the claim checkable, so this script is part of the submission rather than a note in the
README. It validates both the bundles committed under data/fhir/ and a freshly generated one, so a
reviewer can confirm the exporter is still correct and not just the stored files.

The validator is an OPTIONAL dependency on purpose - the pipeline, the dashboard and the tests all
run without it, and this script exits 0 with a clear message if it is not installed. ABDM's NRCeS
implementation guide adds India-specific profiles on top of base R4; passing here means the
resources are structurally valid R4, which is a necessary step towards NRCeS conformance, not a
substitute for it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upvaas.config import DATA_DIR  # noqa: E402
from upvaas.ehr.synth_ehr import generate_cohort  # noqa: E402
from upvaas.interop.fhir import patient_bundle  # noqa: E402

try:
    from fhir.resources.R4B.bundle import Bundle
except ImportError:  # pragma: no cover - optional dependency
    print('fhir.resources is not installed, skipping FHIR validation.\n'
          '  pip install "fhir.resources==7.1.0"   then re-run this script.')
    sys.exit(0)


def check(name: str, payload: dict) -> tuple[bool, str]:
    try:
        b = Bundle.model_validate(payload) if hasattr(Bundle, "model_validate") else Bundle.parse_obj(payload)
    except Exception as exc:                     # noqa: BLE001 - we want the message, whatever it is
        first = str(exc).strip().splitlines()[0]
        return False, first
    kinds = sorted({e.resource.__class__.__name__ for e in b.entry})
    return True, f"{len(b.entry)} resources: {', '.join(kinds)}"


def main() -> int:
    results = []

    stored = sorted((DATA_DIR / "fhir").glob("*.json"))
    for p in stored:
        results.append((p.name, *check(p.name, json.load(open(p)))))

    # also validate a bundle built right now, so the exporter itself is covered
    ehr, _ = generate_cohort(2, seed=7)
    pid = ehr.index[0]
    fresh = patient_bundle({"patient_id": pid, **ehr.loc[pid].to_dict()},
                           risk={"p_hypo": 0.31, "p_hyper": 0.08},
                           cgm_points=[])
    results.append(("<freshly generated>", *check("fresh", json.loads(json.dumps(fresh, default=str)))))

    width = max(len(n) for n, _, _ in results)
    ok = 0
    for name, passed, detail in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
        ok += passed
    print(f"\n{ok}/{len(results)} bundles valid against the FHIR R4B schema "
          f"(fhir.resources).")
    if ok != len(results):
        print("NOTE: base-R4 validity is necessary but not sufficient for ABDM. The NRCeS India IG adds "
              "profiles that must be checked separately before production use.")
        return 1
    print("Base R4 structural validity only. ABDM conformance additionally requires the NRCeS India IG "
          "profiles, which are not checked here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
