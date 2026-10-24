"""Are the artefacts on disk readable by the libraries installed right now?

    python scripts/check_build_env.py            # explain, exit 1 if they are not
    python scripts/check_build_env.py --quiet    # exit code only (used by run.sh)

pandas and scikit-learn both pickle in formats that do not survive a major-version change. pandas 3
stores text columns in a dtype pandas 2 cannot unpickle, and a scikit-learn estimator saved by one
minor version warns or breaks when loaded by another. The pipeline stamps the versions it used into
`artifacts/BUILD_ENV.json`; this compares that stamp with the running interpreter so the failure
shows up here, with a fix, instead of as a 500 from the dashboard twenty minutes later.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upvaas.config import ARTIFACT_DIR  # noqa: E402

STAMP = ARTIFACT_DIR / "BUILD_ENV.json"


def current() -> dict:
    import pandas
    import sklearn
    return {"python": ".".join(str(v) for v in sys.version_info[:2]),
            "pandas": pandas.__version__, "sklearn": sklearn.__version__}


def write_stamp() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(json.dumps(current(), indent=1))


def _major_minor(v: str) -> tuple:
    parts = (v.split(".") + ["0", "0"])[:2]
    return tuple(int("".join(c for c in p if c.isdigit()) or 0) for p in parts)


def check() -> tuple[bool, str]:
    """(compatible, message). Missing stamp counts as incompatible so old builds get refreshed."""
    now = current()
    if not STAMP.exists():
        return False, "artifacts/BUILD_ENV.json is missing, so the artefacts predate this check."
    try:
        was = json.loads(STAMP.read_text())
    except (OSError, ValueError):
        return False, "artifacts/BUILD_ENV.json is unreadable."
    bad = [f"{k}: built with {was.get(k, '?')}, running {now[k]}"
           for k in ("pandas", "sklearn")
           if _major_minor(str(was.get(k, "0"))) != _major_minor(now[k])]
    if bad:
        return False, "library versions changed since the artefacts were built -> " + "; ".join(bad)
    return True, f"artefacts match the installed libraries ({now['pandas']}/{now['sklearn']})."


if __name__ == "__main__":
    ok, msg = check()
    if "--quiet" not in sys.argv:
        print(("OK: " if ok else "STALE: ") + msg)
        if not ok:
            print("\nRebuild them with:\n  bash run.sh --rebuild        # full, ~15 min"
                  "\n  python scripts/run_pipeline.py --quick   # 60 patients, ~4 min")
    sys.exit(0 if ok else 1)
