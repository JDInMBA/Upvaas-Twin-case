"""Start the clinician dashboard.

    python serve.py              -> http://localhost:8000

Uses FastAPI + uvicorn when installed (Swagger docs at /docs); otherwise falls back to a
zero-dependency server built on Python's standard library, so the demo always runs.
"""
from __future__ import annotations

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


def check_artifacts():
    need = [ROOT / "data/cohort/ehr.csv", ROOT / "artifacts/twin_A_full.npz", ROOT / "reports/metrics.json"]
    missing = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
    if missing:
        sys.exit("Artifacts missing: " + ", ".join(missing) + "\nRun first:  python scripts/run_pipeline.py")


def run_fastapi():
    import uvicorn  # noqa: F401
    from api.main import app  # noqa: F401
    print(f"UPVAAS-Twin dashboard on http://localhost:{PORT}  (API docs: /docs)")
    uvicorn.run("api.main:app", host="0.0.0.0", port=PORT)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode())

    def log_message(self, *a):
        pass

    def do_GET(self):
        from upvaas import service
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        dpos = int(q["dpos"]) if "dpos" in q else None
        parts = [p for p in u.path.split("/") if p]
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (ROOT / "dashboard/index.html").read_bytes(), "text/html; charset=utf-8")
            if parts and parts[0] in ("static", "figures"):
                base = ROOT / ("dashboard" if parts[0] == "static" else "reports/figures")
                f = (base / "/".join(parts[1:])).resolve()
                if base.resolve() in f.parents and f.exists():
                    return self._send(200, f.read_bytes(), mimetypes.guess_type(str(f))[0] or "application/octet-stream")
                return self._json({"detail": "not found"}, 404)
            if parts[:2] == ["api", "patients"]:
                return self._json(service.patients(dpos))
            if parts[:2] == ["api", "asha"]:
                return self._json(service.asha(int(q.get("limit", 60))))
            if parts[:2] == ["api", "model-card"]:
                return self._json(service.model_card())
            if parts[:2] == ["api", "patient"] and len(parts) >= 3:
                pid = parts[2]
                sub = parts[3] if len(parts) > 3 else ""
                fn = {"": service.patient, "timeline": service.timeline, "risk": service.risk, "fhir": service.fhir}[sub]
                return self._json(fn(pid, dpos))
            self._json({"detail": "not found"}, 404)
        except KeyError as e:
            self._json({"detail": f"not found: {e}"}, 404)

    def do_POST(self):
        from upvaas import service
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if parts[:2] == ["api", "patient"] and len(parts) == 4 and parts[3] == "whatif":
            dpos = body.pop("dpos", None)
            return self._json(service.whatif(parts[2], dpos, body))
        self._json({"detail": "not found"}, 404)


def run_stdlib():
    print(f"UPVAAS-Twin dashboard on http://localhost:{PORT}  (standard-library server; "
          f"pip install fastapi uvicorn for Swagger docs)")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    check_artifacts()
    try:
        run_fastapi()
    except ImportError:
        run_stdlib()
