"""Capture the dashboard screenshots the slide deck uses.

    python serve.py 8130 &                 # dashboard must be running
    python scripts/capture_screens.py      # writes docs/img/*.png

Needs playwright, which is a developer dependency and deliberately not in requirements.txt:

    pip install playwright && python -m playwright install chromium

The deck falls back to a labelled placeholder when an image is missing, so the build never breaks
if you have not run this.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "img"

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("playwright is not installed.\n"
             "  pip install playwright && python -m playwright install chromium")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8130")
    ap.add_argument("--width", type=int, default=1680)
    ap.add_argument("--height", type=int, default=1050)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": a.width, "height": a.height}, device_scale_factor=2)
        pg.goto(a.url, wait_until="networkidle")
        pg.wait_for_timeout(2500)

        # Pick the patient that makes the point: on a sulfonylurea, fasting today, real hypo risk.
        pid = pg.evaluate("""async () => {
            const d = await fetch('/api/patients').then(r => r.json());
            const p = d.patients.find(x => x.on_su && /vrat|roza|nirjala/i.test(x.today) && x.p_hypo > 0.15)
                   || d.patients.find(x => x.on_su && /vrat|roza|nirjala/i.test(x.today))
                   || d.patients[0];
            return p.id;
        }""")
        pg.evaluate(f"document.querySelector('.pt[data-id=\"{pid}\"]')?.click()")
        pg.wait_for_timeout(3500)
        print(f"  patient on screen: {pid}")

        pg.screenshot(path=str(OUT / "console.png"))
        print(f"  wrote {OUT / 'console.png'}")

        # the Vrat Rehearsal panel, scrolled into view and given time to run
        pg.evaluate("document.querySelector('.rehearsal')?.scrollIntoView({block:'start'})")
        pg.wait_for_timeout(3000)
        panel = pg.query_selector(".rehearsal")
        (panel or pg).screenshot(path=str(OUT / "rehearsal.png"))
        print(f"  wrote {OUT / 'rehearsal.png'}")

        # ASHA recall list
        pg.evaluate("document.querySelector('.tab[data-view=\"asha\"]')?.click()")
        pg.wait_for_timeout(3000)
        pg.evaluate("window.scrollTo(0,0)")
        pg.screenshot(path=str(OUT / "asha.png"))
        print(f"  wrote {OUT / 'asha.png'}")

        # model card
        pg.evaluate("document.querySelector('.tab[data-view=\"card\"]')?.click()")
        pg.wait_for_timeout(3500)
        pg.evaluate("window.scrollTo(0,0)")
        pg.screenshot(path=str(OUT / "modelcard.png"))
        print(f"  wrote {OUT / 'modelcard.png'}")

        # record the demo patient's rehearsal numbers, so the slide headline quotes this run
        import json
        facts = pg.evaluate(f"""async () => {{
            const p = await fetch('/api/patient/{pid}').then(r => r.json());
            const sc = {{day_type: 'navratri_ek_samay', su_action: 'move_to_break_fast'}};
            const w = await fetch('/api/patient/{pid}/whatif',
                {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(sc)}})
                .then(r => r.json());
            return {{id: '{pid}', name: p.ehr.name, su_drug: p.ehr.su_drug,
                    usual_hypo: w.baseline.p_hypo, moved_hypo: w.scenario.p_hypo,
                    usual_hyper: w.baseline.p_hyper, moved_hyper: w.scenario.p_hyper}};
        }}""")
        (OUT / "demo_patient.json").write_text(json.dumps(facts, indent=1))
        print(f"  wrote {OUT / 'demo_patient.json'}  ({facts['usual_hypo']:.0%} -> {facts['moved_hypo']:.0%} hypo)")

        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
