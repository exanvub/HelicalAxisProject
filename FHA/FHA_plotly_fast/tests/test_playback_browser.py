"""Playback in a real browser.

assets/playback.js runs in the browser and, for speed, moves parts of the figures by
reaching into Plotly's internals, with Plotly's own calls as the fallback. No amount of
Python can check that, so this test opens the dashboard in headless Chrome, works the
play button and the slider (tests/playback_driver.js), and compares what the page shows
with numpy:

* the bodies, the axis of the instant, the rings and the time lines sit where the
  analysis says, moved directly and moved through Plotly's own calls;
* a shape slid directly sits on the pixel where Plotly itself draws it;
* play, pause, dragging, running to the end, and a redraw by the server while parked.

Skipped when no Chrome or Chromium is found (set CHROME to its path otherwise).

    python tests/test_playback_browser.py
    pytest tests/test_playback_browser.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import main  # noqa: E402
import FHA_visualizer_fast as fv  # noqa: E402
import recordings as rc  # noqa: E402
import shapes as sh  # noqa: E402

# The simulator, so that the test needs no file and the page is the same everywhere.
CFG = dict(
    source="synthetic", path="", polhemus_unit="cm", markers_1=[], markers_2=[], vicon_frames="least_squares",
    method="incremental_angle", cut=(0, 10**9), step=10.0, rule="net", estimator="endpoints",
    frame="auto", offset=0.0, color="time", max_axes=300, theme="light",
    sim_profile="pulsed", sim_noise="polhemus", sim_out_and_back=False,
    build_up="average", reference_path="", smoothing=0.0, bodies="segments",
)


def find_chrome():
    named = os.environ.get("CHROME")
    places = [named] if named else []
    places += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
               "/Applications/Chromium.app/Contents/MacOS/Chromium",
               r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    places += [shutil.which(name) for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")]
    return next((p for p in places if p and os.path.exists(p)), None)


def run_in_browser(timeout=150):
    """Serve the dashboard with the driver loaded into it, open it in headless Chrome and
    return what the driver reports (None if there is no Chrome)."""
    chrome = find_chrome()
    if chrome is None:
        return None
    import logging

    from werkzeug.serving import make_server

    logging.getLogger("werkzeug").setLevel(logging.ERROR)          # no line per request
    report, arrived = {}, threading.Event()

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            report.update(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            arrived.set()

        def log_message(self, *args):
            pass

    collector = HTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=collector.serve_forever, daemon=True).start()

    read_config, index = main.read_config, main.app.index_string
    main.read_config = lambda: dict(CFG)
    driver = open(os.path.join(HERE, "playback_driver.js")).read()
    address = f"http://127.0.0.1:{collector.server_port}/"
    main.app.index_string = index.replace(
        "{%renderer%}", "{%renderer%}<script>window.__report = " + json.dumps(address) + ";\n" + driver + "</script>")
    server = make_server("127.0.0.1", 0, main.app.server, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile = tempfile.mkdtemp(prefix="fha-chrome-")
    browser = subprocess.Popen(
        [chrome, "--headless=new", "--no-first-run", "--no-default-browser-check", "--enable-unsafe-swiftshader",
         "--remote-debugging-port=0", f"--user-data-dir={profile}", "--window-size=1400,900",
         f"http://127.0.0.1:{server.server_port}/"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        arrived.wait(timeout)
    finally:
        browser.terminate()
        try:
            browser.wait(10)
        except subprocess.TimeoutExpired:
            browser.kill()
        server.shutdown()
        collector.shutdown()
        main.read_config, main.app.index_string = read_config, index
        shutil.rmtree(profile, ignore_errors=True)
    assert report, f"the page did not report back within {timeout} s"
    return report


def numbers(values):
    """Nested lists from the page as an array, None becoming NaN."""
    return np.array([[np.nan if v is None else v for v in row] for row in values], float)


def test_playback_in_the_browser():
    out = run_in_browser()
    if out is None:
        print("      (skipped: no Chrome or Chromium found; set CHROME to its path)")
        return
    assert not out["errors"], out["errors"]

    rec = rc.load_synthetic(CFG["sim_profile"], CFG["sim_noise"])
    an = fv.analyse(rec)                                        # the same settings as CFG
    play = an.play
    n = len(play["index"])
    assert out["steps"] == n and abs(out["rate"] - play["rate"]) < 1e-6
    assert out["initial"]["slider"] == 0 and out["initial"]["max"] == n - 1 and out["initial"]["playing"] == "false"
    assert "Play" in out["initial"]["label"] and "Pause" not in out["initial"]["label"]
    assert out["initial"]["info"].startswith("t = ") and "frame 0" in out["initial"]["info"]

    # 1. every figure on a few steps
    worst, worst_px = 0.0, 0.0
    for k, got in out["placed"].items():
        k = int(k)
        for body in play["bodies"]:
            want = sh.move(body["vertices"], body["pose"][k])
            worst = max(worst, np.abs(np.array(got[body["key"]], float).T - want).max())
        want = sh.move(play["bodies"][1]["vertices"], play["bodies"][1]["pose"][k])[:3]
        for way in ("drawn_direct", "drawn_plotly"):               # what the GL scene holds, both ways
            worst = max(worst, np.abs(np.array(got[way], float) - want).max())
        axis = numbers(got["axis"]).T
        assert np.isfinite(axis).all() == np.isfinite(play["axis"][k]).all(), f"step {k}: axis shown or not"
        if np.isfinite(axis).all():
            worst = max(worst, np.abs(axis - play["axis"][k]).max())
        for figure in ("series", "overview"):
            assert abs(got[figure] - play["t"][k]) < 1e-3, (k, figure, got[figure], play["t"][k])
        for figure, key in (("direction", "tilt"), ("position", "pos"), ("buildup", "build"), ("swing", "swing")):
            there = bool(np.isfinite(play[key][k]).all())
            assert got[figure][2] == there, f"step {k}: ring of {figure} visible={got[figure][2]}"
            if there:
                worst = max(worst, np.abs(np.array(got[figure][:2], float) - play[key][k]).max())
        assert got["how"]["fig-scene"] == "direct", f"step {k}: the 3D view was moved by {got['how']['fig-scene']}"
        assert set(got["how_forced"].values()) == {"plotly"}, got["how_forced"]
        for figure, direct in got["screen_direct"].items():
            plotly = got["screen_plotly"][figure]
            assert (direct is None) == (plotly is None), f"step {k}: {figure} is drawn one way only"
            if direct is not None:
                worst_px = max(worst_px, float(np.abs(np.array(direct) - np.array(plotly)).max()))
        if k == n // 3 + 3:                                        # one step after another: every figure is slid
            slow = {f: how for f, how in got["how"].items() if how != "direct" and f != "fig-removed"}
            assert not slow, f"not moved directly: {slow}"
    assert worst < 2e-3, f"the page is {worst:.1e} {rec.unit} or degrees from the analysis"
    # (Plotly snaps a line to the pixel grid, a slid one keeps its fraction: up to a pixel apart.)
    assert worst_px <= 1.0, f"a shape slid directly sits {worst_px:.2f} px from where Plotly draws it"

    # 2. play and pause
    playing, paused, later = out["playing"], out["paused"], out["paused_later"]
    assert playing["playing"] == "true" and "Pause" in playing["label"] and playing["scene_in_view"]
    expected = play["rate"] * 2.0
    assert 0.6 * expected < playing["step"] < 1.4 * expected, (playing["step"], expected)
    assert playing["slider"] == playing["step"] and f"frame {play['frame'][playing['step']]}" in playing["info"]
    if playing["want_axis"][0] is not None:
        assert abs(playing["axis"][0][0] - playing["want_axis"][0]) < 1e-3, "while playing, the 3D view is on the current step"
    assert paused["playing"] == "false" and "Play" in paused["label"] and paused["step"] == later["step"], "pause holds"

    # 3. out of sight, then scrolled into view
    if not later["overview_in_view"]:
        assert abs(later["overview"] - later["t"]) > 1e-3, "a figure out of sight is left alone"
    assert out["scrolled"]["overview_in_view"] and abs(out["scrolled"]["overview"] - later["t"]) < 1e-3, "and catches up"

    # 4. dragging
    dragged = out["dragged"]
    assert dragged["slider"] == 123 and dragged["step"] == 123 and f"frame {play['frame'][123]}" in dragged["info"]
    if dragged["want_axis"][0] is not None:
        assert abs(dragged["axis"][0][0] - dragged["want_axis"][0]) < 1e-3

    # 5. speed, and the end
    assert out["speed"] == 4
    assert out["ended"]["step"] == n - 1 and out["ended"]["playing"] == "false" and "Play" in out["ended"]["label"]

    # 6. redrawn by the server while parked at the end
    redrawn = out["redrawn"]
    assert "dark" in redrawn["theme"] and redrawn["server_step"] == n - 1, "the server drew the playhead where it was"
    assert redrawn["step"] == n - 1 and abs(redrawn["series"] - play["t"][-1]) < 1e-3

    print(f"      plotly.js {out['plotly']}: all figures move in {out['ms_direct']:.1f} ms directly, "
          f"{out['ms_plotly']:.1f} ms through Plotly's own calls")


if __name__ == "__main__":
    try:
        test_playback_in_the_browser()
        print("PASS  test_playback_in_the_browser\n\n1 passed, 0 failed")
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"FAIL  test_playback_in_the_browser: {type(exc).__name__}: {str(exc)[:600]}")
        traceback.print_exc(limit=4)
        print("\n0 passed, 1 failed")
        sys.exit(1)
