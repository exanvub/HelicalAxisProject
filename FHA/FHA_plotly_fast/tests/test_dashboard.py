"""Drives the dashboard's callbacks through its HTTP endpoint, as the browser does.

No browser and no running server needed (Flask's test client is used).

    python tests/test_dashboard.py
    pytest tests/test_dashboard.py
"""
import base64
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import main  # noqa: E402
import recordings as rc  # noqa: E402

POLHEMUS = os.path.join(rc.TEST_DATA, "Polhemus_test_data")
VICON = os.path.join(rc.TEST_DATA, "Vicon_test_data")
FILE3 = os.path.join(POLHEMUS, "3_90deg_y_with_xxdeg_x_rotationdata.csv")
FILE1 = os.path.join(POLHEMUS, "1_90deg_ydata.csv")
SERVO = os.path.join(VICON, "servo_knee_90degX.c3d")

# A fixed configuration, so the tests do not depend on what config.py currently says.
CFG = dict(
    source="polhemus", path=FILE3, polhemus_unit="cm",
    markers_1=["lateral_epicondyl", "medial_epicondyl", "caput"],
    markers_2=["ankle", "lateral_condyl", "medial_condyl"], vicon_frames="least_squares",
    method="incremental_angle", cut=(100, 1500), step=10.0, rule="net", estimator="endpoints",
    frame="auto", offset=0.0, color="time", max_axes=300, theme="light",
    sim_profile="proportional", sim_noise="polhemus", sim_out_and_back=False,
    build_up="reference", reference_path=FILE1, smoothing=0.0,
)
main.read_config = lambda: dict(CFG)
CLIENT = main.app.server.test_client()

DEFAULTS = {
    "source.value": "polhemus", "file.value": FILE3, "custom-path.value": "",
    "syn-profile.value": "proportional", "syn-noise.value": "polhemus", "syn-return.value": [],
    "cfg.data": json.loads(json.dumps(CFG)),
    "frames.value": [100, 1500], "method.value": "incremental_angle", "step.value": 10,
    "rule.value": "net", "estimator.value": "endpoints", "frame.value": "auto", "color.value": "time",
    "offset.value": 0, "max-axes.value": 300, "theme.value": "light", "download-btn.n_clicks": 1,
    "buildup.value": "reference", "reference.value": FILE1, "smoothing.value": 0,
}


def fire(first_output, **values):
    """Run the callback whose first output is ``first_output`` ("id.property") and
    return {"id.property": value} for every output it updated."""
    known = dict(DEFAULTS)
    known.update(values)
    key = next(k for k in main.app.callback_map if k.strip(".").split("...")[0] == first_output)
    spec = main.app.callback_map[key]
    def arg(dep):
        name = f"{dep['id']}.{dep['property']}"
        assert name in known, f"test needs a value for {name}"
        return dict(id=dep["id"], property=dep["property"], value=known[name])
    outputs = [dict(id=o.split(".")[0], property=o.split(".")[1]) for o in key.strip(".").split("...")]
    body = dict(output=key, outputs=outputs if len(outputs) > 1 else outputs[0],
                inputs=[arg(d) for d in spec["inputs"]], state=[arg(d) for d in spec["state"]],
                changedPropIds=[f"{spec['inputs'][0]['id']}.{spec['inputs'][0]['property']}"])
    reply = CLIENT.post("/_dash-update-component", json=body)
    assert reply.status_code in (200, 204), (reply.status_code, reply.get_data(as_text=True)[:400])
    if reply.status_code == 204:
        return {}
    data = reply.get_json()["response"]
    return {f"{cid}.{prop}": val for cid, props in data.items() for prop, val in props.items()}


def spec_for(**values):
    out = fire("spec.data", **values)
    return out["spec.data"], out


def text_of(node):
    """All text inside a serialised Dash component tree."""
    if node is None:
        return ""
    if isinstance(node, (str, int, float)):
        return str(node)
    if isinstance(node, list):
        return " ".join(text_of(n) for n in node)
    return text_of(node.get("props", {}).get("children"))


def trace_names(fig):
    return [t.get("name") for t in fig["data"] if t.get("name")]


def array(value):
    """A figure array as numpy. Plotly 6 sends numeric arrays base64-encoded
    ({"dtype": ..., "bdata": ...}); older versions send plain lists."""
    if isinstance(value, dict) and "bdata" in value:
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
    return np.asarray(value, dtype=float)


def n_points(fig, name):
    trace = next(t for t in fig["data"] if t.get("name") == name)
    return int(np.isfinite(array(trace["x"])).sum())


# --------------------------------------------------------------------------- layout

def test_layout_contains_every_component_the_callbacks_use():
    layout = json.loads(json.dumps(main.serve_layout(), cls=__import__("plotly").utils.PlotlyJSONEncoder))
    ids = set()

    def walk(node):
        if isinstance(node, dict):
            props = node.get("props", {})
            if "id" in props:
                ids.add(props["id"])
            walk(props.get("children"))
        elif isinstance(node, list):
            for n in node:
                walk(n)

    walk(layout)
    for key, spec in main.app.callback_map.items():
        used = [o.split(".")[0] for o in key.strip(".").split("...")]
        used += [d["id"] for d in spec["inputs"] + spec["state"]]
        missing = [i for i in used if i not in ids]
        assert not missing, f"callback {key} refers to components missing from the layout: {missing}"


def test_page_and_stylesheet_are_served():
    assert CLIENT.get("/").status_code == 200
    css = CLIENT.get("/assets/style.css")
    assert css.status_code == 200 and b".fha" in css.data


# ----------------------------------------------------------------- choosing the data

def test_choosing_a_source_lists_its_recordings_and_selects_the_config_file():
    out = fire("file.options", **{"source.value": "polhemus"})
    paths = [o["value"] for o in out["file.options"]]
    assert FILE3 in paths and len(paths) >= 10
    assert out["file.value"] == FILE3
    assert out["reference.value"] == FILE1 and FILE1 in [o["value"] for o in out["reference.options"]]
    assert out["syn-group.style"] == {"display": "none"} and out["file-group.style"] == {}
    out = fire("file.options", **{"source.value": "vicon"})
    assert all(o["value"].endswith(".c3d") for o in out["file.options"]) and out["file.value"].endswith(".c3d")
    assert out["reference.value"] is None, "a Polhemus reference from config.py is not offered for Vicon data"
    assert all(o["value"].endswith(".c3d") for o in out["reference.options"])
    out = fire("file.options", **{"source.value": "synthetic"})
    assert "file.options" not in out, "the file list is left alone for the simulator"
    assert out["file-group.style"] == {"display": "none"} and out["syn-group.style"] == {}


def test_the_config_file_opens_on_the_config_frame_range_and_others_on_the_whole_recording():
    spec, out = spec_for()
    assert spec["source"] == "polhemus" and spec["path"] == FILE3
    assert out["frames.max"] == 2865 and out["frames.value"] == [100, 1500]
    spec, out = spec_for(**{"file.value": FILE1})
    assert out["frames.value"] == [0, out["frames.max"]] and out["frames.max"] == 2401


def test_a_typed_path_overrides_the_file_list_and_a_bad_one_is_reported():
    spec, out = spec_for(**{"custom-path.value": f"  {FILE1}  "})
    assert spec["path"] == FILE1
    spec, out = spec_for(**{"custom-path.value": "/no/such/file.csv"})
    assert "error" in spec and "File not found" in spec["error"]
    assert "frames.max" not in out, "the frame slider is left as it was"
    shown = fire("root.className", **{"spec.data": spec})
    assert "File not found" in text_of(shown["notes.children"])
    assert "fig-scene.figure" not in shown, "the previous figures stay on screen"


# ------------------------------------------------------------------- the main update

def update(**values):
    spec = values.pop("spec", None) or spec_for(**{k: v for k, v in values.items()
                                                   if k.split(".")[0] in ("source", "file", "custom-path",
                                                                          "syn-profile", "syn-noise", "syn-return")})[0]
    return fire("root.className", **{"spec.data": spec, **values})


def test_default_view_fills_every_panel():
    out = update()
    assert out["root.className"] == "fha theme-light"
    assert out["recording-name.children"] == os.path.basename(FILE3)
    assert len(out["tiles.children"]) == 7 and "Mean angle" in text_of(out["tiles.children"])
    assert "lab frame" in text_of(out["notes.children"]) and "arrival times" in text_of(out["notes.children"])
    scene = trace_names(out["fig-scene.figure"])
    for name in ("Helical axes", "Average axis", "Pivot", "Cross-section plane", "Moving sensor", "Reference sensor"):
        assert name in scene, name
    for fig in ("fig-direction.figure", "fig-position.figure"):
        assert n_points(out[fig], "Helical axes") > 10
    assert n_points(out["fig-buildup.figure"], "Measured") > 100 and n_points(out["fig-swing.figure"], "Tilt built up") > 10
    assert len(out["fig-series.figure"]["data"]) == 3
    assert len(out["fig-overview.figure"]["data"]) == 1
    rows = out["table.children"]["props"]["children"][1]["props"]["children"]
    assert len(rows) == 300, "one table row per axis drawn"
    assert out["step-label.children"] == "Window (deg)" and out["step.disabled"] is False


def test_every_method_and_option_renders():
    for method, step in (("incremental_angle", 10), ("step_angle", 10), ("reference_pose", 10),
                         ("incremental_time", 60), ("all_FHA", 1)):
        out = update(**{"method.value": method, "step.value": step})
        assert "Helical axes" in trace_names(out["fig-scene.figure"]), method
        assert out["step.disabled"] is (method == "all_FHA")
        assert out["step-label.children"] == ("Lag (frames)" if method == "incremental_time" else "Window (deg)")
    for change in ({"rule.value": "path"}, {"estimator.value": "fit"}, {"frame.value": "body"},
                   {"frame.value": "lab"}, {"color.value": "angle"}, {"offset.value": 8.5},
                   {"max-axes.value": 40}, {"theme.value": "dark"}):
        out = update(**change)
        assert "Helical axes" in trace_names(out["fig-scene.figure"]), change
    assert update(**{"theme.value": "dark"})["root.className"] == "fha theme-dark"
    assert n_points(update(**{"max-axes.value": 40})["fig-direction.figure"], "Helical axes") <= 40


def test_blank_number_fields_fall_back_instead_of_failing():
    out = update(**{"step.value": None, "offset.value": None, "max-axes.value": None})
    assert "Helical axes" in trace_names(out["fig-scene.figure"])


def test_a_selection_without_enough_rotation_says_so():
    out = update(**{"frames.value": [0, 200]})
    assert "No axis could be formed" in text_of(out["notes.children"])
    assert out["fig-scene.figure"]["data"] == [] and "No axes" in text_of(out["table.children"])
    assert text_of(out["tiles.children"]).startswith("Axes 0")


def test_out_and_back_recording_keeps_a_sensible_average_axis():
    out = update(**{"file.value": FILE1, "frames.value": [0, 2401]})
    tiles = text_of(out["tiles.children"])
    angle = float(tiles.split("Mean angle to average axis")[1].split()[0])
    assert angle < 1.0, f"a pure hinge moved out and back should give a tight bundle, got {angle} deg"


def built_up(out):
    return float(text_of(out["tiles.children"]).split("Built up off primary axis")[1].split()[0])


def test_build_up_is_measured_against_the_reference_recording():
    # recording 3 (hinge + second rotation by hand), frames of the outbound movement
    against_hinge = update(**{"frames.value": [100, 1682]})
    assert "1_90deg_ydata.csv" in against_hinge["buildup-info.children"]
    assert 23.0 < built_up(against_hinge) < 24.5, "about 23.8 deg against the hinge of recording 1"
    curve = next(t for t in against_hinge["fig-buildup.figure"]["data"] if t.get("name") == "Measured")
    y = array(curve["y"])
    assert y[-1] > 0.98 * y.max(), "against the hinge the second rotation adds up and stays"

    own = update(**{"frames.value": [100, 1682], "buildup.value": "average"})
    assert "own average axis" in own["buildup-info.children"]
    y = array(next(t for t in own["fig-buildup.figure"]["data"] if t.get("name") == "Measured")["y"])
    assert built_up(own) < 12.0 and y[-1] < 0.7 * y.max(), "against its own average axis it reads low and falls back"


def test_build_up_falls_back_and_says_so_when_the_reference_cannot_be_used():
    for reference, expected in ((None, "No reference recording chosen"), ("/no/such/file.csv", "could not be used")):
        out = update(**{"reference.value": reference})
        assert expected in text_of(out["notes.children"]), reference
        assert "own average axis" in out["buildup-info.children"]


def test_reference_picker_is_shown_only_when_it_applies():
    show = lambda how, source: fire("reference-group.style", **{"buildup.value": how, "source.value": source})["reference-group.style"]
    assert show("reference", "polhemus") == {} and show("reference", "vicon") == {}
    assert show("average", "polhemus") == {"display": "none"}
    assert show("reference", "synthetic") == {"display": "none"}, "the simulator brings its own pure hinge"


def test_simulated_build_up_matches_the_true_second_rotation():
    out = update(**{"source.value": "synthetic", "syn-profile.value": "pulsed", "frames.value": [0, 899]})
    assert "simulated pure hinge" in out["buildup-info.children"]
    names = trace_names(out["fig-buildup.figure"])
    assert "True second rotation" in names and "Measured" in names
    assert abs(built_up(out) - 20.0) < 0.3
    own = update(**{"source.value": "synthetic", "syn-profile.value": "pulsed", "frames.value": [0, 899],
                    "buildup.value": "average"})
    assert built_up(own) < 6.0


def test_simulator_shows_the_true_axis():
    out = update(**{"source.value": "synthetic", "syn-profile.value": "pulsed", "frames.value": [0, 899]})
    assert out["recording-name.children"].startswith("Simulated")
    assert "True axis" in trace_names(out["fig-direction.figure"])
    assert "True axis" in trace_names(out["fig-position.figure"])
    assert "True axis" in trace_names(out["fig-series.figure"])
    spec, o = spec_for(**{"source.value": "synthetic", "syn-return.value": ["return"]})
    assert spec["return_trip"] is True and o["frames.max"] > 1000


def test_vicon_recording_reports_marker_fit_and_missing_markers():
    try:
        import ezc3d  # noqa: F401
    except ImportError:
        print("      (skipped: ezc3d not installed)")
        return
    spec, o = spec_for(**{"source.value": "vicon", "file.value": SERVO})
    out = fire("root.className", **{"spec.data": spec, "frames.value": [0, o["frames.max"]]})
    assert "marker fit" in text_of(out["notes.children"]) and "lengths in mm" in text_of(out["notes.children"])
    assert len(out["fig-overview.figure"]["data"]) == 2, "the residual of the marker fit is plotted"
    cfg = dict(DEFAULTS["cfg.data"], markers_1=["LASI", "RASI", "SACR"])
    spec, _ = spec_for(**{"source.value": "vicon", "file.value": SERVO, "cfg.data": cfg})
    assert "Markers not found" in spec["error"] and "LASI" in spec["error"]


# ------------------------------------------------------------------------- smoothing

def component(node, component_id):
    """The serialised component with this id inside a layout tree, or None."""
    if isinstance(node, list):
        return next((hit for hit in (component(n, component_id) for n in node) if hit), None)
    if not isinstance(node, dict):
        return None
    props = node.get("props", {})
    return node if props.get("id") == component_id else component(props.get("children"), component_id)


def mean_angle(out):
    return float(text_of(out["tiles.children"]).split("Mean angle to average axis")[1].split()[0])


def test_smoothing_slider_offers_the_cutoffs_and_opens_on_the_config_value():
    encoder = __import__("plotly").utils.PlotlyJSONEncoder
    slider = component(json.loads(json.dumps(main.serve_layout(), cls=encoder)), "smoothing")["props"]
    assert slider["value"] == 0 and slider["min"] == 0 and slider["max"] == len(main.fv.SMOOTHING_CUTOFFS) == 15
    labels = [slider["marks"][str(i)] for i in range(16)]
    assert labels[0] == "Off" and labels[1] == "20" and labels[5] == "6" and labels[-1] == "0.2"
    assert main.smoothing_position(6) == 5 and main.smoothing_cutoff(5) == 6.0
    assert main.smoothing_position(0) == 0 and main.smoothing_cutoff(0) is None and main.smoothing_cutoff(None) is None
    assert main.smoothing_cutoff(main.smoothing_position(5.6)) == 6.0, "a value in between opens on the nearest"
    CFG["smoothing"] = 2.0
    try:
        opened = component(json.loads(json.dumps(main.serve_layout(), cls=encoder)), "smoothing")["props"]["value"]
    finally:
        CFG["smoothing"] = 0.0
    assert main.smoothing_cutoff(opened) == 2.0


def test_smoothing_off_leaves_everything_as_measured():
    out = update()
    assert "not smoothed" in text_of(out["notes.children"])
    assert re.match(r"Off: the poses are used as measured\. Their frame-to-frame jitter is 0\.00[23]\d?° and 0\.000\d+ cm\.",
                    out["smoothing-info.children"]), out["smoothing-info.children"]
    for fig in ("fig-direction.figure", "fig-position.figure", "fig-series.figure"):
        assert "Without smoothing" not in trace_names(out[fig]), fig
    assert out["fig-removed.figure"]["data"] == [], "nothing was removed, so there is nothing to plot"
    assert len(out["fig-sweep.figure"]["data"]) == 2, "the jitter line and what each cut-off would remove"
    curve = out["fig-sweep.figure"]["data"][-1]
    assert len(array(curve["x"])) == 15 and np.all(np.diff(array(curve["y"])) > 0), "a heavier filter removes more"


def test_smoothing_reports_what_it_removed_and_keeps_the_unsmoothed_axes_in_view():
    outbound = {"frames.value": [100, 1682]}
    plain = update(**outbound)
    out = update(**outbound, **{"smoothing.value": main.smoothing_position(2)})
    assert "low-pass filtered at 2 Hz" in text_of(out["notes.children"])
    info = out["smoothing-info.children"]
    assert re.match(r"Low-pass at 2 Hz removes 0\.04\d° and 0\.00\d+ cm \(RMS\)", info), info
    assert re.search(r", 1\d times their frame-to-frame jitter \(0\.00[23]\d?°\)", info) and "cannot tell which" in info, info
    for fig in ("fig-direction.figure", "fig-position.figure", "fig-series.figure"):
        assert "Without smoothing" in trace_names(out[fig]), fig
    removed = array(out["fig-removed.figure"]["data"][-1]["y"])
    assert len(removed) == 1583 and 0.03 < np.sqrt(np.mean(removed**2)) < 0.05
    assert len(out["fig-sweep.figure"]["data"]) == 3, "the chosen cut-off is marked on the curve"
    # the map keeps its frame while the slider moves, so the two can be compared
    reach = lambda o: o["fig-direction.figure"]["layout"]["xaxis"]["range"][1]
    assert abs(reach(out) - reach(plain)) < 0.03 * reach(plain)
    assert abs(built_up(out) - built_up(plain)) < 0.1, "the build-up does not depend on the smoothing"
    assert abs(mean_angle(out) - mean_angle(plain)) < 0.1, "nor does the average tilt: the fingers are real movement"
    light = update(**outbound, **{"smoothing.value": main.smoothing_position(20)})
    assert "no more than their frame-to-frame jitter" in light["smoothing-info.children"]


def test_smoothing_tightens_a_noisy_hinge():
    sim = {"source.value": "synthetic", "syn-profile.value": "none", "syn-noise.value": "vicon"}
    spec, o = spec_for(**sim)
    whole = {"frames.value": [0, o["frames.max"]]}
    raw = mean_angle(update(spec=spec, **whole))
    smoothed = mean_angle(update(spec=spec, **whole, **{"smoothing.value": main.smoothing_position(1)}))
    assert smoothed < 0.75 * raw, (raw, smoothed)


def test_low_frame_rates_such_as_dynamic_ct_are_handled():
    fv = main.fv
    full = rc.load_synthetic("proportional", "polhemus")
    keep = slice(None, None, 30)                                     # 120 Hz -> 4 frames a second
    slow = rc.Recording("every 30th frame", "synthetic", full.t[keep], full.T1[keep], full.T2[keep], "cm")
    assert 25 <= len(slow) <= 35
    too_high = fv.analyse(slow, smoothing_hz=6.0)
    assert not too_high.smoothing["active"] and too_high.raw is None
    assert "above half the frame rate of this recording (2 Hz)" in main.smoothing_info(too_high)
    assert len(fv.figure_smoothing_removed(too_high).data) == 0
    curve = fv.figure_smoothing_sweep(too_high).data[-1]
    assert max(curve.x) < 2.0, "only cut-offs below half the frame rate are drawn"
    light = fv.analyse(slow, smoothing_hz=1.5)
    assert light.smoothing["active"] and light.smoothing["rms_angle"] > 0
    assert len(light.fha) > 10, "the frames are about 4 deg apart; windows are as narrow as that allows"
    assert any("a window cannot end between two frames" in note for note in light.notes)
    assert light.raw is not None and "Low-pass at 1.5 Hz removes" in main.smoothing_info(light)
    for figure in (fv.figure_direction_map, fv.figure_series, fv.figure_smoothing_removed, fv.figure_smoothing_sweep):
        assert len(figure(light).data) >= 1, figure.__name__
    assert "jitter" not in main.smoothing_info(light), "too few frames per second to measure it: not claimed"


def test_a_recording_that_cannot_be_smoothed_says_so_and_still_opens():
    fv = main.fv
    full = rc.load_synthetic("proportional", "polhemus")
    stamps = full.t.copy()
    stamps[300:] -= 0.5                                              # time runs backwards once
    broken = rc.Recording("bad clock", "synthetic", stamps, full.T1, full.T2, "cm")
    plain = fv.analyse(broken)
    assert len(plain.fha) > 100, "without smoothing the timestamps do not matter"
    message = fv.figure_smoothing_sweep(plain).layout.annotations[0].text
    assert "Sample times must increase" in message
    try:
        fv.analyse(broken, smoothing_hz=2.0)
    except ValueError as exc:
        assert "Sample times must increase" in str(exc)
    else:
        raise AssertionError("smoothing on a broken time base must not pass silently")


# -------------------------------------------------------------------------- download

def test_download_contains_every_axis_with_its_description():
    spec, _ = spec_for()
    out = fire("download.data", **{"spec.data": spec})
    data = out["download.data"]
    assert data["filename"].endswith("_incremental_angle_axes.csv") and data["type"] == "text/csv"
    lines = data["content"].splitlines()
    assert lines[0].startswith("# recording: 3_90deg") and any(l.startswith("# pivot:") for l in lines)
    header = next(l for l in lines if not l.startswith("#")).split(",")
    assert any(l.startswith("# primary axis for the build-up") and "1_90deg_ydata.csv" in l for l in lines)
    for col in ("time_s", "frame", "tilt_deg", "distance", "n_x", "point_z", "weight", "about_primary_deg",
                "off_primary_deg"):
        assert col in header, col
    rows = [l for l in lines if not l.startswith("#")][1:]
    assert len(rows) > 900, "all axes, not only the ones drawn"
    assert all(len(r.split(",")) == len(header) for r in rows)
    assert "# smoothing: none" in lines
    smoothed = fire("download.data", **{"spec.data": spec, "smoothing.value": main.smoothing_position(2)})
    note = next(l for l in smoothed["download.data"]["content"].splitlines() if l.startswith("# smoothing"))
    assert "low-pass at 2 Hz" in note and "removed 0.0" in note and "screw coordinates" in note


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - report every failure, then exit non-zero
            failed += 1
            import traceback
            print(f"FAIL  {name}: {type(exc).__name__}: {str(exc)[:500]}")
            traceback.print_exc(limit=4)
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
