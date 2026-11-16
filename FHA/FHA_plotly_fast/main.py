"""Helical axis dashboard.

    python main.py            then open http://127.0.0.1:8050

Pick a recording, a frame range and a method at the top; everything below follows.
config.py sets what the page opens with — edit it and press "Reload config.py".
"""
import functools
import importlib
import inspect
import json
import os

import numpy as np
from dash import Dash, Input, Output, State, dcc, html, no_update

import config
import FHA_visualizer_fast as fv
import recordings as rc

SOURCES = {"polhemus": "Polhemus", "vicon": "Vicon", "synthetic": "Simulator"}
DATA_TYPES = {"1": "polhemus", "2": "vicon", "3": "synthetic"}
RULES = {"net": "Net rotation between the ends", "path": "Summed frame-to-frame angles (ISB 2025)"}
ESTIMATORS = {"endpoints": "Two end poses (classical FHA)", "fit": "All frames in the window (least squares)"}
FRAMES = {"auto": "Automatic", "lab": "Lab", "body": "Reference sensor"}
BUILD_UP = {"reference": "The axis of a reference recording", "average": "This recording's own average axis"}
GRAPH_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ------------------------------------------------------------------- configuration

def read_config():
    """config.py as a dict, re-read from disk, with defaults for anything missing."""
    importlib.reload(config)
    get = lambda name, default: getattr(config, name, default)
    return dict(
        source=DATA_TYPES.get(str(get("data_type", "1")), "polhemus"),
        path=get("path", ""),
        polhemus_unit=get("polhemus_unit", "cm"),
        markers_1=[get("marker1_j1", ""), get("marker2_j1", ""), get("marker3_j1", "")] + list(get("extra_markers_j1", [])),
        markers_2=[get("marker1_j2", ""), get("marker2_j2", ""), get("marker3_j2", "")] + list(get("extra_markers_j2", [])),
        vicon_frames=get("vicon_frames", "least_squares"),
        method=get("method_type", "incremental_angle"),
        cut=(int(get("cut1", 0)), int(get("cut2", 10**9))),
        step=float(get("step", 10)),
        rule=get("window_rule", "net"),
        estimator=get("estimator", "endpoints"),
        frame=get("display_frame", "auto"),
        offset=float(get("plane_offset", 0.0)),
        color=get("color_by", "time"),
        max_axes=int(get("max_axes", 300)),
        theme=get("theme", "light"),
        sim_profile=get("sim_profile", "proportional"),
        sim_noise=get("sim_noise", "polhemus"),
        sim_out_and_back=bool(get("sim_out_and_back", False)),
        build_up=get("build_up_axis", "reference"),
        reference_path=get("reference_path", ""),
        smoothing=float(get("smoothing_hz", 0) or 0),
    )


@functools.lru_cache(maxsize=8)
def _load(spec_json, _mtime):
    spec = json.loads(spec_json)
    if spec["source"] == "polhemus":
        return rc.load_polhemus(spec["path"], spec.get("unit", "cm"))
    if spec["source"] == "vicon":
        return rc.load_vicon(spec["path"], spec["markers_1"], spec["markers_2"], spec["frames"])
    return rc.load_synthetic(spec["profile"], spec["noise"], return_trip=spec["return_trip"])


def get_recording(spec):
    """Load (or fetch from cache) the recording a spec describes. Raises on bad input."""
    path = spec.get("path")
    if spec["source"] != "synthetic":
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path or '(no file chosen)'}")
        mtime = os.path.getmtime(path)
    else:
        mtime = 0
    return _load(json.dumps(spec, sort_keys=True), mtime)


@functools.lru_cache(maxsize=8)
def _primary_axis(spec_json, _mtime):
    return tuple(fv.primary_axis_of(_load(spec_json, _mtime)))


@functools.lru_cache(maxsize=8)
def _sweep(spec_json, _mtime):
    return fv.smoothing_sweep(_load(spec_json, _mtime))


def get_sweep(spec):
    """What every cut-off on the slider would remove from a recording (cached)."""
    mtime = os.path.getmtime(spec["path"]) if spec["source"] != "synthetic" else 0
    return _sweep(json.dumps(spec, sort_keys=True), mtime)


def smoothing_position(hz):
    """Slider position for a cut-off in Hz: 0 is off, then one position per entry of
    ``fv.SMOOTHING_CUTOFFS`` (the nearest one on a log scale)."""
    if not hz or hz <= 0:
        return 0
    return 1 + int(np.argmin(np.abs(np.log(np.asarray(fv.SMOOTHING_CUTOFFS)) - np.log(hz))))


def smoothing_cutoff(position):
    """Cut-off in Hz for a slider position; ``None`` when the slider is at Off."""
    position = int(position or 0)
    return None if position <= 0 else float(fv.SMOOTHING_CUTOFFS[min(position, len(fv.SMOOTHING_CUTOFFS)) - 1])


def reference_spec(spec, reference_path):
    """The recording whose average axis serves as the primary axis: for the simulator a
    pure hinge on the same rig, otherwise the chosen file read the same way as the
    recording under study."""
    if spec["source"] == "synthetic":
        return dict(spec, profile="none", return_trip=False)
    if not reference_path:
        return None
    return dict(spec, path=reference_path)


def primary_axis(spec, rec, how, reference_path):
    """Axis to measure the build-up against: ``(axis or None, label, notes)``."""
    if how != "reference":
        return None, None, []
    ref_spec = reference_spec(spec, reference_path)
    if ref_spec is None:
        return None, None, ["No reference recording chosen, so the build-up is measured against this "
                            "recording's own average axis."]
    try:
        ref = get_recording(ref_spec)
        mtime = os.path.getmtime(ref_spec["path"]) if ref_spec["source"] != "synthetic" else 0
        axis = np.array(_primary_axis(json.dumps(ref_spec, sort_keys=True), mtime))
    except Exception as exc:
        return None, None, [f"The reference recording could not be used ({type(exc).__name__}: {exc}); "
                            "the build-up is measured against this recording's own average axis."]
    notes = []
    shift = fv.reference_sensor_shift(rec, ref)
    if shift is not None and ref_spec["source"] != "synthetic":
        turn, move = shift
        spread = float(np.linalg.norm(rec.T2[:, :3, 3] - rec.T1[:, :3, 3], axis=1).mean())
        if turn > 1.0 or move > 0.05 * spread:
            notes.append(f"The reference sensor sits {turn:.1f}° and {move:.2g} {rec.unit} away from where it was in "
                         f"{ref.label}. If it was re-mounted in between, that recording's axis does not carry over.")
    label = "a simulated pure hinge on the same rig" if ref_spec["source"] == "synthetic" else ref.label
    return axis, label, notes


def reference_options(source, cfg):
    options = [{"label": label, "value": path} for label, path in rc.discover().get(source, [])]
    known = {o["value"] for o in options}
    ref = cfg["reference_path"]
    if ref and ref not in known and os.path.isfile(ref) and ref.lower().endswith(".c3d") == (source == "vicon"):
        options.insert(0, {"label": os.path.basename(ref) + " (config.py)", "value": ref})
        known.add(ref)
    return options, (ref if ref in known else None)


def file_options(source, cfg):
    found = rc.discover().get(source, [])
    options = [{"label": label, "value": path} for label, path in found]
    known = {o["value"] for o in options}
    if cfg["source"] == source and cfg["path"] and cfg["path"] not in known and os.path.isfile(cfg["path"]):
        options.insert(0, {"label": os.path.basename(cfg["path"]) + " (config.py)", "value": cfg["path"]})
    default = cfg["path"] if cfg["source"] == source and os.path.isfile(cfg["path"]) else (
        options[0]["value"] if options else None)
    return options, default


# --------------------------------------------------------------------------- layout

def field(label, control, width=None, **kw):
    style = {"width": f"{width}px"} if width else {}
    return html.Div([html.Label(label, className="field-label", **kw), control], className="field", style=style)


def card(title, caption, body, class_name=""):
    return html.Section([html.H2(title, className="card-title"), html.P(caption, className="card-caption"), body],
                        className=f"card {class_name}".strip())


def smoothing_slider(cfg):
    marks = {0: "Off"}
    marks.update({i + 1: f"{c:g}" for i, c in enumerate(fv.SMOOTHING_CUTOFFS)})
    # Dash 4 adds a number box next to a slider; it would show the position, not the cut-off
    extra = {"allow_direct_input": False} if "allow_direct_input" in inspect.signature(dcc.Slider).parameters else {}
    return dcc.Slider(id="smoothing", min=0, max=len(fv.SMOOTHING_CUTOFFS), step=1, marks=marks, included=True,
                      value=smoothing_position(cfg["smoothing"]), updatemode="mouseup", **extra)


def serve_layout():
    cfg = read_config()
    options, default_file = file_options(cfg["source"], cfg)
    ref_options, default_ref = reference_options(cfg["source"], cfg)
    dropdown = dict(clearable=False, searchable=False)
    controls = html.Div([
        html.Div([
            html.Div("Recording", className="group-title"),
            html.Div([
                field("Source", dcc.RadioItems(id="source", value=cfg["source"], className="segmented",
                                               options=[{"label": v, "value": k} for k, v in SOURCES.items()])),
                html.Div([
                    field("File", dcc.Dropdown(id="file", options=options, value=default_file, clearable=False,
                                               optionHeight=32), width=330),
                    field("…or a path", dcc.Input(id="custom-path", type="text", debounce=True, value="",
                                                 placeholder="/path/to/recording.csv", className="text-input"),
                          width=230),
                ], id="file-group", className="row"),
                html.Div([
                    field("Movement", dcc.Dropdown(id="syn-profile", value=cfg["sim_profile"],
                                                   options=[{"label": v, "value": k} for k, v in rc.SYNTHETIC_PROFILES.items()],
                                                   **dropdown), width=330),
                    field("Measurement error", dcc.Dropdown(id="syn-noise", value=cfg["sim_noise"],
                                                            options=[{"label": v, "value": k} for k, v in rc.SYNTHETIC_NOISE.items()],
                                                            **dropdown), width=200),
                    field("Path", dcc.Checklist(id="syn-return", value=["return"] if cfg["sim_out_and_back"] else [],
                                                className="check",
                                                options=[{"label": "out and back", "value": "return"}])),
                ], id="syn-group", className="row"),
            ], className="row"),
        ], className="group"),
        html.Div([
            html.Div("Frames", className="group-title"),
            html.Div(dcc.RangeSlider(id="frames", min=0, max=100, step=1, value=[0, 100], allowCross=False,
                                     updatemode="mouseup", marks=None,
                                     tooltip={"placement": "bottom", "always_visible": True}),
                     className="slider"),
        ], className="group grow"),
        html.Div([
            html.Div("Smoothing", className="group-title"),
            html.Div([
                html.Div([html.Label("Low-pass cut-off (Hz) — further right smooths more", className="field-label"),
                          html.Div(smoothing_slider(cfg), className="slider marked")], className="field"),
                html.Div(id="smoothing-info", className="readout"),
            ], className="smoothing-row"),
        ], className="group full"),
        html.Div([
            html.Div("Method", className="group-title"),
            html.Div([
                field("Axes from", dcc.Dropdown(id="method", value=cfg["method"],
                                                options=[{"label": v, "value": k} for k, v in fv.METHOD_LABELS.items()],
                                                **dropdown), width=230),
                field(html.Span("Window (deg)", id="step-label"),
                      dcc.Input(id="step", type="number", min=0.5, step=0.5, value=cfg["step"], debounce=True,
                                className="number-input"), width=110),
                field("A window closes on", dcc.Dropdown(id="rule", value=cfg["rule"],
                                                         options=[{"label": v, "value": k} for k, v in RULES.items()],
                                                         **dropdown), width=300),
                field("Axis estimated from", dcc.Dropdown(id="estimator", value=cfg["estimator"],
                                                          options=[{"label": v, "value": k} for k, v in ESTIMATORS.items()],
                                                          **dropdown), width=300),
            ], className="row"),
        ], className="group"),
        html.Div([
            html.Div("View", className="group-title"),
            html.Div([
                field("Frame", dcc.Dropdown(id="frame", value=cfg["frame"],
                                            options=[{"label": v, "value": k} for k, v in FRAMES.items()],
                                            **dropdown), width=170),
                field("Colour shows", dcc.RadioItems(id="color", value=cfg["color"], className="segmented",
                                                     options=[{"label": "Time", "value": "time"},
                                                              {"label": "Rotation", "value": "angle"}])),
                field("Plane offset", dcc.Input(id="offset", type="number", step="any", value=cfg["offset"],
                                                debounce=True, className="number-input"), width=110),
                field("Axes drawn", dcc.Input(id="max-axes", type="number", min=10, max=5000, step=10,
                                              value=cfg["max_axes"], debounce=True, className="number-input"),
                      width=110),
                field("Theme", dcc.RadioItems(id="theme", value=cfg["theme"], className="segmented",
                                              options=[{"label": "Light", "value": "light"},
                                                       {"label": "Dark", "value": "dark"}])),
            ], className="row"),
        ], className="group"),
        html.Div([
            html.Div("Build-up", className="group-title"),
            html.Div([
                field("Measured against", dcc.Dropdown(id="buildup", value=cfg["build_up"],
                                                       options=[{"label": v, "value": k} for k, v in BUILD_UP.items()],
                                                       **dropdown), width=280),
                html.Div(field("Reference recording (the primary rotation alone)",
                               dcc.Dropdown(id="reference", options=ref_options, value=default_ref, clearable=True,
                                            optionHeight=32, placeholder="choose a recording"), width=330),
                         id="reference-group"),
            ], className="row"),
        ], className="group"),
    ], className="controls")

    graph = lambda gid, height: dcc.Graph(id=gid, config=GRAPH_CONFIG, style={"height": f"{height}px"})
    return html.Div([
        dcc.Store(id="spec", data=dict(cfg=cfg)),
        dcc.Store(id="cfg", data=cfg),
        dcc.Download(id="download"),
        html.Header([
            html.Div([html.H1("Helical axis explorer"), html.Div(id="recording-name", className="subtitle")]),
            html.Div([
                html.A("Reload config.py", href="/", className="button ghost",
                       title="Re-read config.py and reset every control to it"),
                html.Button("Download axes (CSV)", id="download-btn", className="button", n_clicks=0),
            ], className="header-actions"),
        ], className="page-header"),
        controls,
        html.Div(id="notes", className="notes"),
        html.Div(id="tiles", className="tiles"),
        html.Div(id="scale", className="scale"),
        html.Div([
            card("Helical axes in space",
                 "Each line is the axis of one window; the thick line is their average, the dot on it the pivot. "
                 "Drag to rotate.", graph("fig-scene", 700), "span-scene"),
            html.Div([
                card("Direction map",
                     "Which way each axis points: distance from the centre is its angle to the average axis.",
                     graph("fig-direction", 340)),
                card("Position map",
                     "Where each axis crosses the plane. With the plane through the pivot this is as tight as it gets.",
                     graph("fig-position", 340)),
            ], className="stack"),
        ], className="grid-main"),
        html.Div([
            card("How the second rotation builds up",
                 "Rotation away from the primary axis, added up since the first selected frame. A burst of second "
                 "rotation is a step that stays; in the direction map the same burst is a finger that comes back.",
                 html.Div([
                     html.Div([graph("fig-buildup", 340), graph("fig-swing", 340)], className="pair"),
                     html.P(id="buildup-info", className="card-note"),
                 ])),
            card("Along the movement",
                 "How far each axis is from the average, and the rotation its window really spans.",
                 graph("fig-series", 470)),
        ], className="grid-two"),
        card("What the smoothing takes out",
             "A filter cannot tell noise from movement: it removes whatever changes faster than its cut-off.",
             html.Div([
                 html.Div([html.P("What each cut-off would remove from the selected frames; the line is their "
                                  "frame-to-frame jitter. Where the curve runs flat, little in the recording changes "
                                  "at that rate: a cut-off there splits quick from slow, and the exact value hardly "
                                  "matters. Where it climbs steadily, every cut-off is a compromise.",
                                  className="card-note"), graph("fig-sweep", 280)]),
                 html.Div([html.P("What the chosen cut-off removes at each frame. Fine, rapid grass is jitter or "
                                  "vibration. Smooth humps that rise and fall with the movement are the movement "
                                  "itself being flattened.",
                                  className="card-note"), graph("fig-removed", 280)]),
             ], className="pair even")),
        card("Recording", "The whole recording. The shaded band is what is analysed; move it with the Frames slider.",
             dcc.Graph(id="fig-overview", config=GRAPH_CONFIG)),
        card("Axes as a table", "The axes drawn above, one per row. The CSV download has all of them, with their "
             "directions and points.", html.Div(id="table", className="table-wrap")),
    ], id="root", className=f"fha theme-{cfg['theme']}")


app = Dash(__name__, title="Helical axis explorer")
app.layout = serve_layout


# ------------------------------------------------------------------------ callbacks

@app.callback(Output("file", "options"), Output("file", "value"), Output("file-group", "style"),
              Output("syn-group", "style"), Output("reference", "options"), Output("reference", "value"),
              Input("source", "value"), State("cfg", "data"))
def choose_source(source, cfg):
    hide, show = {"display": "none"}, {}
    if source == "synthetic":
        return no_update, no_update, hide, show, no_update, no_update
    options, default = file_options(source, cfg)
    ref_options, default_ref = reference_options(source, cfg)
    return options, default, show, hide, ref_options, default_ref


@app.callback(Output("reference-group", "style"), Input("buildup", "value"), Input("source", "value"))
def show_reference_picker(how, source):
    """The simulator supplies its own reference (a pure hinge on the same rig)."""
    return {} if how == "reference" and source != "synthetic" else {"display": "none"}


@app.callback(Output("spec", "data"), Output("frames", "max"), Output("frames", "value"),
              Input("source", "value"), Input("file", "value"), Input("custom-path", "value"),
              Input("syn-profile", "value"), Input("syn-noise", "value"), Input("syn-return", "value"),
              State("cfg", "data"))
def choose_recording(source, file, custom, profile, noise, back, cfg):
    if source == "synthetic":
        spec = dict(source=source, profile=profile, noise=noise, return_trip="return" in (back or []))
    else:
        path = (custom or "").strip() or file
        spec = dict(source=source, path=path)
        if source == "vicon":
            spec.update(markers_1=cfg["markers_1"], markers_2=cfg["markers_2"], frames=cfg["vicon_frames"])
        else:
            spec.update(unit=cfg["polhemus_unit"])
    try:
        rec = get_recording(spec)
    except Exception as exc:  # shown to the user in the notes line
        return dict(error=f"{type(exc).__name__}: {exc}"), no_update, no_update
    last = len(rec) - 1
    from_config = source == cfg["source"] and spec.get("path") == cfg["path"]
    value = [min(cfg["cut"][0], last - 2), min(cfg["cut"][1], last)] if from_config else [0, last]
    return spec, last, value


# Everything the main callback fills, in the order Dash expects it back.
OUTPUTS = [
    ("root", "className"), ("recording-name", "children"), ("notes", "children"), ("tiles", "children"),
    ("scale", "children"), ("smoothing-info", "children"),
    ("fig-scene", "figure"), ("fig-direction", "figure"), ("fig-position", "figure"),
    ("fig-buildup", "figure"), ("fig-swing", "figure"), ("buildup-info", "children"),
    ("fig-series", "figure"), ("fig-sweep", "figure"), ("fig-removed", "figure"), ("fig-overview", "figure"),
    ("table", "children"), ("step-label", "children"), ("step", "disabled"),
]


def reply(values):
    """The callback's return value from ``{"id.property": value}``; anything not named
    stays as it is on the page."""
    unknown = set(values) - {f"{cid}.{prop}" for cid, prop in OUTPUTS}
    assert not unknown, unknown
    return tuple(values.get(f"{cid}.{prop}", no_update) for cid, prop in OUTPUTS)


@app.callback(
    *[Output(cid, prop) for cid, prop in OUTPUTS],
    Input("spec", "data"), Input("frames", "value"), Input("method", "value"), Input("step", "value"),
    Input("rule", "value"), Input("estimator", "value"), Input("frame", "value"), Input("color", "value"),
    Input("offset", "value"), Input("max-axes", "value"), Input("theme", "value"),
    Input("buildup", "value"), Input("reference", "value"), Input("smoothing", "value"))
def update(spec, frames, method, step, rule, estimator, frame, color, offset, max_axes, theme, how, reference,
           smoothing):
    theme = theme if theme in fv.THEMES else "light"
    always = {"root.className": f"fha theme-{theme}", "step.disabled": method == "all_FHA",
              "step-label.children": "Lag (frames)" if method == "incremental_time" else "Window (deg)"}
    if not spec or "source" not in spec:
        message = (spec or {}).get("error")
        return reply({**always, "notes.children": [html.Div(message, className="note error")] if message else []})
    try:
        rec = get_recording(spec)
        step = float(step) if step else 10.0
        axis, axis_label, axis_notes = primary_axis(spec, rec, how, reference)
        an = fv.analyse(rec, (int(frames[0]), int(frames[1]) + 1), method, step, estimator, rule, frame,
                        float(offset or 0.0), color, int(max_axes or 300), axis, axis_label,
                        smoothing_hz=smoothing_cutoff(smoothing), sweep=get_sweep(spec))
    except Exception as exc:  # shown in the notes line; what is on screen stays
        return reply({**always, "notes.children": [html.Div(f"{type(exc).__name__}: {exc}", className="note error")]})

    notes = [html.Div(text, className="note") for text in an.notes + axis_notes]
    s = an.smoothing
    filtered = f"low-pass filtered at {s['cutoff']:g} Hz" if s["active"] else "not smoothed"
    if an.empty:
        notes.insert(0, html.Div("No axis could be formed: the joint turns less than one window in the selected "
                                 "frames. Widen the frame range or reduce the window.", className="note error"))
    else:
        where = "the lab frame" if an.frame == "lab" else "the frame of the reference sensor"
        notes.insert(0, html.Div(f"{fv.METHOD_LABELS[method]} · axes shown in {where} · lengths in {rec.unit} · "
                                 f"{filtered}.", className="note"))
    tiles = [html.Div([html.Div(label, className="tile-label"),
                       html.Div([html.Span(value, className="tile-value"),
                                 html.Span(unit, className="tile-unit")])], className="tile")
             for label, value, unit in fv.summary(an)]
    return reply({
        **always, "recording-name.children": rec.label, "notes.children": notes, "tiles.children": tiles,
        "scale.children": scale_legend(an, theme), "smoothing-info.children": smoothing_info(an),
        "fig-scene.figure": fv.figure_scene(an, theme), "fig-direction.figure": fv.figure_direction_map(an, theme),
        "fig-position.figure": fv.figure_position_map(an, theme), "fig-buildup.figure": fv.figure_buildup(an, theme),
        "fig-swing.figure": fv.figure_swing_map(an, theme), "buildup-info.children": buildup_info(an),
        "fig-series.figure": fv.figure_series(an, theme), "fig-sweep.figure": fv.figure_smoothing_sweep(an, theme),
        "fig-removed.figure": fv.figure_smoothing_removed(an, theme),
        "fig-overview.figure": fv.figure_overview(rec, an.cut, theme), "table.children": table(an),
    })


def smoothing_info(an):
    """One or two sentences under the slider: what the filter is doing to these frames."""
    s, unit = an.smoothing, an.rec.unit
    jitter, jitter_len = s["jitter"]
    known = bool(np.isfinite(jitter) and jitter > 0)
    if s["cutoff"] is None:
        return "Off: the poses are used as measured." + (
            f" Their frame-to-frame jitter is {jitter:.2g}° and {jitter_len:.2g} {unit}." if known else "")
    if not s["active"]:
        return (f"{s['cutoff']:g} Hz is above half the frame rate of this recording ({s['rate'] / 2:.3g} Hz), "
                "so nothing is removed.")
    text = (f"Low-pass at {s['cutoff']:g} Hz removes {s['rms_angle']:.2g}° and {s['rms_distance']:.2g} {unit} (RMS) "
            "from the selected frames")
    if not known:
        return text + "."
    ratio = s["rms_angle"] / jitter
    if ratio <= 1.5:
        return text + f": no more than their frame-to-frame jitter ({jitter:.2g}°)."
    times = f"{ratio:.1f}" if ratio < 10 else f"{ratio:.0f}"
    return text + (f", {times} times their frame-to-frame jitter ({jitter:.2g}°). What goes beyond the jitter is "
                   "slower tracker error, real movement, or both; a filter cannot tell which.")


def buildup_info(an):
    """One line saying which axis the build-up is measured against, and what that means."""
    if an.empty:
        return ""
    b = an.build
    if b["own"]:
        return ("Measured against this recording's own average axis. That axis is the direction of the whole "
                "rotation, second rotation included, so the build-up reads too low and drifts back toward zero. "
                "To see the second rotation add up, measure against the axis of a recording of the primary "
                "rotation alone.")
    return (f"Measured against the average axis of {b['label']}, which lies {b['angle_to_average']:.1f}° from "
            "this recording's own average axis. Left: how far the moving segment has turned away from that axis. "
            "Right: which way, seen from a frame that turns with the primary rotation.")


def scale_legend(an, theme):
    """One shared key for the colour ramp used in the 3D view and both maps."""
    if an.empty:
        return []
    ramp = fv.THEMES[theme]["ramp"]
    lo, hi = float(np.nanmin(an.color)), float(np.nanmax(an.color))
    fmt = (lambda v: f"{v:.1f} s") if an.params["color_by"] == "time" else (lambda v: f"{v:.0f}°")
    return [html.Span(f"Colour of the axes: {an.color_label}", className="scale-label"),
            html.Span(fmt(lo), className="scale-end"),
            html.Span(className="scale-bar", style={"background": f"linear-gradient(to right, {', '.join(ramp)})"}),
            html.Span(fmt(hi), className="scale-end")]


def table(an):
    cols = fv.axes_table(an, drawn_only=True)
    if not cols:
        return html.Div("No axes for this selection.", className="empty")
    head = html.Thead(html.Tr([html.Th(title.replace("{unit}", an.rec.unit)) for _, title, _ in fv.TABLE_COLUMNS]))
    rows = []
    for r in range(len(cols["time_s"])):
        rows.append(html.Tr([html.Td(fmt.format(int(cols[key][r]) if "d}" in fmt else float(cols[key][r])))
                             for key, _, fmt in fv.TABLE_COLUMNS]))
    return html.Table([head, html.Tbody(rows)], className="axes-table")


@app.callback(Output("download", "data"), Input("download-btn", "n_clicks"),
              State("spec", "data"), State("frames", "value"), State("method", "value"), State("step", "value"),
              State("rule", "value"), State("estimator", "value"), State("frame", "value"),
              State("offset", "value"), State("buildup", "value"), State("reference", "value"),
              State("smoothing", "value"), prevent_initial_call=True)
def download(_, spec, frames, method, step, rule, estimator, frame, offset, how, reference, smoothing):
    if not spec or "source" not in spec:
        return no_update
    rec = get_recording(spec)
    axis, axis_label, _notes = primary_axis(spec, rec, how, reference)
    an = fv.analyse(rec, (int(frames[0]), int(frames[1]) + 1), method, float(step or 10.0), estimator, rule, frame,
                    float(offset or 0.0), primary_axis=axis, primary_label=axis_label,
                    smoothing_hz=smoothing_cutoff(smoothing))
    name = os.path.splitext(rec.label)[0].replace(" ", "_").replace(":", "").replace(",", "")
    return dict(content=fv.axes_csv(an), filename=f"{name}_{method}_axes.csv", type="text/csv")


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 8050)))
