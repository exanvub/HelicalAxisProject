"""Analysis of one recording and the Plotly figures the dashboard shows.

The numerics live in ``fha_core``; this module turns a ``Recording`` plus a handful of
choices into (1) everything worth looking at — the axes, their average, how they
spread, how the joint's two rotations build up — and (2) figures of it.

Two views replace the single cross-section of the earlier version, which mixed
position and direction (research/FHA_deepdive/FINDINGS.md, finding F11):

* the POSITION map: where each axis pierces the plane through the pivot;
* the DIRECTION map: the tilt of each axis from the average axis, in degrees.

Both show what the joint is doing window by window. The BUILD-UP view is their
counterpart: how much rotation has accumulated about a primary axis and away from it
since the first selected frame. A burst of second rotation shows as a finger in the
direction map and as a step that stays in the build-up (finding F15).

SMOOTHING is optional and sits in front of all of that: a low-pass filter on the motion
of the moving body relative to the reference body (``fha_core.smooth_motion``). A filter
cannot tell measurement noise from quick movement, so the analysis also records what
the filter took out, next to the frame-to-frame jitter of the recording, and keeps the
unsmoothed axes to draw underneath (finding F17).
"""
import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

import fha_core as fc

METHOD_LABELS = {
    "incremental_angle": "Moving window (miFHA)",
    "step_angle": "Consecutive windows (fiFHA)",
    "reference_pose": "From a reference pose",
    "incremental_time": "Fixed time lag",
    "all_FHA": "Consecutive frames",
}

# --------------------------------------------------------------------------- theme
# Colours follow the entity everywhere: blue = the helical axes (a one-hue ramp when it
# encodes time or angle), orange = their average axis and pivot, aqua = the true axis of
# a simulated recording, grey = context. Validated for colour-vision deficiency on both
# surfaces (all pairs; worst pair dE 9.2 light / 9.4 dark).
THEMES = {
    "light": dict(
        surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", axes="#2a78d6", average="#eb6834", truth="#1baf7a",
        ramp=["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
              "#184f95", "#104281", "#0d366b"],
    ),
    "dark": dict(
        surface="#1a1a19", page="#0d0d0d", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835", axes="#3987e5", average="#d95926", truth="#199e70",
        ramp=["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec",
              "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"],
    ),
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Cut-offs offered by the dashboard's smoothing slider, in Hz, from light to heavy.
SMOOTHING_CUTOFFS = (20.0, 15.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2)


def _colorscale(th):
    ramp = th["ramp"]
    return [[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)]


# ------------------------------------------------------------------------ analysis

@dataclass
class Analysis:
    rec: object
    cut: tuple
    params: dict
    fha: fc.FHASet
    frame: str                      # "lab" or "body" (frame of the reference body)
    n: np.ndarray                   # (M, 3) axis directions in the display frame
    p: np.ndarray                   # (M, 3) a point on each axis in the display frame
    n_bar: Optional[np.ndarray] = None
    pivot: Optional[np.ndarray] = None
    plane_point: Optional[np.ndarray] = None
    e1: Optional[np.ndarray] = None
    e2: Optional[np.ndarray] = None
    tilt: Optional[np.ndarray] = None          # (M, 2) direction map, deg
    pos: Optional[np.ndarray] = None           # (M, 2) position map, length units
    disp: Optional[dict] = None
    color: Optional[np.ndarray] = None         # (M,) value encoded by colour
    color_label: str = ""
    twist: Optional[np.ndarray] = None         # (N,) rotation about the average axis, deg
    twist_axes: Optional[np.ndarray] = None    # (M,) the same at the frame of each axis
    build: Optional[dict] = None               # accumulated rotation about / away from the primary axis
    path2: Optional[np.ndarray] = None         # (N, 3) origin of the moving body
    path1: Optional[np.ndarray] = None         # (N, 3) origin of the reference body
    draw: Optional[np.ndarray] = None          # indices of the axes drawn (evenly spaced in angle)
    truth: Optional[dict] = None               # simulator: true axis in the same maps
    rotation: Optional[np.ndarray] = None      # (N_all,) rotation from the first frame of the cut, whole recording
    smoothing: Optional[dict] = None           # what the low-pass filter was and what it took out
    raw: Optional[dict] = None                 # the axes without smoothing, in the same maps (only when smoothing)
    notes: list = field(default_factory=list)

    @property
    def empty(self):
        return len(self.fha) == 0

    @property
    def t(self):
        return self.rec.t[self.cut[0]:self.cut[1]]


def reference_is_static(T1, T2):
    """True when the reference body stays put in the lab: it turns by less than 2 deg and
    drifts by less than 5% of the size of the set-up (the larger of the distance between
    the two sensors and the distance the other one travels)."""
    q = fc.quaternions(T1)
    rot = np.degrees(fc.net_angle(q, q[0][None, :])).max()
    drift = np.ptp(T1[:, :3, 3], axis=0).max()
    travel = np.ptp(T2[:, :3, 3], axis=0).max()
    separation = np.linalg.norm(T2[:, :3, 3] - T1[:, :3, 3], axis=1).mean()
    return rot < 2.0 and drift < 0.05 * max(travel, separation, 1e-12)


def smoothing_sweep(rec, cutoffs=SMOOTHING_CUTOFFS):
    """What each cut-off would take out of ``rec``, frame by frame.

    Returns ``cutoffs`` (K,), ``angle`` (K, N) in degrees and ``distance`` (K, N): how far
    the filter moves the pose of the moving body relative to the reference body. Cut-offs
    at or above half the frame rate remove nothing and give zeros. If the recording
    cannot be smoothed at all, ``error`` says why.

    Pass the result to ``analyse`` as ``sweep`` so it is not recomputed for every change
    of the other settings.
    """
    out = dict(cutoffs=np.asarray(cutoffs, float), angle=np.zeros((len(cutoffs), len(rec))),
               distance=np.zeros((len(cutoffs), len(rec))), error=None)
    if len(rec) < 4:
        return out
    raw = fc.relative_poses(rec.T1, rec.T2)
    rate = 1.0 / float(np.median(np.diff(rec.t)))
    try:
        for k, cutoff in enumerate(out["cutoffs"]):
            if cutoff < 0.5 * rate:
                out["angle"][k], out["distance"][k] = fc.removed_by_smoothing(raw, fc.smooth_poses(rec.t, raw, cutoff))
    except ValueError as exc:
        out["error"] = str(exc)
    return out


def _sweep_rms(sweep, c1, c2):
    """RMS of a ``smoothing_sweep`` over frames ``c1:c2``: one value per cut-off."""
    rms = lambda a: np.sqrt(np.mean(np.square(a[:, c1:c2]), axis=1))
    return dict(cutoffs=sweep["cutoffs"], rms_angle=rms(sweep["angle"]), rms_distance=rms(sweep["distance"]),
                error=sweep.get("error"))


def _evenly_in_rotation(fha, max_axes):
    """Indices of at most ``max_axes`` axes spaced evenly in rotation (cumulative
    averaging weight), not in time."""
    if len(fha) <= max_axes:
        return np.arange(len(fha))
    cum = np.cumsum(fha.weight)
    idx = np.unique(np.searchsorted(cum, np.linspace(cum[0], cum[-1], int(max_axes))))
    return idx[idx < len(fha)]


def analyse(rec, cut=(0, None), method="incremental_angle", step=10.0, estimator="endpoints",
            criterion="net", frame="auto", plane_offset=0.0, color_by="time", max_axes=300,
            primary_axis=None, primary_label=None, smoothing_hz=None, sweep=None):
    """Run the FHA pipeline on ``rec[cut]`` and derive everything the figures need.

    smoothing_hz cut-off of a low-pass filter applied to the motion before anything
                 else, in Hz (-3 dB, no time shift; see ``fha_core.smooth``). ``None``
                 or 0 uses the poses as measured. The whole recording is filtered, then
                 cut, so the result at a frame does not depend on where the cut is.
                 ``Analysis.smoothing`` reports what was removed and ``Analysis.raw``
                 holds the unsmoothed axes for comparison.
    sweep        a ``smoothing_sweep`` of ``rec`` computed earlier (optional).

    primary_axis the axis the build-up is measured against, as a vector in the frame of
                 the reference body — typically the average axis of a recording of the
                 primary rotation alone, made with the same reference sensor mounting
                 (see ``primary_axis_of``). ``None`` uses this recording's own average
                 axis, which understates the build-up: that axis already contains the
                 second rotation. ``primary_label`` names where the axis came from.

    frame        "lab", "body" (the reference body's own frame) or "auto": lab when the
                 reference body is static, otherwise its own frame — axes of a joint are
                 only a stable bundle in the frame of one of its segments.
    plane_offset distance of the cross-section plane from the pivot, along the average
                 axis. 0 puts it through the pivot, where the position map is tightest.
    color_by     "time" or "angle" (rotation about the average axis).
    max_axes     how many axes to draw; they are picked evenly in rotation, not in time.
    """
    c1 = int(max(cut[0] or 0, 0))
    c2 = int(len(rec) if cut[1] is None else min(cut[1], len(rec)))
    if c2 - c1 < 3:
        raise ValueError("Select at least three frames.")
    rate = 1.0 / float(np.median(np.diff(rec.t)))
    cutoff = float(smoothing_hz) if smoothing_hz else None
    smoothed = cutoff is not None and cutoff < 0.5 * rate and len(rec) >= 4
    T1_all, T2_all = fc.smooth_motion(rec.t, rec.T1, rec.T2, cutoff) if smoothed else (rec.T1, rec.T2)
    t, T1, T2 = rec.t[c1:c2], T1_all[c1:c2], T2_all[c1:c2]
    params = dict(method=method, step=float(step), estimator=estimator, criterion=criterion,
                  plane_offset=float(plane_offset), color_by=color_by, smoothing_hz=cutoff if smoothed else None)
    fha = fc.compute_fha(t, T1, T2, method, float(step), estimator, criterion)
    Trel = fc.relative_poses(T1, T2)

    # what the filter took out of the selected frames, and their frame-to-frame jitter
    raw_rel = fc.relative_poses(rec.T1[c1:c2], rec.T2[c1:c2])
    gone = fc.removed_by_smoothing(raw_rel, Trel) if smoothed else (np.zeros(c2 - c1), np.zeros(c2 - c1))
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    smoothing = dict(cutoff=cutoff, active=smoothed, rate=rate, angle=gone[0], distance=gone[1],
                     rms_angle=rms(gone[0]), rms_distance=rms(gone[1]), jitter=fc.jitter(raw_rel),
                     sweep=None if sweep is None else _sweep_rms(sweep, c1, c2))

    static = reference_is_static(T1, T2)
    use = ("lab" if static else "body") if frame == "auto" else frame
    notes = list(rec.notes)
    if frame == "auto" and not static:
        notes.append("The reference sensor moves, so everything is drawn in its own frame.")
    if use == "lab" and not static:
        notes.append("The reference sensor moves: in the lab frame the axes are smeared by that "
                     "movement. Use the reference-body frame.")

    if use == "lab":
        n, p = fc.axes_to_lab(fha, T1)
        path2, path1 = T2[:, :3, 3], T1[:, :3, 3]
    else:
        n, p = fha.n, fha.s
        path2, path1 = Trel[:, :3, 3], np.zeros((len(t), 3))

    q = fc.quaternions(Trel)
    rotation_cut = np.degrees(fc.net_angle(q, q[0][None, :]))
    an = Analysis(rec, (c1, c2), params, fha, use, n, p, path2=path2, path1=path1, notes=notes,
                  rotation=rotation_cut, smoothing=smoothing)
    if an.empty:
        return an

    an.n_bar, an.pivot = fc.average_axis(n, p, fha.weight)
    an.e1, an.e2 = fc.plane_basis(an.n_bar, path2.mean(axis=0) - an.pivot)
    an.plane_point = an.pivot + float(plane_offset) * an.n_bar
    an.tilt = fc.tilt_map(n, an.n_bar, an.e1, an.e2)
    an.pos = fc.cross_section(n, p, an.n_bar, an.plane_point, an.e1, an.e2)
    an.disp = fc.dispersion(n, p, an.n_bar, an.pivot, fha.weight)

    # rotation about the average axis, in the frame of the reference body: used to colour
    # and tabulate the axes
    nb1, piv1 = fc.average_axis(fha.n, fha.s, fha.weight)
    an.twist = fc.swing_twist(Trel, nb1)[0]
    an.twist_axes = an.twist[fha.ic]

    # the build-up: rotation accumulated about the primary axis and away from it
    own = primary_axis is None
    axis = nb1 if own else np.asarray(primary_axis, float) / np.linalg.norm(primary_axis)
    if axis @ nb1 < 0:
        axis = -axis
    twist, swing = fc.swing_twist(Trel, axis, turning=True)
    g1, g2 = fc.plane_basis(axis, Trel[0, :3, 3] - piv1)
    apart = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(axis, nb1)), axis @ nb1)))
    an.build = dict(axis=axis, own=own, label=primary_label, angle_to_average=apart, twist=twist,
                    swing=np.c_[swing @ g1, swing @ g2], magnitude=np.linalg.norm(swing, axis=1), truth=None)
    if rec.truth is not None:
        alpha, beta = rec.truth["alpha"][c1:c2], rec.truth["beta"][c1:c2]
        an.build["truth"] = (alpha - alpha[0], beta - beta[0])

    if color_by == "angle":
        an.color, an.color_label = an.twist_axes, "rotation about the average axis (deg)"
    else:
        an.color, an.color_label = fha.t, "time (s)"

    an.draw = _evenly_in_rotation(fha, max_axes)

    if smoothed:
        # the same axes without smoothing, placed in the maps of the smoothed ones
        unsmoothed = fc.compute_fha(t, rec.T1[c1:c2], rec.T2[c1:c2], method, float(step), estimator, criterion)
        if len(unsmoothed):
            rn, rp = fc.axes_to_lab(unsmoothed, rec.T1[c1:c2]) if use == "lab" else (unsmoothed.n, unsmoothed.s)
            spread = fc.dispersion(rn, rp, an.n_bar, an.pivot, unsmoothed.weight)
            an.raw = dict(fha=unsmoothed, tilt=fc.tilt_map(rn, an.n_bar, an.e1, an.e2),
                          pos=fc.cross_section(rn, rp, an.n_bar, an.plane_point, an.e1, an.e2),
                          angle=spread["angle"], distance=spread["distance"],
                          draw=_evenly_in_rotation(unsmoothed, max_axes))

    if rec.truth is not None:
        tn, ts = rec.truth["n"][c1:c2][fha.ic], rec.truth["s"][c1:c2][fha.ic]
        if use == "body":
            Ti = fc.invert(T1[fha.ic])
            tn = np.einsum("kij,kj->ki", Ti[:, :3, :3], tn)
            ts = np.einsum("kij,kj->ki", Ti[:, :3, :3], ts) + Ti[:, :3, 3]
        ok = np.isfinite(tn).all(axis=1)
        tilt = np.full((len(fha), 2), np.nan)
        pos = np.full((len(fha), 2), np.nan)
        tilt[ok] = fc.tilt_map(tn[ok], an.n_bar, an.e1, an.e2)
        pos[ok] = fc.cross_section(tn[ok], ts[ok], an.n_bar, an.plane_point, an.e1, an.e2)
        an.truth = dict(tilt=tilt, pos=pos, n=tn, s=ts)

    if criterion == "path" and method in ("incremental_angle", "step_angle"):
        short = np.mean(np.degrees(fha.phi) < 0.8 * float(step))
        if short > 0.02:
            notes.append(f"{100 * short:.0f}% of the windows span less than 80% of the requested "
                         f"{step:g} deg: summing frame-to-frame angles counts noise and reversals as rotation.")
    if method == "reference_pose":
        notes.append("Reference-pose axes describe the whole displacement since the first selected "
                     "frame, not the movement at each instant.")
    if method == "incremental_angle" and criterion == "net":
        spans = float(np.median(np.degrees(fha.phi)))
        if spans > 1.25 * float(step):
            notes.append(f"The frames are far apart in rotation, so the windows span about {spans:.0f}° where "
                         f"{step:g}° was asked: a window cannot end between two frames.")
    return an


# ------------------------------------------------------------------ figure helpers

def _layout(th, height, margin=None, legend=False):
    return dict(
        template="none", height=height, paper_bgcolor=th["surface"], plot_bgcolor=th["surface"],
        font=dict(family=FONT, size=12, color=th["ink2"]),
        margin=margin or dict(l=56, r=16, t=12, b=44),
        showlegend=bool(legend),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(color=th["ink2"], size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=th["surface"], bordercolor=th["axis"],
                        font=dict(family=FONT, size=12, color=th["ink"])),
    )


def _axis(th, title=None, **kw):
    ax = dict(showgrid=True, gridcolor=th["grid"], gridwidth=1, zeroline=True, zerolinecolor=th["axis"],
              zerolinewidth=1, showline=False, ticks="", tickfont=dict(color=th["muted"], size=11),
              title=dict(text=title or "", font=dict(color=th["ink2"], size=12), standoff=8),
              automargin=True)
    ax.update(kw)
    return ax


def _empty(th, height, message):
    fig = go.Figure()
    fig.update_layout(**_layout(th, height))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(family=FONT, size=13, color=th["muted"]))
    return fig


def _break_gaps(x, *series):
    """Insert NaN where consecutive samples are far apart in x, so lines do not bridge
    stretches that produced no axis (rest, turn-around)."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return (x,) + tuple(np.asarray(s, float) for s in series)
    dx = np.abs(np.diff(x))
    cut = np.flatnonzero(dx > 4 * np.median(dx[dx > 0]) if np.any(dx > 0) else np.zeros(len(dx), bool)) + 1
    out = []
    for s in (x,) + series:
        out.append(np.insert(np.asarray(s, float), cut, np.nan, axis=0))
    return tuple(out)


def _nice_step(span, target=4):
    """A round tick step giving about ``target`` intervals over ``span``."""
    if not np.isfinite(span) or span <= 0:
        return 1.0
    raw = span / target
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _spaced(xy, min_gap):
    """Indices of points at least ``min_gap`` from the previously kept one (first and
    last always kept).

    Markers carry a ring in the surface colour so that overlapping ones stay legible.
    On a smooth trajectory sampled every frame the points sit a fraction of a pixel
    apart; each ring then paints over the marker before it and the whole run vanishes
    into the background, leaving only its last point. Spacing the markers by about
    their own width on screen avoids that; the path line still follows every point.
    """
    keep = [0]
    for k in range(1, len(xy)):
        if not np.isfinite(xy[k]).all():
            continue
        if np.hypot(*(xy[k] - xy[keep[-1]])) >= min_gap:
            keep.append(k)
    if len(xy) > 1 and keep[-1] != len(xy) - 1 and np.isfinite(xy[-1]).all():
        keep.append(len(xy) - 1)
    return np.asarray(keep, dtype=int)


def _hover_axes(an, idx):
    f = an.fha
    return np.c_[f.t[idx], an.twist_axes[idx], an.disp["angle"][idx], an.disp["distance"][idx]]


def _hover_template(unit):
    return ("t = %{customdata[0]:.2f} s<br>rotation %{customdata[1]:.1f}°"
            "<br>tilt from average axis %{customdata[2]:.2f}°"
            "<br>distance from pivot %{customdata[3]:.3f} " + unit + "<extra></extra>")


def _map(an, th, xy, truth_xy, xlabel, ylabel, height, rings=False, origin_label="average axis",
         hover=None, name="Helical axes", label_start=True, ghost_xy=None):
    """Shared layout of the direction map, the position map and the tilt map of the
    build-up: one point per axis drawn, in the order they occur, coloured like the axes.
    ``hover`` is ``(values per axis, template)`` to replace the default tooltip.
    ``ghost_xy`` places the unsmoothed axes (``an.raw``) in the same map; they are drawn
    as a grey line underneath, and the path of the smoothed ones turns blue."""
    if an.empty:
        return _empty(th, height, "No axes for this selection")
    idx = an.draw
    fig = go.Figure()
    order = np.argsort(an.fha.ic[idx])
    idx = idx[order]
    ghost = None
    if ghost_xy is not None and an.raw is not None:
        raw_ic = an.raw["fha"].ic
        gi = an.raw["draw"][np.argsort(raw_ic[an.raw["draw"]])]
        ghost = ghost_xy[gi]
        gx, gy = _break_gaps(raw_ic[gi], ghost[:, 0], ghost[:, 1])[1:]
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", line=dict(color=th["muted"], width=1.5),
                                 name="Without smoothing", hoverinfo="skip"))
    lx, ly = _break_gaps(an.fha.ic[idx], xy[idx, 0], xy[idx, 1])[1:]
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", hoverinfo="skip", showlegend=False,
                             line=dict(color=th["axis"] if ghost is None else th["axes"], width=1.5)))
    if truth_xy is not None and np.isfinite(truth_xy).any():
        tx, ty = _break_gaps(an.fha.ic[idx], truth_xy[idx, 0], truth_xy[idx, 1])[1:]
        fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines", line=dict(color=th["truth"], width=2),
                                 name="True axis", hoverinfo="skip"))
    finite = np.isfinite(xy[idx]).all(axis=1)
    pts = xy[idx][finite]
    reach = np.abs(pts).max() if len(pts) else 1.0
    if ghost is not None and np.isfinite(ghost).any():
        reach = max(reach, np.nanmax(np.abs(ghost)))      # the frame stays put while the smoothing changes
    reach = max(reach, 1e-9) * 1.15
    # the shorter side of the plot spans 2 * reach in roughly (height - margins) pixels
    mk = idx[_spaced(xy[idx], min_gap=9.0 * 2 * reach / max(height - 70, 100))]
    custom, template = (_hover_axes(an, mk), _hover_template(an.rec.unit)) if hover is None else (hover[0][mk], hover[1])
    fig.add_trace(go.Scatter(
        x=xy[mk, 0], y=xy[mk, 1], mode="markers", name=name,
        marker=dict(size=8, color=an.color[mk], colorscale=_colorscale(th), cmin=np.nanmin(an.color),
                    cmax=np.nanmax(an.color), line=dict(width=1.5, color=th["surface"])),
        customdata=custom, hovertemplate=template))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(symbol="cross-thin", size=14, line=dict(width=2, color=th["average"]))))
    step = _nice_step(reach, 3)
    if rings:
        # label the rings on the side of the map away from the data
        centre = pts.mean(axis=0) if len(pts) else np.zeros(2)
        away = -centre / np.hypot(*centre) if np.hypot(*centre) > 0.05 * reach else np.array([0.0, -1.0])
        for r in np.arange(step, reach * 1.01, step):
            fig.add_shape(type="circle", x0=-r, y0=-r, x1=r, y1=r, line=dict(color=th["grid"], width=1),
                          layer="below")
            fig.add_annotation(x=r * away[0], y=r * away[1], text=f"{r:g}°", showarrow=False,
                               xshift=4 if away[0] >= 0 else -4, yshift=1,
                               xanchor="left" if away[0] >= 0 else "right", yanchor="bottom",
                               font=dict(size=10, color=th["muted"]))
    # "start" and "end" sit beyond the two ends of the path, continuing its direction, so
    # they do not land on the markers they name
    if len(mk) > 1:
        ends = ((xy[mk[0]], xy[mk[0]] - xy[mk[1]], "start"), (xy[mk[-1]], xy[mk[-1]] - xy[mk[-2]], "end"))
        for (x, y), away, text in ends[0 if label_start else 1:]:
            norm = np.hypot(*away)
            ux, uy = (away / norm) if norm > 0 else (0.7071, 0.7071)
            fig.add_annotation(x=x, y=y, text=text, showarrow=True, arrowhead=0, arrowwidth=1,
                               arrowcolor=th["muted"], ax=26 * ux, ay=-26 * uy, standoff=6,
                               font=dict(size=11, color=th["ink2"]))
    fig.add_annotation(x=0, y=0, text=origin_label, showarrow=True, arrowhead=0, arrowwidth=1,
                       arrowcolor=th["muted"], ax=-34, ay=30, standoff=8,
                       font=dict(size=11, color=th["ink2"]))
    legend = bool(truth_xy is not None and np.isfinite(truth_xy).any()) or ghost is not None
    fig.update_layout(**_layout(th, height, legend=legend, margin=dict(l=56, r=16, t=28 if legend else 12, b=44)),
                      hovermode="closest")
    fig.update_xaxes(**_axis(th, xlabel, range=[-reach, reach], showgrid=not rings, dtick=step))
    fig.update_yaxes(**_axis(th, ylabel, range=[-reach, reach], showgrid=not rings, dtick=step,
                             scaleanchor="x", scaleratio=1))
    return fig


def figure_direction_map(an, theme="light", height=340):
    th = THEMES[theme]
    truth = None if an.truth is None else an.truth["tilt"]
    return _map(an, th, an.tilt, truth, "tilt toward the moving sensor (deg)", "tilt across (deg)",
                height, rings=True, ghost_xy=None if an.raw is None else an.raw["tilt"])


def figure_position_map(an, theme="light", height=340):
    th = THEMES[theme]
    unit = an.rec.unit
    truth = None if an.truth is None else an.truth["pos"]
    label = "pivot" if an.params["plane_offset"] == 0 else "average axis"
    return _map(an, th, an.pos, truth, f"toward the moving sensor ({unit})", f"across ({unit})",
                height, origin_label=label, ghost_xy=None if an.raw is None else an.raw["pos"])


def figure_scene(an, theme="light", height=700):
    """3D view: the axes, their average axis and pivot, the cross-section plane, the sensors."""
    th = THEMES[theme]
    if an.empty:
        return _empty(th, height, "No axes for this selection")
    idx = an.draw
    n, p = an.n[idx], an.p[idx]
    reach = np.linalg.norm(an.path2 - an.pivot, axis=1).max()
    # The reference sensor is drawn for orientation, unless it sits so far from the joint
    # that framing it would shrink everything else (a segment frame whose origin is far up
    # the limb, or a lab-fixed sensor across the room).
    ref_dist = np.linalg.norm(an.path1 - an.pivot, axis=1).max()
    show_ref = ref_dist <= 4.0 * reach
    if show_ref:
        reach = max(reach, ref_dist)
    spread = np.nanpercentile(np.linalg.norm(an.pos[idx], axis=1), 95) if len(idx) else 0.0
    length = max(1.2 * reach, 6 * spread, 1e-9)
    # Each axis is drawn as a segment centred on its point nearest the pivot. (Centring on
    # where it crosses the plane would fling a badly tilted axis far away and stretch the
    # whole scene around it.)
    centre = p + np.einsum("ki,ki->k", an.plane_point - p, n)[:, None] * n
    a, b = centre - 0.5 * length * n, centre + 0.5 * length * n
    seg = np.full((3 * len(idx), 3), np.nan)
    seg[0::3], seg[1::3] = a, b
    col = np.repeat(an.color[idx], 3)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines", showlegend=False,
                               line=dict(color=col, colorscale=_colorscale(th), cmin=np.nanmin(an.color),
                                         cmax=np.nanmax(an.color), width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", name="Helical axes",
                               line=dict(color=th["axes"], width=4)))

    half = max(1.6 * spread, 0.12 * length)
    corners = np.array([an.plane_point + sx * half * an.e1 + sy * half * an.e2
                        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    fig.add_trace(go.Mesh3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
                            color=th["muted"], opacity=0.16, hoverinfo="skip", flatshading=True, showlegend=False))
    ring = np.vstack([corners, corners[:1]])
    fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="lines", name="Cross-section plane",
                               line=dict(color=th["muted"], width=2), hoverinfo="skip"))

    on_plane = idx[np.abs(an.pos[idx]).max(axis=1) <= half]          # crossings inside the drawn square
    hit = an.plane_point + an.pos[on_plane, :1] * an.e1 + an.pos[on_plane, 1:] * an.e2
    fig.add_trace(go.Scatter3d(x=hit[:, 0], y=hit[:, 1], z=hit[:, 2], mode="markers", showlegend=False,
                               marker=dict(size=3.5, color=an.color[on_plane], colorscale=_colorscale(th),
                                           cmin=np.nanmin(an.color), cmax=np.nanmax(an.color)),
                               customdata=_hover_axes(an, on_plane), hovertemplate=_hover_template(an.rec.unit)))
    if an.truth is not None and np.isfinite(an.truth["pos"]).any():
        tp = an.truth["pos"][idx].copy()
        tp[np.abs(tp).max(axis=1) > half] = np.nan             # beyond the drawn square: leave a gap
        th3 = an.plane_point + tp[:, :1] * an.e1 + tp[:, 1:] * an.e2
        fig.add_trace(go.Scatter3d(x=th3[:, 0], y=th3[:, 1], z=th3[:, 2], mode="lines", name="True axis (on the plane)",
                                   line=dict(color=th["truth"], width=5), hoverinfo="skip"))

    ends = np.vstack([an.pivot - 0.65 * length * an.n_bar, an.pivot + 0.65 * length * an.n_bar])
    fig.add_trace(go.Scatter3d(x=ends[:, 0], y=ends[:, 1], z=ends[:, 2], mode="lines", name="Average axis",
                               line=dict(color=th["average"], width=7), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[an.pivot[0]], y=[an.pivot[1]], z=[an.pivot[2]], mode="markers", name="Pivot",
                               marker=dict(size=6, color=th["average"], line=dict(width=1, color=th["surface"])),
                               hovertemplate="pivot<br>%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>"))

    fig.add_trace(go.Scatter3d(x=an.path2[:, 0], y=an.path2[:, 1], z=an.path2[:, 2], mode="lines",
                               name="Moving sensor", line=dict(color=th["ink2"], width=4), hoverinfo="skip"))
    if show_ref:
        p1 = an.path1
        moving1 = np.ptp(p1, axis=0).max() > 0.02 * max(reach, 1e-12)
        fig.add_trace(go.Scatter3d(
            x=p1[:, 0] if moving1 else [p1[:, 0].mean()], y=p1[:, 1] if moving1 else [p1[:, 1].mean()],
            z=p1[:, 2] if moving1 else [p1[:, 2].mean()], mode="lines" if moving1 else "markers",
            name="Reference sensor", line=dict(color=th["muted"], width=4),
            marker=dict(size=5, color=th["muted"], symbol="diamond"), hoverinfo="skip"))

    unit = an.rec.unit
    ax3 = lambda name: dict(title=dict(text=f"{name} ({unit})", font=dict(color=th["ink2"], size=11)),
                            backgroundcolor=th["surface"], gridcolor=th["grid"], zerolinecolor=th["axis"],
                            showbackground=True, tickfont=dict(color=th["muted"], size=10))
    layout = _layout(th, height, legend=True, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_layout(**layout, scene=dict(xaxis=ax3("x"), yaxis=ax3("y"), zaxis=ax3("z"), aspectmode="data",
                                           camera=dict(eye=dict(x=0.95, y=0.95, z=0.7))),
                      uirevision=f"{an.rec.label}|{an.frame}")
    return fig


def figure_buildup(an, theme="light", height=340):
    """Rotation accumulated away from the primary axis, against rotation about it.

    Unlike the direction map, which shows what happens inside each window, this shows
    the total since the first selected frame: a burst of second rotation is a step up
    that stays, and only a real reversal brings the curve down again.
    """
    th = THEMES[theme]
    if an.empty:
        return _empty(th, height, "No axes for this selection")
    b = an.build
    x, y = np.abs(b["twist"]), b["magnitude"]
    fig = go.Figure()
    has_truth = b["truth"] is not None
    if has_truth:
        fig.add_trace(go.Scatter(x=np.abs(b["truth"][0]), y=np.abs(b["truth"][1]), mode="lines",
                                 name="True second rotation", line=dict(color=th["truth"], width=2),
                                 hovertemplate="about the hinge %{x:.1f}°<br>second rotation %{y:.2f}°<extra>true</extra>"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Measured", line=dict(color=th["axes"], width=2),
                             showlegend=has_truth,
                             hovertemplate="about the primary axis %{x:.1f}°<br>away from it %{y:.2f}°<extra></extra>"))
    top = int(np.nanargmax(y))
    fig.add_trace(go.Scatter(x=[x[top]], y=[y[top]], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(size=8, color=th["axes"], line=dict(width=2, color=th["surface"]))))
    fig.add_annotation(x=x[top], y=y[top], text=f"{y[top]:.1f}°", showarrow=False, xanchor="right", yanchor="bottom",
                       xshift=-8, yshift=2, font=dict(size=12, color=th["ink"]))
    fig.update_layout(**_layout(th, height, legend=has_truth, margin=dict(l=56, r=16, t=44 if has_truth else 22, b=44)),
                      hovermode="closest")
    fig.update_layout(legend=dict(y=1.06))
    fig.update_xaxes(**_axis(th, "rotation about the primary axis (deg)", rangemode="tozero"))
    fig.update_yaxes(**_axis(th, "rotation away from it (deg)", rangemode="tozero"))
    return fig


def figure_swing_map(an, theme="light", height=340):
    """Which way the moving segment has tipped away from the primary axis, as seen from
    a frame that turns with the primary rotation. A second rotation about one carried
    axis runs out along a straight line; each burst moves the point further along it."""
    th = THEMES[theme]
    if an.empty:
        return _empty(th, height, "No axes for this selection")
    b, f = an.build, an.fha
    xy = b["swing"][f.ic]
    values = np.c_[f.t, b["twist"][f.ic], b["magnitude"][f.ic]]
    template = ("t = %{customdata[0]:.2f} s<br>about the primary axis %{customdata[1]:.1f}°"
                "<br>away from it %{customdata[2]:.2f}°<extra></extra>")
    return _map(an, th, xy, None, "toward the moving sensor (deg)", "across (deg)", height, rings=True,
                origin_label="primary axis", hover=(values, template), name="Tilt built up", label_start=False)


def figure_series(an, theme="light", height=470):
    """Three stacked panels on a shared time axis: tilt, distance, window size."""
    th = THEMES[theme]
    if an.empty:
        return _empty(th, height, "No axes for this selection")
    f, unit = an.fha, an.rec.unit
    order = np.argsort(f.ic)
    t = f.t[order]
    tilt, dist, net = an.disp["angle"][order], an.disp["distance"][order], np.degrees(f.phi)[order]
    tt, tilt_g, dist_g, net_g = _break_gaps(f.ic[order], tilt, dist, net)
    tt = np.interp(tt, f.ic[order], t)                      # x of the inserted NaNs is irrelevant
    tt = np.where(np.isnan(tilt_g), np.nan, tt)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09)
    marks = dict(mode="lines+markers" if len(t) < 60 else "lines", line=dict(color=th["axes"], width=2),
                 marker=dict(size=8, color=th["axes"], line=dict(width=1.5, color=th["surface"])))
    has_truth = bool(an.truth is not None and np.isfinite(an.truth["tilt"]).any())
    legend = has_truth or an.raw is not None
    if an.raw is not None:
        rf = an.raw["fha"]
        ro = np.argsort(rf.ic)
        rt, r_tilt, r_dist = _break_gaps(rf.ic[ro], an.raw["angle"][ro], an.raw["distance"][ro])
        rt = np.where(np.isnan(r_tilt), np.nan, np.interp(rt, rf.ic[ro], rf.t[ro]))
        grey = dict(mode="lines", line=dict(color=th["muted"], width=1.5), name="Without smoothing",
                    legendgroup="raw")
        fig.add_trace(go.Scatter(x=rt, y=r_tilt, hovertemplate="%{y:.2f}°<extra>without smoothing</extra>", **grey),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=rt, y=r_dist, showlegend=False,
                                 hovertemplate="%{y:.3f} " + unit + "<extra>without smoothing</extra>", **grey),
                      row=2, col=1)
    fig.add_trace(go.Scatter(x=tt, y=tilt_g, name="Helical axes", showlegend=legend,
                             hovertemplate="%{y:.2f}°<extra>tilt</extra>", **marks), row=1, col=1)
    if has_truth:
        true_tilt = np.linalg.norm(an.truth["tilt"], axis=1)[order]
        fig.add_trace(go.Scatter(x=t, y=true_tilt, mode="lines", name="True axis",
                                 line=dict(color=th["truth"], width=2),
                                 hovertemplate="%{y:.2f}°<extra>true tilt</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=tt, y=dist_g, showlegend=False,
                             hovertemplate="%{y:.3f} " + unit + "<extra>distance</extra>", **marks), row=2, col=1)
    fig.add_trace(go.Scatter(x=tt, y=net_g, showlegend=False,
                             hovertemplate="%{y:.2f}°<extra>window</extra>", **marks), row=3, col=1)
    if an.params["method"] in ("incremental_angle", "step_angle"):
        fig.add_hline(y=an.params["step"], line=dict(color=th["muted"], width=1), row=3, col=1)
        fig.add_annotation(text="requested", x=1.0, xref="x3 domain", y=an.params["step"], yref="y3",
                           xanchor="right", yanchor="bottom", showarrow=False,
                           font=dict(size=10, color=th["muted"]))
    titles = ("tilt from the average axis (deg)", f"distance from the pivot ({unit})",
              "net rotation of each window (deg)")
    for row, title in enumerate(titles, start=1):
        fig.update_yaxes(**_axis(th, None, rangemode="tozero"), row=row, col=1)
        fig.update_xaxes(**_axis(th, "time (s)" if row == 3 else None, showspikes=True, spikemode="across",
                                 spikecolor=th["muted"], spikethickness=1, spikedash="solid"), row=row, col=1)
        dom = "" if row == 1 else str(row)
        fig.add_annotation(text=title, xref=f"x{dom} domain", yref=f"y{dom} domain", x=0.0, y=1.0,
                           xanchor="left", yanchor="bottom", showarrow=False,
                           font=dict(size=11, color=th["ink2"]))
    fig.update_layout(**_layout(th, height, legend=legend, margin=dict(l=56, r=16, t=40 if legend else 24, b=44)),
                      hovermode="x unified")
    fig.update_layout(legend=dict(y=1.04))
    return fig


def _log_ticks(lo, hi):
    """Round tick values (1, 2, 5 per decade; decades only when that gets crowded)
    covering ``lo``..``hi`` on a log axis."""
    decades = np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1)
    ticks = [m * 10.0 ** d for d in decades for m in (1, 2, 5)]
    ticks = [v for v in ticks if lo <= v <= hi]
    if len(ticks) > 7:
        ticks = [v for v in ticks if abs(np.log10(v) - round(np.log10(v))) < 1e-9]
    return ticks


def figure_smoothing_sweep(an, theme="light", height=280):
    """How much each cut-off on the slider would take out of the selected frames.

    The curve is the RMS angle between the measured and the smoothed pose of the moving
    body relative to the reference body. The horizontal line is the frame-to-frame
    jitter of the same frames: while the curve stays at or below it, the filter is
    removing no more than that jitter. Where the curve climbs above it, slower tracker
    error and real movement are going too, and nothing in the data tells them apart.
    """
    th = THEMES[theme]
    s = an.smoothing
    curve = s["sweep"] if s["sweep"] is not None else _sweep_rms(smoothing_sweep(an.rec), *an.cut)
    if curve["error"]:
        return _empty(th, height, curve["error"])
    keep = curve["rms_angle"] > 0
    if not keep.any():
        return _empty(th, height, f"Every cut-off offered is above half the frame rate ({s['rate'] / 2:.3g} Hz)")
    f, y, dist = curve["cutoffs"][keep], curve["rms_angle"][keep], curve["rms_distance"][keep]
    unit = an.rec.unit
    jit = s["jitter"][0]
    has_jitter = bool(np.isfinite(jit) and jit > 0)
    x_range = [f.max() * 1.35, f.min() / 1.35]                       # heavier smoothing to the right, like the slider
    lo, hi = min(y.min(), jit) if has_jitter else y.min(), max(y.max(), jit) if has_jitter else y.max()
    if s["active"] and s["rms_angle"] > 0:
        lo, hi = min(lo, s["rms_angle"]), max(hi, s["rms_angle"])
    y_range = [lo / 1.8, hi * 1.8]
    fig = go.Figure()
    if has_jitter:
        fig.add_trace(go.Scatter(x=x_range, y=[jit, jit], mode="lines", line=dict(color=th["muted"], width=1),
                                 hoverinfo="skip", showlegend=False))
        fig.add_annotation(text=f"frame-to-frame jitter {jit:.2g}°", x=1.0, xref="x domain", y=np.log10(jit), yref="y",
                           xanchor="right", yanchor="bottom", showarrow=False,
                           font=dict(size=10, color=th["muted"]))
    fig.add_trace(go.Scatter(
        x=f, y=y, mode="lines+markers", showlegend=False, line=dict(color=th["axes"], width=2),
        marker=dict(size=8, color=th["axes"], line=dict(width=1.5, color=th["surface"])), customdata=dist,
        hovertemplate="low-pass at %{x:g} Hz<br>removes %{y:.2g}° and %{customdata:.2g} " + unit + " (RMS)<extra></extra>"))
    if s["active"] and s["rms_angle"] > 0:
        fig.add_trace(go.Scatter(x=[s["cutoff"]], y=[s["rms_angle"]], mode="markers", showlegend=False,
                                 hoverinfo="skip", marker=dict(size=14, color=th["axes"],
                                                               line=dict(width=2, color=th["surface"]))))
        # the curve climbs to the right: label above it, except near the left edge
        left = np.log10(x_range[0] / s["cutoff"]) < 0.3 * np.log10(x_range[0] / x_range[1])
        fig.add_annotation(text=f"{s['cutoff']:g} Hz: {s['rms_angle']:.2g}°", x=np.log10(s["cutoff"]),
                           y=np.log10(s["rms_angle"]), showarrow=False, xanchor="left" if left else "right",
                           yanchor="top" if left else "bottom", xshift=10 if left else -10,
                           yshift=-6 if left else 6, font=dict(size=12, color=th["ink"]))
    xt = [v for v in (20, 10, 6, 4, 3, 2, 1, 0.5, 0.3, 0.2) if f.min() * 0.99 <= v <= f.max() * 1.01]
    yt = _log_ticks(*y_range)
    fig.update_layout(**_layout(th, height, margin=dict(l=56, r=16, t=16, b=44)), hovermode="closest")
    fig.update_xaxes(**_axis(th, "low-pass cut-off (Hz) — heavier smoothing to the right", type="log",
                             range=list(np.log10(x_range)), tickvals=xt, ticktext=[f"{v:g}" for v in xt],
                             zeroline=False))
    fig.update_yaxes(**_axis(th, "removed, RMS (deg)", type="log", range=list(np.log10(y_range)), tickvals=yt,
                             ticktext=[f"{v:g}" for v in yt], zeroline=False))
    return fig


def figure_smoothing_removed(an, theme="light", height=280):
    """What the chosen cut-off removes at each selected frame: the angle between the
    measured and the smoothed relative pose. Noise alone gives an even band about as
    high as the frame-to-frame jitter (the horizontal line); humps that come and go with
    the movement are movement being removed."""
    th = THEMES[theme]
    s = an.smoothing
    if not s["active"]:
        return _empty(th, height, "Smoothing is off: nothing is removed" if s["cutoff"] is None else
                      f"{s['cutoff']:g} Hz is above half the frame rate ({s['rate'] / 2:.3g} Hz): nothing is removed")
    t, y, unit = an.t, s["angle"], an.rec.unit
    fig = go.Figure()
    jit = s["jitter"][0]
    if np.isfinite(jit) and jit > 0:
        # the line often runs along the bottom, under the data, so its label goes above the plot
        fig.add_trace(go.Scatter(x=[t[0], t[-1]], y=[jit, jit], mode="lines", line=dict(color=th["muted"], width=1),
                                 hoverinfo="skip", showlegend=False))
        fig.add_annotation(text=f"grey line: frame-to-frame jitter {jit:.2g}°", x=1.0, xref="x domain", y=1.0,
                           yref="y domain", xanchor="right", yanchor="bottom", showarrow=False,
                           font=dict(size=10, color=th["muted"]))
    few = len(t) < 60
    fig.add_trace(go.Scatter(
        x=t, y=y, mode="lines+markers" if few else "lines", showlegend=False,
        line=dict(color=th["axes"], width=2 if few else 1.5),
        marker=dict(size=8, color=th["axes"], line=dict(width=1.5, color=th["surface"])), customdata=s["distance"],
        hovertemplate="%{y:.2g}° and %{customdata:.2g} " + unit + "<extra>removed</extra>"))
    fig.update_layout(**_layout(th, height, margin=dict(l=56, r=16, t=22, b=44)), hovermode="x unified")
    fig.update_xaxes(**_axis(th, "time (s)", showspikes=True, spikemode="across", spikecolor=th["muted"],
                             spikethickness=1, spikedash="solid"))
    fig.update_yaxes(**_axis(th, f"removed at {s['cutoff']:g} Hz, each frame (deg)", rangemode="tozero"))
    return fig


def figure_overview(rec, cut, theme="light", height=None):
    """The whole recording: joint rotation against time, with the selected frames marked."""
    th = THEMES[theme]
    has_quality = rec.quality is not None
    height = height or (300 if has_quality else 200)
    q = fc.quaternions(fc.relative_poses(rec.T1, rec.T2))
    c1 = int(max(cut[0] or 0, 0))
    c2 = int(len(rec) if cut[1] is None else min(cut[1], len(rec)))
    rot = np.degrees(fc.net_angle(q, q[c1][None, :]))
    fig = make_subplots(rows=2 if has_quality else 1, cols=1, shared_xaxes=True, vertical_spacing=0.16)
    fig.add_trace(go.Scatter(x=rec.t, y=rot, mode="lines", line=dict(color=th["axes"], width=2), showlegend=False,
                             hovertemplate="%{y:.1f}°<extra>rotation</extra>"), row=1, col=1)
    fig.add_vrect(x0=rec.t[c1], x1=rec.t[c2 - 1], fillcolor=th["axes"], opacity=0.10, line_width=0, row=1, col=1)
    fig.add_annotation(text="selected", x=rec.t[c1], xref="x", y=1.0, yref="y domain", xanchor="left",
                       yanchor="top", showarrow=False, font=dict(size=10, color=th["ink2"]))
    fig.add_annotation(text="joint rotation from the first selected frame (deg)", xref="x domain", yref="y domain",
                       x=0.0, y=1.0, xanchor="left", yanchor="bottom", showarrow=False,
                       font=dict(size=11, color=th["ink2"]))
    fig.update_yaxes(**_axis(th, None, rangemode="tozero"), row=1, col=1)
    spike = dict(showspikes=True, spikemode="across", spikecolor=th["muted"], spikethickness=1, spikedash="solid")
    if has_quality:
        fig.add_trace(go.Scatter(x=rec.t, y=rec.quality["residual_2"], mode="lines", showlegend=False,
                                 line=dict(color=th["axes"], width=2),
                                 hovertemplate="%{y:.2f} " + rec.unit + "<extra>marker fit</extra>"), row=2, col=1)
        fig.add_annotation(text=f"marker-fit residual of the moving segment ({rec.unit}) — a rigid cluster stays flat",
                           xref="x2 domain", yref="y2 domain", x=0.0, y=1.0, xanchor="left", yanchor="bottom",
                           showarrow=False, font=dict(size=11, color=th["ink2"]))
        fig.update_yaxes(**_axis(th, None, rangemode="tozero"), row=2, col=1)
        fig.update_xaxes(**_axis(th, None, **spike), row=1, col=1)
        fig.update_xaxes(**_axis(th, "time (s)", **spike), row=2, col=1)
    else:
        fig.update_xaxes(**_axis(th, "time (s)", **spike), row=1, col=1)
    fig.update_layout(**_layout(th, height, margin=dict(l=56, r=16, t=24, b=44)), hovermode="x unified")
    return fig


# ----------------------------------------------------------------- numbers and table

def summary(an):
    """Headline numbers as (label, value, unit) tuples."""
    unit = an.rec.unit
    rom = float(np.nanmax(an.rotation)) if an.rotation is not None and len(an.rotation) else float("nan")
    if an.empty:
        return [("Axes", "0", ""), ("Range of motion", f"{rom:.1f}", "deg")]
    f = an.fha
    per10 = float(np.sum(f.weight * np.abs(f.d)) / max(np.sum(f.weight * f.phi), 1e-12) * np.radians(10.0))
    dur = an.t[f.i1] - an.t[f.i0]
    return [
        ("Axes", f"{len(f):,}", ""),
        ("Range of motion", f"{rom:.1f}", "deg"),
        ("Mean angle to average axis", f"{an.disp['angle_mean']:.2f}", "deg"),
        ("RMS distance from pivot", f"{an.disp['distance_rms']:.3g}", unit),
        ("Translation along axis per 10°", f"{per10:.3g}", unit),
        ("Built up off primary axis", f"{np.nanmax(an.build['magnitude']):.1f}", "deg"),
        ("Window, median", f"{np.median(np.degrees(f.phi)):.1f}° · {np.median(dur):.2f} s", ""),
    ]


# (column key, header, number format); "{unit}" in a header becomes the length unit
TABLE_COLUMNS = [
    ("time_s", "Time (s)", "{:.3f}"), ("frame", "Frame", "{:d}"),
    ("rotation_deg", "About average axis (°)", "{:.2f}"), ("off_primary_deg", "Off primary axis (°)", "{:.2f}"),
    ("window_deg", "Window (°)", "{:.2f}"),
    ("window_s", "Window (s)", "{:.3f}"), ("tilt_deg", "Tilt (°)", "{:.3f}"),
    ("tilt_toward_deg", "Tilt toward (°)", "{:.3f}"), ("tilt_across_deg", "Tilt across (°)", "{:.3f}"),
    ("pos_toward", "Position toward ({unit})", "{:.4f}"), ("pos_across", "Position across ({unit})", "{:.4f}"),
    ("distance", "From pivot ({unit})", "{:.4f}"),
    ("translation", "Along axis ({unit})", "{:.4f}"),
]


def axes_table(an, drawn_only=False):
    """One row per axis as a dict of columns (numpy arrays). Directions and points are
    in the display frame; frame numbers refer to the original recording."""
    if an.empty:
        return {}
    f = an.fha
    idx = an.draw if drawn_only else np.arange(len(f))
    idx = idx[np.argsort(f.ic[idx])]
    t = an.t
    cols = {
        "time_s": f.t[idx], "frame": f.ic[idx] + an.cut[0],
        "window_start_frame": f.i0[idx] + an.cut[0], "window_end_frame": f.i1[idx] + an.cut[0],
        "rotation_deg": an.twist_axes[idx],
        "about_primary_deg": an.build["twist"][f.ic[idx]], "off_primary_deg": an.build["magnitude"][f.ic[idx]],
        "window_deg": np.degrees(f.phi[idx]),
        "window_s": t[f.i1[idx]] - t[f.i0[idx]], "tilt_deg": an.disp["angle"][idx],
        "tilt_toward_deg": an.tilt[idx, 0], "tilt_across_deg": an.tilt[idx, 1],
        "pos_toward": an.pos[idx, 0], "pos_across": an.pos[idx, 1],
        "distance": an.disp["distance"][idx], "translation": f.d[idx], "weight": f.weight[idx],
    }
    for k, name in enumerate("xyz"):
        cols[f"n_{name}"] = an.n[idx, k]
    for k, name in enumerate("xyz"):
        cols[f"point_{name}"] = an.p[idx, k]
    return cols


def axes_csv(an):
    """All axes as CSV text, with a header describing the analysis."""
    cols = axes_table(an)
    out = io.StringIO()
    p = an.params
    out.write(f"# recording: {an.rec.label}\n# frames: {an.cut[0]}-{an.cut[1]}\n")
    out.write(f"# method: {p['method']}, step: {p['step']:g}, estimator: {p['estimator']}, "
              f"window rule: {p['criterion']}\n")
    out.write(f"# frame of the axes: {'lab' if an.frame == 'lab' else 'reference body'}; "
              f"length unit: {an.rec.unit}\n")
    s = an.smoothing
    if s["active"]:
        out.write(f"# smoothing: low-pass at {s['cutoff']:g} Hz (-3 dB, no time shift) on the screw coordinates of "
                  f"the pose of the moving body relative to the reference body, whole recording, before the frame "
                  f"cut; removed {s['rms_angle']:.3g} deg and {s['rms_distance']:.3g} {an.rec.unit} RMS from the "
                  f"selected frames (their frame-to-frame jitter: {s['jitter'][0]:.3g} deg)\n")
    else:
        out.write("# smoothing: none\n")
    if not an.empty:
        out.write("# average axis direction: " + " ".join(f"{v:.6f}" for v in an.n_bar) + "\n")
        out.write("# pivot: " + " ".join(f"{v:.6f}" for v in an.pivot) + "\n")
        source = "this recording's own average axis" if an.build["own"] else (an.build["label"] or "given axis")
        out.write("# primary axis for the build-up columns (reference-body frame): "
                  + " ".join(f"{v:.6f}" for v in an.build["axis"]) + f" ({source})\n")
        names = list(cols)
        out.write(",".join(names) + "\n")
        data = np.column_stack([cols[k] for k in names])
        for row in data:
            out.write(",".join(f"{int(v)}" if "frame" in name else f"{v:.6g}" for name, v in zip(names, row)) + "\n")
    return out.getvalue()


# -------------------------------------------------------------- backward compatibility

def calculate_FHA_vectorized(T1, T2):
    """Screw parameters of the displacements ``T2 @ inv(T1)``: ``phi, n, t, s``.

    Kept under its original name for existing scripts. It now goes through
    ``fha_core.screw_from_displacement``, which is exact over the whole 0-180 deg range
    (the original folded angles above 135 deg back) and returns NaN, not a near-zero
    vector, when there is no rotation.
    """
    return fc.screw_from_displacement(np.asarray(T2, float) @ fc.invert(np.asarray(T1, float)))


def primary_axis_of(rec, step=10.0):
    """Average helical axis of a whole recording, in the frame of its reference body.

    Run on a recording of the primary rotation alone (a pure flexion trial, the bench
    hinge without the second rotation), this gives the axis to measure another
    recording's build-up against — provided the reference sensor or marker cluster sat
    on its segment in the same way in both recordings.
    """
    fha = fc.compute_fha(rec.t, rec.T1, rec.T2, "incremental_angle", step)
    if len(fha) == 0:
        raise ValueError(f"{rec.label} turns by less than {step:g} deg, so it has no helical axis to take.")
    return fc.average_axis(fha.n, fha.s, fha.weight)[0]


def reference_sensor_shift(rec, other):
    """How differently the reference sensor sits in the lab in two recordings: (degrees,
    length units). Large values mean it was moved or re-mounted in between — or that the
    whole set-up was moved, which is harmless. Returns None if either one moves in the
    lab during its recording."""
    if not (reference_is_static(rec.T1, rec.T2) and reference_is_static(other.T1, other.T2)):
        return None
    a = Rotation.from_matrix(rec.T1[:, :3, :3]).mean()
    b = Rotation.from_matrix(other.T1[:, :3, :3]).mean()
    turn = float(np.degrees((a.inv() * b).magnitude()))
    shift = float(np.linalg.norm(rec.T1[:, :3, 3].mean(axis=0) - other.T1[:, :3, 3].mean(axis=0)))
    return turn, shift
