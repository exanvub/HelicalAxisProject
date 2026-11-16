# HelicalAxisProject

This repository is dedicated to the collaborative development of the Helical Axis Method for describing motion in human joints. The project aims to explore different approaches and techniques to solve the helical axis problem.

Please note that this project is currently a work in progress and is by no means finished. Contributions from the community are highly encouraged and welcomed.

# Finite Helical Axis (FHA) Analysis Toolkit

The toolkit includes vectorized methods for computing FHA parameters from time-series transformation matrices, with multiple options for time-based and angle-based FHA segmentation. Interactive 3D visualization is included using **Plotly** and **Dash**.

---

## Features

* Efficient **vectorized FHA calculation** between pairs of homogeneous transformation matrices.
* Multiple analysis modes:

  * **All FHA**: Instantaneous FHA between all consecutive frames.
  * **Incremental Time**: FHA computed over fixed time steps.
  * **Step Angle**: FHA computed each time the joint has turned by a set angle.
  * **Incremental Angle**: FHA computed for every frame over a moving window of a set angle.
  * **Reference Pose**: FHA of the displacement since a reference pose.
* **Global frame transformation** of axes and origins.
* **Plane intersection calculation** for spatial referencing.
* **Interactive 3D visualization** of helical axes and motion using Plotly/Dash.
* Includes tools for:

  * Axis direction computation
  * Rotation angle extraction
  * Axis of average motion estimation

---
## FHA Versions

The repository contains several implementations of the FHA analysis to accommodate different needs:

### `FHA_plt`

This version uses **Matplotlib (plt)** for visualization.

* **Slow** to compute and render.
* **Best for understanding the code** due to its simplicity and readability.

### `FHA_plotly_traditional`

This version uses **Plotly** for interactive 3D plots, based on a **traditional (non-vectorized)** computation approach.

* **Slow to compute**, but
* **Faster to render** and
* **Equally understandable** as the `FHA_plt` version.

### `FHA_plotly_fast`

The version under active development: vectorised computation, an interactive dashboard, and
the corrections that came out of the 2026 review of the method (see
[research/FHA_deepdive/FINDINGS.md](research/FHA_deepdive/FINDINGS.md)).

```
cd FHA/FHA_plotly_fast
python main.py          # then open http://127.0.0.1:8050
```

Needs `numpy`, `scipy`, `pandas`, `plotly` and `dash`; `ezc3d` only for Vicon C3D files.

`FHA_plt` and `FHA_plotly_traditional` keep the original algorithm. They are fine for a
single one-way movement but not for out-and-back recordings, where their average axis
cancels out (finding F8 in the research log).

---

## Using the dashboard

Pick a recording, a frame range and a method in the bar at the top; everything below
follows. `config.py` sets what the page opens with — edit it and press **Reload config.py**.

**Recording.** Polhemus CSV, Vicon C3D, or the built-in simulator, which produces a hinge
with an optional second rotation and draws the true axis next to the estimate. Files in
`test_data/` are listed; any other file can be given by path. Vicon marker names come
from `config.py`.

**Method.**

| `method_type` | What it gives |
|---|---|
| `incremental_angle` | Moving window (miFHA): one axis per frame, from a window of `step` degrees centred on that frame |
| `step_angle` | Consecutive windows (fiFHA): one axis per `step` degrees |
| `reference_pose` | Axis of the whole displacement since the first selected frame |
| `incremental_time` | Fixed lag of `step` frames |
| `all_FHA` | Consecutive frames (ill-conditioned; for illustration) |

The window threshold is applied to the **net rotation between the two poses each axis
is computed from**, because that is what makes a helical axis well or ill conditioned.
Every window spans at least `step` degrees of it; frames where that is impossible (at
rest, at a turn-around) get no axis. The earlier rule — summing frame-to-frame angles —
is still available as `window_rule = 'path'` for comparison; it counts noise and
reversals as rotation, and over a whole recording a large share of its windows compare
poses much closer together than asked for. A window can only end on a frame, so with few
frames per degree (dynamic CT, or a small window on a fast movement) the windows come
out wider than asked; the page says by how much.

**Smoothing.** The slider sets a low-pass filter on the motion before the axes are
computed: Off, then 20 Hz down to 0.2 Hz (`smoothing_hz` in `config.py`). The cut-off is
the usual one (−3 dB, no time shift, equivalent to a zero-lag fourth-order Butterworth).
The filter acts on the pose of the moving body relative to the reference body, in screw
coordinates, so it cannot move the axis of a hinge.

A filter cannot tell noise from movement: it removes whatever changes faster than its
cut-off. The page therefore shows what it took out:

* a sentence beside the slider: how much was removed from the selected frames, and how
  many times their frame-to-frame jitter that is;
* the axes without smoothing, as a grey line under the smoothed ones in the direction
  map, the position map and the series;
* the card **What the smoothing takes out**: what each cut-off would remove, and what
  the chosen one removes at each frame.

Look for a flat stretch in that curve. There, little in the recording changes at that
rate; a cut-off inside it separates quick from slow and its exact value hardly matters
(4–10 Hz on the head recordings in `test_data`). Where the curve climbs steadily, as on
the two-hand bench recordings, the fingers are the movement itself: a filter only
shortens them by removing it, and a wider window is the better tool. From about 2 Hz
down the spread of the axes shrinks in every recording tried. Report the cut-off
together with the window size. Details: findings F17–F19 in the research log.

**What the page shows.**

* **Helical axes in space** — the axes, their average axis and pivot, the cross-section
  plane, and the sensor paths.
* **Direction map** — which way each axis points; distance from the centre is its angle
  to the average axis, in degrees.
* **Position map** — where each axis crosses the plane through the pivot (move the plane
  with *Plane offset*).
* **How the second rotation builds up** — how far the moving segment has turned away
  from a primary axis since the first selected frame, against how far it has turned
  about it, and which way. A burst of second rotation is a step that stays.
* **Along the movement** — tilt and distance of each axis over time, and the rotation
  each window really spans.
* **What the smoothing takes out** — how much each cut-off would remove, against the
  frame-to-frame jitter of the recording, and what the chosen cut-off removes frame by
  frame.
* **Recording** — joint rotation over the whole file with the analysed frames shaded; for
  Vicon, also the residual of the marker fit.
* Headline numbers, a table of the axes, and **Download axes (CSV)**.

**Reading two-axis motion.** A helical axis tilts by the *ratio* of the two rotation
rates during its window, not by how much second rotation has built up. If the second
rotation is fed in unevenly — by hand, or by a servo that moves in steps — the axes
swing back and forth ("fingers"). A larger window averages that out at the cost of
resolution. Try the simulator's steady and burst movements to see both cases with the
true axis drawn in.

**The build-up needs a primary axis from another recording.** The build-up panel adds
up the rotation instead of looking at one window at a time, so a burst raises it and
leaves it raised. It must be measured against the primary axis — and a recording's own
average axis is not that: it is the direction of the whole rotation, second rotation
included, so measured against it 20° of second rotation reads as under 4° and falls
back. Record the primary rotation on its own in each session (a pure flexion trial; the
bench hinge without the second rotation) with the reference sensor left where it is,
and choose that recording under *Build-up → Reference recording* (or `reference_path` in
`config.py`). Against the hinge of recording 1, the test recordings 3 and 4 build up
23.8° and 31.4°.

---

## Using the code directly

From inside `FHA/FHA_plotly_fast`:

```python
import recordings, fha_core as fc

rec = recordings.load_polhemus("path/to/file.csv")
fha = fc.compute_fha(rec.t, rec.T1, rec.T2, method="incremental_angle", step=10)

fha.n, fha.s          # direction of each axis and the point on it nearest the origin,
                      # in the frame of the reference body (sensor 1)
fha.phi, fha.d        # rotation and translation along the axis, per window
fha.ic, fha.t         # frame and time each axis belongs to
fha.weight            # averaging weights: every degree of motion counts equally

n_bar, pivot = fc.average_axis(fha.n, fha.s, fha.weight)
spread = fc.dispersion(fha.n, fha.s, n_bar, pivot, fha.weight)   # angle_mean, distance_rms, ...

# rotation accumulated about a primary axis, and away from it, since the first frame
import FHA_visualizer_fast as fv
hinge = fv.primary_axis_of(recordings.load_polhemus("path/to/pure_hinge.csv"))
twist, swing = fc.swing_twist(fc.relative_poses(rec.T1, rec.T2), hinge, turning=True)   # degrees
```

`fc.axes_to_lab(fha, rec.T1)` gives the same axes in the lab frame. Inputs are two
synchronised stacks of 4x4 poses (`T1` reference body, `T2` moving body) and a time vector.

Smoothing is a separate, optional step in front:

```python
T1s, T2s = fc.smooth_motion(rec.t, rec.T1, rec.T2, 6.0)          # low-pass at 6 Hz
fha = fc.compute_fha(rec.t, T1s, T2s, step=10)

raw, smooth = fc.relative_poses(rec.T1, rec.T2), fc.relative_poses(T1s, T2s)
removed_deg, removed_len = fc.removed_by_smoothing(raw, smooth)  # per frame
jitter_deg, jitter_len = fc.jitter(raw)                          # frame-to-frame jitter
```

The time vector must hold sample times. `recordings.load_polhemus` turns Polhemus
arrival stamps into an even time base (`fc.even_time_base`), also when samples were
delivered in bursts; taken at face value those make a filter distort the movement.

Tests: `python tests/test_fha_core.py`, `tests/test_segment_frames.py`,
`tests/test_dashboard.py` (also runnable with pytest).

---

## Contact

For questions or contributions, please contact **nicolas.van.vlasselaer@vub.be** or **matteo.iurato@edu.unige.it**

---

This code is distributed for academic and research purposes only. Please cite appropriately if used in publications.