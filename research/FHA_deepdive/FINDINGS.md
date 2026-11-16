# FHA deep dive — research log

Investigation of why the Moving Increment Angle-Based FHA (miFHA, `incremental_angle`)
produces unstable, back-and-forth "fingers" in the cross-sectional plane when a second
rotation component is introduced, and what to change.

- Dates: 2026-11-15 (two rounds; F14–F16 come from the second, after the user's questions
  on the window threshold and on adding up rotation instead of tracking the axis) and
  2026-11-16 (third round, F17–F19: a smoothing filter, asked for with the note that the
  end goal is human joint kinematics from the motion lab and from dynamic 4D CT; fourth
  round, F20: playing the motion back in the dashboard, with basic shapes for the two
  bodies until bone surfaces are available)
- Code studied: `FHA/FHA_plotly_fast/` as of commit `5b32fa5`. A verbatim copy is kept in
  `original_snapshot/`; "the original" below always means that copy.
- Reference: ISB 2025 presentation, `HelicalAxis-Literature/ISB_2025_Van_Vlasselaer.pdf`
- Python used for all runs: `/Users/nicolas/miniconda3/bin/python` (3.13, numpy 2.3, scipy 1.17,
  plotly 6.5, dash 2.14, ezc3d 1.7)

## Summary — read this first

1. **The fingers are in the motion; the method does not invent them** (F3). A helical
   axis points along the rotation that happens during its window, so with two rotations
   it tilts by `atan(secondary rate / primary rate)` — the *ratio of the two rates*, not
   the amount of second rotation accumulated. On a simulated rig with known ground truth
   the original code returns a smooth arc when the second rotation is fed in steadily
   and fingers only when it arrives in bursts. The Polhemus recordings contain 12 and 17
   such bursts: one hand lowers the beam while the other twists the sensor block.
2. **On clean one-way Polhemus data the original miFHA was already accurate** (F8, F13):
   0.243° spread on the pure hinge, identical to the revised code. The error reduction
   reported at ISB 2025 is reproduced (fiFHA 0.32°, miFHA 0.30° against 3.4° for the
   plain FHA).
3. **The original breaks on out-and-back recordings** (F8, H4, H1). Its average axis is a
   mean of direction vectors that flip sign on the way back, so it cancels: the code
   reports 70–84° "mean angle" on a pure hinge and builds its plane on that axis. Windows
   that straddle the turn-around give axes up to 90° off.
4. **Each axis was assigned to the start of its window instead of the middle** (F4, F7,
   H6). Harmless on a hinge; 4–5× more tracking error once the axis moves.
5. **The original cross-section mostly shows direction, scaled by an arbitrary lever arm**
   (F11, H5): the plane sits 6–10 cm from where the axes converge, inflating the pattern
   up to 4.8×. The revised code shows a position map (plane through the pivot) and a
   direction map (degrees) separately.
6. **Fitting all frames of a window is not an improvement on these instruments** (F6, F7,
   F8, H8 rejected). Tracker error is slow-varying, not white; the fit gives no gain on
   Polhemus and is 20–29% worse on Vicon.
7. **The Vicon two-axis recordings are contaminated by a non-rigid marker cluster** (F9).
   Whether the second servo really jerks cannot be established from them.
8. **What controls the fingers:** the window size (F12), and how evenly the second
   rotation is driven.
9. **The window threshold is now applied to the right quantity and met by every
   window** (F14). The original summed frame-to-frame angles; over whole recordings
   11–89% of its windows compared poses less than 9° apart when 10° was asked for. Now
   the two poses of every window are at least the requested net rotation apart.
10. **Adding up the rotation turns a burst into a step that stays — if it is measured
    against the primary axis** (F15). Against the recording's own average axis it
    cannot: that axis is the direction of the total rotation, so 20° of second rotation
    reads as under 4° and falls back. Against the hinge taken from the pure-hinge
    recording, recordings 3 and 4 build up 23.8° and 31.4° and keep them. An earlier
    statement here, that recording 3's second rotation is "taken back out", was an
    artefact of the wrong axis and is corrected in F12.
11. **The primary axis has to come from outside the recording** (F16). A two-axis fit
    finds it when the second rotation comes in bursts (0.42° from the hinge on
    recording 3) but is not identifiable when the two rotations are coupled, which is
    the case that matters for a joint.

12. **A smoothing filter shortens the fingers, but on the bench recordings it does so by
    removing movement** (F17). The Polhemus jitters by 0.003° from frame to frame; a
    20 Hz filter already takes that out and the fingers hardly change (doubling back
    5.2 → 4.8). Getting them to 2.4 takes 0.5 Hz, which removes 0.24°, eighty times
    the jitter: the hands' bursts themselves. A 20° window without any filter does the
    same (2.3). The build-up of F15 does not move at any cut-off (23.8°).
13. **On recordings of people a filter has a defensible place, and the recording shows
    where** (F17). Plotted against the cut-off, what the filter removes runs flat where
    little in the recording changes at that rate. Head movement: flat from 20 Hz down to
    4 Hz at the level of the jitter; 6 Hz removes 0.03° and shortens the path of the axis
    direction by 14–31% without changing its spread. Knee flexion: a flatter stretch
    between about 6 and 2 Hz in three recordings of four. Below 2 Hz the spread itself
    starts to shrink: movement is going. The bench two-axis recordings have no flat
    stretch at all, so every cut-off there is a compromise.
14. **Where and on what time base the filter acts decides whether it can be trusted**
    (F17, F18). Filtering positions and orientation separately moves the axis of a
    perfect hinge by 1 mm at 1 Hz and 3 mm at 0.5 Hz; filtering the screw coordinates of
    the relative pose moves it by nothing. And 15 of the 87 Polhemus recordings carry
    arrival timestamps delivered in bursts; taken at face value they make a 6 Hz filter
    put up to 0.9° of distortion into a noise-free movement.
15. **Few frames per degree no longer lose their axes** (F19). At 4° per frame the
    10° moving window returned nothing, because a guard against pose jumps took coarse
    frames for jumps. With two dozen frames, as in dynamic CT, a filter helps a little
    in a narrow range (axis error −16% at 1 Hz) and harms below it. Simulated only.

16. **The dashboard plays the motion back** (F20): two basic shapes for the bodies, the
    axis of each instant picked out, and the playhead shown on every plot. This is a
    tool, not a result; F20 records what had to be learnt to make it smooth.

Section 5 lists what was changed in the code, section 6 what to do about the rigs,
section 7 what is still open.

## How to reproduce

```
cd research/FHA_deepdive/scripts
python 01_characterise_data.py     # F1
python 02_reproduce_fingers.py     # F2, F5
python 03_rate_ratio.py            # F3 (b), (c)
python 04_synthetic_original.py    # F3 (a), F4
python calibrate_noise.py          # F6
python 05_benchmark_synthetic.py   # F7
python 06_real_single_axis.py      # F8
python 07_vicon_cluster.py         # F9
python 08_two_axis_views.py        # F12
python 09_plane_placement.py       # F11
python 10_original_defects.py      # F10
python 11_isb_numbers.py           # F13
python 12_window_threshold.py      # F14
python 13_build_up.py              # F15
python 14_two_axis_fit.py          # F16
python 15_smoothing.py             # F17, F18, F19 (about two minutes)
```

Each script prints the numbers quoted here and writes its figure to `figures/`.
`scripts/common.py` loads the original pipeline from `original_snapshot/` and the revised
code from `FHA/FHA_plotly_fast/`. Tests of the revised code:
`python FHA/FHA_plotly_fast/tests/test_fha_core.py` (and `test_segment_frames.py`,
`test_dashboard.py`, `test_playback.py`, `test_playback_browser.py`; the last opens the
dashboard in headless Chrome and prints the timings quoted in F20).

All numbers below were regenerated with the final code. Where the second round changed
the window rule, the tables of F7, F8, F11 and F12 moved in the third digit and were
updated; conclusions did not change. The third round changed the window rule once more
(F19); scripts 01–14 and `calibrate_noise.py` were run with the rule before and after
and print identical output, line for line.

Every finding carries a status:

- **VERIFIED** — reproduced numerically; the script and its output are named.
- **HYPOTHESIS** — follows from reading the code or the math, not yet tested.
- **REJECTED** — tested and found not to hold.

---

## 1. How the original pipeline works (from reading the code)

1. Importers produce two pose streams `T1[i]`, `T2[i]` (4x4, sensor/segment → global).
2. `generate_FHA` forms `Trel[i] = inv(T1[i]) @ T2[i]` — pose of body 2 in the frame of body 1.
3. `calculate_FHA_vectorized(Ta, Tb)` forms `H = Tb @ inv(Ta)`, the displacement of body 2
   between two instants expressed in frame 1, and extracts the screw parameters with the
   Spoor & Veldpaus (1980) formulas: angle `phi`, direction `n`, translation along the axis
   `t = n·v`, and point on the axis `s`.
4. `activate_method` picks which pairs of frames to compare:
   - `all_FHA`: consecutive frames.
   - `incremental_time`: frame `i` vs `i + step`.
   - `step_angle` (fiFHA): non-overlapping windows that close when the summed
     frame-to-frame angle exceeds `step` degrees.
   - `incremental_angle` (miFHA): for every start frame `i`, the first `j` where the summed
     frame-to-frame angle from `i` exceeds `step` degrees.
5. `plotly_visualize_FHA` maps each axis to the global frame with `T1` at the window start,
   averages them into an Average Helical Axis (AHA), and intersects every FHA with a plane
   perpendicular to the AHA.

## 2. Observations from code reading — all HYPOTHESIS until tested below

| # | Where | Observation | Expected consequence |
|---|-------|-------------|----------------------|
| H1 | `activate_method`, `incremental_angle` and `step_angle` | The window closes on the **sum of absolute frame-to-frame angles** (a path length), not on the **net rotation** between the two frames actually compared. Noise and any non-monotonic motion add to the path length without adding net rotation. | The pair `(i, j)` handed to the FHA formula can have a net rotation far below `step`. The FHA is then ill-conditioned again — the exact failure the method was designed to remove. |
| H2 | same | `all_angles[k]` is the angle from frame `k` to `k+1`, so a sum over `k = i..j` spans frames `i` to `j+1`, but the FHA is computed between `i` and `j`. | Off-by-one. If one frame step alone exceeds `step`, the FHA is computed between a frame and itself (identity → division by ~0). |
| H3 | `calculate_FHA_vectorized` | `phi = arcsin(sinPhi)` when `sinPhi <= √2/2`, else `arccos(cosPhi)`. | Wrong above 135°: `arcsin` returns `180° − phi`. `atan2(sinPhi, cosPhi)` is correct over the whole range. |
| H4 | `plotly_visualize_FHA`, AHA | AHA direction is `mean(n_i)`; AHA point is `mean(s_i)`. | Directions that flip sign (flexion then extension) cancel in the mean. A mean of points is not the point closest to all lines. |
| H5 | `plotly_visualize_FHA` | The plane sits 10 units along the AHA from the mean point (`midpoint` of a 20-unit segment). | The cross-section mixes position scatter with direction scatter amplified by a 10-unit lever arm; the two cannot be told apart in the plot. |
| H6 | `plotly_visualize_FHA` | Each FHA is tagged with the time of the window **start** and mapped to global with `T1[start]`. | A window describes motion around its **middle**. Time tagging is biased by half a window, and the bias varies with speed. |
| H7 | `plotly_visualize_FHA` | `p = s + d·n` shifts the plotted origin along the axis by the window translation. | Harmless to the line, but the plotted "origin" is not a meaningful point. |
| H8 | whole pipeline | No filtering; an FHA uses only the two end poses of its window. | All noise in those two poses goes straight into the axis. Frames inside the window are ignored. |

## 3. Findings

### F1 — What is in the test recordings — VERIFIED
`scripts/01_characterise_data.py` → `figures/01_motion_profiles.png`

| Recording | Rate | Reference body | Motion | Noise at rest (relative pose, frame to frame) |
|---|---|---|---|---|
| Polhemus 1–4 | 120 Hz | sensor 1 static (≤0.7° drift) | ~88–90° out **and back** (file `1_..._oneway` is cut to one way) | ~0.003°, ~0.001 cm |
| Polhemus 5 | 120 Hz | **sensor 1 is the mover**, sensor 2 static | 90° out and back | same |
| Vicon servo ×3 | 100 Hz | femur cluster static | ~90° out and back | ~0.02° at rest, far more while moving |

- Vicon positions are in mm (`POINT:UNITS`). The Polhemus CSV carries no unit; it is
  taken to be cm, which puts the moving sensor 16.7 from the hinge axis and matches the
  beam in the test video. The hard-coded plane size (2) and axis length (20) in the
  original plot are therefore 10× smaller, relative to the rig, on Vicon data than on
  Polhemus data.
- Polhemus timestamps are host arrival times: gaps range 0.02–30 ms around a true 8.33 ms.
  No frames are missing (mean gap = 1/120 s exactly), so time should be regularised
  before anything is divided by `dt`.
- Polhemus jitter is tiny, but the pose wanders slowly by 0.03–0.08° and 0.005–0.016 cm
  over 2 s even "at rest" (the beam is hand-held). Errors on this rig are systematic,
  not random.
- The Vicon tibia cluster (`ankle`, `lateral_condyl`, `medial_condyl`) has a 35 mm
  baseline between the two condyle markers and deforms by up to 3.3 mm
  (`servo_X_20Z`); single markers jump up to 5.7 mm in one frame, producing
  one-frame orientation glitches of up to 9.5°. The femur cluster has five markers but
  only three are used.

### F2 — The fingers reproduce with the original code — VERIFIED
`scripts/02_reproduce_fingers.py` → `figures/02_original_cross_sections.png`

Original pipeline, `incremental_angle`, step 10°, outbound phase only:

| Recording | Mean angle FHA–AHA | Cross-section |
|---|---|---|
| Polhemus 1, pure hinge | 0.24° | tight cluster, ±0.1 cm |
| Polhemus 3, hinge + x | 11.1° | fingers, ±2 cm |
| Polhemus 4, hinge + z | 11.1° | fingers, ±3 cm |
| Vicon servo X | 1.7° | cloud, ±3 mm |
| Vicon servo X + 20° Y | 8.6° | fingers and jumps, ±30 mm |
| Vicon servo X + 20° Z | 9.6° | fingers and jumps, ±30 mm |

### F3 — The fingers are in the motion, not invented by the method — VERIFIED
This is the central finding. Three lines of evidence; (a) and (b) are independent of
each other, (c) is a consistency check.

**(a) Ground truth.** `scripts/04_synthetic_original.py` → `figures/04_synthetic_original.png`.
The original code was run on a simulated rig (`FHA/FHA_plotly_fast/synthetic.py`) whose
true helical axis is known analytically at every frame, with Polhemus-level noise. The
four cases differ only in how a 20° second rotation is delivered while the hinge turns 90°:

| Second rotation | Original miFHA cross-section | Direction error vs true axis (window middle) |
|---|---|---|
| none | tight cluster | 0.015° |
| steady, fixed ratio | **smooth arc** | 0.06° |
| phased in smoothly | **smooth curve** | 0.33° |
| in 8 bursts | **8 fingers** | — (true axis itself swings 0–50°) |

With smooth input the original code returns exactly the gradual deviation one expects.
Fingers appear only when the second rotation comes in bursts — and then the true
instantaneous axis swings *more* (0–50° tilt) than the miFHA reports (5–15°), because
the 10° window averages it.

**(b) The real recordings contain bursts.** `scripts/03_rate_ratio.py` → `figures/03_rate_ratio.png`.
Rotation rates were measured directly from the recordings with short central differences
(±0.1 s), independently of any FHA code, and split into a primary rate (about the mean
axis) and a secondary rate (across it):

| Recording | Secondary/primary rate ratio, 10th–90th percentile | Tilt that implies | Bursts of secondary rate (peak finder, prominence ≥ 1°/s) |
|---|---|---|---|
| Polhemus 1, pure hinge | 0.00–0.01 | 0.1–0.6° | 0 |
| Polhemus 3, hinge + x | 0.08–0.48 | 4.8–25.8° | 12, peaks 1.7–6.6°/s |
| Polhemus 4, hinge + z | 0.08–0.44 | 4.6–23.5° | 17, peaks 1.4–9.2°/s |
| Vicon servo X only | 0.01–0.13 | 0.7–7.5° | 16, peaks 1.1–3.9°/s — **noise floor** |
| Vicon servo X + 20° Y | 0.06–0.49 | 3.3–26.2° | 27, of which 9 above 6°/s (up to 14.7°/s) |
| Vicon servo X + 20° Z | 0.06–0.44 | 3.5–23.7° | 32, of which 13 above 6°/s (up to 22.6°/s) |

In Polhemus file 3 the secondary rate arrives as 12 separate pulses, dropping to near
zero in between, while the primary rate keeps going. On Polhemus this measurement has
a signal-to-noise ratio near 100 (rate noise ≈ 0.02°/s). On Vicon it does not: the
single-servo recording, which has no second rotation, already shows "bursts" of up to
3.9°/s from marker noise alone, so on the Vicon files only the bursts above ~6°/s stand
out from that floor — and finding F9 shows that even those coincide with marker error.
The Vicon recordings therefore do not add independent support; the case rests on the
simulation and the Polhemus recordings.

**(c) The miFHA tilt equals the rate ratio.** Same script. The tilt of each original
miFHA axis matches the tilt predicted from the measured rates over the same window:
r = 0.997–1.000 with an RMS difference of 0.46° (file 3) and 0.73° (file 4), against
swings of ±15°. Each finger lines up with one pulse of secondary rotation. On Vicon the
agreement is looser (r = 0.95–0.99, RMS 2.2–3.4°) because both sides of the comparison
carry marker noise.

Caveat on (c): the FHA of a window is, to first order, the direction of the net
rotation over that window, so agreement with integrated rates is expected by
construction. What (c) adds is that nothing *else* — noise, an instability, a bug —
is needed to explain the fingers. The independent evidence is (a) and (b).

**Why.** A helical axis — finite or instantaneous — points along the rotation that
happens *during its window*. For a primary rate ω∥ and a secondary rate ω⊥ it is tilted
from the primary axis by `atan(ω⊥ / ω∥)`. It reports the **ratio of the two rates**, not
how much secondary rotation has accumulated. "Introducing the second rotation
gradually" gives a gradual deviation only if that ratio changes gradually.

**Why the rigs produce bursts.**
- Polhemus rig: the test video shows one hand lowering the hinged beam while the other
  twists the sensor block on top of it. Two hands driving two independent degrees of
  freedom cannot hold a constant rate ratio.
- Servo rig: **not established.** A uniform ramp of 20° over the 5.1 s movement would
  be a steady 3.9°/s; the recordings show a 1–3°/s baseline with isolated bursts of
  6–15°/s (X + 20° Y) and 6–23°/s (X + 20° Z). It is tempting to read those as the
  second servo moving in jumps, but finding F9 shows the bursts coincide with the
  tibia marker cluster changing shape (r = 0.84 on X + 20° Z), which a rigid rotation
  cannot cause. On the Vicon two-axis files, real uneven motion and marker error
  cannot be separated with the present three-marker cluster.

A consequence worth stating, though it is reasoning and was not measured here: where a
second rotation is *mechanically coupled* to the first (the screw-home rotation of the
knee is the usual example), the ratio is set by joint geometry, and to that extent it
does not depend on how evenly the subject moves. The bench rigs have two *independent*
degrees of freedom, so their pattern depends on timing. A rig meant to validate the
method on coupled motion should couple the two rotations mechanically (section 6).

### F4 — Tagging each FHA at its window start biases it in time — VERIFIED (H6)
`scripts/04_synthetic_original.py`. Against ground truth, the original miFHA matches
the true axis at the window **middle** to 0.06° (steady case) and 0.33° (phased-in
case), but compared with the true axis at the window **start** — the time the original
code assigns — the error is 1.05° and 2.1° (max 15°).

### F5 — Windows close early on Vicon data — VERIFIED (H1)
`scripts/02_reproduce_fingers.py`. Requested step 10°; net rotation actually spanned:

| Recording | Median | Minimum | Windows below 8° |
|---|---|---|---|
| Polhemus 1 | 9.88° | 9.11° | 0% |
| Polhemus 3 | 9.60° | 8.75° | 0% |
| Vicon servo X | 9.44° | 8.17° | 0% |
| Vicon servo X + 20° Y | 8.54° | 5.76° | 13.9% |
| Vicon servo X + 20° Z | 8.50° | 4.57° | 31.0% |

Path length exceeds net rotation by 1.6% on Polhemus file 1 (89.5° vs 88.1°) but by
16–45% on the Vicon files (208–272° vs 180–187° for out and back). On Polhemus the
effect is small; on Vicon it is large.

### F6 — Real tracker error is slow-varying, not white — VERIFIED
`scripts/calibrate_noise.py`; the resulting presets live in
`FHA/FHA_plotly_fast/synthetic.py` (`NOISE_PRESETS`) and are used by script 05.

White frame-to-frame jitter alone cannot reproduce the real recordings. Jitter large
enough to match the window statistics of the Vicon single-servo file (σ = 0.02° per
axis per body) gives a simulated FHA scatter of 0.33°; the real scatter is 1.67°.
Adding a smooth, slowly varying error ("wander") reproduces both:

| | Net rotation of 10° windows: median / 5th pct | Mean angle FHA–AHA | Roughness of tilt series |
|---|---|---|---|
| Polhemus file 1 (real) | 9.88° / 9.31° | 0.24° | 0.163 |
| `polhemus` preset: jitter 0.002°, wander 0.03° over 0.4 s | 9.93° / 9.25° | 0.27° | 0.166 |
| Vicon servo_X (real) | 9.44° / 8.73° | 1.67° | 0.241 |
| `vicon` preset: jitter 0.017°, wander 0.11° over 0.06 s | 9.54° / 6.11° | 1.82° | 0.248 |

("Roughness" = std of the frame-to-frame change of the tilt series divided by its std.
The Vicon preset does not reproduce the lower tail of the window sizes.)

Consequence: the error of a helical axis on these instruments is set by pose error that
persists over many frames. It scales as 1/step and is not reduced by averaging frames.

### F7 — Benchmark against ground truth — VERIFIED
`scripts/05_benchmark_synthetic.py` → `figures/05_benchmark_synthetic.png`.
Simulated rig, calibrated error presets, 8 trials each, 10° windows. "Tracking error" is
against the true instantaneous axis at the frame the method assigns the axis to;
"noise error" is against the same method on noise-free poses.

Tracking error, mean (deg):

| Second rotation | Tracker | Original | Net rotation, centred | + all-frames fit |
|---|---|---|---|---|
| none (pure hinge) | Polhemus-like | 0.241 | 0.240 | 0.247 |
| none (pure hinge) | Vicon-like | 1.556 | 1.411 | 1.827 |
| steady | Polhemus-like | 1.049 | 0.230 | 0.237 |
| steady | Vicon-like | 1.784 | 1.400 | 1.810 |
| phased in | Polhemus-like | 2.221 | 0.517 | 0.410 |
| phased in | Vicon-like | 3.030 | 1.669 | 1.960 |
| pulsed | Polhemus-like | 17.63 | 14.82 | 11.47 |

- **Assigning each axis to the window middle (H6) is the fix that matters** whenever the
  axis moves: 4.6× lower tracking error for a steady second rotation and 4.3× for a
  phased-in one on Polhemus-like data (50× and 6× with an ideal, jitter-only tracker).
  On a pure hinge it changes nothing, as expected.
- **Net-rotation windows (H1)** cut the noise error by 9% on Vicon-like data
  (1.556° → 1.411°) and do nothing on Polhemus-like data. The benefit is real but modest.
- **Using every frame in the window (H8) — REJECTED as a default.** It is 3× better
  against pure jitter (0.014° → 0.005°), makes no difference on Polhemus-like error, and
  is 29% *worse* on Vicon-like error (1.411° → 1.827°). With slow-varying error a
  window holds only a few independent samples, and a least-squares slope is then less
  efficient than the difference of the two end poses. It stays available as
  `estimator="fit"` for instruments whose error really is frame-to-frame jitter.
- The "pulsed" row is not an estimation error: the true instantaneous axis swings
  0–50° and any 10° window averages it.

### F8 — Before/after on the real single-axis recordings — VERIFIED
`scripts/06_real_single_axis.py`. On a pure hinge all axes should be one line, so the
spread is the error. Spread is computed identically for every method (lab frame, every
degree of joint rotation weighted equally, sign-safe average axis, least-squares pivot).

| Recording | Original: its own summary | Original axes, common metric (angle / max / distance) | Revised, centred (angle / max / distance) |
|---|---|---|---|
| Polhemus 1, one way | 0.24° | 0.243° / 0.60° / 0.044 cm | 0.243° / 0.59° / 0.044 cm |
| Polhemus 1, out and back | **69.88°** | 0.270° / 8.45° / 0.071 cm | 0.247° / 0.66° / 0.041 cm |
| Polhemus 5 (sensor 1 moves), out and back | **82.07°** | 0.481° / 15.07° / 0.076 cm | 0.407° / 1.07° / 0.059 cm |
| Polhemus 2 (hinge + sliding), out and back | **77.76°** | 1.143° / 89.93° / – | 0.926° / 3.37° / – |
| Vicon servo X, outbound | 1.67° | 1.861° / 6.27° / 1.083 mm | 1.707° / 5.88° / 0.997 mm |
| Vicon servo X, out and back | **83.85°** | 1.952° / 72.07° / 1.979 mm | 1.667° / 5.90° / 0.974 mm |

- **On clean one-way Polhemus data the original estimator was already as good as the
  revised one** (0.243° either way). The miFHA result shown at ISB 2025 stands.
- **On any out-and-back recording the original's average axis collapses (H4 VERIFIED).**
  It averages direction vectors that flip sign on the way back, so they cancel; the
  code then reports 70–84° "mean angle" on a pure hinge, and the plane and
  cross-section built on that axis are meaningless.
- **Windows that straddle the turn-around give garbage axes (H1 VERIFIED):** up to 8°,
  15°, 72° and 90° off on these files. Their path length is 10° but their net rotation
  is near zero. The revised windows drop them; the worst remaining axis is 0.7–5.9° off.
- All-frames fit on real data: 0.247° vs 0.243° (Polhemus 1), 2.05° vs 1.71° (Vicon) —
  no gain, then worse, exactly as the calibrated simulation predicted (F7).
- Least-squares segment frames (`vicon_import.build_reference_frames_lsq`) reduce the
  Vicon spread from 1.707° to 1.602° with the same three markers per segment, and to
  1.600° with all five femur markers. A 6% gain: the moving cluster, not the fit, is
  the limit (F9).

### F9 — The Vicon tibia cluster is not rigid enough — VERIFIED
`scripts/07_vicon_cluster.py`. Residuals are from a least-squares rigid-body fit of the
three tibia markers to their median shape over the recording.

| Recording | Tibia rigid-fit residual (median / max) | Inter-marker distance range while moving | Correlation: cluster-shape change rate vs secondary rotation rate |
|---|---|---|---|
| servo X | 0.047 / 0.146 mm | 0.23–0.50 mm | 0.50 |
| servo X + 20° Y | 0.221 / 0.251 mm | 0.56–1.08 mm | 0.53 |
| servo X + 20° Z | 0.177 / 1.025 mm | 0.84–2.97 mm | **0.84** |

- The two condyle markers are 35 mm apart. A 0.3 mm error on that baseline is about
  0.5° of orientation, against roughly 2.2° of second rotation per 10° window.
- On the single-servo file the measured "secondary rate" (median 0.7°/s) is the same
  size as the apparent rotation implied by cluster-shape change (0.6°/s), consistent
  with it being marker error.
- On servo X + 20° Z the bursts of secondary rotation coincide with the cluster changing
  shape (r = 0.84). A rigid rotation cannot change inter-marker distances, so those
  bursts are at least partly artefact. (A marker that is loosely attached and shaken
  by a real jerk would produce the same correlation; the data cannot tell which.)
- **One marker is out of place for six seconds of servo X + 20° Z.** From frame 710 to
  1312 (7.10–13.12 s: end of flexion, the hold, start of the return) the
  ankle–lateral condyle distance reads 98.4 mm instead of 95.8–96.2 mm, and the
  rigid-fit residual is 0.99 mm against 0.15 mm in the rest of the recording. The
  marker enters that state with a 1.9 mm jump (2.9° of apparent rotation in one frame)
  and leaves it with a 5.7 mm jump (9.4° in one frame, the largest glitch in the data
  set).
- A three-marker cluster flags such errors only weakly: a marker 5.7 mm out of place
  raised the residual by less than 1 mm, because a displacement perpendicular to the
  plane of three markers changes their distances only to second order and is mostly
  absorbed as a rotation. A fourth, non-coplanar marker would make it obvious.
- Of the 11 one-frame orientation jumps above 0.8° in the two-axis files, 6 are spikes
  that return within three frames (single markers moving 0.4–1.3 mm) and 5 are steps
  that persist. The single-servo file has none.

**Implication:** the wild patterns on the Vicon two-axis files should not be interpreted
as joint kinematics until the moving cluster is rebuilt (rigid plate, ≥4 non-coplanar
markers, both baselines ≥ 100 mm). The rigid-fit residual is worth plotting next to
any Vicon result; the dashboard does so.

### F10 — Two further defects of the original, on concrete cases — VERIFIED (H2, H3)
`scripts/10_original_defects.py`.

- **H3.** `calculate_FHA_vectorized` returns 44° for a 136° rotation, 30° for 150°, 10°
  for 170°. Harmless for 10° windows; wrong for any large displacement (for example a
  reference-pose axis of a shoulder elevation).
- **H2.** With `step = 5` on `servo_X_20Z`, the window `[1312, 1312]` compares a frame
  with itself. The "axis" returned has length 0.00026 and is drawn in the
  cross-section at (212, −313) mm.

### F11 — The original cross-section mostly shows direction, scaled by an arbitrary lever arm — VERIFIED (H5)
`scripts/09_plane_placement.py`; the two views are compared side by side in
`figures/08_two_axis_views.png` (columns 1 and 3).

The original plane sits 10 units along the average axis from the mean of the points
`s` — the feet of the perpendiculars dropped from sensor 1 onto each axis. Compared
with the least-squares pivot (the point the axes actually converge on):

| Recording | Original plane, distance from pivot along the axis | Pattern RMS radius at the pivot | at the original plane |
|---|---|---|---|
| Polhemus 1, pure hinge | 10.1 cm | 0.043 cm | 0.078 cm (1.8×) |
| Polhemus 3, hinge + x | 6.4 cm | 0.321 cm | 1.552 cm (4.8×) |
| Polhemus 4, hinge + z | 10.1 cm | 1.079 cm | 2.227 cm (2.1×) |
| Vicon servo X | 16.8 mm | 1.000 mm | 1.283 mm (1.3×) |

Away from the pivot, an axis tilted by τ lands `distance × tan τ` off-centre, so the
picture mixes "where is the axis" with "which way does it point", in proportions set by
where the plane happens to be — which depends on the sensor placement and on the
length unit (10 units is 10 cm on Polhemus data and 10 mm on Vicon data). The revised
pipeline separates the two: a **position map** on the plane through the pivot, and a
**direction map** in degrees, which has no lever arm and no unit.

### F12 — What two-axis motion looks like in the revised views — VERIFIED
`scripts/08_two_axis_views.py` → `figures/08_two_axis_views.png`.

"Doubling back" = path length of the trajectory in the direction map divided by its
end-to-end distance (1 = straight run; large = fingers).

| Case | miFHA 10° | miFHA 20° | miFHA 30° | Reference pose |
|---|---|---|---|---|
| Simulated, steady second rotation | 2.09 | 1.34 | 1.16 | 1.23 |
| Simulated, second rotation in 8 bursts | 7.72 | 5.59 | 4.06 | 3.95 |
| Polhemus 3, hinge + x by hand | 5.16 | 2.31 | 1.77 | 2.43 |
| Polhemus 4, hinge + z by hand | 6.72 | 5.13 | 3.11 | 2.53 |

- A steady second rotation gives a smooth arc in the direction map with any window.
- Fingers shrink as the window grows, because a window wider than the spacing of the
  bursts averages them. On Polhemus file 3 a 30° window gives a nearly monotonic
  trajectory. **The window size is the control for this trade-off**: resolution against
  sensitivity to uneven driving.
- The reference-pose FHA accumulates instead of differentiating, so it is calmer, but
  it is not immune: each burst shows as one tooth of a sawtooth whose size falls off as
  1/(total angle).
- **CORRECTED (see F15).** This entry first said that the "coupling curve" — rotation
  off the *average* axis against rotation about it, the last column of the figure —
  shows the second rotation building up, and that in recording 3 the second rotation
  "rises to about 10° over the first 45° of hinge rotation and is then taken back out,
  ending near −5°". That reading was wrong. It came from measuring against the
  recording's own average axis, which lies 15.9° from the hinge. Measured against the
  hinge, the second rotation of recording 3 rises to 23.8° over the first 45° and then
  **stays**; nothing is taken back out. The long sweep across the direction map is the
  second rotation stopping (rate ratio falling to zero), seen from an average axis that
  sits between the two regimes.

### F13 — The error reduction reported at ISB 2025 is reproduced — VERIFIED
`scripts/11_isb_numbers.py`. Pure-hinge Polhemus recording (one way), mean angle between
the axes and their average axis. ISB 2025 reported FHA 5.01°, fiFHA 0.33°, miFHA 0.34°.

| Method | Original, movement only (frames 480–1200) | Original, with 3 s at rest (100–1248) | Revised, with 3 s at rest |
|---|---|---|---|
| FHA, consecutive frames | 3.43° | **29.17°** | 2.27° |
| fiFHA, 5° | 0.32° | 0.26° | 0.28° |
| miFHA, 5° | 0.30° | 0.26° | 0.27° |
| fiFHA, 10° | 0.25° | 0.21° | 0.19° |
| miFHA, 10° | 0.26° | 0.24° | 0.23° |

- With a 5° step on the movement alone, the original code gives 0.32° and 0.30°, against
  the 0.33° and 0.34° on the slide. The plain-FHA figure depends strongly on the frame
  range: 3.4° for the movement alone, 29° once frames at rest are included, because every
  resting frame contributes a pure-noise axis with full weight. The exact range behind
  the 5.01° on the slide is not known.
- The revised code weights consecutive-frame axes by sin²(φ/2), so resting frames barely
  count (2.3° either way).
- The roughly tenfold reduction from the angle-based windows is real and does not depend
  on the corrections made here.
- The "revised" column uses the dashboard's own measure (axes in the reference sensor's
  frame, every degree of motion weighted equally). F8 compares methods in the lab frame
  with 1° bins, which is why it quotes 0.243° where this table says 0.23°.

### F14 — The window threshold: applied to net rotation, and now met by every window — VERIFIED
`scripts/12_window_threshold.py` → `figures/12_window_threshold.png`.
Raised by the user after the first round: is the threshold computed by summing each
small step, which is itself unstable?

For the original code, yes. What conditions an FHA is the NET rotation between the two
poses it is computed from. The original closes a window when the *sum* of frame-to-frame
angles exceeds the request; that sum grows with every frame of jitter and with every
reversal, whether or not the two end poses have moved apart. Net rotation between the two
poses each window actually compares, request 10°, whole recordings (out and back):

| Recording | Summed steps over the file / largest net rotation | Original: min, share below 9°, share below 5° | Half-windows only: min | Current rule: min / median |
|---|---|---|---|---|
| Polhemus 1, pure hinge | 178.8° / 88.2° | 0.04°, 10.6%, 3.4% | 10.01° | 10.01° / 10.14° |
| Polhemus 3, hinge + x | 187.3° / 87.4° | 0.05°, 16.5%, 3.0% | 9.41° | 10.00° / 10.06° |
| Polhemus 4, hinge + z | 199.2° / 90.3° | 0.82°, 36.1%, 2.8% | 9.61° | 10.00° / 10.08° |
| Vicon servo X | 208.3° / 89.8° | 0.12°, 44.5%, 14.1% | 9.96° | 10.00° / 10.18° |
| Vicon servo X + 20° Z | 271.7° / 93.4° | 0.00°, 89.0%, 25.9% | 9.55° | 10.00° / 10.17° |

- **Original:** 11–89% of windows compare poses less than 9° apart and 3–26% less than
  5° apart; the worst compare almost the same pose (the windows that straddle the
  turn-around). F5 quoted the outbound movement only, where the shortfall is smaller.
- **Current rule** (`fha_core.windows_centred`): for every frame, take the nearest frame
  before and the nearest after that are each half the request away in net rotation; then
  widen, on whichever side reaches less far, until the two ENDS are a full request
  apart. Every window meets the request exactly or better, for requests of 2, 5, 10, 20
  and 30° alike (minima 2.000, 5.000, 10.000, 20.000, 30.000 on recording 3).
- The first revision of this work stopped after the half-windows. Where the direction of
  rotation changes inside a window the ends were then up to 6% short (9.41° on
  recording 3). That gap is now closed.
- Net rotation is read straight from the two poses. Nothing is added up, so noise enters
  once per pose (0.003° on Polhemus, about 0.1° on Vicon, against a 10° window) instead
  of once per frame.
- A frame gets no window where a half would have to span three quarters of the request
  on its own: at a reversal (the halves fold back), and where the pose jumps by that
  much in one frame. On Vicon servo X + 20° Z this drops the windows that swallowed the
  9.4° marker jump (927 → 871 windows).
- **Effect on accuracy: none worth the name on these recordings** — tables F7, F8, F11
  and F12 moved in the third digit. What changed is that the requested window is now a
  guarantee rather than a typical value.

### F15 — Adding up the rotation shows a burst as a step that stays, but only against the primary axis — VERIFIED
`scripts/13_build_up.py` → `figures/13_build_up.png`.
The user's idea after the first round: a burst makes a finger because the axis of each
window tilts while the burst lasts and returns afterwards. Keep track of how much
rotation has *accumulated* instead, so that a burst adds to the total and stays.

The idea is right, with one condition that the first version of the dashboard got wrong.

Simulated rig, Polhemus-like error, 90° about the hinge with 20° of second rotation.
Accumulated rotation away from the axis, maximum / value at the end:

| Second rotation | (1) off-axis part of each window, summed, against the recording's own average axis | (2) swing about the recording's own average axis | (3) swing about the hinge axis, taken from a separate pure-hinge recording |
|---|---|---|---|
| steady | 3.51° / 0.41° | 3.68° / 3.38° | 19.96° / 19.96° |
| phased in | 10.23° / 0.64° | 10.06° / 7.80° | 19.96° / 19.96° |
| in 8 bursts | 3.62° / 0.12° | 4.17° / 3.36° | 19.96° / 19.96° |

Method (3) follows the true second rotation to within 0.13° throughout, as a straight
line, a late ramp or a staircase. Methods (1) and (2) cannot: **the average axis of a
movement is, by definition, the direction of its total rotation — second rotation
included** (here it lies 11.6–12.1° from the hinge). Rotation away from it has to sum
to nearly nothing. This is not a matter of how the adding-up is done; it is the choice
of axis. Method (2) is what the "How the second rotation builds up" panel showed before.

Real recordings. Premise of (3): the hinge, written in the frame of the reference
sensor, is the same in every recording of the session.

| Recording | Own average axis, angle to the hinge of recording 1 | Reference sensor vs recording 1 |
|---|---|---|
| 1, one way | 0.04° | 0.06°, 0.01 cm |
| 2, hinge + sliding | 0.38° | 0.51°, 0.11 cm |
| 3, hinge + x | 15.92° | 0.10°, 0.02 cm |
| 4, hinge + z | 20.25° | 1.40°, 0.34 cm |

Build-up over the outbound movement, maximum / end:

| Recording | (1) windows summed, own axis | (2) swing, own axis | (3) swing, hinge of recording 1 |
|---|---|---|---|
| 1, pure hinge | 0.12° / 0.03° | 0.14° / 0.11° | 0.15° / 0.14° |
| 2, hinge + sliding | 0.44° / 0.35° | 0.45° / 0.38° | 0.57° / 0.36° |
| 3, hinge + x by hand | 10.04° / 0.92° | 11.41° / 7.57° | **23.83° / 23.83°** |
| 4, hinge + z by hand | 7.15° / 1.53° | 8.87° / 5.34° | **31.42° / 31.35°** |

- Against the hinge, recordings 3 and 4 build up 23.8° and 31.4° of second rotation — the
  "xx deg" of their file names — in steps, and the value at the end is the maximum:
  nothing is undone. Recordings 1 and 2, which have no second rotation, stay below 0.6°.
- Seen from a frame that turns with the primary rotation, the tilt runs out along one
  straight line in both recordings (bottom row of the figure): one axis, carried by the
  beam. A wrong primary axis would make that path curl.
- The measure is the angle between the primary axis and the copy of it carried by the
  moving segment, relative to the first selected frame. It depends only on those two
  poses, so it does not drift and noise does not accumulate in it.
- It needs a primary axis from outside the recording under study: here a recording of
  the hinge alone, made with the reference sensor mounted the same way. For recording 4
  the sensor sat 1.4° and 0.34 cm differently, so its 31.4° carries an uncertainty of
  about that size.
- This corrects F12, which read recording 3's second rotation as being "taken back out".

### F16 — Finding the primary axis from the same recording: only when the two rotations vary independently — tested, NOT ADOPTED
`scripts/14_two_axis_fit.py`.
Can the reference recording be dispensed with? A two-axis model was fitted to the
recording under study: one axis fixed in the reference body, one fixed in the moving
body, at a constant angle to each other, `R_t = Rot(a, α_t) · R_ref · Rot(b, β_t)`. The fit
minimises the variance of the angle between `a` and `R_t b` (grid search over both
directions, then least squares). Minimising the variance of its cosine instead collapses
onto two nearly parallel axes.

| Simulated second rotation | Nearest fitted axis to the true hinge: no error / Polhemus-like / Vicon-like | Second rotation found (true 20°) |
|---|---|---|
| in bursts | 0.00° / 0.13° / 0.81° | 19.3° / 19.0° / 21.8° |
| phased in | 0.00° / 0.38° / 1.71° | 17.6° / 18.0° / 17.0° |
| steady (fixed ratio) | **5.66° / 6.80° / 3.01°** | **35.4° / 36.8° / 23.2°** |
| none (pure hinge) | axes come out parallel | 0° / 122° / 417° |

(The amounts are over the frames sampled, which start and end half a window inside the
movement; that accounts for 19.3° and 17.6° in the error-free cases.)

| Real recording (outbound) | Nearest fitted axis to the hinge of recording 1 | Axes apart | Second rotation found | By F15 | Misfit |
|---|---|---|---|---|---|
| 1, pure hinge | 0.61° | 1.8° | 27.3° | 0.15° | 0.004° |
| 3, hinge + x by hand | **0.42°** | 90.8° | 24.1° | 23.8° | 0.014° |
| 4, hinge + z by hand | 6.29° | 103.9° | 23.7° | 31.4° | 0.260° |

- **Recording 3 is an independent confirmation of F15.** Without being given the hinge,
  the fit finds it to 0.42° and puts the second rotation at 24.1° about an axis at
  right angles to it; the reference-recording route gives 23.8°.
- **Steady coupling is not identifiable.** With a fixed ratio, a different pair of axes
  fits with zero misfit even on error-free data, and gives 35° where 20° is true. When a
  second rotation is tied to the first, the movement does not say which part is
  "second"; that takes a convention from outside (a calibration movement, anatomy).
- **A pure hinge is degenerate**: the two axes come out parallel and the split between
  them is arbitrary.
- **Recording 4 comes out biased** (6.3° off the hinge, 23.7° instead of 31.4°), with a
  misfit twenty times that of recording 3: the hand-driven motion is not exactly two
  hinges, and the fit bends the axes to absorb that.
- It works where the test recordings happen to sit — bursts — and fails where a real
  coupled joint would sit. It is therefore not offered in the dashboard; the build-up
  is measured against a reference recording, or against the recording's own average
  axis with a warning.

### F17 — A smoothing filter: what it removes, and how to see whether that is noise — VERIFIED
`scripts/15_smoothing.py` parts A–E, figure `figures/15_smoothing.png`. Dashboard:
`figures/16_dashboard_smoothing_knee.png`, `figures/17_dashboard_smoothing_simulated.png`.

The request was a slider to smooth the motion, with the remark that reducing measurement
noise and suppressing genuine changes in the movement are different things. A low-pass
filter cannot tell them apart: it removes whatever changes faster than its cut-off. So
the work was to build the filter so that it adds no error of its own, and to show beside
it what it takes out.

**The filter.** `fha_core.smooth` is a low-pass with the cut-off defined the usual way
(amplitude down to 70.7% at the cut-off, no time shift). Its response is that of the
zero-lag fourth-order Butterworth filter used in movement analysis, to within 0.1% away
from the ends, so "low-pass filtered at 6 Hz" means what a reader expects. It is
computed as penalised least squares on second differences (the discrete cubic smoothing
spline), one banded solve. That gives no start-up transients at the two ends, accepts
uneven sampling, and works down to four frames. `smooth_poses` applies it to poses,
`smooth_motion` to a pair of bodies. In the dashboard the whole recording is filtered and
then cut, so the result at a frame does not depend on the frame range chosen.

Two measures are used throughout. *Removed*: the RMS angle between the measured and the
smoothed pose of the moving body relative to the reference body. *Frame-to-frame
jitter*: the random part of a single frame, from third- to fifth-order differences
(`fha_core.jitter`, see F19). A filter that removes no more than the jitter is removing
noise; what it removes beyond that is slower tracker error, movement, or both.

**(a) What is filtered matters at low cut-offs.** A noise-free hinge turned stop-and-go,
sensor 20 cm from the axis; how far the filter itself moves the axes (cm, worst axis):

| Cut-off (Hz) | 6 | 3 | 2 | 1 | 0.5 | 0.3 |
|---|---|---|---|---|---|---|
| Positions and orientation filtered separately | 0.0005 | 0.006 | 0.018 | 0.11 | 0.34 | 1.04 |
| Screw coordinates of the relative pose | 0 | 0 | 0 | 0 | 0 | 0 |

Filtering positions pulls a point on an arc toward the chord, by an amount that changes
with speed, so the axis moves. That is what filtering marker or sensor signals amounts
to. At the customary 6 Hz it is harmless; at the cut-offs needed to flatten fingers it
is millimetres. The filter here acts on the screw coordinates of the pose relative to
the pose midway through the movement (rotation vector and translation part of the
logarithm). Motion about one fixed axis is a straight line in those coordinates, so a
hinge stays exactly a hinge at any cut-off.

**(b) Against ground truth** (simulated rig, mean angle between each axis and the true
axis, degrees, 10° windows, six trials):

| | off | 6 Hz | 2 Hz | 1 Hz | 0.5 Hz | 0.3 Hz |
|---|---|---|---|---|---|---|
| Polhemus-like error, pure hinge | 0.262 | 0.260 | 0.260 | 0.254 | 0.217 | 0.167 |
| Polhemus-like, second rotation phased in | 0.519 | 0.518 | 0.531 | 0.652 | 1.225 | 2.212 |
| Vicon-like error, pure hinge | 1.384 | 1.351 | 1.125 | 0.793 | 0.452 | 0.321 |
| Vicon-like, steady second rotation | 1.365 | 1.328 | 1.102 | 0.780 | 0.479 | 0.497 |
| Vicon-like, second rotation phased in | 1.659 | 1.624 | 1.407 | **1.213** | 1.505 | 2.485 |

- On the Polhemus-like rig there is nothing to gain: its error is slow (F6), and passes
  any filter that passes the movement.
- On the Vicon-like rig a filter helps, down to about 1 Hz. Below that it flattens the
  real change of the axis and the error rises again.
- A pure hinge only ever improves, because the filter cannot move a hinge. That makes
  single-axis recordings a poor guide to the right cut-off for a joint.

**(c) The real single-axis bench recordings** (out and back; mean angle to the average
axis, degrees):

| | jitter | off | 6 Hz | 2 Hz | 1 Hz | 0.5 Hz | 0.3 Hz |
|---|---|---|---|---|---|---|---|
| Polhemus 1, pure hinge | 0.0022° | 0.241 | 0.239 | 0.238 | 0.234 | 0.216 | 0.199 |
| — removed by the filter (°) | | 0 | 0.007 | 0.032 | 0.086 | 0.273 | 0.624 |
| Vicon servo X | 0.021° | 1.684 | 1.641 | 1.572 | 1.323 | 0.904 | 0.766 |
| — removed by the filter (°) | | 0 | 0.051 | 0.113 | 0.199 | 0.393 | 0.721 |

**(d) The real two-axis bench recordings: do the fingers go?** Outbound movement;
"fingers" is the doubling back of F12 (path length in the direction map over its
end-to-end distance; 1 = none):

| | | off | 20 Hz | 6 Hz | 2 Hz | 1 Hz | 0.5 Hz | 0.3 Hz |
|---|---|---|---|---|---|---|---|---|
| Polhemus 3 (jitter 0.0029°) | removed (°) | 0 | 0.002 | 0.009 | 0.042 | 0.106 | 0.236 | 0.409 |
| | fingers | 5.16 | 4.84 | 4.48 | 3.89 | 3.18 | 2.44 | 2.02 |
| | mean tilt from average axis (°) | 13.40 | 13.40 | 13.40 | 13.39 | 13.37 | 13.37 | 13.16 |
| | second rotation built up (°) | 23.83 | 23.83 | 23.83 | 23.82 | 23.81 | 23.77 | 23.84 |
| Polhemus 4 (jitter 0.0028°) | removed (°) | 0 | 0.003 | 0.019 | 0.078 | 0.176 | 0.340 | 0.478 |
| | fingers | 6.72 | 6.58 | 5.84 | 4.56 | 3.50 | 2.40 | 2.09 |
| | second rotation built up (°) | 31.42 | 31.42 | 31.42 | 31.41 | 31.39 | 31.39 | 31.38 |
| Vicon X + 20 Y (jitter 0.038°) | removed (°) | 0 | 0.039 | 0.105 | 0.237 | 0.420 | 0.670 | 1.076 |
| | fingers | 16.71 | 13.52 | 10.33 | 6.42 | 4.27 | 2.79 | 1.53 |
| | mean tilt from average axis (°) | 7.81 | 7.79 | 7.73 | 7.61 | 7.35 | 6.60 | 5.60 |

Polhemus 3 without any filter, wider window instead: fingers 5.16 at 10°, 2.31 at 20°,
1.77 at 30°.

- **The expectation holds: a filter shortens the fingers.** How it does so differs by
  instrument.
- **Polhemus: by removing movement.** The jitter is 0.003°. A 20 Hz filter removes
  about that much and the fingers go from 5.2 to 4.8: that part was noise. Everything
  further is the bursts of F3, which come at about 1.3 Hz. Halving the fingers takes
  0.5 Hz and removes 0.24°, eighty times the jitter. Before the movement starts the
  pose of recording 3 scatters by 0.057° while its frame-to-frame jitter is 0.0016°:
  nearly all of this tracker's error is slow, out of a filter's reach.
- **A wider window does the same job and leaves the data alone.** 20° windows give
  2.3, what 0.5 Hz gives. Window and filter both average; the window does it in degrees
  of joint rotation, which can be stated and compared between recordings, the filter in
  Hz, which depends on how fast the movement happened to be performed.
- **Vicon: partly by removing noise.** The fingers of this recording were partly the
  non-rigid cluster (F9), and go from 16.7 to 10.3 at 6 Hz with the mean tilt unchanged.
  Below 2 Hz the mean tilt itself falls: movement is going.
- **The build-up needs no filter.** It ends at 23.8° and 31.4° at every cut-off.

**(e) Recordings of people** (Polhemus, whole recording, 10° windows). *Path* is the
length of the path the axis direction travels in the direction map: wiggle makes it
long. *Flat stretch*: where what the filter removes grows by less than its cut-off falls
(slope below 1 on log-log axes), meaning little in the recording changes at that rate.

| Recording | Jitter (°) | Removed at 6 Hz (°) | Flat stretch | Path at 6 Hz | Mean angle to average axis: off → 6 → 2 → 1 Hz (°) |
|---|---|---|---|---|---|
| Knee flexion, open chain (6) | 0.0048 | 0.064 | 4–1.5 Hz | ×0.77 | 10.18 → 10.13 → 9.80 → 8.95 |
| Knee flexion, closed chain (7) | 0.0047 | 0.066 | none | ×0.88 | 14.06 → 13.98 → 13.39 → 12.04 |
| Knee flexion, closed chain (8) | 0.0033 | 0.049 | 6–2 Hz | ×0.80 | 11.17 → 11.06 → 10.81 → 9.93 |
| Knee flexion, open chain (9) | 0.0051 | 0.056 | 4–1.5 Hz | ×0.81 | 10.40 → 10.37 → 10.21 → 9.69 |
| Head rotation (10) | 0.0223 | 0.036 | 20–4 Hz | ×0.69 | 7.62 → 7.60 → 7.50 → 7.21 |
| Head rotation (11) | 0.0153 | 0.028 | 20–4 Hz | ×0.70 | 7.63 → 7.62 → 7.35 → 6.65 |
| Head lateral flexion (12) | 0.0226 | 0.032 | 20–4 Hz | ×0.86 | 13.49 → 13.50 → 13.29 → 12.65 |

The 74 further recordings of people in sub-folders of the Polhemus test data (50 s
each; read as a group, reported in aggregate; median, 10th–90th percentile):

| Cut-off | Removed (°) | Path | Mean angle to average axis |
|---|---|---|---|
| 20 Hz | 0.011 (0.007–0.021) | ×0.98 (0.96–1.00) | ×1.00 (1.00–1.00) |
| 6 Hz | 0.100 (0.074–0.162) | ×0.84 (0.75–0.90) | ×0.98 (0.96–1.00) |
| 3 Hz | 0.272 (0.175–0.377) | ×0.65 (0.52–0.75) | ×0.94 (0.88–0.98) |
| 2 Hz | 0.415 (0.272–0.567) | ×0.55 (0.41–0.67) | ×0.89 (0.83–0.97) |
| 1 Hz | 1.017 (0.619–1.424) | ×0.37 (0.25–0.51) | ×0.74 (0.65–0.90) |

Their jitter is 0.006° (0.004–0.015°); at 6 Hz the filter removes 16 times that
(6–27 times).

- **Head movement has a clean gap.** From 20 Hz down to 4 Hz the filter removes little
  more than the jitter; below 3 Hz the curve turns steep. A cut-off anywhere from 4 to
  10 Hz takes out the jitter, shortens the path (by 14–31% at 6 Hz) and leaves the
  spread of the axes where it was. The exact value hardly matters, which is what makes
  it defensible.
- **Knee flexion has more between 6 and 20 Hz than jitter** (0.05–0.07° at 6 Hz, ten
  to fifteen times the jitter), and in three recordings of four a flatter stretch
  below it. What that 6–20 Hz content is — the sensor shaking on soft tissue, tremor —
  the recording does not say; it comes and goes with the movement as fine, rapid
  oscillation (`figures/16_dashboard_smoothing_knee.png`, lower right).
- **From about 2 Hz down the spread of the axes shrinks** in every recording: by 2–5%
  at 2 Hz and 5–14% at 1 Hz in the seven, by 11% and 26% in the larger set. That is
  the movement of the axis being averaged away. Whether it is "noise" is a question
  about the joint, not about the instrument.
- **The bench two-axis recordings have no flat stretch** (slope 1.0–1.5 from 20 Hz to
  0.3 Hz on recording 3): quick and slow are not separate there, and no cut-off is
  better founded than another.

**(f) What the dashboard shows.** A slider under "Smoothing" (Off, then 20 to 0.2 Hz;
`smoothing_hz` in `config.py` sets where it opens). Beside it, one sentence: how much
that cut-off removes from the selected frames and how many times their frame-to-frame
jitter that is. In the direction map, the position map and the series, the axes without
smoothing stay visible as a grey line under the smoothed ones, and the maps keep their
scale while the slider moves. A card, "What the smoothing takes out", has the curve of
removed against cut-off with the jitter drawn in, and what the chosen cut-off removes at
each frame: fine, rapid grass is jitter or vibration; smooth humps that rise and fall
with the movement are the movement being flattened. The line of notes under the
controls and the CSV header state the cut-off. The reference recording's axis (F15) is
taken from unsmoothed data: it is an average over all its windows and a filter hardly
changes it.

### F18 — Arrival timestamps have to be turned into sample times before anything is filtered — VERIFIED
`scripts/15_smoothing.py` part F; `fha_core.even_time_base`.

Until now time only labelled the axes: windows are built on rotation, not on time. A
filter puts the timestamps into the arithmetic.

- The Polhemus stamps each sample when it reaches the host. In 72 of the 87 Polhemus
  recordings in the test data a straight line fits the stamps within four sample
  periods, and the loader already replaced them by that line (F1). In the other 15, all
  among the recordings of people, it does not: samples arrive in **bursts** (nothing for
  up to 91 ms, then up to eleven within 0.01 ms of each other); three recordings hold a
  real **pause** (4.7 s, 9.5 s and 62.3 s); in two the host **clock steps back** (by
  57 ms and 134 ms). In all 15 the true period is 8.333 ms (120.00 Hz), found from
  differences over 50 samples.
- Taken at face value, burst stamps say ten samples were taken within a tenth of a
  millisecond. A filter then flattens those ten and leaves a jump before them. A
  noise-free 90° sweep filtered at 6 Hz is moved by up to **0.94°** (RMS 0.24°) on
  burst stamps and by 0.0015° on the repaired time base. On the 13 real recordings whose
  recorded stamps at least increase, a 6 Hz filter removes 1.0 to 7.0 times as much on
  the recorded stamps as on the repaired ones (median 1.1); the excess is distortion.
  The two with a clock step cannot be filtered at all.
- `even_time_base` now takes such stamps apart. An arrival can be late but not early,
  so the true sample times run along the lower edge of the stamps. A lasting rise of
  that edge is time really lost and is kept as a pause; a lasting drop is the clock
  being set back and is ignored. Stamps that do not look like arrival times at all
  (far more breaks than a recording could have) are left as they are, and the loader
  says so. All 15 recordings come out at an even 120 Hz with their three pauses kept.
- `smooth` refuses times that do not increase, and missing values, with a message
  rather than a wrong answer.

### F19 — Few frames per degree: dynamic CT, or a small window on a fast movement — VERIFIED (simulated)
`scripts/15_smoothing.py` part G; `tests/test_fha_core.py`.

There are no CT data in the project, so this is simulation: a 90° movement with 20° of
second rotation phased in over 3.6 s, sampled in 24 or 48 frames, with an independent
error of 0.3° and 0.3 mm per axis on every frame, as if each volume were registered on
its own.

**(a) Coarse frames lost all their axes.** The second round's guard against pose jumps
(F14) dropped a window when one of its halves spanned three quarters of the window. A
half can only end on a frame. At 4° per frame, half of a 10° window is two frames, 8°:
every window was dropped and the dashboard showed "no axis could be formed". A half now
counts as a jump only if its last frame turns more than three times what nine frames in
ten turn. Coarse recordings keep their windows, which are then as narrow as the frames
allow, and the dashboard says so ("the windows span about 20° where 10° was asked").
On the test recordings and the simulator the windows are the same as before for
windows of 5° and more (16 recordings and 16 simulated cases compared, frame for
frame). At 1° the old rule had been dropping windows at peak speed for the same reason:
460 → 622 on knee recording 6, 1246 → 1546 on head recording 10.

**(b) The jitter can be estimated from a couple of dozen frames, roughly.**
`fha_core.jitter` takes differences of the third, fourth and fifth order, each of which
cancels smooth movement, and returns the smallest; if they disagree by more than half,
or there are fewer than 20 frames, it returns nothing. At 100 Hz and above the three
agree within a few percent (0.00310°, 0.00293°, 0.00288° on recording 3). With 24
frames it returns 0.50° where 0.52° is true; over 20 to 30 frames it is good to about a
third. "Frame to frame" is literal: at a few frames per second the whole registration
error of a frame is jitter, and so is any movement too quick for that frame rate.

**(c) What a filter buys with so few frames** (30 trials):

| | off | 2.5 Hz | 2 Hz | 1.5 Hz | 1 Hz | 0.7 Hz | 0.5 Hz |
|---|---|---|---|---|---|---|---|
| 24 frames (6 per second): pose error left (°) | 0.518 | 0.460 | 0.441 | 0.404 | **0.388** | 0.557 | 0.989 |
| — movement removed from the error-free series (°) | 0 | 0.023 | 0.034 | 0.067 | 0.194 | 0.471 | 0.952 |
| — axis against the true axis, 20° windows (°) | 2.47 | 2.31 | 2.27 | 2.18 | **2.06** | 2.17 | 2.55 |
| 48 frames (12 per second): pose error left (°) | 0.519 | 0.360 | 0.323 | **0.289** | 0.319 | 0.549 | 1.004 |
| — axis against the true axis, 20° windows (°) | 2.33 | 1.97 | 1.87 | **1.81** | 1.84 | 2.00 | 2.43 |

(Without any error on the frames the axis figure is 1.26–1.30°: a 20° window against
the instantaneous true axis.)

- The useful range is narrow, about 1 to 2 Hz for a movement of this length, and the
  gain modest: −16% on the axis with 24 frames, −22% with 48.
- One step further down (0.7 Hz) the filter removes more movement than error and the
  result is worse than with no filter.
- Cut-offs at or above half the frame rate do nothing, and the dashboard says so
  instead of pretending.

### F20 — Playing the motion back: what it takes to make it smooth — VERIFIED
`FHA/FHA_plotly_fast/tests/test_playback_browser.py` (headless Chrome), `tests/test_playback.py`;
`assets/playback.js`, `shapes.py`. Dashboard: `figures/09`–`11`, `16`, `17`
(Polhemus, simulator, knee) and `figures/18_dashboard_playback_vicon.png`.

Asked for: meshes of basic shapes for the two rigid bodies, playback of the motion, and
a vertical line on the plots showing where the playback is. Bone surfaces are to
replace the shapes later; none exist yet.

**What is drawn.** A sensor: a block on its origin, joined to the joint (the pivot of
the axes) by a bar of rectangular section, so that a turn about the bar's own length
shows. A Vicon marker cluster: a plate through the three markers that define its frame
and a small block on every marker, joined to the joint by a bar if the nearest marker
is more than a quarter of the cluster's size away. A shape is vertices and triangles in
the coordinates of its body; the page moves the vertices with the pose of that body, so
a bone surface in the same coordinates can replace it without other changes.

**Three things that were not obvious.**

- *Moving anything through Plotly's own calls redraws the whole figure.* Moving a shape
  with `relayout` recomputes every trace of its figure (shape positions are flagged
  as affecting the axis ranges): 5–17 ms per plot. `restyle` on the 3D view rebuilds
  every trace of the scene: about 20 ms. With all figures on screen that is 55–70 ms
  per step. Handing the moved traces straight to the scene's own objects, and sliding
  the drawn shape while updating Plotly's record of where it is, takes **1 to 2 ms** for
  all figures together (plotly.js 2.20, 3.4 and 3.5; Apple M1 Pro). The direct way uses
  Plotly internals, so each step checks what it needs and falls back to Plotly's calls;
  the browser test compares the two against numpy (they agree to the rounding of what
  is sent, under 0.001 cm or degree) and against each other on screen (within a pixel).
- *Dash 4 spends about 90 ms on a tick whose callback changes nothing*, and 140 ms on
  one that moves a slider and a line of text: a play loop written as Dash callbacks (a
  clock ticking, a callback moving the slider, another moving the figures) ran at 25
  frames a second under Dash 2.9 and 2.14 and at 7 under Dash 4.1, although moving the
  figures took 2.5 ms. The loop, the play button and the slider therefore run
  entirely in the browser; Dash delivers the poses once per analysis and is told where
  the playhead stopped. Playback then runs at the display's 60 frames a second under
  all three, and follows the clock, so a slow machine drops frames instead of slowing
  the movement.
- *The 3D view does not cut a mesh at its axis ranges.* The markers of the reference
  cluster of the Vicon rig sit 125–384 mm from the pivot (those of the moving cluster
  23–116 mm); framed whole, it shrinks the joint to a corner of the view, and left
  outside the ranges it is drawn across the whole picture. The view is therefore
  fitted to the axes and the moving body at every instant, the reference body may add
  a margin of that size, and the rest of it is cut off in the mesh itself. An axis
  left to fit itself also stretches and shrinks as the playhead passes its ends, so the
  scene, the build-up and the time axis of "Along the movement" have fixed ranges
  during playback.

**Which axis is "the axis at the playhead".** The one whose window holds the frame,
nearest centre first. At rest and before the first window is full there is none; the
fixed-lag and consecutive-window methods have a window at every frame, so with them an
axis is shown throughout, including the ill-conditioned ones at rest.

---

## 4. Status of the hypotheses

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | Windows close on path length, not net rotation | **VERIFIED.** Small effect on one-way Polhemus data; 9% more noise on Vicon-like data; garbage axes (up to 90° off) at every turn-around. Over whole recordings 11–89% of the original windows span less than 9° of the 10° asked for | F5, F7, F8, F14 |
| H2 | Off-by-one; a frame can be compared with itself | **VERIFIED** on `servo_X_20Z` with step 5° | F10 |
| H3 | Angle folded back beyond 135° | **VERIFIED** (150° returned as 30°) | F10 |
| H4 | Mean of direction vectors cancels on out-and-back motion | **VERIFIED.** Reports 70–84° mean angle on a pure hinge | F8 |
| H5 | Cross-section mixes position with direction × lever arm | **VERIFIED.** Pattern up to 4.8× larger than at the pivot | F11 |
| H6 | Axis tagged at window start, not middle | **VERIFIED.** 4–5× tracking error on smooth two-axis motion | F4, F7 |
| H7 | Plotted origin shifted along the axis by `d` | Confirmed by reading; cosmetic | — |
| H8 | Using only two poses per window wastes information | **REJECTED as a remedy.** Fitting all frames helps only against white jitter; no gain on Polhemus, 20–29% worse on Vicon | F6, F7, F8 |
| — | The fingers are created by the method | **REJECTED.** They are the rate ratio of the two rotations, faithfully reported | F3 |
| — | Adding up rotation instead of looking at the axis direction makes a burst a lasting step (user) | **VERIFIED**, provided it is measured against the primary axis and not the recording's own average axis | F15 |
| — | The primary axis can be found from the recording itself by fitting two axes | **REJECTED as a general tool.** Works for bursts, not identifiable for a coupled rotation, degenerate for a hinge | F16 |
| — | A smoothing filter would reduce the visible fingers (user) | **VERIFIED, with the distinction the user drew.** It does. On the Polhemus bench recordings what it removes to do so is the movement (0.24° at 0.5 Hz against a jitter of 0.003°); on recordings of people a cut-off of 4–10 Hz removes jitter and vibration and leaves the spread of the axes alone | F17 |

---

## 5. What was changed in the code

All in `FHA/FHA_plotly_fast/`. The other two variants (`FHA_plt`, `FHA_plotly_traditional`)
were not touched and still behave as the original.

| File | Change |
|---|---|
| `fha_core.py` (new) | All numerics, no plotting. Screw extraction through the rotation vector (exact 0–180°, NaN instead of a near-zero vector when there is no rotation). Windows that close on **net rotation**, built **symmetrically around the frame they are assigned to** and widened until their two ends are the full requested rotation apart (F14); dropped when the rotation reverses inside them or the pose jumps, but not merely because the frames are coarse (F19). Sign-safe average axis with a least-squares pivot. Averaging weights that make every degree of motion count equally, so slow or paused stretches do not dominate. New `reference_pose` method. Optional `estimator="fit"` and `criterion="path"` (the original window rule) for comparison. `swing_twist`: rotation accumulated about an axis and away from it, optionally as seen from a frame turning with the primary rotation (F15). Third round: `smooth`, `smooth_poses`, `smooth_motion` (low-pass in screw coordinates of the relative pose), `removed_by_smoothing`, `jitter` (F17, F19); `even_time_base` (F18). |
| `synthetic.py` (new) | Simulated two-axis rig with the exact instantaneous axis, four ways of delivering the second rotation, and measurement-error presets calibrated on the test recordings. |
| `recordings.py` (new) | One loader for Polhemus, Vicon and the simulator. Polhemus timestamps are replaced by an even time base, also when samples arrived in bursts, the recording was paused or the host clock was set back (F18); the notes under the controls say what was done. ezc3d is only imported when a C3D file is opened. A Vicon recording also carries where its markers sit on each segment (`points`), for drawing the bodies (F20). |
| `shapes.py` (new) | Basic triangle meshes for the bodies: box, bar, plate, merging, cutting to a box. A mesh is vertices and faces in the coordinates of its body, which is also the form a bone surface would take (F20). |
| `vicon_import.py` | Added `build_reference_frames_lsq`: least-squares rigid-body fit of each marker cluster to its median shape, same frame definition, returns the fit residual. The original `build_reference_frames` is unchanged. |
| `FHA_visualizer_fast.py` | Rewritten: `analyse()` plus the figures. Position map, direction map, 3D scene, build-up curve and tilt map, per-axis series, recording overview. `primary_axis_of()` takes the primary axis from a reference recording. `calculate_FHA_vectorized` kept under its name, now exact. `generate_FHA` and `plotly_visualize_FHA` are gone (their only caller was `main.py`). Third round: `analyse(..., smoothing_hz=)` filters the whole recording before the cut, reports what was removed (`Analysis.smoothing`) and keeps the unsmoothed axes (`Analysis.raw`), which the maps and the series draw underneath in grey; `smoothing_sweep`, `figure_smoothing_sweep`, `figure_smoothing_removed` (F17). Fourth round: `Analysis.play` (frames played, poses and shapes of the two bodies, the axis and the map points of each instant, the fixed box of the 3D view) and `playback_store` (the same as JSON for the browser); every figure takes a `step` and draws the playhead there (F20). |
| `main.py`, `assets/style.css`, `assets/playback.js` | Dashboard with controls for the recording, frame range, smoothing, method, window, window rule, estimator, display frame, plane offset, colour and theme, and for what the build-up is measured against (a reference recording, or the recording's own average axis with a warning); headline numbers; table; CSV export. The smoothing slider has a sentence beside it saying what the chosen cut-off removes, and a card "What the smoothing takes out" (F17). Screenshots: `figures/09_dashboard_polhemus3.png` (recording 3), `figures/10_dashboard_simulated_steady.png` and `figures/11_dashboard_simulated_pulsed.png` (simulator, with the true axis drawn in), `figures/16_dashboard_smoothing_knee.png` (knee flexion at 4 Hz) and `figures/17_dashboard_smoothing_simulated.png` (simulator with Vicon-like error at 1 Hz: smoothed axes against the true axis). Under Dash 4 the sliders and dropdowns now take the dashboard's colours in both themes. Fourth round: a play bar under the 3D view (play, playhead slider, speed, loop) and a choice of how the bodies are drawn; `assets/playback.js` holds the whole of playback and runs in the browser (F20); `figures/18_dashboard_playback_vicon.png`. |
| `config.py` | Same variables as before plus optional ones, among them `smoothing_hz` (0 = off) and `bodies`; it now only sets what the dashboard opens with. |
| `tests/` (new) | 101 tests: numerics against analytic answers (49), marker-frame fit and loaders (8), every dashboard callback driven through its HTTP endpoint (28), playback data and shapes (15), and playback worked in headless Chrome through the real controls (1, skipped without Chrome). Pass on Python 3.13 / Dash 2.14 / Plotly 6.5, on 3.11 / Dash 2.9 / Plotly 5.13 / SciPy 1.12, and on 3.11 / Dash 4.1 / Plotly 6.6. (The Vicon tests are skipped where ezc3d is not installed, as in the second of those.) The page was looked at in a headless browser under all three, in the light and the dark theme. |

Behaviour that differs from the original and should be kept in mind when comparing
numbers with earlier results:

- miFHA axes are assigned to the middle of their window, and there are fewer of them:
  frames at rest and frames within half a window of the start, end or a turn-around get
  no axis (560 instead of 965 on Polhemus file 1, one way).
- "Mean angle to average axis" is weighted so every degree of motion counts equally, and
  the average axis is the optimal direction rather than a mean of vectors. On Polhemus
  file 1 it still reads 0.24°.
- The cross-section is taken through the pivot by default. Set the plane offset to
  reproduce a picture like the original's.
- `window_rule = 'path'` reproduces the original choice of windows (not its tagging or
  its average axis).
- Every moving window spans at least the requested net rotation between its two poses
  (F14). Where the axis changes direction inside a window, the window is a little
  longer in time than before for that reason.
- The build-up panel is measured against a reference recording by default
  (`build_up_axis = 'reference'`, `reference_path` in `config.py`, set to recording 1 of
  the Polhemus bench session). Against a recording's own average axis it understates
  the second rotation (F15), and the panel says so.
- Smoothing is off unless asked for (`smoothing_hz = 0`). When on, it acts on the whole
  recording before the frame range is applied, and the CSV header records the cut-off
  and what it removed.
- With windows of 1–2° more axes are returned than after the second round: the jump
  guard no longer takes coarse frames for jumps (F19). Windows of 5° and more are
  unchanged on every test recording.
- Fifteen of the Polhemus recordings of people get a different time base than before
  (F18). Their axes are the same; the times and window durations shown for them change.

## 6. Recommendations

**For the analysis**
- Use the moving window with net-rotation windows (the default). Report the window size
  with every result: it is the resolution of the method, and results at different window
  sizes are not comparable.
- Read the direction map and the build-up together. The direction map shows what the
  joint does within each window; the build-up shows what has added up. A burst of
  second rotation is a finger in the first and a lasting step in the second.
- Record the primary rotation on its own in every session — a pure flexion trial, the
  bench hinge without the second rotation — with the reference sensor or marker
  cluster left exactly where it is. Its average axis is what the build-up of every
  other recording of that session is measured against (F15). Without it, "how much
  second rotation" has no answer the motion itself can give (F16).
- For a pattern that should not depend on how evenly a movement was performed, use a
  larger window or the reference-pose axes, and say which.
- Do not use the consecutive-frame FHA for anything but illustration; it is the
  ill-conditioned case the ISB 2025 work set out to avoid.

**For smoothing** (F17–F19)
- Bench recordings on the Polhemus rig: leave it off. The instrument's jitter is
  0.003° and its real error is slow; there is nothing for a filter to remove but the
  movement. If the fingers are in the way, widen the window and say so.
- Recordings of people: look at the curve in "What the smoothing takes out" first. If
  it has a flat stretch, set the cut-off inside it; the result then hardly depends on
  the exact value. On the head recordings that is 4–10 Hz, on the knee recordings about
  3–5 Hz. Without a flat stretch, stay with the customary 6 Hz or leave it off.
- Do not move the slider until the picture looks tidy. From about 2 Hz down the spread
  of the axes itself shrinks; that is a statement about the joint that the data did not
  make. Watch the grey line of unsmoothed axes and the mean angle to the average axis:
  when the latter starts to fall, movement is going.
- Report the cut-off with the window size. Both set the resolution of the result.
- Filter the relative pose, as the dashboard does, not marker or sensor signals, once
  the cut-off is below about 3 Hz (F17a).
- Marker data filtered upstream (Nexus, Visual3D) have already had this done to them,
  on the markers. A second filter here adds to it; note both.
- Dynamic CT, a couple of dozen frames: expect little from a filter (F19). For a
  movement of 3–4 s only cut-offs of about 1–2 Hz helped, at either frame rate; a slower
  movement tolerates a lower one. A wider window is the safer way to steady the axes.

**For the bench rigs**
- A two-axis validation needs the two rotations **mechanically coupled** (a linkage, a
  cam, an inclined second hinge, or one oblique hinge), or driven by a robot or by servos
  commanded on a common fine-grained schedule. Two hands, or two hobby servos stepping
  independently, give a rate ratio that depends on timing.
- The simulator (`synthetic.py`, or "Simulator" in the dashboard) shows what a steady,
  a phased-in and a pulsed second rotation should each look like, with the true axis
  drawn in.
- Vicon: rebuild the moving cluster as a rigid plate with at least four non-coplanar
  markers and both baselines of 100 mm or more. With the present 35 mm baseline, marker
  error of 0.1 mm is already 0.15° of orientation, which is the size of the error the
  calibration found (F6).
- Polhemus: the spread on a pure hinge (0.24°) is slow-varying, not jitter, so it will
  not average out. Field distortion and play in the hinge are the candidates; a
  calibration sweep of the empty rig would separate them.

## 7. Open questions and things not done

- **Vicon, servo jerks:** not established (F3, F9). The servo command log, or a repeat
  with a rigid four-marker cluster, would settle it.
- **Source of the Polhemus wander:** not investigated beyond measuring it (F1, F6).
- **Sensor 1 moving in the lab:** handled by showing axes in the reference sensor's
  frame (Polhemus file 5 gives 0.25°), but only that one recording exercises it.
- **In-vivo recordings** (knee and head files 6–12 and the sub-folders of
  `test_data/Polhemus_test_data/`): used in F17 and F18 to see what a filter removes from
  recordings of people. Their helical axes as such — what they say about the joints —
  were not analysed.
- **What the 6–20 Hz content of the knee recordings is** (F17e): shaking of the sensor on
  soft tissue, tremor, or the joint. A recording with the sensor on a rigid mount, or a
  second sensor on the same segment, would tell.
- **Soft-tissue artefact** is slow and tied to the movement. No filter removes it and
  nothing here addresses it.
- **Dynamic CT**: no data and no loader. F19 is simulation with an assumed error of
  0.3° and 0.3 mm per frame; the real size and character of registration error (is it
  independent from frame to frame?) decide whether its conclusions carry over. The
  numerics take any `(t, T1, T2)`; what is missing is a reader for the bone poses.
- **Choosing the cut-off automatically** was left out on purpose. The flat stretch can
  be found by rule, but whether to trust it is a judgement about the movement.
- **Window and filter together.** Both average, one in degrees and one in Hz. Whether a
  filter lets a smaller window be used at the same error was not examined.
- **IMU recordings** (`test_data/IMU_test_data/`): orientation only, no positions. The
  direction of the axis could be computed from them, its position cannot. No loader was
  written.
- **Build-up for a joint without a hinge to refer to.** For the bench rig the hinge
  recording defines the primary axis. For a knee or a neck there is no such recording
  by construction; which movement or anatomical axis should play that role is a
  choice that has to be made and reported, and was not examined here.
- **How well a reference axis carries over between recordings** was checked only through
  the pose of the reference sensor in the lab (0.1° for recording 3, 1.4° for recording
  4). If the whole rig is moved between recordings that check raises a false alarm;
  if only the sensor is re-seated on its segment by a small amount it is the right
  alarm. A repeat of the hinge trial at the end of a session would tell the two apart.
- **Bone surfaces in the playback** (F20): the place for them is there (a mesh in the
  coordinates of its body), reading a mesh file and registering it to the sensor or
  segment frame are not. For skin-mounted sensors that registration is the hard part.
- **Playback leans on Plotly internals for speed** (F20). If a later Plotly changes
  them, playback falls back to Plotly's own calls and becomes slower, not wrong; the
  browser test then fails on "moved directly" and says so.
- **`FHA_plt` and `FHA_plotly_traditional`** still contain every defect listed in
  section 4.
- **`FHA/main_CLI.py`** imports `FHA_visualizer`, which does not exist at that path; it
  did not run before this work and was not touched.
- **The noise presets** reproduce the scatter and smoothness of the axes on one
  single-axis recording per instrument. They are a working model, not a characterisation
  of the trackers.
