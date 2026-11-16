"""Step 15 — what a smoothing filter does to the helical axes (F17-F19).

A low-pass filter removes whatever changes faster than its cut-off. It cannot tell
measurement noise from movement. This script puts numbers on both sides:

  A  the filter itself: does it move the axis of a noise-free hinge?
  B  ground truth (simulated rig): error against the TRUE axis as the cut-off is lowered
  C  real single-axis bench recordings: spread of the axes about one fixed axis
  D  real two-axis bench recordings: the fingers, the build-up, and how much the filter
     removes compared with the frame-to-frame jitter of the recording
  E  recordings of people (knee flexion, head movement): where the curve of "removed
     against cut-off" runs flat, and what a cut-off there does to the axes
  F  the time base: what happens when arrival timestamps are taken at face value
  G  few frames at a low rate, as in dynamic CT

Run:  python 15_smoothing.py          (a few minutes; part E reads about 80 recordings)
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from common import FIG, POLHEMUS, load_polhemus, load_vicon

import fha_core as fc
import polhemus_import as pi
import recordings as rc
import synthetic as syn

CUTOFFS = [None, 20.0, 10.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3]
LABEL = lambda c: "off" if c is None else f"{c:g}"
HEAD = "   " + " " * 35 + " ".join(f"{LABEL(c):>7s}" for c in CUTOFFS)


def angle(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.abs(np.einsum("...i,...i->...", a, b))))


def rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


def removed(t, T1, T2, cutoff, frames=slice(None)):
    """RMS of what the filter takes out of the relative pose: (deg, length)."""
    if cutoff is None:
        return 0.0, 0.0
    raw = fc.relative_poses(T1, T2)
    a, d = fc.removed_by_smoothing(raw[frames], fc.smooth_poses(t, raw, cutoff)[frames])
    return rms(a), rms(d)


def row(label, values, fmt="7.3f"):
    print(f"   {label:>34s} " + " ".join(format(v, fmt) for v in values))


def cutoff_axis(ax, title, ylabel, first=0, log=True):
    ticks = list(range(first, len(CUTOFFS)))
    ax.set_xticks(ticks, [LABEL(CUTOFFS[i]) for i in ticks], fontsize=7)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("cut-off (Hz)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)


# ------------------------------------------------------------------------------ A

def uneven_hinge():
    """A noise-free hinge turned stop-and-go; sensor 20 cm from the axis."""
    fs = 120.0
    t = np.arange(int(8 * fs)) / fs
    speed = np.radians(12.0) * (1 + np.sin(2 * np.pi * t))
    speed[(t < 0.5) | (t > 7.5)] = 0
    T2_0 = np.eye(4); T2_0[:3, 3] = [20.0, 0.0, 3.0]
    T2 = syn._rotation_about_line([0, 1, 0], [0, 0, 0], np.cumsum(speed) / fs) @ T2_0
    T1 = np.tile(np.eye(4), (len(t), 1, 1)); T1[:, :3, 3] = [-6.0, 2.0, 3.0]
    return t, T1, T2


def part_a():
    print("A  NOISE-FREE HINGE, TURNED STOP-AND-GO: how far does the filter itself move the axes? (cm)")
    t, T1, T2 = uneven_hinge()
    print(HEAD)
    for name, screw in (("positions + orientation", False), ("screw coordinates", True)):
        worst = []
        for c in CUTOFFS:
            f = fc.compute_fha(t, *fc.smooth_motion(t, T1, T2, c, screw=screw))
            rel = -f.s + np.linalg.inv(T1[0])[:3, 3]            # hinge point (lab origin) in frame 1, minus axis point
            worst.append(np.linalg.norm(rel - np.einsum("ki,ki->k", rel, f.n)[:, None] * f.n, axis=1).max())
        row(name, worst, "7.4f")


# ------------------------------------------------------------------------------ B

def part_b(axes):
    print("\nB  SIMULATED RIG: mean angle between each axis and the TRUE axis at its frame (deg), 10 deg windows, 6 trials")
    rigs = {"Polhemus-like": dict(noise="polhemus", fs=120.0, move_s=5.5, still_s=1.0),
            "Vicon-like": dict(noise="vicon", fs=100.0, move_s=5.0, still_s=0.5)}
    profiles = (("none", "pure hinge"), ("proportional", "steady second rotation"), ("late", "phased-in"), ("pulsed", "bursts"))
    for ax, (rig_name, rig) in zip(axes, rigs.items()):
        print(f"   {rig_name}")
        print(HEAD + "   best")
        for profile, label in profiles:
            err = np.zeros(len(CUTOFFS)); rem = np.zeros(len(CUTOFFS))
            for seed in range(6):
                d = syn.two_axis_motion(profile=profile, seed=seed, **rig)
                for k, c in enumerate(CUTOFFS):
                    T1, T2 = fc.smooth_motion(d["t"], d["T1"], d["T2"], c)
                    f = fc.compute_fha(d["t"], T1, T2)
                    n, _ = fc.axes_to_lab(f, T1)
                    ok = d["moving"][f.ic]
                    err[k] += angle(n[ok], d["truth_n"][f.ic][ok]).mean() / 6
                    rem[k] += removed(d["t"], d["T1"], d["T2"], c, d["moving"])[0] / 6
            print(f"   {label:>34s} " + " ".join(f"{v:7.3f}" for v in err) + f"   {LABEL(CUTOFFS[int(np.argmin(err))])} Hz")
            if profile == "none":
                row("removed by the filter (deg)", rem)
            if profile != "pulsed":
                ax.plot(range(len(CUTOFFS)), err, marker="o", ms=3, label=label)
        cutoff_axis(ax, f"simulated, {rig_name}: error against the true axis", "error (deg)")


# ------------------------------------------------------------------------------ C

def part_c(ax):
    print("\nC  REAL SINGLE-AXIS BENCH RECORDINGS (out and back): spread of the axes about their average axis")
    for name, loader, key, unit in (("Polhemus 1, pure hinge", load_polhemus, "1_full", "cm"), ("Vicon servo X", load_vicon, "servo_X", "mm")):
        t, T1, T2, _, _ = loader(key)
        t = fc.uniform_time(t)
        ja, jd = fc.jitter(fc.relative_poses(T1, T2))
        print(f"   {name}: frame-to-frame jitter of the relative pose {ja:.4f} deg, {jd:.4f} {unit}")
        print(HEAD)
        ang, dist, rem = [], [], []
        for c in CUTOFFS:
            f = fc.compute_fha(t, *fc.smooth_motion(t, T1, T2, c))
            nb, pv = fc.average_axis(f.n, f.s, f.weight)
            disp = fc.dispersion(f.n, f.s, nb, pv, f.weight)
            ang.append(disp["angle_mean"]); dist.append(disp["distance_rms"]); rem.append(removed(t, T1, T2, c)[0])
        row("mean angle (deg)", ang)
        row(f"RMS distance ({unit})", dist)
        row("removed by the filter (deg)", rem)
        ax.plot(range(len(CUTOFFS)), ang, marker="o", ms=3, label=name)
    cutoff_axis(ax, "real single-axis recordings: spread of the axes", "mean angle to average axis (deg)")


# ------------------------------------------------------------------------------ D

def doubling_back(xy):
    path = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    return path / max(np.linalg.norm(xy[-1] - xy[0]), 1e-9)


def part_d(ax_removed, ax_map, ax_fingers):
    print("\nD  REAL TWO-AXIS BENCH RECORDINGS (outbound)")
    t1, T11, T21, _, _ = load_polhemus("1_full")
    f = fc.compute_fha(fc.uniform_time(t1), T11, T21)
    hinge = fc.average_axis(f.n, f.s, f.weight)[0]
    cases = (("Polhemus 3, hinge + x by hand", load_polhemus, "3_y_plus_x", 100, 1683, "C0"),
             ("Polhemus 4, hinge + z by hand", load_polhemus, "4_y_plus_z", 600, 1901, "C1"),
             ("Vicon servo X + 20 Y", load_vicon, "servo_X_20Y", 200, 760, "C2"))
    for name, loader, key, c1, c2, colour in cases:
        t, T1, T2, _, _ = loader(key)
        t = fc.uniform_time(t)
        raw = fc.relative_poses(T1, T2)
        ja, _ = fc.jitter(raw[c1:c2])
        print(f"   {name}: frame-to-frame jitter {ja:.4f} deg")
        if key == "3_y_plus_x":
            still = Rotation.from_matrix(raw[:90, :3, :3] @ raw[0, :3, :3].T).as_rotvec()
            still = np.degrees(still - still.mean(axis=0))
            print(f"      before the movement starts (first 0.75 s) the pose scatters by {rms(np.linalg.norm(still, axis=1)):.3f} deg "
                  f"while its frame-to-frame jitter is {fc.jitter(raw[:90])[0]:.4f} deg: nearly all of the scatter is slow")
        print(HEAD)
        rows = {k: [] for k in ("removed", "fingers", "tilt", "build")}
        maps = {}
        for c in CUTOFFS:
            T1s, T2s = fc.smooth_motion(t, T1, T2, c)                 # smooth the whole recording, then cut
            fha = fc.compute_fha(t[c1:c2], T1s[c1:c2], T2s[c1:c2])
            nb, pv = fc.average_axis(fha.n, fha.s, fha.weight)
            Trel = fc.relative_poses(T1s[c1:c2], T2s[c1:c2])
            e1, e2 = fc.plane_basis(nb, Trel[:, :3, 3].mean(axis=0) - pv)
            tilt = fc.tilt_map(fha.n, nb, e1, e2)
            rows["removed"].append(removed(t, T1, T2, c, slice(c1, c2))[0])
            rows["fingers"].append(doubling_back(tilt))
            rows["tilt"].append(float(fha.weight @ np.linalg.norm(tilt, axis=1)))
            rows["build"].append(np.linalg.norm(fc.swing_twist(Trel, hinge, turning=True)[1], axis=1).max() if "Polhemus" in name else np.nan)
            maps[c] = tilt
        row("removed by the filter (deg)", rows["removed"])
        row("fingers: doubling back", rows["fingers"], "7.2f")
        row("mean tilt from average axis (deg)", rows["tilt"], "7.2f")
        if "Polhemus" in name:
            row("second rotation built up (deg)", rows["build"], "7.2f")
        ax_removed.plot(range(1, len(CUTOFFS)), rows["removed"][1:], marker="o", ms=3, color=colour, label=name)
        ax_removed.axhline(ja, color=colour, lw=0.8, alpha=0.6)
        ax_fingers.plot(range(len(CUTOFFS)), rows["fingers"], marker="o", ms=3, color=colour, label=name)
        if key == "3_y_plus_x":
            for c, col in ((None, "0.6"), (2.0, "C0"), (0.5, "C3")):
                ax_map.plot(maps[c][:, 0], maps[c][:, 1], color=col, lw=1.2, label=f"cut-off {LABEL(c)}" + ("" if c is None else " Hz"))
            print("   same recording, no filter, wider window instead:")
            for step in (10, 20, 30):
                fha = fc.compute_fha(t[c1:c2], T1[c1:c2], T2[c1:c2], "incremental_angle", step)
                nb, pv = fc.average_axis(fha.n, fha.s, fha.weight)
                Trel = fc.relative_poses(T1[c1:c2], T2[c1:c2])
                e1, e2 = fc.plane_basis(nb, Trel[:, :3, 3].mean(axis=0) - pv)
                print(f"      window {step:2d} deg: doubling back {doubling_back(fc.tilt_map(fha.n, nb, e1, e2)):.2f}")
    cutoff_axis(ax_removed, "bench: what the filter removes (horizontal lines: frame-to-frame jitter)", "removed, RMS (deg)", first=1)
    cutoff_axis(ax_fingers, "bench: fingers in the direction map", "doubling back (1 = no fingers)")
    ax_map.set_aspect("equal", "datalim"); ax_map.grid(alpha=0.3); ax_map.legend(fontsize=7)
    ax_map.set_title("Polhemus 3: direction map at three cut-offs", fontsize=9)
    ax_map.set_xlabel("tilt toward the moving sensor (deg)"); ax_map.set_ylabel("tilt across (deg)")


# ------------------------------------------------------------------------------ E

HUMAN = (("knee flexion, open chain (6)", "6_90deg_knee_flexion_OKC(z-axis)data.csv"),
         ("knee flexion, closed chain (7)", "7_90deg_knee_flexion_CKC_2(z-axis)data.csv"),
         ("knee flexion, closed chain (8)", "8_90deg_knee_flexion_CKC(z-axis)data.csv"),
         ("knee flexion, open chain (9)", "9_90deg_knee_flexion_OKC_2(z-axis)data.csv"),
         ("head rotation (10)", "10_Head_rotationdata.csv"),
         ("head rotation (11)", "11_Head_rotation_2data.csv"),
         ("head lateral flexion (12)", "12_Head_lateral_flexiondata.csv"))


def axes_of(t, T1, T2):
    """Spread of the axes about their average axis (deg) and the length of the path their
    direction travels in the direction map (deg): wiggle makes the path long."""
    f = fc.compute_fha(t, T1, T2)
    nb, pv = fc.average_axis(f.n, f.s, f.weight)
    e1, e2 = fc.plane_basis(nb, fc.relative_poses(T1, T2)[:, :3, 3].mean(axis=0) - pv)
    tilt = fc.tilt_map(f.n, nb, e1, e2)
    order = np.argsort(f.ic)
    hop = np.linalg.norm(np.diff(tilt[order], axis=0), axis=1)
    return fc.dispersion(f.n, f.s, nb, pv, f.weight)["angle_mean"], float(hop[np.diff(f.ic[order]) <= 3].sum())


def study(rec, cutoffs):
    raw = fc.relative_poses(rec.T1, rec.T2)
    out = dict(jitter=fc.jitter(raw)[0], removed=[], spread=[], path=[])
    for c in cutoffs:
        T1s, T2s = fc.smooth_motion(rec.t, rec.T1, rec.T2, c)
        out["removed"].append(0.0 if c is None else rms(fc.removed_by_smoothing(raw, fc.relative_poses(T1s, T2s))[0]))
        spread, path = axes_of(rec.t, T1s, T2s)
        out["spread"].append(spread); out["path"].append(path)
    return out


def part_e(ax_curves, ax_path):
    print("\nE  RECORDINGS OF PEOPLE (Polhemus, whole recording, 10 deg windows)")
    print("   'slope' is the steepness of the removed-against-cut-off curve between neighbouring cut-offs on")
    print("   log-log axes: below about 1 the curve runs flat, i.e. little in the recording changes at that rate")
    freq = np.array(CUTOFFS[1:], float)
    for k, (label, name) in enumerate(HUMAN):
        r = study(rc.load_polhemus(os.path.join(POLHEMUS, name)), CUTOFFS)
        print(f"   {label}: frame-to-frame jitter {r['jitter']:.4f} deg")
        print(HEAD)
        row("removed by the filter (deg)", r["removed"])
        slope = np.diff(np.log(r["removed"][1:])) / np.diff(np.log(1 / freq))
        print(f"   {'slope':>34s} " + " " * 12 + " ".join(f"{v:7.2f}" for v in slope))
        row("mean angle to average axis (deg)", r["spread"], "7.2f")
        row("path of the axis direction (deg)", r["path"], "7.0f")
        group = "knee flexion" if "knee" in label else "head movement"
        style = dict(color="C0" if group == "knee flexion" else "C1", lw=1.2, alpha=0.8, label=group if k in (0, 4) else None)
        ax_curves.plot(range(1, len(CUTOFFS)), r["removed"][1:], **style)
        ax_curves.plot([1, len(CUTOFFS) - 1], [r["jitter"]] * 2, ls=":", color=style["color"], lw=0.8)
        ax_path.plot(range(len(CUTOFFS)), np.array(r["path"]) / r["path"][0], **style)
    t, T1, T2, _, _ = load_polhemus("3_y_plus_x")
    bench = study(rc.Recording("bench 3", "polhemus", fc.uniform_time(t), T1, T2, "cm"), CUTOFFS)
    ax_curves.plot(range(1, len(CUTOFFS)), bench["removed"][1:], color="0.3", lw=1.2, label="bench: hinge + second rotation by hand (3)")
    ax_path.plot(range(len(CUTOFFS)), np.array(bench["path"]) / bench["path"][0], color="0.3", lw=1.2, label="bench (3)")
    cutoff_axis(ax_curves, "people: what the filter removes (dotted: frame-to-frame jitter)", "removed, RMS (deg)", first=1)
    cutoff_axis(ax_path, "people: wiggle of the axis direction", "path length relative to no filter", log=False)

    # The larger set of recordings of people kept in sub-folders of the Polhemus test data.
    # They are read as a group and reported in aggregate only.
    few = [None, 20.0, 6.0, 3.0, 2.0, 1.0]
    paths = [path for _, path in rc.discover()["polhemus"] if os.path.dirname(os.path.relpath(path, POLHEMUS))]
    if not paths:
        return
    table = []
    for path in paths:
        r = study(rc.load_polhemus(path), few)
        table.append([r["jitter"]] + r["removed"][1:] + list(np.array(r["path"][1:]) / r["path"][0])
                     + list(np.array(r["spread"][1:]) / r["spread"][0]))
    table = np.array(table)
    med = lambda col: (f"{np.nanmedian(table[:, col]):.3f} ({np.nanpercentile(table[:, col], 10):.3f}-"
                       f"{np.nanpercentile(table[:, col], 90):.3f})")
    n = len(few) - 1
    print(f"\n   {len(paths)} further recordings of people (sub-folders of the Polhemus test data, 50 s each): median (10th-90th percentile)")
    print(f"   {'frame-to-frame jitter (deg)':>32s} {med(0)}   (could be measured in {int(np.isfinite(table[:, 0]).sum())} of them)")
    for k, c in enumerate(few[1:]):
        print(f"   {'at ' + LABEL(c) + ' Hz: removed (deg)':>32s} {med(1 + k)}   path of the axis direction x {med(1 + n + k)}   "
              f"mean angle to average axis x {med(1 + 2 * n + k)}")
    ratio = table[:, 2] / table[:, 0]
    print(f"   {'removed at 6 Hz / jitter':>32s} {np.nanmedian(ratio):.1f} ({np.nanpercentile(ratio, 10):.1f}-{np.nanpercentile(ratio, 90):.1f})")


# ------------------------------------------------------------------------------ F

def burst_stamps(n, fs=120.0, seed=3):
    """Arrival stamps of an evenly sampling device whose samples reach the host in bursts."""
    rng = np.random.default_rng(seed)
    true = np.arange(n) / fs
    stamp = true + rng.uniform(0, 2e-4, n)
    i = 0
    while i < n:
        if rng.random() < 0.15:
            j = min(i + int(rng.integers(3, 12)), n) - 1
            stamp[i:j + 1] = true[j] + np.arange(j + 1 - i) * 1e-5
            i = j + 1
        else:
            i += 1
    return true, stamp


def part_f(ax):
    print("\nF  TIME BASE: arrival timestamps taken at face value")
    true, stamp = burst_stamps(1200)
    sweep = np.radians(90.0) * 0.5 * (1 - np.cos(np.pi * true / true[-1]))               # one smooth sweep, no noise
    T = syn._rotation_about_line([0, 1, 0], [0, 0, 0], sweep)
    print("   a noise-free hinge sweep of 90 deg in 10 s, smoothed at 6 Hz: how far the filter moves the pose (deg)")
    for label, times, colour in (("stamps as recorded (bursts)", stamp, "C3"), ("time base repaired", fc.uniform_time(stamp), "C0")):
        off = np.degrees(fc.net_angle(fc.quaternions(T), fc.quaternions(fc.smooth_poses(times, T, 6.0))))
        print(f"      {label:30s} worst frame {off.max():.4f}   RMS {rms(off):.4f}")
        ax.plot(true, np.maximum(off, 1e-7), color=colour, lw=1.0, label=label)
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    ax.set_title("time base: a noise-free sweep after a 6 Hz filter", fontsize=9)
    ax.set_xlabel("time (s)"); ax.set_ylabel("moved by the filter (deg)")

    listed = rc.discover()["polhemus"]
    uneven, backwards, pauses, worse = 0, 0, 0, []
    for _, path in listed:
        raw_t = np.asarray(pi.read_data(path)[4], float)
        k = np.arange(len(raw_t))
        slope, intercept = np.polyfit(k, raw_t, 1)
        if np.max(np.abs(raw_t - (slope * k + intercept))) <= 4.0 * slope:
            continue                                         # a straight line fits: ordinary arrival jitter
        uneven += 1
        rec = rc.load_polhemus(path)
        pauses += len(fc.even_time_base(raw_t)[1])
        if np.any(np.diff(raw_t) <= 0):
            backwards += 1
            continue
        rel = fc.relative_poses(rec.T1, rec.T2)
        good = rms(fc.removed_by_smoothing(rel, fc.smooth_poses(rec.t, rel, 6.0))[0])
        bad = rms(fc.removed_by_smoothing(rel, fc.smooth_poses(raw_t, rel, 6.0))[0])
        worse.append(bad / good)
    print(f"   of the {len(listed)} Polhemus recordings in the test data, {uneven} have stamps a straight line does not fit:")
    print(f"      bursts of buffered samples in all of them; {pauses} real pauses in total; the clock runs backwards once in {backwards}")
    print(f"      (those {backwards} cannot be filtered at all on their recorded stamps)")
    if worse:
        print(f"      on the other {len(worse)} a 6 Hz filter 'removes' {np.median(worse):.1f} times as much (median; {np.min(worse):.1f} to "
              f"{np.max(worse):.1f} times) on the recorded stamps")
        print("      as on the repaired time base — the excess is distortion the filter itself puts in")


# ------------------------------------------------------------------------------ G

def ct_series(n_frames, rot_deg=0.3, pos=0.03, seed=0):
    """A smooth two-axis movement in ``n_frames`` frames over 4 s, with an independent error
    on every frame (0.3 deg and 0.3 mm per axis), as from registering each volume of a
    dynamic CT series on its own."""
    d = syn.two_axis_motion(profile="late", noise="none", fs=n_frames / 4.0, move_s=3.6, still_s=0.2)
    rng = np.random.default_rng(seed)
    clean = fc.relative_poses(d["T1"], d["T2"])
    noisy = clean.copy()
    noisy[:, :3, :3] = clean[:, :3, :3] @ Rotation.from_rotvec(np.radians(rot_deg) * rng.standard_normal((len(clean), 3))).as_matrix()
    noisy[:, :3, 3] += pos * rng.standard_normal((len(clean), 3))
    return d, clean, noisy


def part_g(ax):
    print("\nG  FEW FRAMES AT A LOW RATE (dynamic CT): 90 deg + 20 deg phased in over 3.6 s; error 0.3 deg and 0.3 mm per axis on every frame")
    cut = [None, 2.5, 2.0, 1.5, 1.0, 0.7, 0.5]
    for n_frames, colour in ((24, "C0"), (48, "C1")):
        d0, clean, _ = ct_series(n_frames)
        t, T1 = d0["t"], d0["T1"]
        off = lambda T: rms(np.degrees(fc.net_angle(fc.quaternions(T), fc.quaternions(clean))))
        pose, moved, axis, jit = [], [], [], []
        for c in cut:
            p, a = [], []
            for seed in range(30):
                d, _, noisy = ct_series(n_frames, seed=seed)
                smooth = fc.smooth_poses(t, noisy, c)
                p.append(off(smooth))
                f = fc.compute_fha(t, T1, T1 @ smooth, "incremental_angle", 20)
                ok = d["moving"][f.ic]
                a.append(angle(fc.axes_to_lab(f, T1)[0][ok], d["truth_n"][f.ic][ok]).mean())
                if c is None:
                    jit.append(fc.jitter(noisy)[0])
            pose.append(np.mean(p)); axis.append(np.mean(a))
            moved.append(off(fc.smooth_poses(t, clean, c)))
        f0 = fc.compute_fha(t, T1, d0["T2"], "incremental_angle", 20)
        bias = angle(fc.axes_to_lab(f0, T1)[0][d0["moving"][f0.ic]], d0["truth_n"][f0.ic][d0["moving"][f0.ic]]).mean()
        rate = 1.0 / np.median(np.diff(t))
        print(f"   {n_frames} frames at {rate:.0f} per second: true size of the per-frame error {np.sqrt(3) * 0.3:.2f} deg; jitter estimate "
              f"{np.nanmedian(jit):.2f} deg (reported in {100 * np.mean(np.isfinite(jit)):.0f}% of trials); "
              f"windows of 20 deg asked span {np.degrees(np.median(f0.phi)):.0f} deg")
        print("   " + " " * 49 + " ".join(f"{LABEL(c):>7s}" for c in cut))
        for label, values in (("pose error left (deg)", pose), ("movement removed from the clean series (deg)", moved),
                              ("axis against the true axis, 20 deg windows (deg)", axis)):
            print(f"   {label:>48s} " + " ".join(f"{v:7.3f}" for v in values))
        print(f"   {'(the same axes without any error on the frames:':>48s} {bias:7.3f})")
        ax.plot(range(len(cut)), np.array(axis) / axis[0], marker="o", ms=3, color=colour, label=f"{n_frames} frames: axis error")
        ax.plot(range(len(cut)), np.array(pose) / pose[0], marker="s", ms=3, ls="--", color=colour, label=f"{n_frames} frames: pose error")
    ax.set_xticks(range(len(cut)), [LABEL(c) for c in cut], fontsize=7)
    ax.axhline(1.0, color="0.5", lw=0.8); ax.grid(alpha=0.3); ax.legend(fontsize=7); ax.set_ylim(0, 2.5)
    ax.set_title("few frames (CT-like): error relative to no filter", fontsize=9)
    ax.set_xlabel("cut-off (Hz)"); ax.set_ylabel("error / error without filter")


def main():
    fig, axes = plt.subplots(4, 3, figsize=(19, 20))
    part_a()
    part_b(axes[0, :2])
    part_c(axes[0, 2])
    part_d(axes[1, 0], axes[1, 1], axes[1, 2])
    part_e(axes[2, 0], axes[2, 1])
    part_f(axes[2, 2])
    part_g(axes[3, 0])
    axes[3, 1].set_visible(False)
    axes[3, 2].set_visible(False)
    fig.tight_layout()
    out = f"{FIG}/15_smoothing.png"
    fig.savefig(out, dpi=62)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
