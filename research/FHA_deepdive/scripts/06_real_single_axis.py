"""Step 6 — before/after on the REAL single-axis recordings.

On a pure hinge every helical axis should be the same line, so the spread of the axes
is the error. Spread is measured the same way for every method:

  * axes are expressed in the lab frame;
  * every degree of joint rotation counts equally (weights by 1-degree bin of the
    joint angle, outbound and return counted separately), so a method is not rewarded
    or punished for producing many axes during a slow or still stretch;
  * the average axis is the sign-safe optimal direction with the least-squares pivot
    (fha_core.average_axis); "angle" is the weighted mean angle to it and "distance"
    the weighted RMS distance of the axes from the pivot.

The line "as printed by the original code" is the original pipeline's own summary
(plain mean of the direction vectors), for reference.

Run:  python 06_real_single_axis.py
"""
import contextlib
import io
import os

import numpy as np

from common import (POLHEMUS_FILES, VICON, VICON_FILES, load_polhemus, load_vicon, net_angle_deg,
                    orig, run_original)

import fha_core as fc
import vicon_import as vi

STEP = 10
FEMUR3 = ["lateral_epicondyl", "medial_epicondyl", "caput"]
FEMUR5 = FEMUR3 + ["femur_1", "femur_2"]
TIBIA3 = ["ankle", "lateral_condyl", "medial_condyl"]


def angle_bin_weights(angle_deg, tag, width=1.0):
    """Equal total weight for every `width` degrees of joint angle, per pass."""
    turn = np.argmax(angle_deg)
    key = np.floor(angle_deg[tag] / width).astype(int) * 2 + (tag > turn)
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w / w.sum()


def spread(n, p, weights):
    ok = np.isfinite(n).all(axis=1) & np.isfinite(p).all(axis=1)
    n, p, w = n[ok], p[ok], weights[ok] / weights[ok].sum()
    n_bar, pivot = fc.average_axis(n, p, w)
    d = fc.dispersion(n, p, n_bar, pivot, w)
    return d["angle_mean"], d["distance_rms"], d["angle_max"], int(ok.sum())


def evaluate(title, t, T1, T2, c1, c2, unit, position=True, extra=None):
    t, T1, T2 = t[c1:c2], T1[c1:c2], T2[c1:c2]
    angle_from_start = fc.rotation_from_start(fc.relative_poses(T1, T2))
    rows = []

    # original pipeline (snapshot), axes tagged at the window start
    res = run_original(t, T1, T2, T1[:, :3, 3], T2[:, :3, 3], 0, len(t), STEP)
    w = angle_bin_weights(angle_from_start, res["i0"])
    rows.append(("original (path length, tagged at start)", *spread(res["hax_g"], res["origins"], w)))
    printed = res["printed"]

    for label, kw in (("net rotation, forward window", dict(centred=False)),
                      ("net rotation, centred", dict()),
                      ("net rotation, centred, all-frames fit", dict(estimator="fit"))):
        fha = fc.compute_fha(t, T1, T2, "incremental_angle", STEP, **kw)
        n, p = fc.axes_to_lab(fha, T1)
        rows.append((label, *spread(n, p, angle_bin_weights(angle_from_start, fha.ic))))

    for label, (Ta, Tb) in (extra or {}).items():
        fha = fc.compute_fha(t, Ta[c1:c2], Tb[c1:c2], "incremental_angle", STEP)
        n, p = fc.axes_to_lab(fha, Ta[c1:c2])
        ang = fc.rotation_from_start(fc.relative_poses(Ta[c1:c2], Tb[c1:c2]))
        rows.append((label, *spread(n, p, angle_bin_weights(ang, fha.ic))))

    print(f"\n{title}   [frames {c1}:{c2}, peak rotation {angle_from_start.max():.1f} deg]")
    print(f"  as printed by the original code: {printed}")
    print(f"    {'method':54s} {'angle':>8s} {'max':>8s} {'distance':>12s} {'axes':>6s}")
    for label, ang, dist, amax, count in rows:
        dist_s = f"{dist:9.3f} {unit}" if position else f"{'-':>12s}"
        print(f"    {label:54s} {ang:7.3f}° {amax:7.2f}° {dist_s} {count:6d}")
    return rows


def vicon_variants(key):
    """Vicon poses from three ways of building the segment frames."""
    path = os.path.join(VICON, VICON_FILES[key])
    with contextlib.redirect_stdout(io.StringIO()):
        c3d, md, t = vi.read_data(path)
    R1, T1, R2, T2, l1, l2, res3 = vi.build_reference_frames_lsq(md, FEMUR3, TIBIA3)
    R1b, T1b, R2b, T2b, l1b, l2b, res5 = vi.build_reference_frames_lsq(md, FEMUR5, TIBIA3)
    return {
        "centred + least-squares frames (3 + 3 markers)": (T1, T2),
        "centred + least-squares frames (5 femur + 3 tibia)": (T1b, T2b),
    }


def main():
    t, T1, T2, _, _ = load_polhemus("1_oneway")
    evaluate("Polhemus 1 — pure hinge, one way", t, T1, T2, 100, len(t), "cm")
    t, T1, T2, _, _ = load_polhemus("1_full")
    evaluate("Polhemus 1 — pure hinge, out AND back", t, T1, T2, 0, len(t), "cm")
    t, T1, T2, _, _ = load_polhemus("5_by_sensor1")
    evaluate("Polhemus 5 — pure hinge moved by sensor 1, out AND back", t, T1, T2, 0, len(t), "cm")
    t, T1, T2, _, _ = load_polhemus("2_transl")
    evaluate("Polhemus 2 — hinge + sliding translation, out AND back (direction only)", t, T1, T2, 0, len(t), "cm",
             position=False)

    t, T1, T2, _, _ = load_vicon("servo_X")
    extra = vicon_variants("servo_X")
    evaluate("Vicon servo X — single servo, outbound", t, T1, T2, 200, 800, "mm", extra=extra)
    evaluate("Vicon servo X — single servo, out AND back", t, T1, T2, 0, len(t), "mm", extra=extra)


if __name__ == "__main__":
    main()
