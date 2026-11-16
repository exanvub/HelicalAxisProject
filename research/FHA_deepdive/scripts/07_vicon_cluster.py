"""Step 7 — how rigid are the Vicon marker clusters, and what does that do to the axes?

A rigid body keeps its inter-marker distances. Any change in them is measurement error
(or a marker that is not rigidly attached), and on a short baseline it turns into
apparent rotation. This script reports, per recording:

  * the residual of a least-squares rigid-body fit to each cluster;
  * how far the tibia inter-marker distances move while the joint moves;
  * the correlation between the rate of cluster-shape change and the measured
    secondary rotation rate (real rigid rotation would give no correlation);
  * every one-frame orientation jump above 0.8 deg: which marker moved, whether the
    rigid-fit residual notices it, and whether the pose comes back (spike) or stays
    (step).

Run:  python 07_vicon_cluster.py
"""
import contextlib
import io
import os

import numpy as np
from scipy.spatial.transform import Rotation

from common import VICON, VICON_FILES

import fha_core as fc
import vicon_import as vi

FEMUR = ["lateral_epicondyl", "medial_epicondyl", "caput"]
FEMUR5 = FEMUR + ["femur_1", "femur_2"]
TIBIA = ["ankle", "lateral_condyl", "medial_condyl"]
MOVING = {"servo_X": (200, 800), "servo_X_20Y": (200, 760), "servo_X_20Z": (180, 740)}
PAIRS = [(0, 1), (0, 2), (1, 2)]
H = 10  # half-width, in frames, of the central differences (+-0.1 s at 100 Hz)
FS = 100.0


def main():
    for key, fn in VICON_FILES.items():
        with contextlib.redirect_stdout(io.StringIO()):
            c3d, md, t = vi.read_data(os.path.join(VICON, fn))
        X = vi.marker_array(md, TIBIA)
        R1, T1, R2, T2, l1, l2, res = vi.build_reference_frames_lsq(md, FEMUR5, TIBIA)
        Trel = fc.relative_poses(T1, T2)
        q = fc.quaternions(Trel)
        print(f"\n=== {key} ===")
        for name, label in (("residual_j1", "femur (5 markers)"), ("residual_j2", "tibia (3 markers)")):
            r = res[name]
            print(f"  rigid-fit residual, {label}: median {np.median(r):.3f} mm, "
                  f"99th pct {np.percentile(r, 99):.3f} mm, max {r.max():.3f} mm")

        c1, c2 = MOVING[key]
        D = np.stack([np.linalg.norm(X[:, a] - X[:, b], axis=1) for a, b in PAIRS], axis=1)
        k = np.arange(c1 + H, c2 - H)
        Rr = Trel[:, :3, :3]
        w = np.degrees(Rotation.from_matrix(np.einsum("nij,nkj->nik", Rr[k + H], Rr[k - H])).as_rotvec()) * FS / (2 * H)
        main_dir = w.mean(axis=0) / np.linalg.norm(w.mean(axis=0))
        w_par = np.abs(w @ main_dir)
        w_perp = np.linalg.norm(w - np.outer(w @ main_dir, main_dir), axis=1)
        mov = w_par > 0.25 * np.percentile(w_par, 90)
        shape_rate = np.linalg.norm(np.abs(D[k + H] - D[k - H]) * FS / (2 * H), axis=1)   # mm/s
        base = np.median(D[:, 2])
        print("  tibia inter-marker distances while moving: "
              + "; ".join(f"{TIBIA[a]}-{TIBIA[b]} {np.median(D[k[mov], i]):.1f} mm (range {np.ptp(D[k[mov], i]):.2f})"
                          for i, (a, b) in enumerate(PAIRS)))
        print(f"  cluster-shape change rate while moving: median {np.median(shape_rate[mov]):.2f} mm/s "
              f"= {np.degrees(np.median(shape_rate[mov]) / base):.1f} deg/s of apparent rotation on the {base:.0f} mm baseline "
              f"(90th pct {np.degrees(np.percentile(shape_rate[mov], 90) / base):.1f} deg/s)")
        print(f"  measured secondary rotation rate while moving: median {np.median(w_perp[mov]):.1f} deg/s, "
              f"90th pct {np.percentile(w_perp[mov], 90):.1f} deg/s")
        print(f"  correlation, shape-change rate vs secondary rate: r = {np.corrcoef(shape_rate[mov], w_perp[mov])[0, 1]:.2f}")

        if key == "servo_X_20Z":
            # one marker sits out of place between two jumps; compare that stretch with the rest
            Xr, r_tibia = X, vi.rigid_body_fit(X)[3]
            d = np.linalg.norm(Xr[:, 0] - Xr[:, 1], axis=1)
            for a, b in ((0, 709), (710, 1312), (1313, len(d))):
                print(f"  frames {a:4d}-{b:4d} ({t[a]:.2f}-{t[min(b, len(t) - 1)]:.2f} s): tibia residual median "
                      f"{np.median(r_tibia[a:b]):.2f} mm, {TIBIA[0]}-{TIBIA[1]} distance median {np.median(d[a:b]):.2f} mm")
            rest = np.r_[r_tibia[:709], r_tibia[1313:]]
            print(f"  outside 710-1312: tibia residual median {np.median(rest):.2f} mm")

        step = np.degrees(fc.net_angle(q[:-1], q[1:]))
        jumps = np.flatnonzero(step > 0.8)
        if not len(jumps):
            print("  no one-frame orientation jumps above 0.8 deg")
        for j in jumps:
            move = np.linalg.norm(X[j + 1] - X[j], axis=1)                 # per-marker displacement, mm
            lo, hi = max(j - 3, 0), min(j + 4, len(q) - 1)
            across = np.degrees(fc.net_angle(q[lo], q[hi]))                 # still displaced 3 frames later?
            typical = np.median(step[max(j - 30, 0):j + 30]) * (hi - lo)
            kind = "STEP (stays)" if across > typical + 0.5 * step[j] else "spike (returns)"
            print(f"  jump at frame {j}->{j + 1}: {step[j]:.2f} deg | marker moves (mm) "
                  + ", ".join(f"{n} {m:.2f}" for n, m in zip(TIBIA, move))
                  + f" | tibia residual {res['residual_j2'][j]:.2f} -> {res['residual_j2'][j + 1]:.2f} mm | {kind}")


if __name__ == "__main__":
    main()
