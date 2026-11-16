"""Step 10 — two defects of the original shown on concrete cases (F10).

H3  the rotation angle is folded back above 135 degrees.
H2  a window can compare a frame with itself, and its "axis" is still drawn.

Run:  python 10_original_defects.py
"""
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from common import load_vicon, orig, run_original

import fha_core as fc


def main():
    print("H3 — angle returned for a pure rotation about z:")
    print(f"  {'true':>6s} {'original':>9s} {'revised':>8s}")
    for ang in (100, 134, 136, 150, 170):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(np.radians(ang) * np.array([0.0, 0.0, 1.0])).as_matrix()
        phi_old = orig.calculate_FHA_vectorized(np.eye(4)[None], T[None])[0][0]
        phi_new = fc.screw_from_displacement(T)[0]
        print(f"  {ang:5d}° {np.degrees(phi_old):8.1f}° {np.degrees(phi_new):7.1f}°")

    print("\nH2 — Vicon servo_X_20Z, whole recording, incremental_angle:")
    t, T1, T2, l1, l2 = load_vicon("servo_X_20Z")
    for step in (10, 5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = orig.generate_FHA("incremental_angle", t, 0, len(t), step, 1, T1[:, :3, :3], T2[:, :3, :3],
                                    T1, T2, l1, l2)
            res = run_original(t, T1, T2, l1, l2, 0, len(t), step)
        hax, ang, ind = np.asarray(out[0]), np.asarray(out[1]), out[10]
        same = [k for k, pair in enumerate(ind) if pair[0] == pair[1]]
        print(f"  step {step:2d}°: {len(ind)} windows, {len(same)} comparing a frame with itself")
        for k in same:
            print(f"     window {ind[k]}: 'axis' of length {np.linalg.norm(hax[k]):.5f}, angle {np.degrees(ang[k]):.1e}°, "
                  f"drawn in the cross-section at ({res['uv'][k, 0]:.0f}, {res['uv'][k, 1]:.0f}) mm")
        fha = fc.compute_fha(t, T1, T2, "incremental_angle", step)
        print(f"     revised code, same step: {len(fha)} axes, all unit length "
              f"({np.abs(np.linalg.norm(fha.n, axis=1) - 1).max():.1e} max deviation), "
              f"smallest window {np.degrees(fha.phi.min()):.2f}°")


if __name__ == "__main__":
    main()
