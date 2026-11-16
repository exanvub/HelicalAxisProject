"""Step 9 — where the original puts its cross-section plane, and what that does (F11).

The original plane sits 10 units along the average axis from the mean of the points `s`
(feet of the perpendiculars from sensor 1 onto each axis). This script measures how far
that is from the least-squares pivot — the point the axes actually converge on — and
how much larger the cross-section pattern is there than at the pivot.

The same revised axes are used for both planes, so the only thing that differs is where
the plane is.

Run:  python 09_plane_placement.py
"""
import numpy as np

from common import load_polhemus, load_vicon, run_original

import fha_core as fc

CASES = [
    ("Polhemus 1, pure hinge", load_polhemus, "1_oneway", 100, 1248, "cm"),
    ("Polhemus 3, hinge + x", load_polhemus, "3_y_plus_x", 100, 1683, "cm"),
    ("Polhemus 4, hinge + z", load_polhemus, "4_y_plus_z", 600, 1901, "cm"),
    ("Vicon servo X", load_vicon, "servo_X", 200, 800, "mm"),
]


def rms_radius(xy):
    return np.sqrt(np.mean(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1)))


def main():
    print(f"{'recording':26s} {'plane to pivot':>15s} {'radius at pivot':>16s} {'at original plane':>18s} "
          f"{'ratio':>6s} {'as drawn by original':>21s}")
    for title, loader, key, c1, c2, unit in CASES:
        t, T1, T2, l1, l2 = loader(key)
        res = run_original(t, T1, T2, l1, l2, c1, c2, 10)
        T1c, T2c = T1[c1:c2], T2[c1:c2]
        fha = fc.compute_fha(t[c1:c2], T1c, T2c, "incremental_angle", 10)
        n, p = fc.axes_to_lab(fha, T1c)
        n_bar, pivot = fc.average_axis(n, p, fha.weight)
        e1, e2 = fc.plane_basis(n_bar, T2c[:, :3, 3].mean(axis=0) - pivot)
        lever = (res["plane_point"] - pivot) @ n_bar
        at_pivot = rms_radius(fc.cross_section(n, p, n_bar, pivot, e1, e2))
        at_orig = rms_radius(fc.cross_section(n, p, n_bar, pivot + lever * n_bar, e1, e2))
        print(f"{title:26s} {abs(lever):12.2f} {unit} {at_pivot:13.3f} {unit} {at_orig:15.3f} {unit} "
              f"{at_orig / at_pivot:5.1f}x {rms_radius(res['uv']):18.3f} {unit}")


if __name__ == "__main__":
    main()
