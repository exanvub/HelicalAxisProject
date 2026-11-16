"""Step 11 — do the error-reduction figures shown at ISB 2025 hold up? (F13)

ISB 2025, pure rotation, mean angle between the axes and their average axis:
    FHA 5.01 deg    fiFHA 0.33 deg    miFHA 0.34 deg

This runs the ORIGINAL code and the revised code on the one-way pure-hinge Polhemus
recording, for two frame ranges (with and without the still phase before the movement)
and two window sizes.

Run:  python 11_isb_numbers.py
"""
import contextlib
import io
import warnings

import numpy as np

from common import load_polhemus, orig

import fha_core as fc

METHODS = [("FHA, consecutive frames", "all_FHA", 0), ("fiFHA, 5 deg", "step_angle", 5),
           ("miFHA, 5 deg", "incremental_angle", 5), ("fiFHA, 10 deg", "step_angle", 10),
           ("miFHA, 10 deg", "incremental_angle", 10)]
RANGES = [((100, 1248), "frames 100-1248: 3 s at rest, then the movement"),
          ((480, 1200), "frames 480-1200: the movement only")]


def original(t, T1, T2, l1, l2, method, step, c1, c2):
    out = orig.generate_FHA(method, t, c1, c2, step, 1, T1[:, :3, :3], T2[:, :3, :3], T1, T2,
                            np.asarray(l1), np.asarray(l2))
    hax, ang, svec, d, tr1, tr2, time_diff, time_incr, ang_incr, all_angles, ind_incr, ind_step, tcut = out
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        orig.plotly_visualize_FHA(True, T1[c1:c2], T2[c1:c2], method, hax, svec, d, ind_incr, ind_step, 1,
                                  tr1, tr2, tcut, time_incr, ang_incr, step, all_angles, time_diff)
    return float(buf.getvalue().strip().splitlines()[-1].split(":")[1].split()[0]), len(hax)


def revised(t, T1, T2, method, step, c1, c2):
    f = fc.compute_fha(t[c1:c2], T1[c1:c2], T2[c1:c2], method, step)
    n_bar, pivot = fc.average_axis(f.n, f.s, f.weight)
    return fc.dispersion(f.n, f.s, n_bar, pivot, f.weight)["angle_mean"], len(f)


def main():
    warnings.simplefilter("ignore")
    t, T1, T2, l1, l2 = load_polhemus("1_oneway")
    print(f"{'':34s} {'original code':>22s} {'revised code':>22s}")
    for (c1, c2), label in RANGES:
        print(label)
        for name, method, step in METHODS:
            o_ang, o_n = original(t, T1, T2, l1, l2, method, step, c1, c2)
            r_ang, r_n = revised(t, T1, T2, method, max(step, 1), c1, c2)
            print(f"  {name:32s} {o_ang:8.2f} deg ({o_n:4d} axes) {r_ang:8.2f} deg ({r_n:4d} axes)")


if __name__ == "__main__":
    main()
