"""Step 14 — can the primary axis be found from the recording itself? (F16)

Measuring the build-up needs a primary axis (step 13). A recording of the hinge alone
supplies one. This script asks whether one can do without it, by fitting a TWO-AXIS
model to the recording under study: one axis fixed in the reference body, one fixed in
the moving body, such that the angle between them stays constant —
    R_t = Rot(a, alpha_t) . R_ref . Rot(b, beta_t).
The fit minimises the variance of the angle between a and R_t b over both axes (global
search on a grid of directions, then least squares).

Result, in short: it works when the two rotations vary independently (bursts) and is
not identifiable when they vary together; it is degenerate on a plain hinge; and it
is biased when the motion is not exactly two hinges. Not used in the dashboard.

Run:  python 14_two_axis_fit.py
"""
import numpy as np
from scipy.optimize import least_squares

from common import load_polhemus

import fha_core as fc
import synthetic as syn


def sphere(n):
    """n roughly even directions on the unit sphere (Fibonacci lattice)."""
    k = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * k / n), np.pi * (1 + 5 ** 0.5) * k
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def tangent(v):
    ref = np.eye(3)[np.argmin(np.abs(v))]
    e1 = np.cross(v, ref); e1 /= np.linalg.norm(e1)
    return e1, np.cross(v, e1)


def between(R, a, b):
    """Angle (rad) between axis a (reference body) and axis b (moving body), per frame."""
    u = np.einsum("nij,j->ni", R, b)
    return np.arctan2(np.linalg.norm(np.cross(u, a), axis=-1), u @ a)


def fit(R, w, n_grid=200, n_refine=6):
    sw = np.sqrt(w / w.sum())
    dirs = sphere(n_grid)
    U = np.einsum("nij,kj->kni", R, dirs)
    ang = np.arccos(np.clip(np.einsum("kni,ji->kjn", U, dirs), -1, 1))       # (b, a, frame)
    mean = np.einsum("kjn,n->kj", ang, sw**2)
    cost = np.einsum("kjn,n->kj", (ang - mean[..., None]) ** 2, sw**2)
    cost[(mean < np.radians(15)) | (mean > np.radians(165))] = np.inf       # nearly parallel: no split
    seeds = []
    for o in np.argsort(cost, axis=None)[:400]:
        kb, ka = np.unravel_index(o, cost.shape)
        if all(abs(dirs[ka] @ ua) < 0.97 or abs(dirs[kb] @ ub) < 0.97 for ua, ub in seeds):
            seeds.append((dirs[ka], dirs[kb]))
        if len(seeds) == n_refine:
            break
    best = None
    for a0, b0 in seeds:
        ea, eb = tangent(a0), tangent(b0)

        def unpack(x):
            a = a0 + x[0] * ea[0] + x[1] * ea[1]; b = b0 + x[2] * eb[0] + x[3] * eb[1]
            return a / np.linalg.norm(a), b / np.linalg.norm(b)

        def resid(x):
            g = between(R, *unpack(x))
            return sw * (g - np.sum(sw**2 * g))

        sol = least_squares(resid, np.zeros(4), method="lm")
        if best is None or sol.cost < best[2]:
            best = (*unpack(sol.x), sol.cost)
    a, b, cost = best
    return a, b, np.degrees(np.sqrt(2 * cost))


def rotations(R, a, b):
    """Rotation about each fitted axis since the first frame (deg)."""
    u, wv = np.einsum("nij,j->ni", R, b), np.einsum("nji,j->ni", R, a)

    def about(axis, v):
        p = v - np.outer(v @ axis, axis)
        return np.degrees(np.unwrap(np.arctan2(np.cross(p[0][None, :], p) @ axis, p @ p[0])))

    return about(a, u), -about(b, wv)


def angle(u, v):
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), abs(u @ v))))


def sample(t, T1, T2, n=150):
    """Frames spaced evenly in rotation, with the first frame as the reference pose."""
    Trel = fc.relative_poses(T1, T2)
    f = fc.compute_fha(t, T1, T2, "incremental_angle", 10)
    cum = np.cumsum(f.weight)
    pick = np.unique(np.searchsorted(cum, np.linspace(cum[0], cum[-1], n)))
    frames = np.r_[0, f.ic[pick[pick < len(f)]]]
    return Trel[frames, :3, :3], np.ones(len(frames)), frames


def report(label, t, T1, T2, hinge, truth_second=None):
    R, w, frames = sample(t, T1, T2)
    a, b, rms = fit(R, w)
    al, be = rotations(R, a, b)
    apart = float(np.degrees(between(R, a, b)).mean())
    near = min(angle(a, hinge), angle(R[0] @ b, hinge))
    second = min(np.ptp(al), np.ptp(be))
    extra = "" if truth_second is None else f" (true {truth_second:.1f})"
    print(f"  {label:34s} nearest fitted axis is {near:5.2f} deg from the hinge | axes {apart:5.1f} deg apart | "
          f"second rotation {second:5.1f} deg{extra} | misfit {rms:.3f} deg")


def main():
    rig = dict(fs=120.0, move_s=5.5, still_s=1.0)
    for noise in ("none", "polhemus", "vicon"):
        print(f"SIMULATED, measurement error: {noise}")
        for label, kw in (("bursts", dict(profile="pulsed")), ("phased in", dict(profile="late")),
                          ("steady (fixed ratio)", dict(profile="proportional")), ("pure hinge", dict(profile="none"))):
            d = syn.two_axis_motion(noise=noise, seed=1, **kw, **rig)
            hinge = d["T1_clean"][0, :3, :3].T @ d["hinge_axis"]
            report(label, d["t"], d["T1"], d["T2"], hinge, d["beta"][-1])

    print("\nREAL (Polhemus, outbound). 'The hinge' is the average axis of recording 1; the second rotation measured")
    print("against it (step 13) is 0.15, 23.8 and 31.4 deg for recordings 1, 3 and 4.")
    t, T1, T2, _, _ = load_polhemus("1_full")
    f = fc.compute_fha(t, T1, T2, "incremental_angle", 10)
    hinge = fc.average_axis(f.n, f.s, f.weight)[0]
    for label, key, c1, c2 in (("recording 1, pure hinge", "1_full", 0, 1280),
                               ("recording 3, hinge + x by hand", "3_y_plus_x", 100, 1683),
                               ("recording 4, hinge + z by hand", "4_y_plus_z", 600, 1901)):
        t, T1, T2, _, _ = load_polhemus(key)
        report(label, t[c1:c2], T1[c1:c2], T2[c1:c2], hinge)


if __name__ == "__main__":
    main()
