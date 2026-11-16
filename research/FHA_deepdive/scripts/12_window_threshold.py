"""Step 12 — is the window threshold applied to the right quantity? (F14)

What decides whether an FHA is well or ill conditioned is the NET rotation between the
two poses it is computed from. This script measures that net rotation for every window,
under three rules, on whole recordings (out and back where recorded), for a requested
window of 10 degrees:

  summed   ORIGINAL: add up frame-to-frame angles from the start frame until the sum
           exceeds the request.
  halves   net rotation of step/2 from the centre frame, on each side (the first
           revision of this work); the two ends are then up to 6% short of the request
           when the axis changes direction inside the window.
  net      CURRENT: as "halves", then widened until the two ENDS are a full request
           apart.

Run:  python 12_window_threshold.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG, load_polhemus, load_vicon, net_angle_deg, orig

import fha_core as fc

STEP = 10.0
CASES = [
    ("Polhemus 1, pure hinge", load_polhemus, "1_full"),
    ("Polhemus 3, hinge + x", load_polhemus, "3_y_plus_x"),
    ("Polhemus 4, hinge + z", load_polhemus, "4_y_plus_z"),
    ("Vicon servo X", load_vicon, "servo_X"),
    ("Vicon servo X + 20 Z", load_vicon, "servo_X_20Z"),
]


def summed(t, T1, T2):
    out = orig.generate_FHA("incremental_angle", t, 0, len(t), int(STEP), 1, T1[:, :3, :3], T2[:, :3, :3],
                            T1, T2, T1[:, :3, 3], T2[:, :3, 3])
    ind = out[10]
    i0 = np.array([p[0] for p in ind]); i1 = np.array([p[1] for p in ind])
    Trel = np.linalg.inv(T1) @ T2
    return net_angle_deg(Trel[i0, :3, :3], Trel[i1, :3, :3])


def halves(q):
    """Half-windows only, no widening (max_half below 0.5 would drop everything, so the
    widening is switched off by asking for ends only 1e-9 apart)."""
    step = np.radians(STEP)
    back, fwd = fc.first_reach(q, step / 2, -1), fc.first_reach(q, step / 2, +1)
    ic = np.flatnonzero((back >= 0) & (fwd >= 0))
    net = np.degrees(fc.net_angle(q[back[ic]], q[fwd[ic]]))
    return net[net >= 0.7 * STEP]


def line(x):
    return (f"{len(x):5d} windows | min {x.min():6.2f}  1st pct {np.percentile(x, 1):6.2f}  median {np.median(x):6.2f} | "
            f"below 9 deg: {100 * np.mean(x < 9):5.1f}%  below 5 deg: {100 * np.mean(x < 5):5.1f}%")


def main():
    fig, axes = plt.subplots(1, len(CASES), figsize=(4.2 * len(CASES), 3.8), sharey=True)
    for ax, (name, loader, key) in zip(axes, CASES):
        t, T1, T2, _, _ = loader(key)
        q = fc.quaternions(fc.relative_poses(T1, T2))
        path = np.degrees(fc.path_length(q)[-1])
        peak = np.degrees(fc.net_angle(q, q[0][None, :])).max()
        rules = {"summed (original)": summed(t, T1, T2), "halves": halves(q),
                 "net (current)": np.degrees(fc.compute_fha(t, T1, T2, "incremental_angle", STEP).phi)}
        print(f"\n{name}: summed frame-to-frame angles over the file {path:.1f} deg, largest net rotation {peak:.1f} deg")
        for label, x in rules.items():
            print(f"  {label:18s} {line(x)}")
        bins = np.linspace(0, 12, 49)
        for label, x in rules.items():
            ax.hist(np.clip(x, 0, 12), bins=bins, histtype="step", lw=1.4, label=label, density=True)
        ax.axvline(STEP, color="0.5", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("net rotation between the two poses compared (deg)", fontsize=8)
    axes[0].set_ylabel("share of windows (log)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out = f"{FIG}/12_window_threshold.png"
    fig.savefig(out, dpi=80)

    print("\nOther window sizes, Polhemus 3, current rule:")
    t, T1, T2, _, _ = load_polhemus("3_y_plus_x")
    for step in (2, 5, 10, 20, 30):
        phi = np.degrees(fc.compute_fha(t, T1, T2, "incremental_angle", step).phi)
        print(f"  request {step:2d} deg: {len(phi):5d} windows, min {phi.min():7.3f}, median {np.median(phi):6.2f}")
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
