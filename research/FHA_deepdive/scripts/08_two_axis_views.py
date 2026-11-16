"""Step 8 — what the revised pipeline shows on two-axis motion.

Rows: two simulated rigs (steady and pulsed second rotation, ground truth known) and the
two hand-driven Polhemus recordings. Columns:

  1  cross-section exactly as the original code draws it
  2  moving-window FHA (miFHA), revised: direction map — tilt of each axis from the
     average axis, in degrees
  3  same axes: position map — where they cross the plane through the pivot
  4  reference-pose FHA: direction map
  5  rotation off the recording's own AVERAGE axis (swing) against rotation about it (twist).
     This understates the second rotation and makes it look as if it were undone; step 13
     (13_build_up.py) shows why and measures it against the hinge instead.

Also prints how much each trajectory doubles back (path length / end-to-end distance in
the direction map) for several window sizes.

Run:  python 08_two_axis_views.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG, load_polhemus, run_original

import fha_core as fc
import synthetic as syn


def bundle(fha, Trel):
    """Average axis, in-plane basis oriented toward the moving sensor, and both maps."""
    n_bar, pivot = fc.average_axis(fha.n, fha.s, fha.weight)
    e1, e2 = fc.plane_basis(n_bar, Trel[:, :3, 3].mean(axis=0) - pivot)
    return dict(n_bar=n_bar, pivot=pivot, e1=e1, e2=e2,
                tilt=fc.tilt_map(fha.n, n_bar, e1, e2),
                pos=fc.cross_section(fha.n, fha.s, n_bar, pivot, e1, e2))


def doubling_back(xy):
    """Path length over end-to-end distance: 1 for a straight run, large for fingers."""
    path = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    return path / max(np.linalg.norm(xy[-1] - xy[0]), 1e-9), path


def scatter(ax, xy, c, title, xlabel, ylabel, cbar=True):
    ax.plot(xy[:, 0], xy[:, 1], "-", color="0.8", lw=0.6, zorder=1)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap="viridis", s=6, zorder=2)
    if cbar:
        plt.colorbar(sc, ax=ax, label="time (s)")
    ax.axhline(0, color="0.6", lw=0.5); ax.axvline(0, color="0.6", lw=0.5)
    ax.set_aspect("equal", "datalim")
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(alpha=0.3)


def case_synthetic(profile):
    d = syn.two_axis_motion(profile=profile, noise="polhemus", fs=120.0, move_s=8.0, seed=2)
    return d["t"], d["T1"], d["T2"], 0, len(d["t"]), "cm"


def case_polhemus(key, c1, c2):
    t, T1, T2, _, _ = load_polhemus(key)
    return fc.uniform_time(t), T1, T2, c1, c2, "cm"


CASES = [
    ("SIMULATED: second rotation steady", lambda: case_synthetic("proportional")),
    ("SIMULATED: second rotation in 8 bursts", lambda: case_synthetic("pulsed")),
    ("Polhemus 3: hinge + x by hand (outbound)", lambda: case_polhemus("3_y_plus_x", 100, 1683)),
    ("Polhemus 4: hinge + z by hand (outbound)", lambda: case_polhemus("4_y_plus_z", 600, 1901)),
]


def main():
    fig, axes = plt.subplots(len(CASES), 5, figsize=(27, 4.9 * len(CASES)))
    print(f"{'case':42s} {'method':24s} {'axes':>6s} {'doubling back':>14s} {'path (deg)':>11s} {'mean tilt':>10s} {'RMS dist to pivot':>18s}")
    for row, (title, make) in enumerate(CASES):
        t, T1, T2, c1, c2, unit = make()
        res = run_original(t, T1, T2, T1[:, :3, 3], T2[:, :3, 3], c1, c2, 10)
        t, T1, T2 = t[c1:c2], T1[c1:c2], T2[c1:c2]
        Trel = fc.relative_poses(T1, T2)

        sets = {}
        for step in (10, 20, 30):
            sets[f"miFHA {step} deg"] = fc.compute_fha(t, T1, T2, "incremental_angle", step)
        sets["reference pose"] = fc.compute_fha(t, T1, T2, "reference_pose", 10)
        for name, fha in sets.items():
            b = bundle(fha, Trel)
            ratio, path = doubling_back(b["tilt"])
            disp = fc.dispersion(fha.n, fha.s, b["n_bar"], b["pivot"], fha.weight)
            print(f"{title:42s} {name:24s} {len(fha):6d} {ratio:14.2f} {path:11.1f} {disp['angle_mean']:9.2f}° "
                  f"{disp['distance_rms']:13.3f} {unit}")

        mi, ref = sets["miFHA 10 deg"], sets["reference pose"]
        bm, br = bundle(mi, Trel), bundle(ref, Trel)
        scatter(axes[row, 0], res["uv"], res["t"], f"{title}\noriginal code: cross-section 10 units along the axis",
                f"plane axis 1 ({unit})", f"plane axis 2 ({unit})")
        scatter(axes[row, 1], bm["tilt"], mi.t, "revised miFHA, 10 deg windows\ndirection map: tilt from the average axis",
                "tilt toward moving sensor (deg)", "tilt across (deg)")
        scatter(axes[row, 2], bm["pos"], mi.t, "revised miFHA, 10 deg windows\nposition map: plane through the pivot",
                f"toward moving sensor ({unit})", f"across ({unit})")
        scatter(axes[row, 3], br["tilt"], ref.t, "reference-pose FHA\ndirection map",
                "tilt toward moving sensor (deg)", "tilt across (deg)")

        twist, swing = fc.swing_twist(Trel, bm["n_bar"])
        ax = axes[row, 4]
        ax.plot(np.abs(twist), swing @ bm["e1"], label="swing toward moving sensor")
        ax.plot(np.abs(twist), swing @ bm["e2"], label="swing across")
        ax.plot(np.abs(twist), np.linalg.norm(swing, axis=1), "k--", lw=0.8, label="swing magnitude")
        ax.set_title("rotation off the recording's own AVERAGE axis vs rotation about it\n"
                     "(understates the second rotation: see step 13)", fontsize=8.5)
        ax.set_xlabel("twist about the average axis (deg)", fontsize=8); ax.set_ylabel("swing (deg)", fontsize=8)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        print()
    fig.tight_layout()
    path = f"{FIG}/08_two_axis_views.png"
    fig.savefig(path, dpi=62)
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
