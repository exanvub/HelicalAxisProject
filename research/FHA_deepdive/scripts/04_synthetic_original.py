"""Step 4 — ground-truth test of the ORIGINAL pipeline.

Feeds the original miFHA code a simulated rig whose true helical axis is known at every
frame (FHA/FHA_plotly_fast/synthetic.py), with Polhemus-level noise. The four cases
differ ONLY in how the second rotation is delivered as the hinge turns:

  none          pure hinge
  proportional  steady: second rotation in fixed proportion to the hinge angle
  late          phased in smoothly from mid-range
  pulsed        same 20 deg in total, delivered in 8 bursts

If the method invented fingers, they would show up in the smooth cases too.

Run:  python 04_synthetic_original.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG, run_original

import synthetic as syn

STEP = 10
CASES = [
    ("none", "pure hinge"),
    ("proportional", "second rotation steady (fixed ratio)"),
    ("late", "second rotation phased in smoothly"),
    ("pulsed", "second rotation in 8 bursts"),
]


def plane_hit(n, s, p0, nrm):
    """Intersection of lines (s + x n) with the plane through p0 with normal nrm."""
    x = ((p0 - s) @ nrm) / (n @ nrm)
    return s + x[:, None] * n


def main():
    fig, axes = plt.subplots(len(CASES), 3, figsize=(19, 4.6 * len(CASES)))
    print(f"{'case':14s} {'dir err vs truth @window MIDDLE':>34s} {'dir err vs truth @window START':>34s} "
          f"{'in-plane err @middle':>22s} {'spread of truth':>16s}")
    for row, (profile, label) in enumerate(CASES):
        d = syn.two_axis_motion(profile=profile, seed=1)
        t, T1, T2 = d["t"], d["T1"], d["T2"]
        res = run_original(t, T1, T2, T1[:, :3, 3], T2[:, :3, 3], 0, len(t), STEP)
        i0, i1 = res["i0"], res["i1"]
        mid = (i0 + i1) // 2
        ok = d["moving"][mid] & d["moving"][i0]

        ang = lambda a, b: np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", a, b), -1, 1)))
        err_mid = ang(res["hax_g"][ok], d["truth_n"][mid][ok])
        err_start = ang(res["hax_g"][ok], d["truth_n"][i0][ok])

        truth_uv3 = plane_hit(d["truth_n"][mid][ok], d["truth_s"][mid][ok], res["plane_point"], res["normal"])
        tu = np.c_[(truth_uv3 - res["plane_point"]) @ res["e1"], (truth_uv3 - res["plane_point"]) @ res["e2"]]
        eu = res["uv"][ok]
        plane_err = np.linalg.norm(eu - tu, axis=1)
        print(f"{profile:14s} {np.mean(err_mid):14.3f} deg (max {err_mid.max():6.3f}) "
              f"{np.mean(err_start):14.3f} deg (max {err_start.max():6.3f}) "
              f"{np.mean(plane_err):12.3f} (max {plane_err.max():.3f}) {tu.std(0).max():12.3f}")

        ax = axes[row, 0]
        ax.plot(d["alpha"], d["beta"], "C1")
        ax.set_title(f"{label}\ninput: second rotation vs hinge angle", fontsize=9)
        ax.set_xlabel("hinge angle (deg)"); ax.set_ylabel("second rotation (deg)"); ax.grid(alpha=0.3)
        ax.set_ylim(-1, 22)

        ax = axes[row, 1]
        ax.plot(tu[:, 0], tu[:, 1], "-", color="k", lw=2.5, alpha=0.35, label="TRUE axis (ground truth)", zorder=1)
        sc = ax.scatter(eu[:, 0], eu[:, 1], c=t[i0][ok], cmap="magma", s=7, zorder=2, label="original miFHA")
        plt.colorbar(sc, ax=ax, label="time (s)")
        ax.set_aspect("equal", "datalim")
        ax.set_title("cross-section: original miFHA vs the true axis", fontsize=9)
        ax.set_xlabel("plane axis 1 (cm)"); ax.set_ylabel("plane axis 2 (cm)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[row, 2]
        tilt_true = np.degrees(np.arctan(d["slope"]))
        ax.plot(d["alpha"][d["moving"]], tilt_true[d["moving"]], "k", lw=1, label="true instantaneous tilt = atan(rate ratio)")
        hinge_tilt = ang(res["hax_g"][ok], np.tile(d["hinge_axis"], (ok.sum(), 1)))
        ax.plot(d["alpha"][mid][ok], hinge_tilt, "C0", lw=1.4, label=f"original miFHA ({STEP} deg windows)")
        ax.set_title("tilt of the axis away from the hinge", fontsize=9)
        ax.set_xlabel("hinge angle at window middle (deg)"); ax.set_ylabel("deg")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{FIG}/04_synthetic_original.png"
    fig.savefig(path, dpi=75)
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
