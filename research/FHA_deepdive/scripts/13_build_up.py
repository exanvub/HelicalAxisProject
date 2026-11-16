"""Step 13 — keeping track of accumulated rotation instead of the direction of the axis (F15).

The idea: a burst of second rotation makes a finger in the direction map because the
axis tilts while the burst lasts and returns afterwards. If instead the rotation away
from the primary axis is ADDED UP, a burst should raise the total and leave it raised.

This script tests three ways of adding it up, against ground truth (simulated rig) and
on the hand-driven Polhemus recordings:

  (1) sum the off-axis part of each window's rotation, measured against the recording's
      OWN average axis                                    — the idea as first stated
  (2) swing about the recording's OWN average axis       — what the dashboard showed before
  (3) swing about the PRIMARY axis, taken from a recording of the hinge alone
                                                         — what the dashboard shows now

It also checks the premise of (3) on the real data: that the hinge axis, written in the
frame of the reference sensor, is the same in every recording of the session.

Run:  python 13_build_up.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from common import FIG, load_polhemus

import fha_core as fc
import synthetic as syn

RIG = dict(fs=120.0, move_s=5.5, still_s=1.0, noise="polhemus")


def average_axis(t, T1, T2):
    f = fc.compute_fha(t, T1, T2, "incremental_angle", 10)
    return fc.average_axis(f.n, f.s, f.weight)[0]


def angle(u, v):
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), abs(u @ v))))


def build_ups(t, T1, T2, hinge):
    """Accumulated off-axis rotation by the three methods, per frame or per window."""
    Trel = fc.relative_poses(T1, T2)
    own = average_axis(t, T1, T2)
    # (1) consecutive 5-degree windows; off-axis part of each window's rotation vector, summed
    steps = fc.compute_fha(t, T1, T2, "step_angle", 5)
    rho = np.degrees(steps.phi)[:, None] * steps.n
    summed = np.linalg.norm(np.cumsum(rho - np.outer(rho @ own, own), axis=0), axis=1)
    # (2) and (3): swing of the total displacement about an axis
    tw_own, sw_own = fc.swing_twist(Trel, own)
    tw_hinge, sw_hinge = fc.swing_twist(Trel, hinge, turning=True)
    return dict(own_axis=own, summed=summed, summed_frames=steps.i1,
                own=(np.abs(tw_own), np.linalg.norm(sw_own, axis=1)),
                hinge=(np.abs(tw_hinge), np.linalg.norm(sw_hinge, axis=1)), hinge_swing=sw_hinge)


def main():
    fig, axes = plt.subplots(2, 4, figsize=(21, 9))

    # ---------------------------------------------------------------- ground truth
    print("SIMULATED RIG (Polhemus-like error): 90 deg about the hinge, 20 deg of second rotation")
    ref = syn.two_axis_motion(profile="none", seed=11, **RIG)
    hinge = average_axis(ref["t"], ref["T1"], ref["T2"])           # from a separate pure-hinge "recording"
    true_hinge = ref["T1_clean"][0, :3, :3].T @ ref["hinge_axis"]
    print(f"  hinge axis taken from a simulated pure-hinge recording: {angle(hinge, true_hinge):.2f} deg from the true hinge")
    print(f"  {'second rotation':16s} {'own axis vs hinge':>18s} | {'(1) windows summed, own axis':>30s} | {'(2) swing, own axis':>22s} | {'(3) swing, hinge axis':>34s}")
    for col, (profile, label) in enumerate((("proportional", "steady"), ("late", "phased in"), ("pulsed", "in 8 bursts"))):
        d = syn.two_axis_motion(profile=profile, seed=1, **RIG)
        b = build_ups(d["t"], d["T1"], d["T2"], hinge)
        err = np.abs(b["hinge"][1] - d["beta"]).max()
        print(f"  {label:16s} {angle(b['own_axis'], hinge):15.1f} deg | max {b['summed'].max():5.2f}  end {b['summed'][-1]:5.2f} deg"
              f"{'':10s} | max {b['own'][1].max():5.2f}  end {b['own'][1][-1]:5.2f} deg | max {b['hinge'][1].max():5.2f}  end {b['hinge'][1][-1]:5.2f} deg"
              f"  (worst error {err:.2f})")
        if profile != "late":
            ax = axes[0, 0 if profile == "proportional" else 1]
            ax.plot(d["alpha"], d["beta"], color="k", lw=3, alpha=0.25, label="true second rotation")
            ax.plot(*b["hinge"], color="C0", lw=1.6, label="(3) swing about the hinge axis")
            ax.plot(*b["own"], color="C1", lw=1.6, label="(2) swing about the recording's own average axis")
            ax.plot(np.abs(fc.swing_twist(fc.relative_poses(d["T1"], d["T2"]), b["own_axis"])[0])[b["summed_frames"]],
                    b["summed"], "C3.", ms=5, label="(1) windows summed, own average axis")
            ax.set_title(f"simulated: second rotation {label}", fontsize=10)
            ax.set_xlabel("rotation about the axis (deg)"); ax.set_ylabel("accumulated rotation away from it (deg)")
            ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # ------------------------------------------------------------------- real data
    print("\nREAL RECORDINGS — premise: the hinge, in the frame of the reference sensor, is the same in every recording")
    recs = {key: load_polhemus(key) for key in ("1_full", "1_oneway", "2_transl", "3_y_plus_x", "4_y_plus_z")}
    hinge = average_axis(*recs["1_full"][:3])
    base = Rotation.from_matrix(recs["1_full"][1][:, :3, :3]).mean()
    for key, (t, T1, T2, _, _) in recs.items():
        own = average_axis(t, T1, T2)
        turn = np.degrees((base.inv() * Rotation.from_matrix(T1[:, :3, :3]).mean()).magnitude())
        move = np.linalg.norm(T1[:, :3, 3].mean(axis=0) - recs["1_full"][1][:, :3, 3].mean(axis=0))
        print(f"  {key:11s} own average axis is {angle(own, hinge):5.2f} deg from the hinge of recording 1 | "
              f"reference sensor sits {turn:4.2f} deg, {move:4.2f} cm from where it was in recording 1")

    print("\nBuild-up over the outbound movement:")
    print(f"  {'recording':28s} | {'(1) windows summed, own axis':>30s} | {'(2) swing, own axis':>22s} | {'(3) swing, hinge of recording 1':>34s}")
    cases = (("1 pure hinge", "1_full", 0, 1280), ("2 hinge + sliding", "2_transl", 0, 2113),
             ("3 hinge + x by hand", "3_y_plus_x", 100, 1683), ("4 hinge + z by hand", "4_y_plus_z", 600, 1901))
    for name, key, c1, c2 in cases:
        t, T1, T2, _, _ = recs[key]
        b = build_ups(t[c1:c2], T1[c1:c2], T2[c1:c2], hinge)
        mono = b["hinge"][1][-1] / b["hinge"][1].max()
        print(f"  {name:28s} | max {b['summed'].max():5.2f}  end {b['summed'][-1]:5.2f} deg{'':10s} | "
              f"max {b['own'][1].max():5.2f}  end {b['own'][1][-1]:5.2f} deg | max {b['hinge'][1].max():5.2f}  end {b['hinge'][1][-1]:5.2f} deg"
              f"  (end / max = {mono:.2f})")
        if key in ("3_y_plus_x", "4_y_plus_z"):
            col = 2 if key == "3_y_plus_x" else 3
            ax = axes[0, col]
            ax.plot(*b["hinge"], color="C0", lw=1.6, label="(3) swing about the hinge of recording 1")
            ax.plot(*b["own"], color="C1", lw=1.6, label="(2) swing about its own average axis")
            ax.set_title(f"Polhemus {name}", fontsize=10)
            ax.set_xlabel("rotation about the axis (deg)"); ax.set_ylabel("accumulated rotation away from it (deg)")
            ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
            # which way: the swing as seen turning with the primary rotation, in the plane across the hinge
            e1 = np.cross(hinge, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(hinge, e1)
            ax = axes[1, col]
            sc = ax.scatter(b["hinge_swing"] @ e1, b["hinge_swing"] @ e2, c=b["hinge"][0], cmap="viridis", s=4)
            plt.colorbar(sc, ax=ax, label="rotation about the hinge (deg)")
            ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.3)
            ax.set_title(f"Polhemus {name}: which way it tips,\nseen turning with the hinge rotation", fontsize=9)
            ax.set_xlabel("deg"); ax.set_ylabel("deg")

    # simulated tilt maps under the simulated curves
    for col, profile in enumerate(("proportional", "pulsed")):
        d = syn.two_axis_motion(profile=profile, seed=1, **RIG)
        hinge_s = average_axis(ref["t"], ref["T1"], ref["T2"])
        b = build_ups(d["t"], d["T1"], d["T2"], hinge_s)
        e1 = np.cross(hinge_s, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(hinge_s, e1)
        ax = axes[1, col]
        sc = ax.scatter(b["hinge_swing"] @ e1, b["hinge_swing"] @ e2, c=b["hinge"][0], cmap="viridis", s=4)
        plt.colorbar(sc, ax=ax, label="rotation about the hinge (deg)")
        ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.3)
        ax.set_title("simulated: which way it tips,\nseen turning with the hinge rotation", fontsize=9)
        ax.set_xlabel("deg"); ax.set_ylabel("deg")

    fig.tight_layout()
    out = f"{FIG}/13_build_up.png"
    fig.savefig(out, dpi=70)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
