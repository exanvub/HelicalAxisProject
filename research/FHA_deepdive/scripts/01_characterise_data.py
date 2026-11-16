"""Step 1 — what is actually in the test recordings?

For every Polhemus and Vicon test file: sampling rate, duration, how much each sensor
moves, the relative rotation profile, and the noise level during the still phases.
Run:  python 01_characterise_data.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (FIG, POLHEMUS_FILES, VICON_FILES, load_polhemus, load_vicon,
                    net_angle_deg, relative_pose, rotvec_deg)


def describe(name, t, T1, T2):
    dt = np.diff(t)
    Trel = relative_pose(T1, T2)
    Rrel = Trel[:, :3, :3]
    # rotation of body 2 relative to its first relative pose, as a rotation vector
    R0 = Rrel[0]
    rv = rotvec_deg(np.einsum("nij,kj->nik", Rrel, R0))  # Rrel @ R0^T
    total = np.linalg.norm(rv, axis=1)
    step = net_angle_deg(Rrel[:-1], Rrel[1:])  # frame-to-frame angle

    # how much does each sensor itself move in the global frame?
    r1 = net_angle_deg(np.repeat(T1[:1, :3, :3], len(T1), 0), T1[:, :3, :3])
    r2 = net_angle_deg(np.repeat(T2[:1, :3, :3], len(T2), 0), T2[:, :3, :3])
    p1 = np.linalg.norm(T1[:, :3, 3] - T1[0, :3, 3], axis=1)
    p2 = np.linalg.norm(T2[:, :3, 3] - T2[0, :3, 3], axis=1)

    print(f"\n=== {name} ===")
    print(f"frames {len(t)}, duration {t[-1]-t[0]:.2f}s, "
          f"dt median {np.median(dt)*1e3:.2f}ms (min {dt.min()*1e3:.2f}, max {dt.max()*1e3:.2f}) "
          f"-> {1/np.median(dt):.1f} Hz")
    print(f"sensor 1 global motion: rot max {r1.max():.2f} deg, pos max {p1.max():.3f}")
    print(f"sensor 2 global motion: rot max {r2.max():.2f} deg, pos max {p2.max():.3f}")
    print(f"relative rotation from first frame: max {total.max():.2f} deg, "
          f"end {total[-1]:.2f} deg")
    k = np.argmax(total)
    print(f"  rotation vector at max (deg): {np.round(rv[k], 2)} at t={t[k]:.2f}s (frame {k})")
    print(f"frame-to-frame angle: median {np.median(step):.4f} deg, "
          f"95th pct {np.percentile(step,95):.4f}, max {step.max():.4f}")
    print(f"path length (sum of frame-to-frame angles): {step.sum():.1f} deg "
          f"vs peak net rotation {total.max():.1f} deg")
    # exact repeats: frames where the relative pose did not change at all
    rep = np.sum(step == 0.0)
    print(f"frames with exactly zero change: {rep} ({100*rep/len(step):.1f}%)")
    return dict(t=t, rv=rv, total=total, step=step, r1=r1, r2=r2, p1=p1, p2=p2, Trel=Trel)


def main():
    out = {}
    for key in POLHEMUS_FILES:
        t, T1, T2, _, _ = load_polhemus(key)
        out[key] = describe(f"Polhemus {key}  ({POLHEMUS_FILES[key]})", t, T1, T2)
    for key in VICON_FILES:
        t, T1, T2, _, _ = load_vicon(key)
        out[key] = describe(f"Vicon {key}  ({VICON_FILES[key]})", t, T1, T2)

    keys = list(out)
    fig, axes = plt.subplots(len(keys), 2, figsize=(15, 2.6 * len(keys)))
    for row, key in enumerate(keys):
        d = out[key]
        ax = axes[row, 0]
        for c, lab in enumerate("xyz"):
            ax.plot(d["t"], d["rv"][:, c], label=f"rotvec {lab}")
        ax.plot(d["t"], d["total"], "k--", lw=0.8, label="|rotvec|")
        ax.set_ylabel("deg")
        ax.set_title(f"{key}: relative rotation vector (body 2 in frame 1, from first frame)",
                     fontsize=9)
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        ax.grid(alpha=0.3)
        ax = axes[row, 1]
        ax.semilogy(d["t"][1:], np.maximum(d["step"], 1e-5), lw=0.6)
        ax.set_title(f"{key}: frame-to-frame rotation angle (log)", fontsize=9)
        ax.set_ylabel("deg / frame")
        ax.grid(alpha=0.3, which="both")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.tight_layout()
    path = f"{FIG}/01_motion_profiles.png"
    fig.savefig(path, dpi=80)
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
