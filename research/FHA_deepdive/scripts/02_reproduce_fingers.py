"""Step 2 — reproduce the reported behaviour with the ORIGINAL, unmodified pipeline.

Calls generate_FHA + plotly_visualize_FHA from the snapshot of the original code (see
common.py) exactly as main.py did, then pulls the intersection points back out of the returned Plotly figure and
draws them in the 2D coordinates of the cross-sectional plane.

Run:  python 02_reproduce_fingers.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG, load_polhemus, load_vicon, run_original


def panel(axrow, res, title, step):
    uv, tt = res["uv"], res["t"]
    ax = axrow[0]
    ax.plot(uv[:, 0], uv[:, 1], "-", color="0.75", lw=0.5, zorder=1)
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=tt, cmap="magma", s=6, zorder=2)
    plt.colorbar(sc, ax=ax, label="time (s)")
    ax.set_aspect("equal", "datalim")
    ax.set_title(f"{title}\ncross-section, original code", fontsize=9)
    ax.set_xlabel("plane axis 1"); ax.set_ylabel("plane axis 2")
    ax.grid(alpha=0.3)

    ax = axrow[1]
    ax.plot(tt, res["net"], lw=0.8, label="net rotation actually spanned")
    ax.axhline(step, color="r", ls="--", lw=0.8, label=f"requested step = {step} deg")
    ax.set_ylim(0, step * 1.3)
    ax.set_title("net rotation of each window", fontsize=9)
    ax.set_xlabel("window start time (s)"); ax.set_ylabel("deg")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axrow[2]
    ax.plot(tt, uv[:, 0], lw=0.8, label="plane axis 1")
    ax.plot(tt, uv[:, 1], lw=0.8, label="plane axis 2")
    ax.set_title("intersection coordinates vs time", fontsize=9)
    ax.set_xlabel("window start time (s)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axrow[3]
    ax.plot(tt, res["dev"], lw=0.8)
    ax.set_title("angle between FHA and AHA", fontsize=9)
    ax.set_xlabel("window start time (s)"); ax.set_ylabel("deg"); ax.grid(alpha=0.3)


CASES = [
    # name, loader, key, cut1, cut2, step
    ("Polhemus 1: pure 90 deg about y (one way)", load_polhemus, "1_oneway", 100, 1248, 10),
    ("Polhemus 3: 90 deg y + x rotation (config.py default cut)", load_polhemus, "3_y_plus_x", 100, 1500, 10),
    ("Polhemus 4: 90 deg y + z rotation (outbound)", load_polhemus, "4_y_plus_z", 600, 1900, 10),
    ("Vicon: servo 90 deg X only (outbound)", load_vicon, "servo_X", 200, 800, 10),
    ("Vicon: servo 90 deg X + 20 deg Y (outbound)", load_vicon, "servo_X_20Y", 200, 760, 10),
    ("Vicon: servo 90 deg X + 20 deg Z (outbound)", load_vicon, "servo_X_20Z", 180, 740, 10),
]


def main():
    fig, axes = plt.subplots(len(CASES), 4, figsize=(22, 4.4 * len(CASES)))
    for row, (title, loader, key, c1, c2, step) in enumerate(CASES):
        t, T1, T2, loc1, loc2 = loader(key)
        res = run_original(t, T1, T2, loc1, loc2, c1, c2, step)
        panel(axes[row], res, title, step)
        net = res["net"]
        print(f"\n{title}")
        print(f"  {res['printed']}")
        print(f"  windows: {len(net)}; net rotation spanned: min {net.min():.2f}, "
              f"5th pct {np.percentile(net,5):.2f}, median {np.median(net):.2f}, max {net.max():.2f} deg "
              f"(requested {step})")
        print(f"  windows below 80% of the requested step: {100*np.mean(net < 0.8*step):.1f}%")
        print(f"  in-plane spread (std): axis1 {res['uv'][:,0].std():.3f}, axis2 {res['uv'][:,1].std():.3f}")
    fig.tight_layout()
    path = f"{FIG}/02_original_cross_sections.png"
    fig.savefig(path, dpi=70)
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
