"""Step 3 — are the fingers produced by the method, or are they in the motion?

A helical axis points along the rotation that happens DURING its window. If the motion
is a primary rotation (rate w_par about the mean axis) plus a secondary rotation (rate
w_perp across it), the axis tilts away from the mean axis by atan(w_perp / w_par): the
RATIO OF THE TWO RATES, not the amount of secondary rotation accumulated so far.

This script measures both rates directly from the recording (short central differences,
independent of the FHA code) and compares the tilt they predict with the tilt of the
miFHA axes produced by the original pipeline. It also counts the bursts of secondary
rotation with a peak finder.

Run:  python 03_rate_ratio.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from common import FIG, load_polhemus, load_vicon, run_original

BURST_PROMINENCE = 1.0  # deg/s


def analyse(title, loader, key, c1, c2, step, h):
    t, T1, T2, loc1, loc2 = loader(key)
    res = run_original(t, T1, T2, loc1, loc2, c1, c2, step)

    # Uniform time base. Polhemus timestamps are host arrival times that jitter between
    # 0.02 and 30 ms; the true sampling interval is constant (see finding F1).
    fs = (len(t) - 1) / (t[-1] - t[0])
    tc = t[0] + np.arange(c1, c2) / fs
    Trel = np.linalg.inv(T1[c1:c2]) @ T2[c1:c2]
    Rrel = Trel[:, :3, :3]

    # mean axis and in-plane basis, expressed in frame 1 (sensor 1 is static: use mean R1)
    R1m = Rotation.from_matrix(T1[c1:c2, :3, :3]).mean().as_matrix()
    nbar, e1, e2 = R1m.T @ res["normal"], R1m.T @ res["e1"], R1m.T @ res["e2"]

    # angular velocity of body 2 in frame 1, central difference over +-h frames (deg/s)
    k = np.arange(h, len(tc) - h)
    dR = np.einsum("nij,nkj->nik", Rrel[k + h], Rrel[k - h])
    w = np.degrees(Rotation.from_matrix(dR).as_rotvec()) * fs / (2 * h)
    w_par = w @ nbar
    w_p1, w_p2 = w @ e1, w @ e2
    w_perp = np.hypot(w_p1, w_p2)
    moving = np.abs(w_par) > 0.25 * np.percentile(np.abs(w_par), 90)

    # accumulated primary / secondary rotation
    A = np.cumsum(w_par) / fs
    B1, B2 = np.cumsum(w_p1) / fs, np.cumsum(w_p2) / fs

    # miFHA tilt components from the original pipeline, in frame 1
    hax1 = res["hax_g"] @ R1m  # lab -> frame 1 (row-vector form of R1m^T @ n)
    f_par = hax1 @ nbar
    f1, f2 = hax1 @ e1 / f_par, hax1 @ e2 / f_par
    # each window describes the motion around its middle, not its start
    mid = 0.5 * (tc[res["i0"]] + tc[res["i1"]])

    # what the measured rates predict for the same windows: secant of B over A
    ia = np.clip(res["i0"] - h, 0, len(k) - 1)
    ib = np.clip(res["i1"] - h, 0, len(k) - 1)
    dA = A[ib] - A[ia]
    ok = np.abs(dA) > 1.0
    p1 = np.full(len(ia), np.nan); p2 = np.full(len(ia), np.nan)
    p1[ok] = (B1[ib] - B1[ia])[ok] / dA[ok]
    p2[ok] = (B2[ib] - B2[ia])[ok] / dA[ok]

    good = ok & np.isfinite(f1)
    r1 = np.corrcoef(f1[good], p1[good])[0, 1]
    r2 = np.corrcoef(f2[good], p2[good])[0, 1]
    deg = lambda x: np.degrees(np.arctan(x))
    rms = np.sqrt(np.mean((deg(f1[good]) - deg(p1[good])) ** 2 + (deg(f2[good]) - deg(p2[good])) ** 2))

    ratio = w_perp[moving] / np.abs(w_par[moving])
    q10, q50, q90 = np.percentile(ratio, [10, 50, 90])
    pk, _ = find_peaks(np.where(moving, w_perp, 0.0), prominence=BURST_PROMINENCE)
    heights = w_perp[pk]

    print(f"\n{title}")
    print(f"  primary rate while moving: median {np.median(np.abs(w_par[moving])):.1f} deg/s; "
          f"secondary rate: 10th pct {np.percentile(w_perp[moving],10):.2f}, "
          f"median {np.median(w_perp[moving]):.2f}, 90th pct {np.percentile(w_perp[moving],90):.2f} deg/s")
    print(f"  secondary/primary rate ratio: 10th pct {q10:.2f}, median {q50:.2f}, 90th pct {q90:.2f} "
          f"-> tilt {deg(q10):.1f} to {deg(q90):.1f} deg")
    print(f"  bursts of secondary rate (prominence >= {BURST_PROMINENCE} deg/s): {len(pk)}"
          + (f", peaks {heights.min():.1f}-{heights.max():.1f} deg/s, {np.sum(heights > 6)} above 6 deg/s"
             if len(pk) else ""))
    print(f"  miFHA tilt vs tilt predicted from the measured rates over the same windows: "
          f"r = {r1:.3f} (axis 1), {r2:.3f} (axis 2); RMS difference {rms:.2f} deg")
    return dict(title=title, tk=tc[k], w_par=w_par, w_perp=w_perp, A=A, B1=B1, B2=B2, pk=pk,
                mid=mid, f1=f1, f2=f2, p1=p1, p2=p2)


CASES = [
    ("Polhemus 1: pure hinge (one way)", load_polhemus, "1_oneway", 100, 1248, 10, 12),
    ("Polhemus 3: hinge + x by hand", load_polhemus, "3_y_plus_x", 100, 1500, 10, 12),
    ("Polhemus 4: hinge + z by hand", load_polhemus, "4_y_plus_z", 600, 1900, 10, 12),
    ("Vicon: servo X only", load_vicon, "servo_X", 200, 800, 10, 10),
    ("Vicon: servo X + 20 Y", load_vicon, "servo_X_20Y", 200, 760, 10, 10),
    ("Vicon: servo X + 20 Z", load_vicon, "servo_X_20Z", 180, 740, 10, 10),
]


def main():
    out = [analyse(*c) for c in CASES]
    fig, axes = plt.subplots(len(out), 3, figsize=(20, 4.0 * len(out)))
    deg = lambda x: np.degrees(np.arctan(x))
    for row, d in enumerate(out):
        ax = axes[row, 0]
        ax.plot(d["tk"], np.abs(d["w_par"]), label="primary rate (about mean axis)", lw=1)
        ax.plot(d["tk"], d["w_perp"], label="secondary rate (across it)", lw=1)
        ax.plot(d["tk"][d["pk"]], d["w_perp"][d["pk"]], "kv", ms=4, label=f"bursts ({len(d['pk'])})")
        ax.set_title(f"{d['title']}\nrotation rates measured from the recording", fontsize=9)
        ax.set_xlabel("time (s)"); ax.set_ylabel("deg/s"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[row, 1]
        ax.plot(np.abs(d["A"]), d["B1"], label="secondary, plane axis 1")
        ax.plot(np.abs(d["A"]), d["B2"], label="secondary, plane axis 2")
        ax.set_title("secondary rotation accumulated vs primary rotation\n"
                     "(a smooth curve = steady ratio = no fingers; every kink is a finger)", fontsize=9)
        ax.set_xlabel("primary rotation (deg)"); ax.set_ylabel("secondary rotation (deg)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[row, 2]
        ax.plot(d["mid"], deg(d["f1"]), "C0", lw=1.4, label="miFHA tilt, axis 1 (original code)")
        ax.plot(d["mid"], deg(d["p1"]), "k--", lw=0.8, label="predicted from measured rates, axis 1")
        ax.plot(d["mid"], deg(d["f2"]), "C1", lw=1.4, label="miFHA tilt, axis 2 (original code)")
        ax.plot(d["mid"], deg(d["p2"]), "k:", lw=0.9, label="predicted from measured rates, axis 2")
        ax.set_title("tilt of each miFHA axis from the mean axis", fontsize=9)
        ax.set_xlabel("window mid time (s)"); ax.set_ylabel("deg"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{FIG}/03_rate_ratio.png"
    fig.savefig(path, dpi=70)
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
