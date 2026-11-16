"""Step 5 — score the original and the revised methods against ground truth.

Simulated rig (FHA/FHA_plotly_fast/synthetic.py) with measurement error calibrated
against the real test recordings (`NOISE_PRESETS`, finding F6). Each estimated axis is
compared with two references:

  tracking error   vs the TRUE INSTANTANEOUS axis at the frame the method assigns the
                   axis to. Includes the smoothing of the window and any error in the
                   time the axis is assigned to.
  noise error      vs the same method run on the NOISE-FREE poses, same windows. What
                   measurement error alone costs.

Run:  python 05_benchmark_synthetic.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG, net_angle_deg, orig

import fha_core as fc
import synthetic as syn

STEP = 10
RIGS = {
    "jitter only (ideal tracker)": dict(noise="jitter", fs=120.0, move_s=5.5, still_s=1.0),
    "Polhemus-like": dict(noise="polhemus", fs=120.0, move_s=5.5, still_s=1.0),
    "Vicon-like": dict(noise="vicon", fs=100.0, move_s=5.0, still_s=0.5),
}
PROFILES = ("none", "proportional", "late", "pulsed")
LABELS = ["pure hinge", "steady 2nd rotation", "phased-in 2nd rotation", "pulsed 2nd rotation"]
SEEDS = range(8)


def angle(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.einsum("...i,...i->...", a, b)))


def axis_gap(n, p, q):
    """Perpendicular distance from points q to the lines (p, n)."""
    rel = q - p
    return np.linalg.norm(rel - np.einsum("ki,ki->k", rel, n)[:, None] * n, axis=1)


def to_lab(n1, s1, T1, idx):
    n = np.einsum("kij,kj->ki", T1[idx, :3, :3], n1)
    p = np.einsum("kij,kj->ki", T1[idx, :3, :3], s1) + T1[idx, :3, 3]
    return n, p


def original_method(d):
    """Original miFHA: path-length windows, two-pose FHA, axis tagged at the window start."""
    t, T1, T2 = d["t"], d["T1"], d["T2"]
    out = orig.generate_FHA("incremental_angle", t, 0, len(t), STEP, 1, T1[:, :3, :3], T2[:, :3, :3],
                            T1, T2, T1[:, :3, 3], T2[:, :3, 3])
    ind = out[10]
    i0 = np.array([p[0] for p in ind]); i1 = np.array([p[1] for p in ind])
    n, p = to_lab(np.asarray(out[0]), np.asarray(out[2]), T1, i0)
    Tc = np.linalg.inv(d["T1_clean"]) @ d["T2_clean"]
    _, nc, _, sc = orig.calculate_FHA_vectorized(Tc[i0], Tc[i1])
    nc, pc = to_lab(nc, sc, d["T1_clean"], i0)
    Trel = np.linalg.inv(T1) @ T2
    return dict(n=n, p=p, nc=nc, pc=pc, tag=i0, i0=i0, i1=i1, net=net_angle_deg(Trel[i0, :3, :3], Trel[i1, :3, :3]))


def revised(d, **kw):
    fha = fc.compute_fha(d["t"], d["T1"], d["T2"], "incremental_angle", STEP, **kw)
    n, p = fc.axes_to_lab(fha, d["T1"])
    Tc = fc.relative_poses(d["T1_clean"], d["T2_clean"])
    if kw.get("estimator") == "fit":
        _, nc, _, sc = fc.fha_window_fit(Tc, fha.i0, fha.i1, fha.ic)
    else:
        _, nc, _, sc = fc.fha_endpoints(Tc, fha.i0, fha.i1)
    nc, pc = to_lab(nc, sc, d["T1_clean"], fha.ic)
    return dict(n=n, p=p, nc=nc, pc=pc, tag=fha.ic, i0=fha.i0, i1=fha.i1, net=np.degrees(fha.phi))


METHODS = {
    "original: path length, tagged at start": original_method,
    "net rotation, forward, tagged at middle": lambda d: revised(d, centred=False),
    "net rotation, centred": lambda d: revised(d),
    "net rotation, centred, all-frames fit": lambda d: revised(d, estimator="fit"),
}


def score(method, d):
    r = method(d)
    ok = d["moving"][r["tag"]] & d["moving"][r["i0"]] & d["moving"][np.minimum(r["i1"], len(d["t"]) - 1)]
    tag = r["tag"][ok]
    return dict(
        track=angle(r["n"][ok], d["truth_n"][tag]),
        noise=angle(r["n"][ok], r["nc"][ok]),
        noise_pos=axis_gap(r["n"][ok], r["p"][ok], d["hinge_point"][None, :]) if d["profile"] == "none" else np.array([np.nan]),
        net=r["net"][ok],
    )


def main():
    table = {}
    for rig_name, rig in RIGS.items():
        for profile in PROFILES:
            sims = []
            for s in SEEDS:
                d = syn.two_axis_motion(profile=profile, seed=s, **rig)
                d["profile"] = profile
                sims.append(d)
            for name, method in METHODS.items():
                runs = [score(method, d) for d in sims]
                table[rig_name, profile, name] = dict(
                    track=np.mean([r["track"].mean() for r in runs]),
                    noise=np.mean([r["noise"].mean() for r in runs]),
                    noise_pos=np.mean([np.nanmean(r["noise_pos"]) for r in runs]),
                    net_med=np.median(np.concatenate([r["net"] for r in runs])),
                    short=np.mean(np.concatenate([r["net"] for r in runs]) < 0.8 * STEP),
                )

    for rig_name in RIGS:
        print(f"\n=== {rig_name} — mean over {len(SEEDS)} simulated trials, {STEP} deg windows ===")
        for profile, label in zip(PROFILES, LABELS):
            print(f"\n  {label}")
            print(f"    {'method':42s} {'tracking err':>13s} {'noise err':>10s} {'axis-hinge gap':>15s} {'median net':>11s} {'<8 deg':>7s}")
            for name in METHODS:
                r = table[rig_name, profile, name]
                gap = f"{r['noise_pos']:12.4f} cm" if np.isfinite(r["noise_pos"]) else f"{'-':>15s}"
                print(f"    {name:42s} {r['track']:11.3f}° {r['noise']:9.3f}° {gap} {r['net_med']:9.2f}° {100*r['short']:6.1f}%")

    fig, axes = plt.subplots(2, len(RIGS), figsize=(6.4 * len(RIGS), 9.5), sharey="row")
    for col, rig_name in enumerate(RIGS):
        for row, (key, ylabel) in enumerate((("track", "tracking error vs true instantaneous axis (deg)"),
                                             ("noise", "noise error vs noise-free result (deg)"))):
            ax = axes[row, col]
            width = 0.2
            for m, name in enumerate(METHODS):
                vals = [table[rig_name, p, name][key] for p in PROFILES]
                ax.bar(np.arange(len(PROFILES)) + (m - 1.5) * width, vals, width, label=name)
            ax.set_yscale("log")
            ax.set_xticks(np.arange(len(PROFILES)), [l.replace(" 2nd", "\n2nd") for l in LABELS], fontsize=8)
            ax.set_title(f"{rig_name}", fontsize=10)
            ax.grid(alpha=0.3, axis="y", which="both")
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path = f"{FIG}/05_benchmark_synthetic.png"
    fig.savefig(path, dpi=75)
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
