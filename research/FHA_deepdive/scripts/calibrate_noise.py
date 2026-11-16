"""Calibration of the simulator's measurement-error presets (finding F6).

Target: the ORIGINAL miFHA code, run on a simulated pure hinge at the rig's frame rate
and speed, should behave as it does on the real single-axis recordings. Three
statistics are matched:

  net     median and 5th percentile of the net rotation spanned by 10 deg windows
          (sensitive to frame-to-frame jitter, which inflates the path length)
  spread  mean angle between the axes and their average axis (sensitive to the
          amplitude of slow-varying error)
  rough   std of the frame-to-frame change of the tilt series over its std
          (sensitive to the time scale of the slow-varying error)

Run:  python calibrate_noise.py
"""
import numpy as np

from common import load_polhemus, load_vicon, run_original

import synthetic as syn

FMT = "net median %.2f, 5th pct %.2f | spread %.2f deg | roughness %.3f"


def stats(t, T1, T2, c1, c2):
    res = run_original(t, T1, T2, T1[:, :3, 3], T2[:, :3, 3], c1, c2, 10)
    spread = float(res["printed"].split(":")[1].split()[0])
    h = res["hax_g"]
    tilt = np.degrees(np.arctan(np.c_[h @ res["e1"], h @ res["e2"]] / (h @ res["normal"])[:, None]))
    rough = np.mean(np.std(np.diff(tilt, axis=0), axis=0) / np.std(tilt, axis=0))
    return np.array([np.median(res["net"]), np.percentile(res["net"], 5), spread, rough])


def simulated(seeds=16, **kw):
    out = []
    for s in range(seeds):
        d = syn.two_axis_motion(profile="none", seed=s, **kw)
        out.append(stats(d["t"], d["T1"], d["T2"], 0, len(d["t"])))
    return np.mean(out, axis=0)


def main():
    print("REAL recordings, original method, outbound movement:")
    t, T1, T2, _, _ = load_polhemus("1_oneway")
    print("  Polhemus file 1      ", FMT % tuple(stats(t, T1, T2, 100, len(t))))
    t, T1, T2, _, _ = load_vicon("servo_X")
    print("  Vicon servo_X        ", FMT % tuple(stats(t, T1, T2, 200, 800)))

    print("\nSIMULATED pure hinge, original method:")
    print("  white jitter only, Vicon rig (0.02 deg/axis/body):")
    print("                       ", FMT % tuple(simulated(fs=100.0, move_s=5.0, still_s=0.5,
                                                             rot_noise_deg=0.02, pos_noise=0.0067)))
    print("  'polhemus' preset    ", FMT % tuple(simulated(noise="polhemus", fs=120.0, move_s=5.5, still_s=1.0)))
    print("  'vicon' preset       ", FMT % tuple(simulated(noise="vicon", fs=100.0, move_s=5.0, still_s=0.5)))
    print("\npresets:", {k: syn.NOISE_PRESETS[k] for k in ("polhemus", "vicon")})


if __name__ == "__main__":
    main()
