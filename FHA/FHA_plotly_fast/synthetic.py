"""Synthetic two-axis joint with exact ground truth.

Models the bench rigs used in this project: a primary hinge plus an optional secondary
rotation, measured by two pose sensors with adjustable noise. Because the true helical
axis is known at every frame, any FHA method can be scored against it.

The secondary rotation is defined as a function of the primary angle, beta = f(alpha),
so the *coupling path* is explicit:

- "none"          pure hinge
- "proportional"  beta grows in proportion to alpha (steady rate ratio)
- "late"          nothing until `late_start` of the range, then proportional
- "pulsed"        same total beta, delivered in `n_pulses` separate bursts — what
                  happens when the second rotation is applied by hand or by a hobby servo

Poses follow the convention of the importers: T (4x4) maps sensor coordinates to lab
coordinates.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

PROFILES = ("none", "proportional", "late", "pulsed")

# Measurement-error presets, in degrees and in the length unit of the rig (cm).
# "polhemus" and "vicon" were calibrated so that the ORIGINAL miFHA code, run on a
# simulated pure hinge at the rig's frame rate and speed, reproduces what it gives on
# the real single-axis test recordings (research/FHA_deepdive/FINDINGS.md, finding F6):
#
#                              net rotation of 10 deg windows   mean angle   roughness of
#                              median          5th percentile   FHA-AHA      tilt series
#   Polhemus file 1 (real)     9.88            9.31             0.24 deg     0.163
#   "polhemus" preset          9.93            9.25             0.27 deg     0.166
#   Vicon servo_X (real)       9.44            8.73             1.67 deg     0.241
#   "vicon" preset             9.54            6.11             1.82 deg     0.248
#
# Use the vicon preset with fs=100, move_s=5 (the servo rig) and the polhemus preset
# with fs=120, move_s=5.5.
NOISE_PRESETS = {
    "none": dict(rot_noise_deg=0.0, pos_noise=0.0, rot_drift_deg=0.0, pos_drift=0.0, drift_tau_s=0.3),
    "jitter": dict(rot_noise_deg=0.001, pos_noise=0.0005, rot_drift_deg=0.0, pos_drift=0.0, drift_tau_s=0.3),
    "polhemus": dict(rot_noise_deg=0.002, pos_noise=0.001, rot_drift_deg=0.03, pos_drift=0.006, drift_tau_s=0.4),
    "vicon": dict(rot_noise_deg=0.017, pos_noise=0.0057, rot_drift_deg=0.11, pos_drift=0.011, drift_tau_s=0.06),
}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _rotation_about_line(axis, point, angle):
    """4x4 transforms rotating by `angle` (rad, array) about the line (axis, point)."""
    angle = np.atleast_1d(np.asarray(angle, float))
    R = Rotation.from_rotvec(angle[:, None] * _unit(axis)[None, :]).as_matrix()
    point = np.asarray(point, float)
    T = np.tile(np.eye(4), (len(angle), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = point - R @ point
    return T


def _smoothstep(x):
    """C2 step from 0 to 1 on [0, 1] and its derivative."""
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10 - 15 * x + 6 * x**2), 30 * x**2 * (1 - x) ** 2


def _primary_profile(n_move, total):
    """Constant-speed ramp with smooth (minimum-jerk) starts and stops; per-frame rate."""
    u = np.linspace(0.0, 1.0, n_move)
    ramp = 0.15  # fraction of the movement spent accelerating / decelerating
    # piecewise speed profile: smooth rise, plateau, smooth fall — integrate numerically
    rise, _ = _smoothstep(u / ramp)
    fall, _ = _smoothstep((1 - u) / ramp)
    speed = np.minimum(rise, fall)
    alpha = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]))])
    scale = total / alpha[-1]
    return alpha * scale, speed * scale


def coupling(alpha, total_primary, total_secondary, profile="proportional",
             n_pulses=8, pulse_width=0.35, late_start=0.5):
    """Secondary angle beta(alpha) and its slope d(beta)/d(alpha).

    The slope is the rate ratio: the helical axis tilts by atan(slope) from the hinge.
    """
    alpha = np.asarray(alpha, float)
    x = np.abs(alpha) / abs(total_primary)
    if profile == "none" or total_secondary == 0:
        return np.zeros_like(alpha), np.zeros_like(alpha)
    k = total_secondary / abs(total_primary)
    if profile == "proportional":
        return k * np.abs(alpha), np.full_like(alpha, k)
    if profile == "late":
        # the slope rises smoothly from 0 over a transition `w`, then stays constant,
        # so the second rotation is phased in without a jump in the rate ratio
        w = 0.15
        u = np.clip((x - late_start) / w, 0.0, 1.0)
        s, _ = _smoothstep(u)
        area = w * (2.5 * u**4 - 3 * u**5 + u**6) + np.clip(x - late_start - w, 0.0, None)
        peak = total_secondary / (abs(total_primary) * (1 - late_start - w / 2))
        return peak * abs(total_primary) * area, peak * s
    if profile == "pulsed":
        beta = np.zeros_like(alpha)
        slope = np.zeros_like(alpha)
        cell = 1.0 / n_pulses
        for m in range(n_pulses):
            centre = (m + 0.5) * cell
            lo = centre - 0.5 * pulse_width * cell
            s, ds = _smoothstep((x - lo) / (pulse_width * cell))
            beta += s * total_secondary / n_pulses
            slope += ds / (pulse_width * cell) * (total_secondary / n_pulses) / abs(total_primary)
        return beta, slope
    raise ValueError(f"profile must be one of {PROFILES}")


def two_axis_motion(
    primary_deg=90.0, secondary_deg=20.0, profile="proportional",
    n_pulses=8, pulse_width=0.35, late_start=0.5,
    move_s=8.0, still_s=1.0, fs=120.0, return_trip=False,
    hinge_axis=(0.0, 1.0, 0.0), hinge_point=(0.0, 0.0, 0.0),
    secondary_axis=(0.0, 0.0, 1.0), secondary_point=(20.0, 0.0, 0.0), carried=True,
    sensor1_position=(-6.0, 2.0, 3.0), sensor2_position=(20.0, 0.0, 3.0),
    rot_noise_deg=0.001, pos_noise=0.0005,
    rot_drift_deg=0.0, pos_drift=0.0, drift_tau_s=0.3, noise=None, seed=0,
):
    """Simulate the rig and return poses plus the exact instantaneous helical axis.

    Lengths are in whatever unit you choose for the points (cm by default, matching the
    Polhemus recordings). `carried=True` means the secondary axis rides on the moving
    segment (a block twisted on a hinged beam); False fixes it in the lab.

    Two kinds of measurement error, each applied per sensor and per axis:

    * jitter (`rot_noise_deg`, `pos_noise`): white, independent from frame to frame.
      The defaults reproduce the jitter of the Polhemus test recordings at rest (about
      0.003 deg between consecutive frames of the relative pose).
    * wander (`rot_drift_deg`, `pos_drift`, `drift_tau_s`): slowly varying error with
      the given standard deviation and correlation time. Real trackers have it — field
      distortion for electromagnetic systems, marker reconstruction error for optical
      ones — and it is what limits a helical axis in practice, because averaging over
      frames does not remove it.

    `noise` selects a preset from `NOISE_PRESETS` ("none", "jitter", "polhemus",
    "vicon") and overrides the five individual error arguments.

    Returns a dict with
      t, T1, T2            noisy poses, as the importers would deliver them
      T1_clean, T2_clean   noise-free poses
      alpha, beta          primary / secondary angle per frame (deg)
      slope                d(beta)/d(alpha), the rate ratio
      truth_n, truth_s     true instantaneous axis in the LAB frame: unit direction and
                           the point on it closest to the lab origin (NaN while still)
      truth_pitch          translation along the axis per radian
      moving               boolean mask, True where the joint is actually rotating
    """
    if noise is not None:
        if noise not in NOISE_PRESETS:
            raise ValueError(f"noise must be one of {tuple(NOISE_PRESETS)}")
        preset = NOISE_PRESETS[noise]
        rot_noise_deg, pos_noise = preset["rot_noise_deg"], preset["pos_noise"]
        rot_drift_deg, pos_drift = preset["rot_drift_deg"], preset["pos_drift"]
        drift_tau_s = preset["drift_tau_s"]

    n_move = int(round(move_s * fs))
    n_still = int(round(still_s * fs))
    a_move, rate_move = _primary_profile(n_move, np.radians(primary_deg))
    alpha = np.concatenate([np.zeros(n_still), a_move, np.full(n_still, a_move[-1])])
    rate = np.concatenate([np.zeros(n_still), rate_move, np.zeros(n_still)])
    if return_trip:
        alpha = np.concatenate([alpha, alpha[::-1]])
        rate = np.concatenate([rate, -rate[::-1]])
    n = len(alpha)
    t = np.arange(n) / fs

    beta, slope = coupling(alpha, np.radians(primary_deg), np.radians(secondary_deg),
                           profile, n_pulses, pulse_width, late_start)

    a, b = _unit(hinge_axis), _unit(secondary_axis)
    ch, cb = np.asarray(hinge_point, float), np.asarray(secondary_point, float)
    H = _rotation_about_line(a, ch, alpha)
    W = _rotation_about_line(b, cb, beta)

    T1_0 = np.eye(4); T1_0[:3, 3] = sensor1_position
    T2_0 = np.eye(4); T2_0[:3, 3] = sensor2_position
    # give the sensors arbitrary but fixed orientations so nothing is axis-aligned by luck
    T1_0[:3, :3] = Rotation.from_euler("xyz", [8, -5, 12], degrees=True).as_matrix()
    T2_0[:3, :3] = Rotation.from_euler("xyz", [-4, 10, -7], degrees=True).as_matrix()

    motion = H @ W if carried else W @ H
    T2_clean = motion @ T2_0
    T1_clean = np.tile(T1_0, (n, 1, 1))

    # exact spatial twist of body 2 in the lab frame, per unit of primary rotation
    if carried:
        b_now = np.einsum("nij,j->ni", H[:, :3, :3], b)
        cb_now = np.einsum("nij,j->ni", H[:, :3, :3], cb) + H[:, :3, 3]
        omega = a[None, :] + slope[:, None] * b_now
        vel = np.cross(ch, a)[None, :] + slope[:, None] * np.cross(cb_now, b_now)
    else:
        a_now = np.einsum("nij,j->ni", W[:, :3, :3], a)
        ch_now = np.einsum("nij,j->ni", W[:, :3, :3], ch) + W[:, :3, 3]
        omega = slope[:, None] * b[None, :] + a_now
        vel = slope[:, None] * np.cross(cb, b)[None, :] + np.cross(ch_now, a_now)
    w2 = np.einsum("ni,ni->n", omega, omega)
    truth_n = omega / np.sqrt(w2)[:, None]
    truth_s = np.cross(omega, vel) / w2[:, None]
    truth_pitch = np.einsum("ni,ni->n", omega, vel) / w2
    moving = np.abs(rate) > 1e-9
    # direction follows the sense of rotation, so it flips on the way back
    truth_n = truth_n * np.where(rate < 0, -1.0, 1.0)[:, None]
    truth_n[~moving] = np.nan
    truth_s[~moving] = np.nan
    truth_pitch[~moving] = np.nan

    rng = np.random.default_rng(seed)

    def wander(sigma):
        """Smooth stationary error: white noise through a Gaussian kernel of width
        `drift_tau_s`, scaled to standard deviation `sigma`. Smooth on purpose — it
        changes the pose slowly, without adding frame-to-frame jitter."""
        width = max(drift_tau_s * fs, 1e-6)
        pad = int(np.ceil(4 * width))
        e = rng.standard_normal((n + 2 * pad, 3))
        x = gaussian_filter1d(e, width, axis=0, mode="wrap")[pad:pad + n]
        return sigma * x * np.sqrt(2.0 * np.sqrt(np.pi) * width)   # unit variance after filtering

    def add_noise(T):
        out = T.copy()
        rot = np.zeros((n, 3))
        pos = np.zeros((n, 3))
        if rot_noise_deg > 0:
            rot += np.radians(rot_noise_deg) * rng.standard_normal((n, 3))
        if rot_drift_deg > 0:
            rot += wander(np.radians(rot_drift_deg))
        if pos_noise > 0:
            pos += pos_noise * rng.standard_normal((n, 3))
        if pos_drift > 0:
            pos += wander(pos_drift)
        out[:, :3, :3] = T[:, :3, :3] @ Rotation.from_rotvec(rot).as_matrix()
        out[:, :3, 3] = T[:, :3, 3] + pos
        return out

    return dict(
        t=t, T1=add_noise(T1_clean), T2=add_noise(T2_clean),
        T1_clean=T1_clean, T2_clean=T2_clean,
        alpha=np.degrees(alpha), beta=np.degrees(beta), slope=slope,
        truth_n=truth_n, truth_s=truth_s, truth_pitch=truth_pitch, moving=moving,
        hinge_axis=a, hinge_point=ch,
    )
