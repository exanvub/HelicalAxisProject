"""Finite helical axis (FHA) numerics — no plotting, no file I/O.

Conventions
-----------
* A pose ``T`` (4x4) maps the coordinates of a body-fixed frame to a common frame.
* ``relative_poses(T1, T2)`` is the pose of body 2 in the frame of body 1.
* The displacement of body 2 from instant *a* to instant *b*, written in frame 1, is
  ``H = Trel[b] @ inv(Trel[a])``. Every axis returned here is expressed in frame 1
  (the reference body) unless a function says otherwise.
* User-facing angles (``step`` and friends) are in degrees; internals use radians.

What a helical axis reports
---------------------------
The axis of a window points along the rotation that happens *during* that window. For
a primary rotation plus a secondary one, it is tilted from the primary axis by
``atan(secondary rate / primary rate)`` — the ratio of the two rates, not the amount of
secondary rotation accumulated so far. See research/FHA_deepdive/FINDINGS.md.

Methods (``compute_fha``)
-------------------------
``incremental_angle``  miFHA: one window per frame, spanning ``step`` degrees of *net*
                       rotation and centred on the frame it is assigned to.
``step_angle``         fiFHA: consecutive, non-overlapping windows of ``step`` degrees.
``reference_pose``     axis of the displacement from one fixed reference pose to each
                       later pose. Accumulates instead of differentiating.
``incremental_time``   fixed lag of ``step`` frames.
``all_FHA``            consecutive frames.
"""
from dataclasses import dataclass, field
from math import comb

import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import minimum_filter1d
from scipy.spatial.transform import Rotation

METHODS = ("incremental_angle", "step_angle", "reference_pose", "incremental_time", "all_FHA")
ESTIMATORS = ("endpoints", "fit")
CRITERIA = ("net", "path")


# --------------------------------------------------------------------------- poses

def invert(T):
    """Inverse of rigid transforms, exact for any stack shape (..., 4, 4)."""
    T = np.asarray(T, float)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def relative_poses(T1, T2):
    """Pose of body 2 in the frame of body 1, frame by frame."""
    return invert(T1) @ np.asarray(T2, float)


def quaternions(T):
    """Unit quaternions (N, 4) of the rotation part of a pose stack."""
    return Rotation.from_matrix(np.asarray(T, float)[..., :3, :3]).as_quat()


def net_angle(qa, qb):
    """Net rotation angle (rad) between orientations given as unit quaternions.

    Uses the chord between the quaternions, which stays accurate for very small
    angles where ``arccos`` of the dot product loses digits.
    """
    chord = np.minimum(np.linalg.norm(qa - qb, axis=-1), np.linalg.norm(qa + qb, axis=-1))
    return 4.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def even_time_base(t, tolerance=4.0):
    """Sample times of a device that samples evenly but stamps each sample on arrival.

    Returns ``(times, pauses)``. ``pauses`` lists ``(frame, seconds)`` for every place
    where time was really lost just before that frame: the recording was paused or
    frames were dropped.

    Trackers such as the Polhemus stamp each sample when it reaches the host. If a
    straight line through the stamps fits within ``tolerance`` sample periods, that line
    is returned (in the bench recordings the intervals scatter 0.02-30 ms around a true
    8.33 ms). Otherwise the stamps are taken apart. The host often receives samples in
    bursts — nothing for up to 90 ms, then ten at once. That can delay a stamp but never
    advance it, so the true sample times run along the LOWER edge of the stamps. A
    lasting rise of that edge is time that was lost, and is kept as a pause. A lasting
    drop is the host clock being set back; the device kept sampling, so it is ignored.

    Stamps that do not look like this at all (far more breaks than a recording could
    have) are returned unchanged.

    This matters once anything is filtered in time: taken at face value, burst stamps
    say that ten samples were taken within a tenth of a millisecond, and a filter
    would flatten those ten and put a jump before them.
    """
    t = np.asarray(t, float)
    n = len(t)
    if n < 3:
        return t, []
    k = np.arange(n)
    slope, intercept = np.polyfit(k, t, 1)
    if slope > 0 and np.max(np.abs(t - (slope * k + intercept))) <= tolerance * slope:
        return intercept + slope * k, []
    lag = max(1, min(50, (n - 1) // 4))                 # long enough to see through the bursts
    period = float(np.median(t[lag:] - t[:-lag])) / lag
    if not period > 0:
        return t, []
    late = t - period * k                               # delay of each stamp, plus time lost so far
    w = max(3, min(30, n // 4))
    ahead = minimum_filter1d(late, w, origin=-(w // 2), mode="nearest")       # lower edge of frames i .. i+w-1
    behind = minimum_filter1d(late, w, origin=(w - 1) // 2, mode="nearest")   # of frames i-w+1 .. i
    jump = np.r_[0.0, ahead[1:] - behind[:-1]]
    marked = np.flatnonzero(np.abs(jump) > 0.5 * period)
    runs = np.split(marked, np.flatnonzero(np.diff(marked) > 1) + 1) if marked.size else []
    if len(runs) > max(3, n // 200):
        return t, []
    gap = np.r_[0.0, np.diff(t)]
    shift = np.zeros(n)                                 # lasting change of the lower edge, at the frame it happens
    pauses = []
    for run in runs:
        rise = jump[run].sum() > 0
        i = int(run[np.argmax(gap[run])] if rise else run[np.argmin(gap[run])])
        if rise and n - i < max(2, w // 2):
            continue                                    # too close to the end to tell a pause from a late stamp
        shift[i] = jump[i]
        if rise:
            pauses.append((i, float(jump[i])))
    early = late - np.cumsum(shift)
    base = float(np.median(minimum_filter1d(early, w, mode="nearest")))
    if np.mean(early - base < -period) > 0.01:
        return t, []                                    # stamps ahead of their sample time: not arrival times
    return base + period * k + np.cumsum(np.where(shift > 0, shift, 0.0)), pauses


def uniform_time(t, tolerance=4.0):
    """The evenly spaced time base of ``even_time_base``, without the list of pauses."""
    return even_time_base(t, tolerance)[0]


# ----------------------------------------------------------------------- smoothing

def _passes_everything(t, cutoff_hz):
    """True when a low-pass at ``cutoff_hz`` leaves samples taken at times ``t`` as they
    are: no cut-off, fewer than four samples, or a cut-off at or above half the
    sampling rate."""
    if cutoff_hz is None or len(t) < 4:
        return True
    dt = np.median(np.diff(t))
    return not cutoff_hz > 0 or bool(dt > 0 and cutoff_hz >= 0.5 / dt)


def smooth(t, Y, cutoff_hz):
    """Low-pass the columns of ``Y`` (N, k), sampled at times ``t``, with no time shift.

    ``cutoff_hz`` is the frequency at which the amplitude is down to 70.7% (-3 dB).
    The response is practically that of the zero-lag fourth-order Butterworth filter
    usual in movement analysis (within 0.1% of it away from the ends of a recording),
    so a result can be reported as "low-pass filtered at X Hz" in the usual sense.

    It is implemented as penalised least squares on second differences — the discrete
    form of a cubic smoothing spline — solved as one banded system. Compared with
    running a Butterworth filter forwards and backwards that buys three things this
    project needs: no start-up transients at the two ends, any sampling (uneven steps
    are fine), and any length down to four samples, as in dynamic CT. A movement at
    constant velocity is left exactly as it is.

    ``None``, or a cut-off at or above half the sampling rate, returns ``Y`` unchanged.
    """
    t, Y = np.asarray(t, float), np.asarray(Y, float)
    n = len(t)
    if _passes_everything(t, cutoff_hz):
        return Y.copy()
    h = np.diff(t)
    if not np.all(h > 0):
        raise ValueError("Sample times must increase from frame to frame to smooth in time "
                         f"({int(np.sum(~(h > 0)))} do not). See even_time_base.")
    if not np.isfinite(Y).all():
        raise ValueError("Missing values cannot be smoothed; fill or remove those frames first.")
    dt = np.median(h)
    lam = (np.sqrt(2.0) - 1.0) / (16.0 * np.sin(np.pi * cutoff_hz * dt) ** 4)
    # second derivative at each interior sample by divided differences, in units of the
    # median step; the penalty matrix is assembled directly as its five diagonals
    h0, h1 = h[:-1], h[1:]
    a = 2.0 / (h0 * (h0 + h1)) * dt**2
    b = -2.0 / (h0 * h1) * dt**2
    c = 2.0 / (h1 * (h0 + h1)) * dt**2
    w = (h0 + h1) / (2.0 * dt)                          # time each interior sample stands for
    main, off1 = np.zeros(n), np.zeros(n - 1)
    main[:-2] += w * a * a
    main[1:-1] += w * b * b
    main[2:] += w * c * c
    off1[:-1] += w * a * b
    off1[1:] += w * b * c
    off2 = w * a * c
    ab = np.zeros((5, n))
    ab[0, 2:], ab[1, 1:], ab[2] = lam * off2, lam * off1, 1.0 + lam * main
    ab[3, :-1], ab[4, :-2] = lam * off1, lam * off2
    return solve_banded((2, 2), ab, Y)


def _continuous_log(H):
    """Screw coordinates of a stream of displacements ``H`` (N, 4, 4): rotation vector and
    translation part of the SE(3) logarithm, with the rotation vector kept continuous
    along the stream (through 180 deg, up to just under a full turn)."""
    q = Rotation.from_matrix(H[:, :3, :3]).as_quat()
    flip = np.where(np.einsum("ij,ij->i", q[1:], q[:-1]) < 0, -1.0, 1.0)
    q[1:] *= np.cumprod(flip)[:, None]                  # q and -q are the same rotation
    if q[np.argmin(np.linalg.norm(q[:, :3], axis=1)), 3] < 0:
        q = -q                                          # the smallest rotation gets the short branch
    vn = np.linalg.norm(q[:, :3], axis=1)
    theta = 2.0 * np.arctan2(vn, q[:, 3])
    rho = theta[:, None] * q[:, :3] / np.where(vn > 1e-12, vn, 1.0)[:, None]
    small = theta < 1e-2
    ts = np.where(small, 1.0, theta)
    c2 = np.where(small, 1 / 12 + theta**2 / 720, (1.0 - 0.5 * ts / np.tan(0.5 * ts)) / ts**2)
    p = H[:, :3, 3]
    rp = np.cross(rho, p)
    return rho, p - 0.5 * rp + c2[:, None] * np.cross(rho, rp), theta


def _exp_se3(rho, u):
    """Inverse of ``_continuous_log``: displacements (N, 4, 4) from screw coordinates."""
    theta = np.linalg.norm(rho, axis=1)
    small = theta < 1e-2
    ts = np.where(small, 1.0, theta)
    A = np.where(small, 0.5 - theta**2 / 24, (1.0 - np.cos(ts)) / ts**2)
    B = np.where(small, 1 / 6 - theta**2 / 120, (ts - np.sin(ts)) / ts**3)
    ru = np.cross(rho, u)
    H = np.tile(np.eye(4), (len(rho), 1, 1))
    H[:, :3, :3] = Rotation.from_rotvec(rho).as_matrix()
    H[:, :3, 3] = u + A[:, None] * ru + B[:, None] * np.cross(rho, ru)
    return H


def smooth_poses(t, T, cutoff_hz, screw=True):
    """Low-pass a stream of poses (N, 4, 4). See ``smooth`` for what the cut-off means.

    ``screw=True`` (default) smooths the SCREW COORDINATES of each pose relative to the
    pose in the middle of the movement: rotation vector and translation part of the
    logarithm of the displacement. Motion about one fixed axis is a straight line in
    those coordinates, so smoothing moves poses along that axis's motion and never off
    it: a hinge stays exactly a hinge, at any cut-off and any speed profile.

    ``screw=False`` smooths positions and quaternion components separately, which is
    what filtering marker or sensor signals amounts to. For a body turning about an axis
    this pulls every point toward the axis, by an amount that changes with speed, so the
    filter itself moves the helical axis — by millimetres at cut-offs around 1 Hz for a
    sensor 20 cm from the axis (finding F17). It is kept for comparison.

    Smoothing does not know noise from movement. It removes whatever changes faster
    than the cut-off: frame-to-frame jitter first, then slower tracker error, then the
    quick parts of the movement itself. ``removed_by_smoothing`` and ``jitter`` put a
    number on what was taken out and on how much of that can be jitter.
    """
    T = np.asarray(T, float)
    if _passes_everything(np.asarray(t, float), cutoff_hz):
        return T.copy()
    if not screw:
        q = Rotation.from_matrix(T[:, :3, :3]).as_quat()
        flip = np.where(np.einsum("ij,ij->i", q[1:], q[:-1]) < 0, -1.0, 1.0)
        q[1:] *= np.cumprod(flip)[:, None]
        Y = smooth(t, np.c_[q, T[:, :3, 3]], cutoff_hz)
        out = T.copy()
        out[:, :3, :3] = Rotation.from_quat(Y[:, :4] / np.linalg.norm(Y[:, :4], axis=1, keepdims=True)).as_matrix()
        out[:, :3, 3] = Y[:, 4:]
        return out
    q = quaternions(T)
    turned = net_angle(q, q[0][None, :])
    ref = int(np.argmin(np.abs(turned - 0.5 * turned.max())))      # pose midway through the rotation
    rho, u, theta = _continuous_log(T @ invert(T[ref])[None])
    if theta.max() > 2.0 * np.pi - 0.2:
        raise ValueError("The movement covers almost two full turns; smooth it in shorter pieces.")
    Y = smooth(t, np.c_[rho, u], cutoff_hz)
    return _exp_se3(Y[:, :3], Y[:, 3:]) @ T[ref][None]


def smooth_motion(t, T1, T2, cutoff_hz, screw=True):
    """Smooth the motion of body 2 relative to body 1, and body 1 itself.

    The relative pose is what a helical axis is computed from, so that is what is
    smoothed (in screw coordinates, see ``smooth_poses``); body 2 is then put back on
    the smoothed body 1. Returns ``T1_smooth, T2_smooth``.
    """
    if cutoff_hz is None:
        return np.asarray(T1, float).copy(), np.asarray(T2, float).copy()
    T1s = smooth_poses(t, T1, cutoff_hz, screw)
    return T1s, T1s @ smooth_poses(t, relative_poses(T1, T2), cutoff_hz, screw)


def removed_by_smoothing(T_raw, T_smooth):
    """What smoothing changed, frame by frame: the angle (deg) between the raw and the
    smoothed orientation, and the distance between the raw and the smoothed position."""
    angle = np.degrees(net_angle(quaternions(T_raw), quaternions(T_smooth)))
    return angle, np.linalg.norm(np.asarray(T_raw)[:, :3, 3] - np.asarray(T_smooth)[:, :3, 3], axis=1)


def jitter(T):
    """Frame-to-frame jitter of a pose stream: ``(angle in deg, distance)``, each the RMS
    size of the random error of a single frame, or NaN where the recording cannot say.

    Taken from differences of the third, fourth and fifth order. Each cancels smooth
    movement and leaves white noise scaled by a known factor; a median-based spread
    keeps brief events from inflating it. The smallest of the three is returned. On a
    tracker sampling at 100 Hz or more the three agree within a few percent. If they
    disagree by more than half, the differences still hold smooth movement — or there is
    no jitter at all — and the result is NaN, as it is for fewer than 20 frames.

    "Frame to frame" is meant literally: anything that changes independently from one
    frame to the next counts, at the frame rate of the recording. At 100 Hz that is
    electronic noise. At a few frames per second, as in dynamic CT, it is the whole
    registration error of each frame, and also any movement too quick for that frame
    rate to follow. With 20 to 30 frames the number is good to about a third.

    This is the part of the measurement error that low-pass filtering removes before it
    touches the movement. Error that varies slowly — field distortion, marker
    reconstruction, soft tissue — is not in this number, usually dwarfs it on a fast
    tracker, and cannot be told from movement by a filter.
    """
    T = np.asarray(T, float)
    if len(T) < 20:
        return float("nan"), float("nan")
    R = T[:, :3, :3]
    step = Rotation.from_matrix(np.einsum("nji,njk->nik", R[:-1], R[1:])).as_rotvec()    # R_t^T R_t+1

    def level(first_differences):
        found = []
        for order in (3, 4, 5):
            d = np.diff(first_differences, n=order - 1, axis=0)
            spread = 1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0) / np.sqrt(comb(2 * order, order))
            found.append(float(np.sqrt(np.sum(spread**2))))
        return min(found) if max(found) <= 1.5 * min(found) else float("nan")

    return float(np.degrees(level(step))), level(np.diff(T[:, :3, 3], axis=0))


# ----------------------------------------------------------- screw of a displacement

def screw_from_displacement(H, min_angle=1e-7):
    """Helical-axis parameters of rigid displacements ``H`` (..., 4, 4).

    Returns ``phi, n, d, s``:
      phi  rotation angle in [0, pi] (rad)
      n    unit direction; rotation by +phi about n
      d    translation along n
      s    the point of the axis closest to the origin (so ``s . n == 0``)

    Equivalent to Spoor & Veldpaus (1980) eqs. 36-38 but taken through the rotation
    vector, which is well conditioned over the whole range. The original code used
    ``arcsin(sin phi)`` below sin(phi) = 0.707, which returns ``180 deg - phi`` for
    rotations beyond 135 deg. Displacements with ``phi < min_angle`` have no defined
    axis and come back as NaN.
    """
    H = np.asarray(H, float)
    lead = H.shape[:-2]
    R = H[..., :3, :3].reshape(-1, 3, 3)
    v = H[..., :3, 3].reshape(-1, 3)
    if len(R) == 0:                      # nothing to do; older SciPy rejects an empty batch
        return np.zeros(lead), np.zeros(lead + (3,)), np.zeros(lead), np.zeros(lead + (3,))
    rho = Rotation.from_matrix(R).as_rotvec()
    phi = np.linalg.norm(rho, axis=1)
    ok = phi > min_angle
    safe = np.where(ok, phi, 1.0)
    n = rho / safe[:, None]
    d = np.einsum("ij,ij->i", n, v)
    v_perp = v - d[:, None] * n
    s = 0.5 * (v_perp + np.cross(n, v) / np.tan(safe / 2.0)[:, None])
    n[~ok] = np.nan
    s[~ok] = np.nan
    d = np.where(ok, d, np.nan)
    return phi.reshape(lead), n.reshape(lead + (3,)), d.reshape(lead), s.reshape(lead + (3,))


def se3_log(H):
    """Logarithm of rigid displacements: rotation vector ``rho`` and translation part ``u``.

    For a screw motion about a fixed axis through ``c`` with direction ``n`` and pitch
    ``h`` (translation per radian), ``log(H) = theta * (n, c x n + h n)`` exactly, for
    any angle theta. That linearity is what ``fha_window_fit`` relies on.
    """
    H = np.asarray(H, float)
    lead = H.shape[:-2]
    R = H[..., :3, :3].reshape(-1, 3, 3)
    v = H[..., :3, 3].reshape(-1, 3)
    if len(R) == 0:
        return np.zeros(lead + (3,)), np.zeros(lead + (3,))
    rho = Rotation.from_matrix(R).as_rotvec()
    phi2 = np.einsum("ij,ij->i", rho, rho)
    phi = np.sqrt(phi2)
    small = phi < 1e-2
    ps = np.where(small, 1.0, phi)
    c2 = np.where(small, 1 / 12 + phi2 / 720 + phi2**2 / 30240,
                  (1.0 - 0.5 * ps / np.tan(0.5 * ps)) / ps**2)
    rv = np.cross(rho, v)
    u = v - 0.5 * rv + c2[:, None] * np.cross(rho, rv)
    return rho.reshape(lead + (3,)), u.reshape(lead + (3,))


# ------------------------------------------------------------------------- windows

def first_reach(q, threshold, direction=1, block=64):
    """For every frame, the first frame in ``direction`` whose NET rotation from it
    reaches ``threshold`` (rad). Returns -1 where the recording ends first.

    Net rotation is not monotonic along a recording (noise, pauses, reversals), so
    the search walks outward from each frame; it does so for all frames at once in
    blocks, which keeps it fast for recordings of tens of thousands of frames.
    """
    q = np.asarray(q, float)
    n = len(q)
    out = np.full(n, -1, dtype=int)
    cos_half = np.cos(min(threshold, 2 * np.pi) / 2.0)  # |qa.qb| = cos(angle/2)
    active = np.arange(n)
    offset = 1
    while active.size and offset < n:
        steps = offset + np.arange(block)
        idx = active[:, None] + direction * steps[None, :]
        valid = (idx >= 0) & (idx < n)
        dots = np.abs(np.einsum("ak,abk->ab", q[active], q[np.clip(idx, 0, n - 1)]))
        hit = valid & (dots <= cos_half)
        found = hit.any(axis=1)
        out[active[found]] = idx[found, hit[found].argmax(axis=1)]
        active = active[~found & valid[:, -1]]  # drop finished and those out of range
        offset += block
    return out


def path_length(q):
    """Cumulative sum of frame-to-frame rotation angles (rad); ``c[k]`` is the path
    travelled up to frame k. Noise and reversals add to it without adding net rotation."""
    return np.concatenate([[0.0], np.cumsum(net_angle(q[:-1], q[1:]))])


def _first_reach_path(c, threshold):
    """Path-length counterpart of ``first_reach`` (forward only)."""
    j = np.searchsorted(c, c + threshold, side="right")
    return np.where(j < len(c), j, -1)


def windows_centred(q, step, max_half=0.75):
    """miFHA windows: around every frame k, the two frames the FHA will be computed
    between. Returns ``i0, i1, ic`` (start, end, centre).

    What makes an FHA well or ill conditioned is the NET rotation between the two poses
    it is computed from, so that is the quantity the threshold is applied to:

    1. Start from the nearest frame before k and the nearest after k that are each
       ``step/2`` of net rotation away from the pose at k.
    2. On a single axis those two ends are then ``step`` apart. When the direction of
       rotation changes inside the window they are closer, so the window is widened,
       one frame at a time on whichever side reaches less far from k, until the two
       ends are ``step`` apart.

    Every window returned therefore compares two poses at least ``step`` apart in net
    rotation, and is centred, in rotation, on the frame it belongs to.

    A frame gets no window when that cannot be done: at rest; within half a window of
    the start or end of the movement; and where the rotation reverses, since widening
    then makes one half reach ``max_half * step`` on its own (the two halves fold back
    on each other and no single axis describes them). With the default of 0.75 the
    direction of rotation may differ by up to about 95 deg between the two halves.

    The same limit drops a window one of whose halves spans that much from the start
    because the pose JUMPS within a single frame — a marker glitch, or frames lost while
    the joint kept moving. A half can only end on a frame, though, so with few frames per
    degree it overshoots ``step/2`` without any jump: at 4 deg per frame, half of a
    10 deg window is two frames, 8 deg. It counts as a jump only when its last frame
    turns more than three times what nine frames in ten turn. Coarse recordings (dynamic
    CT, or a small ``step`` on a fast movement) therefore keep their windows, which are
    then as narrow as the frames allow and wider than asked; their halves may grow by
    half again before the window counts as folded back.

    Summing frame-to-frame angles instead, as the original code did, counts noise and
    reversals as rotation and returns windows whose ends can be almost the same pose.
    """
    n = len(q)
    back = first_reach(q, step / 2.0, -1)
    fwd = first_reach(q, step / 2.0, +1)
    ic = np.flatnonzero((back >= 0) & (fwd >= 0))
    i0, i1 = back[ic].copy(), fwd[ic].copy()
    span0 = i1 - i0
    h0, h1 = net_angle(q[ic], q[i0]), net_angle(q[ic], q[i1])
    turn = net_angle(q[:-1], q[1:])                               # rotation from each frame to the next
    usual = np.percentile(turn, 90) if turn.size else 0.0
    long = max_half * step
    dead = ((h0 >= long) & (turn[i0] > 3.0 * usual)) | ((h1 >= long) & (turn[i1 - 1] > 3.0 * usual))
    widest = np.maximum(h0, h1)
    limit = np.where(widest < long, long, (max_half / 0.5) * widest)
    active = np.flatnonzero(~dead & (net_angle(q[i0], q[i1]) < step))
    while active.size:
        c, a, b = ic[active], i0[active], i1[active]
        h0, h1 = net_angle(q[c], q[a]), net_angle(q[c], q[b])
        back_side = h0 <= h1
        stop = ((np.maximum(h0, h1) >= limit[active])            # folded back: a reversal
                | ((b - a) > 3 * span0[active] + 2)              # stalled, e.g. the joint came to rest
                | (back_side & (a == 0)) | (~back_side & (b == n - 1)))
        dead[active[stop]] = True
        active, back_side = active[~stop], back_side[~stop]
        i0[active[back_side]] -= 1
        i1[active[~back_side]] += 1
        active = active[net_angle(q[i0[active]], q[i1[active]]) < step]
    return i0[~dead], i1[~dead], ic[~dead]


def windows_forward(q, step, criterion="net"):
    """One window per start frame, closing when ``step`` is reached, by net rotation
    (``criterion="net"``) or by summed frame-to-frame angle (``"path"``, the behaviour
    of the original code). The centre is the frame midway in net rotation."""
    nxt = first_reach(q, step, +1) if criterion == "net" else _first_reach_path(path_length(q), step)
    i0 = np.flatnonzero(nxt >= 0)
    i1 = nxt[i0]
    return i0, i1, _angular_midpoint(q, i0, i1)


def windows_chained(q, step, criterion="net"):
    """fiFHA windows: non-overlapping, each starting where the previous one ended."""
    nxt = first_reach(q, step, +1) if criterion == "net" else _first_reach_path(path_length(q), step)
    chain = [0]
    while nxt[chain[-1]] >= 0:
        chain.append(int(nxt[chain[-1]]))
    chain = np.asarray(chain, dtype=int)
    i0, i1 = chain[:-1], chain[1:]
    return i0, i1, _angular_midpoint(q, i0, i1)


def _angular_midpoint(q, i0, i1):
    """Frame inside each window that is equally far, in net rotation, from both ends."""
    ic = (i0 + i1) // 2
    for w in range(len(i0)):
        if i1[w] - i0[w] > 2:
            m = np.arange(i0[w] + 1, i1[w])
            diff = net_angle(q[m], q[i0[w]][None, :]) - net_angle(q[m], q[i1[w]][None, :])
            ic[w] = m[np.argmin(np.abs(diff))]
    return ic


# ---------------------------------------------------------------------- estimators

def fha_endpoints(Trel, i0, i1):
    """Classical FHA of each window: the screw between its two end poses."""
    return screw_from_displacement(Trel[i1] @ invert(Trel[i0]))


def fha_window_fit(Trel, i0, i1, ic, max_frames=64):
    """Least-squares helical axis of each window, using the frames inside it.

    The classical FHA uses two poses and discards everything between them, so all the
    measurement noise of those two poses lands in the axis. Here every pose in the
    window is written as a displacement from the centre pose and taken through the
    SE(3) logarithm; for motion about a fixed screw axis those logs lie on a straight
    line through the origin of R^6, whose direction is the axis. The line is fitted by
    least squares (principal direction of the rotation part, then regression of the
    translation part), with an intercept so that noise in the centre pose drops out.

    Up to ``max_frames`` evenly spaced frames per window are used. Returns the same
    ``phi, n, d, s`` as ``fha_endpoints``; ``phi`` is the net rotation between the ends.
    """
    i0, i1, ic = (np.asarray(a, dtype=int) for a in (i0, i1, ic))
    if len(i0) == 0:
        return np.zeros(0), np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
    chunk = 1024                                          # bounds memory on long recordings
    if len(i0) > chunk:
        parts = [fha_window_fit(Trel, i0[a:a + chunk], i1[a:a + chunk], ic[a:a + chunk], max_frames)
                 for a in range(0, len(i0), chunk)]
        return tuple(np.concatenate(p) for p in zip(*parts))
    frac = np.linspace(0.0, 1.0, max_frames)
    idx = np.rint(i0[:, None] + frac[None, :] * (i1 - i0)[:, None]).astype(int)   # (W, L)
    D = Trel[idx] @ invert(Trel[ic])[:, None]                                       # (W, L, 4, 4)
    rho, u = se3_log(D)
    rho = rho - rho.mean(axis=1, keepdims=True)
    u = u - u.mean(axis=1, keepdims=True)

    cov = np.einsum("wli,wlj->wij", rho, rho)
    _, vecs = np.linalg.eigh(cov)
    n = vecs[:, :, -1]                                    # principal direction of rotation
    theta = np.einsum("wli,wi->wl", rho, n)               # progress of each frame about n
    forward = np.einsum("wl,l->w", theta, frac - 0.5) >= 0
    n = np.where(forward[:, None], n, -n)                 # +n = sense in which time runs
    theta = np.where(forward[:, None], theta, -theta)

    denom = np.einsum("wl,wl->w", theta, theta)
    ok = denom > 1e-18
    w = np.einsum("wl,wli->wi", theta, u) / np.where(ok, denom, 1.0)[:, None]
    pitch = np.einsum("wi,wi->w", w, n)                   # translation per radian
    s = np.cross(n, w)                                    # axis point closest to origin
    phi = net_angle(*(quaternions(Trel[i]) for i in (i0, i1)))
    n[~ok] = np.nan
    s[~ok] = np.nan
    return phi, n, np.where(ok, pitch * phi, np.nan), s


# ------------------------------------------------------------------------ pipeline

@dataclass
class FHASet:
    """A set of helical axes, expressed in the frame of the reference body (frame 1)."""
    n: np.ndarray            # (M, 3) unit directions
    s: np.ndarray            # (M, 3) point of each axis closest to the frame-1 origin
    phi: np.ndarray          # (M,) rotation of each window (rad)
    d: np.ndarray            # (M,) translation along the axis over the window
    i0: np.ndarray           # (M,) window start frame
    i1: np.ndarray           # (M,) window end frame
    ic: np.ndarray           # (M,) frame the axis is assigned to
    t: np.ndarray            # (M,) time of that frame
    weight: np.ndarray       # (M,) weights for averaging, summing to 1
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.phi)

    def subset(self, mask):
        keep = lambda a: a[mask]
        w = self.weight[mask]
        return FHASet(keep(self.n), keep(self.s), keep(self.phi), keep(self.d), keep(self.i0),
                      keep(self.i1), keep(self.ic), keep(self.t), w / w.sum() if w.sum() > 0 else w,
                      dict(self.meta))


def compute_fha(t, T1, T2, method="incremental_angle", step=10.0, estimator="endpoints",
                criterion="net", centred=True, ref_index=0, ref_frames=10, max_half=0.75):
    """Helical axes of body 2 relative to body 1.

    Parameters
    ----------
    t, T1, T2 : time (N,) and poses (N, 4, 4) of the two bodies in a common frame.
    method    : one of ``METHODS`` (see module docstring).
    step      : window size — degrees for the angle-based methods and the minimum
                rotation for ``reference_pose``; frames for ``incremental_time``.
    estimator : ``"endpoints"`` (classical two-pose FHA) or ``"fit"`` (least squares
                over the frames inside the window; ignored for ``reference_pose`` and
                for windows of fewer than 4 frames).
    criterion : ``"net"`` closes a window on the net rotation between its ends;
                ``"path"`` on the summed frame-to-frame angles, as the original code did.
    centred   : for ``incremental_angle`` with ``criterion="net"``, build each window
                symmetrically around the frame it describes. ``False`` gives one
                forward window per start frame.
    ref_index, ref_frames : ``reference_pose`` only — the reference is the average of
                ``ref_frames`` poses starting at ``ref_index``.
    max_half  : a centred window is dropped if either half would have to span
                ``max_half * step`` on its own for the ends to be ``step`` apart
                (the rotation reversed inside it).

    Returns an ``FHASet``. Axes are in frame 1; use ``axes_to_lab`` to draw them in
    the lab frame.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    if step <= 0:
        raise ValueError("step must be positive")

    t = np.asarray(t, float)
    Trel = relative_poses(T1, T2)
    q = quaternions(Trel)
    n_frames = len(t)
    rad = np.radians(step)
    empty = np.zeros(0, dtype=int)

    if method == "all_FHA":
        i0 = np.arange(n_frames - 1); i1 = i0 + 1; ic = i0
    elif method == "incremental_time":
        lag = max(int(round(step)), 1)
        i0 = np.arange(max(n_frames - lag, 0)); i1 = i0 + lag; ic = i0 + lag // 2
    elif method == "step_angle":
        i0, i1, ic = windows_chained(q, rad, criterion) if n_frames > 1 else (empty, empty, empty)
    elif method == "incremental_angle":
        if criterion == "net" and centred:
            i0, i1, ic = windows_centred(q, rad, max_half)
        else:
            i0, i1, ic = windows_forward(q, rad, criterion)
    else:  # reference_pose
        ref_index = int(np.clip(ref_index, 0, n_frames - 1))
        block = Trel[ref_index:ref_index + max(int(ref_frames), 1)]
        Tref = np.eye(4)
        Tref[:3, :3] = Rotation.from_matrix(block[:, :3, :3]).mean().as_matrix()
        Tref[:3, 3] = block[:, :3, 3].mean(axis=0)
        qref = Rotation.from_matrix(Tref[:3, :3]).as_quat()
        ang = net_angle(q, qref[None, :])
        i1 = np.flatnonzero((ang >= rad) & (np.arange(n_frames) > ref_index))
        i0 = np.full(len(i1), ref_index); ic = i1
        phi, n, d, s = screw_from_displacement(Trel[i1] @ invert(Tref)[None])

    if method != "reference_pose":
        if len(i0) == 0:
            phi = d = np.zeros(0); n = s = np.zeros((0, 3))
        else:
            phi, n, d, s = fha_endpoints(Trel, i0, i1)
            if estimator == "fit":
                wide = (i1 - i0) >= 3
                if wide.any():
                    pf, nf, df, sf = fha_window_fit(Trel, i0[wide], i1[wide], ic[wide])
                    phi[wide], n[wide], d[wide], s[wide] = pf, nf, df, sf

    good = np.isfinite(n).all(axis=1) & np.isfinite(s).all(axis=1)
    i0, i1, ic, phi, n, d, s = (a[good] for a in (i0, i1, ic, phi, n, d, s))

    # Averaging weights. Overlapping windows give one axis per frame, so a slow or
    # paused stretch would otherwise dominate any average. Dividing by the window
    # length in frames makes every degree of motion count equally. Windows of unequal
    # rotation (reference pose, fixed time lag, consecutive frames) get sin^2(phi/2)
    # (Woltring 1990; Ehrig & Heller 2019), which discounts ill-conditioned small angles.
    if method == "incremental_angle":
        weight = 1.0 / np.maximum(i1 - i0, 1)
    elif method == "step_angle":
        weight = np.ones(len(phi))                      # equal angle per window already
    else:
        weight = np.sin(phi / 2.0) ** 2
    weight = weight / weight.sum() if weight.sum() > 0 else weight

    meta = dict(method=method, step=step, estimator=estimator, criterion=criterion,
                centred=bool(centred and criterion == "net" and method == "incremental_angle"),
                n_frames=n_frames)
    return FHASet(n, s, phi, d, i0, i1, ic, t[ic] if len(ic) else np.zeros(0), weight, meta)


# ---------------------------------------------------------- average axis and spread

def average_axis(n, s, weights=None, anchor=None, anchor_strength=1e-3):
    """Average helical axis of a bundle: optimal direction and optimal pivot point.

    Direction: the unit vector maximising the weighted sum of squared cosines to the
    axes (principal eigenvector of sum w n n^T). Unlike a mean of the vectors, it does
    not cancel when half the axes point the other way, as they do when a movement
    goes out and comes back. Its sign follows the weighted majority.

    Pivot: the point minimising the weighted sum of squared distances to the axes
    (Woltring 1990; Ehrig & Heller 2019). Along the average axis the bundle of a hinge
    carries no information — every point on the axis is equally close — so the pivot
    is tied weakly to ``anchor`` in that direction (default: the mean of ``s``).
    ``anchor_strength`` is the angular spread (as sin^2) above which the data decide.
    """
    n, s = np.asarray(n, float), np.asarray(s, float)
    w = np.full(len(n), 1.0 / max(len(n), 1)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    M = np.einsum("k,ki,kj->ij", w, n, n)
    n_bar = np.linalg.eigh(M)[1][:, -1]
    if np.einsum("k,ki,i->", w, n, n_bar) < 0:
        n_bar = -n_bar
    if anchor is None:
        anchor = np.einsum("k,ki->i", w, s)
    A = np.eye(3) - M                                   # sum of w (I - n n^T)
    b = np.einsum("k,ki->i", w, s) - np.einsum("k,ki,kj,kj->i", w, n, n, s)
    P = np.outer(n_bar, n_bar)
    pivot = np.linalg.solve(A + anchor_strength * P, b + anchor_strength * P @ anchor)
    return n_bar, pivot


def plane_basis(n_bar, toward=None):
    """Right-handed in-plane axes ``e1, e2`` of the plane normal to ``n_bar``.

    ``e1`` points along the projection of ``toward`` (e.g. from the pivot to the moving
    sensor), so the cross-section has a physical orientation instead of an arbitrary
    one. Falls back to a fixed choice when ``toward`` is missing or parallel to ``n_bar``.
    """
    n_bar = np.asarray(n_bar, float)
    e1 = None
    if toward is not None:
        cand = np.asarray(toward, float) - np.dot(toward, n_bar) * n_bar
        if np.linalg.norm(cand) > 1e-9 * max(np.linalg.norm(toward), 1e-300):
            e1 = cand / np.linalg.norm(cand)
    if e1 is None:
        ref = np.eye(3)[np.argmin(np.abs(n_bar))]
        e1 = ref - np.dot(ref, n_bar) * n_bar
        e1 /= np.linalg.norm(e1)
    return e1, np.cross(n_bar, e1)


def cross_section(n, s, n_bar, point, e1, e2):
    """Where each axis pierces the plane through ``point`` normal to ``n_bar``, in the
    plane coordinates ``(e1, e2)``. Axes parallel to the plane give NaN."""
    n, s = np.asarray(n, float), np.asarray(s, float)
    denom = n @ n_bar
    denom = np.where(np.abs(denom) > 1e-9, denom, np.nan)
    hit = s + (((point - s) @ n_bar) / denom)[:, None] * n
    rel = hit - point
    return np.c_[rel @ e1, rel @ e2]


def tilt_map(n, n_bar, e1, e2):
    """Direction of each axis relative to the average axis, in degrees.

    Azimuthal-equidistant map: the distance from the origin is the angle between the
    axis (taken as an undirected line) and the average axis, and the bearing is the
    way it leans. Unlike a cross-section taken away from the pivot, it has no length
    unit and no lever arm.
    """
    n = np.asarray(n, float)
    c = n @ n_bar
    flip = np.where(c < 0, -1.0, 1.0)
    c, a, b = np.abs(c), flip * (n @ e1), flip * (n @ e2)
    tilt = np.degrees(np.arctan2(np.hypot(a, b), c))
    bearing = np.arctan2(b, a)
    return np.c_[tilt * np.cos(bearing), tilt * np.sin(bearing)]


def dispersion(n, s, n_bar, pivot, weights=None):
    """Spread of a bundle about its average axis.

    Angles are between undirected lines (0-90 deg). Distances are perpendicular
    distances from the pivot to each axis. ``*_mean`` and ``*_rms`` use the averaging
    weights of the set, so they do not depend on how fast the movement was performed.
    """
    n, s = np.asarray(n, float), np.asarray(s, float)
    w = np.full(len(n), 1.0 / max(len(n), 1)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    # atan2 of (sine, cosine) keeps full precision near 0 deg, where arccos does not
    ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(n, n_bar), axis=1), np.abs(n @ n_bar)))
    rel = pivot - s
    dist = np.linalg.norm(rel - np.einsum("ki,ki->k", rel, n)[:, None] * n, axis=1)
    return dict(
        angle=ang, distance=dist,
        angle_mean=float(w @ ang), angle_rms=float(np.sqrt(w @ ang**2)), angle_max=float(ang.max()),
        distance_mean=float(w @ dist), distance_rms=float(np.sqrt(w @ dist**2)),
        distance_max=float(dist.max()),
    )


def axes_to_lab(fha, T1):
    """Express the axes of an ``FHASet`` in the lab frame.

    Each axis is mapped with the pose of body 1 at the frame the axis is assigned to.
    Returns ``n_lab, p_lab`` where ``p_lab`` is a point on each axis (the image of
    ``s``; it is no longer the point closest to any particular origin).
    """
    T = np.asarray(T1, float)[fha.ic]
    n_lab = np.einsum("kij,kj->ki", T[:, :3, :3], fha.n)
    p_lab = np.einsum("kij,kj->ki", T[:, :3, :3], fha.s) + T[:, :3, 3]
    return n_lab, p_lab


# --------------------------------------------------------- how the rotation builds up

def swing_twist(Trel, axis, ref_index=0, turning=False):
    """Split the rotation of body 2 since ``ref_index`` into a part about ``axis``
    (twist) and the remainder (swing): how much rotation has ACCUMULATED about the
    axis, and how much away from it.

    Returns ``twist`` (N,) in degrees, unwrapped, and ``swing`` (N, 3) in degrees: a
    rotation vector perpendicular to ``axis`` whose length is the angle between
    ``axis`` and the copy of it carried along by body 2. Both depend only on the pose
    at each frame and the pose at ``ref_index`` — nothing is summed along the way, so
    noise does not build up, and a burst of second rotation raises the swing and
    leaves it raised.

    ``turning=False`` gives the swing as seen from the reference body. ``turning=True``
    gives it as seen from a frame that turns with the twist; a second rotation about
    an axis that is carried along by the first (a block twisted on a hinged beam, the
    distal axis of a joint) then keeps one direction, and its path is a straight line.
    The length of the swing is the same either way.

    The choice of ``axis`` decides what this shows. Against the hinge it shows the
    second rotation building up. Against the average helical axis of the same movement
    it cannot: that axis is by definition the direction of the total rotation, so the
    swing about it has to come back towards zero (20 deg of second rotation reads as
    under 4 deg; research/FHA_deepdive/FINDINGS.md, finding F15).
    """
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    R = np.asarray(Trel, float)[:, :3, :3]
    rel = Rotation.from_matrix(R @ R[ref_index].T)        # displacement in frame 1
    q = rel.as_quat()                                     # (x, y, z, w)
    proj = q[:, :3] @ axis
    twist = np.unwrap(2.0 * np.arctan2(proj, q[:, 3]))
    q_twist = Rotation.from_rotvec(twist[:, None] * axis[None, :])
    swing = (q_twist.inv() * rel if turning else rel * q_twist.inv()).as_rotvec()
    return np.degrees(twist), np.degrees(swing)


def rotation_from_start(Trel, ref_index=0):
    """Net rotation angle (deg) of body 2 relative to its pose at ``ref_index``."""
    q = quaternions(Trel)
    return np.degrees(net_angle(q, q[ref_index][None, :]))
