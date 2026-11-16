"""Tests for fha_core against exact, analytically known answers.

Run either way:
    python tests/test_fha_core.py        (no extra packages needed)
    pytest tests/test_fha_core.py
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fha_core as fc  # noqa: E402
import synthetic as syn  # noqa: E402

RNG = np.random.default_rng(42)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def screw(axis, point, angle, translation):
    """Exact 4x4 screw displacement about the line (axis, point)."""
    axis = unit(axis)
    R = Rotation.from_rotvec(angle * axis).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(point) - R @ np.asarray(point) + translation * axis
    return T


def line_angle(a, b):
    """Angle in degrees between directed unit vectors (row-wise).

    atan2(sine, cosine) rather than arccos: arccos cannot resolve angles below about
    1e-6 deg, which is coarser than the exactness these tests check for.
    """
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                                 np.einsum("...i,...i->...", a, b)))


def point_line_distance(p, n, s):
    """Distance from points p to lines (s, n)."""
    rel = p - s
    return np.linalg.norm(rel - np.einsum("...i,...i->...", rel, n)[..., None] * n, axis=-1)


# ----------------------------------------------------------------- screw extraction

def test_screw_recovers_known_axis_over_full_angle_range():
    for angle_deg in (0.01, 1, 10, 45, 90, 134, 136, 150, 170, 179.9):
        axis = unit(RNG.standard_normal(3))
        point = RNG.uniform(-30, 30, 3)
        d = RNG.uniform(-2, 2)
        phi, n, dd, s = fc.screw_from_displacement(screw(axis, point, np.radians(angle_deg), d))
        assert abs(np.degrees(phi) - angle_deg) < 1e-6, (angle_deg, np.degrees(phi))
        assert line_angle(n, axis) < 1e-4
        assert abs(dd - d) < 1e-8
        foot = point - (point @ axis) * axis          # point of the true axis closest to origin
        assert np.linalg.norm(s - foot) < 1e-6 * max(1, 1 / np.radians(angle_deg))
        assert abs(s @ n) < 1e-8


def test_angle_beyond_135_degrees_is_not_folded_back():
    # The original code returned 180 - phi here (arcsin branch chosen for sin(phi) <= 0.707).
    phi, _, _, _ = fc.screw_from_displacement(screw([0, 0, 1], [1, 2, 3], np.radians(160), 0.0))
    assert abs(np.degrees(phi) - 160) < 1e-9


def test_identity_displacement_has_no_axis():
    phi, n, d, s = fc.screw_from_displacement(np.eye(4))
    assert phi == 0 and np.isnan(n).all() and np.isnan(s).all() and np.isnan(d)


def test_screw_is_vectorised_over_leading_dimensions():
    H = np.stack([[screw([1, 0, 0], [0, 1, 0], 0.3, 0.1), screw([0, 1, 0], [2, 0, 0], 0.5, 0.0)]] * 3)
    phi, n, d, s = fc.screw_from_displacement(H)
    assert phi.shape == (3, 2) and n.shape == (3, 2, 3) and s.shape == (3, 2, 3)
    assert np.allclose(phi, [[0.3, 0.5]] * 3)


def test_se3_log_is_linear_along_a_fixed_screw():
    axis, point, pitch = unit([1, 2, -1]), np.array([4.0, -3.0, 7.0]), 0.8
    xi = np.r_[axis, np.cross(point, axis) + pitch * axis]
    for theta in (-1.2, -0.3, 1e-5, 0.4, 2.5):
        rho, u = fc.se3_log(screw(axis, point, theta, pitch * theta))
        assert np.allclose(np.r_[rho, u], theta * xi, atol=1e-9), theta


def test_invert_matches_numpy():
    T = screw(unit(RNG.standard_normal(3)), RNG.standard_normal(3), 1.1, 0.4)
    assert np.allclose(fc.invert(T), np.linalg.inv(T))
    stack = np.stack([T, T @ T])
    assert np.allclose(fc.invert(stack), np.linalg.inv(stack))


# ------------------------------------------------------------------------- windows

def brute_first_reach(q, thr, direction):
    out = np.full(len(q), -1)
    for i in range(len(q)):
        rng = range(i + 1, len(q)) if direction > 0 else range(i - 1, -1, -1)
        for j in rng:
            if fc.net_angle(q[i], q[j]) >= thr:
                out[i] = j
                break
    return out


def test_first_reach_matches_brute_force_on_non_monotonic_motion():
    # a wandering rotation: forward, pause, backward, with noise
    angle = np.concatenate([np.linspace(0, 0.6, 150), np.full(40, 0.6), np.linspace(0.6, 0.1, 120)])
    rot = Rotation.from_rotvec(angle[:, None] * unit([0.2, 1, 0.1])[None, :]) * \
        Rotation.from_rotvec(0.002 * RNG.standard_normal((len(angle), 3)))
    q = rot.as_quat()
    for thr in (0.05, 0.2):
        for direction in (1, -1):
            fast = fc.first_reach(q, thr, direction, block=7)
            assert np.array_equal(fast, brute_first_reach(q, thr, direction)), (thr, direction)


def test_net_angle_small_and_large():
    a = Rotation.identity().as_quat()
    for ang in (1e-9, 1e-4, 0.5, 3.0):
        b = Rotation.from_rotvec([0, 0, ang]).as_quat()
        assert abs(fc.net_angle(a, b) - ang) < 1e-12 + 1e-9 * ang
        assert abs(fc.net_angle(a, -b) - ang) < 1e-12 + 1e-9 * ang  # q and -q are the same rotation


def test_centred_windows_are_symmetric_and_skip_reversal_and_rest():
    d = syn.two_axis_motion(profile="none", return_trip=True, rot_noise_deg=0, pos_noise=0)
    q = fc.quaternions(fc.relative_poses(d["T1"], d["T2"]))
    step = np.radians(10)
    i0, i1, ic = fc.windows_centred(q, step)
    assert len(ic) > 100
    # each half is at least step/2 and the ends are a full step apart
    assert np.all(fc.net_angle(q[ic], q[i0]) >= step / 2 - 1e-12)
    assert np.all(fc.net_angle(q[ic], q[i1]) >= step / 2 - 1e-12)
    assert np.all(fc.net_angle(q[i0], q[i1]) >= step - 1e-12)
    # nothing is assigned to a frame at rest, nor to one within half a step of the turn
    assert d["moving"][ic].all()
    alpha = d["alpha"]
    assert np.all(alpha[ic] >= 5 - 0.2) and np.all(alpha[ic] <= alpha.max() - 5 + 0.2)
    # a window never straddles the reversal
    turn = len(alpha) // 2
    assert not np.any((i0 < turn) & (i1 > turn) & (np.abs(alpha[i0] - alpha[i1]) < 5))


def test_every_centred_window_spans_the_full_step_even_when_the_axis_changes():
    # Two-axis motion: the direction of rotation differs between the two halves of a
    # window, so half-windows of step/2 each leave the ends less than a step apart.
    for profile in ("proportional", "pulsed", "late"):
        d = syn.two_axis_motion(profile=profile, noise="polhemus", return_trip=True, seed=5)
        q = fc.quaternions(fc.relative_poses(d["T1"], d["T2"]))
        for step_deg in (2.0, 10.0, 25.0):
            step = np.radians(step_deg)
            i0, i1, ic = fc.windows_centred(q, step)
            assert len(ic) > 50, (profile, step_deg)
            assert fc.net_angle(q[i0], q[i1]).min() >= step - 1e-12, (profile, step_deg)
            halves = np.maximum(fc.net_angle(q[ic], q[i0]), fc.net_angle(q[ic], q[i1]))
            assert halves.max() < 0.75 * step, "no half may span three quarters of a step on its own"
            # A window may END during the hold at the top (its end pose is then the top
            # pose), but it may not start on the way out and end on the way back.
            moving = np.flatnonzero(d["moving"])
            turn = len(q) // 2
            hold_start, hold_end = moving[moving < turn].max(), moving[moving > turn].min()
            assert not np.any((i0 < hold_start) & (i1 > hold_end)), "no window across the turn-around"


def test_a_window_swallowing_a_jump_is_dropped():
    # a hinge whose pose jumps by 9 deg in one frame (a marker glitch): the frames beside
    # the jump would get a half-window of 9 deg where 5 was asked for
    angle = np.radians(np.r_[np.linspace(0, 40, 400), 49 + np.linspace(0, 40, 400)])
    T2 = np.stack([screw([0, 1, 0], [0, 0, 0], a, 0.0) for a in angle])
    q = fc.quaternions(T2)
    step = np.radians(10)
    i0, i1, ic = fc.windows_centred(q, step)
    assert len(ic) > 300
    assert not np.any((i0 <= 399) & (i1 >= 400)), "no window may contain the jump"
    assert fc.net_angle(q[i0], q[i1]).max() < np.radians(10.5)


def test_coarse_frames_keep_their_windows_and_a_jump_among_them_is_still_dropped():
    hinge = lambda degrees: fc.quaternions(np.stack([screw([0, 1, 0], [0, 0, 0], a, 0.0) for a in np.radians(degrees)]))
    # 4 deg per frame, as in a dynamic CT series: half of a 10 deg window is two frames, 8 deg
    q = hinge(np.arange(0, 92, 4.0))
    i0, i1, ic = fc.windows_centred(q, np.radians(10))
    assert len(ic) == len(q) - 4, "every frame that has two frames on either side gets a window"
    assert np.all(ic - i0 == 2) and np.all(i1 - ic == 2)
    assert np.allclose(np.degrees(fc.net_angle(q[i0], q[i1])), 16.0), "as narrow as the frames allow"
    # the same goes for a small window on a fast movement sampled finely
    q = hinge(np.arange(0, 60, 0.4))
    assert len(fc.windows_centred(q, np.radians(1.0))[2]) == len(q) - 4
    # a 20 deg step among 4 deg steps is a jump: no window may contain it
    q = hinge(np.r_[np.arange(0, 44, 4.0), 60 + np.arange(0, 44, 4.0)])
    i0, i1, ic = fc.windows_centred(q, np.radians(10))
    assert len(ic) >= 10 and not np.any((i0 <= 10) & (i1 >= 11))
    # one frame missing among them (8 deg in one step) is not
    q = hinge(np.r_[np.arange(0, 44, 4.0), 48 + np.arange(0, 44, 4.0)])
    i0, i1, ic = fc.windows_centred(q, np.radians(10))
    assert np.any((i0 <= 10) & (i1 >= 11))


def test_chained_windows_tile_the_motion():
    d = syn.two_axis_motion(profile="none", rot_noise_deg=0, pos_noise=0)
    q = fc.quaternions(fc.relative_poses(d["T1"], d["T2"]))
    i0, i1, ic = fc.windows_chained(q, np.radians(10))
    assert len(i0) == 8 or len(i0) == 9          # 90 deg in 10 deg steps (frame quantisation)
    assert np.array_equal(i0[1:], i1[:-1])
    assert np.all((ic > i0) & (ic < i1))


def test_path_criterion_closes_early_under_noise_but_net_does_not():
    d = syn.two_axis_motion(profile="none", rot_noise_deg=0.05, pos_noise=0.01, seed=3)
    q = fc.quaternions(fc.relative_poses(d["T1"], d["T2"]))
    step = np.radians(10)
    a0, a1, _ = fc.windows_forward(q, step, "net")
    b0, b1, _ = fc.windows_forward(q, step, "path")
    assert np.all(fc.net_angle(q[a0], q[a1]) >= step - 1e-12)
    short = fc.net_angle(q[b0], q[b1]) < 0.8 * step
    assert short.mean() > 0.2, "noise should make path-length windows fall short of the step"


# ---------------------------------------------------------------------- estimators

def lab_axes(fha, d):
    return fc.axes_to_lab(fha, d["T1_clean"])


def test_every_method_recovers_a_pure_hinge_exactly():
    d = syn.two_axis_motion(profile="none", rot_noise_deg=0, pos_noise=0)
    hinge_n, hinge_p = d["hinge_axis"], d["hinge_point"]
    for method, step in (("incremental_angle", 10), ("step_angle", 10), ("reference_pose", 10),
                         ("incremental_time", 60)):
        for estimator in fc.ESTIMATORS:
            fha = fc.compute_fha(d["t"], d["T1"], d["T2"], method, step, estimator)
            assert len(fha) > 3, method
            n, p = lab_axes(fha, d)
            assert line_angle(n, hinge_n).max() < 1e-5, (method, estimator)
            assert point_line_distance(hinge_p, n, p).max() < 1e-6, (method, estimator)
            assert np.abs(fha.d).max() < 1e-7, (method, estimator)


def test_fit_is_exact_on_a_pure_screw_with_pitch():
    # screw with translation along the axis: 0.5 length units per radian
    axis, point, pitch = unit([0.3, 1.0, -0.2]), np.array([5.0, -2.0, 1.0]), 0.5
    theta = np.linspace(0, 1.4, 500)
    T2 = np.stack([screw(axis, point, th, pitch * th) for th in theta]) @ screw([1, 0, 0], [0, 0, 0], 0.3, 2.0)
    T1 = np.tile(np.eye(4), (len(theta), 1, 1))
    for estimator in fc.ESTIMATORS:
        fha = fc.compute_fha(np.arange(len(theta)) / 100, T1, T2, "incremental_angle", 10, estimator)
        assert line_angle(fha.n, axis).max() < 1e-6, estimator
        assert point_line_distance(point, fha.n, fha.s).max() < 1e-7, estimator
        assert np.allclose(fha.d / fha.phi, pitch, atol=1e-7), estimator


def test_centred_window_tracks_the_true_axis_at_its_centre_frame():
    d = syn.two_axis_motion(profile="late", rot_noise_deg=0, pos_noise=0)
    fha = fc.compute_fha(d["t"], d["T1"], d["T2"], "incremental_angle", 10)
    n, p = lab_axes(fha, d)
    err_centre = line_angle(n, d["truth_n"][fha.ic])
    err_start = line_angle(n, d["truth_n"][fha.i0])
    assert err_centre.mean() < 0.4
    assert err_start.mean() > 3 * err_centre.mean(), "tagging at the window start is the worse choice"


def test_fit_beats_endpoints_under_white_noise():
    errs = {e: [] for e in fc.ESTIMATORS}
    for seed in range(4):
        d = syn.two_axis_motion(profile="none", rot_noise_deg=0.03, pos_noise=0.01, seed=seed)
        for e in fc.ESTIMATORS:
            fha = fc.compute_fha(d["t"], d["T1"], d["T2"], "incremental_angle", 10, e)
            n, _ = fc.axes_to_lab(fha, d["T1_clean"])
            errs[e].append(line_angle(n, d["hinge_axis"]).mean())
    assert np.mean(errs["fit"]) < 0.5 * np.mean(errs["endpoints"]), errs


def test_reference_pose_changes_gradually_when_the_second_rotation_is_pulsed():
    d = syn.two_axis_motion(profile="pulsed", rot_noise_deg=0, pos_noise=0)
    mi = fc.compute_fha(d["t"], d["T1"], d["T2"], "incremental_angle", 10)
    ref = fc.compute_fha(d["t"], d["T1"], d["T2"], "reference_pose", 10)
    wiggle = lambda f: np.abs(np.diff(line_angle(f.n, np.tile(d["hinge_axis"], (len(f), 1))), 2)).sum()
    assert wiggle(ref) < 0.25 * wiggle(mi)


def test_empty_result_when_there_is_no_motion():
    T = np.tile(np.eye(4), (200, 1, 1))
    for method in ("incremental_angle", "step_angle", "reference_pose"):
        for estimator in fc.ESTIMATORS:
            fha = fc.compute_fha(np.arange(200) / 100, T, T, method, 10, estimator)
            assert len(fha) == 0, (method, estimator)
            assert fha.n.shape == (0, 3) and fha.s.shape == (0, 3)
    # frames that do not move give no axis for the frame-based methods either
    for method in ("all_FHA", "incremental_time"):
        assert len(fc.compute_fha(np.arange(200) / 100, T, T, method, 5)) == 0, method
    # the low-level functions accept empty batches too
    phi, n, d, s = fc.screw_from_displacement(np.zeros((0, 4, 4)))
    assert phi.shape == (0,) and n.shape == (0, 3)
    rho, u = fc.se3_log(np.zeros((0, 4, 4)))
    assert rho.shape == (0, 3) and u.shape == (0, 3)


def test_bad_arguments_are_rejected():
    T = np.tile(np.eye(4), (10, 1, 1))
    for kwargs in (dict(method="nope"), dict(estimator="nope"), dict(criterion="nope"), dict(step=0)):
        try:
            fc.compute_fha(np.arange(10.0), T, T, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_weights_do_not_depend_on_speed():
    # a 90 deg hinge movement: the first 45 deg in 200 frames, the last 45 deg in 800
    angle = np.radians(np.r_[np.linspace(0, 45, 200, endpoint=False), np.linspace(45, 90, 800)])
    T2 = np.stack([screw([0, 1, 0], [0, 0, 0], a, 0.0) for a in angle]) @ screw([1, 0, 0], [3, 1, 2], 0.2, 0.0)
    T1 = np.tile(np.eye(4), (len(angle), 1, 1))
    fha = fc.compute_fha(np.arange(len(angle)) / 120, T1, T2, "incremental_angle", 10)
    fast_half = np.degrees(angle[fha.ic]) < 45
    assert fast_half.mean() < 0.3, "the fast half contributes far fewer axes..."
    assert abs(fha.weight[fast_half].sum() - 0.5) < 0.05, "...yet the same share of the weight"


# ---------------------------------------------------------- average axis and spread

def test_average_axis_ignores_direction_flips():
    axis, point = unit([1, 3, 2]), np.array([2.0, -1.0, 4.0])
    n = unit(axis)[None, :] + 0.01 * RNG.standard_normal((200, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    s = point + RNG.uniform(-5, 5, (200, 1)) * n            # arbitrary points on each line
    n_flipped = n * np.where(np.arange(200) % 2 == 0, 1.0, -1.0)[:, None]
    nb1, p1 = fc.average_axis(n, s)
    nb2, p2 = fc.average_axis(n_flipped, s)
    assert line_angle(nb1, axis) < 0.2
    assert min(line_angle(nb2, axis), line_angle(-nb2, axis)) < 0.2
    assert np.linalg.norm(np.mean(n_flipped, axis=0)) < 0.1, "a plain mean of the vectors cancels"
    assert np.allclose(p1, p2, atol=1e-9)


def test_pivot_is_the_common_point_of_intersecting_axes():
    centre = np.array([3.0, -7.0, 2.0])
    n = unit([0, 0, 1])[None, :] + 0.3 * RNG.standard_normal((300, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    s = centre + RNG.uniform(-8, 8, (300, 1)) * n
    _, pivot = fc.average_axis(n, s)
    assert np.linalg.norm(pivot - centre) < 1e-2


def test_pivot_of_parallel_axes_falls_back_to_the_anchor_along_the_axis():
    n = np.tile(unit([0, 1, 0]), (50, 1))
    s = np.c_[np.full(50, 2.0), RNG.uniform(-9, 9, 50), np.full(50, -1.0)]
    n_bar, pivot = fc.average_axis(n, s, anchor=np.array([0.0, 4.0, 0.0]))
    assert np.allclose(pivot, [2.0, 4.0, -1.0], atol=1e-9)


def test_cross_section_and_tilt_map_geometry():
    n_bar = np.array([0.0, 0.0, 1.0])
    e1, e2 = fc.plane_basis(n_bar, toward=[5.0, 0.0, 3.0])
    assert np.allclose(e1, [1, 0, 0]) and np.allclose(e2, [0, 1, 0])
    # an axis leaning 10 deg toward +e1, passing through (1, 2, 0)
    n = np.array([[np.sin(np.radians(10)), 0.0, np.cos(np.radians(10))]])
    s = np.array([[1.0, 2.0, 0.0]])
    assert np.allclose(fc.cross_section(n, s, n_bar, np.zeros(3), e1, e2), [[1.0, 2.0]])
    up = fc.cross_section(n, s, n_bar, np.array([0, 0, 10.0]), e1, e2)
    assert np.allclose(up, [[1.0 + 10 * np.tan(np.radians(10)), 2.0]])
    assert np.allclose(fc.tilt_map(n, n_bar, e1, e2), [[10.0, 0.0]])
    assert np.allclose(fc.tilt_map(-n, n_bar, e1, e2), [[10.0, 0.0]])   # undirected line
    # e1, e2, n_bar form a right-handed frame for any normal
    nb = unit(RNG.standard_normal(3))
    a, b = fc.plane_basis(nb)
    assert np.allclose(np.cross(a, b), nb) and abs(a @ nb) < 1e-12


def test_dispersion_of_a_perfect_bundle_is_zero():
    n = np.tile(unit([1, 1, 0]), (20, 1))
    s = np.tile(np.array([0.0, 0.0, 5.0]), (20, 1))
    disp = fc.dispersion(n, s, unit([1, 1, 0]), np.array([0.0, 0.0, 5.0]))
    assert disp["angle_max"] < 1e-6 and disp["distance_max"] < 1e-12


# --------------------------------------------------------- how the rotation builds up

def test_swing_twist_recovers_both_rotations_of_the_rig():
    d = syn.two_axis_motion(profile="proportional", rot_noise_deg=0, pos_noise=0, still_s=0.0)
    # work in the lab frame: with T1 = identity, "frame 1" is the lab
    twist, swing = fc.swing_twist(d["T2_clean"], d["hinge_axis"])
    assert np.allclose(twist, d["alpha"], atol=1e-6)
    assert np.allclose(np.linalg.norm(swing, axis=1), d["beta"], atol=1e-6)
    assert np.abs(swing @ d["hinge_axis"]).max() < 1e-6          # swing is perpendicular to the axis


def test_swing_as_seen_turning_with_the_twist_keeps_one_direction_for_a_carried_axis():
    d = syn.two_axis_motion(profile="pulsed", rot_noise_deg=0, pos_noise=0, still_s=0.0)
    T = d["T2_clean"]
    twist, fixed = fc.swing_twist(T, d["hinge_axis"])
    twist_t, turning = fc.swing_twist(T, d["hinge_axis"], turning=True)
    assert np.allclose(twist, twist_t)
    assert np.allclose(np.linalg.norm(fixed, axis=1), np.linalg.norm(turning, axis=1), atol=1e-9)
    moved = d["beta"] > 0.5
    unit_dir = turning[moved] / np.linalg.norm(turning[moved], axis=1, keepdims=True)
    assert np.abs(unit_dir - unit_dir[0]).max() < 1e-6, "second axis is carried: one direction throughout"
    spread = fixed[moved] / np.linalg.norm(fixed[moved], axis=1, keepdims=True)
    assert np.abs(spread - spread[0]).max() > 0.5, "seen from the reference body the direction sweeps round"
    # bursts add to the swing and it never comes back down
    assert np.all(np.diff(np.linalg.norm(turning, axis=1)) > -1e-9)


def test_swing_about_the_average_axis_cannot_show_a_second_rotation_building_up():
    # 20 deg of second rotation, steadily. Against the hinge the swing reaches 20 deg;
    # against the average axis of the same movement it stays small and ends smaller.
    d = syn.two_axis_motion(profile="proportional", rot_noise_deg=0, pos_noise=0)
    Trel = fc.relative_poses(d["T1"], d["T2"])
    fha = fc.compute_fha(d["t"], d["T1"], d["T2"])
    average, _ = fc.average_axis(fha.n, fha.s, fha.weight)
    hinge = d["T1_clean"][0, :3, :3].T @ d["hinge_axis"]          # in the reference body's frame
    about_hinge = np.linalg.norm(fc.swing_twist(Trel, hinge)[1], axis=1)
    about_average = np.linalg.norm(fc.swing_twist(Trel, average)[1], axis=1)
    assert abs(about_hinge[-1] - 20.0) < 0.01 and np.all(np.diff(about_hinge) > -1e-9)
    assert about_average.max() < 5.0 and about_average[-1] < about_hinge[-1] / 4


def test_swing_twist_unwraps_past_180_degrees():
    theta = np.linspace(0, np.radians(300), 400)
    T = np.stack([screw([0, 0, 1], [0, 0, 0], th, 0.0) for th in theta])
    twist, swing = fc.swing_twist(T, [0, 0, 1])
    assert np.allclose(twist, np.degrees(theta), atol=1e-6)
    assert np.abs(swing).max() < 1e-6


# ----------------------------------------------------------------------- smoothing

def test_smoothing_has_the_stated_cutoff_and_no_time_shift():
    fs = 120.0
    t = np.arange(4000) / fs
    for cutoff in (1.0, 6.0, 15.0):
        for f, expected in ((cutoff, 0.7071), (cutoff / 10, 1.0)):
            y = np.sin(2 * np.pi * f * t)[:, None]
            out = fc.smooth(t, y, cutoff)[1200:-1200, 0]
            assert abs(np.abs(out).max() - expected) < 0.004, (cutoff, f, np.abs(out).max())
    pulse = np.exp(-((t - 10) / 0.2) ** 2)[:, None]
    assert np.argmax(fc.smooth(t, pulse, 2.0)) == np.argmax(pulse)


def test_smoothing_matches_the_usual_zero_lag_butterworth():
    from scipy.signal import butter, sosfiltfilt

    fs = 100.0
    t = np.arange(3000) / fs
    y = np.sin(2 * np.pi * 0.3 * t) + 0.3 * np.sin(2 * np.pi * 2.0 * t) + 0.05 * RNG.standard_normal(len(t))
    for cutoff in (1.0, 6.0):
        ours = fc.smooth(t, y[:, None], cutoff)[:, 0]
        # second-order Butterworth run forwards and backwards; 0.802 puts its -3 dB point on `cutoff`
        usual = sosfiltfilt(butter(2, cutoff / 0.802, fs=fs, output="sos"), y)
        assert np.abs(ours - usual)[300:-300].max() < 0.002, cutoff


def test_smoothing_leaves_steady_movement_and_off_settings_alone():
    t = np.linspace(0, 5, 600)
    line = np.c_[3 * t + 1, -2 * t]
    assert np.abs(fc.smooth(t, line, 0.3) - line).max() < 1e-6
    y = RNG.standard_normal((600, 2))
    assert np.array_equal(fc.smooth(t, y, None), y)
    assert np.array_equal(fc.smooth(t, y, 1e6), y), "a cut-off above half the sampling rate does nothing"
    assert np.array_equal(fc.smooth(t[:3], y[:3], 1.0), y[:3]), "too few samples to smooth"


def test_smoothing_works_on_a_few_unevenly_spaced_frames():
    for n in (4, 6, 15):
        t = np.sort(RNG.uniform(0, 2, n))
        y = np.c_[t**2, np.sin(3 * t)]
        out = fc.smooth(t, y, 0.5)
        assert out.shape == y.shape and np.isfinite(out).all()
    # 15 frames of a slow curve with noise: smoothing brings it closer to the curve
    t = np.linspace(0, 2, 15)
    clean = np.sin(1.5 * t)[:, None]
    worse = better = 0
    for _ in range(200):
        noisy = clean + 0.05 * RNG.standard_normal(clean.shape)
        worse += np.sum((noisy - clean) ** 2)
        better += np.sum((fc.smooth(t, noisy, 1.0) - clean) ** 2)
    assert better < 0.8 * worse


def uneven_hinge(noise=0.0, seed=0):
    """A hinge turned stop-and-go (speed swinging to zero once a second), sensor 20 off the axis."""
    fs = 120.0
    t = np.arange(int(8 * fs)) / fs
    speed = np.radians(12.0) * (1 + np.sin(2 * np.pi * t))
    speed[(t < 0.5) | (t > 7.5)] = 0
    alpha = np.cumsum(speed) / fs
    start = screw([1, 0, 0], [0, 0, 0], 0.2, 0.0)
    start[:3, 3] = [20.0, 0.0, 3.0]
    T2 = np.stack([screw([0, 1, 0], [0, 0, 0], a, 0.0) for a in alpha]) @ start
    T1 = np.tile(screw([0, 0, 1], [0, 0, 0], 0.1, 0.0), (len(t), 1, 1))
    T1[:, :3, 3] = [-6.0, 2.0, 3.0]
    if noise:
        rng = np.random.default_rng(seed)
        T2 = T2 @ np.stack([screw(unit(v), [0, 0, 0], np.radians(noise) * np.linalg.norm(v), 0.0)
                            for v in rng.standard_normal((len(t), 3))])
    return t, T1, T2


def axes_off_hinge(t, T1, T2):
    """Largest angle (deg) and distance of the moving-window axes from the y axis through the origin."""
    fha = fc.compute_fha(t, T1, T2)
    n, p = fc.axes_to_lab(fha, T1)
    hinge = np.tile([0.0, 1.0, 0.0], (len(n), 1))
    return line_angle(n, hinge).max(), point_line_distance(np.zeros(3), n, p).max()


def test_smoothed_poses_stay_rigid():
    d = syn.two_axis_motion(profile="pulsed", noise="vicon", fs=100.0, move_s=5.0, seed=4)
    for screw_coords in (True, False):
        out = fc.smooth_poses(d["t"], d["T2"], 3.0, screw=screw_coords)
        R = out[:, :3, :3]
        assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3), atol=1e-10)
        assert np.allclose(np.linalg.det(R), 1.0) and np.allclose(out[:, 3], [0, 0, 0, 1])
        assert np.array_equal(fc.smooth_poses(d["t"], d["T2"], None, screw=screw_coords), d["T2"])


def test_smoothing_in_screw_coordinates_leaves_a_hinge_exactly_a_hinge():
    t, T1, T2 = uneven_hinge()
    for cutoff in (6.0, 1.0, 0.3):
        angle, distance = axes_off_hinge(t, *fc.smooth_motion(t, T1, T2, cutoff))
        assert angle < 1e-6 and distance < 1e-7, (cutoff, angle, distance)


def test_smoothing_positions_and_orientation_separately_moves_the_axis():
    # the usual way of filtering: with uneven speed the filter itself displaces the hinge
    t, T1, T2 = uneven_hinge()
    _, distance = axes_off_hinge(t, *fc.smooth_motion(t, T1, T2, 1.0, screw=False))
    assert distance > 0.05, "over half a millimetre for a sensor 20 cm from the axis at a 1 Hz cut-off"
    _, mild = axes_off_hinge(t, *fc.smooth_motion(t, T1, T2, 6.0, screw=False))
    assert mild < 0.002, "negligible at the cut-offs normally used"


def test_screw_coordinates_survive_more_than_half_a_turn_and_quaternion_sign_flips():
    angle = np.radians(np.linspace(0, 400, 800))
    T = np.stack([screw([0, 0, 1], [3, 1, 0], a, 0.01 * a) for a in angle])
    t = np.arange(800) / 100
    for screw_coords in (True, False):
        out = fc.smooth_poses(t, T, 2.0, screw=screw_coords)
        turned = np.degrees(fc.net_angle(fc.quaternions(T), fc.quaternions(out)))
        assert turned[50:-50].max() < 1e-3, "a steady rotation must pass unchanged"
    rho, u, theta = fc._continuous_log(T @ fc.invert(T[400])[None])
    assert np.abs(np.diff(rho, axis=0)).max() < 0.02, "rotation vector continuous through 180 deg"
    assert np.abs(fc._exp_se3(rho, u) @ T[400][None] - T).max() < 1e-9


def test_smoothing_reduces_noise_error_of_a_hinge_and_reports_what_it_removed():
    t, T1, T2 = uneven_hinge(noise=0.05, seed=3)
    raw_angle, raw_distance = axes_off_hinge(t, T1, T2)
    angle, distance = axes_off_hinge(t, *fc.smooth_motion(t, T1, T2, 3.0))
    assert angle < 0.4 * raw_angle and distance < 0.4 * raw_distance, (raw_angle, angle, raw_distance, distance)
    raw = fc.relative_poses(T1, T2)
    removed, moved = fc.removed_by_smoothing(raw, fc.relative_poses(*fc.smooth_motion(t, T1, T2, 3.0)))
    jitter_angle, _ = fc.jitter(raw)
    assert removed.shape == (len(raw),)
    assert 0.7 * jitter_angle < np.sqrt(np.mean(removed**2)) < 1.1 * jitter_angle, "only the jitter was removed"
    same_angle, same_dist = fc.removed_by_smoothing(raw, raw)
    assert same_angle.max() < 1e-9 and same_dist.max() == 0


def test_jitter_estimate_finds_the_white_noise_and_ignores_the_movement():
    for sigma in (0.002, 0.02):
        d = syn.two_axis_motion(profile="pulsed", rot_noise_deg=sigma, pos_noise=0.004, seed=6)
        angle, dist = fc.jitter(d["T2"])
        # per-axis noise sigma on three axes -> RMS size sqrt(3) * sigma
        assert abs(angle - np.sqrt(3) * sigma) < 0.15 * np.sqrt(3) * sigma, (sigma, angle)
        assert abs(dist - np.sqrt(3) * 0.004) < 0.15 * np.sqrt(3) * 0.004
    clean = syn.two_axis_motion(profile="pulsed", rot_noise_deg=0, pos_noise=0)
    assert not fc.jitter(clean["T2"])[0] > 2e-4, "movement alone, however jerky, is not counted as jitter"


def ct_like(n_frames, noise_deg, seed=0):
    """A smooth two-axis movement in ``n_frames`` poses with an independent error on each,
    as from registering each volume of a dynamic CT series separately."""
    d = syn.two_axis_motion(profile="late", noise="none", fs=n_frames / 4.0, move_s=3.6, still_s=0.2)
    T = fc.relative_poses(d["T1"], d["T2"])[:n_frames].copy()
    error = Rotation.from_rotvec(np.radians(noise_deg) * np.random.default_rng(seed).standard_normal((len(T), 3)))
    T[:, :3, :3] = T[:, :3, :3] @ error.as_matrix()
    return d["t"][:n_frames], T


def test_jitter_on_a_few_frames_is_rough_but_usable_and_withheld_when_it_cannot_be_told():
    truth = np.sqrt(3) * 0.3
    found = np.array([fc.jitter(ct_like(24, 0.3, seed)[1])[0] for seed in range(40)])
    assert np.isfinite(found).mean() > 0.9, "24 frames with a 0.3 deg error each: almost always reported"
    assert 0.8 * truth < np.nanmedian(found) < 1.35 * truth, np.nanmedian(found)
    assert np.isnan(fc.jitter(ct_like(60, 0.0)[1])[0]), "smooth movement and no error at all: nothing to report"
    assert np.isnan(fc.jitter(ct_like(19, 0.3)[1])[0]), "fewer than 20 frames"
    assert fc.jitter(np.tile(np.eye(4), (200, 1, 1))) == (0.0, 0.0)


def test_a_few_frames_can_be_smoothed_and_a_heavy_filter_does_more_harm_than_good():
    """Dynamic CT: a couple of dozen frames, a few per second."""
    t, clean = ct_like(24, 0.0)
    off = lambda T: np.sqrt(np.mean(np.degrees(fc.net_angle(fc.quaternions(T), fc.quaternions(clean))) ** 2))
    rate = 1.0 / np.median(np.diff(t))
    assert 5.5 < rate < 6.5
    noisy = [ct_like(24, 0.3, seed)[1] for seed in range(8)]
    left = lambda cutoff: np.mean([off(fc.smooth_poses(t, T, cutoff)) for T in noisy])
    assert np.array_equal(fc.smooth_poses(t, noisy[0], 0.501 * rate), noisy[0]), "at half the frame rate nothing is removed"
    assert left(1.5) < 0.9 * left(None), "a light filter takes some of the per-frame error out"
    assert off(fc.smooth_poses(t, clean, 1.5)) < 0.2 * left(None), "and barely touches the movement"
    assert left(0.5) > left(None), "a heavy one removes more movement than error"


# -------------------------------------------------------------------------- timing

def test_uniform_time_removes_arrival_jitter_but_respects_real_gaps():
    true = np.arange(1000) / 120.0
    jittered = true + RNG.uniform(-0.008, 0.008, 1000)
    fixed = fc.uniform_time(jittered)
    assert np.allclose(np.diff(fixed), 1 / 120, rtol=1e-3)
    assert np.abs(fixed - true).max() < 2e-3
    gappy = true.copy()
    gappy[500:] += 0.5                       # half a second of missing frames
    kept, pauses = fc.even_time_base(gappy)
    assert np.allclose(kept, gappy, atol=1e-9) and len(pauses) == 1
    assert pauses[0][0] == 500 and abs(pauses[0][1] - 0.5) < 1e-9


def burst_stamps(n=6000, fs=120.0, seed=3):
    """Arrival stamps of an evenly sampling device whose samples reach the host in bursts:
    nothing for up to 90 ms, then everything at once. Returns (true times, stamps)."""
    rng = np.random.default_rng(seed)
    true = np.arange(n) / fs
    stamp = true + rng.uniform(0, 2e-4, n)
    i = 0
    while i < n:
        if rng.random() < 0.15:
            j = min(i + int(rng.integers(3, 12)), n) - 1
            stamp[i:j + 1] = true[j] + np.arange(j + 1 - i) * 1e-5
            i = j + 1
        else:
            i += 1
    return true, stamp


def test_burst_delivery_is_read_as_even_sampling():
    true, stamp = burst_stamps()
    steps = np.diff(stamp)
    assert steps.min() < 2e-5 and steps.max() > 0.05, "the stamps really are bunched"
    even, pauses = fc.even_time_base(stamp)
    assert pauses == []
    assert np.allclose(np.diff(even), 1 / 120, rtol=1e-4)
    assert np.ptp(even - true) < 1e-3, "the sample times are recovered to within a millisecond"


def test_a_real_pause_is_kept_and_a_clock_set_back_is_not():
    true, stamp = burst_stamps()
    paused = stamp.copy()
    paused[400:] += 3.2
    even, pauses = fc.even_time_base(paused)
    assert [p[0] for p in pauses] == [400] and abs(pauses[0][1] - 3.2) < 2e-3
    steps = np.diff(even)
    assert np.allclose(np.delete(steps, 399), 1 / 120, rtol=1e-4) and abs(steps[399] - 3.2 - 1 / 120) < 2e-3

    stepped = stamp.copy()
    stepped[300:] -= 0.1                                   # host clock corrected during the recording
    assert np.diff(stepped).min() < 0
    even, pauses = fc.even_time_base(stepped)
    assert pauses == [] and np.allclose(np.diff(even), 1 / 120, rtol=1e-4)

    late_end = true + RNG.uniform(0, 2e-4, len(true))
    late_end[2000:2010] = late_end[2009]
    late_end[-1] += 0.06                                   # a late last stamp is not a pause
    even, pauses = fc.even_time_base(late_end)
    assert pauses == [] and np.allclose(np.diff(even), 1 / 120, rtol=1e-4)


def test_stamps_that_are_not_arrival_times_are_left_alone():
    irregular = np.cumsum(np.random.default_rng(4).exponential(1 / 120, 3000))
    even, pauses = fc.even_time_base(irregular)
    assert np.array_equal(even, irregular) and pauses == []


def test_smoothing_refuses_times_that_do_not_increase_and_missing_values():
    d = syn.two_axis_motion(profile="none", noise="polhemus")
    t = d["t"].copy()
    t[300:] -= 0.1
    for bad_t, bad_T2, word in ((t, d["T2"], "increase"), (d["t"], np.where(np.arange(len(t))[:, None, None] == 50, np.nan, d["T2"]), "Missing")):
        try:
            fc.smooth(bad_t, bad_T2[:, :3, 3], 2.0)
        except ValueError as exc:
            assert word in str(exc), exc
        else:
            raise AssertionError(f"expected a ValueError mentioning {word!r}")


def test_smoothing_with_burst_stamps_taken_at_face_value_invents_bursts():
    """Why the time base is repaired before anything is filtered in time."""
    fs = 120.0
    true, stamp = burst_stamps(n=1200)
    angle = np.radians(90.0) * 0.5 * (1 - np.cos(np.pi * true / true[-1]))          # one smooth sweep
    pos = np.c_[20 * np.cos(angle), 20 * np.sin(angle), np.zeros(len(true))]
    worst = lambda times: np.abs(fc.smooth(times, pos, 6.0) - pos).max()
    assert worst(fc.uniform_time(stamp)) < 1e-3, "on the repaired time base a smooth movement passes untouched"
    assert worst(stamp) > 30 * worst(fc.uniform_time(stamp)), "on the raw stamps the filter distorts it"


# -------------------------------------------------------------------------- runner

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
