"""Tests for the least-squares segment frames in vicon_import and for the loaders.

These work on marker arrays directly, so ezc3d is not needed.

    python tests/test_segment_frames.py
    pytest tests/test_segment_frames.py
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fha_core as fc  # noqa: E402
import recordings as rc  # noqa: E402
import vicon_import as vi  # noqa: E402

RNG = np.random.default_rng(7)
# a five-marker cluster in its own frame (mm); the first three define the segment frame
LOCAL = np.array([[0.0, 96.0, 0.0], [-17.0, 0.0, 0.0], [17.0, 0.0, 0.0], [5.0, 40.0, 20.0], [-8.0, 60.0, -15.0]])
NAMES = ["m0", "m1", "m2", "m3", "m4"]


def moving_cluster(n=400, noise=0.0):
    """Markers of a rigid cluster following a known motion, plus that motion."""
    R = Rotation.from_rotvec(np.linspace(0, 1.5, n)[:, None] * np.array([[0.2, 1.0, 0.1]])).as_matrix()
    p = np.c_[np.linspace(0, 50, n), np.zeros(n), np.linspace(10, -20, n)]
    X = np.einsum("nij,mj->nmi", R, LOCAL) + p[:, None, :] + noise * RNG.standard_normal((n, len(LOCAL), 3))
    return {name: dict(x=X[:, i, 0], y=X[:, i, 1], z=X[:, i, 2]) for i, name in enumerate(NAMES)}, R, p


def angle_between(Ra, Rb):
    return np.degrees(np.linalg.norm(Rotation.from_matrix(np.einsum("nij,nkj->nik", Ra, Rb)).as_rotvec(), axis=1))


def test_least_squares_frames_equal_the_three_marker_frames_on_rigid_data():
    md, _, _ = moving_cluster()
    _, T_direct, _, _, _, _ = vi.build_reference_frames(md, "m0", "m1", "m2", "m0", "m1", "m2")
    for names in (NAMES[:3], NAMES):
        _, T_fit, _, _, loc, _, res = vi.build_reference_frames_lsq(md, names, names)
        assert np.abs(T_fit - T_direct).max() < 1e-9, len(names)
        assert res["residual_j1"].max() < 1e-9
        assert np.allclose(loc, T_fit[:, :3, 3])


def test_frames_are_proper_rotations():
    md, _, _ = moving_cluster(noise=0.3)
    R1, T1, R2, T2, _, _, _ = vi.build_reference_frames_lsq(md, NAMES, NAMES[:3])
    for R in (R1, R2):
        assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3), atol=1e-12)
        assert np.allclose(np.linalg.det(R), 1.0)
    assert np.allclose(T1[:, 3], [0, 0, 0, 1])


def test_fit_is_less_noisy_than_the_direct_construction():
    # same three markers, same noise: the direct frame leans on the 34 mm baseline for
    # its in-plane rotation, the fit uses all three markers
    md, R_true, _ = moving_cluster(noise=0.1)
    R_direct, *_ = vi.build_reference_frames(md, "m0", "m1", "m2", "m0", "m1", "m2")
    R_fit3, *_ = vi.build_reference_frames_lsq(md, NAMES[:3], NAMES[:3])
    R_fit5, *_ = vi.build_reference_frames_lsq(md, NAMES, NAMES[:3])
    # compare motion relative to the first frame, which removes the constant frame offset
    rel = lambda R: np.einsum("nij,kj->nik", R, R[0])
    err = lambda R: np.sqrt(np.mean(angle_between(rel(R), rel(R_true)) ** 2))
    assert err(R_fit3) < 0.95 * err(R_direct), (err(R_fit3), err(R_direct))
    assert err(R_fit5) < err(R_fit3)


def test_residual_grows_when_a_marker_is_displaced_in_the_cluster_plane():
    md, _, _ = moving_cluster()
    md["m2"]["x"] = md["m2"]["x"].copy()
    md["m2"]["x"][200:260] += 3.0          # one marker 3 mm out of place for 60 frames
    for names in (NAMES, NAMES[:3]):
        *_, res = vi.build_reference_frames_lsq(md, names, names)
        r = res["residual_j1"]
        assert r[:200].max() < 0.01, "frames with all markers in place fit exactly"
        assert r[200:260].min() > 0.5, "the displaced stretch stands out"


def test_missing_markers_are_reported_not_propagated():
    md, _, _ = moving_cluster()
    md["m1"]["y"] = md["m1"]["y"].copy()
    md["m1"]["y"][10:15] = np.nan
    try:
        vi.build_reference_frames_lsq(md, NAMES[:3], NAMES[:3])
    except ValueError as exc:
        assert "5 frames have missing markers" in str(exc)
    else:
        raise AssertionError("expected a ValueError")
    try:
        vi.build_reference_frames_lsq(md, NAMES[:2], NAMES[:3])
    except ValueError as exc:
        assert "at least three markers" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_polhemus_loader_matches_the_original_importer():
    import polhemus_import as pi

    path = os.path.join(rc.TEST_DATA, "Polhemus_test_data", "1_90deg_ydata_oneway.csv")
    rec = rc.load_polhemus(path)
    q1, q2, loc1, loc2, t = pi.read_data(path)
    _, T1, _, T2 = pi.quaternions_to_matrices(q1, q2, loc1, loc2)
    assert np.array_equal(rec.T1, np.asarray(T1)) and np.array_equal(rec.T2, np.asarray(T2))
    assert rec.unit == "cm" and len(rec) == len(t)
    assert np.allclose(np.diff(rec.t), 1 / 120.0, rtol=1e-3), "arrival-time jitter is removed"
    assert abs((rec.t[-1] - rec.t[0]) - (t[-1] - t[0])) < 0.02, "the duration is preserved"


def test_discovery_lists_motion_recordings_only():
    found = rc.discover()
    names = [label for label, _ in found["polhemus"]]
    assert "1_90deg_ydata.csv" in names
    assert names.index("2_90deg_y_with_translationdata.csv") < names.index("10_Head_rotationdata.csv"), "natural order"
    assert not any("snapshot" in n.lower() for n in names), "single-pose snapshot files are not recordings"
    assert all(path.endswith(".c3d") for _, path in found["vicon"])


def test_simulated_recording_carries_its_true_axis():
    rec = rc.load_synthetic("none", "none")
    assert rec.truth is not None and rec.unit == "cm"
    fha = fc.compute_fha(rec.t, rec.T1, rec.T2)
    n, p = fc.axes_to_lab(fha, rec.T1)
    cosine = np.abs(np.einsum("ki,ki->k", n, rec.truth["n"][fha.ic]))
    assert cosine.min() > 1 - 1e-10


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {str(exc)[:400]}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
