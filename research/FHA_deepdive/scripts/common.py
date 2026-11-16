"""Shared helpers for the FHA deep-dive scripts.

"Before" numbers come from `../original_snapshot/`, a verbatim copy of the pipeline as
it stood at commit 5b32fa5 (before any change made during this investigation). It is
loaded under the names `orig`, `orig_pi`, `orig_vi`. The improved code is imported
normally from FHA/FHA_plotly_fast (`fha_core`, `synthetic`, ...).
"""
import contextlib
import importlib.util
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FAST = os.path.join(REPO, "FHA", "FHA_plotly_fast")
SNAP = os.path.abspath(os.path.join(HERE, "..", "original_snapshot"))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
POLHEMUS = os.path.join(REPO, "test_data", "Polhemus_test_data")
VICON = os.path.join(REPO, "test_data", "Vicon_test_data")

os.makedirs(FIG, exist_ok=True)
if FAST not in sys.path:
    sys.path.insert(0, FAST)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SNAP, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


orig = _load("orig_FHA_visualizer_fast", "FHA_visualizer_fast.py")  # original pipeline
orig_pi = _load("orig_polhemus_import", "polhemus_import.py")
orig_vi = _load("orig_vicon_import", "vicon_import.py")

POLHEMUS_FILES = {
    "1_oneway": "1_90deg_ydata_oneway.csv",
    "1_full": "1_90deg_ydata.csv",
    "2_transl": "2_90deg_y_with_translationdata.csv",
    "3_y_plus_x": "3_90deg_y_with_xxdeg_x_rotationdata.csv",
    "4_y_plus_z": "4_90deg_y_with_xxdeg_z_rotationdata.csv",
    "5_by_sensor1": "5_90deg_y_by_sensor_1data.csv",
}

VICON_FILES = {
    "servo_X": "servo_knee_90degX.c3d",
    "servo_X_20Y": "servo_knee_90degX_and_20Y.c3d",
    "servo_X_20Z": "servo_knee_90degX_and_20Z.c3d",
}

VICON_MARKERS = dict(
    marker1_j1="lateral_epicondyl", marker2_j1="medial_epicondyl", marker3_j1="caput",
    marker1_j2="ankle", marker2_j2="lateral_condyl", marker3_j2="medial_condyl",
)


def load_polhemus(key):
    """Return t, T1, T2 (N,4,4 float arrays), loc1, loc2 for a Polhemus test file."""
    path = os.path.join(POLHEMUS, POLHEMUS_FILES.get(key, key))
    q1, q2, loc1, loc2, t = orig_pi.read_data(path)
    R1, T1, R2, T2 = orig_pi.quaternions_to_matrices(q1, q2, loc1, loc2)
    return np.asarray(t, float), np.asarray(T1, float), np.asarray(T2, float), loc1, loc2


def load_vicon(key):
    """Return t, T1, T2, loc1, loc2 for a Vicon test file (prints suppressed)."""
    path = os.path.join(VICON, VICON_FILES.get(key, key))
    with contextlib.redirect_stdout(io.StringIO()):
        c3d, marker_data, t = orig_vi.read_data(path)
        R1, T1, R2, T2, loc1, loc2 = orig_vi.build_reference_frames(marker_data, **VICON_MARKERS)
    return (np.asarray(t, float), np.asarray(T1, float), np.asarray(T2, float),
            np.asarray(loc1, float), np.asarray(loc2, float))


def relative_pose(T1, T2):
    """Pose of body 2 expressed in the frame of body 1."""
    return np.linalg.inv(T1) @ T2


def rotvec_deg(Rm):
    """Rotation vector(s) in degrees for a stack of rotation matrices."""
    from scipy.spatial.transform import Rotation

    return np.degrees(Rotation.from_matrix(Rm).as_rotvec())


def net_angle_deg(Ra, Rb):
    """Net rotation angle in degrees between two stacks of rotation matrices."""
    Rrel = np.einsum("nij,nkj->nik", Rb, Ra)  # Rb @ Ra^T
    c = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def run_original(t, T1, T2, loc1, loc2, cut1, cut2, step, method="incremental_angle", nn=1):
    """Run the ORIGINAL pipeline exactly as main.py did and unpack what it drew.

    Returns the cross-section coordinates recovered from the Plotly figure, plus the
    net rotation each window really spans and each axis in lab coordinates.
    """
    R1, R2 = T1[:, :3, :3], T2[:, :3, :3]
    (hax, ang, svec, d, tr1, tr2, time_diff, time_incr, ang_incr, all_angles,
     ind_incr, ind_step, tcut) = orig.generate_FHA(
        method, t, cut1, cut2, step, nn, R1, R2, T1, T2, np.asarray(loc1), np.asarray(loc2))
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        fig = orig.plotly_visualize_FHA(
            True, T1[cut1:cut2], T2[cut1:cut2], method, hax, svec, d, ind_incr, ind_step, nn,
            tr1, tr2, tcut, time_incr, ang_incr, step, all_angles, time_diff)
    printed = buf.getvalue().strip().splitlines()[-1]

    traces = {tr.name: tr for tr in fig.data}
    xyz = lambda tr: np.c_[tr.x, tr.y, tr.z].astype(float)
    inter = xyz(traces["Intersection Points"])
    aha = xyz(traces["Average Helical Axis"])
    origins = xyz(traces["FHA Origins"])
    normal = aha[1] - aha[0]
    normal /= np.linalg.norm(normal)
    mid = aha.mean(axis=0)
    # same in-plane basis as the original code
    arb = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    e1 = np.cross(normal, arb); e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1); e2 /= np.linalg.norm(e2)
    uv = np.c_[(inter - mid) @ e1, (inter - mid) @ e2]

    Trel = np.linalg.inv(T1[cut1:cut2]) @ T2[cut1:cut2]
    if method == "incremental_angle":
        i0 = np.array([p[0] for p in ind_incr]); i1 = np.array([p[1] for p in ind_incr])
    else:
        i0 = np.array(ind_step[:-1]); i1 = np.array(ind_step[1:])
    net = net_angle_deg(Trel[i0, :3, :3], Trel[i1, :3, :3])
    hax_g = np.einsum("nij,nj->ni", T1[cut1:cut2][i0, :3, :3], np.asarray(hax))
    dev = np.degrees(np.arccos(np.clip(hax_g @ normal, -1, 1)))
    return dict(uv=uv, t=np.asarray(time_incr), net=net, dev=dev, i0=i0, i1=i1, tcut=tcut,
                printed=printed, hax_g=hax_g, origins=origins, normal=normal, aha_start=aha[0],
                plane_point=mid, e1=e1, e2=e2, all_angles=all_angles)
