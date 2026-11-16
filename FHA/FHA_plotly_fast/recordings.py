"""Load a recording — Polhemus CSV, Vicon C3D or the built-in simulator — into one
common structure the analysis and the dashboard work from."""
import contextlib
import glob
import io
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

import fha_core as fc
import synthetic as syn

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DATA = os.path.abspath(os.path.join(HERE, "..", "..", "test_data"))
POLHEMUS_COLUMNS = ("Time", "w1", "x1", "y1", "z1", "loc1_x", "loc1_y", "loc1_z",
                    "w2", "x2", "y2", "z2", "loc2_x", "loc2_y", "loc2_z")

SYNTHETIC_PROFILES = {
    "none": "Pure hinge",
    "proportional": "Hinge + steady second rotation",
    "late": "Hinge + second rotation phased in smoothly",
    "pulsed": "Hinge + second rotation in bursts",
}
SYNTHETIC_NOISE = {
    "none": "No measurement error",
    "polhemus": "Polhemus-like error",
    "vicon": "Vicon-like error",
}


@dataclass
class Recording:
    label: str
    source: str                     # "polhemus" | "vicon" | "synthetic"
    t: np.ndarray                   # (N,) seconds, evenly spaced where the device samples evenly
    T1: np.ndarray                  # (N, 4, 4) pose of the reference body in the lab
    T2: np.ndarray                  # (N, 4, 4) pose of the moving body in the lab
    unit: str                       # length unit of the positions
    truth: Optional[dict] = None    # simulator only: true axis per frame, in the lab frame
    quality: Optional[dict] = None  # Vicon only: rigid-fit residual per segment, per frame
    points: Optional[tuple] = None  # Vicon only: per body, where its markers sit in its own frame (m, 3)
    notes: list = field(default_factory=list)

    def __len__(self):
        return len(self.t)


def _poses(quat_wxyz, loc):
    """Poses from Polhemus quaternions (w, x, y, z) and positions."""
    T = np.tile(np.eye(4), (len(loc), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(np.roll(np.asarray(quat_wxyz, float), -1, axis=1)).as_matrix()
    T[:, :3, 3] = loc
    return T


def load_polhemus(path, unit="cm"):
    """Polhemus CSV with two sensors: timestamps, quaternions and positions.

    The file carries no length unit; ``unit`` is a label only ("cm" matches the
    dimensions of the bench rig in the test recordings).
    """
    import polhemus_import as pi

    q1, q2, loc1, loc2, t = pi.read_data(path)
    rec = Recording(os.path.basename(path), "polhemus", np.asarray(t, float),
                    _poses(q1, loc1), _poses(q2, loc2), unit)
    even, pauses = fc.even_time_base(rec.t)
    if not np.array_equal(even, rec.t):
        steps = np.diff(rec.t)
        note = (f"Timestamps are arrival times (intervals {steps.min() * 1e3:.2f} to {steps.max() * 1e3:.0f} ms); "
                f"replaced by an even {1 / np.median(np.diff(even)):.1f} Hz time base")
        if pauses:
            frame, lost = max(pauses, key=lambda p: p[1])
            note += (f", keeping {len(pauses)} pause{'s' if len(pauses) > 1 else ''} "
                     f"(the longest {lost:.1f} s, before frame {frame})")
        rec.notes.append(note + ".")
        rec.t = even
    elif np.ptp(np.diff(rec.t)) > 0.5 * np.median(np.diff(rec.t)):
        rec.notes.append("The timestamps are uneven and could not be read as arrival times of an evenly "
                         "sampling device, so they are used as they are. Check them before smoothing.")
    return rec


def load_vicon(path, markers_1, markers_2, frames="least_squares"):
    """Vicon C3D. ``markers_1`` / ``markers_2`` list the markers of the reference and the
    moving segment; the first three of each define the segment frame.

    ``frames="least_squares"`` fits one rigid body to all listed markers of a segment
    (see vicon_import.build_reference_frames_lsq); ``"three_marker"`` builds the frame
    directly from the first three markers, frame by frame, as the original code did.
    """
    import vicon_import as vi

    with contextlib.redirect_stdout(io.StringIO()):
        c3d, marker_data, t = vi.read_data(path)
    labels = list(c3d["parameters"]["POINT"]["LABELS"]["value"])
    missing = [m for m in list(markers_1) + list(markers_2) if m not in labels]
    if missing:
        raise ValueError(f"Markers not found in {os.path.basename(path)}: {', '.join(missing)}. "
                         f"Available: {', '.join(l for l in labels if not l.startswith('*'))}")
    units = c3d["parameters"]["POINT"]["UNITS"]["value"]
    unit = units[0] if len(units) else "mm"
    quality = None
    if frames == "least_squares":
        _, T1, _, T2, _, _, res = vi.build_reference_frames_lsq(marker_data, list(markers_1), list(markers_2))
        quality = dict(residual_1=res["residual_j1"], residual_2=res["residual_j2"])
    elif frames == "three_marker":
        _, T1, _, T2, _, _ = vi.build_reference_frames(marker_data, *list(markers_1)[:3], *list(markers_2)[:3])
    else:
        raise ValueError("frames must be 'least_squares' or 'three_marker'")
    if np.isnan(T1).any() or np.isnan(T2).any():
        raise ValueError("Some frames have missing markers; fill the gaps or use markers that are always visible.")
    rec = Recording(os.path.basename(path), "vicon", np.asarray(t, float), np.asarray(T1, float),
                    np.asarray(T2, float), unit, quality=quality)

    def on_segment(T, names):
        """The markers of a segment in that segment's own frame: the median over the recording."""
        inverse = fc.invert(np.asarray(T, float))
        local = np.einsum("nij,nmj->nmi", inverse[:, :3, :3], vi.marker_array(marker_data, names)) + inverse[:, None, :3, 3]
        return np.median(local, axis=0)

    rec.points = (on_segment(rec.T1, list(markers_1)), on_segment(rec.T2, list(markers_2)))
    if quality is not None:
        r2 = quality["residual_2"]
        rec.notes.append(f"Moving-segment marker fit: median residual {np.median(r2):.2f} {unit}, "
                         f"worst {r2.max():.2f} {unit}.")
    return rec


def load_synthetic(profile="proportional", noise="polhemus", secondary_deg=20.0, n_pulses=8,
                   return_trip=False, seed=0):
    """Simulated rig with a known true axis (see synthetic.two_axis_motion)."""
    rig = dict(fs=100.0, move_s=5.0, still_s=0.5) if noise == "vicon" else dict(fs=120.0, move_s=5.5, still_s=1.0)
    d = syn.two_axis_motion(profile=profile, noise=noise, secondary_deg=secondary_deg, n_pulses=n_pulses,
                            return_trip=return_trip, seed=seed, **rig)
    label = f"Simulated: {SYNTHETIC_PROFILES[profile].lower()}, {SYNTHETIC_NOISE[noise].lower()}"
    truth = dict(n=d["truth_n"], s=d["truth_s"], moving=d["moving"], alpha=d["alpha"], beta=d["beta"])
    return Recording(label, "synthetic", d["t"], d["T1"], d["T2"], "cm", truth=truth)


def is_polhemus_csv(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().strip().split(",")
            rows = sum(1 for _ in zip(range(60), fh))
    except OSError:
        return False
    return header[0] == "Time" and all(c in header for c in POLHEMUS_COLUMNS) and rows >= 50


def discover(root=TEST_DATA):
    """Recordings found under the test-data folder: {source: [(label, path), ...]}."""
    found = {"polhemus": [], "vicon": []}
    pol = os.path.join(root, "Polhemus_test_data")
    for path in sorted(glob.glob(os.path.join(pol, "**", "*.csv"), recursive=True),
                       key=lambda p: (os.path.dirname(p) != pol, _natural(os.path.relpath(p, pol)))):
        if is_polhemus_csv(path):
            found["polhemus"].append((os.path.relpath(path, pol), path))
    vic = os.path.join(root, "Vicon_test_data")
    for path in sorted(glob.glob(os.path.join(vic, "**", "*.c3d"), recursive=True)):
        found["vicon"].append((os.path.relpath(path, vic), path))
    return found


def _natural(name):
    """Sort key that orders '2_...' before '10_...'."""
    import re

    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]
