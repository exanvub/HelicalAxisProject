"""Playback: the body shapes, what is sent to the browser, and the playhead in the figures.

The part that runs in the browser (assets/playback.js) is checked in a real browser by
tests/test_playback_browser.py; here everything it is given is checked against numpy.

    python tests/test_playback.py
    pytest tests/test_playback.py
"""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import FHA_visualizer_fast as fv  # noqa: E402
import fha_core as fc  # noqa: E402
import recordings as rc  # noqa: E402
import shapes as sh  # noqa: E402

VICON = os.path.join(rc.TEST_DATA, "Vicon_test_data", "servo_knee_90degX_and_20Y.c3d")
MARKERS = (["lateral_epicondyl", "medial_epicondyl", "caput"], ["ankle", "lateral_condyl", "medial_condyl"])


def everywhere(body):
    """The vertices of a body at every playhead position: (K, V, 3)."""
    pose = body["pose"]
    return np.einsum("kij,vj->kvi", pose[:, :3, :3], body["vertices"]) + pose[:, None, :3, 3]


def outward(vertices, faces):
    """True if every face of a convex solid is wound so that its normal points away from the middle."""
    middle = vertices.mean(axis=0)
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    return bool(np.all(np.einsum("ij,ij->i", np.cross(b - a, c - a), (a + b + c) / 3 - middle) > 0))


# ------------------------------------------------------------------------------ shapes

def test_basic_shapes_are_closed_solids_of_the_right_size():
    v, f = sh.box((2, 3, 4), centre=(1, 1, 1))
    assert v.shape == (8, 3) and f.shape == (12, 3)
    assert abs(sh.volume(v, f) - 24) < 1e-12 and outward(v, f)
    assert np.allclose(v.min(axis=0), [0, -0.5, -1]) and np.allclose(v.max(axis=0), [2, 2.5, 3])

    v, f = sh.bar((1, 0, 0), (1, 5, 0), width=2, thickness=1)
    assert abs(sh.volume(v, f) - 10) < 1e-12 and outward(v, f)
    assert np.allclose(v[:, 1].min(), 0) and np.allclose(v[:, 1].max(), 5), "it runs from one end point to the other"
    assert np.allclose(np.ptp(v, axis=0), [1, 5, 2]) or np.allclose(np.ptp(v, axis=0), [2, 5, 1])
    slanted = sh.bar((0, 0, 0), (3, 4, 12), 0.5, 0.25)
    assert abs(sh.volume(*slanted) - 13 * 0.5 * 0.25) < 1e-12 and outward(*slanted)

    v, f = sh.plate([[0, 0, 0], [4, 0, 0], [0, 3, 0]], thickness=0.5)
    assert abs(sh.volume(v, f) - 3.0) < 1e-12 and outward(v, f)
    assert np.allclose(sorted(set(np.round(v[:, 2], 12))), [-0.25, 0.25]), "the three points lie in its middle plane"
    assert sh.volume(*sh.plate([[0, 0, 0], [0, 3, 0], [4, 0, 0]], 0.5)) > 0, "whichever way round the points are given"

    both = sh.merge(sh.box((1, 1, 1)), sh.box((2, 2, 2), centre=(5, 0, 0)))
    assert both[0].shape == (16, 3) and both[1].max() == 15 and abs(sh.volume(*both) - 9) < 1e-12
    assert sh.merge()[0].shape == (0, 3)


def test_moving_a_shape_is_the_pose_applied_to_its_vertices():
    v, _ = sh.box((1, 2, 3))
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [20, -35, 70], degrees=True).as_matrix()
    pose[:3, 3] = [4, 5, 6]
    moved = sh.move(v, pose)
    assert np.allclose(moved, (pose @ np.c_[v, np.ones(len(v))].T).T[:, :3])
    assert np.allclose(sh.move(moved, np.linalg.inv(pose)), v)


def test_cutting_a_shape_to_a_box_keeps_exactly_what_is_inside():
    v, f = sh.box((2, 2, 2))                              # from -1 to 1
    area = lambda v, f: sum(np.linalg.norm(np.cross(v[b] - v[a], v[c] - v[a])) / 2 for a, b, c in f)
    cv, cf = sh.clip_box(v, f, (-0.5, -5, -5), (5, 5, 5))
    assert abs(cv[:, 0].min() + 0.5) < 1e-12 and abs(cv[:, 0].max() - 1) < 1e-12
    assert abs(area(cv, cf) - 16.0) < 1e-12, "one whole face and three quarters of four others; the cut stays open"
    whole = sh.clip_box(v, f, (-5, -5, -5), (5, 5, 5))
    assert np.array_equal(whole[0], v) and np.array_equal(whole[1], f), "nothing to cut: the same mesh comes back"
    assert sh.clip_box(v, f, (2, 2, 2), (3, 3, 3))[0].shape == (0, 3), "all outside"


# ---------------------------------------------------------------------- playback data

def test_playhead_positions_cover_the_selected_frames_and_no_others():
    rec = rc.load_synthetic("proportional", "polhemus")
    an = fv.analyse(rec, (40, 700))
    p = an.play
    assert p["index"][0] == 0 and p["index"][-1] == 659 and np.all(np.diff(p["index"]) > 0)
    assert len(p["index"]) == 660, "fewer frames than the limit: every frame is a position"
    assert np.array_equal(p["frame"], p["index"] + 40) and np.allclose(p["t"], rec.t[40:700])
    assert abs(p["rate"] - 120.0) < 1e-6, "positions per second of recording"
    assert np.allclose(p["rotation"], an.rotation)

    long = rc.Recording("long", "synthetic", np.arange(5000) / 100.0, np.tile(rec.T1[:1], (5000, 1, 1)),
                        rec.T2[np.minimum(np.arange(5000) // 6, len(rec) - 1)], "cm")
    p = fv.analyse(long).play
    assert len(p["index"]) == fv.PLAY_STEPS and p["index"][-1] == 4999
    assert abs(p["rate"] - 100.0 * (fv.PLAY_STEPS - 1) / 4999) < 1e-6


def test_bodies_are_placed_by_the_poses_the_axes_were_computed_from():
    rec = rc.load_synthetic("pulsed", "polhemus")
    lab = fv.analyse(rec, frame="lab").play
    assert [b["key"] for b in lab["bodies"]] == ["body1", "body2"]
    assert np.allclose(lab["bodies"][0]["pose"], rec.T1[lab["index"]]) and np.allclose(lab["bodies"][1]["pose"], rec.T2[lab["index"]])
    body = fv.analyse(rec, frame="body").play
    assert np.allclose(body["bodies"][0]["pose"], np.eye(4)), "in the frame of the reference body, that body stands still"
    assert np.allclose(body["bodies"][1]["pose"], fc.relative_poses(rec.T1, rec.T2)[body["index"]])
    smooth = fv.analyse(rec, frame="lab", smoothing_hz=2.0).play
    T1s, T2s = fc.smooth_motion(rec.t, rec.T1, rec.T2, 2.0)
    assert np.allclose(smooth["bodies"][1]["pose"], T2s[smooth["index"]]), "with smoothing on, the smoothed poses"


def test_a_segment_runs_from_the_joint_to_its_sensor():
    rec = rc.load_synthetic("none", "none")                       # a pure hinge, no error
    an = fv.analyse(rec)
    for body in an.play["bodies"]:
        where = everywhere(body)
        # some vertex of the bar is always at the joint, and the block is always on the sensor
        to_joint = np.linalg.norm(where - an.pivot, axis=2).min(axis=1)
        to_sensor = np.linalg.norm(where - body["pose"][:, None, :3, 3], axis=2).min(axis=1)
        size = fv._scene_scale(an)["arm"]
        assert to_joint.max() < 0.08 * size, (body["key"], to_joint.max(), size)
        assert to_sensor.max() < 0.12 * size
        assert sh.volume(body["vertices"], body["faces"]) > 0
    blocks = fv.analyse(rec, bodies="blocks").play["bodies"]
    assert all(len(b["vertices"]) == 8 for b in blocks), "blocks only: one block per sensor"
    assert fv.analyse(rec, bodies="none").play["bodies"] == []


def test_the_axis_at_the_playhead_is_the_one_whose_window_holds_that_frame():
    rec = rc.load_synthetic("late", "polhemus")
    for method, step in (("incremental_angle", 10), ("step_angle", 10), ("incremental_time", 40), ("reference_pose", 10)):
        an = fv.analyse(rec, method=method, step=step)
        p, f = an.play, an.fha
        held = p["axis_index"] >= 0
        assert 0.3 < held.mean() <= 1.0, (method, held.mean())      # a fixed time lag has a window at every frame
        if method in ("incremental_angle", "reference_pose"):
            assert not held[0], "at rest before the movement there is no axis"
        a = p["axis_index"][held]
        if method == "reference_pose":
            assert np.all(np.abs(f.ic[a] - p["index"][held]) <= 1)
        else:
            assert np.all((f.i0[a] <= p["index"][held]) & (p["index"][held] <= f.i1[a])), method
            # and no other window holding the frame has its centre nearer
            k = np.flatnonzero(held)[len(a) // 2]
            holding = np.flatnonzero((f.i0 <= p["index"][k]) & (p["index"][k] <= f.i1))
            assert abs(f.ic[p["axis_index"][k]] - p["index"][k]) == np.abs(f.ic[holding] - p["index"][k]).min()
        if method != "reference_pose":
            any_window = np.array([np.any((f.i0 <= i) & (i <= f.i1)) for i in p["index"]])
            assert np.array_equal(held, any_window), f"{method}: a frame inside a window has an axis, and only such a frame"
        # the two ends drawn lie on that axis, and the map points are its points
        ends = p["axis"][held]
        along = ends[:, 1] - ends[:, 0]
        assert np.allclose(np.cross(along, an.n[a]), 0, atol=1e-9 * np.linalg.norm(along, axis=1).max())
        off_axis = np.cross(ends[:, 0] - an.p[a], an.n[a])
        assert np.abs(off_axis).max() < 1e-9 * max(1.0, np.abs(an.p).max())
        assert np.allclose(p["tilt"][held], an.tilt[a]) and np.allclose(p["pos"][held], an.pos[a])
        assert np.isnan(p["axis"][~held]).all() and np.isnan(p["tilt"][~held]).all()
        assert np.allclose(p["build"][:, 1], an.build["magnitude"][p["index"]])
        assert np.allclose(p["swing"], an.build["swing"][p["index"]])


def test_the_box_of_the_3d_view_holds_everything_at_every_instant():
    for rec in (rc.load_synthetic("pulsed", "vicon", return_trip=True), rc.load_synthetic("none", "polhemus")):
        for frame in ("lab", "body"):
            an = fv.analyse(rec, frame=frame)
            low, high = an.play["box"]
            for body in an.play["bodies"]:
                where = everywhere(body).reshape(-1, 3)
                assert np.all(where >= low - 1e-9) and np.all(where <= high + 1e-9), body["key"]
            ends = an.play["axis"][np.isfinite(an.play["axis"]).all(axis=(1, 2))].reshape(-1, 3)
            assert np.all(ends >= low) and np.all(ends <= high)
            assert np.all(an.path2 >= low) and np.all(an.path2 <= high)
            ranges = fv.figure_scene(an, step=0).layout.scene
            assert np.allclose([ranges.xaxis.range, ranges.yaxis.range, ranges.zaxis.range], np.c_[low, high])


def test_a_reference_body_far_from_the_joint_is_cut_off_instead_of_shrinking_the_view():
    rec = rc.load_synthetic("proportional", "polhemus")
    near = fv.analyse(rec).play
    T1 = rec.T1.copy()
    T1[:, :3, 3] += [900.0, 0.0, 0.0]                             # the reference sensor is across the room
    far_rec = rc.Recording("far", "synthetic", rec.t, T1, rec.T2, "cm")
    an = fv.analyse(far_rec, frame="lab")
    far = an.play
    size = lambda p: float((p["box"][1] - p["box"][0]).max())
    assert size(far) < 3.3 * size(near), (size(far), size(near))
    assert [b["key"] for b in far["bodies"]] == ["body1", "body2"], "its bar still shows, coming in from outside"
    where = everywhere(far["bodies"][0]).reshape(-1, 3)
    assert np.all(where >= far["box"][0] - 1e-6) and np.all(where <= far["box"][1] + 1e-6)
    assert np.linalg.norm(where - an.pivot, axis=1).min() < 0.1 * size(near), "and still reaches the joint"
    assert np.abs(where[:, 0]).max() < 400, "but not the sensor"


def test_marker_clusters_are_drawn_as_a_plate_through_their_markers():
    try:
        import ezc3d  # noqa: F401
    except ImportError:
        print("      (skipped: ezc3d not installed)")
        return
    rec = rc.load_vicon(VICON, *MARKERS)
    assert rec.points[0].shape == (3, 3) and rec.points[1].shape == (3, 3)
    # the frame convention puts the origin midway between markers 2 and 3, in the plane of the three
    for points in rec.points:
        assert np.allclose(points[1] + points[2], 0, atol=0.5) and np.abs(points[:, 2]).max() < 0.5       # mm
    an = fv.analyse(rec, (200, 760))
    moving = [b for b in an.play["bodies"] if b["key"] == "body2"][0]
    extent = max(np.linalg.norm(rec.points[1][i] - rec.points[1][j]) for i, j in ((0, 1), (1, 2), (0, 2)))
    for marker in rec.points[1]:
        near = np.linalg.norm(moving["vertices"] - marker, axis=1)
        assert near.min() < 0.07 * extent, "plate and block sit on the marker"
    assert np.all(np.abs(moving["vertices"][:, 2]) < 0.05 * extent), "a flat cluster gives a flat body"
    scale = fv._scene_scale(an)
    assert scale["length"] >= 0.8 * extent - 1e-9, "axes are drawn long enough to show beside the cluster"
    # the reference cluster sits far from this joint: what is drawn of it stays inside the box
    low, high = an.play["box"]
    for body in an.play["bodies"]:
        where = everywhere(body).reshape(-1, 3)
        assert np.all(where >= low - 1e-6) and np.all(where <= high + 1e-6)


def test_nothing_breaks_when_there_are_no_axes():
    rec = rc.load_synthetic("proportional", "polhemus")
    an = fv.analyse(rec, (0, 60))                                  # the rig is still at rest
    assert an.empty and len(an.play["index"]) == 60
    assert np.isnan(an.play["axis"]).all() and all(len(b["vertices"]) == 8 for b in an.play["bodies"])
    store = fv.playback_store(an)
    assert store["n"] == 60 and store["axis"][0] == [None] * 6


def test_few_frames_as_in_dynamic_ct_play_frame_by_frame():
    full = rc.load_synthetic("late", "polhemus")
    keep = slice(None, None, 30)
    slow = rc.Recording("every 30th frame", "synthetic", full.t[keep], full.T1[keep], full.T2[keep], "cm")
    p = fv.analyse(slow).play
    assert len(p["index"]) == len(slow) and abs(p["rate"] - 4.0) < 1e-6
    assert (p["axis_index"] >= 0).sum() > 5


# ------------------------------------------------------------- what the browser is sent

def rotation_from_quaternion(q):
    """The formula assets/playback.js uses, row by row."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_the_store_is_plain_json_and_reproduces_the_poses():
    rec = rc.load_synthetic("pulsed", "polhemus")
    an = fv.analyse(rec)
    store = fv.playback_store(an)
    text = json.dumps(store, allow_nan=False)                     # no NaN, no numpy types
    assert len(text) < 1_500_000
    back = json.loads(text)
    n = back["n"]
    assert n == len(an.play["index"]) and len(back["t"]) == n and len(back["frame"]) == n and len(back["axis"]) == n
    assert set(back["points"]) == {"fig-direction", "fig-position", "fig-buildup", "fig-swing"}
    assert back["lines"] == ["fig-series", "fig-removed", "fig-overview"]
    assert all(len(v) == n for v in back["points"].values())
    assert back["axis"][0] == [None] * 6 and back["points"]["fig-direction"][0] == [None, None]
    for sent, body in zip(back["bodies"], an.play["bodies"]):
        assert sent["meta"] == f"playback:{body['key']}"
        q, p, v = np.array(sent["q"]), np.array(sent["p"]), np.array(sent["v"])
        assert np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-6)
        for k in (0, n // 3, n - 1):
            placed = v @ rotation_from_quaternion(q[k]).T + p[k]
            assert np.abs(placed - sh.move(body["vertices"], body["pose"][k])).max() < 2e-4, (body["key"], k)
    k = n // 2
    assert np.allclose(np.array(back["axis"][k]).reshape(2, 3), an.play["axis"][k], atol=1e-4)


# ----------------------------------------------------------- the playhead in the figures

def playhead(fig):
    shapes = [s for s in fig.layout.shapes if s.name == "playhead"]
    assert len(shapes) <= 1
    return shapes[0] if shapes else None


def test_every_figure_shows_the_playhead_where_the_step_puts_it():
    rec = rc.load_synthetic("pulsed", "polhemus")
    an = fv.analyse(rec, smoothing_hz=6.0)
    p = an.play
    k = len(p["index"]) // 2
    assert p["axis_index"][k] >= 0

    scene = fv.figure_scene(an, step=k)
    by_meta = {t.meta: t for t in scene.data if t.meta}
    assert set(by_meta) == {"playback:body1", "playback:body2", "playback:axis"}
    for body in p["bodies"]:
        t = by_meta[f"playback:{body['key']}"]
        assert np.allclose(np.c_[t.x, t.y, t.z], sh.move(body["vertices"], body["pose"][k]))
        assert np.array_equal(np.c_[t.i, t.j, t.k], body["faces"])
    axis = by_meta["playback:axis"]
    assert np.allclose(np.c_[axis.x, axis.y, axis.z], p["axis"][k])
    names = [t.name for t in scene.data if t.name and t.showlegend is not False]
    assert "Moving body" in names and "Reference body" in names and "Axis at the playhead" in names

    for figure, key in ((fv.figure_direction_map, "tilt"), (fv.figure_position_map, "pos"),
                        (fv.figure_buildup, "build"), (fv.figure_swing_map, "swing")):
        ring = playhead(figure(an, step=k))
        assert ring.type == "circle" and ring.xsizemode == "pixel" and ring.visible is True
        assert np.allclose([ring.xanchor, ring.yanchor], p[key][k]), key
    hidden = playhead(fv.figure_direction_map(an, step=0))
    assert hidden.visible is False, "no axis at rest: the ring is there but hidden"
    assert playhead(fv.figure_buildup(an, step=0)).visible is True, "the build-up has a value at every frame"

    for fig in (fv.figure_series(an, step=k), fv.figure_smoothing_removed(an, step=k),
                fv.figure_overview(rec, an.cut, playhead=p["t"][k])):
        line = playhead(fig)
        assert line.type == "line" and line.yref == "paper" and (line.y0, line.y1) == (0, 1)
        assert abs(line.x0 - p["t"][k]) < 1e-9 and line.x0 == line.x1
    series = fv.figure_series(an, step=k)
    assert series.layout.xaxis.range[0] <= an.t[0] and series.layout.xaxis.range[1] >= an.t[-1], \
        "the time axis spans the selected frames, so the playhead never leaves it"


def test_without_a_step_the_figures_are_as_before():
    rec = rc.load_synthetic("pulsed", "polhemus")
    an = fv.analyse(rec)
    scene = fv.figure_scene(an)
    assert not [t for t in scene.data if t.meta] and scene.layout.scene.xaxis.range is None
    assert "Moving sensor" in [t.name for t in scene.data]
    for fig in (fv.figure_direction_map(an), fv.figure_buildup(an), fv.figure_series(an), fv.figure_overview(rec, an.cut)):
        assert playhead(fig) is None
    hidden = fv.figure_scene(fv.analyse(rec, bodies="none"), step=3)
    assert [t.meta for t in hidden.data if t.meta] == ["playback:axis"], "no bodies, but the axis of the instant"
    assert "Moving sensor" in [t.name for t in hidden.data]


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - report every failure, then exit non-zero
            failed += 1
            import traceback
            print(f"FAIL  {name}: {type(exc).__name__}: {str(exc)[:500]}")
            traceback.print_exc(limit=4)
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
