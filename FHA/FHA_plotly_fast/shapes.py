"""Basic triangle meshes that stand in for a rigid body until its real surface is known.

A mesh here is ``(vertices, faces)``: vertices (V, 3) in the coordinates of the body it
belongs to, faces (F, 3) as indices into them, wound counter-clockwise seen from
outside. A bone surface read from a file has the same form, so it can take the place of
these shapes without anything else changing: the dashboard only ever moves vertices
with the pose of their body.
"""
import numpy as np

# two triangles per face of a box whose corner number is 4*bx + 2*by + bz (b = 0 low, 1 high)
_BOX_FACES = np.array([
    (0, 1, 3), (0, 3, 2), (4, 7, 5), (4, 6, 7),      # low x, high x
    (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),      # low y, high y
    (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),      # low z, high z
])


def box(size, centre=(0.0, 0.0, 0.0), axes=None):
    """A box with edge lengths ``size`` along the three ``axes`` (rows; the body's own
    x, y, z when omitted), centred on ``centre``."""
    size, centre = np.asarray(size, float), np.asarray(centre, float)
    axes = np.eye(3) if axes is None else np.asarray(axes, float)
    corners = np.array([[bx, by, bz] for bx in (0, 1) for by in (0, 1) for bz in (0, 1)], float) - 0.5
    return centre + (corners * size) @ axes, _BOX_FACES.copy()


def bar(start, end, width, thickness):
    """A straight bar of rectangular section from ``start`` to ``end``.

    The section is rectangular, not round, so that a turn of the body about the bar's
    own length shows. Its wide side is laid along whichever of the body's x, y, z is
    most nearly square to the bar, which keeps its faces lined up with the body's axes.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = np.linalg.norm(end - start)
    if not length > 0:
        raise ValueError("A bar needs two different end points.")
    along = (end - start) / length
    hint = np.eye(3)[np.argmin(np.abs(along))]
    wide = hint - (hint @ along) * along
    wide /= np.linalg.norm(wide)
    return box((length, width, thickness), 0.5 * (start + end), np.array([along, wide, np.cross(along, wide)]))


def plate(corners, thickness):
    """A flat triangular plate through three points, ``thickness`` thick: the simplest
    solid that shows where three markers on a segment are and how the segment turns."""
    a, b, c = (np.asarray(p, float) for p in corners)
    normal = np.cross(b - a, c - a)
    size = np.linalg.norm(normal)
    if not size > 0:
        raise ValueError("The three points of a plate must not lie on one line.")
    half = 0.5 * thickness * normal / size
    vertices = np.array([a - half, b - half, c - half, a + half, b + half, c + half])
    faces = np.array([(3, 4, 5), (0, 2, 1),                       # top, bottom
                      (0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4), (2, 0, 3), (2, 3, 5)])   # the three sides
    return vertices, faces


def merge(*meshes):
    """Several meshes as one."""
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        vertices.append(np.asarray(v, float))
        faces.append(np.asarray(f, int) + offset)
        offset += len(v)
    if not vertices:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(vertices), np.vstack(faces)


def clip_box(vertices, faces, low, high):
    """The part of a mesh inside the box with corners ``low`` and ``high`` (sides along
    x, y, z). Triangles that cross a side are cut there; the cuts are left open."""
    vertices, low, high = np.asarray(vertices, float), np.asarray(low, float), np.asarray(high, float)
    if len(vertices) and np.all(vertices >= low) and np.all(vertices <= high):
        return vertices, np.asarray(faces, int)                    # all of it is inside: nothing to cut
    kept, triangles = [], []
    for triangle in np.asarray(faces, int):
        polygon = [vertices[i] for i in triangle]
        for axis in range(3):
            for bound, side in ((low[axis], 1.0), (high[axis], -1.0)):
                # one pass of Sutherland-Hodgman: keep what lies on the inner side of this face of the box
                inside = [side * (p[axis] - bound) >= 0 for p in polygon]
                cut = []
                for k, p in enumerate(polygon):
                    q, q_in = polygon[k - 1], inside[k - 1]
                    if inside[k] != q_in:
                        cut.append(q + (p - q) * (bound - q[axis]) / (p[axis] - q[axis]))
                    if inside[k]:
                        cut.append(p)
                polygon = cut
                if len(polygon) < 3:
                    break
            if len(polygon) < 3:
                break
        if len(polygon) >= 3:
            first = len(kept)
            kept.extend(polygon)
            triangles.extend((first, first + k, first + k + 1) for k in range(1, len(polygon) - 1))
    if not kept:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.array(kept), np.array(triangles, int)


def volume(vertices, faces):
    """Volume enclosed by a closed mesh; positive when the faces are wound outward."""
    a, b, c = (np.asarray(vertices, float)[np.asarray(faces, int)[:, k]] for k in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def move(vertices, pose):
    """Vertices given in a body's coordinates, placed by the pose (4x4) of that body."""
    pose = np.asarray(pose, float)
    return np.asarray(vertices, float) @ pose[:3, :3].T + pose[:3, 3]
