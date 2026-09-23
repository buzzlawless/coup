"""Up-closed convex sets in [0, 1]^n, the currency of the bluffing solver.

A set here is everything at or above some convex combination of a finite list
of points -- ``conv(V) + R^n_+`` -- and is stored as its lower extreme points
``V``.  In the bluffing game such a set is what the uninformed player can
*guarantee*: a vector with one entry per card the informed player might hold,
each an upper bound on that card's chance of winning.

Only three operations are needed, and ``support`` is how a belief turns a set
back into a single number.

Everything goes through Qhull.  To keep it well-posed, every set is completed
upward to a box ``[0, BOX]^n`` before a hull is taken: that makes each set
full-dimensional however degenerate its points are, and the box's corner gives
halfspace intersection a guaranteed interior point.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

#: Upper face of the working box.  Values live in [0, 1]; anything above is
#: only there to make the sets bounded and full-dimensional.
BOX = 1.5
EPS = 1e-9


def _dedupe(points: np.ndarray) -> np.ndarray:
    return np.unique(np.round(points, 12), axis=0)


def _raised(points: np.ndarray) -> np.ndarray:
    """Every point with every subset of its coordinates lifted to BOX."""
    n = points.shape[1]
    out = []
    for mask in itertools.product((False, True), repeat=n):
        q = points.copy()
        q[:, list(mask)] = BOX
        out.append(q)
    return np.vstack(out)


def prune(points) -> np.ndarray:
    """The lower extreme points of conv(points) + R^n_+."""
    pts = _dedupe(np.atleast_2d(np.asarray(points, dtype=float)))
    if len(pts) == 1:
        return pts
    # drop plainly dominated points first; it keeps the hull small
    keep = [i for i, p in enumerate(pts)
            if not any(j != i and np.all(pts[j] <= p + EPS) and np.any(pts[j] < p - EPS)
                       for j in range(len(pts)))]
    pts = pts[keep]
    if len(pts) == 1:
        return pts
    cloud = _raised(pts)
    try:
        hull = ConvexHull(cloud)
    except QhullError:
        hull = ConvexHull(cloud, qhull_options="QJ")
    extreme = set(hull.vertices.tolist())
    kept = [i for i in range(len(pts)) if i in extreme]   # the unlifted copies come first
    return pts[kept] if kept else pts


def union(sets) -> np.ndarray:
    """conv of the union: the other side choosing, possibly at random."""
    return prune(np.vstack(sets))


def intersect(sets) -> np.ndarray:
    """The intersection: the informed side choosing, with its choice observed."""
    sets = [np.atleast_2d(s) for s in sets]
    if len(sets) == 1:
        return sets[0]
    # a set of all of R^n_+ (its only point is 0) constrains nothing
    sets = [s for s in sets if not (len(s) == 1 and np.all(s[0] <= EPS))] or [sets[0]]
    if len(sets) == 1:
        return sets[0]
    if all(len(s) == 1 for s in sets):
        return np.max(np.vstack(sets), axis=0, keepdims=True)
    n = sets[0].shape[1]
    halfspaces = []
    for s in sets:
        cloud = _raised(s)
        try:
            hull = ConvexHull(cloud)
        except QhullError:
            hull = ConvexHull(cloud, qhull_options="QJ")
        halfspaces.append(hull.equations)
    hs = np.unique(np.round(np.vstack(halfspaces), 12), axis=0)
    inside = np.full(n, (1.0 + BOX) / 2)
    try:
        cut = HalfspaceIntersection(hs, inside)
    except QhullError:
        cut = HalfspaceIntersection(hs, inside, qhull_options="QJ")
    pts = cut.intersections
    pts = pts[np.all(pts <= 1.0 + 1e-7, axis=1)]
    pts = np.clip(pts, 0.0, 1.0)
    return prune(pts)


def support(points: np.ndarray, belief) -> float:
    """min over the set of belief . z: the value once a belief is fixed."""
    return float(np.min(np.atleast_2d(points) @ np.asarray(belief, dtype=float)))
