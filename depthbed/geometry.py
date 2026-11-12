"""Geometry: plane fitting, unprojection and the orthographic ray-cast.

Everything here is deliberately numpy-only so it can run on a Pi Zero as well
as on a desktop. The one expensive routine is ``ortho_map``; it is analytic
(a plane projects to a homography) so it costs a few array ops, not a
per-pixel loop.

Frames matter here. The plane is fitted in the depth camera frame, but the
pretty image comes from the RGB camera. Rather than hope the two agree, we
build the in-plane coordinate frame *once* (in the depth frame) and carry the
origin + basis vectors into the RGB frame explicitly. Doing it the naive way
-- calling ``basis()`` on each camera's copy of the plane -- silently offsets
the two views by the stereo baseline.
"""

from __future__ import annotations

import numpy as np

from .types import Intrinsics, Plane


def points_from_depth(depth: np.ndarray, intr: Intrinsics, stride: int = 1):
    """Unproject a depth image. Returns (points HxWx3, valid HxW)."""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride].astype(np.float64)
    z = depth[::stride, ::stride].astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    x = (xs - intr.cx) / intr.fx * z
    y = (ys - intr.cy) / intr.fy * z
    return np.stack([x, y, z], axis=-1), valid


def _plane_from_3(p0, p1, p2):
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n)
    if ln < 1e-9:
        return None
    n = n / ln
    return n, float(n @ p0)


def fit_plane_lsq(points: np.ndarray):
    """Total-least-squares plane through points (SVD)."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    d = float(n @ c)
    resid = float(np.median(np.abs(points @ n - d)))
    return Plane(n, d), resid


def orient_outward(plane: Plane, points: np.ndarray) -> Plane:
    """Make the normal point away from the camera origin, so that
    ``height()`` is positive for surfaces between camera and plane."""
    if float(np.mean(points @ plane.normal)) < 0:
        return Plane(-plane.normal, -plane.d)
    return plane


def fit_plane_ransac(points: np.ndarray, thresh: float = 0.002, iters: int = 400,
                     seed: int = 0, max_points: int = 20000):
    """RANSAC plane. Returns (Plane, inlier_mask, residual_median).

    ``residual_median`` is the workhorse diagnostic: if it is more than a
    couple of millimetres on a genuinely flat surface, the intrinsics you were
    handed do not describe this image.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 10:
        raise ValueError("not enough points for a plane fit")

    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        sample = points[rng.choice(len(points), max_points, replace=False)]
    else:
        sample = points

    best_n, best_d, best_count = None, 0.0, -1
    for _ in range(iters):
        s = sample[rng.choice(len(sample), 3, replace=False)]
        cand = _plane_from_3(*s)
        if cand is None:
            continue
        n, d = cand
        count = int(np.count_nonzero(np.abs(sample @ n - d) < thresh))
        if count > best_count:
            best_n, best_d, best_count = n, d, count

    if best_n is None:
        raise ValueError("RANSAC failed to find a plane")

    inl_sample = np.abs(sample @ best_n - best_d) < thresh
    plane, resid = fit_plane_lsq(sample[inl_sample])
    plane = orient_outward(plane, sample[inl_sample])
    full_inl = np.abs(points @ plane.normal - plane.d) < thresh
    return plane, full_inl, resid


def plane_frame(plane: Plane):
    """A stable in-plane frame anchored on the plane itself.

    Returns (origin, e1, e2) in the frame the plane was expressed in. ``e1``
    is roughly the camera's +X projected onto the plane, so the result reads
    like the picture you would take looking straight down.
    """
    e1, e2 = plane.basis()
    return plane.normal * plane.d, e1, e2


def transform_frame(R, t, origin, e1, e2):
    """Carry a plane frame into another camera frame (x' = R x + t)."""
    return R @ origin + t, R @ e1, R @ e2


def frame_coords(origin, e1, e2, points: np.ndarray):
    """(N,3) -> (N,2) in-plane metres, relative to ``origin``."""
    rel = np.asarray(points) - origin
    return np.stack([rel @ e1, rel @ e2], axis=-1)


def rgb_footprint(intr_rgb: Intrinsics, R_depth_to_rgb, plane: Plane,
                  origin, e1, e2):
    """The quadrilateral, in plane coordinates (mm), that the RGB frame sees.

    A plane maps to an image by a homography, so the image border maps to a
    convex quadrilateral on the plane. That is exactly the region a rectified
    photo can contain.

    Returns ``(4, 2)`` points in image-border order, or ``None`` if a corner
    ray misses the plane (which only happens when the plane is behind the
    camera or edge-on).
    """
    T = np.asarray(R_depth_to_rgb, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    # Do this in the RGB frame, from the RGB camera centre. Rays start at the
    # origin of *that* camera, which is 37 mm away from the depth camera's --
    # intersecting depth-frame rays from the depth origin puts the footprint
    # out by exactly that baseline.
    plane_rgb = plane.transform(R, t)
    o_a, e1_a, e2_a = transform_frame(R, t, origin, e1, e2)

    pix = np.array([[0.0, 0.0],
                    [intr_rgb.width - 1.0, 0.0],
                    [intr_rgb.width - 1.0, intr_rgb.height - 1.0],
                    [0.0, intr_rgb.height - 1.0]])
    xd = (pix[:, 0] - intr_rgb.cx) / intr_rgb.fx
    yd = (pix[:, 1] - intr_rgb.cy) / intr_rgb.fy
    xn, yn = intr_rgb.undistort(xd, yd)
    dirs = np.stack([xn, yn, np.ones_like(xn)], axis=1)

    denom = dirs @ plane_rgb.normal
    if np.any(denom <= 1e-6):
        return None
    s = plane_rgb.d / denom
    if np.any(s <= 0):
        return None

    P = dirs * s[:, None]
    return frame_coords(o_a, e1_a, e2_a, P) * 1000.0


def _half_planes(poly: np.ndarray):
    """``A @ p <= C`` for the interior of a convex polygon (any winding)."""
    p = np.asarray(poly, dtype=np.float64)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                        - p[:, 1] * np.roll(p[:, 0], -1))
    if area < 0:
        p = p[::-1].copy()                       # make it counter-clockwise
    A, C = [], []
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        # Interior is left of a->b:  -dy*x + dx*y > dy*ax - dx*ay, i.e.
        # dy*x - dx*y < dy*ax - dx*ay.
        A.append([dy, -dx])
        C.append(dy * a[0] - dx * a[1])
    return np.asarray(A), np.asarray(C)


def _largest_rect_inside(A: np.ndarray, C: np.ndarray,
                         bounds, steps: int = 48):
    """Largest axis-aligned rectangle inside ``A p <= C``.

    Why bother: a rectified photo of an oblique surface is a rectangle-shaped
    window onto a quadrilateral patch. Exporting that patch's bounding box
    leaves big black wedges in the corners. The largest rectangle that fits
    inside keeps only real pixels -- which is what "a picture" means. Solved
    by scanning the two x edges and tightening y from the half-planes; the
    constraint set is convex, so the inner problem is exact.
    """
    xmin, ymin, xmax, ymax = bounds          # same order as (lo, hi)
    if not (xmax > xmin and ymax > ymin):
        return None

    def y_range(x0, x1):
        lo, hi = -np.inf, np.inf
        for (a, b), c in zip(A, C):
            if b > 1e-12:
                hi = min(hi, (c - a * x0) / b, (c - a * x1) / b)
            elif b < -1e-12:
                lo = max(lo, (c - a * x0) / b, (c - a * x1) / b)
            else:
                if a * x0 > c + 1e-9 or a * x1 > c + 1e-9:
                    return 0.0, 0.0
        return lo, hi

    def scan(xs):
        best, best_area = None, 0.0
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                lo, hi = y_range(xs[i], xs[j])
                area = (xs[j] - xs[i]) * (hi - lo)
                if area > best_area:
                    best, best_area = (xs[i], xs[j], lo, hi), area
        return best

    xs = np.linspace(xmin, xmax, steps)
    best = scan(xs)
    if best is None:
        return None
    # one local refinement around the winner
    pad = (xmax - xmin) / (steps - 1)
    xs = np.linspace(max(xmin, best[0] - pad), min(xmax, best[1] + pad), steps)
    best = scan(xs) or best
    return (max(best[0], xmin), max(best[2], ymin),
            min(best[1], xmax), min(best[3], ymax))      # x0, y0, x1, y1


def ortho_map(origin, e1, e2, intr: Intrinsics, u0_mm: float, v0_mm: float,
              out_w: int, out_h: int, mm_per_px: float):
    """Build a remap that ray-casts a *plane* into an orthographic image.

    For every output pixel we compute the 3D point on the plane, project it
    through the real camera (including lens distortion), and store the source
    pixel. Because the target is a plane this is analytic — no iteration and
    no per-pixel depth.
    """
    origin = np.asarray(origin, dtype=np.float64)
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)

    step = mm_per_px / 1000.0
    us = u0_mm / 1000.0 + np.arange(out_w, dtype=np.float64) * step
    vs = v0_mm / 1000.0 + np.arange(out_h, dtype=np.float64) * step
    U, V = np.meshgrid(us, vs)

    P = origin[None, None, :] + U[..., None] * e1[None, None, :] + V[..., None] * e2[None, None, :]

    z = P[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xn = P[..., 0] / z
        yn = P[..., 1] / z
    xd, yd = intr.distort(xn, yn)
    map_x = (xd * intr.fx + intr.cx).astype(np.float32)
    map_y = (yd * intr.fy + intr.cy).astype(np.float32)

    bad = (~np.isfinite(map_x)) | (~np.isfinite(map_y)) | (z <= 0)
    map_x[bad] = -1e6
    map_y[bad] = -1e6
    return map_x, map_y
