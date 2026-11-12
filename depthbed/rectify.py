"""Depth + colour -> an orthographic photo and a height field, in millimetres.

No segmentation, no "find the part". The only thing chosen here is the
reference plane, and it is reported and can be overridden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry as geo
from .types import Capture, Plane


@dataclass
class Rectified:
    rgb: np.ndarray                  # orthographic colour photo
    height_mm: np.ndarray            # float32, height above the reference plane
    mm_per_px: float
    u0_mm: float                     # plane coordinate of pixel (0, 0)
    v0_mm: float
    plane: Plane                     # reference plane, depth camera frame
    stats: dict = field(default_factory=dict)

    @property
    def extent_mm(self):
        """Photo size in millimetres, pixel centre to pixel centre."""
        h, w = self.height_mm.shape
        return (w - 1) * self.mm_per_px, (h - 1) * self.mm_per_px


def auto_plane(cap: Capture, thresh: float | None = None, seed: int = 0):
    """The dominant flat surface. Reported in stats, never trusted blindly.

    ``thresh`` is in metres and scales with working range when not given.
    Stereo depth error grows as z^2, so a fixed 4 mm that is right at 350 mm
    rejects almost the whole wall at 1.2 m -- and then the fit is just noise.
    """
    depth = np.asarray(cap.depth, dtype=np.float64)
    depth[depth <= 0] = np.nan
    pts, valid = geo.points_from_depth(depth, cap.intr_depth)
    if valid.sum() < 500:
        raise ValueError("not enough valid depth to find a plane")
    if thresh is None:
        z = float(np.nanmedian(depth))
        thresh = float(np.clip(0.004 * (z / 0.35) ** 2, 0.003, 0.020))
    plane, inliers, resid = geo.fit_plane_ransac(pts[valid], thresh=thresh,
                                                iters=500, seed=seed)
    return plane, float(inliers.mean()), float(resid)


def _canvas_extent(uv_mm: np.ndarray, margin_mm: float):
    lo = np.percentile(uv_mm, 0.5, axis=0)
    hi = np.percentile(uv_mm, 99.5, axis=0)
    return lo - margin_mm, hi + margin_mm


def _scene_band_m(plane_distance_m: float) -> float:
    """How far from the reference plane a point can be and still be "the
    scene", rather than the room behind it or a hand in front of it."""
    return max(0.08, 0.30 * abs(plane_distance_m))


def rectify(cap: Capture, plane: Plane | None = None, mm_per_px: float = 0.1,
            ref_height_mm: float = 0.0, margin_mm: float = 5.0,
            max_px: int = 4000) -> Rectified:
    """Rectify onto ``plane``.

    ``ref_height_mm`` is the surface that comes out metrically exact: leave it
    at 0 to make the base sheet true. Everything not at that height is
    displaced by its own height -- which is why the height field is returned
    alongside, rather than hidden inside a segmentation step.
    """
    depth = np.asarray(cap.depth, dtype=np.float64)
    depth[depth <= 0] = np.nan

    stats: dict = {}
    if plane is None:
        plane, frac, resid = auto_plane(cap)
        stats["plane_inlier_frac"] = round(frac, 3)
        stats["plane_residual_mm"] = round(resid * 1000, 3)
    else:
        pts, valid = geo.points_from_depth(depth, cap.intr_depth)
        res = np.abs(pts[valid] @ plane.normal - plane.d)
        stats["plane_inlier_frac"] = round(float((res < 0.004).mean()), 3)
        stats["plane_residual_mm"] = round(float(np.median(res)) * 1000, 3)

    pts, valid = geo.points_from_depth(depth, cap.intr_depth)
    if valid.sum() < 500:
        raise ValueError("not enough valid depth")

    origin, e1, e2 = geo.plane_frame(plane)
    h_m = plane.height(pts)
    near = valid & (np.abs(h_m) <= _scene_band_m(plane.d))
    if near.sum() < 500:
        raise ValueError("the reference plane does not contain enough of the "
                         "scene to rectify")

    uv = geo.frame_coords(origin, e1, e2, pts[near]) * 1000.0     # mm
    h_mm = h_m[near] * 1000.0
    stats["range_mm"] = round(float(np.nanmedian(depth[near])) * 1000, 1)

    # Canvas = where depth saw the plane, clipped to what the RGB frame can
    # show, then the largest axis-aligned rectangle inside the overlap. Sizing
    # this from *all* valid depth turns a wall into a 44 x 18 m sheet: a few
    # far returns stretch the percentiles by three orders of magnitude.
    lo, hi = _canvas_extent(uv, margin_mm)
    bounds = np.array([lo[0], lo[1], hi[0], hi[1]])
    data_rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                          [hi[0], hi[1]], [lo[0], hi[1]]])
    A, C = geo._half_planes(data_rect)

    fp = geo.rgb_footprint(cap.intr_rgb, cap.T_depth_to_rgb, plane,
                           origin, e1, e2)
    if fp is not None:
        fa, fc = geo._half_planes(fp)
        A = np.vstack([A, fa])
        C = np.concatenate([C, fc])
        ilo = np.maximum(lo, fp.min(axis=0))
        ihi = np.minimum(hi, fp.max(axis=0))
        if np.all(ihi - ilo > 10.0):
            bounds = np.array([ilo[0], ilo[1], ihi[0], ihi[1]])

    inner = geo._largest_rect_inside(A, C, bounds)
    if inner is not None and inner[2] - inner[0] > 20 and inner[3] - inner[1] > 20:
        lo = np.array([inner[0], inner[1]])
        hi = np.array([inner[2], inner[3]])
    else:
        lo = np.array([bounds[0], bounds[1]])
        hi = np.array([bounds[2], bounds[3]])

    field = hi - lo
    # A nearly edge-on plane has a footprint that runs to the horizon; the
    # "photo" would be a couple of pixels of content in a metres-wide canvas.
    if float(np.max(field)) > 10.0 * max(float(plane.d), 1e-6) * 1000.0:
        raise ValueError(
            "the dominant surface is seen too edge-on to rectify; aim more "
            f"squarely at it (field would be {field[0]:.0f} x {field[1]:.0f} mm "
            f"at {plane.d * 1000:.0f} mm)")

    scale = float(mm_per_px)
    out_w = int(np.ceil(field[0] / scale)) + 1
    out_h = int(np.ceil(field[1] / scale)) + 1
    if max(out_w, out_h) > max_px:
        scale *= max(out_w, out_h) / max_px
        out_w = int(np.ceil(field[0] / scale)) + 1
        out_h = int(np.ceil(field[1] / scale)) + 1
        stats["scale_limited"] = True

    u0, v0 = float(lo[0]), float(lo[1])
    stats["mm_per_px"] = round(scale, 4)
    stats["ortho_size"] = [out_w, out_h]
    stats["field_mm"] = [round(float(field[0]), 1), round(float(field[1]), 1)]

    # --- colour: ray-cast the plane at the requested reference height ---
    T = np.asarray(cap.T_depth_to_rgb, dtype=np.float64)
    R, tr = T[:3, :3], T[:3, 3]
    o_a, e1_a, e2_a = geo.transform_frame(R, tr, origin, e1, e2)
    # Raising the projection plane by ``ref_height_mm`` means moving its
    # origin towards the camera, i.e. against the outward normal.
    o_a = o_a - (R @ plane.normal) * (ref_height_mm / 1000.0)

    mx, my = geo.ortho_map(o_a, e1_a, e2_a, cap.intr_rgb, u0, v0,
                           out_w, out_h, scale)
    rgb = cv2.remap(cap.rgb, mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # --- height: scatter the 3D surface into a grid, from above ---
    # Not a warp: a warp would project each point down along the camera ray,
    # displacing heights sideways by h*tan(theta). Scattering from above, the
    # tallest surface wins each cell, which is what an orthographic view sees.
    #
    # The grid is built at the depth sampling density and then upsampled; a
    # 0.1 mm grid fed by 0.44 mm depth samples would be 80% interpolation.
    sample_mm = max(stats["range_mm"] / cap.intr_depth.fx, mm_per_px)
    # Size the coarse grid so upscaling to the photo grid is an exact factor:
    # cv2.resize maps w -> W, so W/w must equal sample_mm/scale. Rounding with
    # ceil instead stretches the field by ~1% (a 50 mm level set read 49.4).
    h_w = max(2, int(round(out_w * scale / sample_mm)))
    h_h = max(2, int(round(out_h * scale / sample_mm)))

    iu = np.rint((uv[:, 0] - u0) / sample_mm).astype(np.int64)
    iv = np.rint((uv[:, 1] - v0) / sample_mm).astype(np.int64)
    inside = (iu >= 0) & (iu < h_w) & (iv >= 0) & (iv < h_h)
    cells = iv[inside] * h_w + iu[inside]
    hh = h_mm[inside]

    # Per cell: mean of the samples within BAND_MM of the highest sample in
    # that cell. Plain max is occlusion-correct at a step but reads ~0.25 mm
    # high on a flat sheet; plain mean smears the step. This does neither.
    BAND_MM = 1.0
    cell_max = np.full(h_h * h_w, -np.inf, dtype=np.float64)
    np.maximum.at(cell_max, cells, hh)
    keep = hh >= cell_max[cells] - BAND_MM
    totals = np.zeros(h_h * h_w, dtype=np.float64)
    counts = np.zeros(h_h * h_w, dtype=np.float64)
    np.add.at(totals, cells[keep], hh[keep])
    np.add.at(counts, cells[keep], 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        coarse = np.where(counts > 0, totals / counts, np.nan).reshape(h_h, h_w)

    seen_coarse = np.isfinite(coarse)
    # Fill gaps only near measured data (a cell or two). Anything further is
    # invention, and a measuring tool must not do it.
    coarse = _fill_nearest(coarse, max_cells=3)

    seen = cv2.resize(seen_coarse.astype(np.uint8), (out_w, out_h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    grid = cv2.resize(np.nan_to_num(coarse, nan=0.0), (out_w, out_h),
                      interpolation=cv2.INTER_LINEAR)
    grid[~seen] = np.nan

    stats["height_mm_per_px"] = round(float(sample_mm), 4)
    stats["measured_frac"] = round(float(seen_coarse.mean()), 3)

    return Rectified(rgb=rgb, height_mm=grid.astype(np.float32),
                     mm_per_px=scale, u0_mm=u0, v0_mm=v0, plane=plane,
                     stats=stats)


def _fill_nearest(grid: np.ndarray, max_cells: float = 3.0) -> np.ndarray:
    """Nearest-neighbour fill of small gaps. Beyond ``max_cells`` from any
    measurement the cell stays NaN, because guessing there would be fiction."""
    holes = ~np.isfinite(grid)
    if not holes.any():
        return grid
    if holes.all():
        return np.full_like(grid, np.nan)
    dist, labels = cv2.distanceTransformWithLabels(
        holes.astype(np.uint8), cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(labels.max() + 1, dtype=np.float64)
    flat = grid.ravel()
    lab = labels.ravel()
    good = ~holes.ravel()
    lut[lab[good]] = flat[good]
    filled = lut[lab].reshape(grid.shape)
    filled[(holes & (dist > max_cells))] = np.nan
    return filled
