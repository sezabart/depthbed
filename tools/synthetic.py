"""Synthetic scene with known ground truth.

The scene is a *heightfield on a plane*: a tilted backing surface plus a set of
cylindrical bumps of known radius and height. That model makes the ground truth
trivial to state and to assert on -- and it means no test depends on any
"find the part" logic, because there isn't any.

It can print a grid and marker patches onto the base plane, which gives the
rectification test something real to check itself against.
"""

from __future__ import annotations

import numpy as np

from depthbed.types import Capture, Intrinsics


def _rotation(rx: float, ry: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx


def _intrinsics(w: int, h: int, hfov_deg: float) -> Intrinsics:
    f = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return Intrinsics(fx=f, fy=f, cx=(w - 1) / 2.0, cy=(h - 1) / 2.0,
                      width=w, height=h, dist=None)


class _Scene:
    def __init__(self, work_dist_m, tilt_deg, bumps, grid_mm, markers):
        self.n = _rotation(np.radians(tilt_deg * 0.6), np.radians(tilt_deg)) @ np.array([0.0, 0.0, 1.0])
        self.n /= np.linalg.norm(self.n)
        p0 = np.array([0.0, 0.0, work_dist_m])
        self.d = float(self.n @ p0)
        self.bumps = [tuple(b) for b in bumps]      # (cu, cv, r, h) mm, plane coords
        self.grid_mm = grid_mm
        self.markers = list(markers)                # (u, v, size_mm)

        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(self.n @ ref)) > 0.95:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = ref - self.n * (self.n @ ref)
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.n, self.e1)
        self.origin = self.n * self.d

    def height_mm(self, u, v):
        h = np.zeros(np.shape(u), dtype=np.float64)
        for cu, cv, r, hh in self.bumps:
            inside = (u - cu) ** 2 + (v - cv) ** 2 < r * r
            h = np.where(inside, np.maximum(h, hh), h)
        return h

    def render(self, intr: Intrinsics, eye_offset=(0.0, 0.0, 0.0)):
        ys, xs = np.mgrid[0:intr.height, 0:intr.width].astype(np.float64)
        x = (xs - intr.cx) / intr.fx
        y = (ys - intr.cy) / intr.fy
        inv = 1.0 / np.sqrt(x * x + y * y + 1.0)
        rx, ry, rz = x * inv, y * inv, inv

        o = np.asarray(eye_offset, dtype=np.float64)
        denom = self.n[0] * rx + self.n[1] * ry + self.n[2] * rz
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.d - self.n @ o) / denom
        P = np.stack([o[0] + rx * t, o[1] + ry * t, o[2] + rz * t], axis=-1)

        rel = P - self.origin
        u = (rel @ self.e1) * 1000.0
        v = (rel @ self.e2) * 1000.0
        h = self.height_mm(u, v)
        z = P[..., 2] - h / 1000.0 * self.n[2]
        return z, u, v, h

    def shade(self, u, v, h):
        img = np.zeros(np.shape(u) + (3,), np.float32)
        img[..., :] = (198, 192, 184)
        if self.grid_mm:
            gu = np.abs((u % self.grid_mm) - self.grid_mm / 2) < 0.3
            gv = np.abs((v % self.grid_mm) - self.grid_mm / 2) < 0.3
            img[np.logical_or(gu, gv)] = (80, 80, 88)
        on = h > 0.05
        tint = (h / max(float(h.max()), 1.0))[..., None]
        img = np.where(on[..., None], np.array([110, 128, 112], np.float32) + tint * 80.0, img)
        for mu, mv, ms in self.markers:
            hit = (np.abs(u - mu) < ms / 2) & (np.abs(v - mv) < ms / 2)
            img[hit] = (20, 20, 250)
        return np.clip(img, 0, 255)


def make_scene(work_dist_m: float = 0.35, tilt_deg: float = 4.0,
               bumps=((45.0, -15.0, 25.0, 8.0),), grid_mm: float = 10.0,
               markers=(), depth_size=(1280, 800), rgb_size=(1613, 1216),
               depth_hfov=80.0, rgb_hfov=66.0, depth_noise_mm: float = 1.0,
               invalid_frac: float = 0.01, baseline_m: float = 0.0375,
               seed: int = 0) -> Capture:
    rng = np.random.default_rng(seed)
    scene = _Scene(work_dist_m, tilt_deg, bumps, grid_mm, markers)

    intr_d = _intrinsics(*depth_size, depth_hfov)
    intr_r = _intrinsics(*rgb_size, rgb_hfov)

    z, _, _, _ = scene.render(intr_d)
    depth = z + rng.normal(0.0, depth_noise_mm / 1000.0, z.shape)
    depth[rng.random(depth.shape) < invalid_frac] = np.nan

    # The RGB camera is a genuinely different viewpoint. That parallax is the
    # whole reason the rectification is non-trivial.
    t = np.array([-baseline_m, 0.0, 0.0])
    _, ur, vr, hr = scene.render(intr_r, eye_offset=-t)
    rgb = scene.shade(ur, vr, hr) + rng.normal(0.0, 1.5, ur.shape + (3,))

    T = np.eye(4)
    T[:3, 3] = t

    return Capture(
        rgb=np.clip(rgb, 0, 255).astype(np.uint8),
        depth=depth.astype(np.float32),
        intr_rgb=intr_r,
        intr_depth=intr_d,
        T_depth_to_rgb=T,
        meta={"ground_truth": {
            "bumps": [tuple(b) for b in bumps],
            "tilt_deg": tilt_deg,
            "work_dist_mm": work_dist_m * 1000.0,
            "baseline_mm": baseline_m * 1000.0,
            "markers": [list(m) for m in markers],
        }},
    )


def stepped_scene(**kw) -> Capture:
    """Two levels: a 4 mm ring of radius 25 with an 8 mm boss of radius 12.

    Lets a level-set test ask for "the outline at 6 mm" and expect a 24 mm
    circle -- exactly what a person does when tracing in CAD.
    """
    kw.setdefault("tilt_deg", 0.0)
    return make_scene(bumps=((0.0, 0.0, 12.0, 8.0), (0.0, 0.0, 25.0, 4.0)), **kw)
