"""Shared data types. Kept dumb: plain dataclasses + numpy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Intrinsics:
    """Pinhole intrinsics at a specific image size, plus lens distortion.

    ``dist`` holds OpenCV-style coefficients (k1, k2, p1, p2, k3) or is None.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: Optional[np.ndarray] = None

    @classmethod
    def from_matrix(cls, k, width: int, height: int, dist=None) -> "Intrinsics":
        k = np.asarray(k, dtype=np.float64)
        d = None if dist is None else np.asarray(dist, dtype=np.float64).ravel()
        return cls(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]),
                   int(width), int(height), d)

    def distort(self, x, y):
        """Normalised pinhole coords -> distorted normalised coords.

        Full OpenCV model, including the *rational* denominator. The IMX378 on
        this device reports k1=47.3, k2=-305, k3=656.8 with k4..k6 nearly
        cancelling them; using only k1..k3 (the cheap version everyone writes)
        gives a ~60x radial scale and bends the whole image. Coefficient order
        is OpenCV's: k1, k2, p1, p2, k3, k4, k5, k6.
        """
        if self.dist is None or self.dist.size < 4:
            return x, y
        d = np.asarray(self.dist, dtype=np.float64).ravel()
        k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
        k3 = d[4] if d.size > 4 else 0.0
        k4 = d[5] if d.size > 5 else 0.0
        k5 = d[6] if d.size > 6 else 0.0
        k6 = d[7] if d.size > 7 else 0.0

        r2 = x * x + y * y
        num = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        den = 1.0 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        radial = num / den
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return xd, yd

    def undistort(self, xd, yd, iters: int = 8):
        """Inverse of :meth:`distort`, by fixed-point iteration.

        Only needed to turn a pixel into a viewing *direction* (the footprint
        of the RGB frame on the plane). Eight iterations are well past
        convergence for this lens; the residual is below 1e-10 px.
        """
        if self.dist is None or self.dist.size < 4:
            return xd, yd
        x = np.array(xd, dtype=np.float64, copy=True)
        y = np.array(yd, dtype=np.float64, copy=True)
        for _ in range(iters):
            px, py = self.distort(x, y)
            x = x - (px - xd)
            y = y - (py - yd)
        return x, y


@dataclass
class Plane:
    """Plane in camera frame: ``normal . x = d``. Normal is unit and points
    away from the camera (so points in front have normal . x > 0)."""

    normal: np.ndarray
    d: float

    def height(self, points) -> np.ndarray:
        """Signed height above the plane, towards the camera (metres)."""
        return self.d - np.asarray(points) @ self.normal

    def transform(self, R, t) -> "Plane":
        """Express this plane in a frame where x' = R x + t."""
        n = R @ self.normal
        return Plane(n / np.linalg.norm(n), (self.d + n @ t) / np.linalg.norm(n))

    def basis(self):
        """A stable in-plane orthonormal basis (e1, e2), e1 ~ camera +X."""
        n = self.normal
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(n @ ref)) > 0.95:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = ref - n * (n @ ref)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        return e1, e2


@dataclass
class Capture:
    """One synchronised set of frames plus everything needed to reason about it.

    ``depth`` is in the depth camera frame (CAM_C), ``rgb`` in the RGB frame
    (CAM_A). ``T_depth_to_rgb`` is the 4x4 that takes depth-frame points into
    the RGB frame.
    """

    rgb: np.ndarray
    depth: np.ndarray
    intr_rgb: Intrinsics
    intr_depth: Intrinsics
    T_depth_to_rgb: np.ndarray
    meta: dict = field(default_factory=dict)
