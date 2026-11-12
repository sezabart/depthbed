"""Save a single capture to disk and replay it forever.

The point: you only have to hold the camera once. After that, changing the
outline level is a re-run against the same frozen capture, not another shot.

Colour is stored as JPEG (a 12 MP raw frame is 36 MB) and depth as a compressed
float32 array, so a capture is a few megabytes and one file.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .types import Capture, Intrinsics

MAGIC = "depthbed-capture-1"


def save_capture(cap: Capture, path) -> Path:
    path = Path(path)
    ok, buf = cv2.imencode(".jpg", cap.rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("could not encode RGB")

    meta = {
        "magic": MAGIC,
        "intr_rgb": vars(cap.intr_rgb) | {"dist": None if cap.intr_rgb.dist is None
                                          else list(map(float, cap.intr_rgb.dist))},
        "intr_depth": vars(cap.intr_depth) | {"dist": None if cap.intr_depth.dist is None
                                              else list(map(float, cap.intr_depth.dist))},
        "T": np.asarray(cap.T_depth_to_rgb).tolist(),
        "meta": {k: v for k, v in cap.meta.items() if isinstance(v, (str, int, float, bool))},
    }
    # Pass a file handle: np.savez appends ".npz" to a path that lacks it,
    # which silently writes somewhere other than where you asked.
    with open(path, "wb") as fh:
        np.savez_compressed(
            fh,
            rgb_jpeg=np.frombuffer(buf.tobytes(), dtype=np.uint8),
            depth=np.asarray(cap.depth, dtype=np.float32),
            meta=np.array(json.dumps(meta)),
        )
    return path


def _intr(d):
    dist = d.get("dist")
    return Intrinsics(fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
                      width=int(d["width"]), height=int(d["height"]),
                      dist=None if dist is None else np.asarray(dist, dtype=np.float64))


def load_capture(path) -> Capture:
    z = np.load(Path(path), allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if meta.get("magic") != MAGIC:
        raise ValueError(f"{path} is not a DepthBed capture")
    rgb = cv2.imdecode(z["rgb_jpeg"], cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError("RGB payload in capture is corrupt")
    return Capture(
        rgb=rgb,
        depth=z["depth"].astype(np.float32),
        intr_rgb=_intr(meta["intr_rgb"]),
        intr_depth=_intr(meta["intr_depth"]),
        T_depth_to_rgb=np.asarray(meta["T"], dtype=np.float64),
        meta=meta.get("meta", {}),
    )


class ReplayCamera:
    """Camera-shaped wrapper around a saved capture (same interface as OakCamera)."""

    def __init__(self, path):
        self.path = path
        self._cap = load_capture(path)
        self.info = {"usb_speed": "replay", "usb2": False, "mode": "replay",
                     "source": str(path)}
        self._preview = None

    def open(self):
        return self

    def close(self):
        pass

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def preview(self):
        if self._preview is None:
            self._preview = cv2.resize(self._cap.rgb, (640, 480),
                                       interpolation=cv2.INTER_AREA)
        return self._preview

    def capture(self, timeout: float = 12.0):
        return self._cap
