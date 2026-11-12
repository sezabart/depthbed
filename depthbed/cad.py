"""CAD output: outlines in millimetres, and a DXF you can open and use.

The drawing frame is the photo's frame: origin at its lower-left corner, X to
the right, Y up, millimetres. Insert the photo as an underlay with its
lower-left corner at (0, 0) and its size set to `width_mm` x `height_mm` and
the picture lines up with the geometry exactly. No scaling, no rotation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def outline(rect, level_mm: float, min_area_mm2: float = 4.0):
    """Closed loops where the height crosses ``level_mm``, in drawing mm.

    This is the operator's selection: you choose the height, you get the
    contour there. Nothing decides which object matters.

    Loops under a few square millimetres are depth noise, not features, and
    are dropped -- otherwise a noisy surface fills the DXF with specks. Change
    ``min_area_mm2`` if you genuinely need 1 mm holes.
    """
    h = np.nan_to_num(rect.height_mm, nan=-np.inf)
    mask = (h >= level_mm).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    min_px = min_area_mm2 / (rect.mm_per_px ** 2)
    _, img_h_mm = rect.extent_mm

    loops = []
    for c in contours:
        if cv2.contourArea(c) < min_px:
            continue
        # 1 px tolerance: 0.1 mm at the usual scale. Keeps the DXF editable
        # instead of handing CAD a 500-vertex circle.
        p = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(np.float64)
        if len(p) < 3:
            continue
        # pixel -> mm, Y flipped for CAD (Y up)
        x = p[:, 0] * rect.mm_per_px
        y = img_h_mm - p[:, 1] * rect.mm_per_px
        loops.append(np.stack([x, y], axis=1))

    loops.sort(key=lambda p: -abs(_signed_area(p)))   # big loops first, holes after
    return loops


def _signed_area(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _pairs(*items) -> str:
    return "".join(f"{code}\n{value}\n" for code, value in items)


def write_dxf(loops, path, layer: str = "OUTLINE") -> Path:
    """Minimal DXF R12: one closed POLYLINE per loop, millimetres, Y up.

    R12 is the least surprising thing to hand a CAD program, and closed
    polylines are directly extrudable / offsettable after import.
    """
    path = Path(path)
    loops = [np.asarray(p, dtype=np.float64) for p in loops if len(p) >= 3]
    if loops:
        allp = np.concatenate(loops, axis=0)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
    else:
        lo = hi = np.zeros(2)

    out = [
        _pairs((0, "SECTION"), (2, "HEADER"),
               (9, "$ACADVER"), (1, "AC1009"),
               (9, "$INSUNITS"), (70, 4),            # 4 = millimetres
               (9, "$EXTMIN"), (10, f"{lo[0]:.6f}"), (20, f"{lo[1]:.6f}"), (30, "0.0"),
               (9, "$EXTMAX"), (10, f"{hi[0]:.6f}"), (20, f"{hi[1]:.6f}"), (30, "0.0"),
               (0, "ENDSEC")),
        _pairs((0, "SECTION"), (2, "ENTITIES")),
    ]
    for loop in loops:
        out.append(_pairs((0, "POLYLINE"), (8, layer), (66, 1), (70, 1),
                          (10, "0.0"), (20, "0.0"), (30, "0.0")))
        for x, y in loop:
            out.append(_pairs((0, "VERTEX"), (8, layer),
                              (10, f"{x:.6f}"), (20, f"{y:.6f}"), (30, "0.0")))
        out.append(_pairs((0, "SEQEND"), (8, layer)))
    out.append(_pairs((0, "ENDSEC"), (0, "EOF")))
    path.write_text("".join(out))
    return path
