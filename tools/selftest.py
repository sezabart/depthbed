#!/usr/bin/env python3
"""Ground-truth tests for the rectifier and the CAD output.

Nothing here decides what "the part" is. It checks that geometry printed at
known plane coordinates lands there, that known heights read back, that the
outline at a level has the right size, and that the DXF says what we meant.

Run: .venv/bin/python tools/selftest.py
"""

import os
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from synthetic import make_scene, stepped_scene          # noqa: E402
from depthbed.rectify import rectify                     # noqa: E402
from depthbed import cad                                 # noqa: E402

OUT = "results/selftest"
MARKERS = [(-60.0, -40.0, 4.0), (60.0, -40.0, 4.0),
           (-60.0, 40.0, 4.0), (60.0, 40.0, 4.0), (0.0, 0.0, 4.0)]
# At 25 degrees the RGB frame only covers part of the plane, and the export is
# cropped to the largest clean rectangle inside it. Keep these markers well
# inside that crop so the test measures geometry, not the crop edge.
MARKERS_INNER = [(-40.0, -25.0, 4.0), (40.0, -25.0, 4.0),
                 (-40.0, 25.0, 4.0), (40.0, 25.0, 4.0), (0.0, 0.0, 4.0)]


def marker_centroids_mm(rect):
    """Find the blue marker patches in the photo and return their mm coords."""
    b, g, r = (rect.rgb[..., 0].astype(int), rect.rgb[..., 1].astype(int),
               rect.rgb[..., 2].astype(int))
    mask = ((r > 150) & (b < 90) & (g < 90)).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 80:
            continue
        found.append((rect.u0_mm + cents[i][0] * rect.mm_per_px,
                      rect.v0_mm + cents[i][1] * rect.mm_per_px))
    return found


def dxf_loops(path):
    """Read back the loops our own writer produced, to prove what is on disk."""
    lines = [ln.strip() for ln in open(path).read().splitlines()]
    pairs = list(zip(lines[0::2], lines[1::2]))
    loops, loop, vertex = [], None, None
    for code, val in pairs:
        if code == "0":
            if vertex is not None:
                loop.append((vertex["10"], vertex["20"]))
                vertex = None
            if val == "POLYLINE":
                loop = []
                loops.append(loop)
            elif val == "VERTEX":
                vertex = {}
        elif vertex is not None and code in ("10", "20"):
            vertex[code] = float(val)
    return [np.array(p) for p in loops]


def main():
    os.makedirs(OUT, exist_ok=True)
    failures = []

    # ------------------------------------------------- 1. rectification fidelity
    # 25 degrees is the honest test of the whole point: a strongly oblique
    # photo must come back with the printed geometry in the right place.
    for seed, tilt, markers in ((0, 4.0, MARKERS), (1, 0.0, MARKERS),
                                (2, 7.0, MARKERS), (5, 25.0, MARKERS_INNER)):
        cap = make_scene(seed=seed, tilt_deg=tilt, markers=markers)
        rect = rectify(cap)
        found = marker_centroids_mm(rect)
        print(f"\n--- rectification, seed {seed}, tilt {tilt} deg ---")
        print(f"  plane residual {rect.stats.get('plane_residual_mm')} mm | "
              f"{rect.stats['ortho_size']} px @ {rect.stats['mm_per_px']} mm/px")

        if len(found) != len(markers):
            failures.append(f"seed {seed}: found {len(found)} markers, "
                            f"expected {len(markers)}")
            continue

        errs = []
        for tu, tv, _ in markers:
            # nearest match, not sort order: two markers can share a coordinate
            d = [np.hypot(fu - tu, fv - tv) for fu, fv in found]
            k = int(np.argmin(d))
            fu, fv = found[k]
            errs.append(d[k])
            print(f"    marker ({tu:6.1f},{tv:6.1f}) -> ({fu:6.2f},{fv:6.2f})"
                  f"   err {d[k]:5.3f} mm")
        worst = max(errs)
        if worst > 0.30:
            failures.append(f"seed {seed}: worst marker error {worst:.3f} mm")

    # ------------------------------------------------------------ 2. heights
    cap = make_scene(seed=3, bumps=((45.0, -15.0, 25.0, 8.0),))
    rect = rectify(cap)

    def at(uu, vv, half_mm=2.0):
        # Sample in millimetres, not pixels: the canvas scale depends on the
        # working range and a fixed pixel offset walked onto the bump skirt.
        iu = int(round((uu - rect.u0_mm) / rect.mm_per_px))
        iv = int(round((vv - rect.v0_mm) / rect.mm_per_px))
        n = max(1, int(round(half_mm / rect.mm_per_px)))
        return rect.height_mm[iv - n:iv + n, iu - n:iu + n]

    bump = at(45.0, -15.0)
    flat = at(85.0, 20.0)
    print("\n--- height field ---")
    print(f"  on 8 mm bump : {np.nanmedian(bump):6.3f} mm (truth 8.000)")
    print(f"  on bare sheet: {np.nanmedian(flat):6.3f} mm (truth 0.000)")
    if abs(np.nanmedian(bump) - 8.0) > 0.30:
        failures.append(f"bump height {np.nanmedian(bump):.3f} != 8.0")
    if abs(np.nanmedian(flat)) > 0.30:
        failures.append(f"base height {np.nanmedian(flat):.3f} != 0.0")

    # ------------------------------------------- 3. outlines, in drawing mm
    cap = stepped_scene(seed=4)
    rect = rectify(cap)
    w_mm, h_mm = rect.extent_mm
    print("\n--- outlines on a 4 mm ring / 8 mm boss ---")
    for level, expect_dia in ((6.0, 24.0), (2.0, 50.0)):
        loops = cad.outline(rect, level)
        if not loops:
            failures.append(f"level {level}: no outline")
            continue
        p = loops[0]
        dia_w = p[:, 0].max() - p[:, 0].min()
        dia_h = p[:, 1].max() - p[:, 1].min()
        print(f"  level {level:4.1f} mm -> {len(loops)} loop(s), bbox "
              f"{dia_w:6.2f} x {dia_h:6.2f} mm (truth {expect_dia:.1f})")
        if abs(dia_w - expect_dia) > 1.2 or abs(dia_h - expect_dia) > 1.2:
            failures.append(f"level {level}: bbox {dia_w:.2f}x{dia_h:.2f} != {expect_dia}")
        inside = (p[:, 0] >= -1e-6).all() and (p[:, 0] <= w_mm + 1e-6).all() \
            and (p[:, 1] >= -1e-6).all() and (p[:, 1] <= h_mm + 1e-6).all()
        if not inside:
            failures.append(f"level {level}: outline leaves the drawing frame")

    # ----------------------------------------------------- 4. DXF round trip
    loops = cad.outline(rect, 6.0)
    path = os.path.join(OUT, "outline-6.00mm.dxf")
    cad.write_dxf(loops, path)
    back = dxf_loops(path)
    print(f"\n--- DXF ---\n  {path}: {len(back)} loop(s)")
    if len(back) != len(loops):
        failures.append(f"dxf: {len(back)} loops != {len(loops)}")
    else:
        for a, b in zip(loops, back):
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-5):
                failures.append("dxf: vertices do not round-trip")
                break
        print(f"  vertices match: {sum(len(a) for a in loops)} points, Y up, mm")

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS - rectification, height field, outlines and DXF match ground truth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
