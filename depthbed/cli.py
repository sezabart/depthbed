"""One command, one shot, one DXF.

  python -m depthbed                 # live preview; SPACE shoots, q quits
  python -m depthbed --once          # one shot, no window
  python -m depthbed --from cap.ocap --level 2.5
  python -m depthbed --once --save cap.ocap

Each shot writes a folder with the rectified photo, the outline as a DXF in
millimetres, and a small JSON of the numbers behind it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import cad
from .rectify import rectify


def _make_camera(args):
    if args.replay:
        from .capture_io import ReplayCamera
        return ReplayCamera(args.replay)
    from .camera import OakCamera
    return OakCamera()


def shoot(cam, args) -> Path | None:
    """Capture, rectify, write the folder. Returns the folder, or None."""
    t0 = time.time()
    try:
        cap = cam.capture()
    except Exception as e:  # noqa: BLE001 - device errors are not interesting
        print(f"capture failed: {type(e).__name__}: {e}")
        return None

    if args.save:
        from .capture_io import save_capture
        save_capture(cap, args.save)

    try:
        rect = rectify(cap, mm_per_px=args.mm_per_px, ref_height_mm=args.ref)
    except ValueError as e:
        print(f"cannot rectify: {e}")
        return None

    w_mm, h_mm = rect.extent_mm

    # A level set below the depth noise is not an outline, it is the noise.
    # At 1.2 m the plane residual is ~8 mm, and a 1 mm level yields thousands
    # of loops around speckle. Say so instead of handing CAD the speckle.
    noise_mm = float(rect.stats.get("plane_residual_mm") or 0.0)
    if args.level < 1.5 * noise_mm:
        loops = []
        outline_note = (f"skipped: level {args.level:.2f} mm is inside the depth "
                        f"noise (residual {noise_mm:.2f} mm; needs "
                        f">{1.5 * noise_mm:.1f} mm)")
    else:
        loops = cad.outline(rect, args.level)
        outline_note = None

    out = _fresh_dir(Path(args.out) / time.strftime("%Y%m%d-%H%M%S"))
    photo = out / "ortho.png"
    dxf = out / f"outline-{args.level:.2f}mm.dxf"
    cv2.imwrite(str(photo), rect.rgb)
    if outline_note is None:
        cad.write_dxf(loops, dxf)

    h = rect.height_mm[np.isfinite(rect.height_mm)]
    meta = {
        "mm_per_px": rect.stats["mm_per_px"],
        "width_mm": round(w_mm, 2),
        "height_mm": round(h_mm, 2),
        "level_mm": args.level,
        "ref_mm": args.ref,
        "u0_mm": round(rect.u0_mm, 2),
        "v0_mm": round(rect.v0_mm, 2),
        "photo": photo.name,
        "dxf": dxf.name if outline_note is None else None,
        "loops": len(loops),
        "outline_note": outline_note,
        "height_range_mm": [round(float(np.min(h)), 2), round(float(np.max(h)), 2)],
        "quality": {k: rect.stats[k] for k in
                    ("plane_residual_mm", "plane_inlier_frac", "range_mm",
                     "height_mm_per_px", "measured_frac") if k in rect.stats},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    q = meta["quality"]
    print(f"\nshot in {time.time() - t0:.1f} s   range {q['range_mm']} mm   "
          f"plane residual {q['plane_residual_mm']} mm "
          f"({q['plane_inlier_frac'] * 100:.0f}% coverage)")
    print(f"  photo    {photo}   {w_mm:.1f} x {h_mm:.1f} mm "
          f"@ {meta['mm_per_px']} mm/px")
    if outline_note:
        print(f"  cad      {outline_note}")
    else:
        print(f"  cad      {dxf}   {len(loops)} closed loop(s) at {args.level:.2f} mm")
        if not loops:
            print(f"  no outline at {args.level:.2f} mm (height runs "
                  f"{meta['height_range_mm'][0]:.1f} .. "
                  f"{meta['height_range_mm'][1]:.1f} mm) -- pick another --level")
    print(f"  underlay insert {photo.name} at (0, 0), size "
          f"{w_mm:.1f} x {h_mm:.1f} mm"
          + ("" if outline_note else ", to line it up with the DXF"))
    sys.stdout.flush()
    return out


def _fresh_dir(path: Path) -> Path:
    """mkdir, but never overwrite: two shots in one second get -2, -3, ..."""
    candidate, n = path, 1
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.name}-{n}")
    candidate.mkdir(parents=True)
    return candidate


def _open(cam) -> bool:
    try:
        cam.open()
        return True
    except Exception as e:  # noqa: BLE001 - device errors are not interesting
        print(f"cannot open the camera: {type(e).__name__}: {e}")
        return False


def preview(cam, args) -> int:
    if not _open(cam):
        return 1
    live = "DepthBed  -  SPACE shoots, q quits"
    cv2.namedWindow(live, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(live, 960, 720)
    try:
        while True:
            frame = cam.preview()
            if frame is not None:
                cv2.imshow(live, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (32, 13):
                out = shoot(cam, args)
                if out is not None:
                    photo = cv2.imread(str(out / "ortho.png"))
                    if photo is not None:
                        h, w = photo.shape[:2]
                        s = min(1.0, 1000 / w, 700 / h)
                        cv2.imshow("last shot", cv2.resize(photo, (int(w * s), int(h * s))))
    finally:
        cv2.destroyAllWindows()
        cam.close()
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="depthbed",
        description="Rectified photo + CAD outline from an OAK-D Pro.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--once", action="store_true", help="one shot, no preview window")
    ap.add_argument("--from", dest="replay", metavar="CAPTURE",
                    help="replay a saved capture instead of using the camera")
    ap.add_argument("--save", metavar="CAPTURE", help="also save the capture")
    ap.add_argument("--out", metavar="DIR", default="out", help="output directory")
    ap.add_argument("--level", type=float, default=1.0,
                    help="outline height above the base plane, mm")
    ap.add_argument("--mm-per-px", type=float, default=0.1,
                    help="asked-for photo scale; limited by 4000 px")
    ap.add_argument("--ref", type=float, default=0.0,
                    help="height of the face that must be metrically exact, mm")
    args = ap.parse_args()

    cam = _make_camera(args)
    try:
        if args.replay or args.once:
            if not args.replay and not _open(cam):
                return 1
            try:
                return 0 if shoot(cam, args) is not None else 1
            finally:
                cam.close()
        return preview(cam, args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
