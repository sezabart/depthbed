# DepthBed

A depth camera that undoes perspective, and hands the result to CAD.

Point an OAK-D Pro at a flat surface from any angle. Each shot produces a
rectified photograph of that surface and a DXF outline at a height you choose,
both in millimetres, both in the same frame. Nothing is detected or
segmented — the tool fits one plane and reports it; what matters in the frame
is your call.

![A synthetic grid photographed 30 degrees off-axis, and the same grid
rectified](ortho-demo.png)

## What a shot gives you

```
out/<time>/
  outline-1.00mm.dxf   closed polylines, mm, Y up
  ortho.png            rectified photo, same frame and scale
  meta.json            mm_per_px, width_mm, height_mm, level, plane quality
```

Both files use the **photo's frame**: origin at its lower-left corner, X right,
Y up, millimetres. So:

- import the DXF and it is already the right size — no scaling;
- to line the photo up underneath it, insert `ortho.png` with its lower-left
  corner at `(0, 0)` and its size set to `width_mm × height_mm` from
  `meta.json`.

The outline is a level set: every boundary where the surface crosses
`--level` mm above the base plane. That number is the operator's selection, so
one part can be drawn at its footprint (1 mm) or at a raised face (10 mm) from
the same capture, without rescanning (`--from cap.ocap --level 10`).

## How the photo is made

1. **Find the dominant plane** with RANSAC on the depth map. The inlier
   threshold follows working range (stereo error grows as z²): 4 mm at
   350 mm, clamped at 20 mm for a 1 m scene.
2. **Build the plane frame once**, in the depth camera's frame, then carry the
   origin and basis into the RGB frame with the calibrated extrinsic. Building
   it separately in each frame offsets the two views by the 37 mm baseline.
3. **Frame the shot.** A plane maps to an image by a homography, so the RGB
   frame's border maps to a quadrilateral on the plane. The canvas is the
   largest axis-aligned rectangle inside (that quadrilateral ∩ where depth
   actually saw the plane) — a full picture, no black wedges, no metres-wide
   sheet made of three stray depth returns.
4. **Warp the 12 MP colour image analytically**: point on the plane → real
   projection through the RGB camera (full rational distortion, not just
   k1..k3) → one `remap`. A plane is a homography, so no per-pixel depth is
   needed and the 12 MP detail survives.
5. **Measure height separately**, because only one plane can be exact. The 3D
   surface is scattered from above onto the same grid and the top surface wins
   each cell. This is deliberately not a warp: warping would displace heights
   sideways by `h·tan θ`. Cells the camera could not see stay empty, and the
   outline simply does not run through them.

## Measured accuracy

`tools/selftest.py`, synthetic scene with known geometry, 1 mm depth noise, 1%
dropouts, RGB and depth cameras 37.5 mm apart. Markers printed at known plane
coordinates, located in the rectified photo:

| tilt | worst marker error over a ±60 × ±40 mm field |
|---|---|
| 0° | 0.06 mm |
| 4° | 0.10 mm |
| 7° | 0.08 mm |
| 25° (inner markers) | 0.05 mm |

That exercises intrinsics, distortion, inter-camera extrinsics and the
ray-cast as one chain: sub-0.1 mm over 120 mm, or 0.08%.

| what | result | truth |
|---|---|---|
| height on an 8 mm bump | 8.02 mm | 8.00 |
| height on bare sheet | 0.14 mm | 0.00 |
| outline at 6 mm on a 24 mm boss | 23.6 mm | 24.0 |
| outline at 2 mm on a 50 mm ring | 49.3 mm | 50.0 |

Outlines read ~1.5% small: half-pixel convention plus one bilinear upsample
from the depth sampling grid. Absolute scale comes from the device's factory
calibration (baseline 37.3 mm, depth fx 800 px); the tests confirm the maths
preserves it, not that this lens was calibrated well.

## Usage

```bash
.venv/bin/python -m depthbed                 # preview; SPACE shoots, q quits
.venv/bin/python -m depthbed --once          # one shot
.venv/bin/python -m depthbed --once --level 4
.venv/bin/python -m depthbed --once --save cap.ocap
.venv/bin/python -m depthbed --from cap.ocap --level 4
.venv/bin/python -m depthbed --from examples/demo.ocap   # no camera
```

| flag | meaning |
|---|---|
| `--level MM` | outline height above the base plane (default 1.0) |
| `--mm-per-px MM` | asked-for scale; capped at a 4000 px photo, real value in `meta.json` |
| `--ref MM` | the face that must be metrically exact (default 0, the base) |
| `--out DIR` | where the per-shot folders go (default `out/`) |
| `--save CAPTURE`, `--from CAPTURE` | keep / replay a capture |

A shot is ~0.6 s once the pipeline is up (plus ~3 s for the first one, which
waits for exposure and the stereo matcher). Save a capture if you want to keep
tuning the level offline.

One guard worth knowing: if `--level` is inside the depth noise, the outline
would be thousands of specks, so **no DXF is written** and the message tells
you the level above which it becomes meaningful. This matters when the surface
is far away — stereo noise grows as z², so at 1.2 m a 1 mm level is meaningless
while at 350 mm it is fine. The photo is always written.

Hardware setup (once, needs root):

```bash
sudo sh -c 'echo SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"03e7\", MODE=\"0666\" \
  > /etc/udev/rules.d/80-movidius.rules && udevadm control --reload-rules && udevadm trigger'
```

## Limits worth knowing

- **Planar faces only, really.** A curved face rectifies as well as its
  departure from the plane allows.
- **The base is assumed flat.** A warped table is a warped datum.
- **Occlusion**: a face hidden behind a taller step is not measured and is not
  invented. The outline stops there.
- **Shiny metal** punches holes in depth. The IR dot projector helps; matte
  powder helps more.
- **A very oblique shot loses corners**: rectifying 30–40° leaves a smaller
  clean rectangle than shooting square-on. Past that it refuses rather than
  emitting nonsense.
- **Roll is not corrected.** Drawing X/Y follow the camera's. Mount it level,
  or rotate the DXF.

## Two hardware facts that cost time

- The RGB distortion is the OpenCV **rational** model: `k1=47.3, k2=-305,
  k3=656.8` with a nearly-cancelling `k4..k6`. Using only `k1..k3` gives a
  ~60× radial scale and bends the image.
- The first ~1.5 s after pipeline start is unusable (exposure and the stereo
  matcher have not settled), so the capture waits it out.
