"""OAK-D Pro device layer (DepthAI v3 API).

Two things worth knowing before reading the code:

* The depth map is aligned to the RIGHT mono camera (CAM_C) and rectified, so
  it is pinhole with (nominally) no distortion. The pretty 12MP image comes
  from CAM_A, which is a *different* physical camera 37.5 mm away. Everything
  downstream has to respect that, which is why ``Capture`` carries both sets
  of intrinsics plus the 4x4 between them.
* This board may well enumerate at USB 2.0 (480 Mb/s, ~30 MB/s real). That is
  fine for a button-press scanner, but it is not fine for streaming raw 12 MP.
  Hence the MJPEG encoder on the still path and a small preview.
"""

from __future__ import annotations

import threading
import time

import numpy as np

import depthai as dai

from .types import Capture, Intrinsics

RGB_SIZE = (4032, 3040)
DEPTH_SIZE = (1280, 800)
PREVIEW_SIZE = (640, 480)


class OakCamera:
    def __init__(self, preview_size=PREVIEW_SIZE, depth_size=DEPTH_SIZE,
                 rgb_size=RGB_SIZE, dot_projector: bool = True):
        self.preview_size = tuple(preview_size)
        self.depth_size = tuple(depth_size)
        self.rgb_size = tuple(rgb_size)
        self.dot_projector = dot_projector

        self._lock = threading.RLock()
        self._pipeline = None
        self._device = None
        self._queues = {}
        self.intr_rgb: Intrinsics | None = None
        self.intr_depth: Intrinsics | None = None
        self.T_depth_to_rgb: np.ndarray | None = None
        self.info: dict = {}

    # ------------------------------------------------------------------ setup
    def open(self) -> "OakCamera":
        with self._lock:
            if self._pipeline is not None:
                return self

            p = dai.Pipeline()

            cam_a = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            # NV12 explicitly: the ISP default here is BGR888p and the hardware
            # MJPEG encoder only accepts NV12 / YUV400p.
            rgb_out = cam_a.requestOutput(
                self.rgb_size, type=dai.ImgFrame.Type.NV12, fps=2.0)
            prev_out = cam_a.requestOutput(self.preview_size, fps=15.0)

            # JPEG on the device, not raw 12 MP over the wire: at USB 2 a raw
            # still is ~18 MB and would monopolise the link for most of a second.
            enc = p.create(dai.node.VideoEncoder)
            enc.setDefaultProfilePreset(1.0, dai.VideoEncoderProperties.Profile.MJPEG)
            enc.setQuality(95)
            rgb_out.link(enc.input)

            cam_b = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            cam_c = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            left = cam_b.requestOutput(self.depth_size, fps=8.0)
            right = cam_c.requestOutput(self.depth_size, fps=8.0)

            stereo = p.create(dai.node.StereoDepth).build(
                left, right, presetMode=dai.node.StereoDepth.PresetMode.DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setExtendedDisparity(True)
            stereo.setSubpixel(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_C)
            stereo.setRectification(True)

            self._queues = {
                "depth": stereo.depth.createOutputQueue(maxSize=2, blocking=False),
                "right": stereo.rectifiedRight.createOutputQueue(maxSize=1, blocking=False),
                "preview": prev_out.createOutputQueue(maxSize=1, blocking=False),
                "rgb": enc.bitstream.createOutputQueue(maxSize=1, blocking=False),
            }

            p.start()
            self._pipeline = p

            try:
                dev = p.getDefaultDevice()
            except Exception:  # noqa: BLE001 - older builds need the explicit ctor
                dev = dai.Device(p)
            self._device = dev

            speed = str(dev.getUsbSpeed())
            self.info["usb_speed"] = speed
            self.info["usb2"] = "HIGH" in speed.upper() and "SUPER" not in speed.upper()

            if self.dot_projector:
                try:
                    dev.setIrLaserDotProjectorIntensity(1.0)   # max; helps blank surfaces
                    self.info["dot_projector"] = "on"
                except Exception as e:  # noqa: BLE001
                    self.info["dot_projector"] = f"unavailable: {e}"

            self._read_calibration(dev)
            self._opened_at = time.time()
            return self

    def _read_calibration(self, dev):
        calib = dev.readCalibration()
        w, h = self.rgb_size
        dw, dh = self.depth_size

        k_a = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, w, h)
        dist_a = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
        self.intr_rgb = Intrinsics.from_matrix(k_a, w, h, dist_a)

        k_c = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, dw, dh)
        # CAM_C arrives rectified, so it is pinhole by construction: no distortion.
        self.intr_depth = Intrinsics.from_matrix(k_c, dw, dh, None)

        T = calib.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C,
                                      dai.CameraBoardSocket.CAM_A,
                                      unit=dai.LengthUnit.METER)
        T = np.array(T, dtype=np.float64)
        if T.shape != (4, 4):
            T = np.vstack([np.array(T[:3]), [0, 0, 0, 1]])
        self.T_depth_to_rgb = T

        self.info.update({
            "baseline_mm": round(float(np.linalg.norm(T[:3, 3])) * 1000, 2),
            "rgb_fx": round(self.intr_rgb.fx, 2),
            "depth_fx": round(self.intr_depth.fx, 2),
            "rgb_dist": np.round(dist_a, 4).tolist(),
        })

    # ---------------------------------------------------------------- preview
    def preview(self):
        """Latest preview frame (BGR), or None if nothing has arrived yet."""
        with self._lock:
            if self._pipeline is None:
                return None
            pkt = self._queues["preview"].tryGet()
            return None if pkt is None else pkt.getCvFrame()

    def _drain(self, name):
        q = self._queues[name]
        last = None
        while True:
            pkt = q.tryGet()
            if pkt is None:
                return last
            last = pkt

    def capture(self, timeout: float = 12.0, temporal_frames: int = 3) -> Capture:
        """Wait for depth + one full-res RGB still, and pair them.

        ``temporal_frames`` depth frames are combined with a per-pixel median.
        Stereo noise is close to independent frame to frame, so 3 frames cut it
        by ~1.7x for free -- and the subject is a stationary object on a table,
        so there is nothing to lose. Costs about 0.4 s.
        """
        import cv2

        # The first second after the pipeline starts is not trustworthy: auto
        # exposure is still settling and the stereo matcher has not converged.
        # Capturing cold costs ~4x the valid depth, so wait it out.
        warm = 1.5 - (time.time() - getattr(self, "_opened_at", 0.0))
        if warm > 0:
            time.sleep(warm)

        t0 = time.time()
        frames: list = []
        rgb_pkt = None
        with self._lock:
            if self._pipeline is None:
                raise RuntimeError("camera not open")
            self._drain("depth")
            self._drain("rgb")
            while time.time() - t0 < timeout:
                pkt = self._queues["depth"].tryGet()
                if pkt is not None and len(frames) < temporal_frames:
                    frames.append(pkt.getFrame())
                if rgb_pkt is None:
                    rgb_pkt = self._drain("rgb")
                if len(frames) >= temporal_frames and rgb_pkt is not None:
                    break
                time.sleep(0.015)

        if not frames or rgb_pkt is None:
            raise TimeoutError(
                f"timed out waiting for frames (depth={len(frames)}, "
                f"rgb={rgb_pkt is not None}) - is the device still connected?"
            )

        stack = np.stack(frames).astype(np.float32)
        stack[stack <= 0] = np.nan
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN columns
            depth = np.nanmedian(stack, axis=0) / 1000.0            # mm -> m
        depth[~np.isfinite(depth)] = np.nan

        try:
            raw = np.frombuffer(bytes(rgb_pkt.getData()), dtype=np.uint8)
        except TypeError:
            raw = np.asarray(rgb_pkt.getData(), dtype=np.uint8).ravel()
        rgb = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError("could not decode the JPEG still from the device")

        return Capture(
            rgb=rgb,
            depth=depth,
            intr_rgb=self.intr_rgb,
            intr_depth=self.intr_depth,
            T_depth_to_rgb=self.T_depth_to_rgb,
            meta={"capture_ms": round((time.time() - t0) * 1000), **self.info},
        )

    def close(self):
        with self._lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception:  # noqa: BLE001
                    pass
            self._pipeline = None
            self._queues = {}

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
