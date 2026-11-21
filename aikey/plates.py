"""License plates for cameras that rely on the AI Key for plate recognition.

For those cameras Protect sends the vehicles it tracked (vehicleMeta). The key looks
for a plate on each vehicle with fast-alpr (a YOLOv9 plate detector and a plate OCR
model, ONNX) and reports the text on the vehicle with a crop of the plate.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger("aikey.plates")


class PlateReader:
    def __init__(self, threads: int) -> None:
        import onnxruntime as ort
        from fast_alpr import ALPR

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end",
                         ocr_model="cct-s-v2-global-model", ocr_device="cpu",
                         detector_sess_options=opts)
        self.lock = threading.Lock()

    def read(self, vehicle: Image.Image) -> Optional[tuple[str, float, Image.Image, tuple]]:
        """The best plate on a vehicle crop: (text, confidence, plate crop, box)."""
        frame = np.asarray(vehicle.convert("RGB"))[:, :, ::-1].copy()     # BGR, as OpenCV reads
        with self.lock:
            results = self.alpr.predict(frame)
        best = None
        for r in results:
            if not r.ocr or not r.ocr.text.strip():
                continue
            conf = r.ocr.confidence
            conf = float(np.mean(conf)) if isinstance(conf, (list, tuple)) else float(conf)
            score = conf * float(r.detection.confidence)
            if best is None or score > best[0]:
                best = (score, r)
        if best is None:
            return None
        score, r = best
        b = r.detection.bounding_box
        pad = int(0.15 * (b.y2 - b.y1))
        box = (max(0, b.x1 - pad), max(0, b.y1 - pad), b.x2 + pad, b.y2 + pad)
        text = r.ocr.text.strip().replace("_", "").upper()
        return text, round(score, 3), vehicle.crop(box), box
