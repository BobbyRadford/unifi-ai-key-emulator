"""Finding people in a frame for Protect's person re-identification.

Protect's "find this person" sends RequestAI :7445/segment-video with a camera and
time. The key answers each person in that frame with an outline (per-mille points
the UI draws as a clickable shape), a box, and the re-id embedding that Protect then
compares with every stored person's. YOLO11s finds the people; the outline is their
box, which is enough to pick someone.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from PIL import Image

log = logging.getLogger("aikey.people")


class PersonDetector:
    SIZE = 640

    def __init__(self, path: str, threads: int, min_conf: float = 0.4) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.min_conf = min_conf
        self.lock = threading.Lock()

    def detect(self, img: Image.Image) -> list[tuple[float, float, float, float, float]]:
        """People as (x0, y0, x1, y1, confidence) in pixels of img."""
        W, H = img.size
        scale = self.SIZE / max(W, H)
        w, h = int(W * scale), int(H * scale)
        canvas = Image.new("RGB", (self.SIZE, self.SIZE), (114, 114, 114))
        px, py = (self.SIZE - w) // 2, (self.SIZE - h) // 2
        canvas.paste(img.resize((w, h)), (px, py))
        x = np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255.0
        with self.lock:
            out = self.sess.run(None, {self.inp: x})[0][0]      # [84, 8400]
        conf = out[4]                                           # COCO class 0: person
        keep = conf >= self.min_conf
        boxes, conf = out[:4, keep].T, conf[keep]
        dets = []
        for (cx, cy, bw, bh), c in zip(boxes, conf):
            x0, y0 = (cx - bw / 2 - px) / scale, (cy - bh / 2 - py) / scale
            x1, y1 = (cx + bw / 2 - px) / scale, (cy + bh / 2 - py) / scale
            dets.append((max(0, x0), max(0, y0), min(W, x1), min(H, y1), float(c)))
        return _nms(dets)


def _nms(dets: list, iou: float = 0.5) -> list:
    dets = sorted(dets, key=lambda d: -d[4])
    out = []
    for d in dets:
        if all(_iou(d, o) < iou for o in out):
            out.append(d)
    return out


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def segments(img: Image.Image, detector: PersonDetector, reid) -> list[dict]:
    """The fetchSegments results for one frame."""
    img = img.convert("RGB")
    W, H = img.size
    people = detector.detect(img)
    if not people:
        return []
    crops = [img.crop((int(x0), int(y0), int(x1), int(y1))) for x0, y0, x1, y1, _ in people]
    embeds = reid.embed(crops)
    out = []
    for (x0, y0, x1, y1, _), emb in zip(people, embeds):
        X0, Y0, X1, Y1 = (round(1000 * float(v), 1) for v in (x0 / W, y0 / H, x1 / W, y1 / H))
        outline = [[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]]
        out.append({"embedding": [round(float(v), 5) for v in emb],
                    "bbox": [X0, Y0, round(X1 - X0, 1), round(Y1 - Y0, 1)],
                    "contour": outline, "contours": [outline], "objectType": "person"})
    log.info("Found %d people for re-id", len(out))
    return out
