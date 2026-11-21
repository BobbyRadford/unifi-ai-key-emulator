"""Faces for cameras that rely on the AI Key for face recognition.

For those cameras Protect sends the people it tracked (personMeta). The key finds
a face inside each person (SCRFD from InsightFace's buffalo_s pack, onnxruntime),
embeds it and estimates its pose, and reports it linked to the person.

The embedding has to come from Ubiquiti's own face model. The NVR's face daemon
(ubnt_ai_daemon) matches every face, from G6 cameras and AI Keys alike, against one
database of known and unknown people, and the G6 cameras embed faces with the
MobileFaceNet the NVR ships as /usr/lib/ubnt_face_model/SmartFace_extract (ncnn).
On the cameras' own face crops it reproduces their stored embeddings at a cosine
similarity of about 0.9, where an InsightFace ArcFace scores 0. SmartFace_pose is
the head-pose model next to it. Both are copied from the NVR into
<data>/smartface; they aren't redistributable, so they aren't in this repository.
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
import zipfile

import numpy as np
from PIL import Image

log = logging.getLogger("aikey.faces")

PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
# Where the eyes, nose and mouth corners go in a 112x112 face (InsightFace's ArcFace
# template). SmartFace_extract takes faces aligned the same way.
ARCFACE_POINTS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                           [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def fetch_pack(models_dir: str) -> str:
    """Downloads and unpacks buffalo_s once; returns its directory."""
    out = os.path.join(models_dir, "buffalo_s")
    if os.path.exists(os.path.join(out, "det_500m.onnx")):
        return out
    os.makedirs(out, exist_ok=True)
    zpath = out + ".zip"
    log.info("Downloading %s", PACK_URL)
    urllib.request.urlretrieve(PACK_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.endswith(".onnx"):
                with open(os.path.join(out, os.path.basename(name)), "wb") as f:
                    f.write(z.read(name))
    os.remove(zpath)
    return out


def _ncnn(smartface_dir: str, name: str, threads: int):
    import ncnn

    net = ncnn.Net()
    net.opt.num_threads = threads
    net.opt.use_vulkan_compute = False
    base = os.path.join(smartface_dir, name)
    if net.load_param(base + ".param") or net.load_model(base + ".bin"):
        raise FileNotFoundError(f"{base}.param/.bin missing; copy them from the NVR's "
                                "/usr/lib/ubnt_face_model (see ai-key/README.md)")
    return net


def _run(net, x: np.ndarray, outputs: list[str], input_name: str = "input") -> list[np.ndarray]:
    """Runs an ncnn net on a CHW float image. ncnn.Mat wraps the array without
    copying, so x has to stay referenced until the outputs are copied out."""
    import ncnn

    x = np.ascontiguousarray(x, np.float32)
    mat = ncnn.Mat(x)
    ex = net.create_extractor()
    ex.input(input_name, mat)
    return [np.array(ex.extract(name)[1]).reshape(-1).copy() for name in outputs]


class FaceModel:
    DET_SIZE = 320
    POSE_SIZE = 64

    def __init__(self, models_dir: str, smartface_dir: str, threads: int) -> None:
        import onnxruntime as ort

        pack = fetch_pack(models_dir)
        ort.set_default_logger_severity(3)   # SCRFD warns about its dynamic output shapes on every run
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.det = ort.InferenceSession(os.path.join(pack, "det_500m.onnx"), opts,
                                        providers=["CPUExecutionProvider"])
        self.extract = _ncnn(smartface_dir, "SmartFace_extract", threads)
        self.posenet = _ncnn(smartface_dir, "SmartFace_pose", threads)
        self.lock = threading.Lock()

    def detect(self, img: Image.Image, min_score: float = 0.5) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """Faces as (box x0,y0,x1,y1 in pixels, score, 5 landmarks), best first."""
        W, H = img.size
        S = self.DET_SIZE
        scale = S / max(W, H)
        w, h = max(1, int(W * scale)), max(1, int(H * scale))
        canvas = np.zeros((S, S, 3), np.float32)
        canvas[:h, :w] = np.asarray(img.convert("RGB").resize((w, h)), np.float32)
        x = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None]
        with self.lock:
            outs = self.det.run(None, {self.det.get_inputs()[0].name: x})
        faces = []
        for i, stride in enumerate((8, 16, 32)):
            scores, boxes, kps = outs[i].reshape(-1), outs[i + 3].reshape(-1, 4), outs[i + 6].reshape(-1, 10)
            n = S // stride
            ys, xs = np.mgrid[:n, :n]
            centers = np.repeat(np.stack([xs.ravel(), ys.ravel()], 1) * stride, 2, axis=0).astype(np.float32)
            for j in np.where(scores >= min_score)[0]:
                cx, cy = centers[j]
                d = boxes[j] * stride
                box = np.array([cx - d[0], cy - d[1], cx + d[2], cy + d[3]]) / scale
                pts = (kps[j].reshape(5, 2) * stride + centers[j]) / scale
                faces.append((box, float(scores[j]), pts))
        faces.sort(key=lambda f: -f[1])
        kept = []
        for f in faces:
            if all(_iou(f[0], k[0]) < 0.4 for k in kept):
                kept.append(f)
        return kept

    def embed(self, img: Image.Image, landmarks: np.ndarray) -> np.ndarray:
        """Unit-length 512-d SmartFace embedding of the face aligned on its landmarks
        (BGR, (x - 127.5) / 128, as the cameras feed it)."""
        face = np.asarray(_align(img.convert("RGB"), landmarks), np.float32)[:, :, ::-1]
        with self.lock:
            (v,) = _run(self.extract, ((face - 127.5) / 128).transpose(2, 0, 1), ["embedding"])
        return (v / max(np.linalg.norm(v), 1e-9)).astype(np.float32)

    def pose(self, img: Image.Image, box) -> dict:
        """Head pose in degrees, {yaw, pitch, roll}, like the cameras' facePose. The
        model scores 66 bins of 3 degrees from -99 for each angle."""
        w, h = box[2] - box[0], box[3] - box[1]
        crop = img.convert("RGB").crop((box[0] - 0.2 * w, box[1] - 0.2 * h, box[2] + 0.2 * w, box[3] + 0.2 * h))
        x = (np.asarray(crop.resize((self.POSE_SIZE, self.POSE_SIZE)), np.float32) - 127.5) / 128
        with self.lock:
            outs = _run(self.posenet, x.transpose(2, 0, 1), ["fc_yaw", "fc_pitch", "fc_roll"])
        angles = []
        for v in outs:
            p = np.exp(v - v.max())
            angles.append(round(float((p / p.sum() * np.arange(66)).sum() * 3 - 99), 2))
        return dict(zip(("yaw", "pitch", "roll"), angles))


def _align(img: Image.Image, pts: np.ndarray) -> Image.Image:
    """Warps the face so its landmarks land on ARCFACE_POINTS (a similarity transform)."""
    src, dst = pts.astype(np.float64), ARCFACE_POINTS.astype(np.float64)
    # Least squares for dst = [a -b; b a] src + t, solved for the inverse (dst -> src),
    # which is what PIL's affine transform takes.
    A, y = [], []
    for (u, v), (X, Y) in zip(dst, src):
        A += [[u, -v, 1, 0], [v, u, 0, 1]]
        y += [X, Y]
    a, b, tx, ty = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)[0]
    return img.transform((112, 112), Image.AFFINE, (a, -b, tx, b, a, ty), resample=Image.BILINEAR)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
