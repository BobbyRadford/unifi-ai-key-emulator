"""The models behind the AI Key, all on CPU.

- Clip: OpenAI CLIP ViT-L/14, the model Protect names ("clip-ViT-L-14") and tunes
  its search thresholds for (similarity = (cosine - 0.08) / 0.32). It embeds detection
  crops and search text, and picks tags from Protect's vocabulary zero-shot, by
  comparing a crop with every tag's text embedding.
- Reid: DINOv2-large, whose 1024-d CLS embedding fills Protect's re-id column. Protect
  matches re-id by L2 distance <= 20..30, so unit vectors are scaled by REID_SCALE.
- Whisper: faster-whisper for speech to text.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import threading
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger("aikey.models")

CLIP_MODEL = "openai/clip-vit-large-patch14"
REID_MODEL = "facebook/dinov2-large"
REID_SCALE = 28.0
TAGS_CSV = os.path.join(os.path.dirname(__file__), "ram_tags.csv")


def load_tags() -> list[str]:
    """The tags Protect accepts: enabled rows of its fixtures/ram_tags.csv."""
    with open(TAGS_CSV, newline="") as f:
        return [row[1].strip() for row in csv.reader(f, delimiter="\t")
                if len(row) >= 3 and row[2].strip() in ("true", "1")]


class Clip:
    def __init__(self, cache_dir: str, threads: int, model_name: str = CLIP_MODEL) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        torch.set_num_threads(threads)
        self.torch = torch
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.lock = threading.Lock()
        self.tags = load_tags()
        self.tag_embeds = self._tag_embeddings(cache_dir)

    def _tag_embeddings(self, cache_dir: str) -> np.ndarray:
        key = hashlib.md5(("\n".join(self.tags) + self.model_name).encode()).hexdigest()[:12]
        path = os.path.join(cache_dir, f"tag_embeds_{key}.npy")
        if os.path.exists(path):
            return np.load(path)
        log.info("Embedding %d tags (first start only)", len(self.tags))
        out = [self.text([f"a photo of a {t}." for t in self.tags[i:i + 256]])
               for i in range(0, len(self.tags), 256)]
        emb = np.concatenate(out)
        np.save(path, emb)
        return emb

    def image(self, images: list[Image.Image]) -> np.ndarray:
        """Unit-length 768-d embeddings, one row per image."""
        with self.lock, self.torch.no_grad():
            x = self.proc(images=images, return_tensors="pt")
            f = _tensor(self.model.get_image_features(**x))
        return _normalize(f.numpy())

    def text(self, texts: list[str]) -> np.ndarray:
        with self.lock, self.torch.no_grad():
            x = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True)
            f = _tensor(self.model.get_text_features(**x))
        return _normalize(f.numpy())

    def tag(self, emb: np.ndarray, top: int = 8, margin: float = 0.035,
            floor: float = 0.17) -> list[dict]:
        """Tags whose text is close to the image: the best few within `margin` of the best."""
        sims = self.tag_embeds @ emb
        idx = np.argsort(sims)[::-1][:top]
        best = float(sims[idx[0]])
        out = []
        for i in idx:
            s = float(sims[i])
            if s < floor or s < best - margin:
                break
            out.append({"tag": self.tags[i], "confScore": round(min(1.0, max(0.0, (s - 0.08) / 0.32)), 3)})
        return out


class Reid:
    def __init__(self, threads: int) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = AutoModel.from_pretrained(REID_MODEL).eval()
        self.proc = AutoImageProcessor.from_pretrained(REID_MODEL)
        self.lock = threading.Lock()

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """1024-d, unit length times REID_SCALE."""
        with self.lock, self.torch.no_grad():
            x = self.proc(images=[_pad_square(i) for i in images], return_tensors="pt")
            cls = self.model(**x).last_hidden_state[:, 0]
        return _normalize(cls.numpy()) * REID_SCALE


class Whisper:
    def __init__(self, size: str, threads: int) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads)
        self.lock = threading.Lock()

    def transcribe(self, mp4: bytes) -> list[tuple[float, float, str]]:
        """Speech in an MP4's audio as (start s, end s, text)."""
        audio = decode_audio(mp4)
        if audio.size == 0:
            return []
        with self.lock:
            segments, info = self.model.transcribe(audio, vad_filter=True, beam_size=1)
            out = [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
        log.info("Transcribed %d segments (%s)", len(out), info.language)
        return out


def decode_audio(data: bytes, rate: int = 16000) -> np.ndarray:
    """An MP4's first audio track as mono float32 samples at `rate`. faster-whisper's
    own decoder passes an argument PyAV 14 removed, so the key decodes it here."""
    import io

    import av

    with av.open(io.BytesIO(data)) as c:
        if not c.streams.audio:
            return np.zeros(0, np.float32)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        chunks = []
        for frame in c.decode(c.streams.audio[0]):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return (np.concatenate(chunks).astype(np.float32) / 32768) if chunks else np.zeros(0, np.float32)


def crop(img: Image.Image, coord: Optional[list], pad: float = 0.1) -> Image.Image:
    """Crops a per-mille [x, y, w, h] box, a little larger, as Protect's thumbnails are."""
    if not coord or len(coord) < 4:
        return img
    W, H = img.size
    x, y, w, h = (float(c) / 1000 for c in coord[:4])
    x0, y0 = max(0, (x - w * pad) * W), max(0, (y - h * pad) * H)
    x1, y1 = min(W, (x + w * (1 + pad)) * W), min(H, (y + h * (1 + pad)) * H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return img
    return img.crop((int(x0), int(y0), int(x1), int(y1)))


def _pad_square(img: Image.Image) -> Image.Image:
    """DINOv2's processor center-crops; pad tall person crops instead of cutting them."""
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), (124, 116, 104))
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def _tensor(out):
    if hasattr(out, "shape"):
        return out
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, name, None)
        if v is not None:
            return v
    raise TypeError(type(out))


def _normalize(a: np.ndarray) -> np.ndarray:
    return (a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)).astype(np.float32)
