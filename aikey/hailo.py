"""Hailo-10H versions of the slow models, for a key running on a Raspberry Pi 5 with an
AI HAT+ 2 (HailoRT 5.2.0).

The key uses one when its HEF is in <data>/hailo/, and the CPU model otherwise:
- clip_vit_l_14_laion2B_image_encoder.hef: Hailo's precompiled LAION-2B CLIP ViT-L/14
  image encoder, paired with laion/CLIP-ViT-L-14-laion2B-s32B-b82K for text embeddings
  on the CPU.
- osnet_x1_0.hef: Hailo's precompiled OSNet person re-ID model (512-d, zero-padded to
  1024-d for Protect's pgvector column while preserving exact L2 distances).
- Whisper-Small.hef: Hailo's precompiled Whisper. The CPU voice filter the key already
  uses (Silero, from faster-whisper) runs first, because Whisper makes up sentences
  from noise and the HEF has no filter of its own. It transcribes as English, where
  the CPU model detects the language.
The image HEFs take RGB as uint8 and normalize on the chip.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np
from PIL import Image

from aikey.models import REID_SCALE, Clip, _normalize, decode_audio

log = logging.getLogger("aikey.hailo")

CLIP_HEF = "clip_vit_l_14_laion2B_image_encoder.hef"
CLIP_HAILO_MODEL = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
REID_HEF = "osnet_x1_0.hef"
REID_DIM = 1024
WHISPER_HEF = "Whisper-Small.hef"
RATE = 16000

# Hailo's LAION-2B CLIP HEF bakes in OpenAI CLIP's mean/std while HF's LAION-2B weights
# expect [0.5, 0.5, 0.5] / [0.5, 0.5, 0.5]. Remapping uint8 pixels with this LUT makes
# the HEF's internal normalization produce exact [-1, 1] inputs without clipping.
_OAI_MEAN = np.array([122.7709383, 116.7460125, 104.09373615], dtype=np.float32)
_OAI_STD = np.array([68.5005327, 66.6321579, 70.32316305], dtype=np.float32)
_CLIP_NORM_MAX = 35.0

_vdevice = None
_vdevice_lock = threading.Lock()


def hef_path(data: str, name: str) -> str | None:
    """<data>/hailo/<name>, if it exists and a Hailo device is present."""
    path = os.path.join(data, "hailo", name)
    return path if os.path.exists(path) and os.path.exists("/dev/hailo0") else None


def vdevice():
    """The process's one VDevice. Its scheduler shares the chip between models."""
    global _vdevice
    with _vdevice_lock:
        if _vdevice is None:
            from hailo_platform import HailoSchedulingAlgorithm, VDevice

            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            _vdevice = VDevice(params)
        return _vdevice


class Hef:
    """A single-input, single-output HEF: uint8 NHWC images in, float32 rows out."""

    def __init__(self, path: str) -> None:
        from hailo_platform import FormatType

        self.model = vdevice().create_infer_model(path)
        self.model.input().set_format_type(FormatType.UINT8)
        self.model.output().set_format_type(FormatType.FLOAT32)
        self.out_shape = tuple(self.model.output().shape)
        self.configured = self.model.configure()
        self.lock = threading.Lock()
        log.info("Loaded %s: %s -> %s", os.path.basename(path), self.model.input().shape, self.out_shape)

    def run(self, x: np.ndarray) -> np.ndarray:
        # Keep input arrays referenced until run() completes so buffers aren't freed early.
        ins = [np.ascontiguousarray(img) for img in x]
        outs = [np.empty(self.out_shape, np.float32) for _ in ins]
        with self.lock:
            bindings = []
            for img, out in zip(ins, outs):
                b = self.configured.create_bindings()
                b.input().set_buffer(img)
                b.output().set_buffer(out)
                bindings.append(b)
            self.configured.run(bindings, 10000)
        return np.stack([o.reshape(-1) for o in outs])


def _uint8(proc, images: list[Image.Image]) -> np.ndarray:
    """A Hugging Face image processor's resize and crop, without its normalization."""
    x = proc(images=images, do_rescale=False, do_normalize=False, return_tensors="np")["pixel_values"]
    return np.clip(np.rint(x), 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)


class HailoClip(Clip):
    """LAION-2B CLIP with its image tower on the Hailo."""

    def __init__(self, cache_dir: str, threads: int, path: str) -> None:
        super().__init__(cache_dir, threads, model_name=CLIP_HAILO_MODEL)
        self.hef = Hef(path)

    def image(self, images: list[Image.Image]) -> np.ndarray:
        x = _uint8(self.proc.image_processor, images)
        x_adj = np.clip(np.rint(_OAI_MEAN + (x.astype(np.float32) - 127.5) * (_OAI_STD / 127.5)),
                        0, 255).astype(np.uint8)
        raw = self.hef.run(x_adj)
        norms = np.linalg.norm(raw, axis=1)
        out = _normalize(raw)
        # Rare quantized activation spikes (norm >> ~20) fall back to CPU for that crop.
        bad = [i for i, n in enumerate(norms) if not np.isfinite(n) or n > _CLIP_NORM_MAX]
        if bad:
            log.warning("CLIP HEF norm spike on %d/%d crop(s); falling back to CPU", len(bad), len(images))
            out[bad] = super().image([images[i] for i in bad])
        return out


class HailoReid:
    """Person re-ID on the Hailo: OSNet (512-d) zero-padded to 1024-d, unit length times REID_SCALE."""

    def __init__(self, path: str) -> None:
        self.hef = Hef(path)

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        x = np.stack([np.asarray(im.convert("RGB").resize((128, 256), Image.BILINEAR), dtype=np.uint8)
                      for im in images])
        vec = _normalize(self.hef.run(x)) * REID_SCALE
        if vec.shape[1] < REID_DIM:
            vec = np.pad(vec, ((0, 0), (0, REID_DIM - vec.shape[1])))
        return vec.astype(np.float32)


class HailoWhisper:
    """Whisper-Small on the Hailo, run on the stretches the voice filter keeps."""

    def __init__(self, path: str) -> None:
        from hailo_platform.genai import Speech2Text

        self.s2t = Speech2Text(vdevice(), path)
        self.lock = threading.Lock()

    def transcribe(self, mp4: bytes) -> list[tuple[float, float, str]]:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        audio = decode_audio(mp4)
        out = []
        for span in get_speech_timestamps(audio, VadOptions()) if audio.size else []:
            off = span["start"] / RATE
            with self.lock:
                segs = self.s2t.generate_all_segments(audio[span["start"]:span["end"]], language="en",
                                                      timeout_ms=120000)
            out += [(off + s.start_sec, off + s.end_sec, s.text.strip()) for s in segs if s.text.strip()]
        log.info("Transcribed %d segments on the Hailo", len(out))
        return out
