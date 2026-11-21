"""Runs the emulated AI Key: python -m aikey --nvr <nvr-ip> --mac <mac> --ip <host-ip>"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from aikey.device import SKU, Device
from aikey.nlsearch import NlSearch
from aikey.nvr import Nvr
from aikey.tasks import Tasks

log = logging.getLogger("aikey")


BACKENDS = ("auto", "cpu", "hailo")


def resolve_hef(name: str, data: str, hef: str, mode: str = "auto") -> str | None:
    """Returns the HEF path when `mode` selects Hailo, or None to use CPU.

    auto  uses Hailo when /dev/hailo0 and <data>/hailo/<hef> exist, CPU otherwise.
    cpu   forces CPU even when a Hailo device and HEF are present.
    hailo requires Hailo and raises FileNotFoundError if the device or HEF is missing.
    """
    from aikey import hailo

    if mode not in BACKENDS:
        raise ValueError(f"Invalid backend {mode!r} for {name}; expected one of {BACKENDS}")
    if mode == "cpu":
        return None
    path = hailo.hef_path(data, hef)
    if mode == "hailo" and path is None:
        raise FileNotFoundError(f"Hailo backend requested for {name}, but /dev/hailo0 or {hef} is missing")
    return path


class Models:
    """Loads each model on first use, once, off the event loop."""

    def __init__(self, data: str, threads: int, whisper_size: str,
                 clip_backend: str = "auto", reid_backend: str = "auto",
                 whisper_backend: str = "auto", reid_hef: str | None = None) -> None:
        self.data, self.threads, self.whisper_size = data, threads, whisper_size
        self.clip_backend = clip_backend
        self.reid_backend = reid_backend
        self.whisper_backend = whisper_backend
        self.reid_hef = reid_hef or None
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    def _get(self, name: str, factory):
        task = self._tasks.get(name)
        if task is None or (task.done() and task.exception() is not None):
            log.info("Loading %s", name)
            task = self._tasks[name] = asyncio.ensure_future(self._load(factory))
        return asyncio.shield(task)

    async def _load(self, factory):
        # One at a time: transformers' lazy imports aren't thread-safe.
        async with self._lock:
            return await asyncio.to_thread(factory)

    # CLIP, re-id and Whisper run on a Hailo-10H when enabled and their HEFs are in <data>/hailo/.
    async def clip(self):
        from aikey import hailo
        from aikey.models import Clip
        path = resolve_hef("clip", self.data, hailo.CLIP_HEF, self.clip_backend)
        return await self._get("clip", lambda: hailo.HailoClip(self.data, self.threads, path) if path
                               else Clip(self.data, self.threads))

    async def reid(self):
        from aikey import hailo
        from aikey.models import Reid
        path = resolve_hef("reid", self.data, self.reid_hef or hailo.REID_HEF, self.reid_backend)
        return await self._get("reid", lambda: hailo.HailoReid(path) if path else Reid(self.threads))

    async def whisper(self):
        from aikey import hailo
        from aikey.models import Whisper
        path = resolve_hef("whisper", self.data, hailo.WHISPER_HEF, self.whisper_backend)
        return await self._get("whisper", lambda: hailo.HailoWhisper(path) if path
                               else Whisper(self.whisper_size, self.threads))

    async def faces(self):
        from aikey.faces import FaceModel
        return await self._get("faces", lambda: FaceModel(os.path.join(self.data, "models"),
                                                            os.path.join(self.data, "smartface"), self.threads))

    async def plates(self):
        from aikey.plates import PlateReader
        return await self._get("plates", lambda: PlateReader(self.threads))

    async def people(self):
        from aikey.people import PersonDetector
        path = os.path.join(self.data, "yolo11s.onnx")
        return await self._get("people", lambda: PersonDetector(path, self.threads))


def main() -> None:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nvr", default=env("AIKEY_NVR"), required=env("AIKEY_NVR") is None)
    p.add_argument("--mac", default=env("AIKEY_MAC"), required=env("AIKEY_MAC") is None)
    p.add_argument("--ip", default=env("AIKEY_IP"), required=env("AIKEY_IP") is None,
                   help="address Protect reaches this host's Postgres at")
    p.add_argument("--data", default=env("AIKEY_DATA", "/data"))
    p.add_argument("--token", default=env("AIKEY_TOKEN") or None,
                   help="one-time adoption token, from devices/manage-payload?model=UP-AI-KEY")
    p.add_argument("--threads", type=int, default=int(env("AIKEY_THREADS", "4")))
    p.add_argument("--whisper", default=env("AIKEY_WHISPER", "small"))
    p.add_argument("--clip-backend", choices=BACKENDS, default=env("AIKEY_CLIP_BACKEND", "auto"))
    p.add_argument("--reid-backend", choices=BACKENDS, default=env("AIKEY_REID_BACKEND", "auto"))
    p.add_argument("--whisper-backend", choices=BACKENDS, default=env("AIKEY_WHISPER_BACKEND", "auto"))
    p.add_argument("--reid-hef", default=env("AIKEY_REID_HEF") or None,
                   help="HEF filename in <data>/hailo/ for person re-id (default: osnet_x1_0.hef)")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    cert, key = os.path.join(a.data, "client.crt"), os.path.join(a.data, "client.key")
    mac = a.mac.replace(":", "").upper()

    async def run() -> None:
        device = Device(a.nvr, mac, a.ip, cert, key, os.path.join(a.data, "state.json"),
                        os.path.join(a.data, "messages.jsonl"))
        nvr = Nvr(a.nvr, mac, cert, key, SKU)
        models = Models(a.data, a.threads, a.whisper,
                        clip_backend=a.clip_backend, reid_backend=a.reid_backend,
                        whisper_backend=a.whisper_backend, reid_hef=a.reid_hef)
        tasks = Tasks(device, nvr, models)
        search = NlSearch(a.nvr, mac, cert, key, models.clip, nvr)
        # Load models up front: tag embeddings take a while on first start, and re-id
        # requests must be answered within 15 s.
        async def preload() -> None:
            for load in (models.clip, models.reid, models.people):
                try:
                    await load()
                except Exception:
                    log.exception("Loading a model failed; it will be retried on first use")
        asyncio.ensure_future(preload())
        await asyncio.gather(device.run(a.token), tasks.run(), search.run())

    asyncio.run(run())


if __name__ == "__main__":
    main()
