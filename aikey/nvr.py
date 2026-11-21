"""Protect's internal API for AI Keys: https://<nvr>:7444/internal/aiprocessors/...

The AI Key fetches media and posts results there. Protect lets a request through
when it carries X-Ident (the key's MAC) and X-Type, and presents the client
certificate it pinned at adoption.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import ssl
from typing import Any, Optional

import aiohttp
from PIL import Image

log = logging.getLogger("aikey.nvr")


class NvrError(RuntimeError):
    def __init__(self, method: str, path: str, status: int, text: str) -> None:
        super().__init__(f"{method} {path} -> {status}: {text[:200]}")
        self.status = status


class Nvr:
    def __init__(self, host: str, mac: str, cert: str, key: str, sku: str) -> None:
        self.base = f"https://{host}:7444"
        self.headers = {"X-Ident": mac, "X-Type": sku}
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.load_cert_chain(cert, key)
        self.ssl = ctx
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers, timeout=aiohttp.ClientTimeout(total=120),
                connector=aiohttp.TCPConnector(ssl=self.ssl, limit=8))
        return self._session

    def url(self, path: str) -> str:
        return path if path.startswith("http") else self.base + path

    async def get(self, path: str, params: Optional[dict] = None) -> bytes:
        async with self.session.get(self.url(path), params=params) as r:
            if r.status >= 300:
                raise NvrError("GET", path, r.status, await r.text())
            return await r.read()

    async def frames(self, camera: str, times: list[int], channel: int = 0) -> dict[int, Image.Image]:
        """Full frames at wall-clock times (ms), decoded from MP4 exports.

        Protect's snapshot/generate fails for recent moments and intermittently for
        older ones, so frames come from the recording. An export starts at the
        keyframe before `start` and ends exactly at `end`, so a moment lies
        (end - moment) before the clip's end. Moments close together share an
        export; a long event gets several short ones rather than one long 4K clip.
        """
        out: dict[int, Image.Image] = {}
        for group in _clusters(sorted(times), CLIP_GAP_MS):
            start, end = group[0] - 500, group[-1] + 500
            clip = await self.get("/internal/aiprocessors/video/export", {
                "camera": camera, "channel": str(channel), "type": "rotating", "start": str(start),
                "end": str(end), "mute": "true", "format": "mp4", "createEvent": "false"})
            out.update(await asyncio.to_thread(_pick_frames, clip, end, group))
        return out

    async def frame(self, camera: str, ts: int, channel: int = 0) -> Image.Image:
        return (await self.frames(camera, [ts], channel))[ts]

    async def post_json(self, path: str, body: Any) -> str:
        async with self.session.post(self.url(path), data=json.dumps(body),
                                     headers={"Content-Type": "application/json"}) as r:
            text = await r.text()
            if r.status >= 300:
                raise RuntimeError(f"POST {path} -> {r.status}: {text[:300]}")
            return text

    async def post_multipart(self, path: str, parts: list[tuple[str, Any, str]]) -> str:
        """parts: (name, value, content type). JSON values are serialized."""
        form = aiohttp.FormData()
        for name, value, ctype in parts:
            if ctype == "application/json":
                value = json.dumps(value)
            kw = {"filename": f"{name}.jpg"} if ctype.startswith("image/") else {}
            form.add_field(name, value, content_type=ctype, **kw)
        async with self.session.post(self.url(path), data=form) as r:
            text = await r.text()
            if r.status >= 300:
                raise RuntimeError(f"POST {path} -> {r.status}: {text[:300]}")
            return text


CLIP_GAP_MS = 4000    # moments further apart than this get separate exports


def _clusters(times: list[int], gap: int) -> list[list[int]]:
    """Sorted times split wherever two neighbors are more than `gap` apart."""
    groups: list[list[int]] = []
    for t in times:
        if groups and t - groups[-1][-1] <= gap:
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups


def _pick_frames(clip: bytes, end: int, times: list[int]) -> dict[int, Image.Image]:
    import av

    with av.open(io.BytesIO(clip)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or 30)
        if stream.frames:
            last = (stream.frames - 1) / fps
        else:
            last = c.duration / 1e6 - 1 / fps
        targets = {ts: last - (end - ts) / 1000 for ts in times}
        half = 0.5 / fps + 0.002
        out: dict[int, Image.Image] = {}
        f = None
        for f in c.decode(stream):
            t = float(f.time or 0)
            for ts, want in targets.items():
                if ts not in out and abs(t - want) <= half:   # converting 4K frames is slow
                    out[ts] = f.to_image()
            if len(out) == len(targets):
                break
        if f is None:
            raise RuntimeError("export had no frames")
        for ts in targets:
            out.setdefault(ts, f.to_image())                  # beyond the clip: its last frame
    return out
