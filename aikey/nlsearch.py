"""The natural-language search channel: wss://<nvr>:7444/wss/nl-search/v1.

The AI Key opens it (ucp4 framing, X-Ident, the pinned client certificate) and
Protect sends requests on it:
- NL_PARSE {querySentence, model}: answer the query's CLIP text embedding, plus the
  object types it names. Object types make Protect search only those detections
  (by their smartDetectType label) ranked by similarity. With none, it falls back
  to a plain similarity search.
- IMAGE_SEARCH {imgUri}: answer the uploaded photo's CLIP image embedding.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import ssl
import time
from typing import Any

import websockets
from PIL import Image

from aikey import protocol

log = logging.getLogger("aikey.nlsearch")

OBJECT_WORDS = {
    "person": r"person|people|man|men|woman|women|boy|girl|kid|child|children|guy|lady|someone|"
              r"somebody|human|baby|toddler|pedestrian|delivery (?:man|driver)",
    "vehicle": r"car|cars|truck|trucks|van|suv|sedan|vehicle|vehicles|bus|motorcycle|pickup|jeep",
    "animal": r"dog|dogs|cat|cats|animal|animals|bird|birds|deer|pet|pets|squirrel|raccoon|fox|"
              r"horse|cow|bear|rabbit|coyote",
    "package": r"package|packages|parcel|box|boxes|delivery",
}


def object_types(sentence: str) -> list[str]:
    s = sentence.lower()
    return [t for t, words in OBJECT_WORDS.items() if re.search(rf"\b(?:{words})\b", s)]


class NlSearch:
    def __init__(self, nvr: str, mac: str, cert: str, key: str, clip_ready, nvr_api) -> None:
        self.url = f"wss://{nvr}:7444/wss/nl-search/v1"
        self.mac = mac
        self.cert, self.key = cert, key
        self.clip_ready = clip_ready      # async callable returning the Clip model
        self.api = nvr_api

    async def run(self) -> None:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.load_cert_chain(self.cert, self.key)
        delay = 5
        while True:
            try:
                async with websockets.connect(
                        self.url, ssl=ctx, subprotocols=["ucp4"], max_size=None,
                        additional_headers={"X-Ident": self.mac, "X-Type": "UP-AI-KEY"},
                        ping_interval=20, ping_timeout=20, open_timeout=15) as ws:
                    log.info("Search channel connected")
                    delay = 5
                    async for msg in ws:
                        if isinstance(msg, str):
                            msg = msg.encode()
                        for header, body in protocol.decode(msg):
                            if header.get("type") == "request":
                                asyncio.create_task(self._answer(ws, header, body))
            except websockets.InvalidStatus as e:
                log.warning("Search channel refused: %s", e.response.status_code)
            except (OSError, websockets.WebSocketException) as e:
                log.warning("Search channel lost: %s", e)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)

    async def _answer(self, ws, header: dict, body: Any) -> None:
        action = header.get("action")
        reply = {"type": "response", "id": header.get("id"), "action": action,
                 "errorCode": 0, "error": ""}
        out: Any = {}
        try:
            clip = await self.clip_ready()
            if action == "NL_PARSE":
                q = (body or {}).get("querySentence", "")
                emb = await asyncio.to_thread(clip.text, [q])
                out = {"keyTags": [], "objectTypes": object_types(q), "txtEmbed": emb[0].tolist(),
                       "model": "clip-ViT-L-14", "dim": 768, "exact_match": False}
                log.info("Search %r -> %s", q, out["objectTypes"] or "any object")
            elif action == "IMAGE_SEARCH":
                data = await self.api.get(body["imgUri"])
                img = Image.open(io.BytesIO(data)).convert("RGB")
                emb = await asyncio.to_thread(clip.image, [img])
                out = {"imgEmbed": emb[0].tolist()}
                log.info("Image search embedded")
            else:
                raise ValueError(f"unsupported action {action}")
        except Exception as e:
            log.exception("Search request %s failed", action)
            reply.update(errorCode=1, error=str(e) or type(e).__name__)
        reply["timestamp"] = int(time.time() * 1000)
        await ws.send(protocol.encode(reply, out))
