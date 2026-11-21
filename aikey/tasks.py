"""The AI work Protect sends the AI Key over the device websocket.

Each request is acknowledged at once (Protect ignores the reply body), queued, and
its result posted to the request's resUrl on Protect's internal API:
- recognizeKeyFrames: per detected object, the frame at its thumbnail time is
  cropped to the object and gets a CLIP embedding (what search compares), tags, and
  for people a re-id embedding. For cameras that use the AI Key for faces and
  plates, Protect also sends the tracked people and vehicles, which get a face
  (aikey.faces) and plate (aikey.plates) lookup. Everything goes to recognize-anything
  as one multipart upload: "ram", "face" and "lpr" parts plus a JPEG per face/plate.
- speechToText: the event's audio is transcribed with Whisper.
- enhanceImage: answered with "unchanged".
- RequestAI: reverification is echoed (keeps the camera's own labels), on-demand
  "describe this moment" gets the frame's tags as text, and segment-video (person
  re-id) gets the people in the frame with their re-id embeddings.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from typing import Any, Optional

from PIL import Image

from aikey.device import Device
from aikey.models import crop
from aikey.nvr import Nvr, NvrError
from aikey.people import segments

log = logging.getLogger("aikey.tasks")


class Tasks:
    def __init__(self, device: Device, nvr: Nvr, models) -> None:
        self.device = device
        self.nvr = nvr
        self.models = models            # aikey.__main__.Models, loads lazily
        self.queue: asyncio.Queue = asyncio.Queue()
        self.busy = 0
        self.counts = {"recognizeKeyFrames": 0, "speechToText": 0}
        for action in ("recognizeKeyFrames", "speechToText", "enhanceImage", "RequestAI"):
            device.handlers[action] = self._enqueue
        device.queue_info = self.queue_info

    def queue_info(self) -> dict:
        waiting = [j[0] for j in self.queue._queue]   # type: ignore[attr-defined]
        return {"UI_TASK_NUM": len(waiting) + self.busy,
                "UI_AUTO_RAM": waiting.count("recognizeKeyFrames"),
                "UI_AUTO_STT": waiting.count("speechToText")}

    async def _enqueue(self, header: dict, body: Any) -> dict:
        body = body or {}
        target = body.get("targetUri", "")
        if "segment-video" in target or "on_demand_inference" in target:
            # Someone is waiting in the UI and Protect gives up after timeoutMs (15 s),
            # so these skip the queue.
            asyncio.create_task(self._now(body))
        else:
            self.queue.put_nowait((header.get("action"), body))
        return {}

    async def _now(self, body: dict) -> None:
        try:
            await self._do_RequestAI(body)
        except Exception:
            log.exception("RequestAI %s failed", body.get("targetUri"))

    async def run(self) -> None:
        while True:
            action, body = await self.queue.get()
            self.busy = 1
            t0 = time.time()
            try:
                await getattr(self, f"_do_{action}")(body)
                if action in self.counts:
                    self.counts[action] += 1
            except NvrError as e:
                if e.status != 404:
                    raise
                # Protect answers 404 when it has no recording for the moment asked.
                log.warning("%s skipped (event %s): %s", action, body.get("event"), e)
            except Exception:
                log.exception("%s failed (event %s)", action, body.get("event"))
            finally:
                self.busy = 0
            log.info("%s for event %s took %.1fs", action, body.get("event"), time.time() - t0)

    # ---- recognizeKeyFrames ------------------------------------------------

    async def _do_recognizeKeyFrames(self, body: dict) -> None:
        people, vehicles = _tracks(body.get("personMeta")), _tracks(body.get("vehicleMeta"))
        extra = sorted({ts for t in (people, vehicles) for boxes in t.values() for ts, _ in boxes})
        frames = await self._frames(body, extra)
        objects = await self._objects(body, frames)
        clip = await self.models.clip()
        t0 = time.time()
        thumbs, moments = [], {}
        if objects:
            embeds = await asyncio.to_thread(clip.image, [o["image"] for o in objects])
            persons = [i for i, o in enumerate(objects) if o["type"] == "person" and o["tracker"] is not None]
            reids = {}
            if persons:
                reid = await self.models.reid()
                vecs = await asyncio.to_thread(reid.embed, [objects[i]["image"] for i in persons])
                reids = dict(zip(persons, vecs))
            for i, o in enumerate(objects):
                tags = clip.tag(embeds[i])
                moments.setdefault(o["ms"], {})
                for t in tags:
                    old = moments[o["ms"]].get(t["tag"])
                    if old is None or old["confScore"] < t["confScore"]:
                        moments[o["ms"]][t["tag"]] = t
                if o["tracker"] is None:
                    continue
                entry = {"keyMomentMs": o["ms"], "trackerID": o["tracker"], "tags": tags,
                         "imgEmbed": [round(float(v), 6) for v in embeds[i]]}
                if i in reids:
                    entry["reidEmbed"] = [round(float(v), 5) for v in reids[i]]
                thumbs.append(entry)
                log.info("Event %s %s #%s: %s", body.get("event"), o["type"], o["tracker"],
                         ", ".join(t["tag"] for t in tags) or "no tags")
        infer = int((time.time() - t0) * 1000)
        ram = {"cameraId": body.get("camera"), "eventId": body.get("event"), "description": "",
               "status": "ok", "inferBoxMs": 0, "inferTagMs": infer, "inferTxtMs": 0,
               "preProcessMs": 0, "timeElapsedMs": infer,
               "keyMomentsTags": [{"keyMomentMs": ms, "tags": list(t.values())}
                                  for ms, t in sorted(moments.items())],
               "thumbnailTags": thumbs}
        parts = [("ram", ram, "application/json")]
        if people and frames:
            parts += await self._faces(body, people, frames)
        if vehicles and frames:
            parts += await self._plates(body, vehicles, frames)
        await self.nvr.post_multipart(body.get("resUrl", "/internal/aiprocessors/recognize-anything"), parts)

    async def _frames(self, body: dict, extra: list[int]) -> dict[int, Image.Image]:
        """Every frame a video request needs, from one export."""
        if body.get("ramType") not in ("video", "videoWithRecognition"):
            return {}
        times = {int(ms) for ms in body.get("thumbnailMs", [])} | set(extra)
        if not times:
            return {}
        return await self.nvr.frames(body["camera"], sorted(times), body.get("channel", 0))

    async def _objects(self, body: dict, frames: dict[int, Image.Image]) -> list[dict]:
        """The detected objects of a request: {ms, tracker, type, image}."""
        kind = body.get("ramType")
        out = []
        if kind in ("video", "videoWithRecognition"):
            for ms, meta in zip(body.get("thumbnailMs", []), body.get("thumbnailMeta", [])):
                for roi in _rois(meta):
                    out.append({"ms": int(ms), "tracker": _tracker(roi), "type": _type(roi),
                                "image": crop(frames[int(ms)], roi.get("coord"))})
        elif kind == "multipleImages":
            for im in body.get("images", []):
                out.append({"ms": int(im["keyMoment"]), "tracker": _int(im.get("trackerId")),
                            "type": im.get("objectType"),
                            "image": await self._image(self.nvr.get(im["reqUrl"]))})
        elif kind == "image":
            # Smart audio events: no objects, so only event-level tags.
            out.append({"ms": int(body.get("keyMoment") or body.get("start")), "tracker": None,
                        "type": None, "image": await self._image(self.nvr.get(body["reqUrl"]))})
        else:
            log.warning("Unknown ramType %r", kind)
        return out

    async def _faces(self, body: dict, people: dict, frames: dict) -> list:
        """The face part and face crops, for cameras that use the AI Key for faces."""
        model = await self.models.faces()
        t0 = time.time()
        tracks, attrs, snaps, images = [], {}, [], []
        for person, boxes in people.items():
            best = None
            for ts, coord in boxes:
                frame = frames[ts]
                pbox = _pixels(frame, coord)
                pcrop = frame.crop(pbox)
                found = await asyncio.to_thread(model.detect, pcrop, 0.6)
                if not found:
                    continue
                box, score, pts = found[0]
                if box[2] - box[0] < 20:
                    continue
                rank = score * (box[2] - box[0])
                if best is None or rank > best[0]:
                    best = (rank, ts, frame, pcrop, pbox, box, score, pts)
            if best is None:
                continue
            _, ts, frame, pcrop, pbox, box, score, pts = best
            pose = await asyncio.to_thread(model.pose, pcrop, box)
            fx0, fy0, fx1, fy1 = box[0] + pbox[0], box[1] + pbox[1], box[2] + pbox[0], box[3] + pbox[1]
            tid = FACE_TRACKER_OFFSET + person
            quality = round(score * 100)
            tracks.append({"timestamp": ts, "descriptors": [_descriptor(
                frame, (fx0, fy0, fx1, fy1), "face", tid, score)]})
            W, H = frame.size
            # The same attributes a G6 camera sends; landmarks are per mille of the frame.
            attrs[str(tid)] = {"confidence": quality, "qualityScore": quality, "facePose": pose,
                               "faceLandmarks": [round(1000 * (v + o) / n) for x, y in pts
                                                 for v, o, n in ((x, pbox[0], W), (y, pbox[1], H))],
                               "faceMask": {"confidence": quality, "val": "face"}, "matchedName": "",
                               "namesTopK": [], "objectType": "face", "topKCandidate": [],
                               "linkedPersonTrackerID": person}
            # Like the cameras, leave the embedding off faces turned too far away to
            # identify. Protect still saves them, as faces of no one in particular.
            if abs(pose["yaw"]) <= MAX_FACE_ANGLE and abs(pose["pitch"]) <= MAX_FACE_ANGLE:
                emb = await asyncio.to_thread(model.embed, pcrop, pts)
                attrs[str(tid)]["faceEmbed"] = [round(float(v), 6) for v in emb]
            snaps.append(_snapshot(ts, "face", tid))
            images.append((str(tid), _jpeg(frame.crop(_square((fx0, fy0, fx1, fy1), frame, 0.4))), "image/jpeg"))
        if not snaps:
            return []
        log.info("Event %s: %d face(s)", body.get("event"), len(snaps))
        ms = int((time.time() - t0) * 1000)
        face = {"cameraId": body.get("camera"), "eventId": body.get("event"), "status": "ok",
                "inferMs": ms, "preProcessMs": 0, "timeElapsedMs": ms, "eventTracks": tracks,
                "faceAttrs": attrs, "faceSnapshots": snaps}
        return [("face", face, "application/json")] + images

    async def _plates(self, body: dict, vehicles: dict, frames: dict) -> list:
        """The lpr part and plate crops, for cameras that use the AI Key for plates."""
        reader = await self.models.plates()
        t0 = time.time()
        tracks, attrs, snaps, images = [], {}, [], []
        for vehicle, boxes in vehicles.items():
            best = None
            for ts, coord in boxes:
                frame = frames[ts]
                vbox = _pixels(frame, coord)
                got = await asyncio.to_thread(reader.read, frame.crop(vbox))
                if got and (best is None or got[1] > best[1][1]):
                    best = (ts, got, vbox, frame)
            if best is None:
                continue
            ts, (text, conf, plate, pbox), vbox, frame = best
            box = (pbox[0] + vbox[0], pbox[1] + vbox[1], pbox[2] + vbox[0], pbox[3] + vbox[1])
            tracks.append({"timestamp": ts, "descriptors": [_descriptor(frame, box, "licensePlate", vehicle, conf, text)]})
            attrs[str(vehicle)] = {"matchedName": text, "namesTopK": [text], "objectType": "licensePlate",
                                   "topKCandidate": [{"confidence": conf, "name": text}]}
            snaps.append(_snapshot(ts, "licensePlate", vehicle))
            images.append((str(vehicle), _jpeg(plate), "image/jpeg"))
            log.info("Event %s: vehicle #%s plate %s (%.2f)", body.get("event"), vehicle, text, conf)
        if not snaps:
            return []
        ms = int((time.time() - t0) * 1000)
        lpr = {"cameraId": body.get("camera"), "eventId": body.get("event"), "status": "ok",
               "inferMs": ms, "preProcessMs": 0, "timeElapsedMs": ms, "eventTracks": tracks,
               "lprAttrs": attrs, "lprSnapshots": snaps}
        return [("lpr", lpr, "application/json")] + images

    @staticmethod
    async def _image(fetch) -> Image.Image:
        return Image.open(io.BytesIO(await fetch)).convert("RGB")

    # ---- speechToText ------------------------------------------------------

    async def _do_speechToText(self, body: dict) -> None:
        whisper = await self.models.whisper()
        audio = await self.nvr.get(body["reqUrl"])
        segs = await asyncio.to_thread(whisper.transcribe, audio)
        start = int(body.get("start") or 0)
        stt = [{"startMs": start + int(a * 1000), "endMs": start + int(b * 1000), "text": t}
               for a, b, t in segs]
        await self.nvr.post_json(body.get("resUrl", "/internal/aiprocessors/speech-to-text"),
                                 {"camera": body["camera"], "event": body["event"], "stt": stt})

    # ---- the rest ----------------------------------------------------------

    async def _do_enhanceImage(self, body: dict) -> None:
        await self.nvr.post_multipart(body.get("resUrl", "/internal/aiprocessors/image/enhanced"), [
            ("camera", str(body.get("camera", "")), "text/plain"),
            ("type", "face", "text/plain"),
            ("smartDetectObject", str(body.get("smartDetectObject", "")), "text/plain"),
            ("file", b"", "image/jpeg")])

    async def _do_RequestAI(self, body: dict) -> None:
        target = body.get("targetUri", "")
        payload = body.get("payload") or {}
        if "second_verifier" in target:
            await self.nvr.post_json(body["resUrl"], _reverification(payload))
        elif "on_demand_inference" in target:
            clip = await self.models.clip()
            img = await self.nvr.frame(payload["cameraId"], int(payload["timestamp"]))
            tags = clip.tag((await asyncio.to_thread(clip.image, [img]))[0], top=10, margin=0.05)
            text = "Likely in view: " + ", ".join(t["tag"] for t in tags) if tags else "Nothing recognized."
            await self.nvr.post_json(body["resUrl"], {"description": text})
        elif "segment-video" in target:
            frame = await self.nvr.frame(payload["cameraId"], int(payload["timestamp"]))
            detector, reid = await self.models.people(), await self.models.reid()
            results = await asyncio.to_thread(segments, frame, detector, reid)
            await self.nvr.post_json(body["resUrl"], {"results": results})
        else:
            log.warning("Unhandled RequestAI %s", target)


def _reverification(payload: dict) -> dict:
    """Confirms every object as the camera labelled it."""
    params = payload.get("params") or {}
    results = []
    for ms, meta in zip(params.get("thumbnailMs", []), params.get("thumbnailMeta", [])):
        for roi in (meta or {}).get("roi") or []:
            conf = float(roi.get("confidence") or 100)
            results.append({"thumbnailMs": ms, "trackerID": roi.get("trackerID"),
                            "objectType": roi.get("objectType"), "isValidDetection": True,
                            "detectedAs": roi.get("objectType"),
                            "detectionConfidence": conf / 100 if conf > 1 else conf})
    return {"model": "second_verifier_mlabel", "action": "classify", "inference_time_ms": 0,
            "result": {"cameraId": params.get("camera"), "eventId": params.get("event"),
                       "verificationResults": results, "status": "ok", "preProcessMs": 0,
                       "inferMs": 0, "timeElapsedMs": 0}}


MAX_FACE_ANGLE = 60             # degrees of yaw or pitch past which a face gets no embedding
FACE_TRACKER_OFFSET = 100000    # face trackers are new objects; keep their ids clear of the camera's


def _rois(meta) -> list[dict]:
    rois = (meta or {}).get("roi") or []
    return rois if isinstance(rois, list) else [rois]


def _tracker(roi: dict) -> Optional[int]:
    return _int(roi.get("trackerID", roi.get("trackerId")))


def _type(roi: dict) -> Optional[str]:
    return roi.get("objectType") or (roi.get("attributes") or {}).get("objectType")


def _tracks(meta: Optional[list], per_track: int = 3) -> dict[int, list[tuple[int, list]]]:
    """Per tracker id, its largest few boxes as (time ms, per-mille coord)."""
    tracks: dict[int, list] = {}
    for m in meta or []:
        for roi in _rois(m):
            tid, coord = _tracker(roi), roi.get("coord")
            if tid is None or not coord or len(coord) < 4 or m.get("ts") is None:
                continue
            tracks.setdefault(tid, []).append((int(m["ts"]), coord))
    return {t: sorted(b, key=lambda x: -(x[1][2] * x[1][3]))[:per_track] for t, b in tracks.items()}


def _pixels(img: Image.Image, coord: list) -> tuple[int, int, int, int]:
    W, H = img.size
    x, y, w, h = (float(c) / 1000 for c in coord[:4])
    return (max(0, int(x * W)), max(0, int(y * H)), min(W, int((x + w) * W)), min(H, int((y + h) * H)))


def _square(box, img: Image.Image, pad: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    W, H = img.size
    return (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
            int(min(W, cx + side / 2)), int(min(H, cy + side / 2)))


def _descriptor(img: Image.Image, box, kind: str, tracker: int, conf: float, name: str = "") -> dict:
    """A smart-detect track descriptor, as cameras send them (per-mille [x, y, w, h])."""
    W, H = img.size
    x0, y0, x1, y1 = box
    coord = [round(1000 * x0 / W), round(1000 * y0 / H), round(1000 * (x1 - x0) / W), round(1000 * (y1 - y0) / H)]
    return {"boxColor": "", "confidenceLevel": round(100 * float(conf)), "coord": coord, "coord3d": [],
            "lines": [], "name": name, "objectType": kind, "stationary": False, "tag": "",
            "trackerID": tracker, "zones": []}


def _snapshot(ts: int, kind: str, tracker: int) -> dict:
    return {"clockBestMonotonic": ts, "clockBestWall": ts, "smartDetectHeatmap": "",
            "smartDetectSnapshot": f"{tracker}.jpg", "smartDetectSnapshotName": f"{tracker}.jpg",
            "smartDetectSnapshotType": kind, "trackerID": tracker}


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
