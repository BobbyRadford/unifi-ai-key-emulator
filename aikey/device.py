"""The AI Key as a UniFi Protect device: the ucp4 websocket and its housekeeping requests.

Protect adopts a UCP4 device that connects to wss://<nvr>:7442 with a client
certificate and a one-time token from GET /proxy/protect/api/devices/manage-payload
?model=UP-AI-KEY. It pins the certificate's fingerprint, so later connections need
no token. After connecting, Protect sends requests (getInfo, diskInfo,
changeUserPassword, getTaskQueueInfo every 5 s, AI tasks, ...) and this answers them.
Handlers for AI work are registered by aikey.tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

import websockets

from aikey import protocol

log = logging.getLogger("aikey.device")

FIRMWARE = "2.2.8"           # newest AI-Key firmware; Protect forces updates outside ~2.2.0
SYSID = "0xa5f0"             # UP-AI-KEY
SKU = "UP-AI-KEY"

Handler = Callable[[dict, Any], Awaitable[Any]]


class DeviceError(Exception):
    """Raised by a handler to answer with a non-zero errorCode."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


class Device:
    def __init__(self, nvr: str, mac: str, ip: str, cert: str, key: str, state_path: str,
                 message_log: Optional[str] = None) -> None:
        self.nvr = nvr
        self.mac = mac.replace(":", "").upper()
        self.ip = ip
        self.cert, self.key = cert, key
        self.state_path = state_path
        self.message_log = message_log
        self.state = self._load_state()
        self.started = time.time()
        self.handlers: dict[str, Handler] = {}
        self.on_password: Optional[Callable[[str], Awaitable[None]]] = None
        self.queue_info: Callable[[], dict] = lambda: {}
        self.metrics: Callable[[], dict] = lambda: {}
        self.console: dict = {}
        self.settings: dict = {}
        self._ws = None
        self._pending: dict[str, asyncio.Future] = {}
        for action, fn in {
            "getInfo": self._get_info,
            "networkStatus": lambda h, b: self._ok({"linkSpeedMbps": 1000}),
            "changeUserPassword": self._change_password,
            "sshService": lambda h, b: self._ok({}),
            "diskInfo": lambda h, b: self._ok({"storageSize": "256"}),
            "updateTimezone": lambda h, b: self._ok({}),
            "changeAiInferAgentSettings": self._store_settings,
            "setConsoleInfo": self._console_info,
            "updateLcmSettings": lambda h, b: self._ok({}),
            "getTaskQueueInfo": self._queue_info,
            "systemMetrics": self._system_metrics,
            "locate": lambda h, b: self._ok({}),
            "reboot": lambda h, b: self._ok({}),
            "factoryReset": self._factory_reset,
            "updateFirmware": self._refuse,
            "firmwareState": lambda h, b: self._ok({}),
        }.items():
            self.handlers[action] = fn

    # ---- state -------------------------------------------------------------

    def _load_state(self) -> dict:
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {"deviceId": str(uuid.uuid4())}

    def save_state(self) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.state_path)

    # ---- connection --------------------------------------------------------

    def _ssl(self) -> ssl.SSLContext:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE        # the NVR's devices.crt is self-signed
        ctx.load_cert_chain(self.cert, self.key)
        return ctx

    def _headers(self, token: Optional[str]) -> dict:
        h = {"x-ident": self.mac, "x-type": SKU, "x-sysid": SYSID, "x-mode": "0",
             "x-ip": self.ip, "x-version": FIRMWARE, "x-device-id": self.state["deviceId"],
             "x-adopted": "true" if self.state.get("adopted") else "false"}
        if token:
            h["x-token"] = token
        return h

    async def run(self, token: Optional[str] = None) -> None:
        """Connects and reconnects forever. The token is only used until adoption."""
        delay = 2
        while True:
            use_token = None if self.state.get("adopted") else token
            try:
                async with websockets.connect(
                        f"wss://{self.nvr}:7442/", ssl=self._ssl(), subprotocols=["ucp4"],
                        additional_headers=self._headers(use_token), max_size=None,
                        ping_interval=15, ping_timeout=10, open_timeout=15) as ws:
                    self._ws = ws
                    log.info("Connected to %s", self.nvr)
                    if not self.state.get("adopted"):
                        self.state["adopted"] = True
                        self.save_state()
                    delay = 2
                    async for msg in ws:
                        if isinstance(msg, str):
                            msg = msg.encode()
                        for header, body in protocol.decode(msg):
                            asyncio.create_task(self._dispatch(header, body))
            except websockets.InvalidStatus as e:
                log.error("Protect refused the connection: %s", e.response.status_code)
            except (OSError, websockets.WebSocketException) as e:
                log.warning("Connection lost: %s", e)
            finally:
                self._ws = None
                for fut in self._pending.values():
                    fut.cancel()
                self._pending.clear()
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)

    async def send(self, header: dict, body: Any = None) -> None:
        if self._ws is None:
            raise ConnectionError("not connected")
        header.setdefault("timestamp", int(time.time() * 1000))
        self._log("out", header, body)
        await self._ws.send(protocol.encode(header, body))

    async def request(self, action: str, body: Any = None, timeout: float = 10) -> Any:
        """Sends a request to Protect and waits for its response body."""
        rid = str(uuid.uuid4())
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self.send({"type": "request", "action": action, "id": rid}, body)
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(rid, None)

    async def _dispatch(self, header: dict, body: Any) -> None:
        kind = header.get("type")
        if kind == "response":
            fut = self._pending.get(header.get("id"))
            if fut and not fut.done():
                fut.set_result(body)
            return
        self._log("in", header, body)
        if kind != "request":
            return
        action = header.get("action")
        reply = {"type": "response", "id": header.get("id"), "action": action,
                 "errorCode": 0, "error": ""}
        out: Any = {}
        fn = self.handlers.get(action)
        try:
            if fn is None:
                log.warning("Unhandled request %s: %s", action, _short(body))
                raise DeviceError(f"unsupported action {action}")
            out = await fn(header, body)
        except DeviceError as e:
            reply.update(errorCode=e.code, error=str(e))
        except Exception as e:   # answer anyway, so Protect doesn't wait out its timeout
            log.exception("Handler for %s failed", action)
            reply.update(errorCode=1, error=str(e) or type(e).__name__)
        try:
            await self.send(reply, out)
        except ConnectionError:
            pass

    def _log(self, direction: str, header: dict, body: Any) -> None:
        if header.get("action") in ("getTaskQueueInfo", "systemMetrics") or not self.message_log:
            return
        with open(self.message_log, "a") as f:
            f.write(json.dumps({"t": time.time(), "dir": direction, "header": header,
                                "body": _short(body, 4000)}) + "\n")

    # ---- housekeeping handlers --------------------------------------------

    @staticmethod
    async def _ok(body: Any) -> Any:
        return body

    async def _get_info(self, header: dict, body: Any) -> dict:
        flag = {"enabled": True, "version": "v1"}
        return {
            "version": FIRMWARE, "sw_version": FIRMWARE, "hardwareRevision": "1",
            "uptime": int(time.time() - self.started),
            "anonymousDeviceId": self.state["deviceId"], "poeType": "802.3bt",
            "featureFlags": {
                "storageSize": 256, "poeType": "802.3bt", "hasLcdScreen": False,
                "supportLocate": True, "supportFaceEnhancement": flag,
                "supportRetroactiveProcessing": flag, "supportRecognizeAnything": flag,
                "supportAiSummary": flag, "supportFaceRecognition": flag,
                "supportLicensePlateRecognition": flag, "supportImageSearch": flag,
                "supportPersonReId": flag, "supportTts": flag, "supportDeepMode": False,
                "aiMode": "basic",
            },
        }

    async def _change_password(self, header: dict, body: dict) -> dict:
        new = body.get("passwordNew")
        if new and new != self.state.get("password"):
            if self.on_password:
                await self.on_password(new)
            self.state["password"] = new
            self.save_state()
            log.info("Protect set a new device password")
        return {}

    async def _store_settings(self, header: dict, body: dict) -> dict:
        self.settings.update(body or {})
        return {}

    async def _console_info(self, header: dict, body: dict) -> dict:
        self.console = (body or {}).get("controller", {})
        return {}

    async def _queue_info(self, header: dict, body: Any) -> dict:
        info = {"UI_AUDIO_RAM": 0, "UI_AUTO_FACE_ENHANCE": 0, "UI_AUTO_RAM": 0,
                "UI_AUTO_STT": 0, "UI_MANUAL_FACE_ENHANCE": 0, "UI_TASK_NUM": 0}
        info.update(self.queue_info())
        return info

    async def _system_metrics(self, header: dict, body: Any) -> dict:
        load = os.getloadavg()[0] / (os.cpu_count() or 1) * 100
        try:
            with open("/proc/meminfo") as f:
                mem = {l.split(":")[0]: int(l.split()[1]) for l in f}
            mem_pct = 100 * (1 - mem["MemAvailable"] / mem["MemTotal"])
        except Exception:
            mem_pct = 0
        out = {"systemMetrics": [{"name": "CPU", "value": round(min(load, 100), 1)},
                                 {"name": "GPU", "value": 0},
                                 {"name": "Memory", "value": round(mem_pct, 1)}],
               "aiFeatureMetrics": {}}
        out.update(self.metrics())
        return out

    async def _factory_reset(self, header: dict, body: Any) -> dict:
        log.warning("Protect unadopted the AI Key; a new token is needed to adopt again")
        self.state.pop("adopted", None)
        self.state.pop("password", None)
        self.save_state()
        return {}

    async def _refuse(self, header: dict, body: Any) -> dict:
        raise DeviceError("firmware updates are not supported")


def _short(body: Any, limit: int = 400) -> Any:
    if isinstance(body, (bytes, bytearray)):
        return f"<{len(body)} bytes>"
    s = json.dumps(body, default=str)
    return body if len(s) <= limit else s[:limit] + "..."
