"""Framing of UniFi Protect's "ucp4" device websocket.

Each binary message holds one packet: a header part and a body part, each
  u8 kind (1 header, 2 body) | u8 format | u8 deflated | u8 0 | u32be length | bytes
with format 1 = JSON, 2 = UTF-8 string, 3 = raw bytes, and zlib when deflated.
The header is always JSON: {"type": "request"|"response"|"event"|"log", ...}.
(Protect's DeviceConnection, module 412614 in 7.3.74's service.js.)
"""

from __future__ import annotations

import json
import struct
import zlib
from typing import Any

HEADER, BODY = 1, 2
JSON_FMT, STRING_FMT, BYTES_FMT = 1, 2, 3


def encode_part(kind: int, obj: Any, deflate: bool = False) -> bytes:
    if isinstance(obj, (bytes, bytearray)):
        fmt, raw = BYTES_FMT, bytes(obj)
    elif isinstance(obj, str):
        fmt, raw = STRING_FMT, obj.encode()
    else:
        fmt, raw = JSON_FMT, json.dumps(obj, separators=(",", ":")).encode()
    if deflate:
        raw = zlib.compress(raw)
    return struct.pack(">BBBBI", kind, fmt, int(deflate), 0, len(raw)) + raw


def encode(header: dict, body: Any = None) -> bytes:
    """One packet. An empty JSON body breaks Protect's parser, so None becomes {}."""
    return encode_part(HEADER, header) + encode_part(BODY, {} if body is None else body)


def decode(data: bytes) -> list[tuple[dict, Any]]:
    """Splits a message into (header, body) packets."""
    parts, i = [], 0
    while i + 8 <= len(data):
        kind, fmt, deflated, _, n = struct.unpack_from(">BBBBI", data, i)
        raw = data[i + 8:i + 8 + n]
        i += 8 + n
        if deflated:
            raw = zlib.decompress(raw)
        if fmt == JSON_FMT:
            val = json.loads(raw) if raw else {}
        elif fmt == STRING_FMT:
            val = raw.decode(errors="replace")
        else:
            val = raw
        parts.append((kind, val))
    packets, header = [], None
    for kind, val in parts:
        if kind == HEADER:
            if header is not None:
                packets.append((header, None))
            header = val
        else:
            packets.append((header or {}, val))
            header = None
    if header is not None:
        packets.append((header, None))
    return packets
