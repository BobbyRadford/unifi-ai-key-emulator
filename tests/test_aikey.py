import struct

from PIL import Image

from aikey import protocol
from aikey.models import crop, load_tags
from aikey.nlsearch import object_types
from aikey.tasks import _reverification


def test_packet_round_trip():
    data = protocol.encode({"type": "request", "action": "getInfo", "id": "1"}, {"a": 1})
    assert data[:4] == bytes([1, 1, 0, 0])
    assert protocol.decode(data) == [({"type": "request", "action": "getInfo", "id": "1"}, {"a": 1})]


def test_deflated_and_binary_parts():
    raw = protocol.encode_part(1, {"type": "event"}, deflate=True) + protocol.encode_part(2, b"\x00\x01")
    assert protocol.decode(raw) == [({"type": "event"}, b"\x00\x01")]


def test_empty_body_is_an_object():
    data = protocol.encode({"type": "response", "id": "x"})
    kind, fmt, _, _, n = struct.unpack_from(">BBBBI", data, len(data) - 10)
    assert data.endswith(b"{}")


def test_object_types():
    assert object_types("man in a red shirt") == ["person"]
    assert object_types("white pickup truck at night") == ["vehicle"]
    assert object_types("a dog and a mailman") == ["animal"]
    assert object_types("something blue") == []


def test_tags_are_protects_enabled_set():
    tags = load_tags()
    assert len(tags) == 5375 and "person" in tags and "red car" in tags


def test_crop_uses_per_mille_boxes():
    img = Image.new("RGB", (1000, 500))
    c = crop(img, [100, 100, 200, 400], pad=0)
    assert c.size == (200, 200)


def test_reverification_echoes_labels():
    out = _reverification({"params": {"camera": "c", "event": "e", "thumbnailMs": [5],
                                      "thumbnailMeta": [{"ts": 5, "roi": [{"trackerID": 3, "objectType": "person", "confidence": 87}]}]}})
    r = out["result"]["verificationResults"][0]
    assert r == {"thumbnailMs": 5, "trackerID": 3, "objectType": "person", "isValidDetection": True,
                 "detectedAs": "person", "detectionConfidence": 0.87}


def test_tracks_keep_largest_boxes_per_tracker():
    from aikey.tasks import _tracks
    meta = [{"ts": t, "roi": {"trackerId": 7, "coord": [0, 0, w, w]}} for t, w in ((1, 10), (2, 50), (3, 30), (4, 40))]
    meta.append({"ts": 5, "roi": [{"trackerID": 9, "coord": [0, 0, 5, 5]}]})
    out = _tracks(meta)
    assert [ts for ts, _ in out[7]] == [2, 4, 3] and list(out) == [7, 9]


def test_descriptor_is_per_mille():
    from aikey.tasks import _descriptor
    d = _descriptor(Image.new("RGB", (2000, 1000)), (200, 100, 600, 500), "face", 100007, 0.91)
    assert d["coord"] == [100, 100, 200, 400] and d["confidenceLevel"] == 91 and d["trackerID"] == 100007


def test_face_alignment_maps_landmarks():
    import numpy as np
    from aikey.faces import ARCFACE_POINTS, _align
    img = Image.new("RGB", (400, 400))
    # Paint white dots at the landmarks scaled 2x and shifted; after alignment they
    # should land on ArcFace's template points.
    pts = ARCFACE_POINTS * 2 + 50
    px = img.load()
    for x, y in pts:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                px[int(x) + dx, int(y) + dy] = (255, 255, 255)
    out = np.asarray(_align(img, pts).convert("L"))
    for x, y in ARCFACE_POINTS:
        assert out[int(round(y)), int(round(x))] > 100


def test_decode_audio_resamples_to_16k_mono():
    import io

    import av
    import numpy as np
    from aikey.models import decode_audio
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as c:
        s = c.add_stream("aac", rate=48000, layout="stereo")
        t = np.arange(48000 * 2) / 48000
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for i in range(0, len(tone), 1024):
            chunk = np.stack([tone[i:i + 1024]] * 2)
            if chunk.shape[1] < 1024:
                break
            f = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
            f.sample_rate = 48000
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)
    audio = decode_audio(buf.getvalue())
    assert audio.dtype == np.float32 and abs(len(audio) - 32000) < 1600 and 0.15 < np.abs(audio).max() < 0.5


def test_frame_exports_split_far_apart_moments():
    from aikey.nvr import _clusters
    assert _clusters([1000, 2000, 9000, 9500, 30000], 4000) == [[1000, 2000], [9000, 9500], [30000]]


def test_backend_resolution(monkeypatch, tmp_path):
    import pytest
    from aikey import hailo
    from aikey.__main__ import resolve_hef

    hef_file = tmp_path / "hailo" / "model.hef"
    hef_file.parent.mkdir()
    hef_file.write_bytes(b"hef")

    monkeypatch.setattr(hailo, "hef_path", lambda d, n: str(hef_file) if n == "model.hef" else None)
    assert resolve_hef("clip", str(tmp_path), "model.hef", "auto") == str(hef_file)
    assert resolve_hef("clip", str(tmp_path), "model.hef", "hailo") == str(hef_file)
    assert resolve_hef("clip", str(tmp_path), "model.hef", "cpu") is None
    assert resolve_hef("clip", str(tmp_path), "missing.hef", "auto") is None
    with pytest.raises(FileNotFoundError):
        resolve_hef("clip", str(tmp_path), "missing.hef", "hailo")
    with pytest.raises(ValueError):
        resolve_hef("clip", str(tmp_path), "model.hef", "gpu")
