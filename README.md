# UniFi AI Key Emulator

An open-source emulator for the UniFi Protect AI Key (`UP-AI-KEY`). Adopt it into UniFi Protect to enable AI Key features across every camera on your NVR using your own hardware:

- Natural-language search ("man in a red shirt", "white pickup truck") and search by photo.
- Object tags on events, chosen from Protect's tag list (`aikey/ram_tags.csv`).
- Person re-identification across cameras over the last 24 hours.
- Speech to text on speech events.
- Face and license plate recognition for cameras that do not run them on-camera.

Tested against UniFi Protect 7.3. Protect 7.3 does not use a VLM for these features, so no VLM is required.

## Prerequisites (both platforms)

Before starting the containers on any machine, prepare the `data/` directory once:

1. Generate a client certificate and keep it safe, since Protect pins its fingerprint at adoption:
   ```sh
   mkdir -p data/smartface
   openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 7300 \
     -subj "/CN=UP-AI-KEY" -keyout data/client.key -out data/client.crt
   ```
2. Export `yolo11s.onnx` for person re-identification detection into `data/`:
   ```sh
   docker run --rm -v "$PWD/data:/out" -w /out python:3.12-slim sh -c \
     "apt-get update -qq && apt-get install -y -qq libxcb1 libgl1 libglib2.0-0 >/dev/null &&
      pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu &&
      pip install -q ultralytics onnx onnxslim && yolo export model=yolo11s.pt format=onnx imgsz=640 opset=17"
   ```
3. Copy Ubiquiti's SmartFace models from your Protect NVR into `data/smartface/` (they are Ubiquiti proprietary files and are not included in this repository):
   ```sh
   scp 'root@<nvr-ip>:/usr/lib/ubnt_face_model/SmartFace_{extract,pose}.*' data/smartface/
   ```
4. If your NVR is on an isolated VLAN, add a firewall rule allowing TCP port 5432 from the NVR IP to this host's IP so Protect can connect to `ai-key-db`.
5. Pick an unused MAC address for your emulated AI Key (any locally administered or unused `28:70:4E:...` address works) and fetch a one-time adoption token from Protect's API:
   ```
   GET https://<nvr-ip>/proxy/protect/api/devices/manage-payload?model=UP-AI-KEY
   ```
   Use the `mgmt.token` value as `AIKEY_ADOPTION_TOKEN` in `.env`, then clear it once Protect shows the AI Key connected.

## Deployment platforms

Two Docker Compose configurations share one codebase and one multi-target `Dockerfile`.

### 1. x86_64 Linux (CPU only, no GPU)

This is the default configuration in `docker-compose.yml` (`target: cpu`). Every model runs on CPU.

```sh
cp .env.example .env
# Edit .env with your NVR IP, host IP, MAC address, Postgres password, and adoption token.
docker compose up -d --build
```

### 2. Raspberry Pi 5 with AI HAT+ 2 (Hailo-10H)

Layer `docker-compose.hailo.yml` on top of `docker-compose.yml` (`target: hailo`). It installs HailoRT 5.2.0 inside the container and passes `/dev/hailo0` through.

1. Make sure HailoRT 5.2.0 host drivers and firmware are installed on the Pi and `/dev/hailo0` exists.
2. Download the matching `hailort_5.2.0_arm64.deb` and `hailort-5.2.0-cp312-cp312-linux_aarch64.whl` from the Hailo Developer Zone into `hailo/`.
3. Download Hailo's official precompiled Hailo-10H (v5.2.0) HEF models into `data/hailo/`:
   ```sh
   ./scripts/download-hailo-models.sh
   ```
   All three HEFs come straight from Hailo's public Model Zoo and developer downloads, so no custom-compiled models are needed.
4. Configure `.env`, build, and start:
   ```sh
   cp .env.example .env
   docker compose -f docker-compose.yml -f docker-compose.hailo.yml up -d --build
   ```

## Granular per-model backend selection

Each heavy model chooses its backend independently via `AIKEY_CLIP_BACKEND`, `AIKEY_REID_BACKEND`, and `AIKEY_WHISPER_BACKEND` (each accepts `auto`, `cpu`, or `hailo`, defaulting to `auto`):

- `auto` uses the Hailo-10H when both `/dev/hailo0` and that model's HEF exist in `data/hailo/`, and falls back to CPU otherwise.
- `cpu` forces CPU even if a Hailo device and HEF are present.
- `hailo` requires the Hailo device and HEF and fails fast if either is missing.

You can also download only a subset of HEFs (for example `./scripts/download-hailo-models.sh whisper reid`) or switch the person re-ID HEF with `AIKEY_REID_HEF=repvgg_a0_person_reid_512.hef` after running `./scripts/download-hailo-models.sh repvgg`.

Note that the CPU and Hailo paths for CLIP and re-ID use different models (OpenAI CLIP + DINOv2 on CPU vs LAION-2B CLIP + OSNet on Hailo). Their embedding spaces do not mix, so changing a model's backend on an existing deployment requires a fresh Postgres database or a retroactive re-index in Protect.

## Models

| Task | CPU backend | Hailo-10H backend |
|---|---|---|
| Image search & tags | `openai/clip-vit-large-patch14` (768-d) | `clip_vit_l_14_laion2B_image_encoder.hef` + `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` text tower (768-d) |
| Person re-ID | `facebook/dinov2-large` CLS token (1024-d) | `osnet_x1_0.hef` (512-d, zero-padded to 1024-d to preserve L2 distance) |
| Speech to text | `faster-whisper` (`small`, int8) | `Whisper-Small.hef` behind Silero VAD |
| Person detection (re-ID) | `yolo11s.onnx` (CPU) | `yolo11s.onnx` (CPU) |
| Faces | InsightFace SCRFD + Ubiquiti `SmartFace` via ncnn (CPU) | Same (CPU) |
| License plates | `fast-alpr` YOLOv9 + plate OCR (CPU) | Same (CPU) |

On Hailo, `clip_vit_l_14_laion2B_image_encoder.hef` bakes in OpenAI CLIP's channel mean and standard deviation while HuggingFace's LAION-2B weights expect `[0.5, 0.5, 0.5]`. `aikey/hailo.py` remaps uint8 pixels with an affine transform before inference so the chip receives exact `[-1, 1]` inputs without clipping, and falls back to CPU on the rare crop (~1%) that triggers a quantized activation norm spike.

## Measured performance

Replaying 49 captured Protect requests (10 audio-event images, 25 multi-crop object events, 7 video keyframe events, 7 speech events) with 4 CPU threads:

| Metric | x86_64 Linux VM (CPU) | Raspberry Pi 5 (CPU) | Raspberry Pi 5 + Hailo-10H |
|---|---|---|---|
| Total replay (49 requests) | 178.1 s | 295.4 s | 24.7 s |
| Audio-event image (`im`) median | 1.10 s | 2.42 s | 0.08 s |
| Object crops (`mi`) median / p90 | 4.18 s / 9.09 s | 4.72 s / 18.88 s | 0.18 s / 0.50 s |
| Video event (`rkf`) median | 3.29 s | 4.81 s | 0.63 s |
| Speech to text (`stt`) median | 3.74 s | 8.46 s | 0.74 s |
| CLIP image call median | 1.917 s | 2.437 s | 0.082 s |
| Person re-ID call median | 2.236 s | 2.610 s | 0.013 s |
| Whisper call median | 3.740 s | 8.458 s | 0.735 s |
| Peak RSS | 2.71 GB | 4.79 GB | 3.04 GB |

## How Protect talks to an AI Key

- An AI Key is a `ucp4` device that connects to `wss://<nvr>:7442` with a client certificate and one-time adoption token. Protect pins the certificate fingerprint on adoption, so no token is needed on restarts. `aikey/device.py` handles this connection and `aikey/protocol.py` handles packet framing.
- The key hosts a Postgres database with `pgvector` on port 5432 (`ai-key-db`). Protect connects directly, runs its own migrations, and performs every insert and vector search itself.
- Protect sends `recognizeKeyFrames`, `speechToText`, `enhanceImage`, and `RequestAI` over the device websocket. The key fetches frames and clips from Protect's internal API on port 7444 and posts results back (`aikey/nvr.py`, `aikey/tasks.py`).
- The key opens a second websocket (`wss://<nvr>:7444/wss/nl-search/v1`) to answer `NL_PARSE` and `IMAGE_SEARCH` queries (`aikey/nlsearch.py`).

## Checking and testing

- Watch task and search logs with `docker logs -f ai-key`.
- Inspect raw Protect messages in `data/messages.jsonl`.
- Count indexed detections with `docker exec ai-key-db psql -U postgres -d unifi-protect -c 'select count(*) from "ramDetections"'`.
- Run the unit tests with:
  ```sh
  docker run --rm --entrypoint python unifi-ai-key-emulator:cpu -m pytest -q tests
  ```

## Notes and limits

- Protect's `snapshot/generate` endpoint fails for recent moments, so the key extracts frames from short MP4 exports instead (`aikey/nvr.py`).
- Protect runs retroactive processing only once per AI Key. To trigger it again, reset `retroactiveProcessing` to `{"state": "not_started"}` in the NVR's `aiprocessors` table (Postgres port 5433), restart `unifi-protect`, and POST `/proxy/protect/api/aiprocessors/retroactive-processing/start`.
- Faces turned more than 60 degrees away are saved without an embedding so they do not mis-match known people.
- "AI summary" and deep understanding are not implemented. "Describe this moment" returns the frame's top tags.
