# Emulated UniFi AI Key.
#
# Two build targets share the same Python dependencies:
#   cpu   (default) x86_64 or arm64 Linux with no GPU
#   hailo           Raspberry Pi 5 with an AI HAT+ 2 (Hailo-10H, HailoRT 5.2.0)
FROM python:3.12-slim-bookworm AS base

RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir transformers==4.46.3 faster-whisper==1.1.0 aiohttp==3.11.9 \
        websockets==14.1 onnxruntime==1.20.1 'fast-alpr[onnx]==0.4.0' ncnn==1.0.20261020 pillow numpy pytest

WORKDIR /app
ENV HF_HOME=/data/hf
ENTRYPOINT ["python", "-m", "aikey"]

FROM base AS cpu
COPY aikey /app/aikey
COPY tests /app/tests

FROM base AS hailo
# Place hailort_5.2.0_arm64.deb and hailort-5.2.0-cp312-cp312-linux_aarch64.whl in hailo/
# before building this target.
COPY hailo/hailort_5.2.0_arm64.deb hailo/hailort-5.2.0-cp312-cp312-linux_aarch64.whl /tmp/
RUN dpkg --unpack /tmp/hailort_5.2.0_arm64.deb && ldconfig \
    && pip install --no-cache-dir /tmp/hailort-5.2.0-cp312-cp312-linux_aarch64.whl \
    && rm -f /tmp/*.deb /tmp/*.whl
COPY aikey /app/aikey
COPY tests /app/tests
