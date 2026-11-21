#!/bin/bash
# Downloads Hailo's official precompiled Hailo-10H (v5.2.0) HEF models into data/hailo/.
#
# Usage:
#   ./scripts/download-hailo-models.sh                   # downloads clip, reid (osnet), whisper
#   ./scripts/download-hailo-models.sh clip whisper      # downloads only the named models
#   ./scripts/download-hailo-models.sh repvgg            # optional faster Re-ID alternative
set -euo pipefail

DEST="${1:-}"
if [ -d "$DEST" ]; then
  shift
else
  DEST="$(cd "$(dirname "$0")/.." && pwd)/data/hailo"
fi
mkdir -p "$DEST"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(clip reid whisper)
fi

fetch() {
  local file="$1" url="$2"
  if [ -f "$DEST/$file" ]; then
    echo "already present: $DEST/$file"
    return
  fi
  echo "downloading $file..."
  curl -fL --progress-bar "$url" -o "$DEST/$file.part"
  mv "$DEST/$file.part" "$DEST/$file"
}

for m in "${MODELS[@]}"; do
  case "$m" in
    clip)
      fetch clip_vit_l_14_laion2B_image_encoder.hef \
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/clip_vit_l_14_laion2B_image_encoder.hef"
      ;;
    reid|osnet)
      fetch osnet_x1_0.hef \
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/osnet_x1_0.hef"
      ;;
    repvgg)
      fetch repvgg_a0_person_reid_512.hef \
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/repvgg_a0_person_reid_512.hef"
      ;;
    whisper)
      fetch Whisper-Small.hef \
        "https://dev-public.hailo.ai/v5.2.0/blob/Whisper-Small.hef"
      ;;
    *)
      echo "unknown model: $m (expected: clip, reid, repvgg, whisper)" >&2
      exit 1
      ;;
  esac
done

ls -lh "$DEST"
