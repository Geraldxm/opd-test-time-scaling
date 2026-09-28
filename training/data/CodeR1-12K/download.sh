#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/downloads/train-00000-of-00001.parquet"
PART="$DEST.incomplete"
URL="https://huggingface.co/datasets/ganler/code-r1-12k/resolve/main/data/train-00000-of-00001.parquet"
SHA256="ba9df84a1a629849b3684a80ec0331aa011ec19f3033bc776888a4eabd08dc0d"

mkdir -p "$(dirname "$DEST")"
if [[ -f "$DEST" ]] && echo "$SHA256  $DEST" | sha256sum --check --status; then
  echo "Verified existing source: $DEST"
  exit 0
fi

curl --fail --location --retry 3 --continue-at - --output "$PART" "$URL"
if ! echo "$SHA256  $PART" | sha256sum --check; then
  rm -f "$PART"
  exit 1
fi
mv "$PART" "$DEST"
echo "Downloaded and verified: $DEST"
