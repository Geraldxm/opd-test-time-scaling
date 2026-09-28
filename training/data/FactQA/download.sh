#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/downloads/train.parquet"
PART="$DEST.incomplete"
REVISION="b7d80abfee334a7a91cb377544f09180d58b34f6"
URL="https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/resolve/$REVISION/train.parquet"
SHA256="c3cc21e862a8469105de666101578cbff23cdc77e91a803cef102622c89cc4f6"

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
