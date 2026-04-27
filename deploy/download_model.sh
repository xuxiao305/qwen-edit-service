#!/usr/bin/env bash
# Background downloader for Qwen-Image-Edit-2511 from hf-mirror.
# Resumable. Logs to $LOG.
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="${REPO:-Qwen/Qwen-Image-Edit-2511}"
DEST="${DEST:-/project/qwen_edit/models/Qwen-Image-Edit-2511}"
LOG="${LOG:-/project/qwen_edit/download.log}"

mkdir -p "$DEST"
mkdir -p "$(dirname "$LOG")"

echo "[$(date)] Start download $REPO -> $DEST" >> "$LOG"

wget -q -O /tmp/repo.json --tries=3 --timeout=30 "$HF_ENDPOINT/api/models/$REPO"
FILES=$(python3 -c "import json; d=json.load(open('/tmp/repo.json')); [print(s['rfilename']) for s in d.get('siblings',[])]")

for f in $FILES; do
  url="$HF_ENDPOINT/$REPO/resolve/main/$f"
  out="$DEST/$f"
  mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then
    echo "[$(date)] SKIP (exists) $f" >> "$LOG"
    continue
  fi
  echo "[$(date)] GET $f" >> "$LOG"
  wget -q -c -O "$out" --tries=5 --timeout=60 "$url" 2>>"$LOG"
  if [ $? -ne 0 ]; then
    echo "[$(date)] FAIL $f" >> "$LOG"
  else
    sz=$(stat -c%s "$out")
    echo "[$(date)] OK  $f ($sz bytes)" >> "$LOG"
  fi
done

echo "[$(date)] DONE. Total size:" >> "$LOG"
du -sh "$DEST" >> "$LOG"
