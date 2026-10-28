#!/usr/bin/env bash
# Fetch the Chinese corpora used for draft-vocabulary frequency selection.
# Everything lands in datasets/raw/ which is gitignored.
set -u
cd "$(dirname "$0")/.." || exit 1

mkdir -p datasets/raw/coig_cqia datasets/raw/wikipedia_zh datasets/raw/firefly
LOG=logs/download.log
: > "$LOG"

fetch() { # url, dest
  local url="$1" dest="$2"
  echo "=== $(date +%H:%M:%S) $dest" >> "$LOG"
  curl -L --fail --retry 3 --retry-delay 2 -C - \
       --connect-timeout 20 --speed-time 60 --speed-limit 1024 \
       -o "$dest" "$url" >> "$LOG" 2>&1
  echo "curl=$? size=$(stat -c %s "$dest" 2>/dev/null || echo 0)" >> "$LOG"
}

fetch "https://huggingface.co/datasets/m-a-p/COIG-CQIA/resolve/main/COIG-CQIA-full.jsonl" \
      "datasets/raw/coig_cqia/COIG-CQIA-full.jsonl"

fetch "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00000-of-00006.parquet" \
      "datasets/raw/wikipedia_zh/train-00000-of-00006.parquet"

fetch "https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M/resolve/main/firefly-train-1.1M.jsonl" \
      "datasets/raw/firefly/firefly-train-1.1M.jsonl"

echo "=== final" >> "$LOG"
ls -l datasets/raw/*/ >> "$LOG" 2>&1
du -sh datasets/raw >> "$LOG" 2>&1
echo "DONE" >> "$LOG"
