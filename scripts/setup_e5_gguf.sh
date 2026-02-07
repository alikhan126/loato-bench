#!/usr/bin/env bash
# Download E5-Mistral-7B GGUF model for local embedding inference.
# Requires: huggingface-cli (pip install huggingface_hub)

set -euo pipefail

MODEL_DIR="data/models/e5-mistral"
REPO="second-state/E5-Mistral-7B-Instruct-Embedding-GGUF"
FILE="e5-mistral-7b-instruct-Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$FILE" ]; then
    echo "Model already downloaded: $MODEL_DIR/$FILE"
    exit 0
fi

echo "Downloading $FILE from $REPO ..."
huggingface-cli download "$REPO" "$FILE" --local-dir "$MODEL_DIR"
echo "Done: $MODEL_DIR/$FILE"
