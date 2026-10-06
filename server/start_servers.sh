#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Hugging Face cache -> /data_shared
export HF_HOME=/data_shared/tasyl/.cache/huggingface
export HF_HUB_CACHE=/data_shared/tasyl/.cache/huggingface/hub
export HF_XET_CACHE=/data_shared/tasyl/.cache/huggingface/xet

# Temporary files -> /data_shared
export TMPDIR=/data_shared/tasyl/tmp

VLLM="$SCRIPT_DIR/vllm-env/bin/vllm"
CLM="$SCRIPT_DIR/.venv/bin/clm-serve"

VLLM_PORT=8090
CLM_PORT=8700

VLLM_PID_FILE="$SCRIPT_DIR/vllm.pid"
CLM_PID_FILE="$SCRIPT_DIR/clm.pid"

VLLM_LOG="$SCRIPT_DIR/vllm.log"
CLM_LOG="$SCRIPT_DIR/clm.log"

echo "======================================"
echo " Starting local LLM servers"
echo "======================================"

# Ha már futnak, ne indítsuk el újra
if [ -f "$VLLM_PID_FILE" ] && kill -0 "$(cat "$VLLM_PID_FILE")" 2>/dev/null; then
    echo "vLLM is already running (PID $(cat "$VLLM_PID_FILE"))"
else
    echo "Starting Qwen3-8B embedding server..."

    ./vllm-env/bin/vllm serve Qwen/Qwen3-8B \
      --served-model-name qwen3-8b \
      --task embed \
      --dtype half \
      --max-model-len 2048 \
      --enforce-eager \
      --port "$VLLM_PORT" \
      > "$VLLM_LOG" 2>&1 &

    VLLM_PID=$!
    echo "$VLLM_PID" > "$VLLM_PID_FILE"

    echo "   PID: $VLLM_PID"
    echo "   Log: $VLLM_LOG"
fi


echo ""
echo "Waiting for vLLM to become ready..."

until curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" > /dev/null 2>&1; do
    sleep 2
done

echo "vLLM is ready on port $VLLM_PORT"


# CLM indítása
if [ -f "$CLM_PID_FILE" ] && kill -0 "$(cat "$CLM_PID_FILE")" 2>/dev/null; then
    echo "CLM is already running (PID $(cat "$CLM_PID_FILE"))"
else
    echo ""
    echo "Starting CLM server..."

    ./\.venv/bin/clm-serve \
      --port "$CLM_PORT" \
      --emb-url "http://127.0.0.1:$VLLM_PORT/v1/embeddings" \
      > "$CLM_LOG" 2>&1 &

    CLM_PID=$!
    echo "$CLM_PID" > "$CLM_PID_FILE"

    echo "   PID: $CLM_PID"
    echo "   Log: $CLM_LOG"
fi


echo ""
echo "======================================"
echo " Servers started"
echo "======================================"
echo ""
echo "Qwen3-8B embeddings: http://127.0.0.1:$VLLM_PORT"
echo "CLM server:          http://127.0.0.1:$CLM_PORT"
echo ""
echo "Logs:"
echo "  tail -f $VLLM_LOG"
echo "  tail -f $CLM_LOG"
echo ""