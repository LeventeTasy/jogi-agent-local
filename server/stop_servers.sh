#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VLLM_PID_FILE="$SCRIPT_DIR/vllm.pid"
CLM_PID_FILE="$SCRIPT_DIR/clm.pid"

echo "======================================"
echo " Stopping local LLM servers"
echo "======================================"

# CLM
if [ -f "$CLM_PID_FILE" ]; then
    CLM_PID=$(cat "$CLM_PID_FILE")

    if kill -0 "$CLM_PID" 2>/dev/null; then
        echo "Stopping CLM (PID $CLM_PID)..."
        kill "$CLM_PID"
    else
        echo "CLM process not running."
    fi

    rm -f "$CLM_PID_FILE"
else
    echo "CLM PID file not found."
fi


# vLLM
if [ -f "$VLLM_PID_FILE" ]; then
    VLLM_PID=$(cat "$VLLM_PID_FILE")

    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM (PID $VLLM_PID)..."
        kill "$VLLM_PID"
    else
        echo "vLLM process not running."
    fi

    rm -f "$VLLM_PID_FILE"
else
    echo "vLLM PID file not found."
fi

echo ""
echo "Servers stopped."