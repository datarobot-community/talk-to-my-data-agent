#!/usr/bin/env bash

# Configure environment
export UV_CACHE_DIR=.uv

# Ensure we run from the app code directory
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR" || exit

if command -v uv >/dev/null 2>&1; then
    if [ ! -d ".venv" ]; then
        uv sync
    fi
    exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --proxy-headers --timeout-keep-alive 300
else
    # Fallback when uv is not installed (e.g. pre-built image + requirements.txt build path)
    # Set PYTHONPATH so app and core (src layout: core/src/core/) are importable
    export PYTHONPATH="${SCRIPT_DIR}/core/src:${SCRIPT_DIR}:${PYTHONPATH:-}"
    exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --proxy-headers --timeout-keep-alive 300
fi
