#!/usr/bin/env bash

WORKING_DIR=$(pwd)

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR" || exit

export UV_CACHE_DIR="${WORKING_DIR}/.uv"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

if [ -f "/.datarobot-pre-bundled" ]; then
    echo "Building prebundled env ${WORKING_DIR} ${UV_CACHE_DIR}"

    PREBUNDLED_ROOT=/home/notebooks/talk-to-my-data-agent
    PREBUNDLED_UV_CACHE="$PREBUNDLED_ROOT/.cache/uv"
    PREBUNDLED_VENV="$PREBUNDLED_ROOT/app_backend/.venv"

    # Keep port 8080 alive to pass health checks while dependencies are being synchronized.
    python3 -m http.server 8080 >/tmp/prebundled-bootstrap-http.log 2>&1 &
    TEMP_SERVER_PID=$!

    mkdir -p "$UV_CACHE_DIR"
    if [ -d "$PREBUNDLED_UV_CACHE" ]; then
        echo "Copying UV cache"
        cp -r "$PREBUNDLED_UV_CACHE"/. "$UV_CACHE_DIR"/
    fi

    echo "Syncing .venv"

    if [ -d "$PREBUNDLED_VENV" ]; then
        rm -rf ".venv"
        cp -r "$PREBUNDLED_VENV" ".venv"
    fi

    echo "Running uv sync"

    uv sync

    echo "Starting real server"

    kill "$TEMP_SERVER_PID"
    wait "$TEMP_SERVER_PID"
    
    exec uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --proxy-headers --timeout-keep-alive 300
elif command -v uv >/dev/null 2>&1; then
    exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --proxy-headers --timeout-keep-alive 300
else
    # Fallback when uv is not installed (e.g. pre-built image + requirements.txt build path)
    # Set PYTHONPATH so app and core (src layout: core/src/core/) are importable
    export PYTHONPATH="${SCRIPT_DIR}/core/src:${SCRIPT_DIR}:${PYTHONPATH:-}"
    exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --proxy-headers --timeout-keep-alive 300
fi
