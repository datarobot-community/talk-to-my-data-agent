#!/usr/bin/env bash

WORKING_DIR=$(pwd)

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR" || exit

export UV_CACHE_DIR="${WORKING_DIR}/.uv"

# pyproject.toml build path: deps installed system-wide by the DR platform.
# core/ is a symlink, so include core/src on PYTHONPATH for the src layout.
export PYTHONPATH="${SCRIPT_DIR}/core/src:${SCRIPT_DIR}:${PYTHONPATH:-}"

if [ "${UV_OFFLINE:-0}" = "1" ] && [ -n "${VENV_PATH:-}" ]; then
    # Airgapped dr-environment image: the platform installs deps offline from the image's
    # wheelhouse with `uv pip install --system`, which skips the kernel venv at the front of
    # PATH and installs into the system Python. Drop the venv from PATH so python3 resolves
    # to that same interpreter.
    PATH=":${PATH}:"
    while [[ "$PATH" == *":${VENV_PATH}/bin:"* ]]; do
        PATH="${PATH//:${VENV_PATH}\/bin:/:}"
    done
    PATH="${PATH#:}"
    export PATH="${PATH%:}"
fi

exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --proxy-headers --timeout-keep-alive 300
