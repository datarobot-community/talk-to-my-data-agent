#!/usr/bin/env sh

if [ -f "/.datarobot-pre-bundled" ]; then
    exit 0
else
    python3 -m pip install --no-cache-dir pipx
    PIPX_GLOBAL_BIN_DIR=/usr/bin python3 -m pipx install --global uv
    export UV_CACHE_DIR=.uv
    uv sync
fi

