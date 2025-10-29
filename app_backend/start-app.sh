#!/usr/bin/env bash

export PORT=${PORT:-"8080"}
DEV_MODE=${DEV_MODE:-false}
LOG_LEVEL="info"
EXTRA_OPTS=(--proxy-headers)

if [ "$(echo "$DEV_MODE" | tr '[:upper:]' '[:lower:]')" = "true" ]; then
  EXTRA_OPTS+=("--reload")
  # Disable telemetry in dev mode to avoid OTLP connection errors
  export DISABLE_TELEMETRY=true
fi

python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --log-level $LOG_LEVEL "${EXTRA_OPTS[@]}"
