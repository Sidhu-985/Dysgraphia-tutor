#!/usr/bin/env bash
# Phase E orchestration against a REAL uvicorn server (synthetic traffic).
# Usage: scripts/run_phase_e.sh <python> <evidence_dir> [port]
set -u
PY="$1"; OUT="$2"; PORT="${3:-8770}"
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
export STORAGE_DB_PATH="$OUT/phase_e_live.db"
rm -f "$STORAGE_DB_PATH" "$OUT/failure_modes.json" "$OUT/failure_modes.json.restart_state.json"
UVICORN="$(dirname "$PY")/uvicorn"
start() {
  "$UVICORN" app.main:app --host 127.0.0.1 --port "$PORT" >> "$OUT/uvicorn.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 60); do curl -s "127.0.0.1:$PORT/health" >/dev/null && return 0; sleep 1; done
  echo "server did not start"; return 1
}
stop() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
: > "$OUT/uvicorn.log"
start
curl -s "127.0.0.1:$PORT/health" > "$OUT/health.json"
"$PY" scripts/live_scenarios.py --base-url "http://127.0.0.1:$PORT" --out "$OUT/live_scenarios.json"; S=$?
"$PY" scripts/phase_e_failure_modes.py --base-url "http://127.0.0.1:$PORT" --out "$OUT/failure_modes.json" --phase all; F=$?
"$PY" scripts/phase_e_failure_modes.py --base-url "http://127.0.0.1:$PORT" --out "$OUT/failure_modes.json" --phase before; B=$?
stop; sleep 1; start
"$PY" scripts/phase_e_failure_modes.py --base-url "http://127.0.0.1:$PORT" --out "$OUT/failure_modes.json" --phase after; A=$?
stop
echo "server 5xx responses: $(grep -c ' 500 ' "$OUT/uvicorn.log")"
echo "exit codes: scenarios=$S failure_modes=$F restart_before=$B restart_after=$A"
exit $(( S || F || B || A ))
