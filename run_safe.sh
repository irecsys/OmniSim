#!/bin/bash
# run_safe.sh — wrapper that ensures killing this script also kills the Python child.
# Usage: bash run_safe.sh [run.py arguments...]

CHILD_PID=""

_cleanup() {
    if [ -n "$CHILD_PID" ]; then
        kill -9 "$CHILD_PID" 2>/dev/null || true
    fi
}
trap _cleanup EXIT SIGTERM SIGINT SIGHUP

python run.py "$@" &
CHILD_PID=$!
wait $CHILD_PID
