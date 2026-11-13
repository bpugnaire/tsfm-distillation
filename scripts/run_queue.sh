#!/usr/bin/env bash
# Run study stages one after another, after an optional PID finishes (no overlapping runs on the Mac).
# Usage: scripts/run_queue.sh [--after PID] configs/study/a.yaml configs/study/b.yaml ...
set -u
cd "$(dirname "$0")/.."
if [ "${1:-}" = "--after" ]; then
  while kill -0 "$2" 2>/dev/null; do sleep 60; done
  shift 2
fi
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  echo "$(date '+%F %T') start $name" >> runs/queue.log
  uv run python -W ignore scripts/study.py "$cfg" > "runs/$name.log" 2>&1
  echo "$(date '+%F %T') end $name (exit $?)" >> runs/queue.log
  uv run python scripts/build_web_data.py > /dev/null 2>&1
done
