#!/usr/bin/env bash
# Overlapped queue: one process builds corpora + teacher labels (CPU) for every study in order, while another trains and
# evaluates (GPU) as soon as each arm's data is ready. Results are identical to scripts/run_queue.sh; only wall time changes.
# Usage: scripts/run_pipeline.sh configs/study/a.yaml configs/study/b.yaml ...
set -u
cd "$(dirname "$0")/.."
(for cfg in "$@"; do uv run python -W ignore scripts/study.py "$cfg" --prepare-only > "runs/$(basename "$cfg" .yaml).prepare.log" 2>&1; done) &
prep=$!
export PIPELINE_PREP_PID=$prep  # the trainer fails fast if the preparer dies (see wait_for_data)
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  echo "$(date '+%F %T') start $name (pipelined)" >> runs/queue.log
  uv run python -W ignore scripts/study.py "$cfg" --wait-for-data > "runs/$name.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') end $name (exit $rc)" >> runs/queue.log
  uv run python scripts/build_web_data.py > /dev/null 2>&1
done
wait $prep
