#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$$" > worker.pid
sleep "${WORKER_SLEEP_SECONDS:-0.05}"
printf '0\n' > worker.exitcode
touch worker.completed
: > DONE
