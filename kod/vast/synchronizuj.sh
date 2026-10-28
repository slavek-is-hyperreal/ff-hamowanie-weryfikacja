#!/usr/bin/env bash
# Pull results from one or more instances and print counts only (PROTOKOL_v6 §8). Never prints accuracies.
# Usage: vast/synchronizuj.sh PLAN.json HOST:PORT [HOST:PORT ...]
set -uo pipefail
PLAN="$1"; shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${HAMEXP_PY:-$ROOT/../eksperyment_ff_od_zera/.venv/bin/python}"
LOG="$ROOT/wyniki/sync.log"
exec 9>"$ROOT/wyniki/.sync.lock"; flock -n 9 || exit 0
{
  date '+%F %T'
  for hp in "$@"; do
    rsync -az --timeout=120 -e "ssh -p ${hp#*:} -o ConnectTimeout=30 -o BatchMode=yes" \
      "root@${hp%%:*}:/workspace/ham/wyniki/" "$ROOT/wyniki/" --exclude 'locks/' 2>&1 | grep -v -i 'welcome\|have fun'
  done
  cd "$ROOT/kod" && MKL_CBWR=COMPATIBLE "$PY" -m hamexp.sync_verify "$ROOT/wyniki" | head -1
  MKL_CBWR=COMPATIBLE "$PY" -m hamexp.monitor "$ROOT/plany/$PLAN" --results "$ROOT/wyniki"
} >> "$LOG" 2>&1
tail -1 "$LOG"
