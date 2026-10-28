#!/usr/bin/env bash
# Start a stage plan on the instance in the background (PROTOKOL_v6 §8).
# Usage: vast/uruchom.sh SSH_HOST SSH_PORT PLAN_NAME WORKERS [DEVICE=cpu]  (§8.3: all stages on CPU)
set -euo pipefail
HOST="$1"; PORT="$2"; PLAN="$3"; K="$4"; DEV="${5:-cpu}"
COMMIT="$(git -C "$(dirname "$0")" rev-parse HEAD)"
ssh -p "$PORT" "root@$HOST" "cd /workspace/ham/kod; MKL_CBWR=COMPATIBLE HAMEXP_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime HAMEXP_COMMIT=$COMMIT nohup python -m hamexp.runner ../plany/$PLAN --out ../wyniki --workers $K --device $DEV > ../wyniki/log_${PLAN%.json}.txt 2>&1 < /dev/null & echo uruchomiono $PLAN k=$K device=$DEV commit=$COMMIT"
