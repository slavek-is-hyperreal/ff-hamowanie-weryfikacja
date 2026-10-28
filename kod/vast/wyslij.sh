#!/usr/bin/env bash
# Copy code + data to a Vast.ai instance and verify data hashes there (PROTOKOL_v6 §6, §8).
# Usage: vast/wyslij.sh SSH_HOST SSH_PORT
set -euo pipefail
HOST="$1"; PORT="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # eksperyment_hamowanie_od_zera
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
$SSH "root@$HOST" "mkdir -p /workspace/ham/kod /workspace/ham/dane /workspace/ham/plany /workspace/ham/wyniki"
rsync -az --delete -e "$SSH" --exclude '__pycache__' "$ROOT/kod/" "root@$HOST:/workspace/ham/kod/"
rsync -az -e "$SSH" "$ROOT/dane/" "root@$HOST:/workspace/ham/dane/" --exclude "raw/*/t10k-*" --exclude "raw/qmnist/"
[ -d "$ROOT/plany" ] && rsync -az -e "$SSH" "$ROOT/plany/" "root@$HOST:/workspace/ham/plany/"
$SSH "root@$HOST" "cd /workspace/ham/kod && python -c 'from hamexp import data; [data.verify_file(r) for r in data.load_sums() if \"t10k\" not in r and \"qmnist\" not in r]; print(\"hashe danych: OK\")' && python -c 'import torch,numpy,platform; print(platform.python_version(), torch.__version__, torch.version.cuda, numpy.__version__, torch.cuda.get_device_name(0))'"
# Test data (MNIST/F-MNIST t10k, QMNIST) are NOT copied here; only after the checkpoint hashes are pushed (§5.8 step 2):
#   vast/wyslij_test.sh SSH_HOST SSH_PORT
