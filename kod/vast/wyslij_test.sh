#!/usr/bin/env bash
# §5.8 step 2: copy test data to the instance – ONLY after HASHE_CHECKPOINTOW.txt has been committed and pushed.
set -euo pipefail
HOST="$1"; PORT="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
git -C "$ROOT" log --oneline -1 -- wyniki/HASHE_CHECKPOINTOW.txt | grep -q . || { echo "brak zacommitowanego pliku hashy"; exit 1; }
SSH="ssh -p $PORT"
rsync -az -e "$SSH" "$ROOT/dane/" "root@$HOST:/workspace/ham/dane/"
$SSH "root@$HOST" "cd /workspace/ham/kod && python -c 'from hamexp import data; [data.verify_file(r) for r in data.load_sums()]; print(\"hashe danych testowych: OK\")'"
