#!/usr/bin/env bash
# §5.8 step 2: copy test data to the instance – ONLY after HASHE_CHECKPOINTOW.txt has been committed and pushed.
set -euo pipefail
HOST="$1"; PORT="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL=$(sha256sum "$ROOT/wyniki/HASHE_CHECKPOINTOW.txt" | cut -d' ' -f1)
REMOTE=$(gh api repos/slavek-is-hyperreal/ff-hamowanie-weryfikacja/contents/wyniki/HASHE_CHECKPOINTOW.txt --jq .content | base64 -d | sha256sum | cut -d' ' -f1)
[ "$LOCAL" = "$REMOTE" ] || { echo "plik hashy nie jest wypchnięty na publiczne repozytorium (lokalny != GitHub)"; exit 1; }
SSH="ssh -p $PORT"
rsync -az -e "$SSH" "$ROOT/dane/" "root@$HOST:/workspace/ham/dane/"
$SSH "root@$HOST" "cd /workspace/ham/kod && python -c 'from hamexp import data; [data.verify_file(r) for r in data.load_sums()]; print(\"hashe danych testowych: OK\")'"
