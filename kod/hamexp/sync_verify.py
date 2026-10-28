"""Local confirmation of synced stage-C checkpoints (PROTOKOL_v6 §5.8): a C run counts as synced only
when its local checkpoint matches the sha256 in its run record (NaN runs have no checkpoint).

Usage:  python -m hamexp.sync_verify WYNIKI
"""
import glob
import json
import os
import sys

from .runner import sha256


def main(argv=None):
    results = (argv or sys.argv[1:])[0]
    ok, bad = 0, []
    for path in sorted(glob.glob(os.path.join(results, "C", "runs", "*.json"))):
        with open(path) as f:
            run = json.load(f)
        if run["nan"]:
            ok += 1
            continue
        ck = os.path.join(results, "C", "ckpt", run["id"] + ".pt")
        if os.path.exists(ck) and sha256(ck) == run["ckpt_sha256"]:
            ok += 1
        else:
            bad.append(run["id"])
    print(f"etap C potwierdzone lokalnie: {ok}  niezgodne/brak: {len(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
