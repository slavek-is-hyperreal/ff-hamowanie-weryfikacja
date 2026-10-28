"""Progress monitor (PROTOKOL_v6 §8.2): counts only — never shows any accuracy.

Usage:  python -m hamexp.monitor PLAN.json --results WYNIKI
"""
import argparse
import json
import os

from .runner import job_id


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--results", required=True)
    a = ap.parse_args(argv)
    with open(a.plan) as f:
        jobs = json.load(f)
    done = err = nan = 0
    for j in jobs:
        base = os.path.join(a.results, j["stage"], "runs", job_id(j))
        if os.path.exists(base + ".json"):
            done += 1
            with open(base + ".json") as f:
                nan += bool(json.load(f)["nan"])
        elif os.path.exists(base + ".error"):
            err += 1
    print(f"ukończone {done}/{len(jobs)}  błędy {err}  biegi z NaN {nan}")


if __name__ == "__main__":
    main()
