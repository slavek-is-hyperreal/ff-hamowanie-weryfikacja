"""Blind test evaluation (PROTOKOL_v6 §5.8).

Step 1 (on the instance):  python -m hamexp.evaluate_test --results WYNIKI --hashes H.txt --plan ../plany/C.json --write-hashes
Step 3 (one command):      python -m hamexp.evaluate_test --results WYNIKI --hashes H.txt --plan ../plany/C.json --device cuda
Step 4 (locally):          python -m hamexp.evaluate_test --results WYNIKI --hashes H.txt --plan ../plany/C.json --verify-predictions

The hash file lists every stage-C run: ``<sha256 of checkpoint>  C/ckpt/<id>.pt`` for normal runs and
``NAN <sha256 of run record>  <id>`` for runs that stopped with NaN (§3.4). Evaluation refuses to run
unless every run of the plan is covered and every hash matches. Per-example correctness (bool) is
written to WYNIKI/test/C/<id>.pt for readouts main / no_l0 / cross (FF) or main (BP), on the dataset's
official test set and, for MNIST models, on QMNIST-50k.
"""
import argparse
import glob
import json
import os

import torch

from . import data, ff, train
from .runner import job_id, sha256


def _runs(results):
    return sorted(glob.glob(os.path.join(results, "C", "runs", "*.json")))


def check_complete(results, plan_path):
    """§5.8 / review K1-01: the set of stage-C run records must equal the plan; no .error files."""
    with open(plan_path) as f:
        planned = {job_id(j) for j in json.load(f)}
    have = {os.path.basename(r)[:-5] for r in _runs(results)}
    errors = glob.glob(os.path.join(results, "C", "runs", "*.error"))
    if planned != have or errors:
        raise SystemExit(f"Etap C niekompletny: brak {len(planned - have)}, nadmiarowe {len(have - planned)}, "
                         f"błędy {len(errors)} – ewaluacja testowa zabroniona (§5.8)")
    return sorted(os.path.join(results, "C", "runs", i + ".json") for i in planned)


def write_hashes(results, path, plan_path):
    """Step 1 of §5.8: checkpoints (or NAN record entries) plus every run record and validation file
    of stage C (failure status is fixed before the test is opened, §3.4; review K1-10)."""
    runs = check_complete(results, plan_path)
    lines = []
    for rp in runs:
        with open(rp) as f:
            r = json.load(f)
        if r["nan"]:
            lines.append(f"NAN {sha256(rp)}  {r['id']}")
        else:
            ck = os.path.join("C", "ckpt", r["id"] + ".pt")
            lines.append(f"{sha256(os.path.join(results, ck))}  {ck}")
        for rel in (os.path.join("C", "runs", r["id"] + ".json"), os.path.join("C", "zaslepione", r["id"] + ".json")):
            lines.append(f"FILE {sha256(os.path.join(results, rel))}  {rel}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(runs)


def read_hashes(path):
    ck, nan, files = {}, {}, {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "NAN":
                nan[parts[2]] = parts[1]
            elif parts[0] == "FILE":
                files[parts[2]] = parts[1]
            else:
                ck[parts[1]] = parts[0]
    return ck, nan, files


def verify(results, hashes_path, plan_path):
    runs = check_complete(results, plan_path)
    ck, nan, files = read_hashes(hashes_path)
    ids = {os.path.basename(r)[:-5] for r in runs}
    listed = {os.path.basename(k)[:-3] for k in ck} | set(nan)
    if listed != ids:
        raise SystemExit("plik hashy nie odpowiada dokładnie zbiorowi biegów Etapu C")
    need_files = {os.path.join("C", sub, i + ".json") for i in ids for sub in ("runs", "zaslepione")}
    if set(files) != need_files:
        raise SystemExit("plik hashy nie zawiera dokładnie wpisów FILE dla wszystkich rekordów i plików zaslepione Etapu C")
    for rel, d in files.items():
        if sha256(os.path.join(results, rel)) != d:
            raise SystemExit(f"hash mismatch: {rel}")
    for rp in runs:
        with open(rp) as f:
            r = json.load(f)
        if r["nan"]:
            if nan.get(r["id"]) != sha256(rp):
                raise SystemExit(f"NAN record hash mismatch: {r['id']}")
        else:
            rel = os.path.join("C", "ckpt", r["id"] + ".pt")
            d = sha256(os.path.join(results, rel))
            if ck.get(rel) != d or r["ckpt_sha256"] != d:
                raise SystemExit(f"checkpoint hash mismatch: {rel}")
    return runs


def evaluate(results, hashes_path, device, plan_path):
    runs = verify(results, hashes_path, plan_path)
    data.enable_test_access()
    sets = {}
    for ds, part in (("mnist", "test"), ("fmnist", "test"), ("qmnist", "test50k")):
        x, y = data.load(ds, part)
        sets[ds] = (data.normalize_pixels(x).to(device), y.to(device))
    out_dir = os.path.join(results, "test", "C")
    os.makedirs(out_dir, exist_ok=True)
    for rp in runs:
        with open(rp) as f:
            r = json.load(f)
        out_path = os.path.join(out_dir, r["id"] + ".pt")
        if os.path.exists(out_path):
            continue
        job = r["job"]
        names = [job["dataset"]] + (["qmnist"] if job["dataset"] == "mnist" else [])
        res = {}
        if r["nan"]:
            for nm in names:
                x, y = sets[nm]
                p = train.nan_predictions(len(y), device)
                res[nm] = {k: (v == y).cpu() for k, v in p.items()} if job["kind"] == "ff" else {"main": (p["main"] == y).cpu()}
        else:
            ckpt = torch.load(os.path.join(results, "C", "ckpt", r["id"] + ".pt"), map_location=device, weights_only=False)
            if job["kind"] == "ff":
                net = ff.FFNet(data.IN_DIM, job["cfg"]["lr"], job["cfg"].get("theta", 0.0), 0, device,
                               theta_layers=job["cfg"].get("theta_layers"))
                for layer, st in zip(net.layers, ckpt["state"]["layers"]):
                    layer.load_state_dict(st)
                net.bcm = ckpt["state"]["bcm"]
                for nm in names:
                    x, y = sets[nm]
                    p = ff.predict(net, x, data.encode, job["arm"])
                    res[nm] = {k: (v == y).cpu() for k, v in p.items()}
            else:
                net = ff.build_bp(784, 0, device)
                net.load_state_dict(ckpt["state"])
                for nm in names:
                    x, y = sets[nm]
                    res[nm] = {"main": (train.predict_bp(net, x) == y).cpu()}
        torch.save(res, out_path + ".tmp")
        os.replace(out_path + ".tmp", out_path)
    # §5.8 step 4: hashes of the prediction files, compared locally after download
    preds = sorted(glob.glob(os.path.join(out_dir, "*.pt")))
    with open(os.path.join(results, "HASHE_PREDYKCJI.txt"), "w") as f:
        f.write("\n".join(f"{sha256(p)}  {os.path.relpath(p, results)}" for p in preds) + "\n")
    return len(runs)


def verify_predictions(results, plan_path=None):
    """Run locally after download: every prediction file matches HASHE_PREDYKCJI.txt and (with a plan)
    the set of prediction files equals the set of planned stage-C runs."""
    listed = set()
    with open(os.path.join(results, "HASHE_PREDYKCJI.txt")) as f:
        for line in f:
            if line.strip():
                d, rel = line.split()
                listed.add(rel)
                if not os.path.exists(os.path.join(results, rel)) or sha256(os.path.join(results, rel)) != d:
                    raise SystemExit(f"prediction hash mismatch: {rel}")
    if plan_path:
        with open(plan_path) as f:
            want = {os.path.join("test", "C", job_id(j) + ".pt") for j in json.load(f)}
        if listed != want:
            raise SystemExit("zbiór plików predykcji nie odpowiada planowi Etapu C")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--hashes", required=True)
    ap.add_argument("--plan", required=True, help="stage C plan; incomplete stage C is refused (§5.8)")
    ap.add_argument("--verify-predictions", action="store_true", help="step 4: check downloaded predictions")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write-hashes", action="store_true")
    a = ap.parse_args(argv)
    if a.write_hashes:
        print(f"{write_hashes(a.results, a.hashes, a.plan)} runs hashed -> {a.hashes}")
        return
    if a.verify_predictions:
        verify_predictions(a.results, a.plan)
        print("predykcje zgodne z HASHE_PREDYKCJI.txt")
        return
    from .runner import setup_torch
    setup_torch(a.device)
    print(f"evaluated {evaluate(a.results, a.hashes, a.device, a.plan)} runs")


if __name__ == "__main__":
    main()
