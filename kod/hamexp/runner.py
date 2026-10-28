"""Job runner (PROTOKOL_v6 §5, §8). Adapted from ``eksperyment_ff_od_zera/kod/ffexp/runner.py``.

Usage:  python -m hamexp.runner PLAN.json --out WYNIKI --workers K --device cuda

A plan is a JSON list of jobs. Each finished job writes (atomically):
  OUT/<stage>/runs/<id>.json        run record – never any accuracy (nan flag, epochs, E1/E4 logs, versions)
  OUT/<stage>/zaslepione/<id>.json  validation accuracy and curve (E6); read only by selection /
                                    choose_n scripts before step 3 of §5.8, which print no means
  OUT/C/ckpt/<id>.pt                weights (stage C only; NaN runs get no checkpoint, §3.4)
  OUT/C/zaslepione/<id>.pt          validation predictions (stage C)
Finished jobs are skipped on restart; failures write <id>.error.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback

CKPT_STAGES = {"C"}


def job_id(job) -> str:
    parts = [job["stage"], job["dataset"], job["arm"], f"s{job['seed']}"]
    if "point" in job:
        parts.append(f"p{job['point']}")
    return "__".join(parts)


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions(torch):
    v = {"python": platform.python_version(), "torch": torch.__version__, "host": platform.node(),
         "tf32": bool(torch.backends.cuda.matmul.allow_tf32), "MKL_CBWR": os.environ.get("MKL_CBWR")}
    if torch.cuda.is_available():
        v["cuda"] = torch.version.cuda
        v["gpu"] = torch.cuda.get_device_name(0)
        try:
            import subprocess
            v["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                         capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    v["code_commit"] = os.environ.get("HAMEXP_COMMIT", "unknown")
    return v


def run_job(job, datasets, out, device, torch):
    from . import data, ff, train

    stage = job["stage"]
    jid = job_id(job)
    dirs = {k: os.path.join(out, stage, k) for k in ("runs", "zaslepione", "ckpt")}
    for k in ("runs", "zaslepione"):
        os.makedirs(dirs[k], exist_ok=True)
    run_path = os.path.join(dirs["runs"], jid + ".json")
    if os.path.exists(run_path):
        return "skip"

    x, y = datasets[job["dataset"]]
    sp = data.make_split(job["seed"])
    tr, va = sp.train_idx.to(device), sp.val_idx.to(device)
    t0 = time.time()
    if job["kind"] == "ff":
        rec, net = train.train_ff(job, x[tr], y[tr], x[va], y[va], device)
        pred = train.nan_predictions(len(va), device) if rec["nan"] else ff.predict(net, x[va], data.encode, job["arm"])
        state = None if rec["nan"] else net.state_dict()
        extra = {"bcm_theta_final": (list(net.bcm) if job["arm"] == "BCM" and not rec["nan"] else None)}
    else:
        rec, net = train.train_bp(job, x[tr], y[tr], x[va], y[va], device)
        main = torch.zeros(len(va), dtype=torch.long, device=device) if rec["nan"] else train.predict_bp(net, x[va])
        pred = {"main": main}
        state = None if rec["nan"] else net.state_dict()
        extra = {}
    val_acc = (pred["main"] == y[va]).float().mean().item() * 100.0

    ckpt_sha = None
    if stage in CKPT_STAGES and state is not None:
        os.makedirs(dirs["ckpt"], exist_ok=True)
        ckpt_path = os.path.join(dirs["ckpt"], jid + ".pt")
        torch.save({"job": job, "state": state}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        ckpt_sha = sha256(ckpt_path)
    if stage in CKPT_STAGES:
        pp = os.path.join(dirs["zaslepione"], jid + ".pt")
        torch.save({"val_idx": sp.val_idx, **{k: v.cpu() for k, v in pred.items()}}, pp + ".tmp")
        os.replace(pp + ".tmp", pp)

    _atomic_write_json(os.path.join(dirs["zaslepione"], jid + ".json"),
                       {"id": jid, "val_acc": val_acc, "val_curve": rec.get("val_curve", [])})
    record = {
        "id": jid, "job": job, "nan": rec["nan"], "nan_epoch": rec["nan_epoch"],
        "epochs_run": rec["epochs_run"], "train_time_sec": rec["train_time_sec"],
        "wall_time_sec": time.time() - t0, "bcm_theta": rec.get("bcm_theta", []),
        "dw_norm": rec.get("dw_norm", []), "versions": _versions(torch), "ckpt_sha256": ckpt_sha, **extra,
    }
    _atomic_write_json(run_path, record)
    return "done"


def _lock_dir(out, stage):
    d = os.path.join(out, stage, "locks")
    os.makedirs(d, exist_ok=True)
    return d


def _claim(out, stage, jid) -> bool:
    if os.path.exists(os.path.join(out, stage, "runs", jid + ".json")):
        return False
    try:
        fd = os.open(os.path.join(_lock_dir(out, stage), jid), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def clear_locks(plan_path, out):
    with open(plan_path) as f:
        stages = {j["stage"] for j in json.load(f)}
    for stage in stages:
        d = _lock_dir(out, stage)
        for name in os.listdir(d):
            os.remove(os.path.join(d, name))


def setup_torch(device_name):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def load_train_sets(names, device, torch):
    from . import data

    out = {}
    for ds in sorted(names):
        x, y = data.load(ds, "train")
        out[ds] = (data.normalize_pixels(x).to(device), y.to(device))
    return out


def worker(rank, plan_path, out, device_name):
    torch = setup_torch(device_name)
    device = torch.device(device_name)
    with open(plan_path) as f:
        jobs = json.load(f)
    datasets = load_train_sets({j["dataset"] for j in jobs}, device, torch)
    for job in jobs:
        jid = job_id(job)
        if not _claim(out, job["stage"], jid):
            continue
        err_path = os.path.join(out, job["stage"], "runs", jid + ".error")
        try:
            status = run_job(job, datasets, out, device, torch)
            if os.path.exists(err_path):
                os.remove(err_path)
            print(f"[w{rank}] {status} {jid}", flush=True)
        except Exception:
            os.makedirs(os.path.dirname(err_path), exist_ok=True)
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
            print(f"[w{rank}] ERROR {jid}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    clear_locks(args.plan, args.out)
    if args.workers == 1:
        worker(0, args.plan, args.out, args.device)
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, args.plan, args.out, args.device)) for r in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    sys.exit(max(p.exitcode for p in procs))


if __name__ == "__main__":
    main()
