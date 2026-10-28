"""Vectorized job runner (ODSTEPSTWA: execution change). Same outputs as ``runner.run_job`` for every job.

Usage:  python -m hamexp.runner_batched PLAN.json --out WYNIKI --device cuda --batch 64 --workers 2

Jobs of the same (stage, dataset, arm) are trained together in chunks of up to ``--batch`` runs
(hamexp.batched). Each worker claims whole chunks via lock files; finished jobs are skipped on restart.
"""
import argparse
import json
import os
import sys
import time
import traceback

from .runner import CKPT_STAGES, _atomic_write_json, _versions, clear_locks, job_id, load_train_sets, setup_torch, sha256


def chunks(jobs, size):
    groups = {}
    for j in jobs:
        groups.setdefault((j["stage"], j["dataset"], j["arm"]), []).append(j)
    out = []
    for key in sorted(groups):
        g = groups[key]
        for i in range(0, len(g), size):
            out.append(g[i:i + size])
    return out


def _claim_chunk(out, chunk):
    stage = chunk[0]["stage"]
    lock_dir = os.path.join(out, stage, "locks")
    os.makedirs(lock_dir, exist_ok=True)
    name = "chunk__" + job_id(chunk[0]) + f"__n{len(chunk)}"
    try:
        fd = os.open(os.path.join(lock_dir, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def run_chunk(chunk, datasets, out, device, torch, batch_size):
    from . import batched, train

    todo = [j for j in chunk if not os.path.exists(os.path.join(out, j["stage"], "runs", job_id(j) + ".json"))]
    if not todo:
        return 0
    stage, ds = todo[0]["stage"], todo[0]["dataset"]
    x, y = datasets[ds]
    t0 = time.time()
    if todo[0]["kind"] == "ff":
        recs, net, va = batched.train_batch(todo, x, y, device)
        preds = net.predict(x[va])
    else:
        recs, net, va = batched.train_batch_bp(todo, x, y, device)
        preds = {"main": net.predict(x[va])}
    wall = time.time() - t0
    dirs = {k: os.path.join(out, stage, k) for k in ("runs", "zaslepione", "ckpt")}
    for k in ("runs", "zaslepione"):
        os.makedirs(dirs[k], exist_ok=True)
    vers = dict(_versions(torch), batched=len(todo))
    for r, j in enumerate(todo):
        jid = job_id(j)
        rec = recs[r]
        if rec["nan"]:
            pr = {k: torch.zeros_like(v[r]) for k, v in preds.items()}
        else:
            pr = {k: v[r] for k, v in preds.items()}
        yv = y[va[r]]
        val_acc = (pr["main"] == yv).float().mean().item() * 100.0
        ckpt_sha = None
        if stage in CKPT_STAGES and not rec["nan"]:
            os.makedirs(dirs["ckpt"], exist_ok=True)
            cp = os.path.join(dirs["ckpt"], jid + ".pt")
            torch.save({"job": j, "state": net.run_state(r)}, cp + ".tmp")
            os.replace(cp + ".tmp", cp)
            ckpt_sha = sha256(cp)
        if stage in CKPT_STAGES:
            pp = os.path.join(dirs["zaslepione"], jid + ".pt")
            torch.save({"val_idx": va[r].cpu(), **{k: v.cpu() for k, v in pr.items()}}, pp + ".tmp")
            os.replace(pp + ".tmp", pp)
        _atomic_write_json(os.path.join(dirs["zaslepione"], jid + ".json"),
                           {"id": jid, "val_acc": val_acc, "val_curve": rec.get("val_curve", [])})
        extra = {}
        if j["kind"] == "ff":
            extra["bcm_theta_final"] = (net.bcm[r].tolist() if j["arm"] == "BCM" and not rec["nan"] else None)
        record = {"id": jid, "job": j, "nan": rec["nan"], "nan_epoch": rec["nan_epoch"], "epochs_run": rec["epochs_run"],
                  "train_time_sec": wall / len(todo), "wall_time_sec": wall / len(todo),
                  "bcm_theta": rec.get("bcm_theta", []), "dw_norm": rec.get("dw_norm", []), "versions": vers,
                  "ckpt_sha256": ckpt_sha, **extra}
        _atomic_write_json(os.path.join(dirs["runs"], jid + ".json"), record)
    return len(todo)


def worker(rank, plan_path, out, device_name, batch_size):
    torch = setup_torch(device_name)
    device = torch.device(device_name)
    with open(plan_path) as f:
        jobs = json.load(f)
    datasets = load_train_sets({j["dataset"] for j in jobs}, device, torch)
    for ch in chunks(jobs, batch_size):
        if not _claim_chunk(out, ch):
            continue
        try:
            n = run_chunk(ch, datasets, out, device, torch, batch_size)
            print(f"[w{rank}] done {n} {job_id(ch[0])}..", flush=True)
        except Exception:
            err = os.path.join(out, ch[0]["stage"], "runs", "chunk__" + job_id(ch[0]) + ".error")
            os.makedirs(os.path.dirname(err), exist_ok=True)
            with open(err, "w") as f:
                f.write(traceback.format_exc())
            print(f"[w{rank}] ERROR {job_id(ch[0])}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    clear_locks(a.plan, a.out)
    if a.workers == 1:
        worker(0, a.plan, a.out, a.device, a.batch)
        return
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, a.plan, a.out, a.device, a.batch)) for r in range(a.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    sys.exit(max(p.exitcode for p in procs))


if __name__ == "__main__":
    main()
