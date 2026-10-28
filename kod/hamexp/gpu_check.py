"""On-instance checks (PROTOKOL_v6 §8.2 T6, §8.3 benchmark).

Usage:  python -m hamexp.gpu_check --out LOG.json --k-list 1,4,8,12,16

1. T6: the same run twice gives identical weights and validation predictions; and a run alone vs. one of
   k parallel processes gives identical predictions (independence of k);
2. throughput: full-size runs (20 000 examples, 25 epochs, validation every epoch) on smoke-test seeds
   (69001+, outside every pool) for k parallel processes; runs/hour per k.
"""
import argparse
import json
import multiprocessing as mp
import time

SMOKE_SEED = 69001


def _run(args):
    device, seed, epochs = args
    from .runner import setup_torch
    torch = setup_torch(device)
    from . import data, ff, train
    x, y = data.load("mnist", "train")
    xn = data.normalize_pixels(x).to(device)
    y = y.to(device)
    sp = data.make_split(seed)
    tr, va = sp.train_idx.to(device), sp.val_idx.to(device)
    job = {"seed": seed, "arm": "BCM", "cfg": {"lr": 1e-3, "theta": 2.0}}
    t = time.time()
    rec, net = train.train_ff(job, xn[tr], y[tr], xn[va], y[va], device, epochs=epochs, log_val=(epochs == 25))
    pred = ff.predict(net, xn[va], data.encode, "BCM")["main"].cpu()
    return {"sec": time.time() - t, "pred": pred.tolist(), "w0": net.layers[0].weight.detach().cpu().sum().item()}


def determinism(device):
    a = _run((device, SMOKE_SEED, 2))
    b = _run((device, SMOKE_SEED, 2))
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        par = pool.map(_run, [(device, SMOKE_SEED, 2)] + [(device, SMOKE_SEED + i, 2) for i in range(1, 4)])
    return {"T6_powtorka": a["pred"] == b["pred"] and a["w0"] == b["w0"],
            "T6_niezaleznosc_od_k": a["pred"] == par[0]["pred"] and a["w0"] == par[0]["w0"]}


def determinism_ok(device="cuda"):
    r = determinism(device)
    return all(r.values())


def throughput(device, k_list):
    out = {}
    ctx = mp.get_context("spawn")
    for k in k_list:
        t = time.time()
        try:
            with ctx.Pool(k) as pool:
                per = pool.map(_run, [(device, SMOKE_SEED + 100 + i, 25) for i in range(k)])
        except Exception as e:
            out[k] = {"error": type(e).__name__}
            break
        wall = time.time() - t
        out[k] = {"wall_sec": wall, "runs_per_hour": 3600.0 * k / wall, "mean_run_sec": sum(p["sec"] for p in per) / k}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k-list", default="1,4,8,12,16")
    a = ap.parse_args(argv)
    res = {"determinism": determinism(a.device)}
    res["throughput"] = throughput(a.device, [int(k) for k in a.k_list.split(",")])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
