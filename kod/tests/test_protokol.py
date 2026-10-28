"""Code tests T1–T8 (PROTOKOL_v6 §8.2). Run: python -m pytest -q tests/ (from kod/)."""
import hashlib
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KOD = os.path.dirname(HERE)
sys.path.insert(0, KOD)

from hamexp import data, ff, streams, train  # noqa: E402

SRC = os.path.join(KOD, "zrodlo")


def _src_module():
    """Frozen copy of the source campaign; hashes must match SHA256SUMS (§8.2 T1)."""
    with open(os.path.join(SRC, "SHA256SUMS")) as f:
        for line in f:
            digest, name = line.split()
            with open(os.path.join(SRC, name), "rb") as g:
                assert hashlib.sha256(g.read()).hexdigest() == digest, name
    sys.path.insert(0, SRC)
    import siec_ff  # noqa: E402
    return siec_ff


def _source_encode(x, y):
    """Source input: x/255 with the label overwriting pixels 0–9 (value 1.0) – test T1 only."""
    out = x.clone()
    out[:, :10] = 0.0
    if isinstance(y, int):
        out[:, y] = 1.0
    else:
        out[torch.arange(len(x)), y] = 1.0
    return out


def _small_data(n=256):
    x, y = data.load("mnist", "train")
    return x[:n], y[:n]


def _run_source(siec_ff, arm, cfg, xtr, ytr, orders, wrongs, seed, epochs=4, switch=2):
    torch.manual_seed(streams.init_seed(seed))
    start = {"STD": "standard", "CUR": "standard", "ANN": "standard", "REV": "hamowanie",
             "BCM": "standard_bcm", "PLW": "standard"}[arm]
    if arm == "PLW":
        teta = list(cfg["theta_layers"])
    elif start == "hamowanie":
        teta = None
    else:
        teta = cfg["theta"]
    net = siec_ff.SiecFF(784, ff.N_LAYERS, ff.WIDTH, start, lr=cfg["lr"], teta_override=teta)
    for ep in range(epochs):
        if ep == switch:
            if arm == "CUR":
                net.przelacz_formule("hamowanie")
            elif arm == "REV":
                net.przelacz_formule("standard", teta_override=cfg["theta"])
            elif arm == "ANN":
                for o in net.optymalizatory:
                    for g in o.param_groups:
                        g["lr"] = cfg["lr"] * 0.5
        for bi, s in enumerate(range(0, len(xtr), ff.BATCH)):
            idx = orders[ep][s:s + ff.BATCH]
            xb, yb = xtr[idx], ytr[idx]
            net.krok_treningowy(_source_encode(xb, yb), _source_encode(xb, wrongs[ep][bi]))
    return [w.liniowa.weight.detach() for w in net.warstwy], net


@pytest.mark.parametrize("arm", ff.FF_ARMS)
def test_T1_matches_source(arm):
    siec_ff = _src_module()
    xtr, ytr = _small_data()
    g = torch.Generator().manual_seed(7)
    epochs, switch = 4, 2
    orders = [torch.randperm(len(xtr), generator=g) for _ in range(epochs)]
    nb = (len(xtr) + ff.BATCH - 1) // ff.BATCH
    wrongs = [[data.wrong_labels(ytr[orders[e][b * ff.BATCH:(b + 1) * ff.BATCH]], g) for b in range(nb)]
              for e in range(epochs)]
    cfg = {"lr": 1e-3, "theta": 8.0, "theta_layers": [8.0 + i for i in range(ff.N_LAYERS)]}
    job = {"seed": 60001, "arm": arm, "cfg": cfg}
    _, net = train.train_ff(job, xtr, ytr, xtr[:10], ytr[:10], "cpu", encode=_source_encode,
                            orders=orders, wrongs=wrongs, epochs=epochs, switch=switch, log_val=False)
    w_src, src = _run_source(siec_ff, arm, cfg, xtr, ytr, orders, wrongs, 60001, epochs, switch)
    for a, b in zip(net.weights_snapshot(), w_src):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
    if arm == "BCM":
        assert all(abs(a - b) < 1e-4 for a, b in zip(net.bcm, src.tety))


def test_T2_schedules():
    assert ff.phase("CUR", 11) == ("standard", "theta", 1.0)
    assert ff.phase("CUR", 12) == ("hamowanie", "zero", 1.0)
    assert ff.phase("ANN", 12) == ("standard", "theta", 0.5)
    assert ff.phase("REV", 11) == ("hamowanie", "zero", 1.0)
    assert ff.phase("REV", 12) == ("standard", "theta", 1.0)
    net = ff.FFNet(794, 1e-3, 1.0, 1, "cpu")
    net.set_lr_factor(0.5)
    assert all(g["lr"] == 5e-4 for o in net.opts for g in o.param_groups)


def test_T3_bcm_update():
    torch.manual_seed(0)
    net = ff.FFNet(794, 1e-3, 2.0, 1, "cpu")
    x = torch.rand(8, 794)
    terms = net.layer_goodness_terms(x)
    expected0 = 0.99 * 2.0 + 0.01 * 0.5 * (terms[0][0].mean() + terms[0][0].mean()).item()
    net.step(x, x, "standard", "bcm")
    assert abs(net.bcm[0] - expected0) < 1e-6


def test_T4_split_and_test_lock():
    s = data.make_split(60001)
    assert len(s.train_idx) == data.N_TRAIN and len(s.val_idx) == data.N_VAL
    assert not set(s.train_idx.tolist()) & set(s.val_idx.tolist())
    with pytest.raises(data.TestAccessError):
        data.load("mnist", "test")


def test_T5_streams_identical_across_arms():
    a = torch.randperm(100, generator=streams.order_gen(63001, 3))
    b = torch.randperm(100, generator=streams.order_gen(63001, 3))
    assert torch.equal(a, b)
    assert streams.seed_from(1, 2, 3) != streams.seed_from(1, 2, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="T6 runs on the GPU instance (gpu_check)")
def test_T6_gpu_determinism():
    from hamexp import gpu_check
    assert gpu_check.determinism_ok()


def test_T7_nan_handling():
    xtr, ytr = _small_data(128)
    xn = data.normalize_pixels(xtr)
    job = {"seed": 60001, "arm": "STD", "cfg": {"lr": 1e-3, "theta": float("inf")}}
    rec, _ = train.train_ff(job, xn, ytr, xn[:10], ytr[:10], "cpu", epochs=3, log_val=False)
    assert rec["nan"] is True
    p = train.nan_predictions(5, "cpu")
    assert all(torch.equal(v, torch.zeros(5, dtype=torch.long)) for v in p.values())


def test_T8_input_encoding():
    x, y = _small_data(32)
    xn = data.normalize_pixels(x)
    e = data.encode(xn, y)
    assert e.shape[1] == data.IN_DIM
    assert torch.allclose(e[:, :784].norm(dim=1), torch.ones(32), atol=1e-6)
    assert torch.allclose(e[torch.arange(32), 784 + y], torch.full((32,), data.LABEL_VALUE))
    assert torch.equal(e[:, :784], xn)


def test_T7_pipeline_with_nan(tmp_path, monkeypatch):
    """§8.2 T7 (review K1-09): runner → write_hashes (NAN entry) → verify → evaluate for a NaN run and a
    normal run, on smoke-test seeds (69001+), 1 epoch, with mocked test sets (the real test is not opened)."""
    from hamexp import evaluate_test, runner
    out = str(tmp_path)
    jobs = [{"stage": "C", "kind": "ff", "dataset": "mnist", "arm": "STD", "seed": 69001, "cfg": {"lr": 1e-3, "theta": 1.0}},
            {"stage": "C", "kind": "ff", "dataset": "mnist", "arm": "CUR", "seed": 69002,
             "cfg": {"lr": 1e-3, "theta": float("inf")}}]
    plan = os.path.join(out, "C.json")
    json.dump(jobs, open(plan, "w"))
    orig = train.train_ff
    monkeypatch.setattr(train, "train_ff", lambda *a, **k: orig(*a, **{**k, "epochs": 1}))
    torch_mod = runner.setup_torch("cpu")
    ds = runner.load_train_sets({"mnist"}, torch_mod.device("cpu"), torch_mod)
    for j in jobs:
        assert runner.run_job(j, ds, out, torch_mod.device("cpu"), torch_mod) == "done"
    assert json.load(open(os.path.join(out, "C", "runs", runner.job_id(jobs[1]) + ".json")))["nan"] is True
    h = os.path.join(out, "H.txt")
    evaluate_test.write_hashes(out, h, plan)
    txt = open(h).read()
    assert "NAN " in txt and "C/ckpt/" in txt and "FILE " in txt
    real_load = data.load

    def fake_load(dataset, part):
        if (dataset, part) in data.TEST_PARTS:
            n = 200
            return torch.rand(n, 784), torch.randint(0, 10, (n,))
        return real_load(dataset, part)
    monkeypatch.setattr(data, "load", fake_load)
    assert evaluate_test.evaluate(out, h, "cpu", plan) == 2
    res_nan = torch.load(os.path.join(out, "test", "C", runner.job_id(jobs[1]) + ".pt"), weights_only=False)
    assert set(res_nan) == {"mnist", "qmnist"} and set(res_nan["mnist"]) == {"main", "no_l0", "cross"}
    assert evaluate_test.verify_predictions(out)
    os.remove(os.path.join(out, "C", "runs", runner.job_id(jobs[0]) + ".json"))
    with pytest.raises(SystemExit):
        evaluate_test.verify(out, h, plan)
