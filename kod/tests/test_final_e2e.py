"""End-to-end test of final.py on a synthetic results directory (§7.9): every report field is produced."""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hamexp import final  # noqa: E402
from hamexp.ff import FF_ARMS  # noqa: E402

ACC = {"STD": 0.90, "CUR": 0.92, "ANN": 0.92, "REV": 0.88, "BCM": 0.90, "PLW": 0.90, "BP": 0.95}


def _make(tmp, n_seeds=10):
    rng = np.random.default_rng(0)
    for ds in ("mnist", "fmnist"):
        for arm in FF_ARMS + ("BP",):
            for s in range(n_seeds):
                rid = f"C__{ds}__{arm}__s{63001 + s}"
                nan = arm == "REV" and s == 0
                for sub in ("runs", "zaslepione"):
                    os.makedirs(os.path.join(tmp, "C", sub), exist_ok=True)
                rec = {"id": rid, "job": {"stage": "C", "kind": "bp" if arm == "BP" else "ff", "dataset": ds, "arm": arm,
                                          "seed": 63001 + s, "cfg": {}},
                       "nan": nan, "bcm_theta": [[2.0 + l for l in range(10)]] * 25 if arm == "BCM" else [],
                       "dw_norm": [[0.1] * 10] * 25}
                json.dump(rec, open(os.path.join(tmp, "C", "runs", rid + ".json"), "w"))
                json.dump({"id": rid, "val_acc": 9.9 if nan else 100 * ACC[arm], "val_curve": [80.0] * 25},
                          open(os.path.join(tmp, "C", "zaslepione", rid + ".json"), "w"))
                os.makedirs(os.path.join(tmp, "test", "C"), exist_ok=True)
                sets = {ds: 10000, **({"qmnist": 50000} if ds == "mnist" else {})}
                res = {}
                for nm, n in sets.items():
                    p = 0.098 if nan else ACC[arm] + rng.normal(0, 0.003)
                    c = torch.from_numpy(rng.random(n) < p)
                    res[nm] = {"main": c} if arm == "BP" else {"main": c, "no_l0": c, "cross": c}
                torch.save(res, os.path.join(tmp, "test", "C", rid + ".pt"))
    os.makedirs(os.path.join(tmp, "A", "runs"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "A", "zaslepione"), exist_ok=True)
    for p in range(3):
        rid = f"A__mnist__STD__s61001__p{p}"
        json.dump({"id": rid, "job": {"dataset": "mnist", "arm": "STD", "point": str(p), "cfg": {"lr": 1e-3 * (p + 1), "theta": 1.0}}},
                  open(os.path.join(tmp, "A", "runs", rid + ".json"), "w"))
        json.dump({"id": rid, "val_acc": 90.0 + p}, open(os.path.join(tmp, "A", "zaslepione", rid + ".json"), "w"))


def _files(tmp, drop_arm=None):
    sel = {f"{ds}|{a}": {"point": 0, "cfg": {"lr": 1e-3, "theta": 1.0}} for ds in ("mnist", "fmnist") for a in FF_ARMS + ("BP",)}
    if drop_arm:
        sel[drop_arm]["cfg"] = None
    json.dump(sel, open(os.path.join(tmp, "SEL.json"), "w"))
    json.dump({"N": 10, "N_power_table": {"table": {"H2": {"0": {"joint": {"R": 0.2}}}}}},
              open(os.path.join(tmp, "P.json"), "w"))


def _args(tmp, out):
    return ["--results", tmp, "--params", os.path.join(tmp, "P.json"), "--sel", os.path.join(tmp, "SEL.json"),
            "--out", out, "--boot", "200"]


def test_refuses_incomplete_stage_c(tmp_path):
    import pytest
    tmp = str(tmp_path)
    _make(tmp)
    _files(tmp)
    os.remove(os.path.join(tmp, "test", "C", "C__mnist__CUR__s63003.pt"))
    with pytest.raises(SystemExit):
        final.main(_args(tmp, os.path.join(tmp, "W.json")))


def test_untrainable_arm(tmp_path):
    import shutil
    tmp = str(tmp_path)
    _make(tmp)
    _files(tmp, drop_arm="fmnist|PLW")
    for f in os.listdir(os.path.join(tmp, "C", "runs")):
        if "__fmnist__PLW__" in f:
            rid = f[:-5]
            for sub, ext in (("runs", ".json"), ("zaslepione", ".json")):
                os.remove(os.path.join(tmp, "C", sub, rid + ext))
            os.remove(os.path.join(tmp, "test", "C", rid + ".pt"))
    out = os.path.join(tmp, "W.json")
    final.main(_args(tmp, out))
    rep = json.load(open(out))
    assert rep["confirmatory"]["H3"]["joint"]["category"] == "NIEROZSTRZYGNIETY"
    assert "nietrenowalne" in rep["confirmatory"]["H3"]["note"]


def test_final_end_to_end(tmp_path):
    tmp = str(tmp_path)
    _make(tmp)
    _files(tmp)
    out = os.path.join(tmp, "W.json")
    final.main(_args(tmp, out))
    rep = json.load(open(out))
    assert set(rep["confirmatory"]) == {"H1", "H2", "H3", "H4"}
    for h in rep["confirmatory"]:
        c = rep["confirmatory"][h]
        assert c["joint"]["category"]
        assert c["stability_joint"]
        for ds in ("mnist", "fmnist"):
            assert "hl_pp" in c["per_dataset"][ds] and "stability" in c["per_dataset"][ds]
        s = rep["sensitivity"][h]
        for k in ("a_bootstrap_hl", "b_ttest", "c_qmnist", "d_no_failed_pairs", "e_S_0.25", "e_S_1.0"):
            assert k in s
    assert rep["confirmatory"]["H1"]["joint"]["class"] == "P"  # CUR 0.92 vs STD 0.90 → +2 p.p.
    assert rep["confirmatory"]["H4"]["joint"]["class"] == "G"
    assert rep["accuracy"]["mnist|REV"]["n_nan"] == 1
    for k in ("E1_bcm_theta", "E2_bp_gap", "E3_readouts", "E4_dw_norm", "E5_landscape", "E6_val_curves", "E7_ann_minus_std"):
        assert rep["exploratory"][k]
    assert os.path.exists(os.path.join(tmp, "rysunki", "E6_krzywe_mnist.png"))
    for k in ("notes_7_4", "H1_x_H2", "boundary_5_5", "versions", "power_final_N"):
        assert k in rep
    v = rep["confirmatory"]["H1"]["per_dataset"]["mnist"]
    for k in ("hl_pp", "hl_ci_pp", "mean_pp", "t_ci_pp", "failures", "stability", "holm_better_comparators_inf_S"):
        assert k in v
    b = rep["sensitivity"]["H1"]["a_bootstrap_hl"]["per_dataset"]["mnist"]
    assert "ci_pp_1-2a" in b and "p" in b
    assert "per_dataset" in rep["sensitivity"]["H1"]["e_S_0.25"]
