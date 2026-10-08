"""Independent post-run consistency checks for the fair-control artifacts."""
import argparse
import re
import platform
from importlib.metadata import version
from common import *


def verify(dataset):
    folder = HERE / "fair_control" / dataset
    assert (folder / "summary.json").exists(), f"Incomplete: {dataset}"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert sha(SOURCE / DATA[dataset]["file"]) == DATA[dataset]["sha256"]
    for filename, expected_hash in manifest["source_sha256"].items():
        assert sha(HERE / "code_snapshot" / filename) == expected_hash
    for key, file in (("runner_sha256", "fair_control.py"), ("common_sha256", "common.py"), ("protocol_sha256", "PROTOCOL.md")):
        assert manifest[key] == sha(HERE / file), f"Modified locked {key}"
    lock = json.loads((folder / "selection_lock.json").read_text())
    checks = json.loads((folder / "implementation_checks.json").read_text())
    assert checks["unknown_attack_mask_verified"]
    assert checks["DT-GNN"]["batched_single_graph_equivalence"]
    assert all(checks[name]["node_loss_reaches_encoder"] for name in manifest["models"])
    coordinator = folder / "execution_coordinator.json"
    if coordinator.exists():
        assert json.loads(coordinator.read_text())["coordinator_sha256"] == sha(HERE / "parallel_fair_control.py")
    development, test = load_cache(dataset, "development"), load_cache(dataset, "test")
    meta = load_meta(dataset)
    labels = test["labels"].numpy()
    lookup = c.class_root_table(development, len(meta["label_names"]))
    records = []
    for model in manifest["models"]:
        choices = []
        for lr in manifest["learning_rates"]:
            prefix = folder / f"validation__{model}__lr{int(lr * 1000000)}"
            history = json.loads(prefix.with_suffix(".history.json").read_text())
            assert [r["epoch"] for r in history] == list(range(1, 19))
            assert all(np.isfinite(r["loss"]) for r in history)
            for row in history:
                np.testing.assert_allclose(c.balanced_key(row), row["selection_score"])
            best = max(history, key=lambda r: (r["selection_score"], r["auc"], -r["epoch"]))
            result = json.loads(prefix.with_suffix(".result.json").read_text())
            assert result["epoch"] == best["epoch"] and result["learning_rate"] == lr
            choices.append(result)
        selected = max(choices, key=lambda r: (r["selection_score"], r["auc"], -r["epoch"], -r["learning_rate"]))
        assert lock["selected"][model] == selected
        for seed in SEEDS:
            prefix = folder / f"final__{model}__seed{seed}"
            result = json.loads(prefix.with_suffix(".result.json").read_text())
            history = json.loads(prefix.with_suffix(".history.json").read_text())
            assert result["epochs"] == selected["epoch"] == len(history)
            assert result["learning_rate"] == selected["learning_rate"]
            assert result["selection_lock_sha256"] == sha(folder / "selection_lock.json")
            assert result["checkpoint_sha256"] == sha(prefix.with_suffix(".pt"))
            with np.load(prefix.with_suffix(".predictions.npz")) as pred:
                np.testing.assert_array_equal(labels, pred["labels"])
                np.testing.assert_array_equal(test["raw_indices"].numpy(), pred["raw_indices"])
                assert np.isfinite(pred["probabilities"]).all() and np.isfinite(pred["root_scores"]).all()
                np.testing.assert_allclose(pred["probabilities"].sum(1), 1, atol=1e-6)
                assert pred["root_scores"].min() >= 0 and pred["root_scores"].max() <= 1
                actual = c.evaluate(pred["probabilities"], pred["root_scores"], test, lookup)
                predicted = pred["probabilities"].argmax(1)
                for class_id, name in enumerate(meta["label_names"]):
                    tp = int(((labels == class_id) & (predicted == class_id)).sum())
                    n_true = int((labels == class_id).sum())
                    n_pred = int((predicted == class_id).sum())
                    np.testing.assert_allclose(tp / max(n_true, 1), result["per_class"][name]["recall"])
                    np.testing.assert_allclose(2 * tp / max(n_true + n_pred, 1), result["per_class"][name]["f1-score"])
            for metric in METRICS:
                np.testing.assert_allclose(actual[metric], result["metrics"][metric], atol=1e-9)
            np.testing.assert_allclose(actual["class_lookup_soft"]["top1"], result["metrics"]["class_lookup_soft"]["top1"], atol=1e-9)
            records.append({"model": model, "seed": seed, "metrics_recomputed": True,
                            "checkpoint_hash_verified": True, "selection_lock_verified": True})
    indices = np.load(HERE / "data_cache" / dataset / "split_indices.npz")
    for train_name in ("train", "development"):
        assert not np.intersect1d(indices[train_name], indices["test"]).size
    scheduling_check = None
    if dataset == "IGCPS":
        serial_log = (HERE / "fair_progress.log").read_text(encoding="utf-8", errors="replace")
        serial_rows = re.findall(r"final__DT-GNN__seed22 epoch=(\d+)/15 loss=([0-9.]+)", serial_log)
        parallel_history = json.loads((folder / "final__DT-GNN__seed22.history.json").read_text())
        for epoch, loss in serial_rows:
            np.testing.assert_allclose(float(loss), parallel_history[int(epoch)-1]["loss"], atol=5.1e-6, rtol=0)
        scheduling_check = {"restarted_fit": "DT-GNN seed22", "serial_epochs_compared": len(serial_rows),
                            "all_restarted_losses_match_serial_log": True}
    audit = {"dataset": dataset, "all_passed": True, "final_fits": len(records),
             "equal_validation_budget_verified": True, "unchanged_locked_code_verified": True,
             "identical_test_rows_verified": True, "no_train_test_overlap": True,
             "per_class_detection_recomputed": True, "scheduling_equivalence": scheduling_check, "records": records}
    dump(folder / "postrun_verification.json", audit)
    dump(HERE / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("torch", "torch-geometric", "numpy", "pandas", "scipy", "scikit-learn")},
        "threads_per_fit": 2, "parallel_workers_per_dataset": 2,
        "hardware_note": "CPU-only execution; elapsed fit times are not a controlled speed benchmark."})
    print(f"POST-RUN VERIFIED {dataset}: {len(records)} fits; all metrics, hashes, masks, selection locks, and test rows match", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=list(DATA), choices=list(DATA))
    args = parser.parse_args()
    torch.set_num_threads(2)
    for dataset in args.dataset:
        verify(dataset)
