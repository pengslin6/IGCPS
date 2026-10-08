"""Check saved fixed-policy predictions and comparison provenance."""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "unified_detection_20260929"))
import unified as u
sys.path.insert(0, str(ROOT / "experiments_20260928"))
import common as audit


def confusion(y, prediction, classes):
    cm = np.bincount(y * classes + prediction, minlength=classes * classes).reshape(classes, classes)
    tp, support, predicted = cm.diagonal(), cm.sum(1), cm.sum(0)
    per_f1 = 2 * tp / np.maximum(support + predicted, 1)
    return cm, float(100 * per_f1.mean())


def main():
    report = {"all_passed": True, "datasets": {}, "fixed_prediction_files": 0}
    results = u.read(HERE / "results.json")
    lock = u.read(HERE / "selection_lock.json")
    fitted = u.read(HERE / "fit_lock.json")
    assert lock["test_read_for_selection"] is False
    assert fitted["test_used_for_fitting"] is False
    for path, digest in fitted["fitted_model_hashes"].items():
        assert u.sha(path) == digest
    for dataset in u.DATASETS:
        test = u.graph(dataset, "test")
        development = u.graph(dataset, "development")
        labels = test["labels"].numpy()
        raw = test["raw_indices"].numpy()
        lookup = audit.c.class_root_table(development, len(u.meta(dataset)["label_names"]))
        focus = u.meta(dataset)["label_names"].index("NM" if dataset == "IGCPS" else "Attack6")
        passed = 0
        for model in ["DA-TGT", "DT-GNN"]:
            for seed in u.SEEDS:
                path = HERE / dataset / f"fixed_full__{model}__seed{seed}.npz"
                with np.load(path) as prediction:
                    np.testing.assert_array_equal(prediction["labels"], labels)
                    np.testing.assert_array_equal(prediction["raw_indices"], raw)
                    probs = prediction["probabilities"]
                    roots = prediction["root_scores"]
                np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)
                assert np.isfinite(probs).all() and (probs >= 0).all()
                cm, f1 = confusion(labels, probs.argmax(1), probs.shape[1])
                saved = u.read(HERE / dataset / f"fixed_full__{model}__seed{seed}.json")["metrics"]
                np.testing.assert_allclose(f1, saved["f1"], atol=1e-9)
                np.testing.assert_allclose(100 * (cm[0].sum() - cm[0, 0]) / cm[0].sum(),
                                           saved["normal_fpr"], atol=1e-9)
                np.testing.assert_allclose(100 * cm[focus, focus] / cm[focus].sum(),
                                           saved["focus_recall"], atol=1e-9)
                ranking = audit.c.ranking(probs @ lookup, test)
                np.testing.assert_allclose(ranking["top1"], saved["lookup_top1"], atol=1e-9)
                if model == "DA-TGT":
                    previous = np.load(ROOT / "unified_detection_20260929/pair_refinement" /
                                       dataset / f"final_seed{seed}.npz")
                    np.testing.assert_array_equal(probs, previous["probabilities"])
                passed += 1
        report["datasets"][dataset] = {"prediction_files_checked": passed,
                                       "row_count": len(labels),
                                       "same_selected_policy_across_architectures":
                                           lock["selection"]["DA-TGT"]["selected"]["policy"] ==
                                           lock["selection"]["DT-GNN"]["selected"]["policy"]}
        report["fixed_prediction_files"] += passed
    (HERE / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
