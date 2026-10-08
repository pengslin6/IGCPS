"""Prepare data, audit reference labels, and replay the retained five-seed models."""
import argparse
import gc
import json
from pathlib import Path
from common import *
from sklearn.metrics import classification_report, confusion_matrix


def analyze(dataset):
    out = HERE / "archived_analysis" / dataset
    out.mkdir(parents=True, exist_ok=True)
    meta = load_meta(dataset)
    development, test = load_cache(dataset, "development"), load_cache(dataset, "test")
    adjacency = torch.load(HERE / "data_cache" / dataset / "adjacency.pt", weights_only=True)
    lookup = c.class_root_table(development, len(meta["label_names"]))
    labels, targets = test["labels"].numpy(), test["root_targets"].numpy()
    eligible = test["root_mask"].numpy() > 0
    rows, class_rows, verified = [], [], []
    archive = SOURCE / DATA[dataset]["archive"]
    for seed in SEEDS:
        checkpoint = archive / f"HGAN-Trace__seed{seed}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        candidate = c.joint.Candidate(**payload["candidate"])
        model = c.joint.DATGTJoint(test["features"].shape[-1], meta["n_nodes"], meta["n_net"], len(meta["label_names"]), candidate)
        missing = model.load_state_dict(payload["state_dict"], strict=False)
        assert not set(missing.missing_keys) - {"detection_pool.weight", "detection_pool.bias"}
        assert not missing.unexpected_keys
        raw_probs, roots = c.collect(model, test, adjacency)
        probs = raw_probs.copy()
        if dataset == "IGCPS":
            lock = json.loads((archive / f"calibration_lock__seed{seed}.json").read_text())
            probs = conditional.locked_probabilities(raw_probs, lock)
            with np.load(archive / f"predictions__seed{seed}.npz") as saved:
                np.testing.assert_allclose(probs, saved["probabilities"], atol=2e-5)
                np.testing.assert_allclose(roots, saved["root_scores"], atol=2e-5)
                np.testing.assert_array_equal(labels, saved["labels"])
        metrics = c.evaluate(probs, roots, test, lookup)
        expected = json.loads((archive / f"result__seed{seed}.json").read_text())
        for metric in METRICS:
            np.testing.assert_allclose(metrics[metric], expected[metric], atol=1e-4, err_msg=f"{dataset} seed{seed}: {metric}")
        verified.append({"seed": seed, "checkpoint_sha256": sha(checkpoint), "all_metrics_match": True, "metrics": metrics})
        np.savez_compressed(out / f"replayed__seed{seed}.npz", labels=labels, raw_probabilities=raw_probs,
                            probabilities=probs, root_scores=roots, lookup_scores=probs @ lookup,
                            raw_indices=test["raw_indices"].numpy())
        head_metrics = ranking_per_row(roots, targets)
        lookup_metrics = ranking_per_row(probs @ lookup, targets)
        for metric in ("top1", "mrr", "ndcg5"):
            np.testing.assert_allclose(head_metrics[metric][eligible].mean(), metrics[metric], atol=1e-6)
            np.testing.assert_allclose(lookup_metrics[metric][eligible].mean(), metrics["class_lookup_soft"][metric], atol=1e-6)
        for group, mask in groups(labels, probs, meta["label_names"]).items():
            mask = mask & eligible
            row = {"dataset": dataset, "seed": seed, "group": group, "eligible_windows": int(mask.sum())}
            for metric in head_metrics:
                row[f"head_{metric}"] = float(head_metrics[metric][mask].mean()) if mask.any() else None
                row[f"lookup_{metric}"] = float(lookup_metrics[metric][mask].mean()) if mask.any() else None
                row[f"head_minus_lookup_{metric}"] = float((head_metrics[metric][mask]-lookup_metrics[metric][mask]).mean()) if mask.any() else None
            rows.append(row)
        for stage, probability in [("raw", raw_probs), ("adjusted", probs)]:
            report = classification_report(labels, probability.argmax(1), labels=np.arange(len(meta["label_names"])), target_names=meta["label_names"], output_dict=True, zero_division=0)
            for name in meta["label_names"]:
                class_rows.append({"dataset":dataset,"seed":seed,"stage":stage,"class":name,**report[name]})
        dump(out / f"confusion__seed{seed}.json", {"raw":confusion_matrix(labels,raw_probs.argmax(1)).tolist(),"adjusted":confusion_matrix(labels,probs.argmax(1)).tolist()})
        print(f"REPLAY VERIFIED {dataset} seed={seed} Acc={metrics['accuracy']:.4f} Head/Lookup Top1={metrics['top1']:.4f}/{metrics['class_lookup_soft']['top1']:.4f}", flush=True)
        del model, payload
        gc.collect()
    pd.DataFrame(rows).to_csv(out / "subgroups_per_seed.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out / "class_detection_per_seed.csv", index=False)
    frame = pd.DataFrame(rows)
    numeric = [col for col in frame if col.startswith(("head_", "lookup_"))]
    summary = frame.groupby("group", sort=False)[numeric].agg(["mean","std","count"])
    summary.columns = ["__".join(v) for v in summary.columns]
    summary = summary.join(frame.groupby("group")["eligible_windows"].agg(["min","max","mean"]).add_prefix("support_"))
    summary.to_csv(out / "subgroups_summary.csv")
    dump(out / "verification.json", verified)
    print(f"SUBGROUP ANALYSIS COMPLETE {dataset}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", choices=list(DATA), default=list(DATA))
    parser.add_argument("--stage", choices=["prepare", "analyze", "all"], default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    for dataset in args.dataset:
        if args.stage in ("prepare", "all"):
            prepare(dataset)
        if args.stage in ("analyze", "all"):
            analyze(dataset)
