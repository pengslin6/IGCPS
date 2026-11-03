"""Evaluate detection and reference-node ranking over repeated graph fits."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits
import torch
from .infer import Pipeline
from .utils import load_cache, write_json


def metrics(probabilities, scores, cache):
    labels = cache["labels"].numpy()
    predicted = probabilities.argmax(1)
    targets = cache["reference_targets"].numpy()
    eligible = cache["reference_mask"].numpy() > 0
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranked = np.take_along_axis(targets, order, axis=1)
    ranks = np.argmax(ranked > 0, axis=1) + 1
    discount = 1 / np.log2(np.arange(2, min(5, targets.shape[1]) + 2))
    dcg = (ranked[:, :len(discount)] * discount).sum(1)
    ideal = np.asarray([discount[:min(int(row.sum()), 5)].sum() for row in targets])
    report = {"accuracy": accuracy_score(labels, predicted) * 100,
              "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0) * 100,
              "normal_fpr": float(np.mean(predicted[labels == 0] != 0) * 100) if (labels == 0).any() else None,
              "top1": float(np.mean(ranks[eligible] == 1) * 100) if eligible.any() else None,
              "mrr": float(np.mean(1 / ranks[eligible]) * 100) if eligible.any() else None,
              "ndcg5": float(np.mean(dcg[eligible] / ideal[eligible]) * 100) if eligible.any() else None}
    try:
        report["auc"] = float(roc_auc_score(np.eye(probabilities.shape[1])[labels], probabilities, average="macro") * 100)
    except ValueError:
        report["auc"] = None
    return report


def evaluate(data, model, output, seeds, device, batch_size, variants):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cache = load_cache(Path(data) / "test.pt")
    with np.load(Path(data) / "history_test.npz", allow_pickle=False) as source:
        arrays = dict(source)
    np.testing.assert_array_equal(cache["raw_indices"].numpy(), arrays["raw_indices"])
    np.testing.assert_array_equal(cache["labels"].numpy(), arrays["labels"])
    all_results = {}
    for variant in variants:
        rows, counts = [], None
        for seed in seeds:
            pipeline = Pipeline(model, seed, device)
            result = pipeline.predict(cache["features"], arrays, batch_size, variant)
            row = metrics(result["probabilities"], result["lookup_scores"], cache)
            row.update(seed=seed, variant=variant)
            rows.append(row)
            cm = confusion_matrix(arrays["labels"], result["predictions"], labels=range(len(pipeline.meta["label_names"])))
            counts = cm if counts is None else counts + cm
            np.savez_compressed(output / ("%s_seed%d.npz" % (variant, seed)), labels=arrays["labels"],
                                raw_indices=arrays["raw_indices"], **result)
            write_json(output / ("%s_seed%d_classwise.json" % (variant, seed)),
                classification_report(arrays["labels"], result["predictions"], output_dict=True, zero_division=0))
            print(variant, seed, row, flush=True)
        summary = {}
        for name in ("accuracy", "macro_f1", "auc", "normal_fpr", "top1", "mrr", "ndcg5"):
            values = [r[name] for r in rows if r[name] is not None]
            summary[name] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.} if values else None
        all_results[variant] = {"runs": rows, "summary": summary, "confusion_counts": counts.tolist(),
            "count_unit": "prediction instances summed across graph seeds", "query_windows_per_seed": len(cache["labels"])}
    write_json(output / "summary.json", all_results)
    if "full" in all_results:
        pd.DataFrame(all_results["full"]["runs"]).rename(columns={"macro_f1": "f1"}).to_csv(output / "runs.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared dataset directory")
    parser.add_argument("--model", type=Path, required=True, help="Fitted dataset directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--variants", nargs="+", choices=["full", "without_pair", "pair_only", "graph_only"], default=["full"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        evaluate(args.data, args.model, args.output, args.seeds, args.device, args.batch_size, args.variants)
