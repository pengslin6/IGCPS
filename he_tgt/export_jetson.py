"""Export fitted HE-TGT weights and numeric history trees for Jetson inference."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .infer import Pipeline
from .utils import write_json


def export_trees(bank, prefix, model, scaler, schema, setting):
    w, n = schema["total_width"], schema["network_width"]
    if setting["view"] == "network_history":
        columns = np.r_[np.concatenate([np.arange(n) + w * k for k in range(4)]), np.arange(w * 4, w * 4 + 3)]
    elif setting["view"] == "network_with_physical_changes":
        columns = np.r_[np.arange(n), np.arange(w, w * 4 + 3)]
    else:
        columns = np.arange(w * 4 + 3)
    trees, roots, classes, offset = [], [], [], 0
    for iteration in model._predictors:
        for cls, predictor in enumerate(iteration):
            nodes = predictor.nodes.copy()
            if nodes["is_categorical"].any():
                raise ValueError("Exporter expects numeric split trees")
            roots.append(offset)
            classes.append(cls)
            nodes["left"] += offset
            nodes["right"] += offset
            offset += len(nodes)
            trees.append(nodes)
    nodes = np.concatenate(trees)
    values = {"columns": columns, "mean": scaler.mean_, "scale": scaler.scale_,
              "baseline": model._baseline_prediction, "roots": np.asarray(roots, np.int64),
              "tree_class": np.asarray(classes), "max_depth": np.asarray(nodes["depth"].max())}
    for source, target in [("feature_idx", "feature"), ("num_threshold", "threshold"),
                           ("missing_go_to_left", "missing_left"), ("left", "left"),
                           ("right", "right"), ("is_leaf", "leaf"), ("value", "value")]:
        values[target] = nodes[source].copy()
    for name, array in values.items():
        bank[prefix + name] = array


def export(model_dir, seed, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pipeline = Pipeline(model_dir, seed, "cpu")
    weights = {"weight/" + key: value.detach().numpy() for key, value in pipeline.model.state_dict().items()}
    weights.update(adjacency=pipeline.adjacency.numpy(), reference_table=pipeline.table)
    np.savez_compressed(output / "graph.npz", **weights)
    bank = {}
    prior = pipeline.prior
    export_trees(bank, "prior/", prior["model"], prior["scaler"], prior["schema"], prior["setting"])
    for (a, b), fitted in pipeline.pairs["pairs"].items():
        # The fixed policy beta=1 uses the tree's conditional probability.
        export_trees(bank, "%d_%d/" % (a, b), fitted["tree"], fitted["scaler"],
                     pipeline.pairs["schema"], pipeline.pairs["setting"])
    np.savez_compressed(output / "trees.npz", **bank)
    write_json(output / "model.json", {"format_version": 1, "dataset": pipeline.meta["dataset"], "seed": seed,
        "architecture": {"input_dim": pipeline.width, "n_nodes": pipeline.meta["n_nodes"],
                         "n_net": pipeline.meta["n_net"], "n_classes": len(pipeline.meta["label_names"])},
        "label_names": pipeline.meta["label_names"], "node_names": pipeline.meta["node_names"],
        "history_width": prior["schema"]["total_width"] * 4 + 3,
        "pairs": [list(pair) for pair in pipeline.pairs["pairs"]],
        "policy": {"gamma": .25, "beta": 1., "rho": .5}})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", type=Path, help="Optional prepared graph cache for an inference input file")
    parser.add_argument("--history", type=Path, help="Aligned history NPZ; required with --graph")
    parser.add_argument("--queries", type=int, default=128)
    args = parser.parse_args()
    if bool(args.graph) != bool(args.history) or args.queries < 1:
        parser.error("Provide both --graph and --history and a positive --queries value")
    output = export(args.model, args.seed, args.output)
    if args.graph:
        graph = torch.load(args.graph, map_location="cpu", weights_only=True)
        features = graph["features"] if isinstance(graph, dict) else graph
        with np.load(args.history, allow_pickle=False) as arrays:
            if len(features) != len(arrays["history"]):
                raise ValueError("Graph/history row counts differ")
            if isinstance(graph, dict) and "raw_indices" in graph:
                np.testing.assert_array_equal(graph["raw_indices"].numpy(), arrays["raw_indices"])
            np.savez_compressed(output / "input.npz", graph=features[:args.queries].numpy(), history=arrays["history"][:args.queries])
    print("Exported", output)
