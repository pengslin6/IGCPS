"""Jetson inference with PyTorch graph weights and numeric tree arrays.

This module needs only NumPy and PyTorch; scikit-learn is used at export time.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .model import HETGT


def tree_probabilities(bank, prefix, history):
    def arr(name):
        return bank[prefix + name]
    x = np.asarray(history[:, arr("columns")], dtype=np.float64)
    x = np.clip((x - arr("mean")) / arr("scale"), -10, 10)
    current = np.broadcast_to(arr("roots"), (len(x), len(arr("roots")))).copy()
    rows = np.arange(len(x))[:, None]
    for _ in range(int(arr("max_depth")) + 1):
        leaf = arr("leaf")[current]
        if leaf.all():
            break
        value = x[rows, arr("feature")[current]]
        left = value <= arr("threshold")[current]
        left = np.where(np.isnan(value), arr("missing_left")[current], left)
        current = np.where(leaf, current, np.where(left, arr("left")[current], arr("right")[current]))
    if not arr("leaf")[current].all():
        raise ValueError("Invalid tree traversal")
    values = arr("value")[current]
    logits = np.broadcast_to(arr("baseline").reshape(1, -1), (len(x), arr("baseline").size)).copy()
    for index in range(logits.shape[1]):
        logits[:, index] += values[:, arr("tree_class") == index].sum(1)
    if logits.shape[1] == 1:
        # exp(-logaddexp(...)) gives a numerically stable sigmoid.
        positive = np.exp(-np.logaddexp(0., -logits[:, 0]))
        return np.column_stack([1 - positive, positive])
    logits -= logits.max(1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(1, keepdims=True)


class EdgePipeline:
    def __init__(self, bundle, device="cuda"):
        bundle = Path(bundle)
        self.meta = json.loads((bundle / "model.json").read_text(encoding="utf-8"))
        if self.meta.get("format_version") != 1:
            raise ValueError("Unsupported bundle format")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a JetPack-compatible PyTorch build")
        self.model = HETGT(**self.meta["architecture"])
        with np.load(bundle / "graph.npz", allow_pickle=False) as arrays:
            state = {key: torch.from_numpy(arrays["weight/" + key].copy()) for key in self.model.state_dict()}
            self.model.load_state_dict(state, strict=True)
            self.adjacency = torch.from_numpy(arrays["adjacency"].copy()).to(self.device)
            self.table = arrays["reference_table"].copy()
        self.model.to(self.device).eval()
        with np.load(bundle / "trees.npz", allow_pickle=False) as arrays:
            self.bank = {key: arrays[key].copy() for key in arrays.files}

    def predict(self, graph, history, batch_size=1):
        graph = np.asarray(graph)
        history = np.asarray(history)
        shape = (self.meta["architecture"]["n_nodes"], self.meta["architecture"]["input_dim"])
        if graph.ndim != 3 or tuple(graph.shape[1:]) != shape or graph.dtype != np.float32:
            raise ValueError("Graph must be float32 [queries, nodes, input_dim]")
        if history.ndim != 2 or history.shape != (len(graph), self.meta["history_width"]):
            raise ValueError("History shape does not match this bundle")
        if batch_size < 1 or not len(graph) or not np.isfinite(graph).all() or not np.isfinite(history).all():
            raise ValueError("Expected finite nonempty inputs and a positive batch size")
        outputs = []
        with torch.no_grad():
            for start in range(0, len(graph), batch_size):
                logits = self.model(torch.from_numpy(graph[start:start + batch_size]).to(self.device), self.adjacency)
                outputs.append(logits.softmax(-1).cpu().numpy())
        base = np.concatenate(outputs)
        prior = tree_probabilities(self.bank, "prior/", history)
        p = .75 * base + .25 * prior
        final = p.copy()
        top2 = np.sort(np.argsort(p, axis=1, kind="stable")[:, -2:], axis=1)
        for a, b in self.meta["pairs"]:
            mask = (top2[:, 0] == a) & (top2[:, 1] == b)
            if mask.any():
                conditional = tree_probabilities(self.bank, "%d_%d/" % (a, b), history[mask])[:, 1]
                mass = p[mask, a] + p[mask, b]
                refined = .5 * p[mask, b] / np.maximum(mass, 1e-15) + .5 * conditional
                final[mask, b], final[mask, a] = mass * refined, mass * (1 - refined)
        scores = final @ self.table
        return {"probabilities": final, "predictions": final.argmax(1), "graph_probabilities": base,
                "lookup_scores": scores, "reference_order": np.argsort(scores, axis=1)[:, ::-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--input", type=Path, help="NPZ with aligned graph and history arrays")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.smoke_test:
        model = HETGT(63, 50, 36, 6).to(args.device).eval()
        with torch.no_grad():
            probabilities = model(torch.zeros(1, 50, 63, device=args.device), torch.eye(50, device=args.device)).softmax(-1)
        print("Synthetic graph smoke test:", tuple(probabilities.shape), "device:", probabilities.device)
        print("Random weights and synthetic inputs: no detection result or timing measurement.")
        return
    if not all([args.bundle, args.input, args.output]):
        parser.error("--bundle, --input and --output are required for fitted inference")
    if args.output.exists():
        parser.error("Output already exists")
    with np.load(args.input, allow_pickle=False) as arrays:
        result = EdgePipeline(args.bundle, args.device).predict(arrays["graph"], arrays["history"], args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **result)
    print("Saved", len(result["predictions"]), "predictions to", args.output)


if __name__ == "__main__":
    main()
