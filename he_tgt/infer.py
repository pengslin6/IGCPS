"""Inference with fitted graph, history and class-reference models."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import torch
from .model import HETGT
from .history_models import probabilities, pair_probabilities, decision, POLICY
from .utils import config, device_for, read_json


class Pipeline:
    def __init__(self, directory, seed=11, device="auto"):
        directory = Path(directory)
        self.meta = read_json(directory / "metadata.json")
        self.device = device_for(device)
        checkpoint = torch.load(directory / ("graph_seed%d.pt" % seed), map_location="cpu", weights_only=True)
        expected = config(self.meta["dataset"])["graph"]
        if checkpoint["dataset"] != self.meta["dataset"] or checkpoint["seed"] != seed:
            raise ValueError("Checkpoint identity mismatch")
        if checkpoint["settings"] != {k: expected[k] for k in ["learning_rate", "epochs"]}:
            raise ValueError("Checkpoint settings mismatch")
        state = checkpoint["state_dict"]
        self.width = state["input_projection.0.weight"].shape[1]
        self.model = HETGT(self.width, self.meta["n_nodes"], self.meta["n_net"], len(self.meta["label_names"]))
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.adjacency = torch.load(directory / "adjacency.pt", map_location="cpu", weights_only=True).to(self.device)
        self.prior = joblib.load(directory / "prior.joblib")
        self.pairs = joblib.load(directory / "pairs.joblib")
        self.table = np.load(directory / "reference_table.npy", allow_pickle=False)
        if self.table.shape != (len(self.meta["label_names"]), self.meta["n_nodes"]) or not np.isfinite(self.table).all():
            raise ValueError("Invalid reference table")

    def predict(self, features, history, batch_size=128, variant="full"):
        features = torch.as_tensor(features)
        if features.ndim != 3 or tuple(features.shape[1:]) != (self.meta["n_nodes"], self.width):
            raise ValueError("Input graph shape does not match the fitted model")
        if features.dtype != torch.float32 or not torch.isfinite(features).all() or len(features) == 0 or batch_size < 1:
            raise ValueError("Expected a nonempty finite float32 graph batch")
        if len(history["history"]) != len(features) or not np.isfinite(history["history"]).all():
            raise ValueError("Invalid or misaligned history inputs")
        with torch.no_grad():
            graph = np.concatenate([self.model(x.to(self.device), self.adjacency).softmax(-1).cpu().numpy()
                                    for x in features.split(batch_size)])
        if variant == "graph_only":
            final = graph
        else:
            prior = probabilities(self.prior, history)
            if variant == "without_pair":
                final = (1 - POLICY["gamma"]) * graph + POLICY["gamma"] * prior
            elif variant in ("full", "pair_only"):
                policy = POLICY if variant == "full" else dict(POLICY, gamma=0.)
                final = decision(graph, prior, pair_probabilities(self.pairs, history), policy)
            else:
                raise ValueError("Unknown component-removal variant")
        scores = final @ self.table
        return {"probabilities": final, "predictions": final.argmax(1), "graph_probabilities": graph,
                "lookup_scores": scores, "reference_order": np.argsort(scores, axis=1)[:, ::-1]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    graph = torch.load(args.graph, map_location="cpu", weights_only=True)
    features = graph["features"] if isinstance(graph, dict) else graph
    with np.load(args.history, allow_pickle=False) as arrays:
        history = {k: arrays[k] for k in ("static", "history")}
    result = Pipeline(args.model, args.seed, args.device).predict(features, history, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **result)
    print("Saved", len(features), "predictions to", args.output)
