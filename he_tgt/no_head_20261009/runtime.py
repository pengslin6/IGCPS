"""Inference for the current no-node-head graph/history/reference-lookup pipeline."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
import joblib
from model import NodeFreeGraph

POLICY = {"gamma": .25, "beta": 1., "rho": .5}
SEEDS = (11, 22, 33, 44, 55)
SETTINGS = {"IGCPS": {"learning_rate": .002, "epochs": 17},
            "TE-CUP-SEC": {"learning_rate": .0007, "epochs": 16}}


def configure(artifact_root):
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source / "unified_detection_20260929"))
    import unified as u
    import pair_model as pair
    root = Path(artifact_root).resolve()
    u.ROOT, u.MODEL, u.AUDIT = root, root / "detection_history_final_20260929", root / "experiments_20260928"
    pair.OUT = root / "unified_detection_20260929/pair_refinement"
    return u, pair


def lookup(development, classes):
    labels = development["labels"].numpy()
    targets = development["root_targets"].numpy()
    eligible = development["root_mask"].numpy() > 0
    table = np.zeros((classes, targets.shape[1]), np.float32)
    for c in range(classes):
        mask = (labels == c) & eligible
        if mask.any():
            table[c] = targets[mask].mean(0)
    return table


class Pipeline:
    def __init__(self, artifact_root, checkpoint_root, dataset, seed=11, device="auto", reference_table=None):
        if dataset not in SETTINGS or seed not in SEEDS:
            raise ValueError("Unsupported dataset/seed")
        self.u, self.pair = configure(artifact_root)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.dataset, self.seed, self.meta = dataset, seed, self.u.meta(dataset)
        path = Path(checkpoint_root) / dataset / f"no_head_seed{seed}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing no-head checkpoint: {path}; see REPRODUCING.md")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("dataset") != dataset or checkpoint.get("seed") != seed or checkpoint.get("settings") != SETTINGS[dataset]:
            raise ValueError("Checkpoint identity/settings do not match the current model")
        state = checkpoint["state_dict"]
        if any("root_scorer" in name for name in state):
            raise ValueError("The current entry requires a no-node-head checkpoint")
        self.width = state["input_projection.0.weight"].shape[1]
        self.graph = NodeFreeGraph(self.width, self.meta)
        self.graph.load_state_dict(state, strict=True)
        self.graph.to(self.device).eval()
        self.adjacency = torch.load(self.u.AUDIT / "data_cache" / dataset / "adjacency.pt", map_location="cpu", weights_only=True).to(self.device)
        if self.u.read(self.pair.OUT / "final_fit_lock.json")["policy"] != POLICY:
            raise ValueError("History artifacts use a different refinement policy")
        self.prior = joblib.load(self.pair.OUT / dataset / "final_prior.joblib")
        self.pairs = joblib.load(self.pair.OUT / dataset / "final_pairs.joblib")
        self.table = (lookup(self.u.graph(dataset, "development"), len(self.meta["label_names"]))
                      if reference_table is None else np.load(reference_table, allow_pickle=False))
        if self.table.shape != (len(self.meta["label_names"]), self.meta["n_nodes"]) or not np.isfinite(self.table).all() or (self.table < 0).any() or (self.table > 1).any():
            raise ValueError("Invalid development-fitted reference table")

    def predict(self, features, history, batch_size=128):
        features = torch.as_tensor(features)
        if batch_size < 1 or features.ndim != 3 or tuple(features.shape[1:]) != (self.meta["n_nodes"], self.width):
            raise ValueError("Graph shape or batch size does not match the model")
        if features.dtype != torch.float32 or not torch.isfinite(features).all():
            raise ValueError("Graph features must be finite float32 values")
        for key in ("static", "history"):
            if key not in history or len(history[key]) != len(features) or not np.isfinite(history[key]).all():
                raise ValueError(f"Invalid or misaligned history field: {key}")
        if len(features) == 0:
            raise ValueError("At least one query is required")
        with torch.inference_mode():
            graph = np.concatenate([self.graph(x.to(self.device), self.adjacency).softmax(-1).cpu().numpy()
                                    for x in features.split(batch_size)])
        prior = self.u.probabilities(self.prior, history)
        pairs = self.pair.pair_probabilities(self.pairs, history)
        probabilities = self.pair.decision(graph, prior, pairs, POLICY)
        scores = probabilities @ self.table
        return {"probabilities": probabilities, "predictions": probabilities.argmax(1),
                "graph_probabilities": graph, "lookup_scores": scores,
                "reference_order": np.argsort(scores, axis=1)[:, ::-1]}
