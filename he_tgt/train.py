"""Fit the fixed HE-TGT graph and history models on development data."""
import argparse
import random
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from threadpoolctl import threadpool_limits
from .model import HETGT
from .history_models import fit_prior, fit_pairs, PRIOR_SETTING, PAIR_SETTING, POLICY
from .utils import config, device_for, load_cache, read_json, reference_table, sha256, write_json


def fit_graph(cache, meta, adjacency, settings, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = HETGT(cache["features"].shape[-1], meta["n_nodes"], meta["n_net"], len(meta["label_names"])).to(device)
    weights = torch.bincount(cache["labels"], minlength=len(meta["label_names"])).float()
    weights = torch.sqrt(weights.max() / weights.clamp_min(1))
    weights = (weights / weights.min()).to(device)
    loader = DataLoader(TensorDataset(cache["features"], cache["labels"], cache["recency"]),
                        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=.0005)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18, eta_min=settings["learning_rate"] * .1)
    adjacency = adjacency.to(device)
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        total = 0.
        for x, y, recency in loader:
            x, y, recency = x.to(device), y.to(device), recency.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (F.cross_entropy(model(x, adjacency), y, weight=weights, reduction="none") * recency).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(y)
        schedule.step()
        row = {"epoch": epoch, "loss": total / len(cache["labels"]), "seconds": time.perf_counter() - started}
        history.append(row)
        print("seed", seed, "epoch", epoch, "/", settings["epochs"], "loss", round(row["loss"], 6), flush=True)
    return model, history


def train(data, output, dataset, seeds, device="auto", threads=2):
    data = Path(data) / dataset
    output = Path(output) / dataset
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    cfg, meta = config(dataset), read_json(data / "metadata.json")
    cache = load_cache(data / "development.pt")
    adjacency = torch.load(data / "adjacency.pt", map_location="cpu", weights_only=True)
    with np.load(data / "history_development.npz", allow_pickle=False) as source:
        history_data = dict(source)
    np.testing.assert_array_equal(cache["raw_indices"].numpy(), history_data["raw_indices"])
    np.testing.assert_array_equal(cache["labels"].numpy(), history_data["labels"])
    settings = {key: cfg["graph"][key] for key in ("learning_rate", "epochs")}
    write_json(output / "metadata.json", meta)
    torch.save(adjacency, output / "adjacency.pt")
    np.save(output / "reference_table.npy", reference_table(cache, len(meta["label_names"])))
    with threadpool_limits(limits=threads):
        prior = fit_prior(history_data, meta["history_schema"], PRIOR_SETTING)
        pairs = fit_pairs(history_data, meta["history_schema"], PAIR_SETTING)
        joblib.dump(prior, output / "prior.joblib", compress=3)
        joblib.dump(pairs, output / "pairs.joblib", compress=3)
        for seed in seeds:
            model, history = fit_graph(cache, meta, adjacency, settings, seed, device_for(device))
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "dataset": dataset, "seed": seed, "settings": settings}, output / ("graph_seed%d.pt" % seed))
            write_json(output / ("training_seed%d.json" % seed), history)
    write_json(output / "fit_lock.json", {"dataset": dataset, "seeds": seeds, "policy": POLICY,
        "settings": settings, "test_used_for_fitting": False,
        "inputs": {p.name: sha256(p) for p in [data / "development.pt", data / "history_development.npz", data / "metadata.json"]},
        "files": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=["IGCPS", "TE-CUP-SEC"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    train(args.data, args.output, args.dataset, args.seeds, args.device, args.threads)
