"""Prepare graph caches and strictly past history from a fused observation CSV."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from baselines import components as tr
from baselines import protocol as exp
from baselines.training import build_development_test
from .history import make_history
from .utils import config, sha256, write_json


def capped(data, cap, seed, name):
    raw = np.asarray(data["original_indices"]).copy()
    output = tr.cap_split_per_class(data, cap, seed, name)
    if cap > 0:
        output["original_indices"] = raw[output["original_indices"]]
    return output


def graph_cache(data, builder, recency_strength):
    targets = np.zeros((len(data["y"]), builder.n_nodes), np.float32)
    eligible = np.zeros(len(data["y"]), np.float32)
    features = []
    for row, label in enumerate(data["y"]):
        features.append(builder.get_node_features_with_history(row, data["X_net_feat"], data["X_phy"],
                                                              data["src_ips"], data["dst_ips"]))
        refs = tr.infer_true_root_nodes(row, data, builder)
        if int(label) > 0 and refs:
            targets[row, sorted(refs)] = 1
            eligible[row] = 1
    recency = np.ones(len(data["y"]), np.float32)
    for label in np.unique(data["y"]):
        ids = np.flatnonzero(data["y"] == label)
        if len(ids) > 1:
            weights = np.exp(recency_strength * (np.linspace(0, 1, len(ids), dtype=np.float32) - 1))
            recency[ids] = weights / max(float(weights.mean()), 1e-8)
    return {"features": torch.tensor(np.asarray(features), dtype=torch.float32),
            "labels": torch.tensor(data["y"], dtype=torch.long),
            "reference_targets": torch.from_numpy(targets), "reference_mask": torch.from_numpy(eligible),
            "recency": torch.from_numpy(recency), "raw_indices": torch.as_tensor(data["original_indices"]).long()}


def prepare(csv, dataset, output, allow_new_data=False):
    cfg = config(dataset)
    digest = sha256(csv)
    if not allow_new_data and digest.lower() != cfg["fusion_csv_sha256"].lower():
        raise ValueError("CSV differs from the paper's source; use --allow-new-data for an explicitly new run")
    output = Path(output) / dataset
    output.mkdir(parents=True, exist_ok=False)
    protocol = exp.load_protocol(csv)
    dev, test = build_development_test(protocol.raw)
    caps = cfg["caps_per_class"]
    full = {"train": protocol.train, "validation": protocol.val, "development": dev, "test": test}
    data = {key: capped(value, caps["train" if key == "development" else key],
                       42 if key in ("train", "development") else 43 if key == "validation" else 44, key)
            for key, value in full.items()}
    builder = exp.make_builder(protocol.raw, history_length=1, temporal=False, preserve_network_features=True)
    adjacency = builder.build_static_adjacency(use_cross_layer=False, use_phy_chain=True, use_net_edges=True)
    adjacency = adjacency + np.eye(builder.n_nodes, dtype=np.float32)
    torch.save(torch.tensor(adjacency / adjacency.sum(1, keepdims=True).clip(min=1), dtype=torch.float32), output / "adjacency.pt")
    indices = {key: value["original_indices"] for key, value in data.items()}
    indices.update({key + "_full": value["original_indices"] for key, value in full.items()})
    for a, b in [("train", "validation"), ("development", "test")]:
        if np.intersect1d(indices[a], indices[b]).size:
            raise ValueError("Overlapping split indices")
    np.savez_compressed(output / "split_indices.npz", **indices)
    caches = {}
    for split, values in data.items():
        caches[split] = graph_cache(values, builder, cfg["graph"]["recency_strength"])
        torch.save(caches[split], output / (split + ".pt"))

    # Recreate the history scaler/vocabularies on full development context only.
    network, physical = protocol.raw["net_feat_cols"], protocol.raw["phy_cols"]
    columns = network + physical
    strings = {"Time", "Source IP", "Destination IP", "Protocol Type"}
    frame = pd.read_csv(csv, usecols=columns + ["Time", "Source IP", "Destination IP"],
                        dtype={c: np.float32 for c in columns if c not in strings})
    times = pd.to_datetime(frame["Time"], format="mixed", utc=True).astype("int64").to_numpy()
    order = np.argsort(times, kind="stable")
    frame, times = frame.iloc[order].reset_index(drop=True), times[order]
    development = frame.iloc[indices["development_full"]]
    vocab = {c: {v: i for i, v in enumerate(sorted(development[c].astype(str).unique()))}
             for c in columns if not pd.api.types.is_numeric_dtype(development[c])}
    values = np.empty((len(frame), len(columns)), np.float32)
    for j, c in enumerate(columns):
        values[:, j] = (frame[c].astype(str).map(vocab[c]).fillna(-1) if c in vocab else
                        pd.to_numeric(frame[c], errors="coerce").fillna(0)).to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(values[indices["development_full"]])
    values = scaler.transform(values).astype(np.float32)
    streams = pd.factorize(frame["Source IP"].astype(str) + "\x00" + frame["Destination IP"].astype(str), sort=True)[0]
    joblib.dump({"scaler": scaler, "vocabularies": vocab, "columns": columns,
                 "network_columns": network, "physical_columns": physical,
                 "fit_raw_indices": indices["development_full"]}, output / "preprocessing.joblib", compress=3)
    for split in ["development", "test"]:
        context = indices["development_full"] if split == "development" else np.arange(len(frame))
        arrays = make_history(values[context], times[context], streams[context], context, indices[split])
        arrays.update(labels=caches[split]["labels"].numpy(), recency=caches[split]["recency"].numpy())
        np.savez_compressed(output / ("history_" + split + ".npz"), **arrays)
    write_json(output / "metadata.json", {"dataset": dataset, "data_sha256": digest,
        "n_nodes": builder.n_nodes, "n_net": builder.n_net, "label_names": protocol.raw["label_names"],
        "node_names": list(builder.net_nodes) + list(builder.phy_nodes),
        "history_schema": {"network_width": len(network), "total_width": len(columns), "classes": protocol.raw["n_classes"]},
        "split_counts": {key: len(value["y"]) for key, value in data.items()}, "config": cfg})
    print("Prepared", dataset, "in", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dataset", choices=["IGCPS", "TE-CUP-SEC"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-new-data", action="store_true")
    args = parser.parse_args()
    prepare(args.csv, args.dataset, args.output, args.allow_new_data)
