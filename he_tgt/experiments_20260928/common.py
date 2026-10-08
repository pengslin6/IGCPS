"""Shared, hash-checked data and metrics for the reviewer-requested audit."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code_snapshot"))
import numpy as np
import pandas as pd
import torch
import controlled_graph_revision as c
import hgan_conditional_calibration as conditional

SOURCE = Path(r"C:\Users\pcsys\Desktop\论文\HGAN-Trace")
DATA = {
    "IGCPS": {"file": "sr_com_causal_v2.csv", "sha256": "6c8aaea31e4820fee7230ba35e04ce3907edaa5f5a56051cd5ec427d1f5998df", "archive": "hgan_causal_multiseed_ns_corrected_20260907"},
    "TE-CUP-SEC": {"file": "tecupsec_information_physical_causal_v3.csv", "sha256": "092e56fb31a684c16826202c209bb8827af4933e9e7edf046d9888afb55c4c6f", "archive": "paired_te_main_20260831"},
}
SEEDS = [11, 22, 33, 44, 55]
METRICS = ["accuracy", "f1", "auc", "top1", "mrr", "ndcg5"]
CONFIDENCE_EDGES = [0.0, 0.6, 0.8, 0.9, 1.000001]
c.joint.HISTORY_LENGTH, c.joint.USE_TEMPORAL = 1, False


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=c.exp.jsonable), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def cap_with_indices(data, cap, seed, name):
    raw_indices = np.asarray(data["original_indices"]).copy()
    out = c.tr.cap_split_per_class(data, cap, seed=seed, split_name=name)
    if cap > 0:
        out["original_indices"] = raw_indices[out["original_indices"]]
    return out


def load_raw_splits(dataset):
    spec = DATA[dataset]
    path = SOURCE / spec["file"]
    digest = sha(path)
    if digest != spec["sha256"]:
        raise RuntimeError(f"Data provenance mismatch for {dataset}: {digest}")
    print(f"Verified data hash: {dataset} {digest}", flush=True)
    protocol = c.exp.load_protocol(path)
    builder = c.joint.make_builder(protocol.raw, c.CANDIDATE)
    development, test = c.temporal.build_development_test(protocol.raw)
    np.testing.assert_array_equal(test["original_indices"], protocol.test["original_indices"])
    _, train_cap, val_cap, test_cap, recency = c.DATASETS[dataset]
    c.joint.RECENCY_STRENGTH = recency
    splits = {"all": protocol.raw, "train_full": protocol.train,
              "validation_full": protocol.val, "development_full": development,
              "test_full": test}
    splits["train"] = cap_with_indices(protocol.train, train_cap, 42, "Train")
    splits["validation"] = cap_with_indices(protocol.val, val_cap, 43, "Validation")
    splits["development"] = cap_with_indices(development, train_cap, 42, "Development")
    splits["test"] = cap_with_indices(test, test_cap, 44, "Test")
    for a, b in [("train", "validation"), ("train", "test"), ("development", "test")]:
        assert not np.intersect1d(splits[a]["original_indices"], splits[b]["original_indices"]).size
    return protocol, builder, splits


def prepare(dataset):
    out = HERE / "data_cache" / dataset
    if (out / "complete.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    protocol, builder, splits = load_raw_splits(dataset)
    raw = protocol.raw
    node_names = [str(n) for n in list(builder.net_nodes) + list(builder.phy_nodes)]
    refs = [tuple(sorted(c.tr.infer_true_root_nodes(i, raw, builder))) for i in range(len(raw["y"]))]
    sets = sorted(set(refs))
    set_id = {value: i for i, value in enumerate(sets)}
    codes = np.asarray([set_id[value] for value in refs], dtype=np.int16)
    np.savez_compressed(out / "all_reference_labels.npz", labels=raw["y"], reference_set_id=codes,
                        times=np.asarray(raw["times"], dtype=str), source=np.asarray(raw["src_ips"], dtype=str))
    reference_sets = [{"id": i, "node_indices": list(value), "nodes": [node_names[j] for j in value]} for i, value in enumerate(sets)]
    statistics, frequencies, split_indices = [], [], {}
    for split, data in splits.items():
        indices = np.arange(len(raw["y"])) if split == "all" else np.asarray(data["original_indices"])
        split_indices[split] = indices
        np.testing.assert_array_equal(raw["y"][indices], data["y"])
        np.testing.assert_array_equal(np.asarray(raw["times"])[indices], data["times"])
        for label, name in enumerate(raw["label_names"]):
            selected = indices[raw["y"][indices] == label]
            values, counts = np.unique(codes[selected], return_counts=True)
            valid = np.asarray([bool(sets[int(v)]) for v in values]) & (label != 0)
            elig_count = int(counts[valid].sum())
            p = counts[valid] / max(elig_count, 1)
            statistics.append({"dataset": dataset, "split": split, "class_id": label, "class": name,
                "windows": len(selected), "eligible_windows": elig_count,
                "distinct_reference_sets": int(valid.sum()),
                "modal_set_share": float(p.max()) if len(p) else None,
                "reference_set_entropy_bits": float(-(p * np.log2(p)).sum()) if len(p) else None,
                "distinct_source_endpoints": int(len(np.unique(raw["src_ips"][selected])))})
            for value, count in zip(values, counts):
                frequencies.append({"dataset": dataset, "split": split, "class": name,
                    "reference_set_id": int(value), "nodes": " | ".join(node_names[j] for j in sets[int(value)]),
                    "windows": int(count), "class_share": float(count / max(len(selected), 1))})
    pd.DataFrame(statistics).to_csv(out / "label_statistics.csv", index=False)
    pd.DataFrame(frequencies).to_csv(out / "reference_set_frequencies.csv", index=False)
    np.savez_compressed(out / "split_indices.npz", **split_indices)
    metadata = {"dataset": dataset, "source_path": str(SOURCE / DATA[dataset]["file"]),
        "data_sha256": DATA[dataset]["sha256"], "reference_policy": c.tr.ROOT_REFERENCE_POLICY,
        "node_names": node_names, "n_net": builder.n_net, "n_nodes": builder.n_nodes,
        "label_names": raw["label_names"], "reference_sets": reference_sets,
        "count_unit": "fused observation windows; independent incident identities are unavailable",
        "split_counts": {k: len(v["y"]) for k, v in splits.items()},
        "split_indices_sha256": {k: array_sha(v) for k, v in split_indices.items()},
        "code_sha256": {p.name: sha(p) for p in (HERE / "code_snapshot").glob("*.py")}}
    dump(out / "metadata.json", metadata)
    torch.save(c.joint.static_adjacency(builder, c.CANDIDATE), out / "adjacency.pt")
    for split in ("train", "validation", "development", "test"):
        cache = c.joint.build_cache(splits[split], builder)
        cache["raw_indices"] = torch.as_tensor(split_indices[split], dtype=torch.long)
        # Check cache targets directly against the audited full-recording rules.
        for i, raw_i in enumerate(split_indices[split]):
            expected = refs[int(raw_i)] if int(cache["labels"][i]) > 0 else ()
            actual = tuple(torch.nonzero(cache["root_targets"][i], as_tuple=True)[0].tolist())
            assert expected == actual
        torch.save(cache, out / f"{split}.pt")
        print(f"Cached {dataset} {split}: {len(cache['labels'])} rows, {tuple(cache['features'].shape)}", flush=True)
        del cache
        gc.collect()
    dump(out / "complete.json", {"verified": True, "data_sha256": DATA[dataset]["sha256"]})
    print(f"LABEL AUDIT COMPLETE {dataset}", flush=True)


def load_cache(dataset, split):
    return torch.load(HERE / "data_cache" / dataset / f"{split}.pt", map_location="cpu", weights_only=True)


def load_meta(dataset):
    return json.loads((HERE / "data_cache" / dataset / "metadata.json").read_text(encoding="utf-8"))


def ranking_per_row(scores, targets):
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranked = np.take_along_axis(targets, order, axis=1)
    eligible = targets.sum(1) > 0
    ranks = np.argmax(ranked > 0, axis=1) + 1
    discount = 1 / np.log2(np.arange(2, min(5, targets.shape[1]) + 2))
    dcg = (ranked[:, :len(discount)] * discount).sum(1)
    ideal = np.asarray([discount[:min(int(row.sum()), 5)].sum() for row in targets])
    tied = np.isclose(scores, scores.max(1, keepdims=True), rtol=0, atol=1e-8)
    return {"top1": np.where(eligible, (ranks == 1) * 100., np.nan),
            "mrr": np.where(eligible, 100. / ranks, np.nan),
            "ndcg5": np.where(eligible, 100. * dcg / np.maximum(ideal, 1e-12), np.nan),
            "top1_tie_expected": np.where(eligible, 100. * (tied * targets).sum(1) / tied.sum(1), np.nan)}


def groups(labels, probabilities, label_names):
    correct = labels == probabilities.argmax(1)
    confidence = probabilities.max(1)
    result = {"all": np.ones(len(labels), dtype=bool), "class_correct": correct, "class_wrong": ~correct}
    for i, name in enumerate(label_names):
        result[f"class:{name}"] = labels == i
    for lo, hi in zip(CONFIDENCE_EDGES[:-1], CONFIDENCE_EDGES[1:]):
        result[f"confidence:{lo:.1f}-{min(hi,1.):.1f}"] = (confidence >= lo) & (confidence < hi)
    if "NM" in label_names:
        result["NM_to_PM"] = (labels == label_names.index("NM")) & (probabilities.argmax(1) == label_names.index("PM"))
    return result
