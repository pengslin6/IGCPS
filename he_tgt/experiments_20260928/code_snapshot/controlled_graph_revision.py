"""Validation-locked encoder controls for the manuscript revision.

These are controlled adaptations, not reproductions of the June manuscript.
Existing repeatedly inspected recordings remain retrospective test data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GATv2Conv

import reviewer_experiments as exp
import single_hgan_joint_experiment as joint
import temporal_robust_experiments as temporal
import trace as tr

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "IGCPS": ("sr_com_causal_v2.csv", 0, 0, 0, 1.0),
    "TE-CUP-SEC": ("tecupsec_information_physical_causal_v3.csv", 1000, 500, 4000, 2.0),
}
METRICS = ("accuracy", "f1", "auc", "top1", "mrr", "ndcg5")
ENCODERS = ("mean_transformer", "gat_transformer", "mean_cross", "gat_cross")
ABLATIONS = ("full", "shared_type", "no_local", "no_exchange", "separate_tasks")
CANDIDATE = next(c for c in joint.CANDIDATES if c.name == "typed_full_robust")


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=exp.jsonable), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TypedGATBlock(nn.Module):
    """GATv2 messages on the same observed local edges, with type-specific weights."""

    def __init__(self, dim, heads, dropout, n_net, shared=False):
        super().__init__()
        self.n_net = n_net
        self.shared = shared
        self.net = GATv2Conv(dim, dim // heads, heads=heads, dropout=dropout,
                             add_self_loops=False, share_weights=True)
        self.phy = None if shared else GATv2Conv(
            dim, dim // heads, heads=heads, dropout=dropout,
            add_self_loops=False, share_weights=True)
        self.norm = nn.LayerNorm(dim)
        self._edges = {}

    def apply_domain(self, nodes, adjacency, conv, key):
        batch, count, dim = nodes.shape
        cache_key = (key, batch, count, nodes.device)
        # The supplied adjacency is immutable throughout an experiment run.
        if cache_key not in self._edges:
            target, source = torch.where(adjacency > 0)
            edge = torch.stack([source, target])
            offsets = torch.arange(batch, device=nodes.device) * count
            self._edges[cache_key] = (edge[:, None, :] + offsets[None, :, None]).reshape(2, -1)
        output = conv(nodes.reshape(batch * count, dim), self._edges[cache_key])
        return output.reshape(batch, count, dim)

    def forward(self, nodes, adjacency):
        n = self.n_net
        net = self.apply_domain(nodes[:, :n], adjacency[:n, :n], self.net, "net")
        phy = self.apply_domain(nodes[:, n:], adjacency[n:, n:],
                                self.net if self.shared else self.phy, "phy")
        return self.norm(nodes + F.gelu(torch.cat([net, phy], dim=1)))


class CrossExchange(nn.Module):
    """Batched version of the retained encoder's gated bidirectional attention."""

    def __init__(self, dim, heads, dropout, n_net):
        super().__init__()
        self.n_net = n_net
        self.net_from_phy = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.phy_from_net = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.net_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.phy_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.net_norm = nn.LayerNorm(dim)
        self.phy_norm = nn.LayerNorm(dim)

    def forward(self, sequence):
        token, nodes = sequence[:, :1], sequence[:, 1:]
        net, phy = nodes[:, :self.n_net], nodes[:, self.n_net:]
        nmsg, _ = self.net_from_phy(net, phy, phy, need_weights=False)
        pmsg, _ = self.phy_from_net(phy, net, net, need_weights=False)
        net = self.net_norm(net + self.net_gate(torch.cat([net, nmsg], -1)) * nmsg)
        phy = self.phy_norm(phy + self.phy_gate(torch.cat([phy, pmsg], -1)) * pmsg)
        context = token + 0.5 * (net.mean(1, keepdim=True) + phy.mean(1, keepdim=True))
        return torch.cat([context, net, phy], 1)


class NoExchange(nn.Module):
    def __init__(self, n_net):
        super().__init__()
        self.n_net = n_net

    def forward(self, sequence):
        token, nodes = sequence[:, :1], sequence[:, 1:]
        context = token + 0.5 * (nodes[:, :self.n_net].mean(1, keepdim=True)
                                 + nodes[:, self.n_net:].mean(1, keepdim=True))
        return torch.cat([context, nodes], 1)


class ControlledGraph(joint.DATGTJoint):
    def __init__(self, input_dim, n_nodes, n_net, n_classes, encoder, ablation="full"):
        candidate = replace(CANDIDATE, decouple_detection_root=ablation == "separate_tasks")
        super().__init__(input_dim, n_nodes, n_net, n_classes, candidate)
        self.encoder_name, self.ablation = encoder, ablation
        if ablation == "shared_type":
            self.phy_adapter = self.net_adapter
            self.type_embedding.requires_grad_(False)
            with torch.no_grad():
                self.type_embedding.zero_()
        if encoder.startswith("gat"):
            self.local_blocks = nn.ModuleList([
                TypedGATBlock(candidate.model_dim, candidate.n_heads, candidate.dropout,
                              n_net, shared=ablation == "shared_type") for _ in range(2)])
        if encoder.endswith("cross"):
            self.global_encoder = CrossExchange(candidate.model_dim, candidate.n_heads,
                                                 candidate.dropout, n_net)
        if ablation == "no_local":
            self.local_blocks = nn.ModuleList()
        if ablation == "no_exchange":
            self.global_encoder = NoExchange(n_net)


def supervised_root_loss(logits, targets, eligible, labels):
    supervised = joint.root_objective(logits, targets, eligible, 0.0)
    normal = labels == 0
    # An unannotated attack is unknown, not a negative root annotation.
    negative = F.binary_cross_entropy_with_logits(logits[normal], targets[normal]) if normal.any() else logits.sum() * 0
    return supervised + CANDIDATE.normal_root_weight * negative


def class_root_table(cache, n_classes):
    labels = cache["labels"].numpy()
    roots = cache["root_targets"].numpy()
    eligible = cache["root_mask"].numpy() > 0
    table = np.zeros((n_classes, roots.shape[1]), dtype=np.float32)
    for label in range(n_classes):
        use = eligible & (labels == label)
        if use.any():
            table[label] = roots[use].mean(0)
    return table


def ranking(scores, cache):
    eligible = cache["root_mask"].numpy() > 0
    truth = cache["root_targets"].numpy()[eligible]
    scores = np.asarray(scores)[eligible]
    # Preserve the existing paper's ranking rule; ties also get sensitivity bounds.
    if not len(truth):
        return {"top1": 0., "mrr": 0., "ndcg5": 0., "eligible": 0,
                "top1_tie_lower": 0., "top1_tie_upper": 0., "top1_tie_expected": 0.}
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranked = np.take_along_axis(truth, order, axis=1)
    ranks = np.argmax(ranked > 0, axis=1) + 1
    discount = 1 / np.log2(np.arange(2, min(5, truth.shape[1]) + 2))
    dcg = (ranked[:, :len(discount)] * discount).sum(1)
    ideal = np.array([discount[:min(int(row.sum()), 5)].sum() for row in truth])
    ties = np.isclose(scores, scores.max(1, keepdims=True), rtol=0, atol=1e-8)
    tied_correct = (ties * truth).sum(1)
    return {"top1": float((ranks == 1).mean() * 100),
            "mrr": float((1 / ranks).mean() * 100),
            "ndcg5": float((dcg / ideal).mean() * 100),
            "eligible": int(eligible.sum()),
            "top1_tie_lower": float((tied_correct == ties.sum(1)).mean() * 100),
            "top1_tie_upper": float((tied_correct > 0).mean() * 100),
            "top1_tie_expected": float((tied_correct / ties.sum(1)).mean() * 100)}


def collect(model, cache, adjacency, batch_size=128):
    model.eval()
    probs, roots = [], []
    with torch.inference_mode():
        for (features,) in DataLoader(TensorDataset(cache["features"]), batch_size=batch_size):
            out = model(features, adjacency)
            probs.append(out["anomaly_logits"].softmax(-1).numpy())
            roots.append(out["root_logits"].sigmoid().numpy())
    return np.concatenate(probs), np.concatenate(roots)


def evaluate(probs, roots, cache, lookup=None):
    labels = cache["labels"].numpy()
    result = {"accuracy": float(accuracy_score(labels, probs.argmax(1)) * 100),
              "f1": float(f1_score(labels, probs.argmax(1), average="macro", zero_division=0) * 100),
              "auc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro") * 100),
              **ranking(roots, cache)}
    if lookup is not None:
        result["class_lookup_soft"] = ranking(probs @ lookup, cache)
        result["class_lookup_hard"] = ranking(lookup[probs.argmax(1)], cache)
        result["class_lookup_oracle_diagnostic_only"] = ranking(lookup[labels], cache)
    return result


def balanced_key(metrics):
    f1, root = metrics["f1"], metrics["top1"]
    return 2 * f1 * root / max(f1 + root, 1e-12)


def build_model(cache, builder, encoder, ablation, seed):
    joint.set_seed(seed)
    return ControlledGraph(cache["features"].shape[-1], builder.n_nodes, builder.n_net,
                           int(cache["labels"].max()) + 1, encoder, ablation)


def train(cache, builder, adjacency, encoder, ablation, seed, epochs, output,
          validation=None):
    model = build_model(cache, builder, encoder, ablation, seed)
    weights = joint.class_weights(cache["labels"])
    loader = DataLoader(TensorDataset(*(cache[k] for k in
                        ("features", "labels", "root_targets", "root_mask", "recency"))),
                        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CANDIDATE.learning_rate,
                                 weight_decay=CANDIDATE.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18,
                                                          eta_min=CANDIDATE.learning_rate * .1)
    history, best, best_state = [], None, None
    lookup = class_root_table(cache, int(cache["labels"].max()) + 1)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total = 0.
        for features, labels, roots, eligible, recency in loader:
            out = model(features, adjacency)
            loss = (F.cross_entropy(out["anomaly_logits"], labels, weight=weights,
                                    reduction="none") * recency).mean()
            if ablation != "separate_tasks":
                loss = loss + CANDIDATE.root_weight * supervised_root_loss(
                    out["root_logits"], roots, eligible, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(labels)
        scheduler.step()
        row = {"epoch": epoch, "loss": total / len(cache["labels"]),
               "seconds": time.perf_counter() - started}
        if validation is not None:
            probs, roots = collect(model, validation, adjacency)
            row.update(evaluate(probs, roots, validation, lookup))
            row["selection_score"] = balanced_key(row)
            key = (row["selection_score"], row["auc"], -epoch)
            if best is None or key > best[0]:
                best = (key, row)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(row)
        dump(output.with_suffix(".history.json"), history)
        detail = f" val Acc/F1/Top1={row['accuracy']:.2f}/{row['f1']:.2f}/{row['top1']:.2f}" if validation is not None else ""
        print(f"{output.stem} epoch={epoch}/{epochs} loss={row['loss']:.4f} train={row['seconds']:.1f}s{detail}", flush=True)
    if validation is not None:
        model.load_state_dict(best_state)
    if ablation == "separate_tasks":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.root_scorer.parameters():
            p.requires_grad_(True)
        root_optimizer = torch.optim.AdamW(model.root_scorer.parameters(),
                                           lr=CANDIDATE.learning_rate, weight_decay=CANDIDATE.weight_decay)
        for epoch in range(epochs):
            model.eval()
            model.root_scorer.train()
            total = 0.
            for features, labels, roots, eligible, _ in loader:
                out = model(features, adjacency)
                loss = supervised_root_loss(out["root_logits"], roots, eligible, labels)
                root_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.root_scorer.parameters(), 1.)
                root_optimizer.step()
                total += loss.item() * len(labels)
            print(f"{output.stem} frozen-encoder root epoch={epoch+1}/{epochs} loss={total/len(cache['labels']):.4f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "candidate": asdict(CANDIDATE),
                "encoder": encoder, "ablation": ablation, "seed": seed,
                "epochs": best[1]["epoch"] if best else epochs}, output.with_suffix(".pt"))
    return model, best[1] if best else None, lookup


def prepare(dataset, final=False):
    filename, train_cap, val_cap, test_cap, recency = DATASETS[dataset]
    joint.RECENCY_STRENGTH = recency
    protocol = exp.load_protocol(ROOT / filename)
    builder = joint.make_builder(protocol.raw, CANDIDATE)
    if final:
        development, test = temporal.build_development_test(protocol.raw)
        development = tr.cap_split_per_class(development, train_cap, seed=42, split_name="Development")
        test = tr.cap_split_per_class(test, test_cap, seed=44, split_name="Test")
        cache, monitor = joint.build_cache(development, builder), joint.build_cache(test, builder)
    else:
        train_data = tr.cap_split_per_class(protocol.train, train_cap, seed=42, split_name="Train")
        val_data = tr.cap_split_per_class(protocol.val, val_cap, seed=43, split_name="Validation")
        cache, monitor = joint.build_cache(train_data, builder), joint.build_cache(val_data, builder)
    del protocol
    gc.collect()
    return builder, joint.static_adjacency(builder, CANDIDATE), cache, monitor


def timing(model, cache, adjacency):
    model.eval()
    sample = cache["features"][:1]
    with torch.inference_mode():
        for _ in range(10):
            model(sample, adjacency)
        measurements = []
        for _ in range(50):
            start = time.perf_counter()
            model(sample, adjacency)
            measurements.append((time.perf_counter() - start) * 1000)
    return {"batch1_median_ms": float(np.median(measurements)),
            "batch1_p95_ms": float(np.percentile(measurements, 95)),
            "parameters": sum(p.numel() for p in model.parameters())}


def validation(args):
    rows = []
    for dataset in DATASETS:
        builder, adjacency, cache, val = prepare(dataset)
        for encoder in ENCODERS:
            prefix = args.out / f"validation__{dataset}__{encoder}"
            result_file = prefix.with_suffix(".result.json")
            if result_file.exists():
                rows.append(json.loads(result_file.read_text()))
                continue
            model, best, lookup = train(cache, builder, adjacency, encoder, "full", 42,
                                         args.epochs, prefix, val)
            result = {"dataset": dataset, "encoder": encoder, **best, **timing(model, val, adjacency)}
            dump(result_file, result)
            rows.append(result)
            del model
            gc.collect()
        del cache, val
    scores = {encoder: float(np.mean([r["selection_score"] for r in rows if r["encoder"] == encoder])) for encoder in ENCODERS}
    selected = max(ENCODERS, key=lambda e: (scores[e], -ENCODERS.index(e)))
    lock = {"encoder": selected, "selection_seed": 42, "scores": scores,
            "epochs": {r["dataset"]: r["epoch"] for r in rows if r["encoder"] == selected},
            "test_used_for_selection": False,
            "scope": "Retrospective existing recordings; no new independent acquisition",
            "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    dump(args.out / "validation_lock.json", lock)
    dump(args.out / "validation_results.json", rows)
    print("VALIDATION LOCK " + json.dumps(lock), flush=True)


def final(args):
    lock = json.loads((args.out / "validation_lock.json").read_text())
    for dataset in DATASETS:
        builder, adjacency, cache, test = prepare(dataset, final=True)
        for ablation in ABLATIONS:
            for seed in args.seeds:
                prefix = args.out / f"final__{dataset}__{ablation}__seed{seed}"
                result_file = prefix.with_suffix(".result.json")
                if result_file.exists():
                    continue
                model, _, lookup = train(cache, builder, adjacency, lock["encoder"], ablation,
                                          seed, lock["epochs"][dataset], prefix)
                probs, roots = collect(model, test, adjacency)
                result = {"dataset": dataset, "encoder": lock["encoder"], "ablation": ablation,
                          "seed": seed, "epochs": lock["epochs"][dataset],
                          **evaluate(probs, roots, test, lookup), **timing(model, test, adjacency)}
                dump(result_file, result)
                np.savez_compressed(prefix.with_suffix(".predictions.npz"),
                                    probabilities=probs, root_scores=roots)
                print("FINAL RESULT " + json.dumps(result), flush=True)
                del model
                gc.collect()
        del cache, test
    records = [json.loads(p.read_text()) for p in args.out.glob("final__*.result.json")]
    dump(args.out / "final_results.json", records)
    pd.json_normalize(records).to_csv(args.out / "final_results.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["validation", "final", "all"], default="all")
    parser.add_argument("--out", type=Path, default=ROOT / "controlled_graph_revision_20260906")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    with (args.out / "progress.log").open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout = exp.Tee(sys.__stdout__, log)
        manifest_path = args.out / "protocol.json"
        manifest = {"source_sha256": sha256(__file__), "candidate": asdict(CANDIDATE),
                    "datasets": DATASETS, "encoders": ENCODERS, "ablations": ABLATIONS,
                    "validation_epochs": args.epochs, "selection_seed": 42, "final_seeds": args.seeds,
                    "threads": args.threads, "torch": torch.__version__,
                    "calibration": False, "ensemble": False, "distillation": False,
                    "selection": "One encoder across both datasets; maximize mean validation harmonic mean of macro-F1 and Top1; epoch ties use AUC then earlier epoch",
                    "unknown_attack_root_labels": "Excluded from root loss, not treated as all-negative",
                    "schema": "Known full-recording endpoint and category vocabulary, training-only numeric scaling",
                    "test_status": "Previously inspected recordings, retrospective only",
                    "dataset_sha256": {d: sha256(ROOT / item[0]) for d, item in DATASETS.items()},
                    "import_sha256": {p: sha256(ROOT / p) for p in
                                      ("trace.py", "single_hgan_joint_experiment.py", "reviewer_experiments.py", "temporal_robust_experiments.py")}}
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous != json.loads(json.dumps(manifest)):
                raise RuntimeError("Frozen experiment protocol differs. Use a new output folder; do not overwrite old runs.")
        else:
            dump(manifest_path, manifest)
        if args.stage in ("validation", "all"):
            validation(args)
        if args.stage in ("final", "all"):
            final(args)


if __name__ == "__main__":
    main()
