"""Audit the final single-encoder DA-TGT checkpoint without retraining."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

import graphsage_bias_calibration as calibration
import hgan_conditional_calibration as conditional
import reviewer_experiments as exp
import temporal_robust_experiments as temporal


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "sr_com_causal_v2.csv")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "single_hgan_joint_causal_rootfix_validation_20260825",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=ROOT / "hgan_conditional_calibration_20260825",
    )
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "final_hgan_audit_20260825"
    )
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    protocol = exp.load_protocol(args.csv)
    _, test = temporal.build_development_test(protocol.raw, dev_ratio=0.8)
    checkpoint = args.model_dir / "HGAN-Trace-Joint__seed42.pt"
    model, _, builder, cache, adjacency = conditional.load_model(
        checkpoint, protocol.raw, test
    )
    lock = json.loads(
        (args.calibration_dir / "validation_lock.json").read_text(encoding="utf-8")
    )

    raw_probs, root_scores = conditional.collect(model, cache, adjacency)
    probs = conditional.locked_probabilities(raw_probs, lock)
    labels = cache["labels"].numpy()
    predictions = probs.argmax(axis=1)
    aggregate = calibration.metrics(labels, probs)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=range(len(test["label_names"])),
        zero_division=0,
    )

    ece = exp.calibration_metrics(labels, predictions, probs)
    per_class = [
        {
            "class": str(test["label_names"][index]),
            "precision": float(100 * precision[index]),
            "recall": float(100 * recall[index]),
            "f1": float(100 * f1[index]),
            "support": int(support[index]),
        }
        for index in range(len(test["label_names"]))
    ]

    sample = cache["features"][:1]
    model.eval()
    with torch.no_grad():
        for _ in range(100):
            model(sample, adjacency)
        latencies = []
        for _ in range(args.repeats):
            started = time.perf_counter_ns()
            output = model(sample, adjacency)
            base = torch.softmax(output["anomaly_logits"], dim=-1).cpu().numpy()
            conditional.locked_probabilities(base, lock)
            latencies.append((time.perf_counter_ns() - started) / 1e6)

    state_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name not in {"detection_pool.weight", "detection_pool.bias"}
    )
    result = {
        **aggregate,
        "ece15": float(ece["ece15"]),
        "brier": float(ece["brier"]),
        "nll": float(ece["nll"]),
        "entropy_error_spearman": float(ece["entropy_error_spearman"]),
        "reliability_bins": ece["reliability_bins"],
        "per_class": per_class,
        "confusion_matrix": ece["confusion_matrix"].tolist()
        if "confusion_matrix" in ece
        else None,
        "parameters": int(state_parameters + 12),
        "base_parameters": int(state_parameters),
        "conditional_head_parameters": 12,
        "latency_ms": {
            "median": float(statistics.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "mean": float(np.mean(latencies)),
            "repeats": int(args.repeats),
            "batch_size": 1,
            "scope": "one encoder forward plus folded-head-equivalent adjustment",
        },
        "test_samples": int(len(labels)),
        "single_encoder": True,
        "ensemble": False,
        "distillation": False,
    }
    (args.outdir / "audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
