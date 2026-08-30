"""Synthetic graph-size latency audit for the final DA-TGT architecture."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import single_hgan_joint_experiment as joint


ROOT = Path(__file__).resolve().parent


def adjacency(nodes: int) -> torch.Tensor:
    matrix = np.eye(nodes, dtype=np.float32)
    for index in range(nodes - 1):
        matrix[index, index + 1] = 1.0
        matrix[index + 1, index] = 1.0
    matrix /= matrix.sum(axis=1, keepdims=True)
    return torch.tensor(matrix, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=ROOT / "final_hgan_audit_20260825" / "scaling.json"
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    candidate = joint.CANDIDATES[1]
    rows = []
    for nodes in args.sizes:
        network_nodes = max(2, nodes // 2)
        model = joint.DATGTJoint(
            input_dim=63,
            n_nodes=nodes,
            n_net=network_nodes,
            n_classes=6,
            candidate=candidate,
        ).eval()
        features = torch.randn(1, nodes, 63)
        graph = adjacency(nodes)
        with torch.no_grad():
            for _ in range(30):
                model(features, graph)
            durations = []
            for _ in range(args.repeats):
                started = time.perf_counter_ns()
                model(features, graph)
                durations.append((time.perf_counter_ns() - started) / 1e6)
        row = {
            "nodes": nodes,
            "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95)),
            "mean_ms": float(np.mean(durations)),
            "parameters": int(sum(p.numel() for p in model.parameters())),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "device": "cpu",
        "batch_size": 1,
        "repeats_per_size": args.repeats,
        "synthetic_topology": "self loops plus bidirectional chain",
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
