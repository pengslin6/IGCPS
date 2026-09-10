"""Export current DA-TGT detection and root-ranking reports.

The exporter loads the validation-locked checkpoints used by the revised
manuscript.  It does not retrain, tune, or alter either model.  Reports contain
all anomalous samples in the final temporal test partitions and explicitly
mark samples for which the released dataset has no represented exact-node
reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import graphsage_bias_calibration as calibration
import hgan_conditional_calibration as conditional
import reviewer_experiments as exp
import temporal_robust_experiments as temporal
import trace as tr


DEFAULT_ROOT = Path(__file__).resolve().parent
METRIC_KEYS = ("accuracy", "f1", "auc", "top1", "mrr", "ndcg5")
SEEDS = (11, 22, 33, 44, 55)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_policy(checkpoint: Path, dataset: str) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = payload.get("root_reference_policy")
    if dataset == "IGCPS" and policy != tr.ROOT_REFERENCE_POLICY:
        raise ValueError("IGCPS report requires a checkpoint refitted with the corrected NS references")
    return {"checkpoint_reference_policy": policy or "TE-CUP-SEC mapping unchanged; legacy metadata"}


def node_name(builder, index: int) -> str:
    """Return a stable, human-readable graph node name."""
    if index < builder.n_net:
        values = getattr(builder, "unique_ips", None) or getattr(
            builder, "ip_list", None
        )
        if values is not None and index < len(values):
            return f"Endpoint:{values[index]}"
        inverse = {value: key for key, value in builder.ip_to_idx.items()}
        return f"Endpoint:{inverse.get(index, index)}"
    local = index - builder.n_net
    return f"Process:{builder.phy_cols[local]}"


def scalar_at(values, index: int, default: str = "N/A") -> str:
    if values is None or index >= len(values):
        return default
    value = values[index]
    if value is None:
        return default
    return str(value)


def expected_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(payload[key]) for key in METRIC_KEYS if key in payload}


def calculate_metrics(labels: np.ndarray, probs: np.ndarray,
                      roots: np.ndarray, data: dict, builder) -> dict:
    result = calibration.metrics(labels, probs)
    root = tr.compute_strict_traceback_metrics(roots, data, builder)
    result.update({
        "top1": float(root["rca"]),
        "mrr": float(root["mrr"]),
        "ndcg5": float(root["ndcg"]),
        "mean_reference_rank": float(root["apd"]),
        "trace_eval_total": int(root["trace_eval_total"]),
        "trace_eval_skipped": int(root["trace_eval_skipped"]),
    })
    return result


def validate_metrics(name: str, actual: dict, expected: dict,
                     tolerance: float = 1e-5) -> None:
    failures = []
    for key, expected_value in expected.items():
        difference = abs(float(actual[key]) - expected_value)
        if difference > tolerance:
            failures.append(
                f"{key}: actual={actual[key]:.10f}, "
                f"expected={expected_value:.10f}, diff={difference:.10f}"
            )
    if failures:
        raise RuntimeError(
            f"{name} report metrics do not match the locked result:\n  "
            + "\n  ".join(failures)
        )


def write_report(path: Path, dataset_name: str, protocol_description: str,
                 data: dict, builder, probs: np.ndarray, roots: np.ndarray,
                 metrics: dict, checkpoint: Path, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(data["y"], dtype=int)
    predictions = probs.argmax(axis=1)
    label_names = [str(value) for value in data["label_names"]]
    confusion = np.asarray(metrics["confusion_matrix"], dtype=int)

    with path.open("w", encoding="utf-8", newline="\n") as report:
        report.write("=" * 96 + "\n")
        report.write(f"DA-TGT Root-Localization Report: {dataset_name}\n")
        report.write("=" * 96 + "\n")
        report.write("Model: single shared DA-TGT encoder, K=1\n")
        report.write(f"Protocol: {protocol_description}\n")
        report.write(f"Checkpoint: {checkpoint.name}\n")
        for key, value in (metadata or {}).items():
            report.write(f"{key}: {value}\n")
        report.write("Report scope: one fixed-seed checkpoint, not the five-seed mean\n")
        report.write(f"Evaluation reference policy: {tr.ROOT_REFERENCE_POLICY}\n")
        if dataset_name == "IGCPS":
            report.write("NS references: high-pressure and sub-high-pressure switches only\n")
            report.write("NS excludes the medium-pressure-heater switch\n")
            report.write("The production loader rejects checkpoints lacking the corrected NS supervision policy\n")
        report.write("The exporter performs inference only; it does not train or select a model\n")
        report.write("Reference provenance: scenario-derived; not independent sample-wise causal annotations\n")
        report.write("Root output: supervised exact-node ranking, not a causal path\n")
        report.write("Top-k detail: five highest node-head sigmoid scores\n\n")

        report.write("[Detection and Localization Summary]\n")
        report.write(f"  Test samples: {len(labels)}\n")
        report.write(f"  Anomaly samples: {int(np.count_nonzero(labels))}\n")
        report.write(f"  Accuracy: {metrics['accuracy']:.6f}%\n")
        report.write(f"  Macro F1: {metrics['f1']:.6f}%\n")
        report.write(f"  Macro one-vs-rest AUC: {metrics['auc']:.6f}%\n")
        report.write(f"  Root Top-1: {metrics['top1']:.6f}%\n")
        report.write(f"  Root MRR: {metrics['mrr']:.6f}%\n")
        report.write(f"  Root NDCG@5: {metrics['ndcg5']:.6f}%\n")
        report.write(
            f"  Mean first-reference rank: {metrics['mean_reference_rank']:.6f}\n"
        )
        report.write(
            f"  Root-evaluable anomaly samples: {metrics['trace_eval_total']}\n"
        )
        report.write(
            "  Anomaly samples skipped from exact-root metrics: "
            f"{metrics['trace_eval_skipped']}\n\n"
        )

        report.write("[Per-Class Detection Counts]\n")
        for class_id, class_name in enumerate(label_names):
            support = int(confusion[class_id].sum())
            predicted = int(confusion[:, class_id].sum())
            correct = int(confusion[class_id, class_id])
            recall = 100.0 * correct / max(support, 1)
            report.write(
                f"  {class_name}: true={support}, predicted={predicted}, "
                f"correct={correct}, recall={recall:.6f}%\n"
            )

        report.write("\n" + "-" * 96 + "\n")
        report.write("[Anomalous Sample Details]\n")
        report.write("-" * 96 + "\n")

        times = data.get("times")
        source_ips = data.get("src_ips")
        destination_ips = data.get("dst_ips")
        for sample_index in np.flatnonzero(labels != 0):
            sample_index = int(sample_index)
            true_id = int(labels[sample_index])
            predicted_id = int(predictions[sample_index])
            references = tr.infer_true_root_nodes(sample_index, data, builder)
            ranking = np.argsort(roots[sample_index])[::-1]
            first_reference_rank = next(
                (
                    rank
                    for rank, node_index in enumerate(ranking, start=1)
                    if int(node_index) in references
                ),
                None,
            )

            report.write(f"\nSample index: {sample_index}\n")
            report.write(f"Time: {scalar_at(times, sample_index)}\n")
            report.write(
                f"Flow endpoint pair: {scalar_at(source_ips, sample_index)} -> "
                f"{scalar_at(destination_ips, sample_index)}\n"
            )
            report.write(f"True class: {label_names[true_id]}\n")
            report.write(
                f"Predicted class: {label_names[predicted_id]} "
                f"(confidence={probs[sample_index, predicted_id]:.6f})\n"
            )
            report.write(
                "Exact-root eligible: " + ("yes" if references else "no") + "\n"
            )
            report.write(
                "Reference node(s): "
                + (
                    ", ".join(
                        node_name(builder, int(index))
                        for index in sorted(references)
                    )
                    if references
                    else "N/A (excluded from exact-root metrics)"
                )
                + "\n"
            )
            report.write(
                "First reference rank: "
                + (str(first_reference_rank) if first_reference_rank else "N/A")
                + "\n"
            )
            report.write("Top-5 ranked nodes:\n")
            for rank, node_index in enumerate(ranking[:5], start=1):
                node_index = int(node_index)
                marker = " [REFERENCE]" if node_index in references else ""
                report.write(
                    f"  {rank}. {node_name(builder, node_index)} "
                    f"score={roots[sample_index, node_index]:.6f}{marker}\n"
                )


def load_igcps(root: Path, seed: int = 11):
    csv_path = root / "sr_com_causal_v2.csv"
    protocol = exp.load_protocol(csv_path)
    _, test = temporal.build_development_test(protocol.raw, dev_ratio=0.8)
    checkpoint = (
        root / "hgan_causal_multiseed_ns_corrected_20260907"
        / f"HGAN-Trace__seed{seed}.pt"
    )
    validate_checkpoint_policy(checkpoint, "IGCPS")
    model, _, builder, cache, adjacency = conditional.load_model(
        checkpoint, protocol.raw, test
    )
    raw_probs, roots = conditional.collect(model, cache, adjacency)
    lock = json.loads(
        (checkpoint.parent / f"calibration_lock__seed{seed}.json")
        .read_text(encoding="utf-8")
    )
    probs = conditional.locked_probabilities(raw_probs, lock)
    metrics = calculate_metrics(cache["labels"].numpy(), probs, roots, test, builder)
    expected = expected_metrics(checkpoint.parent / f"result__seed{seed}.json")
    validate_metrics("IGCPS corrected-supervision checkpoint", metrics, expected)
    with np.load(checkpoint.parent / f"predictions__seed{seed}.npz") as saved:
        np.testing.assert_array_equal(saved["labels"], cache["labels"].numpy())
        np.testing.assert_allclose(probs, saved["probabilities"], atol=2e-5)
        np.testing.assert_allclose(roots, saved["root_scores"], atol=2e-5)
    ns_id = list(test["label_names"]).index("NS")
    for index in np.flatnonzero(test["y"] == ns_id):
        references = tr.infer_true_root_nodes(int(index), test, builder)
        assert len(references) == 2
        assert {builder.phy_cols[i - builder.n_net] for i in references} == set(tr.IGCPS_NS_ROOT_COLUMNS)
    return test, builder, probs, roots, metrics, checkpoint


def load_tecupsec(root: Path, seed: int = 11):
    csv_path = root / "tecupsec_information_physical_causal_v3.csv"
    protocol = exp.load_protocol(csv_path)
    _, test = temporal.build_development_test(protocol.raw, dev_ratio=0.8)
    test = tr.cap_split_per_class(test, 4000, seed=44, split_name="Test")
    checkpoint = (
        root / "paired_te_main_20260831"
        / f"HGAN-Trace__seed{seed}.pt"
    )
    model, _, builder, cache, adjacency = conditional.load_model(
        checkpoint, protocol.raw, test
    )
    probs, roots = conditional.collect(model, cache, adjacency)
    metrics = calculate_metrics(cache["labels"].numpy(), probs, roots, test, builder)
    expected = expected_metrics(
        checkpoint.parent / f"result__seed{seed}.json"
    )
    validate_metrics("TE-CUP-SEC", metrics, expected)
    return test, builder, probs, roots, metrics, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", choices=("all", "IGCPS", "TE-CUP-SEC"), default="all")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=11)
    parser.add_argument(
        "--outdir", type=Path, default=DEFAULT_ROOT / "traceback_reports_current"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    outdir = args.outdir.resolve()
    torch.set_num_threads(4)

    if args.dataset in ("all", "IGCPS"):
        print("Loading IGCPS final checkpoint...", flush=True)
        data, builder, probs, roots, metrics, checkpoint = load_igcps(root, args.seed)
        igcps_path = outdir / "igcps_traceback_report.txt"
        write_report(
            igcps_path,
            "IGCPS",
            "class-wise chronological first 80% development / last 20% test; "
            "validation-locked separate 12-coefficient conditional probability adjustment",
            data, builder, probs, roots, metrics, checkpoint,
            {"seed": args.seed, "checkpoint_relative_path": str(checkpoint.relative_to(root)),
             "checkpoint_sha256": file_sha256(checkpoint),
             "calibration_lock_sha256": file_sha256(checkpoint.parent / f"calibration_lock__seed{args.seed}.json"),
             **validate_checkpoint_policy(checkpoint, "IGCPS")},
        )
        print(json.dumps({key: metrics[key] for key in METRIC_KEYS}), flush=True)
        print(f"Wrote {igcps_path}", flush=True)

    if args.dataset in ("all", "TE-CUP-SEC"):
        print("Loading TE-CUP-SEC final checkpoint...", flush=True)
        data, builder, probs, roots, metrics, checkpoint = load_tecupsec(root, args.seed)
        te_path = outdir / "te_cup_sec_traceback_report.txt"
        write_report(
            te_path,
            "TE-CUP-SEC",
            "class-wise chronological first 80% development / last 20% test; "
            "test capped at 4,000 samples per class with seed 44",
            data, builder, probs, roots, metrics, checkpoint,
            {"seed": args.seed, "checkpoint_relative_path": str(checkpoint.relative_to(root)),
             "checkpoint_sha256": file_sha256(checkpoint),
             **validate_checkpoint_policy(checkpoint, "TE-CUP-SEC")},
        )
        print(json.dumps({key: metrics[key] for key in METRIC_KEYS}), flush=True)
        print(f"Wrote {te_path}", flush=True)


if __name__ == "__main__":
    main()
