"""Extract representative root-ranking examples from the final HGAN checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import hgan_conditional_calibration as conditional
import reviewer_experiments as exp
import temporal_robust_experiments as temporal
import trace as tr


ROOT = Path(__file__).resolve().parent


def node_name(builder, index: int) -> str:
    if index < builder.n_net:
        values = getattr(builder, "unique_ips", None) or getattr(builder, "ip_list", None)
        if values is not None and index < len(values):
            return f"Endpoint:{values[index]}"
        inverse = {value: key for key, value in builder.ip_to_idx.items()}
        return f"Endpoint:{inverse.get(index, index)}"
    local = index - builder.n_net
    return f"Process:{builder.phy_cols[local]}"


def main() -> None:
    protocol = exp.load_protocol(ROOT / "sr_com_causal_v2.csv")
    _, test = temporal.build_development_test(protocol.raw, dev_ratio=0.8)
    checkpoint = (
        ROOT / "single_hgan_joint_causal_rootfix_validation_20260825"
        / "HGAN-Trace-Joint__seed42.pt"
    )
    model, _, builder, cache, adjacency = conditional.load_model(
        checkpoint, protocol.raw, test
    )
    _, root_scores = conditional.collect(model, cache, adjacency)
    output = []
    for class_id in range(1, len(test["label_names"])):
        candidates = np.where(np.asarray(test["y"]) == class_id)[0]
        chosen = None
        for sample_index in candidates:
            references = tr.infer_true_root_nodes(sample_index, test, builder)
            if not references:
                continue
            order = np.argsort(-root_scores[sample_index])
            if int(order[0]) in references:
                chosen = int(sample_index)
                break
        if chosen is None:
            chosen = int(candidates[0])
        references = tr.infer_true_root_nodes(chosen, test, builder)
        order = np.argsort(-root_scores[chosen])[:5]
        output.append({
            "class": str(test["label_names"][class_id]),
            "test_index": chosen,
            "reference_nodes": [node_name(builder, int(index)) for index in sorted(references)],
            "top5": [
                {
                    "rank": rank,
                    "node": node_name(builder, int(index)),
                    "score": float(root_scores[chosen, index]),
                    "is_reference": bool(int(index) in references),
                }
                for rank, index in enumerate(order, start=1)
            ],
        })
    target = ROOT / "final_hgan_audit_20260825" / "root_examples.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
