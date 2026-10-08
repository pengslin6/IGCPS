"""Locked full-pipeline comparison of HE-TGT and information-matched DT-GNN."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "experiments_20260928"))
import fair_control as fc
import common as audit
sys.path.insert(0, str(ROOT / "unified_detection_20260929"))
import pair_model as pm
import unified as u

np, torch = u.np, u.torch
DATASETS = u.DATASETS
SEEDS = u.SEEDS
MODELS = ["DA-TGT", "DT-GNN"]
FAIR = ROOT / "experiments_20260928/fair_control"
PAIR = ROOT / "unified_detection_20260929/pair_refinement"
METRICS = ["accuracy", "f1", "normal_fpr", "anomaly_recall", "focus_recall", "focus_f1",
           "direct_top1", "lookup_top1", "lookup_mrr", "lookup_ndcg5"]


def fixed_sources():
    sources = [HERE / "PROTOCOL.md", Path(__file__), ROOT / "experiments_20260928/PROTOCOL.md",
               ROOT / "experiments_20260928/fair_control.py", PAIR / "final_fit_lock.json",
               PAIR / "selection_lock.json"]
    for dataset in DATASETS:
        sources += [FAIR / dataset / "selection_lock.json", u.MODEL / dataset / "train.npz",
                    PAIR / dataset / "final_prior.joblib", PAIR / dataset / "final_pairs.joblib"]
        for model in MODELS:
            chosen = u.read(FAIR / dataset / "selection_lock.json")["selected"][model]
            lr = round(chosen["learning_rate"] * 1e6)
            sources.append(FAIR / dataset / f"validation__{model}__lr{lr}.pt")
            for seed in SEEDS:
                sources += [FAIR / dataset / f"final__{model}__seed{seed}.pt",
                            FAIR / dataset / f"final__{model}__seed{seed}.predictions.npz"]
        sources.append(FAIR / dataset / "final__DT-GNN__seed11.predictions.npz")
    return {str(p.resolve()): u.sha(p) for p in sorted(set(sources))}


def check_fitted_assets():
    locked = u.read(PAIR / "final_fit_lock.json")
    for path, digest in locked["hashes"].items():
        assert u.sha(path) == digest, path
    return locked


def graph_validation(dataset, model_name):
    graph = u.graph(dataset, "validation")
    selected = u.read(FAIR / dataset / "selection_lock.json")["selected"][model_name]
    lr = round(selected["learning_rate"] * 1e6)
    path = FAIR / dataset / f"validation__{model_name}__lr{lr}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["model"] == model_name
    model = fc.build_model(model_name, graph, u.meta(dataset), 42)
    model.load_state_dict(checkpoint["state_dict"])
    adjacency = torch.load(u.AUDIT / "data_cache" / dataset / "adjacency.pt", weights_only=True)
    probs, _ = audit.c.collect(model, graph, adjacency)
    if model_name == "DA-TGT":
        original = u.base_validation(dataset)
        np.testing.assert_allclose(probs, original, atol=1e-7, rtol=1e-6)
    return probs


def validation_inputs():
    outputs = {}
    for dataset in DATASETS:
        arr = u.data(dataset, "validation")
        graph = u.graph(dataset, "validation")
        for key in ["labels", "raw_indices"]:
            np.testing.assert_array_equal(arr[key], graph[key].numpy())
        prior_file = u.HERE / dataset / "screen/full_history__leaves7_min60.npz"
        prior = np.load(prior_file)["probabilities"]
        pair_outputs = {}
        for setting in pm.settings():
            saved = np.load(PAIR / dataset / "screen" / f"{setting['name']}.npz")
            pair_outputs[setting["name"]] = {tuple(map(int, key.split("_"))): saved[key]
                                              for key in saved.files}
        outputs[dataset] = {"arr": arr, "prior": prior, "pairs": pair_outputs,
                            "graph": {model: graph_validation(dataset, model) for model in MODELS}}
        for model in MODELS:
            assert len(outputs[dataset]["graph"][model]) == len(arr["labels"])
    return outputs


def select():
    check_fitted_assets()
    source_hashes = fixed_sources()
    inputs = validation_inputs()
    baseline = {model: {dataset: u.metrics(dataset, inputs[dataset]["arr"]["labels"],
                        inputs[dataset]["graph"][model].argmax(1)) for dataset in DATASETS}
                for model in MODELS}
    results = {}
    for model in MODELS:
        rows = []
        for setting in pm.settings():
            for beta, gamma, rho in itertools.product([0., .5, 1.], [0., .25, .5], [.5, .75, 1.]):
                policy = {"beta": beta, "gamma": gamma, "rho": rho}
                measured = {}
                for dataset in DATASETS:
                    block = inputs[dataset]
                    probabilities = pm.decision(block["graph"][model], block["prior"],
                                                block["pairs"][setting["name"]], policy)
                    measured[dataset] = u.metrics(dataset, block["arr"]["labels"], probabilities.argmax(1))
                gains = [measured[d]["f1"] - baseline[model][d]["f1"] for d in DATASETS]
                mean_fpr = float(np.mean([measured[d]["normal_fpr"] for d in DATASETS]))
                rows.append({"name": setting["name"] + f"__beta{beta}_gamma{gamma}_rho{rho}",
                             "setting": setting, "policy": policy, "metrics": measured,
                             "worst_f1_gain": float(min(gains)), "mean_f1_gain": float(np.mean(gains)),
                             "mean_fpr": mean_fpr})
        assert len(rows) == 108
        ranked = sorted(rows, key=lambda r: (-r["worst_f1_gain"], -r["mean_f1_gain"],
                                           r["mean_fpr"], r["name"]))
        results[model] = {"selected": ranked[0], "candidate_count": len(rows),
                          "graph_validation_baseline": baseline[model], "all_candidates": rows}
        print("SELECTED", model, ranked[0]["name"], ranked[0]["worst_f1_gain"], flush=True)
    lock = {"source_hashes": source_hashes, "test_read_for_selection": False,
            "same_policy_budget": True, "selection": results}
    if (HERE / "selection_lock.json").exists():
        assert u.read(HERE / "selection_lock.json") == lock
    else:
        u.dump(HERE / "selection_lock.json", lock)
    print("VALIDATION SELECTION LOCKED", flush=True)


def saved_graph(dataset, model_name, seed, graph, arr):
    path = FAIR / dataset / f"final__{model_name}__seed{seed}.predictions.npz"
    with np.load(path) as saved:
        for key in ["labels", "raw_indices"]:
            np.testing.assert_array_equal(saved[key], graph[key].numpy())
            np.testing.assert_array_equal(saved[key], arr[key])
        probabilities, roots = saved["probabilities"], saved["root_scores"]
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    return probabilities, roots


def measured(dataset, probs, roots, graph, lookup):
    classification = u.metrics(dataset, graph["labels"].numpy(), probs.argmax(1))
    ranking = audit.c.evaluate(probs, roots, graph, lookup)
    return {**{key: classification[key] for key in u.KEYS},
            "auc": ranking["auc"], "direct_top1": ranking["top1"],
            "direct_mrr": ranking["mrr"], "direct_ndcg5": ranking["ndcg5"],
            "lookup_top1": ranking["class_lookup_soft"]["top1"],
            "lookup_mrr": ranking["class_lookup_soft"]["mrr"],
            "lookup_ndcg5": ranking["class_lookup_soft"]["ndcg5"]}


def summarize(rows):
    return {metric: {"mean": float(np.mean([r["metrics"][metric] for r in rows])),
                     "sd": float(np.std([r["metrics"][metric] for r in rows], ddof=1))}
            for metric in METRICS + ["auc", "direct_mrr", "direct_ndcg5"]}


def fit_selected():
    check_fitted_assets()
    lock = u.read(HERE / "selection_lock.json")
    assert fixed_sources() == lock["source_hashes"]
    fitted = {}
    for model in MODELS:
        selected = lock["selection"][model]["selected"]
        for dataset in DATASETS:
            path = HERE / dataset / f"equal_budget_pairs__{model}.joblib"
            if not path.exists():
                arr = dict(np.load(u.MODEL / dataset / "train.npz"))
                assert np.array_equal(arr["raw_indices"], u.graph(dataset, "development")["raw_indices"].numpy())
                head = pm.fit(arr, u.feature_metadata(dataset), selected["setting"])
                path.parent.mkdir(parents=True, exist_ok=True)
                u.joblib.dump(head, path, compress=3)
            fitted[str(path.resolve())] = u.sha(path)
            print("FITTED", model, dataset, selected["setting"]["name"], flush=True)
    new_lock = {"selection_lock_sha256": u.sha(HERE / "selection_lock.json"),
                "fitted_model_hashes": fitted, "test_used_for_fitting": False}
    if (HERE / "fit_lock.json").exists():
        assert u.read(HERE / "fit_lock.json") == new_lock
    else:
        u.dump(HERE / "fit_lock.json", new_lock)
    print("SELECTED FINAL HEADS LOCKED", flush=True)


def evaluate():
    check_fitted_assets()
    locked = u.read(HERE / "selection_lock.json")
    assert locked["test_read_for_selection"] is False
    assert fixed_sources() == locked["source_hashes"]
    fitted_lock = u.read(HERE / "fit_lock.json")
    assert fitted_lock["selection_lock_sha256"] == u.sha(HERE / "selection_lock.json")
    assert fitted_lock["test_used_for_fitting"] is False
    for path, digest in fitted_lock["fitted_model_hashes"].items():
        assert u.sha(path) == digest
    fixed = u.read(PAIR / "final_fit_lock.json")
    results = {"design": "retrospective information-matched full-pipeline comparison",
               "fixed_policy": fixed["policy"], "validation_selection": {}, "datasets": {}}
    for model in MODELS:
        chosen = locked["selection"][model]
        results["validation_selection"][model] = {"candidate_count": chosen["candidate_count"],
                                                   "selected": chosen["selected"]}
    for dataset in DATASETS:
        graph = u.graph(dataset, "test")
        arr = dict(np.load(u.MODEL / dataset / "test.npz"))
        for key in ["labels", "raw_indices"]:
            np.testing.assert_array_equal(graph[key].numpy(), arr[key])
        dev = u.graph(dataset, "development")
        lookup = audit.c.class_root_table(dev, len(u.meta(dataset)["label_names"]))
        prior_head = u.joblib.load(PAIR / dataset / "final_prior.joblib")
        pair_head = u.joblib.load(PAIR / dataset / "final_pairs.joblib")
        prior = u.probabilities(prior_head, arr)
        pairs = pm.pair_probabilities(pair_head, arr)
        rows_by_condition = {condition: {model: [] for model in MODELS}
                             for condition in ["graph_only", "fixed_full", "equal_budget_full"]}
        chosen_pairs = {}
        for model in MODELS:
            selected = locked["selection"][model]["selected"]
            selected_head = u.joblib.load(HERE / dataset / f"equal_budget_pairs__{model}.joblib")
            assert selected_head["setting"] == selected["setting"]
            chosen_pairs[model] = pm.pair_probabilities(selected_head, arr)
        for seed in SEEDS:
            for model in MODELS:
                base, roots = saved_graph(dataset, model, seed, graph, arr)
                conditions = {"graph_only": base,
                              "fixed_full": pm.decision(base, prior, pairs, fixed["policy"])}
                chosen = locked["selection"][model]["selected"]
                conditions["equal_budget_full"] = pm.decision(base, prior, chosen_pairs[model],
                                                                chosen["policy"])
                if model == "DA-TGT":
                    reference = np.load(PAIR / dataset / f"final_seed{seed}.npz")
                    np.testing.assert_array_equal(conditions["fixed_full"], reference["probabilities"])
                for condition, probs in conditions.items():
                    row = {"dataset": dataset, "model": model, "seed": seed,
                           "condition": condition, "metrics": measured(dataset, probs, roots, graph, lookup)}
                    rows_by_condition[condition][model].append(row)
                    dest = HERE / dataset / f"{condition}__{model}__seed{seed}.json"
                    u.dump(dest, row)
                np.savez_compressed(HERE / dataset / f"fixed_full__{model}__seed{seed}.npz",
                                    probabilities=conditions["fixed_full"], root_scores=roots,
                                    labels=arr["labels"], raw_indices=arr["raw_indices"])
        results["datasets"][dataset] = {}
        for condition, model_rows in rows_by_condition.items():
            results["datasets"][dataset][condition] = {model: summarize(model_rows[model]) for model in MODELS}
            results["datasets"][dataset][condition]["paired_DA_minus_DT"] = {
                metric: [float(model_rows["DA-TGT"][i]["metrics"][metric] -
                               model_rows["DT-GNN"][i]["metrics"][metric]) for i in range(len(SEEDS))]
                for metric in METRICS}
            print("RESULT", dataset, condition,
                  {model: results["datasets"][dataset][condition][model]["f1"]["mean"] for model in MODELS},
                  flush=True)
    u.dump(HERE / "results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["select", "fit_selected", "evaluate"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with u.threadpool_limits(limits=2):
        globals()[args.phase]()
