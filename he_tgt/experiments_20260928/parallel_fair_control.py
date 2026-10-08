"""Schedule independent control fits concurrently without changing the locked fit routine.

Uses exactly fair_control.fit and its selection/output rules. Do not run this
coordinator concurrently with fair_control.py on the same dataset.
"""
import argparse
import concurrent.futures
import json
import time
import fair_control as f


def worker(dataset, phase, model_name, rate, seed, epochs):
    f.torch.set_num_threads(2)
    folder = f.OUT / dataset
    meta = f.load_meta(dataset)
    adjacency = f.torch.load(f.HERE / "data_cache" / dataset / "adjacency.pt", weights_only=True)
    if phase == "validation":
        prefix = folder / f"validation__{model_name}__lr{int(rate * 1000000)}"
    else:
        prefix = folder / f"final__{model_name}__seed{seed}"
    result_path = prefix.with_suffix(".result.json")
    if result_path.exists():
        return json.loads(result_path.read_text())
    cache = f.load_cache(dataset, "train" if phase == "validation" else "development")
    evaluation = f.load_cache(dataset, "validation" if phase == "validation" else "test")
    model, history, elapsed = f.fit(model_name, cache, meta, adjacency, seed, rate, epochs, prefix,
                                    evaluation if phase == "validation" else None)
    if phase == "validation":
        best = max(history, key=lambda r: (r["selection_score"], r["auc"], -r["epoch"]))
        result = {"model": model_name, "learning_rate": rate, "epoch": best["epoch"],
                  "selection_score": best["selection_score"], "auc": best["auc"],
                  "validation": best, "elapsed_seconds": elapsed}
    else:
        lookup = f.c.class_root_table(cache, len(meta["label_names"]))
        probs, roots = f.c.collect(model, evaluation, adjacency)
        metrics = f.c.evaluate(probs, roots, evaluation, lookup)
        per_class = f.classification_report(evaluation["labels"].numpy(), probs.argmax(1),
            target_names=meta["label_names"], output_dict=True, zero_division=0)
        result = {"dataset": dataset, "model": model_name, "seed": seed, "epochs": epochs,
                  "learning_rate": rate, "training_seconds": elapsed,
                  "selection_lock_sha256": f.sha(folder / "selection_lock.json"), "metrics": metrics,
                  "per_class": per_class, "checkpoint_sha256": f.sha(prefix.with_suffix(".pt"))}
        f.np.savez_compressed(prefix.with_suffix(".predictions.npz"), labels=evaluation["labels"].numpy(),
            probabilities=probs, root_scores=roots, raw_indices=evaluation["raw_indices"].numpy())
    f.dump(result_path, result)
    print(f"COMPLETED {dataset} {phase} {model_name} seed={seed} lr={rate}", flush=True)
    return result


def run(dataset):
    folder = f.OUT / dataset
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    # Initialization is copied from the serial coordinator; the fit implementation
    # itself is always imported unchanged from the locked original runner.
    meta = f.load_meta(dataset)
    train = f.load_cache(dataset, "train")
    adjacency = f.torch.load(f.HERE / "data_cache" / dataset / "adjacency.pt", weights_only=True)
    f.checks(dataset, train, meta, adjacency)
    del train
    f.gc.collect()
    manifest = {"dataset": dataset, "data_sha256": f.DATA[dataset]["sha256"],
        "protocol_sha256": f.sha(f.HERE / "PROTOCOL.md"), "runner_sha256": f.sha(f.__file__),
        "common_sha256": f.sha(f.HERE / "common.py"), "source_sha256": meta["code_sha256"],
        "torch": f.torch.__version__, "threads": 2, "models": f.MODELS,
        "learning_rates": f.LEARNING_RATES, "validation_epochs": f.MAX_EPOCHS,
        "selection_seed": 42, "final_seeds": f.SEEDS, "test_status": "retrospective",
        "input_noise": .02, "calibration": False, "pretraining": False, "frozen_encoder": False,
        "loss": "weighted recency CE + .30*(positive-weighted eligible node BCE + .03*true-normal node BCE)",
        "native_dropout_preserved": True, "identical_architectures_or_parameter_counts": False,
        "run_started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ("runner_sha256", "common_sha256", "protocol_sha256", "data_sha256"):
            assert old[key] == manifest[key]
    else:
        f.dump(manifest_path, manifest)
    f.dump(folder / "execution_coordinator.json", {"coordinator_sha256": f.sha(__file__),
        "workers": 2, "threads_per_fit": 2, "fit_function": "unchanged fair_control.fit",
        "reason": "Schedule independent fits concurrently; no change to data, training, budget or selection.",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    lock_path = folder / "selection_lock.json"
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        if not lock_path.exists():
            jobs = [pool.submit(worker, dataset, "validation", name, lr, 42, f.MAX_EPOCHS)
                    for name in f.MODELS for lr in f.LEARNING_RATES]
            results = [job.result() for job in jobs]
            selections = {name: max([r for r in results if r["model"] == name],
                key=lambda r: (r["selection_score"], r["auc"], -r["epoch"], -r["learning_rate"])) for name in f.MODELS}
            f.dump(lock_path, {"dataset": dataset, "test_used_for_this_selection": False,
                "retrospective_dataset": True, "selected": selections,
                "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        lock = json.loads(lock_path.read_text())
        jobs = [pool.submit(worker, dataset, "final", name, lock["selected"][name]["learning_rate"],
                            seed, lock["selected"][name]["epoch"]) for name in f.MODELS for seed in f.SEEDS]
        for job in concurrent.futures.as_completed(jobs):
            job.result()
    # The unchanged serial runner verifies its manifest, skips completed fits,
    # and writes the exact same final aggregation/paired summary as a serial run.
    f.run(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(f.DATA))
    args = parser.parse_args()
    f.torch.set_num_threads(2)
    run(args.dataset)
