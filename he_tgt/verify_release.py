"""Verify source integrity and optionally replay the locked inference interface."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import statistics

HERE = Path(__file__).resolve().parent


def verify(artifact_root=None):
    manifest = json.loads((HERE / "SOURCE_SHA256.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = HERE / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    scripts = list(HERE.rglob("*.py"))
    for path in scripts:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    metrics = json.loads((HERE / "he_tgt_revision_20260929/metrics.json").read_text(encoding="utf-8"))
    for dataset, record in metrics["datasets"].items():
        for variant, summary in record["summary"].items():
            expected = summary["f1"]["mean"]
            values = [row["variants"][variant]["f1"] for row in record["rows"]]
            assert abs(statistics.mean(values) - expected) < 1e-10, (dataset, variant)
    report = {"copied_source_files_verified": len(manifest), "python_files_parsed": len(scripts),
              "compact_f1_means_verified": 12, "training_performed": False,
              "inference_replay": "not run; external assets were not supplied"}
    if artifact_root is None:
        return report
    from run_he_tgt import load_pipeline
    from sklearn.metrics import f1_score
    predict, u, p = load_pipeline(artifact_root)
    u.torch.set_num_threads(2)
    checks = []
    with u.threadpool_limits(limits=2):
        for dataset in u.DATASETS:
            graph = u.torch.load(u.AUDIT / "data_cache" / dataset / "test.pt", map_location="cpu", weights_only=False)
            with u.np.load(u.MODEL / dataset / "test.npz") as source:
                rows = u.np.unique(u.np.linspace(0, len(source["labels"]) - 1, 128, dtype=int))
                history = {key: source[key][rows] for key in ["static", "history"]}
                labels = source["labels"].copy()
                raw_indices = source["raw_indices"].copy()
            u.np.testing.assert_array_equal(raw_indices, graph["raw_indices"].numpy())
            for seed in u.SEEDS:
                got = predict.predict_from_graph(dataset, seed, graph["features"][rows], history)
                with u.np.load(p.OUT / dataset / f"final_seed{seed}.npz") as saved:
                    u.np.testing.assert_array_equal(labels, saved["labels"])
                    u.np.testing.assert_array_equal(raw_indices, saved["raw_indices"])
                    u.np.testing.assert_array_equal(got["predictions"], saved["predictions"][rows])
                    for field in ["probabilities", "root_scores"]:
                        u.np.testing.assert_allclose(got[field], saved[field][rows], atol=1e-6, rtol=0)
                    f1 = float(f1_score(labels, saved["predictions"], average="macro") * 100)
                expected = next(row for row in metrics["datasets"][dataset]["rows"] if row["seed"] == seed)["variants"]["HE-TGT"]["f1"]
                assert abs(f1 - expected) < 1e-10
                checks.append({"dataset": dataset, "seed": seed, "inference_rows": len(rows),
                               "full_saved_rows": len(labels), "macro_f1": f1})
                print(f"Verified {dataset} seed {seed}", flush=True)
    report["inference_replay"] = checks
    report["probability_and_node_score_absolute_tolerance"] = 1e-6
    report["predicted_classes_identical"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.artifact_root)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
