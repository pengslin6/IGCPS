"""Verify source integrity and optionally replay all ten no-head checkpoints."""
from pathlib import Path
import argparse, ast, hashlib, json, sys

HERE = Path(__file__).resolve().parent

def verify(artifact_root=None, checkpoint_root=None):
    manifest = json.loads((HERE / "SOURCE_SHA256.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected, name
    scripts = list(HERE.rglob("*.py"))
    assert set(manifest) == {p.relative_to(HERE).as_posix() for p in scripts}, "Source manifest coverage mismatch"
    for path in scripts:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    report = {"source_files_verified": len(manifest), "python_files_parsed": len(scripts),
              "training_performed": False, "inference_replay": "not requested"}
    if artifact_root is None:
        return report
    if checkpoint_root is None:
        raise ValueError("--checkpoint-root is required for inference verification")
    sys.path.insert(0, str(HERE / "no_head_20261009"))
    from runtime import Pipeline, torch, np, SETTINGS, SEEDS
    from threadpoolctl import threadpool_limits
    from sklearn.metrics import f1_score
    torch.set_num_threads(2)
    checks = []
    with threadpool_limits(limits=2):
        for dataset in SETTINGS:
            graph = torch.load(Path(artifact_root)/"experiments_20260928/data_cache"/dataset/"test.pt", map_location="cpu", weights_only=True)
            with np.load(Path(artifact_root)/"detection_history_final_20260929"/dataset/"test.npz", allow_pickle=False) as arrays:
                rows = np.unique(np.linspace(0,len(arrays["labels"])-1,128,dtype=int))
                history = {key: arrays[key][rows] for key in ("static","history")}
                labels, raw = arrays["labels"].copy(), arrays["raw_indices"].copy()
            np.testing.assert_array_equal(raw,graph["raw_indices"].numpy())
            for seed in SEEDS:
                model = Pipeline(artifact_root, checkpoint_root, dataset, seed, "cpu")
                got = model.predict(graph["features"][rows],history)
                with np.load(Path(checkpoint_root)/dataset/f"without_head_seed{seed}.predictions.npz",allow_pickle=False) as saved:
                    np.testing.assert_array_equal(labels,saved["labels"])
                    np.testing.assert_array_equal(raw,saved["raw_indices"])
                    np.testing.assert_array_equal(got["predictions"],saved["probabilities"][rows].argmax(1))
                    for field in ("probabilities","graph_probabilities","lookup_scores"):
                        np.testing.assert_allclose(got[field],saved[field][rows],atol=1e-6,rtol=0)
                    f1 = float(f1_score(labels,saved["probabilities"].argmax(1),average="macro")*100)
                checks.append({"dataset":dataset,"seed":seed,"replayed_rows":len(rows),"full_saved_rows":len(labels),
                               "macro_f1":f1,"graph_parameters":sum(p.numel() for p in model.graph.parameters())})
                print(f"Verified {dataset} seed {seed}",flush=True)
    report["inference_replay"]=checks
    report["absolute_probability_tolerance"]=1e-6
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root",type=Path)
    parser.add_argument("--checkpoint-root",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists():parser.error("Refusing to overwrite the output")
    text=json.dumps(verify(args.artifact_root,args.checkpoint_root),indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as stream:stream.write(text)
    print(text)
