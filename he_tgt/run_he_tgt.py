"""Run unchanged HE-TGT source with an explicit external artifact root."""
from pathlib import Path
import argparse
import sys


def load_pipeline(artifact_root):
    source = Path(__file__).resolve().parent
    sys.path.insert(0, str(source / "unified_detection_20260929"))
    import predict_pair
    import unified as u
    import pair_model as p
    import current_model

    root = Path(artifact_root).resolve()
    u.ROOT = root
    u.MODEL = root / "detection_history_final_20260929"
    u.RUNTIME = u.MODEL / "retained_runtime"
    u.AUDIT = root / "experiments_20260928"
    p.OUT = root / "unified_detection_20260929/pair_refinement"
    current_model.ASSETS = u.RUNTIME / "assets"
    lock_path = p.OUT / "final_fit_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing HE-TGT lock: {lock_path}. See REPRODUCING.md.")
    policy = u.read(lock_path)["policy"]
    if policy != {"beta": 1.0, "gamma": 0.25, "rho": 0.5}:
        raise ValueError(f"Artifacts do not use the current manuscript policy: {policy}")
    return predict_pair, u, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["IGCPS", "TE-CUP-SEC"], required=True)
    parser.add_argument("--seed", type=int, choices=[11, 22, 33, 44, 55], default=11)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite the output")
    predict, u, _ = load_pipeline(args.artifact_root)
    u.torch.set_num_threads(2)
    graph = u.torch.load(args.graph, map_location="cpu", weights_only=True)
    features = graph["features"] if isinstance(graph, dict) else graph
    with u.np.load(args.history, allow_pickle=False) as source:
        history = {key: source[key] for key in ["static", "history"]}
    with u.threadpool_limits(limits=2):
        result = predict.predict_from_graph(args.dataset, args.seed, features, history)
    with args.output.open("xb") as stream:
        u.np.savez_compressed(stream, **result)
    print(f"Saved {len(features)} predictions to {args.output}")


if __name__ == "__main__":
    main()
