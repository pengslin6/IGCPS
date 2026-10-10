"""Run the current no-node-head HE-TGT model with external artifacts."""
from pathlib import Path
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["IGCPS", "TE-CUP-SEC"], required=True)
    parser.add_argument("--seed", type=int, choices=[11, 22, 33, 44, 55], default=11)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--reference-table", type=Path)
    parser.add_argument("--device", default="auto", help="A PyTorch device or auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite the output")
    sys.path.insert(0, str(Path(__file__).resolve().parent / "no_head_20261009"))
    from runtime import Pipeline, torch, np
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    graph = torch.load(args.graph, map_location="cpu", weights_only=True)
    features = graph["features"] if isinstance(graph, dict) else graph
    with np.load(args.history, allow_pickle=False) as source:
        history = {key: source[key] for key in ("static", "history")}
    with threadpool_limits(limits=2):
        model = Pipeline(args.artifact_root, args.checkpoint_root, args.dataset, args.seed, args.device, args.reference_table)
        result = model.predict(features, history, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **result)
    print(f"Saved {len(features)} predictions to {args.output}")

if __name__ == "__main__":
    main()
