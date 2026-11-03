"""Run the nine comparison models with the paper's locked training settings."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from he_tgt.utils import config, sha256, write_json
from . import protocol as exp
from . import components as tr
from .training import build_development_test, run_seed, summarize


def run(args):
    cfg = config(args.dataset)
    if not args.allow_new_data and sha256(args.csv).lower() != cfg["fusion_csv_sha256"].lower():
        raise ValueError("Source differs from the paper; use --allow-new-data for a new run")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = exp.load_protocol(args.csv)
    development, test = build_development_test(protocol.raw)
    development = tr.cap_split_per_class(development, cfg["caps_per_class"]["train"], seed=42, split_name="Development")
    test = tr.cap_split_per_class(test, cfg["caps_per_class"]["test"], seed=44, split_name="Test")
    monitor = tr.cap_split_per_class(development, 8, seed=43, split_name="Epoch monitor")
    epochs = {item["model"]: item["best_epoch"] for item in cfg["baseline_epochs"]}
    models = args.models or list(epochs)
    if set(models).difference(epochs):
        raise ValueError("Unknown baseline")
    rows = []
    for model in models:
        for seed in args.seeds:
            rows.append(run_seed(model, seed, protocol, development, monitor, test, epochs[model],
                                 cfg["baseline_lock"]["selected_recency_strength"], args.output, args.device))
            pd.DataFrame(rows).to_csv(args.output / "runs.csv", index=False)
    write_json(args.output / "summary.json", summarize(pd.DataFrame(rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dataset", choices=["IGCPS", "TE-CUP-SEC"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--allow-new-data", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        run(args)
