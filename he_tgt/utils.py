"""Shared paths, configuration and checkpoint helpers."""
import hashlib
import json
from pathlib import Path
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config(dataset):
    return read_json(Path(__file__).resolve().parents[1] / "configs" / (dataset + ".json"))


def device_for(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def load_cache(path):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    # Accept archived array names when importing the paper's retained caches.
    for before, after in [("root_targets", "reference_targets"), ("root_mask", "reference_mask")]:
        if before in cache and after not in cache:
            cache[after] = cache.pop(before)
    return cache


def reference_table(cache, n_classes):
    import numpy as np
    labels = cache["labels"].numpy()
    targets = cache["reference_targets"].numpy()
    eligible = cache["reference_mask"].numpy() > 0
    table = np.zeros((n_classes, targets.shape[1]), np.float32)
    for label in range(n_classes):
        mask = (labels == label) & eligible
        if mask.any():
            table[label] = targets[mask].mean(0)
    return table
