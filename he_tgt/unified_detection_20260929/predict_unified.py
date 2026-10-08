"""Common inference entry: no dataset-specific class routing or thresholds."""
import argparse
from pathlib import Path
import unified as u
from unified import torch, np, joblib, HERE, DATASETS, SEEDS, threadpool_limits
from current_model import load_backbone


def predict_from_graph(dataset, seed, features, history):
    head = joblib.load(HERE/dataset/'final_head.joblib')
    lock = u.read(HERE/'final_fit_lock.json')
    assert len(features) == len(history['history']) and len(features) > 0
    model = load_backbone(dataset, seed, features.shape[-1])
    adjacency = torch.load(u.RUNTIME/'assets'/dataset/'adjacency.pt', weights_only=True)
    probabilities, roots = [], []
    with torch.inference_mode():
        for batch in features.split(128):
            output = model(batch, adjacency)
            probabilities.append(output['anomaly_logits'].softmax(-1).numpy())
            roots.append(output['root_logits'].sigmoid().numpy())
    base = np.concatenate(probabilities)
    result = u.predict(head, history, base, lock['alpha'])
    result['root_scores'] = np.concatenate(roots)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=11)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Refusing to overwrite existing output'
    torch.set_num_threads(2)
    graph = torch.load(args.graph, map_location='cpu', weights_only=False)
    features = graph['features'] if isinstance(graph, dict) else graph
    with np.load(args.history) as source:
        history = {key: source[key] for key in ['static', 'history']}
    with threadpool_limits(limits=2):
        np.savez_compressed(args.output, **predict_from_graph(args.dataset, args.seed, features, history))
