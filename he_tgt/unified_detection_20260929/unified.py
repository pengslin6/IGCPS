"""One multiclass history detector and one mixing rule for both datasets."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / 'detection_history_final_20260929'
RUNTIME = MODEL / 'retained_runtime'
AUDIT = ROOT / 'experiments_20260928'
sys.path.insert(0, str(RUNTIME))
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from backbone import DATGTJoint, CANDIDATE
from history import make_history
import history_head

DATASETS = ['IGCPS', 'TE-CUP-SEC']
SEEDS = [11, 22, 33, 44, 55]
KEYS = ['accuracy', 'f1', 'normal_fpr', 'anomaly_recall', 'focus_recall', 'focus_precision', 'focus_f1']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def meta(dataset):
    return read(AUDIT / 'data_cache' / dataset / 'metadata.json')


def graph(dataset, split):
    return torch.load(AUDIT / 'data_cache' / dataset / f'{split}.pt', map_location='cpu', weights_only=False)


def data(dataset, split):
    return dict(np.load(HERE / dataset / f'{split}.npz'))


def feature_metadata(dataset):
    # Read only the schema here, not the fitted final scaler parameters.
    prep = joblib.load(MODEL / dataset / 'preprocessing.joblib')
    return {'network_width': len(prep['network_columns']), 'total_width': len(prep['columns']),
            'classes': len(meta(dataset)['label_names'])}


def prepare():
    dataset = 'IGCPS'
    output = HERE / dataset
    output.mkdir(parents=True, exist_ok=True)
    metadata = meta(dataset)
    assert sha(metadata['source_path']) == metadata['data_sha256']
    schema = joblib.load(MODEL / dataset / 'preprocessing.joblib')
    columns = schema['columns']
    strings = ['Time', 'Source IP', 'Destination IP', 'Protocol Type']
    frame = pd.read_csv(metadata['source_path'], usecols=columns + strings[:3],
                        dtype={c: np.float32 for c in columns if c not in strings})
    times = pd.to_datetime(frame['Time'], format='mixed', utc=True).astype('int64').to_numpy()
    order = np.argsort(times, kind='stable')
    times = times[order]
    frame = frame.iloc[order].reset_index(drop=True)
    reference = np.load(AUDIT / 'data_cache' / dataset / 'all_reference_labels.npz')['times']
    np.testing.assert_array_equal(times, pd.to_datetime(reference, format='mixed', utc=True).astype('int64'))
    splits = dict(np.load(AUDIT / 'data_cache' / dataset / 'split_indices.npz'))
    # Exclude test rows before forming vocabulary, values, normalization or history.
    frame = frame.loc[splits['development_full']].copy()
    train_frame = frame.loc[splits['train_full']]
    vocab = {c: {v: i for i, v in enumerate(sorted(train_frame[c].astype(str).unique()))}
             for c in columns if not pd.api.types.is_numeric_dtype(frame[c])}
    matrix = np.empty((len(frame), len(columns)), dtype=np.float32)
    for j, c in enumerate(columns):
        matrix[:, j] = (frame[c].astype(str).map(vocab[c]).fillna(-1) if c in vocab
                        else pd.to_numeric(frame[c], errors='coerce').fillna(0)).to_numpy(dtype=np.float32)
    assert np.isfinite(matrix).all()
    raw_ids = frame.index.to_numpy()
    train_mask = np.isin(raw_ids, splits['train_full'])
    scaler = StandardScaler().fit(matrix[train_mask])
    values = scaler.transform(matrix).astype(np.float32)
    stream = pd.factorize(frame['Source IP'].astype(str) + '\x00' + frame['Destination IP'].astype(str), sort=True)[0]
    joblib.dump({'scaler': scaler, 'vocabularies': vocab, 'columns': columns,
                 'fit_raw_indices': raw_ids[train_mask]}, output / 'development_preprocessing.joblib', compress=3)
    checks = []
    for split in ['train', 'validation']:
        cache = graph(dataset, split)
        mask = train_mask if split == 'train' else np.ones(len(frame), dtype=bool)
        arr = make_history(values[mask], times[raw_ids[mask]], stream[mask], raw_ids[mask], cache['raw_indices'].numpy())
        arr.update(labels=cache['labels'].numpy(), recency=cache['recency'].numpy())
        # Independent direct search reconstructs every history membership, age and block.
        context = np.flatnonzero(mask)
        lookup = {int(raw): i for i, raw in enumerate(raw_ids)}
        for row, raw in enumerate(arr['raw_indices']):
            q = lookup[int(raw)]
            ids = context[(stream[context] == stream[q]) & (times[raw_ids[context]] < times[raw])
                          & (times[raw] - times[raw_ids[context]] <= 30_000_000_000)][-8:]
            valid = arr['history_indices'][row] >= 0
            np.testing.assert_array_equal(arr['history_indices'][row][valid], raw_ids[ids])
            expected = np.zeros(values.shape[1] * 4 + 3, dtype=np.float32)
            w = values.shape[1]
            expected[:w] = values[q]
            if len(ids):
                age = (times[raw] - times[raw_ids[ids]]) / 1e9
                np.testing.assert_array_equal(arr['history_ages'][row][valid], age)
                expected[w:2*w] = values[q] - values[ids[-1]]
                expected[2*w:3*w] = values[q] - values[ids].mean(0)
                expected[3*w:4*w] = values[ids].std(0)
                expected[-3:] = [len(ids)/8, np.log1p(age[-1]), np.log1p(age[0])]
            np.testing.assert_array_equal(arr['history'][row], expected)
        np.savez_compressed(output / f'{split}.npz', **arr)
        checks.append({'split': split, 'rows_checked': len(arr['labels']), 'all_passed': True})
    te_dev = RUNTIME / 'assets/TE-CUP-SEC/development'
    (HERE / 'TE-CUP-SEC').mkdir(exist_ok=True)
    for split in ['train', 'validation']:
        arrays = dict(np.load(te_dev / f'standard_{split}.npz'))
        cache = graph('TE-CUP-SEC', split)
        for key in ['labels', 'raw_indices']:
            np.testing.assert_array_equal(arrays[key], cache[key].numpy())
        np.savez_compressed(HERE / 'TE-CUP-SEC' / f'{split}.npz', **arrays)
    dump(HERE / 'preparation.json', {'igcps_checks': checks, 'te_verified_cache': str(te_dev / 'data_verification.json'),
        'development_scaler_fit': 'training rows only', 'test_features_used_for_fitting': False,
        'feature_metadata': {d: feature_metadata(d) for d in DATASETS}})
    print('PREPARATION VERIFIED', flush=True)


def base_validation(dataset):
    cache = graph(dataset, 'validation')
    metadata = meta(dataset)
    selected = read(AUDIT / 'fair_control' / dataset / 'selection_lock.json')['selected']['DA-TGT']
    checkpoint = AUDIT / 'fair_control' / dataset / f"validation__DA-TGT__lr{round(selected['learning_rate']*1e6)}.pt"
    model = DATGTJoint(cache['features'].shape[-1], metadata['n_nodes'], metadata['n_net'], len(metadata['label_names']), CANDIDATE)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict'])
    model.eval()
    adjacency = torch.load(AUDIT / 'data_cache' / dataset / 'adjacency.pt', weights_only=True)
    with torch.inference_mode():
        probs = np.concatenate([model(batch, adjacency)['anomaly_logits'].softmax(-1).numpy() for batch in cache['features'].split(128)])
    return probs


def incumbent_validation(dataset, train, val, probs):
    # Dataset-specific code is confined to reproducing OLD comparators, never the new predictor.
    pred = probs.argmax(1)
    if dataset == 'IGCPS':
        def network_values(g):
            x = g['features']
            active = (x[:, :36, 4].abs() > .5).to(x.dtype)
            return ((x[:, :36, 6:] * active.unsqueeze(-1)).sum(1) / active.sum(1, keepdim=True).clamp_min(1)).numpy()
        tr_graph, val_graph = graph(dataset, 'train'), graph(dataset, 'validation')
        mask = np.isin(train['labels'], [2, 3])
        scaler = StandardScaler().fit(network_values(tr_graph)[mask])
        model = LogisticRegression(C=1, solver='lbfgs', class_weight='balanced', tol=1e-8, max_iter=2000, random_state=0)
        model.fit(scaler.transform(network_values(tr_graph)[mask]), train['labels'][mask], sample_weight=train['recency'][mask])
        p = model.predict_proba(scaler.transform(network_values(val_graph)))
        route = np.isin(pred, [2, 3])
        pred[route] = np.where(p[route, 0] >= .3, 2, 3)
        _, head = next(history_head.fit(dataset, {'kind': 'history_lr', 'view': 'network', 'epoch': 0}, train, 42))
        conditional = history_head.predict(head, val)
        route = np.isin(pred, [0, 2])
        pred[route] = np.where(conditional[route, 1] >= .5, 2, 0)
    else:
        head = joblib.load(ROOT / 'te_attack6_20260929/screen/network_with_physical_changes__leaves15_min60.joblib')
        width = val['static'].shape[1]
        ids = np.r_[np.arange(21), np.arange(width, width*4+3)]
        p = head['model'].predict_proba(np.clip(head['scaler'].transform(val['history'][:, ids].astype(np.float64)), -10, 10))
        route = np.isin(pred, [0, 6])
        pred[route] = np.where(p[route, 1] >= .5, 6, 0)
    return pred


def settings():
    return [{'name': f'{view}__leaves{leaves}_min{minimum}', 'view': view, 'leaves': leaves, 'minimum': minimum}
            for view in ['network_history', 'network_with_physical_changes', 'full_history']
            for leaves in [7, 15] for minimum in [15, 60]]


def features(arr, schema, setting):
    w, n = schema['total_width'], schema['network_width']
    assert arr['history'].shape[1] == w*4+3
    if setting['view'] == 'network_history':
        ids = np.r_[np.concatenate([np.arange(n)+w*k for k in range(4)]), np.arange(w*4, w*4+3)]
    elif setting['view'] == 'network_with_physical_changes':
        ids = np.r_[np.arange(n), np.arange(w, w*4+3)]
    else:
        ids = np.arange(w*4+3)
    return arr['history'][:, ids].astype(np.float64)


def fit(arr, schema, setting):
    x = features(arr, schema, setting)
    scaler = StandardScaler().fit(x)
    y = arr['labels']
    weights = np.empty(len(y), dtype=np.float64)
    for label in range(schema['classes']):
        ids = np.flatnonzero(y == label)
        ids = ids[np.argsort(arr['raw_indices'][ids], kind='stable')]
        weights[ids] = len(y) / (schema['classes']*len(ids)) * np.linspace(1, 2, len(ids))
    weights /= weights.mean()
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=setting['leaves'],
        min_samples_leaf=setting['minimum'], l2_regularization=5, max_bins=128, early_stopping=False, random_state=0)
    model.fit(np.clip(scaler.transform(x), -10, 10), y, sample_weight=weights)
    np.testing.assert_array_equal(model.classes_, np.arange(schema['classes']))
    return {'setting': setting, 'schema': schema, 'scaler': scaler, 'model': model}


def probabilities(head, arr):
    return head['model'].predict_proba(np.clip(head['scaler'].transform(features(arr, head['schema'], head['setting'])), -10, 10))


def predict(head, arr, base_probabilities, alpha):
    p = probabilities(head, arr)
    combined = (1-alpha)*base_probabilities + alpha*p
    return {'predictions': combined.argmax(1), 'probabilities': combined, 'history_probabilities': p}


def metrics(dataset, y, pred):
    k = len(meta(dataset)['label_names'])
    cm = np.bincount(y*k+pred, minlength=k*k).reshape(k, k)
    tp, counts, pos = cm.diagonal(), cm.sum(1), cm.sum(0)
    recall = 100*tp/np.maximum(counts, 1)
    precision = 100*tp/np.maximum(pos, 1)
    f1 = 200*tp/np.maximum(counts+pos, 1)
    focus = meta(dataset)['label_names'].index('NM' if dataset == 'IGCPS' else 'Attack6')
    return {'accuracy': float(100*tp.sum()/len(y)), 'f1': float(f1.mean()),
        'normal_fpr': float(100*(counts[0]-tp[0])/counts[0]),
        'anomaly_recall': float(100*(counts[1:].sum()-cm[1:, 0].sum())/counts[1:].sum()),
        'focus_recall': float(recall[focus]), 'focus_precision': float(precision[focus]), 'focus_f1': float(f1[focus]),
        'per_class_recall': recall.tolist(), 'per_class_precision': precision.tolist(), 'per_class_f1': f1.tolist(),
        'confusion_matrix': cm.tolist()}


def halves(arr):
    parts = {'early': [], 'late': []}
    for label in np.unique(arr['labels']):
        ids = np.flatnonzero(arr['labels'] == label)
        ids = ids[np.argsort(arr['raw_indices'][ids], kind='stable')]
        parts['early'].extend(ids[:len(ids)//2])
        parts['late'].extend(ids[len(ids)//2:])
    return {'full': np.arange(len(arr['labels'])), **{k: np.sort(v) for k, v in parts.items()}}


def guards(a, b, final=False):
    limits = [('f1', .25), ('anomaly_recall', -.5), ('focus_recall', -2), ('focus_f1', -1)]
    if final:
        limits.append(('accuracy', -.25))
    failed = [key for key, gain in limits if a[key] < b[key]+gain-1e-8]
    if a['normal_fpr'] > b['normal_fpr']+.5+1e-8:
        failed.append('normal_fpr')
    return failed


def screen():
    lock = {'hashes': {str(p.resolve()): sha(p) for p in [Path(__file__), HERE/'PROTOCOL.md', HERE/'preparation.json']},
            'settings': settings(), 'alphas': [.25, .5, .75, 1.], 'selection_uses_test': False}
    if (HERE/'manifest.json').exists():
        assert read(HERE/'manifest.json') == lock
    else:
        dump(HERE/'manifest.json', lock)
    baseline, arrays, base, schemas = {}, {}, {}, {}
    for dataset in DATASETS:
        train, val = data(dataset, 'train'), data(dataset, 'validation')
        assert not np.intersect1d(train['raw_indices'], val['raw_indices']).size
        schemas[dataset] = feature_metadata(dataset)
        arrays[dataset] = train, val
        base[dataset] = base_validation(dataset)
        incumbent = incumbent_validation(dataset, train, val, base[dataset])
        baseline[dataset] = {part: metrics(dataset, val['labels'][ids], incumbent[ids]) for part, ids in halves(val).items()}
        np.savez_compressed(HERE/dataset/'validation_baseline.npz', probabilities=base[dataset], predictions=incumbent,
                            labels=val['labels'], raw_indices=val['raw_indices'])
        print('BASELINE', dataset, {k: round(baseline[dataset]['full'][k], 4) for k in KEYS}, flush=True)
    dump(HERE/'development_baselines.json', baseline)
    candidates = []
    for setting in settings():
        outputs = {}
        for dataset in DATASETS:
            start = time.perf_counter()
            train, val = arrays[dataset]
            output = HERE/dataset/'screen'/setting['name']
            output.parent.mkdir(exist_ok=True)
            if output.with_suffix('.joblib').exists():
                head = joblib.load(output.with_suffix('.joblib'))
            else:
                head = fit(train, schemas[dataset], setting)
                joblib.dump(head, output.with_suffix('.joblib'), compress=3)
            outputs[dataset] = probabilities(head, val)
            np.savez_compressed(output.with_suffix('.npz'), probabilities=outputs[dataset], labels=val['labels'], raw_indices=val['raw_indices'])
            print('SCREEN', dataset, setting['name'], 'seconds', round(time.perf_counter()-start, 2), flush=True)
        for alpha in lock['alphas']:
            measured, failures = {}, []
            for dataset in DATASETS:
                val = arrays[dataset][1]
                pred = ((1-alpha)*base[dataset]+alpha*outputs[dataset]).argmax(1)
                measured[dataset] = {part: metrics(dataset, val['labels'][ids], pred[ids]) for part, ids in halves(val).items()}
                failures += [dataset+':full:'+f for f in guards(measured[dataset]['full'], baseline[dataset]['full'])]
                for part in ['early', 'late']:
                    a, b = measured[dataset][part], baseline[dataset][part]
                    if a['f1'] < b['f1']-1-1e-8:
                        failures.append(dataset+':'+part+':f1')
                    if a['normal_fpr'] > b['normal_fpr']+2+1e-8:
                        failures.append(dataset+':'+part+':normal_fpr')
            gains = [measured[d]['full']['f1']-baseline[d]['full']['f1'] for d in DATASETS]
            candidates.append({'name': setting['name']+f'__alpha{alpha}', 'setting': setting, 'alpha': alpha,
                'metrics': measured, 'worst_gain': min(gains), 'mean_gain': float(np.mean(gains)), 'failed_guards': failures,
                'dimensions': sum(features(arrays[d][0], schemas[d], setting).shape[1] for d in DATASETS)})
    eligible = [c for c in candidates if not c['failed_guards']]
    ranked = sorted(eligible or candidates, key=lambda c: (-c['worst_gain'], -c['mean_gain'], c['dimensions'],
                    c['setting']['leaves'], -c['setting']['minimum'], c['name']))
    dump(HERE/'selection_lock.json', {'selected': ranked[0], 'eligible_count': len(eligible), 'candidates': candidates,
        'manifest_sha256': sha(HERE/'manifest.json'), 'diagnostic_only': not bool(eligible), 'test_used': False})
    print('SELECTED', ranked[0]['name'], 'eligible', len(eligible), 'gain', ranked[0]['worst_gain'], flush=True)


def final_fit():
    selected = read(HERE/'selection_lock.json')['selected']
    for path, digest in read(HERE/'manifest.json')['hashes'].items():
        assert sha(path) == digest
    files = [Path(__file__), HERE/'PROTOCOL.md', HERE/'selection_lock.json', HERE/'manifest.json']
    for dataset in DATASETS:
        arr = dict(np.load(MODEL/dataset/'train.npz'))
        head = fit(arr, feature_metadata(dataset), selected['setting'])
        joblib.dump(head, HERE/dataset/'final_head.joblib', compress=3)
        files += [HERE/dataset/'final_head.joblib', MODEL/dataset/'train.npz', MODEL/dataset/'preprocessing.joblib']
    dump(HERE/'final_fit_lock.json', {'hashes': {str(p.resolve()): sha(p) for p in files}, 'alpha': selected['alpha'],
        'setting': selected['setting'], 'new_test_inputs_used': False})
    print('FINAL HEADS LOCKED', flush=True)


def evaluate():
    lock = read(HERE/'final_fit_lock.json')
    for path, digest in lock['hashes'].items():
        assert sha(path) == digest
    report = {}
    for dataset in DATASETS:
        arr = dict(np.load(MODEL/dataset/'test.npz'))
        head = joblib.load(HERE/dataset/'final_head.joblib')
        rows = []
        for seed in SEEDS:
            base = dict(np.load(MODEL/dataset/f'seed{seed}.predictions.npz'))
            incumbent = base if dataset == 'IGCPS' else dict(np.load(ROOT/'te_attack6_20260929'/f'final_seed{seed}.npz'))
            for key in ['labels', 'raw_indices']:
                np.testing.assert_array_equal(base[key], arr[key])
                np.testing.assert_array_equal(incumbent[key], arr[key])
            result = predict(head, arr, base['base_probabilities'], lock['alpha'])
            result.update(root_scores=base['root_scores'], labels=arr['labels'], raw_indices=arr['raw_indices'])
            np.savez_compressed(HERE/dataset/f'final_seed{seed}.npz', **result)
            row = {'seed': seed, 'metrics': metrics(dataset, arr['labels'], result['predictions']),
                'baseline_metrics': metrics(dataset, arr['labels'], incumbent['predictions']),
                'retained_metrics': metrics(dataset, arr['labels'], base['predictions'])}
            dump(HERE/dataset/f'final_seed{seed}.json', row)
            rows.append(row)
        means = {field: {key: float(np.mean([r[field][key] for r in rows])) for key in KEYS}
                 for field in ['metrics', 'baseline_metrics', 'retained_metrics']}
        failed = guards(means['metrics'], means['baseline_metrics'], final=True)
        improved = sum(r['metrics']['f1'] > r['baseline_metrics']['f1'] for r in rows)
        if improved < 4:
            failed.append('seed_consistency')
        report[dataset] = {'rows': rows, 'means': means, 'failed_guards': failed, 'improved_seeds': improved}
        print('FINAL', dataset, means, 'failures', failed, flush=True)
    dump(HERE/'comparison.json', {'datasets': report, 'both_passed': all(not r['failed_guards'] for r in report.values()),
        'development_eligible': read(HERE/'selection_lock.json')['eligible_count'] > 0, 'test_used_for_selection': False,
        'same_settings_both_datasets': True, 'retrospective': True})


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'screen', 'final_fit', 'evaluate'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        globals()[args.phase]()
