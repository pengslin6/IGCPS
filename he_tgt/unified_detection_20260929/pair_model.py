"""Universal all-pairs refinement, identical algorithm on both datasets."""
import argparse
import itertools
import time
import warnings
import unified as u
from unified import np, torch, joblib, HERE, MODEL, ROOT, DATASETS, SEEDS, KEYS, read, dump, sha, threadpool_limits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning

OUT = HERE/'pair_refinement'
PRIOR = {'name': 'full_history__leaves7_min60', 'view': 'full_history', 'leaves': 7, 'minimum': 60}


def settings():
    return [{'name': f'{view}__min{minimum}', 'view': view, 'minimum': minimum}
        for view in ['network_history', 'network_with_physical_changes'] for minimum in [30, 60]]


def fit(arr, schema, setting, linear_source=None):
    x = u.features(arr, schema, setting)
    output = {'schema': schema, 'setting': setting, 'pairs': {}}
    for a, b in itertools.combinations(range(schema['classes']), 2):
        mask = np.isin(arr['labels'], [a, b])
        y = arr['labels'][mask]
        raw = arr['raw_indices'][mask]
        scaler = StandardScaler().fit(x[mask])
        values = np.clip(scaler.transform(x[mask]), -10, 10)
        weights = np.empty(len(y), dtype=np.float64)
        for label in [a, b]:
            ids = np.flatnonzero(y == label)
            ids = ids[np.argsort(raw[ids], kind='stable')]
            weights[ids] = len(y)/(2*len(ids))*np.linspace(1, 2, len(ids))
        weights /= weights.mean()
        if linear_source is None:
            linear = LogisticRegression(C=1, solver='lbfgs', tol=1e-8, max_iter=3000, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                linear.fit(values, y, sample_weight=weights)
        else:
            linear = linear_source['pairs'][(a, b)]['linear']
            np.testing.assert_array_equal(scaler.mean_, linear_source['pairs'][(a, b)]['scaler'].mean_)
        tree = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15,
            min_samples_leaf=setting['minimum'], l2_regularization=5, max_bins=128, early_stopping=False, random_state=0)
        tree.fit(values, y, sample_weight=weights)
        for classifier in [linear, tree]:
            np.testing.assert_array_equal(classifier.classes_, [a, b])
        output['pairs'][(a, b)] = {'scaler': scaler, 'linear': linear, 'tree': tree}
    return output


def pair_probabilities(head, arr):
    x = u.features(arr, head['schema'], head['setting'])
    return {pair: np.stack([m.predict_proba(np.clip(items['scaler'].transform(x), -10, 10))[:, 1]
                           for m in [items['linear'], items['tree']]], axis=-1)
            for pair, items in head['pairs'].items()}


def decision(base, prior, pairs, setting):
    p = (1-setting['gamma'])*base+setting['gamma']*prior
    output = p.copy()
    top2 = np.sort(np.argsort(p, axis=1, kind='stable')[:, -2:], axis=1)
    for (a, b), probs in pairs.items():
        mask = (top2[:, 0] == a) & (top2[:, 1] == b)
        mass = p[mask, a]+p[mask, b]
        conditional = (1-setting['beta'])*probs[mask, 0]+setting['beta']*probs[mask, 1]
        refined = (1-setting['rho'])*p[mask, b]/np.maximum(mass, 1e-15)+setting['rho']*conditional
        output[mask, b] = mass*refined
        output[mask, a] = mass*(1-refined)
    np.testing.assert_allclose(output.sum(1), 1, atol=1e-6)
    return output


def screen():
    OUT.mkdir(exist_ok=True)
    files = [u.Path(__file__), HERE/'PAIR_PROTOCOL.md', HERE/'unified.py', HERE/'selection_lock.json']
    lock = {'hashes': {str(p.resolve()): sha(p) for p in files}, 'settings': settings(), 'new_test_inputs_used': False}
    if (OUT/'manifest.json').exists():
        assert read(OUT/'manifest.json') == lock
    else:
        dump(OUT/'manifest.json', lock)
    baseline = read(HERE/'development_baselines.json')
    arrays = {d: (u.data(d, 'train'), u.data(d, 'validation')) for d in DATASETS}
    base = {d: dict(np.load(HERE/d/'validation_baseline.npz'))['probabilities'] for d in DATASETS}
    prior = {d: dict(np.load(HERE/d/'screen'/f"{PRIOR['name']}.npz"))['probabilities'] for d in DATASETS}
    candidates, linear_cache = [], {}
    for setting in settings():
        pair_outputs = {}
        for dataset in DATASETS:
            start = time.perf_counter()
            output = OUT/dataset/'screen'/setting['name']
            output.parent.mkdir(parents=True, exist_ok=True)
            cache_key = dataset, setting['view']
            if output.with_suffix('.joblib').exists():
                head = joblib.load(output.with_suffix('.joblib'))
            else:
                head = fit(arrays[dataset][0], u.feature_metadata(dataset), setting, linear_cache.get(cache_key))
                joblib.dump(head, output.with_suffix('.joblib'), compress=3)
            linear_cache[cache_key] = head
            pair_outputs[dataset] = pair_probabilities(head, arrays[dataset][1])
            np.savez_compressed(output.with_suffix('.npz'), **{f'{a}_{b}': p for (a, b), p in pair_outputs[dataset].items()})
            print('PAIR SCREEN', dataset, setting['name'], 'pairs', len(head['pairs']), 'seconds', round(time.perf_counter()-start, 2), flush=True)
        for beta, gamma, rho in itertools.product([0., .5, 1.], [0., .25, .5], [.5, .75, 1.]):
            policy = {'beta': beta, 'gamma': gamma, 'rho': rho}
            measured, failures = {}, []
            for dataset in DATASETS:
                val = arrays[dataset][1]
                pred = decision(base[dataset], prior[dataset], pair_outputs[dataset], policy).argmax(1)
                measured[dataset] = {part: u.metrics(dataset, val['labels'][ids], pred[ids]) for part, ids in u.halves(val).items()}
                failures += [dataset+':full:'+f for f in u.guards(measured[dataset]['full'], baseline[dataset]['full'])]
                for part in ['early', 'late']:
                    a, b = measured[dataset][part], baseline[dataset][part]
                    if a['f1'] < b['f1']-1-1e-8:
                        failures.append(dataset+':'+part+':f1')
                    if a['normal_fpr'] > b['normal_fpr']+2+1e-8:
                        failures.append(dataset+':'+part+':normal_fpr')
            gains = [measured[d]['full']['f1']-baseline[d]['full']['f1'] for d in DATASETS]
            candidates.append({'name': setting['name']+f'__beta{beta}_gamma{gamma}_rho{rho}', 'setting': setting,
                'policy': policy, 'metrics': measured, 'failed_guards': failures, 'worst_gain': min(gains),
                'mean_gain': float(np.mean(gains)), 'dimensions': sum(u.features(arrays[d][0], u.feature_metadata(d), setting).shape[1] for d in DATASETS)})
    eligible = [c for c in candidates if not c['failed_guards']]
    ranked = sorted(eligible or candidates, key=lambda c: (len(c['failed_guards']), -c['worst_gain'], -c['mean_gain'],
        c['dimensions'], -c['setting']['minimum'], c['name']))
    dump(OUT/'selection_lock.json', {'selected': ranked[0], 'eligible_count': len(eligible), 'candidates': candidates,
        'diagnostic_only': not bool(eligible), 'test_used': False, 'manifest_sha256': sha(OUT/'manifest.json')})
    print('PAIR SELECTED', ranked[0]['name'], 'eligible', len(eligible), 'worst gain', ranked[0]['worst_gain'], 'failures', ranked[0]['failed_guards'], flush=True)


def final_fit():
    for path, digest in read(OUT/'manifest.json')['hashes'].items():
        assert sha(path) == digest
    assert read(OUT/'development_verification.json')['all_passed']
    selected = read(OUT/'selection_lock.json')['selected']
    files = [u.Path(__file__), HERE/'unified.py', HERE/'PAIR_PROTOCOL.md', OUT/'manifest.json', OUT/'selection_lock.json',
             OUT/'development_verification.json']
    for dataset in DATASETS:
        arr = dict(np.load(MODEL/dataset/'train.npz'))
        schema = u.feature_metadata(dataset)
        head = fit(arr, schema, selected['setting'])
        prior = u.fit(arr, schema, PRIOR)
        joblib.dump(head, OUT/dataset/'final_pairs.joblib', compress=3)
        joblib.dump(prior, OUT/dataset/'final_prior.joblib', compress=3)
        files += [OUT/dataset/'final_pairs.joblib', OUT/dataset/'final_prior.joblib', MODEL/dataset/'train.npz', MODEL/dataset/'preprocessing.joblib']
        print('PAIR FINAL FIT', dataset, flush=True)
    dump(OUT/'final_fit_lock.json', {'hashes': {str(p.resolve()): sha(p) for p in files}, 'policy': selected['policy'],
        'setting': selected['setting'], 'new_test_inputs_used': False})
    print('PAIR FINAL FIT LOCKED', flush=True)


def predict(head, prior, arr, base, policy):
    scores = decision(base, u.probabilities(prior, arr), pair_probabilities(head, arr), policy)
    return {'predictions': scores.argmax(1), 'probabilities': scores}


def evaluate():
    lock = read(OUT/'final_fit_lock.json')
    for path, digest in lock['hashes'].items():
        assert sha(path) == digest
    report = {}
    for dataset in DATASETS:
        arr = dict(np.load(MODEL/dataset/'test.npz'))
        head = joblib.load(OUT/dataset/'final_pairs.joblib')
        prior = joblib.load(OUT/dataset/'final_prior.joblib')
        pair_probs = pair_probabilities(head, arr)
        prior_probs = u.probabilities(prior, arr)
        rows = []
        for seed in SEEDS:
            base = dict(np.load(MODEL/dataset/f'seed{seed}.predictions.npz'))
            incumbent = base if dataset == 'IGCPS' else dict(np.load(ROOT/'te_attack6_20260929'/f'final_seed{seed}.npz'))
            for key in ['labels', 'raw_indices']:
                np.testing.assert_array_equal(base[key], arr[key])
                np.testing.assert_array_equal(incumbent[key], arr[key])
            scores = decision(base['base_probabilities'], prior_probs, pair_probs, lock['policy'])
            result = {'predictions': scores.argmax(1), 'probabilities': scores, 'root_scores': base['root_scores'],
                      'labels': arr['labels'], 'raw_indices': arr['raw_indices']}
            np.savez_compressed(OUT/dataset/f'final_seed{seed}.npz', **result)
            row = {'seed': seed, 'metrics': u.metrics(dataset, arr['labels'], result['predictions']),
                'baseline_metrics': u.metrics(dataset, arr['labels'], incumbent['predictions']),
                'retained_metrics': u.metrics(dataset, arr['labels'], base['predictions'])}
            dump(OUT/dataset/f'final_seed{seed}.json', row)
            rows.append(row)
        means = {field: {key: float(np.mean([r[field][key] for r in rows])) for key in KEYS}
                 for field in ['metrics', 'baseline_metrics', 'retained_metrics']}
        failed = u.guards(means['metrics'], means['baseline_metrics'], final=True)
        improved = sum(r['metrics']['f1'] > r['baseline_metrics']['f1'] for r in rows)
        if improved < 4:
            failed.append('seed_consistency')
        report[dataset] = {'rows': rows, 'means': means, 'failed_guards': failed, 'improved_seeds': improved}
        print('PAIR FINAL', dataset, means, 'failures', failed, flush=True)
    dump(OUT/'comparison.json', {'datasets': report, 'both_passed': all(not r['failed_guards'] for r in report.values()),
        'development_eligible': read(OUT/'selection_lock.json')['eligible_count'] > 0, 'test_used_for_selection': False,
        'same_settings_both_datasets': True, 'retrospective': True})


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['screen', 'final_fit', 'evaluate'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        globals()[args.phase]()
