"""Separate metric and scalar decision reconstruction for the shared pair model."""
import argparse
import itertools
import pair_model as p
import unified as u
from unified import np, torch, joblib, HERE, MODEL, ROOT, DATASETS, SEEDS, KEYS, read, dump, sha, threadpool_limits
from verify import check_metrics


def scalar_decision(base, prior, pairs, policy):
    result = []
    for i, (b, h) in enumerate(zip(base, prior)):
        row = (1-policy['gamma'])*b+policy['gamma']*h
        classes = sorted(range(len(row)), key=lambda j: (row[j], j))[-2:]
        a, c = sorted(classes)
        mass = row[a]+row[c]
        binary = (1-policy['beta'])*pairs[(a, c)][i, 0]+policy['beta']*pairs[(a, c)][i, 1]
        value = (1-policy['rho'])*row[c]/max(mass, 1e-15)+policy['rho']*binary
        row[a], row[c] = mass*(1-value), mass*value
        result.append(row)
    return np.array(result)


def development():
    for path, digest in read(p.OUT/'manifest.json')['hashes'].items():
        assert sha(path) == digest
    selection = read(p.OUT/'selection_lock.json')
    selected = selection['selected']
    baseline = read(HERE/'development_baselines.json')
    assert read(HERE/'development_verification.json')['all_passed']
    for dataset in DATASETS:
        train, val = u.data(dataset, 'train'), u.data(dataset, 'validation')
        base = dict(np.load(HERE/dataset/'validation_baseline.npz'))['probabilities']
        prior = dict(np.load(HERE/dataset/'screen'/f"{p.PRIOR['name']}.npz"))['probabilities']
        saved = joblib.load(p.OUT/dataset/'screen'/f"{selected['setting']['name']}.joblib")
        refit = p.fit(train, saved['schema'], selected['setting'])
        refit_probs = p.pair_probabilities(refit, val)
        for pair, values in p.pair_probabilities(saved, val).items():
            np.testing.assert_array_equal(values, refit_probs[pair])
        saved_probs = {}
        for setting in p.settings():
            source = dict(np.load(p.OUT/dataset/'screen'/f"{setting['name']}.npz"))
            saved_probs[setting['name']] = {tuple(map(int, key.split('_'))): value for key, value in source.items()}
        for candidate in selection['candidates']:
            pairs = saved_probs[candidate['setting']['name']]
            prediction = scalar_decision(base, prior, pairs, candidate['policy']).argmax(1)
            for part, ids in u.halves(val).items():
                check_metrics(dataset, {'labels': val['labels'][ids]}, prediction[ids], candidate['metrics'][dataset][part])
        print('PAIR DEVELOPMENT VERIFIED', dataset, flush=True)
    dump(p.OUT/'development_verification.json', {'all_passed': True, 'candidate_dataset_pairs': 216,
        'selected_all_pair_heads_refitted': 51, 'decision_verification': 'separate scalar implementation',
        'metric_verification': 'sklearn', 'selection_sha256': sha(p.OUT/'selection_lock.json'), 'verifier_sha256': sha(__file__)})


def final():
    from predict_pair import predict_from_graph
    lock = read(p.OUT/'final_fit_lock.json')
    for path, digest in lock['hashes'].items():
        assert sha(path) == digest
    comparison = read(p.OUT/'comparison.json')
    classifier_params = []
    for dataset in DATASETS:
        train = dict(np.load(MODEL/dataset/'train.npz'))
        test = dict(np.load(MODEL/dataset/'test.npz'))
        head = joblib.load(p.OUT/dataset/'final_pairs.joblib')
        prior = joblib.load(p.OUT/dataset/'final_prior.joblib')
        refit = p.fit(train, head['schema'], lock['setting'])
        prior_refit = u.fit(train, prior['schema'], p.PRIOR)
        pairs = p.pair_probabilities(head, test)
        pair_refit = p.pair_probabilities(refit, test)
        for pair, values in pairs.items():
            np.testing.assert_array_equal(values, pair_refit[pair])
            classifier_params.append({key: head['pairs'][pair][key].get_params() for key in ['linear', 'tree']})
        prior_probs = u.probabilities(prior, test)
        np.testing.assert_array_equal(prior_probs, u.probabilities(prior_refit, test))
        graph = u.graph(dataset, 'test')
        subset = np.unique(np.linspace(0, len(test['labels'])-1, 128, dtype=int))
        for seed in SEEDS:
            saved = dict(np.load(p.OUT/dataset/f'final_seed{seed}.npz'))
            base = dict(np.load(MODEL/dataset/f'seed{seed}.predictions.npz'))
            incumbent = base if dataset == 'IGCPS' else dict(np.load(ROOT/'te_attack6_20260929'/f'final_seed{seed}.npz'))
            scores = scalar_decision(base['base_probabilities'], prior_probs, pairs, lock['policy'])
            np.testing.assert_array_equal(scores, saved['probabilities'])
            np.testing.assert_array_equal(scores.argmax(1), saved['predictions'])
            for key in ['raw_indices', 'labels', 'root_scores']:
                np.testing.assert_array_equal(saved[key], base[key])
            row = read(p.OUT/dataset/f'final_seed{seed}.json')
            for field, predictions in [('metrics', saved['predictions']), ('baseline_metrics', incumbent['predictions']), ('retained_metrics', base['predictions'])]:
                check_metrics(dataset, test, predictions, row[field])
            interface = predict_from_graph(dataset, seed, graph['features'][subset], {k: test[k][subset] for k in ['static', 'history']})
            np.testing.assert_array_equal(interface['predictions'], saved['predictions'][subset])
            np.testing.assert_allclose(interface['probabilities'], saved['probabilities'][subset], atol=1e-6)
            np.testing.assert_allclose(interface['root_scores'], saved['root_scores'][subset], atol=1e-6)
        for field, values in comparison['datasets'][dataset]['means'].items():
            for key, value in values.items():
                np.testing.assert_allclose(value, np.mean([r[field][key] for r in comparison['datasets'][dataset]['rows']]), atol=1e-10)
        print('PAIR FINAL VERIFIED', dataset, flush=True)
    assert all(params == classifier_params[0] for params in classifier_params)
    dump(p.OUT/'final_verification.json', {'all_passed': True, 'full_test_seed_pairs_checked': 10,
        'all_pairs_and_prior_deterministic_refits': True, 'same_hyperparameters_all_classes_and_datasets': True,
        'scalar_decision_reconstruction': True, 'sklearn_metric_reconstruction': True,
        'public_interface_rows_per_dataset_seed': 128, 'reference_node_scores_unchanged': True,
        'comparison_sha256': sha(p.OUT/'comparison.json'), 'verifier_sha256': sha(__file__)})


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['development', 'final'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        globals()[args.phase]()
