"""Separate sklearn metric reconstruction, deterministic refits and interface checks."""
import argparse
import unified as u
from unified import np, torch, joblib, HERE, MODEL, ROOT, DATASETS, SEEDS, KEYS, read, dump, sha, threadpool_limits
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def independent_metrics(dataset, y, prediction):
    names = u.meta(dataset)['label_names']
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, labels=range(len(names)), zero_division=0)
    focus = names.index('NM' if dataset == 'IGCPS' else 'Attack6')
    return {'accuracy': accuracy_score(y, prediction)*100, 'f1': f1.mean()*100,
        'normal_fpr': (prediction[y == 0] != 0).mean()*100,
        'anomaly_recall': (prediction[y != 0] != 0).mean()*100,
        'focus_recall': recall[focus]*100, 'focus_precision': precision[focus]*100, 'focus_f1': f1[focus]*100,
        'per_class_recall': recall*100, 'per_class_precision': precision*100, 'per_class_f1': f1*100,
        'confusion_matrix': confusion_matrix(y, prediction, labels=range(len(names)))}


def check_metrics(dataset, arr, prediction, expected):
    measured = independent_metrics(dataset, arr['labels'], prediction)
    for key, value in measured.items():
        np.testing.assert_allclose(value, expected[key], atol=1e-9, rtol=1e-10)


def development():
    lock = read(HERE/'manifest.json')
    for path, digest in lock['hashes'].items():
        assert sha(path) == digest
    selection = read(HERE/'selection_lock.json')
    baseline = read(HERE/'development_baselines.json')
    checked = 0
    source_hashes = {}
    for dataset in DATASETS:
        train, val = u.data(dataset, 'train'), u.data(dataset, 'validation')
        base = dict(np.load(HERE/dataset/'validation_baseline.npz'))
        for key in ['raw_indices', 'labels']:
            np.testing.assert_array_equal(base[key], val[key])
        for split in ['train', 'validation']:
            source_hashes[str(HERE/dataset/f'{split}.npz')] = sha(HERE/dataset/f'{split}.npz')
        if dataset == 'TE-CUP-SEC':
            dev = u.RUNTIME/'assets/TE-CUP-SEC/development'
            assert read(dev/'data_verification.json')['all_passed']
            for split in ['train', 'validation']:
                assert sha(HERE/dataset/f'{split}.npz') == sha(dev/f'standard_{split}.npz')
        for part, ids in u.halves(val).items():
            check_metrics(dataset, {'labels': val['labels'][ids]}, base['predictions'][ids], baseline[dataset][part])
        selected = selection['selected']
        head = joblib.load(HERE/dataset/'screen'/f"{selected['setting']['name']}.joblib")
        refit = u.fit(train, head['schema'], selected['setting'])
        np.testing.assert_array_equal(u.probabilities(head, val), u.probabilities(refit, val))
        for candidate in selection['candidates']:
            saved = dict(np.load(HERE/dataset/'screen'/f"{candidate['setting']['name']}.npz"))
            for key in ['raw_indices', 'labels']:
                np.testing.assert_array_equal(saved[key], val[key])
            prediction = ((1-candidate['alpha'])*base['probabilities'] + candidate['alpha']*saved['probabilities']).argmax(1)
            for part, ids in u.halves(val).items():
                check_metrics(dataset, {'labels': val['labels'][ids]}, prediction[ids], candidate['metrics'][dataset][part])
            checked += 1
        print('DEVELOPMENT VERIFIED', dataset, flush=True)
    dump(HERE/'development_verification.json', {'all_passed': True, 'config_dataset_pairs': checked,
        'selected_head_exact_refits': 2, 'source_hashes': source_hashes,
        'selection_sha256': sha(HERE/'selection_lock.json'), 'verifier_sha256': sha(__file__)})


def final():
    from predict_unified import predict_from_graph
    lock = read(HERE/'final_fit_lock.json')
    for path, digest in lock['hashes'].items():
        assert sha(path) == digest
    comparison = read(HERE/'comparison.json')
    params = []
    checked = 0
    for dataset in DATASETS:
        train = dict(np.load(MODEL/dataset/'train.npz'))
        test = dict(np.load(MODEL/dataset/'test.npz'))
        head = joblib.load(HERE/dataset/'final_head.joblib')
        params.append(head['model'].get_params())
        assert head['setting'] == lock['setting']
        refit = u.fit(train, head['schema'], lock['setting'])
        np.testing.assert_array_equal(u.probabilities(head, test), u.probabilities(refit, test))
        graph = u.graph(dataset, 'test')
        subset = np.unique(np.linspace(0, len(test['labels'])-1, 128, dtype=int))
        for seed in SEEDS:
            saved = dict(np.load(HERE/dataset/f'final_seed{seed}.npz'))
            base = dict(np.load(MODEL/dataset/f'seed{seed}.predictions.npz'))
            incumbent = base if dataset == 'IGCPS' else dict(np.load(ROOT/'te_attack6_20260929'/f'final_seed{seed}.npz'))
            result = u.predict(head, test, base['base_probabilities'], lock['alpha'])
            for key in ['predictions', 'probabilities', 'history_probabilities']:
                np.testing.assert_array_equal(result[key], saved[key])
            for key in ['raw_indices', 'labels', 'root_scores']:
                np.testing.assert_array_equal(saved[key], base[key])
            row = read(HERE/dataset/f'final_seed{seed}.json')
            for field, prediction in [('metrics', saved['predictions']), ('baseline_metrics', incumbent['predictions']),
                                       ('retained_metrics', base['predictions'])]:
                check_metrics(dataset, test, prediction, row[field])
            interface = predict_from_graph(dataset, seed, graph['features'][subset], {k: test[k][subset] for k in ['static', 'history']})
            np.testing.assert_array_equal(interface['predictions'], saved['predictions'][subset])
            # Matrix kernels can differ very slightly when batching changes.
            np.testing.assert_allclose(interface['probabilities'], saved['probabilities'][subset], atol=1e-6)
            np.testing.assert_allclose(interface['root_scores'], saved['root_scores'][subset], atol=1e-6)
            checked += 1
        for field, values in comparison['datasets'][dataset]['means'].items():
            for key, value in values.items():
                expected = np.mean([r[field][key] for r in comparison['datasets'][dataset]['rows']])
                np.testing.assert_allclose(value, expected, atol=1e-10)
        print('FINAL VERIFIED', dataset, flush=True)
    assert params[0] == params[1]
    dump(HERE/'final_verification.json', {'all_passed': True, 'full_test_predictions_checked': checked,
        'separate_metric_implementation': 'sklearn', 'final_exact_refits': 2,
        'identical_classifier_hyperparameters': True, 'identical_feature_and_decision_rules': True,
        'public_interface_rows_per_dataset_seed': 128, 'reference_node_scores_unchanged': True,
        'comparison_sha256': sha(HERE/'comparison.json'), 'verifier_sha256': sha(__file__)})


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['development', 'final'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        globals()[args.phase]()
